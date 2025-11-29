#!/usr/bin/env python3
"""
Script 01: Collect Demonstration Data

This script collects demonstration data for training the Merlin policy.
It collects trajectories from the FetchReach environment with obstacles,
storing observations, actions, goals, and achieved goals.
"""

import os
import sys
import argparse
import numpy as np
import pickle
from tqdm import tqdm
import gymnasium as gym
from PIL import Image
import mujoco

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.fetch_reach_with_obstacles import create_fetch_reach_with_obstacles


def check_collision(env):
    """
    Check for collisions using MuJoCo's contact detection system.
    Returns True if there is a collision between the robot and obstacles/table.
    """
    mj_model = env.unwrapped.model
    mj_data = env.unwrapped.data
    
    # Get obstacle and table geom IDs
    try:
        obstacle_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle1_geom")
    except:
        obstacle_geom_id = -1
    
    table_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "table0")
    table_geom_ids = [i for i in range(mj_model.ngeom) if mj_model.geom_bodyid[i] == table_body_id]
    
    # Get robot geom IDs
    robot_geom_ids = []
    for i in range(mj_model.ngeom):
        geom_body = mj_model.geom_bodyid[i]
        body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, geom_body)
        if body_name and body_name.startswith("robot0"):
            robot_geom_ids.append(i)
    
    # Combine obstacle and table geom IDs
    obstacle_table_geom_ids = [obstacle_geom_id] + table_geom_ids if obstacle_geom_id >= 0 else table_geom_ids
    
    # Check for contacts
    for i in range(mj_data.ncon):
        contact = mj_data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        if (g1 in robot_geom_ids and g2 in obstacle_table_geom_ids) or \
           (g2 in robot_geom_ids and g1 in obstacle_table_geom_ids):
            return True
    
    return False


def collect_demonstration_data(args):
    """
    Collect demonstration data by running episodes.
    
    Args:
        args: Command line arguments
    """
    # Create environment
    render_mode = "human" if args.render else "rgb_array"
    env = create_fetch_reach_with_obstacles(render_mode=render_mode)
    env._max_episode_steps = args.max_steps_per_episode
    
    # Create data directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    episode_length = args.max_steps_per_episode
    num_chunks = (args.num_episodes + args.chunk_size - 1) // args.chunk_size
    
    print(f"Starting data collection...")
    print(f"Total episodes: {args.num_episodes}")
    print(f"Episodes per chunk: {args.chunk_size}")
    print(f"Max steps per episode: {episode_length}")
    print(f"Output directory: {args.output_dir}")
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * args.chunk_size
        end_idx = min((chunk_idx + 1) * args.chunk_size, args.num_episodes)
        current_chunk_size = end_idx - start_idx
        
        # Initialize arrays for current chunk
        o = np.zeros((current_chunk_size, episode_length, 10))
        u = np.zeros((current_chunk_size, episode_length-1, 4))
        ag = np.zeros((current_chunk_size, episode_length, 3))
        g = np.zeros((current_chunk_size, episode_length-1, 3))
        h = np.zeros((current_chunk_size, episode_length, 1))
        
        if args.save_images:
            im = np.zeros((current_chunk_size, episode_length, 84, 84, 3), dtype=np.uint8)
        
        for traj_idx in tqdm(range(current_chunk_size), desc=f"Collecting chunk {chunk_idx + 1}/{num_chunks}"):
            actual_idx = start_idx + traj_idx
            obs, _ = env.reset(seed=args.seed + actual_idx if args.seed is not None else None)
            
            # Store initial observation
            o[traj_idx, 0] = obs["observation"]
            ag[traj_idx, 0] = obs["observation"][:3]
            
            if args.save_images:
                im[traj_idx, 0] = (np.array(Image.fromarray(env.render()).resize((84, 84), Image.Resampling.LANCZOS))).astype(np.uint8)
            
            actual_length = episode_length
            for step_index in range(episode_length-1):
                # Get action (random for demonstration data collection)
                action = env.action_space.sample()
                u[traj_idx, step_index] = action
                
                # Take step
                next_obs, _, terminated, truncated, _ = env.step(action)
                
                # Check for collision in the next state
                if check_collision(env):
                    h[traj_idx, step_index] = 1  # Set horizon to 1 if collision occurs
                
                # Store observation
                o[traj_idx, step_index + 1] = next_obs["observation"]
                ag[traj_idx, step_index + 1] = next_obs["observation"][:3]
                
                if args.save_images:
                    im[traj_idx, step_index + 1] = (np.array(Image.fromarray(env.render()).resize((84, 84), Image.Resampling.LANCZOS))).astype(np.uint8)
                
                # Store goal
                g[traj_idx, step_index] = next_obs["desired_goal"][:3]
                
                # Update observation for next step
                obs = next_obs
                
                if terminated or truncated:
                    actual_length = step_index + 2
                    break
            
            # Store final goal for all steps
            final_goal = obs["desired_goal"][:3]
            g[traj_idx, :] = final_goal
            
            # If episode ended early, repeat the last frame and horizon value
            if actual_length < episode_length:
                last_horizon = h[traj_idx, actual_length-1]
                o[traj_idx, actual_length:] = o[traj_idx, actual_length-1]
                ag[traj_idx, actual_length:] = ag[traj_idx, actual_length-1]
                h[traj_idx, actual_length:] = last_horizon
                if args.save_images:
                    last_frame = im[traj_idx, actual_length-1]
                    im[traj_idx, actual_length:] = last_frame
        
        # Prepare buffers for current chunk
        buffer = {
            'o': o,
            'u': u,
            'g': g,
            'ag': ag,
            'h': h
        }
        
        # Save current chunk
        chunk_path = os.path.join(args.output_dir, f'buffer_chunk_{chunk_idx}.pkl')
        with open(chunk_path, 'wb') as f:
            pickle.dump(buffer, f)
        print(f"Saved chunk {chunk_idx + 1}/{num_chunks} to {chunk_path}")
        
        if args.save_images:
            image_buffer = {
                'o': im,
                'u': u,
                'g': g,
                'ag': ag,
                'h': h
            }
            image_chunk_path = os.path.join(args.output_dir, f'image_buffer_chunk_{chunk_idx}.pkl')
            with open(image_chunk_path, 'wb') as f:
                pickle.dump(image_buffer, f)
            print(f"Saved image chunk {chunk_idx + 1}/{num_chunks} to {image_chunk_path}")
    
    # Combine all chunks
    print("Combining chunks...")
    combined_buffer = {
        'o': [], 'u': [], 'g': [], 'ag': [], 'h': []
    }
    
    for chunk_idx in range(num_chunks):
        chunk_path = os.path.join(args.output_dir, f'buffer_chunk_{chunk_idx}.pkl')
        with open(chunk_path, 'rb') as f:
            chunk_data = pickle.load(f)
            for key in combined_buffer:
                combined_buffer[key].append(chunk_data[key])
    
    # Stack arrays
    for key in combined_buffer:
        combined_buffer[key] = np.concatenate(combined_buffer[key], axis=0)
    
    # Save combined data
    final_path = os.path.join(args.output_dir, 'buffer.pkl')
    with open(final_path, 'wb') as f:
        pickle.dump(combined_buffer, f)
    print(f"Saved combined data to {final_path}")
    
    if args.save_images:
        combined_image_buffer = {
            'o': [], 'u': [], 'g': [], 'ag': [], 'h': []
        }
        for chunk_idx in range(num_chunks):
            image_chunk_path = os.path.join(args.output_dir, f'image_buffer_chunk_{chunk_idx}.pkl')
            with open(image_chunk_path, 'rb') as f:
                chunk_data = pickle.load(f)
                for key in combined_image_buffer:
                    combined_image_buffer[key].append(chunk_data[key])
        
        for key in combined_image_buffer:
            combined_image_buffer[key] = np.concatenate(combined_image_buffer[key], axis=0)
        
        image_final_path = os.path.join(args.output_dir, 'image_buffer.pkl')
        with open(image_final_path, 'wb') as f:
            pickle.dump(combined_image_buffer, f)
        print(f"Saved combined image data to {image_final_path}")
    
    # Clean up chunk files
    if args.cleanup_chunks:
        for chunk_idx in range(num_chunks):
            chunk_path = os.path.join(args.output_dir, f'buffer_chunk_{chunk_idx}.pkl')
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
            if args.save_images:
                image_chunk_path = os.path.join(args.output_dir, f'image_buffer_chunk_{chunk_idx}.pkl')
                if os.path.exists(image_chunk_path):
                    os.remove(image_chunk_path)
        print("Cleaned up chunk files")
    
    print(f"\n=== Data Collection Complete ===")
    print(f"Total trajectories: {args.num_episodes}")
    print(f"Data saved to: {final_path}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Collect demonstration data for Merlin training')
    parser.add_argument('--num-episodes', type=int, default=2000,
                      help='Number of episodes to collect')
    parser.add_argument('--chunk-size', type=int, default=100,
                      help='Number of episodes per chunk')
    parser.add_argument('--max-steps-per-episode', type=int, default=50,
                      help='Maximum steps per episode')
    parser.add_argument('--output-dir', type=str, default='../data/demonstration',
                      help='Directory to save collected data')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--render', action='store_true',
                      help='Render the environment for human viewing')
    parser.add_argument('--save-images', action='store_true',
                      help='Save image observations')
    parser.add_argument('--cleanup-chunks', action='store_true',
                      help='Remove chunk files after combining')
    
    args = parser.parse_args()
    
    collect_demonstration_data(args)


if __name__ == "__main__":
    main()

