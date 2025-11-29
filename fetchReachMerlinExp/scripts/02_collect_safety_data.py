#!/usr/bin/env python3
"""
Script 02: Collect Safety Data

This script collects safety-labeled data for training the safety model.
It runs episodes and labels actions as safe (1.0) or unsafe (0.0) based on
whether they lead to collisions. Steps before collisions are also labeled as unsafe.
"""

import os
import sys
import argparse
import numpy as np
import pickle
from tqdm import tqdm
import time
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


def collect_safety_data(args):
    """
    Collect safety data by running episodes for the full duration.
    Each data point contains: observation, action, safety_score
    Episodes continue until max_steps_per_episode regardless of collisions.
    When a collision occurs, the previous unsafe_steps_before_collision steps are marked as unsafe.
    """
    # Create environment
    render_mode = "human" if args.render else "rgb_array"
    env = create_fetch_reach_with_obstacles(render_mode=render_mode)
    env._max_episode_steps = args.max_steps_per_episode
    
    # Initialize data storage
    collected_data = {
        'observations': [],
        'actions': [],
        'safety_scores': [],
        'episode_info': []
    }
    
    total_episodes = 0
    total_data_points = 0
    
    print(f"Starting safety data collection...")
    print(f"Target episodes: {args.num_episodes}")
    print(f"Unsafe steps before collision: {args.unsafe_steps_before_collision}")
    print(f"Max steps per episode: {args.max_steps_per_episode}")
    print(f"Render mode: {render_mode}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    with tqdm(total=args.num_episodes, desc="Collecting episodes") as pbar:
        while total_episodes < args.num_episodes:
            # Reset environment
            obs, _ = env.reset(seed=args.seed + total_episodes if args.seed is not None else None)
            episode_start_time = time.time()
            
            # Store episode trajectory
            episode_observations = []
            episode_actions = []
            episode_safety_scores = []
            collision_steps = []  # Track when collisions occur
            
            # Run episode for full duration
            for step in range(args.max_steps_per_episode):
                # Generate random action
                action = env.action_space.sample()
                
                # Store current observation and action
                episode_observations.append(obs['observation'].copy())
                episode_actions.append(action.copy())
                
                # Take the action
                next_obs, _, terminated, truncated, _ = env.step(action)
                
                # Check for collision
                if check_collision(env):
                    collision_steps.append(step)
                
                obs = next_obs
                
                if terminated or truncated:
                    break
            
            # Now assign safety scores based on collision history
            num_steps = len(episode_observations)
            episode_safety_scores = [1.0] * num_steps  # Start with all safe
            
            # Mark steps as unsafe based on collision history
            for collision_step in collision_steps:
                # Mark the previous unsafe_steps_before_collision steps as unsafe
                start_step = max(0, collision_step - args.unsafe_steps_before_collision + 1)
                end_step = collision_step + 1
                
                for i in range(start_step, end_step):
                    if i < num_steps:  # Ensure we don't go out of bounds
                        episode_safety_scores[i] = 0.0
            
            # Add episode data to collected data
            collected_data['observations'].extend(episode_observations)
            collected_data['actions'].extend(episode_actions)
            collected_data['safety_scores'].extend(episode_safety_scores)
            
            # Store episode info
            episode_info = {
                'episode_id': total_episodes,
                'data_points': len(episode_observations),
                'duration': time.time() - episode_start_time,
                'collisions_occurred': len(collision_steps),
                'collision_steps': collision_steps.copy(),
                'safe_actions': sum(1 for s in episode_safety_scores if s == 1.0),
                'unsafe_actions': sum(1 for s in episode_safety_scores if s == 0.0)
            }
            collected_data['episode_info'].append(episode_info)
            
            total_episodes += 1
            total_data_points += len(episode_observations)
            pbar.update(1)
            
            # Update progress
            pbar.set_postfix({
                'Data Points': total_data_points,
                'Avg/Episode': f"{total_data_points/total_episodes:.1f}",
                'Last Episode': len(episode_observations),
                'Collisions': len(collision_steps)
            })
            
            # Save intermediate results every 50 episodes
            if total_episodes % 50 == 0:
                intermediate_path = os.path.join(args.output_dir, f'safety_data_intermediate_{total_episodes}.pkl')
                with open(intermediate_path, 'wb') as f:
                    pickle.dump(collected_data, f)
                print(f"\nSaved intermediate data: {intermediate_path}")
    
    # Convert lists to numpy arrays for efficiency
    collected_data['observations'] = np.array(collected_data['observations'])
    collected_data['actions'] = np.array(collected_data['actions'])
    collected_data['safety_scores'] = np.array(collected_data['safety_scores'])
    
    # Save final data
    final_path = os.path.join(args.output_dir, 'safety_data_final.pkl')
    with open(final_path, 'wb') as f:
        pickle.dump(collected_data, f)
    
    # Print summary statistics
    safety_scores = collected_data['safety_scores']
    safe_actions = np.sum(safety_scores == 1.0)
    unsafe_actions = np.sum(safety_scores == 0.0)
    
    print(f"\n=== Safety Data Collection Complete ===")
    print(f"Total episodes: {total_episodes}")
    print(f"Total data points: {total_data_points}")
    print(f"Safe actions: {safe_actions} ({safe_actions/total_data_points*100:.1f}%)")
    print(f"Unsafe actions: {unsafe_actions} ({unsafe_actions/total_data_points*100:.1f}%)")
    print(f"Average data points per episode: {total_data_points/total_episodes:.1f}")
    print(f"Data saved to: {final_path}")
    
    # Print episode statistics
    episodes_with_collision = sum(1 for info in collected_data['episode_info'] if info['collisions_occurred'] > 0)
    total_collisions = sum(info['collisions_occurred'] for info in collected_data['episode_info'])
    print(f"Episodes with collisions: {episodes_with_collision}/{total_episodes} ({episodes_with_collision/total_episodes*100:.1f}%)")
    print(f"Total collisions across all episodes: {total_collisions}")
    print(f"Average collisions per episode: {total_collisions/total_episodes:.2f}")
    
    env.close()
    return collected_data


def main():
    parser = argparse.ArgumentParser(description='Collect safety data for training safety model')
    parser.add_argument('--num-episodes', type=int, default=1000,
                      help='Number of episodes to collect')
    parser.add_argument('--max-steps-per-episode', type=int, default=100,
                      help='Maximum steps per episode')
    parser.add_argument('--unsafe-steps-before-collision', type=int, default=5,
                      help='Number of steps before collision to label as unsafe')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='../data/safety',
                      help='Directory to save collected data')
    parser.add_argument('--render', action='store_true',
                      help='Render the environment for human viewing')
    
    args = parser.parse_args()
    
    collect_safety_data(args)


if __name__ == "__main__":
    main()

