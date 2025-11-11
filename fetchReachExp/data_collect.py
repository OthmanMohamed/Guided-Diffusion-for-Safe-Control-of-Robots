import gymnasium as gym
import numpy as np
from PIL import Image
import os
import pickle
from tqdm import tqdm
import mujoco

def check_collision(env):
    """
    Check for collisions using MuJoCo's contact detection system.
    Returns True if there is a collision between the robot and obstacles/table.
    """
    mj_model = env.unwrapped.model
    mj_data = env.unwrapped.data
    
    # Get obstacle and table geom IDs
    obstacle_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle1_geom")
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
    obstacle_table_geom_ids = [obstacle_geom_id] + table_geom_ids
    
    # Check for contacts
    for i in range(mj_data.ncon):
        contact = mj_data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        if (g1 in robot_geom_ids and g2 in obstacle_table_geom_ids) or \
           (g2 in robot_geom_ids and g1 in obstacle_table_geom_ids):
            return True
    
    return False

def collect_and_save_data(num_episodes=2000, chunk_size=100):
    env = gym.make("FetchReach-v2", render_mode="rgb_array")
    episode_length = 50

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Calculate number of chunks
    num_chunks = (num_episodes + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, num_episodes)
        current_chunk_size = end_idx - start_idx

        # Initialize arrays for current chunk
        o = np.zeros((current_chunk_size, episode_length, 10))
        u = np.zeros((current_chunk_size, episode_length-1, 4))
        ag = np.zeros((current_chunk_size, episode_length, 3))
        g = np.zeros((current_chunk_size, episode_length-1, 3))
        h = np.zeros((current_chunk_size, episode_length, 1))
        im = np.zeros((current_chunk_size, episode_length, 84, 84, 3), dtype=np.uint8)

        for traj_idx in tqdm(range(current_chunk_size), desc=f"Collecting chunk {chunk_idx + 1}/{num_chunks}"):
            actual_idx = start_idx + traj_idx
            obs, _ = env.reset()
            
            # Store initial observation
            o[traj_idx, 0] = obs["observation"]
            ag[traj_idx, 0] = obs["observation"][:3]
            # Convert to uint8 when storing the image
            im[traj_idx, 0] = (np.array(Image.fromarray(env.render()).resize((84, 84), Image.Resampling.LANCZOS))).astype(np.uint8)
            
            actual_length = episode_length
            for step_index in range(episode_length-1):
                # Get action
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
                
                # Store image as uint8
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
                last_frame = im[traj_idx, actual_length-1]
                last_horizon = h[traj_idx, actual_length-1]
                im[traj_idx, actual_length:] = last_frame
                o[traj_idx, actual_length:] = o[traj_idx, actual_length-1]
                ag[traj_idx, actual_length:] = ag[traj_idx, actual_length-1]
                h[traj_idx, actual_length:] = last_horizon

        # Prepare buffers for current chunk
        buffer = {
            'o': o,
            'u': u,
            'g': g,
            'ag': ag,
            'h': h
        }

        image_buffer = {
            'o': im,
            'u': u,
            'g': g,
            'ag': ag,
            'h': h
        }

        # Save current chunk
        try:
            with open(f'data/buffer_chunk_{chunk_idx}.pkl', 'wb') as f:
                pickle.dump(buffer, f)
            
            with open(f'data/image_buffer_chunk_{chunk_idx}.pkl', 'wb') as f:
                pickle.dump(image_buffer, f)
            
            print(f"Successfully saved chunk {chunk_idx + 1}/{num_chunks}")
        except Exception as e:
            print(f"Error saving chunk {chunk_idx + 1}: {str(e)}")
            continue

    # Combine all chunks
    try:
        print("Combining chunks...")
        combined_buffer = {
            'o': [], 'u': [], 'g': [], 'ag': [], 'h': []
        }
        combined_image_buffer = {
            'o': [], 'u': [], 'g': [], 'ag': [], 'h': []
        }

        for chunk_idx in range(num_chunks):
            with open(f'data/buffer_chunk_{chunk_idx}.pkl', 'rb') as f:
                chunk_data = pickle.load(f)
                for key in combined_buffer:
                    combined_buffer[key].append(chunk_data[key])

            with open(f'data/image_buffer_chunk_{chunk_idx}.pkl', 'rb') as f:
                chunk_data = pickle.load(f)
                for key in combined_image_buffer:
                    combined_image_buffer[key].append(chunk_data[key])

        # Stack arrays
        for key in combined_buffer:
            combined_buffer[key] = np.concatenate(combined_buffer[key], axis=0)
            combined_image_buffer[key] = np.concatenate(combined_image_buffer[key], axis=0)

        # Save combined data
        with open('data/buffer_obstacles_env_val.pkl', 'wb') as f:
            pickle.dump(combined_buffer, f)
        
        with open('data/image_buffer_obstacles_env_val.pkl', 'wb') as f:
            pickle.dump(combined_image_buffer, f)

        # Clean up chunk files
        for chunk_idx in range(num_chunks):
            os.remove(f'data/buffer_chunk_{chunk_idx}.pkl')
            os.remove(f'data/image_buffer_chunk_{chunk_idx}.pkl')

        print("Successfully combined and saved all data")
    except Exception as e:
        print(f"Error combining chunks: {str(e)}")

if __name__ == "__main__":
    collect_and_save_data(200)