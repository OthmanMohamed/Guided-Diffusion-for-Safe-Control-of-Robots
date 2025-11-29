#!/usr/bin/env python3
"""
Script 05: Evaluate Policy

This script evaluates a trained Merlin policy without safety integration.
It measures performance metrics including success rate, discounted returns, and collision rate.
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
import mujoco

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import math

from components.normalizer import normalizer
from envs.fetch_reach_with_obstacles import create_fetch_reach_with_obstacles


def timestep_embedding(timesteps, dim=32, max_period=50):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    if len(timesteps.shape) == 1:
        args = timesteps.float() * freqs
    else:
        args = timesteps.float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Policy(nn.Module):
    def __init__(self, state_shape, goal_dim,
                 timestep_embedding_dim, hidden_dim, output_dim,
                 max_path_length, max_action):
        super(Policy, self).__init__()
        self.timestep_embedding_dim = timestep_embedding_dim
        self.output_dim = output_dim
        self.max_path_length = max_path_length
        self.max_action = max_action
        if len(state_shape) == 1:
            self.cnn = False
            state_dim = state_shape[0]
            self.conv_layers = nn.Identity()
            self.dense_layers = nn.Sequential(
                nn.Linear(state_dim+goal_dim+timestep_embedding_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.cnn = True
            h, w, c = state_shape
            self.conv_layers = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Flatten(-3, -1),
            )
            with torch.no_grad():
                embedding_dim = int(np.prod(self.conv_layers(torch.zeros(1, c, h, w)).shape[1:]))
            self.dense_layers = nn.Sequential(
                nn.Linear(embedding_dim+goal_dim+timestep_embedding_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x, g, t):
        x = self.conv_layers(x)
        t = timestep_embedding(t, self.timestep_embedding_dim, self.max_path_length)
        x = torch.cat([x, g, t], dim=-1)
        x = self.dense_layers(x)
        mu, sigma = torch.split(x, self.output_dim//2, dim=-1)
        sigma = torch.nn.functional.softplus(sigma) + 1e-5
        return self.max_action * torch.tanh(mu), sigma


def discounted_return(rewards, gamma, reward_offset=True):
    L = len(rewards)
    if type(rewards[0]) == np.ndarray and len(rewards[0]):
        rewards = np.array(rewards).T
    else:
        rewards = np.array(rewards).reshape(1, L)
    if reward_offset:
        rewards += 1
    discount_weights = np.power(gamma, np.arange(L)).reshape(1, -1)
    dis_return = (rewards * discount_weights).sum(axis=1)
    undis_return = rewards.sum(axis=1)
    return dis_return.mean(), undis_return.mean()


def is_valid_goal(goal_pos):
    """
    Check if a goal position is valid (not inside the obstacle).
    The obstacle is a box at position (1.25, 0.75, 0.42) with size (0.05, 0.05, 0.4).
    """
    obstacle_pos = np.array([1.25, 0.75, 0.42])
    obstacle_size = np.array([0.05, 0.05, 0.4])
    
    # Check if goal is inside the obstacle's bounding box
    is_inside = (
        abs(goal_pos[0] - obstacle_pos[0]) < obstacle_size[0] and
        abs(goal_pos[1] - obstacle_pos[1]) < obstacle_size[1] and
        abs(goal_pos[2] - obstacle_pos[2]) < obstacle_size[2]
    )
    return not is_inside


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


def evaluate_checkpoint(checkpoint_path, env_name, image_obs=False, num_episodes=100, seed=0, 
                       human_render=False, step=500):
    """
    Evaluate a checkpointed model deterministically.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint (.pt file)
        env_name (str): Name of the environment (e.g., 'FetchReach')
        image_obs (bool): Whether the model uses image observations
        num_episodes (int): Number of evaluation episodes
        seed (int): Random seed for reproducibility
        human_render (bool): Whether to render the environment for human viewing
        step (int): Step number for normalizer loading
    
    Returns:
        tuple: (discounted_return, undiscounted_return, success_rate, collision_rate)
    """
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment with appropriate render mode
    render_mode = "human" if human_render else "rgb_array"
    env = create_fetch_reach_with_obstacles(render_mode=render_mode)
    env._max_episode_steps = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get environment dimensions
    action_space = env.action_space
    action_dim = action_space.shape[0]
    goal_space = env.observation_space['desired_goal']
    goal_dim = goal_space.shape[0]
    
    # Get state shape
    obs, _ = env.reset()
    if image_obs:
        state_shape = (84, 84, 3)  # Standard image size
    else:
        state_shape = obs['observation'].shape
    
    # Initialize policy
    policy = Policy(
        state_shape=state_shape,
        goal_dim=goal_dim,
        timestep_embedding_dim=32,
        hidden_dim=256,
        output_dim=2*action_dim,
        max_path_length=50,
        max_action=float(action_space.high[0])
    ).to(device)
    
    # Get checkpoint directory
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # Load checkpoint
    policy.load_state_dict(torch.load(checkpoint_path))
    policy.eval()
    
    # Load normalizers
    if not image_obs:
        o_norm = normalizer(size=10)
        o_norm.load_normalizer(os.path.join(checkpoint_dir, f'o_norm_{step}.pt'))
    g_norm = normalizer(size=3)
    g_norm.load_normalizer(os.path.join(checkpoint_dir, f'g_norm_{step}.pt'))
    
    # Generate fixed evaluation setups with valid goals
    eval_setups = []
    max_attempts = 1000  # Maximum attempts to find valid goals
    attempts = 0
    
    while len(eval_setups) < num_episodes and attempts < max_attempts:
        eval_seed = seed + len(eval_setups) + attempts
        obs, _ = env.reset(seed=eval_seed)
        goal_pos = obs['desired_goal'].copy()
        
        if is_valid_goal(goal_pos):
            eval_setups.append({
                'seed': eval_seed,
                'goal': goal_pos
            })
        attempts += 1
    
    if len(eval_setups) < num_episodes:
        print(f"Warning: Could only find {len(eval_setups)} valid goals after {max_attempts} attempts")
    
    # Evaluate
    dis_returns = []
    undis_returns = []
    successes = []
    collisions = []
    
    for i in range(len(eval_setups)):
        setup = eval_setups[i]
        obs, _ = env.reset(seed=setup['seed'])
        
        rewards = []
        done = False
        episode_collision = False
        
        for t in range(50):  # max_path_length
            if image_obs:
                frame = env.render()
                img = Image.fromarray(frame)
                img = img.resize((84, 84), Image.BILINEAR)
                state = np.array(img)
                state = np.transpose(state, (2, 0, 1)) / 255.
            else:
                state = obs['observation']
                state = o_norm.normalize(state)
            
            goal = obs['desired_goal']
            goal = g_norm.normalize(goal)
            
            with torch.no_grad():
                input = (
                    torch.from_numpy(np.array(state, dtype=np.float32)).to(device),
                    torch.from_numpy(np.array(goal, dtype=np.float32)).to(device),
                    torch.from_numpy(np.array([1], dtype=np.float32)).to(device)  # test_horizon
                )
                action, _ = policy(*input)
            
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            
            # Check for collision using MuJoCo's contact detection
            if check_collision(env):
                episode_collision = True
                if human_render:
                    print(f"Episode {i+1} - Step {t}: COLLISION DETECTED!")
            
            done = terminated or truncated
            success = info['is_success']
            rewards.append(reward)
            
            if human_render:
                env.render()
            
            if done:
                break
        
        dis_return, undis_return = discounted_return(rewards, 0.98, reward_offset=True)
        dis_returns.append(dis_return)
        undis_returns.append(undis_return)
        successes.append(success)
        collisions.append(episode_collision)
    
    dis_return = np.mean(np.array(dis_returns))
    undis_return = np.mean(np.array(undis_returns))
    success_rate = np.mean(np.array(successes))
    collision_rate = np.mean(np.array(collisions))
    
    print(f"\n=== Evaluation Results ===")
    print(f"Discounted Return: {dis_return:.3f}")
    print(f"Undiscounted Return: {undis_return:.3f}")
    print(f"Success Rate: {success_rate:.3f}")
    print(f"Collision Rate: {collision_rate:.3f}")
    
    return dis_return, undis_return, success_rate, collision_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--env', type=str, default='FetchReach', help='Environment name')
    parser.add_argument('--image-obs', action='store_true', help='Whether the model uses image observations')
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--step', type=int, default=500, help='Step number for normalizer loading')
    parser.add_argument('--human-render', action='store_true', help='Whether to render the environment for human viewing')
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        env_name=args.env,
        image_obs=args.image_obs,
        num_episodes=args.num_episodes,
        seed=args.seed,
        step=args.step,
        human_render=args.human_render
    )


if __name__ == "__main__":
    main()

