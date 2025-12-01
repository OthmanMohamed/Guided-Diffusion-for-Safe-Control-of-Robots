#!/usr/bin/env python3
"""
Script 06: Evaluate Policy with Safety

This script evaluates a trained Merlin policy with safety model integration.
It measures performance metrics and safety scores during evaluation.

Note: Obstacles must be added directly to the FetchReach environment source files.
See README.md for instructions on adding obstacles.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import mujoco

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.normalizer import normalizer
from envs.fetch_reach_with_obstacles import create_fetch_reach_with_obstacles

import math

# Import Policy and discounted_return (duplicate from 04_train_merlin_policy)
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


# Import SafetyModel
class SafetyModel(nn.Module):
    """Neural network for predicting safety scores."""
    
    def __init__(self, observation_dim=10, action_dim=4, hidden_dims=[128, 64, 32]):
        super(SafetyModel, self).__init__()
        input_dim = observation_dim + action_dim
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=1)
        return self.network(x)


def is_valid_goal(goal_pos):
    """
    Check if a goal position is valid (not inside any obstacles).
    
    Note: This function uses example obstacle positions. Adjust the obstacle_pos
    and obstacle_size arrays to match the obstacles defined in your environment source files.
    The obstacles should already be defined in the FetchReach environment XML/model files.
    
    Args:
        goal_pos: Goal position as numpy array [x, y, z]
    
    Returns:
        bool: True if goal is valid (not inside obstacles), False otherwise
    """
    # Example obstacle positions - adjust these to match your actual obstacle definitions
    # These should correspond to obstacles defined in the environment source files
    obstacle_pos = np.array([1.25, 0.75, 0.42])  # Example obstacle position
    obstacle_size = np.array([0.05, 0.05, 0.4])  # Example obstacle size
    
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
    
    Note: This function assumes obstacles are already defined in the environment source files.
    The obstacle names (e.g., "obstacle1_geom") should match those in your environment XML/model files.
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


def load_safety_model(safety_model_path, device):
    """
    Load the trained safety model.
    
    Args:
        safety_model_path (str): Path to the safety model checkpoint
        device (torch.device): Device to load the model on
    
    Returns:
        SafetyModel: Loaded safety model
    """
    # Load the checkpoint
    checkpoint = torch.load(safety_model_path, map_location=device)
    
    # Initialize the safety model with the same architecture as used during training
    safety_model = SafetyModel(
        observation_dim=10,  # FetchReach observation dimension
        action_dim=4,        # FetchReach action dimension
        hidden_dims=[128, 64, 64, 32]  # Default architecture from training
    ).to(device)
    
    # Load the model state
    safety_model.load_state_dict(checkpoint['model_state_dict'])
    safety_model.eval()
    
    print(f"Safety model loaded from: {safety_model_path}")
    print(f"Model was trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")
    
    return safety_model


def predict_safety_score(safety_model, observation, action, device):
    """
    Predict safety score for a given observation and action.
    
    Args:
        safety_model (SafetyModel): Loaded safety model
        observation (np.ndarray): Current observation
        action (np.ndarray): Current action
        device (torch.device): Device to run inference on
    
    Returns:
        float: Safety score between 0 and 1
    """
    with torch.no_grad():
        # Convert to tensors
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
        
        # Get safety prediction
        safety_score = safety_model(obs_tensor, action_tensor)
        
        return safety_score.item()


def evaluate_checkpoint_with_safety(checkpoint_path, safety_model_path, env_name, image_obs=False,
                                  num_episodes=100, seed=0, human_render=False, step=500,
                                  print_safety_scores=False):
    """
    Evaluate a checkpointed model deterministically with safety model integration.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint (.pt file)
        safety_model_path (str): Path to the safety model checkpoint (.pt file)
        env_name (str): Name of the environment (e.g., 'FetchReach')
        image_obs (bool): Whether the model uses image observations
        num_episodes (int): Number of evaluation episodes
        seed (int): Random seed for reproducibility
        human_render (bool): Whether to render the environment for human viewing
        step (int): Step number for normalizer loading
        print_safety_scores (bool): Whether to print safety scores at each step
    
    Returns:
        tuple: (discounted_return, undiscounted_return, success_rate, collision_rate, avg_safety_score)
    """
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment with appropriate render mode
    render_mode = "human" if human_render else "rgb_array"
    env = create_fetch_reach_with_obstacles(render_mode=render_mode)
    env._max_episode_steps = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load safety model
    safety_model = load_safety_model(safety_model_path, device)
    
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
    episode_safety_scores = []
    
    for i in range(len(eval_setups)):
        setup = eval_setups[i]
        obs, _ = env.reset(seed=setup['seed'])
        
        rewards = []
        done = False
        episode_collision = False
        episode_safety = []
        
        if print_safety_scores:
            print(f"\n=== Episode {i+1}/{len(eval_setups)} ===")
            print(f"Goal: {setup['goal']}")
        
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
            
            # Get safety score for current observation and action
            safety_score = predict_safety_score(safety_model, obs['observation'], action.cpu().numpy(), device)
            episode_safety.append(safety_score)
            
            # Print safety score if requested
            if print_safety_scores:
                safety_status = "SAFE" if safety_score > 0.5 else "UNSAFE"
                print(f"Step {t:2d}: Safety Score = {safety_score:.4f} ({safety_status})")
            
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            
            # Check for collision using MuJoCo's contact detection
            if check_collision(env):
                episode_collision = True
                if print_safety_scores:
                    print(f"Step {t:2d}: COLLISION DETECTED!")
            
            done = terminated or truncated
            success = info['is_success']
            rewards.append(reward)
            
            if human_render:
                env.render()
            
            if done:
                break
        
        # Episode summary
        avg_episode_safety = np.mean(episode_safety)
        episode_safety_scores.append(avg_episode_safety)
        
        dis_return, undis_return = discounted_return(rewards, 0.98, reward_offset=True)
        dis_returns.append(dis_return)
        undis_returns.append(undis_return)
        successes.append(success)
        collisions.append(episode_collision)
        
        if print_safety_scores:
            print(f"Episode {i+1} Summary:")
            print(f"  Success: {success}")
            print(f"  Collision: {episode_collision}")
            print(f"  Average Safety Score: {avg_episode_safety:.4f}")
            print(f"  Discounted Return: {dis_return:.3f}")
            print(f"  Undiscounted Return: {undis_return:.3f}")
    
    # Overall results
    dis_return = np.mean(np.array(dis_returns))
    undis_return = np.mean(np.array(undis_returns))
    success_rate = np.mean(np.array(successes))
    collision_rate = np.mean(np.array(collisions))
    avg_safety_score = np.mean(np.array(episode_safety_scores))
    
    print(f"\n=== Overall Evaluation Results ===")
    print(f"Discounted Return: {dis_return:.3f}")
    print(f"Undiscounted Return: {undis_return:.3f}")
    print(f"Success Rate: {success_rate:.3f}")
    print(f"Collision Rate: {collision_rate:.3f}")
    print(f"Average Safety Score: {avg_safety_score:.4f}")
    
    # Safety analysis
    print(f"\n=== Safety Analysis ===")
    safe_episodes = sum(1 for score in episode_safety_scores if score > 0.5)
    unsafe_episodes = len(episode_safety_scores) - safe_episodes
    print(f"Episodes with avg safety > 0.5: {safe_episodes}/{len(episode_safety_scores)} ({100*safe_episodes/len(episode_safety_scores):.1f}%)")
    print(f"Episodes with avg safety <= 0.5: {unsafe_episodes}/{len(episode_safety_scores)} ({100*unsafe_episodes/len(episode_safety_scores):.1f}%)")
    
    # Correlation analysis
    if len(collisions) > 0:
        collision_safety_correlation = np.corrcoef(collisions, episode_safety_scores)[0, 1]
        print(f"Correlation between collisions and safety scores: {collision_safety_correlation:.3f}")
    
    return dis_return, undis_return, success_rate, collision_rate, avg_safety_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--safety-model', type=str, required=True,
                      help='Path to the safety model checkpoint')
    parser.add_argument('--env', type=str, default='FetchReach', help='Environment name')
    parser.add_argument('--image-obs', action='store_true', help='Whether the model uses image observations')
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--step', type=int, default=500, help='Step number for normalizer loading')
    parser.add_argument('--human-render', action='store_true', help='Whether to render the environment for human viewing')
    parser.add_argument('--print-safety-scores', action='store_true', help='Print safety scores at each step')
    args = parser.parse_args()

    evaluate_checkpoint_with_safety(
        checkpoint_path=args.checkpoint,
        safety_model_path=args.safety_model,
        env_name=args.env,
        image_obs=args.image_obs,
        num_episodes=args.num_episodes,
        seed=args.seed,
        step=args.step,
        human_render=args.human_render,
        print_safety_scores=args.print_safety_scores
    )


if __name__ == "__main__":
    main()

