#!/usr/bin/env python3
"""
Script 04: Train Merlin Policy

This script trains the Merlin policy using offline demonstration data.
The policy learns to reach goals in the FetchReach environment with obstacles.

Note: Obstacles must be added directly to the FetchReach environment source files.
See README.md for instructions on adding obstacles.
"""

import os
import sys
import pickle
import math
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Independent
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.buffer import ReplayBuffer
from components.normalizer import normalizer
from envs.fetch_reach_with_obstacles import create_fetch_reach_with_obstacles


HER_RATIO_TUNED = {
    'FetchReach': {'expert': 1.0, 'random': 1.0},
}

HORIZON_TUNED = {
    'FetchReach': {'expert': 1, 'random': 1},
}


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


def modify_state_to_goal(env_name, states, goal_dim):
    if env_name.startswith('FetchReach'):
        goal = states[:, :goal_dim]
    elif env_name.startswith('Fetch'):
        goal = states[:, goal_dim:2*goal_dim]
    else:
        goal = states[:, :goal_dim]
    return goal


def discounted_return(rewards, gamma, reward_offset=True):
    L = len(rewards)
    if type(rewards[0]) == np.ndarray and len(rewards[0]):
        rewards = np.array(rewards).T
    else:
        rewards = np.array(rewards).reshape(1, L)

    if reward_offset:
        rewards += 1   # positive offset

    discount_weights = np.power(gamma, np.arange(L)).reshape(1, -1)
    dis_return = (rewards * discount_weights).sum(axis=1)
    undis_return = rewards.sum(axis=1)
    return dis_return.mean(), undis_return.mean()


class OfflineMerlin:
    def __init__(self,
        env,
        dataset_path,
        device,
        seed,
        max_timesteps,
        max_path_length,
        test_horizon,
        her_ratio,
        image_obs=False,
        obstacles=False
    ):
        self.env = env
        self.dataset_path = dataset_path
        self.action_space = env.action_space
        self.action_dim = self.action_space.shape[0]
        self.action_dtype = self.action_space.dtype
        self.goal_space = env.observation_space['desired_goal']
        self.goal_dim = self.goal_space.shape[0]
        self.goal_dtype = self.goal_space.dtype
        self.image_obs = image_obs
        self.obstacles = obstacles

        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)

        with open(self.dataset_path, 'rb') as f:
            data = pickle.load(f)

        states = data['o'][0,0]
        self.state_shape = states.shape
        self.state_dtype = states.dtype

        # Setup directories
        base_dir = './logs'
        if obstacles:
            base_dir = os.path.join(base_dir, "obstacles")
        os.makedirs(base_dir, exist_ok=True)
        
        self.log_dir = os.path.join(base_dir, 'logs')
        self.checkpoint_dir = os.path.join(base_dir, 'checkpoints')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.device = device
        self.max_timesteps = max_timesteps
        self.max_path_length = max_path_length
        self.test_horizon = test_horizon
        self.her_ratio = her_ratio

        max_buffer_size = 2000

        self.policy = Policy(
            self.state_shape,
            self.goal_dim,
            32, 256,
            2*self.action_dim,
            max_path_length,
            self.action_space.high[0]
        ).to(self.device)

        self.dataset_replay_buffer = ReplayBuffer(
            self.state_shape,
            self.state_dtype,
            self.action_dim,
            self.action_dtype,
            self.goal_dim,
            self.goal_dtype,
            self.max_path_length,
            self.her_ratio,
            buffer_size=max_buffer_size,
        )

        self.policy_updates_per_step = None
        self.optimizer = optim.Adam(self.policy.parameters(), lr=5e-4)
        self.batch_size = 512

        # create the normalizer
        if not self.image_obs:
            self.o_norm = normalizer(size=self.state_shape[0])
        self.g_norm = normalizer(size=self.goal_dim)

        self.setup_data_and_models()

    def preprocess_and_load_dataset(self):
        with open(self.dataset_path, 'rb') as f:
            data = pickle.load(f)

        states = data['o']
        actions = data['u']
        goals = data['g'][:, 0]
        achieved_goals = data['ag']

        n_trajectories = states.shape[0]
        for i in range(n_trajectories):
            self.dataset_replay_buffer.add_trajectory(states[i], actions[i], goals[i], achieved_goals[i])

    def setup_data_and_models(self):
        self.preprocess_and_load_dataset()
        start = time.time()
        print('Model setup time: {}'.format(time.time()-start))

    def buffer_sample(self, buffer, batch_size):
        data = []
        for sub_buffer, ratio in buffer:
            sub_batch_size = int(batch_size * ratio + 0.1)
            sub_data = sub_buffer.sample_batch(sub_batch_size)
            data.append(sub_data)
            
        output = []
        for item in zip(*data):
            sub_data = np.concatenate(item, axis=0)
            output.append(sub_data)
        return output

    def train_policy(self):
        # setup buffer
        buffer = [(self.dataset_replay_buffer, 1.)]

        # train policy
        if self.policy_updates_per_step is None:
            self.policy_updates_per_step = (self.max_path_length) * 10

        training_steps = 0
        train_losses = []

        best_score = 0
        dis_returns, undis_returns, successes = [], [], []

        while training_steps < self.max_timesteps:
            training_steps += self.policy_updates_per_step
            _losses = []

            self.policy.train()
            for _ in range(self.policy_updates_per_step):
                _states, _actions, _goals, _, _horizons, _, _ = self.buffer_sample(buffer, self.batch_size)

                if not self.image_obs:
                    self.o_norm.update(_states)
                    self.o_norm.recompute_stats()
                
                self.g_norm.update(_goals)
                self.g_norm.recompute_stats()

                if self.image_obs:
                    _states = np.transpose(_states, (0, 3, 1, 2)) / 255.
                else:
                    _states = self.o_norm.normalize(_states)
                _goals = self.g_norm.normalize(_goals)

                _states = torch.from_numpy(np.array(_states, dtype=np.float32)).to(self.device)
                _goals = torch.from_numpy(np.array(_goals, dtype=np.float32)).to(self.device)
                _actions = torch.from_numpy(np.array(_actions, dtype=np.float32)).to(self.device)
                _horizons = torch.from_numpy(np.array(_horizons, dtype=np.float32)).to(self.device)
    
                mus, sigmas = self.policy(_states, _goals, _horizons)
                dist = Independent(Normal(mus, sigmas), 1)
                loss = -dist.log_prob(_actions).mean()

                self.optimizer.zero_grad()
                loss.backward()

                # Perform gradient clipping
                max_grad_norm = 1.0
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)

                self.optimizer.step()
                _losses.append(np.mean(np.array(loss.item())))

            train_losses.append(np.mean(np.array(_losses)))
            
            print("Timesteps: {}, Loss: {}".format(training_steps, np.mean(np.array(loss.item()))))
            if training_steps % 500 == 0:
                dis, undis, succ = self.eval_policy(100, training_steps)
                dis_returns.append(dis)
                undis_returns.append(undis)
                successes.append(succ)
                np.save(os.path.join(self.log_dir, 'dis_returns.npy'), np.array(dis_returns))
                np.save(os.path.join(self.log_dir, 'undis_returns.npy'), np.array(undis_returns))
                np.save(os.path.join(self.log_dir, 'successes.npy'), np.array(successes))

                if dis > best_score:
                    torch.save(self.policy.state_dict(), os.path.join(self.checkpoint_dir, f'best_policy_{training_steps}.pt'))
                    # Save normalizers
                    if not self.image_obs:
                        self.o_norm.save_normalizer(os.path.join(self.checkpoint_dir, f'o_norm_{training_steps}.pt'))
                    self.g_norm.save_normalizer(os.path.join(self.checkpoint_dir, f'g_norm_{training_steps}.pt'))
                    best_score = dis
            
        dis, undis, succ = self.eval_policy(100, training_steps)
        dis_returns.append(dis)
        undis_returns.append(undis)
        successes.append(succ)
        np.save(os.path.join(self.log_dir, 'dis_returns.npy'), np.array(dis_returns))
        np.save(os.path.join(self.log_dir, 'undis_returns.npy'), np.array(undis_returns))
        np.save(os.path.join(self.log_dir, 'successes.npy'), np.array(successes))

    def eval_policy(self, num_episodes, iterations):
        dis_returns = []
        undis_returns = []
        successes = []
        self.policy.eval()
        
        for i in range(num_episodes):
            obs, _ = self.env.reset(seed=i)
            
            rewards = []
            done = False
            
            for t in range(self.max_path_length):
                if self.image_obs:
                    frame = self.env.render()
                    img = Image.fromarray(frame)
                    img = img.resize((84, 84), Image.BILINEAR)
                    state = np.array(img)
                    state = np.transpose(state, (2, 0, 1)) / 255.
                else:
                    state = obs['observation']
                    state = self.o_norm.normalize(state)
                goal = obs['desired_goal']
                goal = self.g_norm.normalize(goal)

                with torch.no_grad():
                    input = (
                        torch.from_numpy(np.array(state, dtype=np.float32)).to(self.device),
                        torch.from_numpy(np.array(goal, dtype=np.float32)).to(self.device),
                        torch.from_numpy(np.array([self.test_horizon], dtype=np.float32)).to(self.device)
                    )
                    action, _ = self.policy(*input)
                
                obs, reward, terminated, truncated, info = self.env.step(action.cpu().numpy())
                
                done = terminated or truncated
                success = info['is_success']
                rewards.append(reward)

                if done:
                    break

            dis_return, undis_return = discounted_return(rewards, 0.98, reward_offset=True)
            dis_returns.append(dis_return)
            undis_returns.append(undis_return)
            successes.append(success)
        
        dis_return = np.mean(np.array(dis_returns))
        undis_return = np.mean(np.array(undis_returns))
        success = np.mean(np.array(successes))
        print("Iterations: {}, Dis Return: {}, Undis Return: {}, Success: {}".format(
            iterations, dis_return, undis_return, success))

        return dis_return, undis_return, success


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", type=str, default='FetchReach')
    parser.add_argument("--variant", type=str, default='expert')
    parser.add_argument("--max-timesteps", type=int, default=5e5)
    parser.add_argument("--max-path-length", type=int, default=50)
    parser.add_argument("--test-horizon", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--her-ratio", type=float, default=1.0)
    parser.add_argument("--image-obs", action='store_true')
    parser.add_argument("--obstacles", action='store_true')
    parser.add_argument("--dataset-path", type=str, required=True,
                      help='Path to demonstration dataset')
    
    args = parser.parse_args()

    # override with tuned hyperparameters
    args.her_ratio = HER_RATIO_TUNED[args.task][args.variant]
    args.test_horizon = HORIZON_TUNED[args.task][args.variant]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment
    render_mode = "rgb_array"
    env = create_fetch_reach_with_obstacles(render_mode=render_mode)
    env._max_episode_steps = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = OfflineMerlin(env, args.dataset_path, device, args.seed, args.max_timesteps,
                            args.max_path_length, args.test_horizon, args.her_ratio,
                            image_obs=args.image_obs, obstacles=args.obstacles)
    model.train_policy()


if __name__ == "__main__":
    main()

