import os
import pickle
import math
import argparse
import time

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Independent

from src.envs import register_envs
from src.components.buffer import ReplayBuffer
from src.components.dynamics_model import train_dynamics, train_reverse_bc
from src.components.representations import train_contrastive_encoder
from src.components.normalizer import normalizer
from src.components.trajectory_stitching import StateTree
import mujoco

from PIL import Image


HER_RATIO_TUNED = {
    'PointReach': {'expert': 1.0, 'random': 1.0},
    'PointRooms': {'expert': 0.2, 'random': 1.0},
    'Reacher': {'expert': 0.2, 'random': 1.0},
    'SawyerReach': {'expert': 1.0, 'random': 1.0},
    'SawyerDoor': {'expert': 0.2, 'random': 1.0},
    'FetchReach': {'expert': 1.0, 'random': 1.0},
    'FetchPush': {'expert': 0.0, 'random': 0.2},
    'FetchPick': {'expert': 0.0, 'random': 0.5},
    'FetchSlide': {'expert': 0.2, 'random': 0.8},
    'HandReach': {'expert': 1.0, 'random': 1.0},
}

HORIZON_TUNED = {
    'PointReach': {'expert': 1, 'random': 1},
    'PointRooms': {'expert': 1, 'random': 1},
    'Reacher': {'expert': 5, 'random': 5},
    'SawyerReach': {'expert': 1, 'random': 1},
    'SawyerDoor': {'expert': 5, 'random': 5},
    'FetchReach': {'expert': 1, 'random': 1},
    'FetchPush': {'expert': 20, 'random': 20},
    'FetchPick': {'expert': 10, 'random': 50},
    'FetchSlide': {'expert': 10, 'random': 10},
    'HandReach': {'expert': 1, 'random': 1},
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
    def __init__(self, state_shape, goal_dim,\
                 timestep_embedding_dim, hidden_dim, output_dim, \
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


def build_env(task, render_mode="rgb_array"):
    env_dict = {
        'PointReach': 'Point2DLargeEnv-v1',
        'PointRooms': 'Point2D-FourRoom-v1',
        'Reacher': 'Reacher-v2',
        'SawyerReach': 'SawyerReachXYZEnv-v1',
        'SawyerDoor': 'SawyerDoor-v0',
        'FetchReach': 'FetchReach-v2',
        'FetchPush': 'FetchPush-v2',
        'FetchPick': 'FetchPickAndPlace-v2',
        'FetchSlide': 'FetchSlide-v2',
        'HandReach': 'HandReach-v1',
    }
    env_id = env_dict[task]
    if task.startswith("Fetch"):
        env = gym.make(env_id, render_mode=render_mode)
    else:
        env = gym.make(env_id)
    if env_id.startswith('Fetch'):
        from envs.multi_world_wrapper import FetchGoalWrapper
        env._max_episode_steps = 50
        # env = FetchGoalWrapper(env)
    elif env_id.startswith('Hand'):
        env._max_episode_steps = 50
    elif env_id.startswith('Sawyer'):
        from envs.multi_world_wrapper import SawyerGoalWrapper
        env = SawyerGoalWrapper(env)
        if not hasattr(env, '_max_episode_steps'):
            env = gym.wrappers.TimeLimit(env, max_episode_steps=50)
    elif env_id.startswith('Point'):
        from envs.multi_world_wrapper import PointGoalWrapper
        env = gym.wrappers.TimeLimit(env, max_episode_steps=50)
        env = PointGoalWrapper(env)
    elif env_id.startswith('Reacher'):
        from envs.multi_world_wrapper import ReacherGoalWrapper
        env._max_episode_steps = 50
        env = ReacherGoalWrapper(env)
    else:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=50)
    return env

def modify_state_to_goal(env_name, states, goal_dim):
    if env_name.startswith('Point'):
        goal = states
    elif env_name.startswith('SawyerReach'):
        goal = states
    elif env_name.startswith('SawyerDoor'):
        goal = states[:, -goal_dim:]
    elif env_name.startswith('Reacher'):
        goal = states[:, -goal_dim-1:-1]
    elif env_name.startswith('FetchReach'):
        goal = states[:, :goal_dim]
    elif env_name.startswith('Fetch'):
        goal = states[:, goal_dim:2*goal_dim]
    elif env_name.startswith('HandReach'):
        goal = states[:, -goal_dim:]
    return goal

def setup_logging(base_dir, env_name, variant, seed, diffusion_rollout, diffusion_nonparam, obstacles):
    if obstacles: base_dir = os.path.join(base_dir, "obstacles/")
    log_path = os.path.join(base_dir, 'logs', env_name, variant, str(seed))
    checkpoint_path = os.path.join(base_dir, 'checkpoints', env_name, variant, str(seed))
    buffer_path = os.path.join(log_path, 'diffusion_buffers', env_name, variant, str(seed))
    if diffusion_nonparam:
        log_path = os.path.join(log_path, 'forward_diffusion_nonparam')
        checkpoint_path = os.path.join(checkpoint_path, 'forward_diffusion_nonparam')
        buffer_path = os.path.join(buffer_path, 'forward_diffusion_nonparam')
    else:
        log_path = os.path.join(log_path, 'forward_diffusion_model')
        checkpoint_path = os.path.join(checkpoint_path, 'forward_diffusion_model')
        buffer_path = os.path.join(buffer_path, 'forward_diffusion_model')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(buffer_path, exist_ok=True)
    return log_path, checkpoint_path, buffer_path


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
        latent=False,
        diffusion_nonparam=False,
        diffusion_rollout=True,
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
        
        # Generate fixed evaluation setups
        self.eval_setups = []
        for i in range(100):  # 100 evaluation episodes
            # Use a different seed for each evaluation setup
            eval_seed = seed + i
            obs, _ = env.reset(seed=eval_seed)
            self.eval_setups.append({
                'seed': eval_seed,
                'goal': obs['desired_goal'].copy()
            })

        if diffusion_nonparam:
            assert diffusion_rollout, 'Diffusion nonparam only supported with rollout'

        with open(self.dataset_path, 'rb') as f:
            data = pickle.load(f)

        states = data['o'][0,0]
        self.state_shape = states.shape
        self.state_dtype = states.dtype

        variant = dataset_path.split('/')[-3]
        env_name = dataset_path.split('/')[-2]
        if self.image_obs:
            self.log_dir, self.checkpoint_dir, self.buffer_dir = setup_logging('./logging/img_obs', \
                                                                                env_name, variant, seed,
                                                                                diffusion_rollout,
                                                                                diffusion_nonparam, obstacles)
        else:
            self.log_dir, self.checkpoint_dir, self.buffer_dir = setup_logging('./logging/state_obs',
                                                                                env_name, variant, seed,
                                                                                diffusion_rollout,
                                                                                diffusion_nonparam, obstacles)

        self.device = device
        self.max_timesteps = max_timesteps
        self.max_path_length = max_path_length
        self.test_horizon = test_horizon
        self.her_ratio = her_ratio
        self.latent = latent
        self.encoder = None

        if dataset_path.split('/')[-2] in ['FetchPick', 'FetchPush', 'FetchSlide', 'HandReach']:
            max_buffer_size = 40000
        else:
            max_buffer_size = 2000

        self.dynamics = None
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

        return

    def setup_data_and_models(self):
        self.preprocess_and_load_dataset()
        start = time.time()
        print('Model setup time: {}'.format(time.time()-start))        

    def forward_model_noise(self):
        states = np.zeros((self.rollout_batch_size, self.max_path_length, *self.state_shape))
        actions = np.zeros((self.rollout_batch_size, self.max_path_length-1, self.action_dim))
        achieved_states = np.zeros((self.rollout_batch_size, self.max_path_length, self.goal_dim))

        next_state, goal = self.dataset_replay_buffer.sample_goals(self.rollout_batch_size)

        T = self.max_path_length 
        states[:, T-1] = next_state
        achieved_states[:, T-1] = goal

        h = None
        for t in range(1, T):
            action = self.reverse_bc.select_action(next_state)
            state, h = self.reverse_dynamics.step(T-t-1, next_state, action, h)
            
            achieved_state = modify_state_to_goal(self.env.unwrapped.spec.id, state, self.goal_dim)

            states[:, T-t-1] = state
            actions[:, T-t-1] = action
            achieved_states[:, T-t-1] = achieved_state

            next_state = state

        states = states[~np.isnan(states).any(axis=1).any(axis=-1)]
        actions = actions[~np.isnan(actions).any(axis=1).any(axis=-1)]
        achieved_states = achieved_states[~np.isnan(achieved_states).any(axis=1).any(axis=-1)]
        return states, actions, achieved_states, goal

    def forward_tree_noise(self):
        states = np.zeros((self.rollout_batch_size, self.max_path_length, *self.state_shape))
        actions = np.zeros((self.rollout_batch_size, self.max_path_length-1, self.action_dim))
        achieved_states = np.zeros((self.rollout_batch_size, self.max_path_length, self.goal_dim))

        next_state, goal = self.dataset_replay_buffer.sample_goals(self.rollout_batch_size)

        T = self.max_path_length 
        states[:, T-1] = next_state
        achieved_states[:, T-1] = goal

        for t in range(1, T):
            buffer_idx = self.tree.query(next_state)

            delete_idx = np.nonzero(buffer_idx % self.max_path_length == 0)[0]
            buffer_idx = np.delete(buffer_idx, delete_idx)
            if len(buffer_idx) == 0:
                states = np.zeros(0)
                break
            goal = np.delete(goal, delete_idx, axis=0)
            states = np.delete(states, delete_idx, axis=0)
            actions = np.delete(actions, delete_idx, axis=0)
            achieved_states = np.delete(achieved_states, delete_idx, axis=0)

            traj_idx = buffer_idx // self.max_path_length
            time_idx = buffer_idx % self.max_path_length

            state, action, next_state, achieved_state = self.dataset_replay_buffer.get_transition(traj_idx, time_idx)

            states[:, T-t-1] = state
            actions[:, T-t-1] = action
            achieved_states[:, T-t-1] = achieved_state

            next_state = state

        return states, actions, achieved_states, goal

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
        dis_returns, undis_returns, successess = [], [], []

        if os.path.exists(os.path.join(self.checkpoint_dir, 'best_policy.pt')) and False:
            self.policy.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, "best_policy.pt")))
            while training_steps < self.max_timesteps:
                training_steps += self.policy_updates_per_step
                for _ in range(self.policy_updates_per_step):
                    _states, _actions, _goals, _, _horizons, _ = self.buffer_sample(buffer, self.batch_size)
                    self.o_norm.update(_states)
                    self.g_norm.update(_goals)
                    self.o_norm.recompute_stats()
                    self.g_norm.recompute_stats()
                    print(f"{training_steps}", end="\r")
        else:
            while training_steps < self.max_timesteps:
                training_steps += self.policy_updates_per_step
                _losses = []

                self.policy.train()
                for _ in range(self.policy_updates_per_step):
                    _states, _actions, _goals, _, _horizons, _ = self.buffer_sample(buffer, self.batch_size)

                    if not self.image_obs:
                        self.o_norm.update(_states)
                        self.o_norm.recompute_stats()
                    
                    self.g_norm.update(_goals)                        
                    self.g_norm.recompute_stats()

                    if self.image_obs:
                        _states = np.transpose(_states, (0, 3, 1, 2)) / 255.
                        # _states.fill(0)
                        # _states = np.random.rand(*_states.shape)
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
                    max_grad_norm = 1.0  # Set the maximum gradient norm threshold
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)

                    self.optimizer.step()
                    _losses.append(np.mean(np.array(loss.item())))

                train_losses.append(np.mean(np.array(_losses)))
                
                print("Timesteps: {}, Loss: {}".format(training_steps, np.mean(np.array(loss.item()))))
                if training_steps % 500 == 0:
                    dis, undis, succ = self.eval_policy(100, training_steps)
                    dis_returns.append(dis)
                    undis_returns.append(undis)
                    successess.append(succ)
                    np.save(os.path.join(self.log_dir, '{}_{}_dis_returns.npy'.format(int(10*self.her_ratio), self.test_horizon)), np.array(dis_returns))
                    np.save(os.path.join(self.log_dir, '{}_{}_undis_returns.npy'.format(int(10*self.her_ratio), self.test_horizon)), np.array(undis_returns))
                    np.save(os.path.join(self.log_dir, '{}_{}_successes.npy'.format(int(10*self.her_ratio), self.test_horizon)), np.array(successess))

                    if dis > best_score or True:
                        torch.save(self.policy.state_dict(), os.path.join(self.checkpoint_dir, f'best_policy_{training_steps}.pt'))
                        # Save normalizers
                        if not self.image_obs:
                            self.o_norm.save_normalizer(os.path.join(self.checkpoint_dir, f'o_norm_{training_steps}.pt'))
                        self.g_norm.save_normalizer(os.path.join(self.checkpoint_dir, f'g_norm_{training_steps}.pt'))
                        best_score = dis
            
        dis, undis, succ = self.eval_policy(100, training_steps)
        dis_returns.append(dis)
        undis_returns.append(undis)
        successess.append(succ)
        np.save(os.path.join(self.log_dir, '{}_{}_dis_returns.npy'.format(int(10*self.her_ratio), self.test_horizon)), np.array(dis_returns))
        np.save(os.path.join(self.log_dir, '{}_{}_undis_returns.npy'.format(int(10*self.her_ratio), self.test_horizon)), np.array(undis_returns))
        np.save(os.path.join(self.log_dir, '{}_{}_successes.npy'.format(int(10*self.her_ratio), self.test_horizon)), np.array(successess))

    def is_valid_goal(self, goal_pos):
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

    def check_collision(self, env):
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

    def eval_policy(self, num_episodes, iterations, plot=False):
        dis_returns = []
        undis_returns = []
        successes = []
        collisions = []
        self.policy.eval()
        save_dir = f"eval_images"
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate fixed evaluation setups with valid goals
        eval_setups = []
        max_attempts = 1000  # Maximum attempts to find valid goals
        attempts = 0
        
        while len(eval_setups) < num_episodes and attempts < max_attempts:
            eval_seed = 0 + len(eval_setups) + attempts
            obs, _ = self.env.reset(seed=eval_seed)
            goal_pos = obs['desired_goal'].copy()
            
            if self.is_valid_goal(goal_pos):
                eval_setups.append({
                    'seed': eval_seed,
                    'goal': goal_pos
                })
            attempts += 1
        
        if len(eval_setups) < num_episodes:
            print(f"Warning: Could only find {len(eval_setups)} valid goals after {max_attempts} attempts")
        
        for i in range(len(eval_setups)):
            # Use the fixed evaluation setup
            setup = eval_setups[i]
            obs, _ = self.env.reset(seed=setup['seed'])
            
            rewards = []
            states, actions, goals = [], [], []
            done = False
            episode_collision = False
            
            for t in range(self.max_path_length):
                traj_dir = os.path.join(save_dir, f"traj_{i}")
                os.makedirs(traj_dir, exist_ok=True)
                
                if self.image_obs:
                    frame = self.env.render()
                    img = Image.fromarray(frame)
                    img.save(os.path.join(traj_dir, f"frame_{t}.png"))
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
                states.append(obs['observation'])
                actions.append(action.cpu().numpy())
                goals.append(obs['desired_goal'])

                obs, reward, terminated, truncated, info = self.env.step(action.cpu().numpy())
                
                # Check for collision using MuJoCo's contact detection
                if self.check_collision(self.env):
                    episode_collision = True
                
                done = terminated or truncated
                success = info['is_success']
                rewards.append(reward)

                if done:
                    break

            dis_return, undis_return = discounted_return(rewards, 0.98, reward_offset=True)
            dis_returns.append(dis_return)
            undis_returns.append(undis_return)
            successes.append(success)
            collisions.append(episode_collision)

            if plot:
                states.append(obs['observation'])
                states = np.array(states)
                actions = np.array(actions)
                goals = np.array(goals)
                fig, ax = plt.subplots(figsize=(6, 6))
                self.env.plot_trajectory(ax, states, actions, goals[0])
                fig.savefig('{}_{}.png'.format(self.env.unwrapped.spec.id, i))
        
        dis_return = np.mean(np.array(dis_returns))
        undis_return = np.mean(np.array(undis_returns))
        success = np.mean(np.array(successes))
        collision_rate = np.mean(np.array(collisions))
        print("Iterations: {}, Dis Return: {}, Undis Return: {}, Success: {}, Collision Rate: {}".format(
            iterations, dis_return, undis_return, success, collision_rate))

        return dis_return, undis_return, success

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


def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    # env args
    parser.add_argument("--task", type=str, default='PointRooms')
    parser.add_argument("--variant", type=str, default='expert')
    parser.add_argument("--max-timesteps", type=int, default=5e5)
    parser.add_argument("--max-path-length", type=int, default=50)
    parser.add_argument("--test-horizon", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--her-ratio", type=float, default=0.2)
    parser.add_argument("--stitching-radius", type=float, default=0.9999)
    parser.add_argument("--image-obs", action='store_true')
    parser.add_argument("--obstacles", action='store_true')
    return parser.parse_args()

def evaluate_checkpoint(checkpoint_path, env_name, image_obs=False, num_episodes=100, seed=0):
    """
    Evaluate a checkpointed model deterministically.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint (.pt file)
        env_name (str): Name of the environment (e.g., 'FetchReach')
        image_obs (bool): Whether the model uses image observations
        num_episodes (int): Number of evaluation episodes
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (discounted_return, undiscounted_return, success_rate)
    """
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment
    env = build_env(env_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = OfflineMerlin(
        env=env,
        dataset_path="dummy_path",  # Not used for evaluation
        device=device,
        seed=seed,
        max_timesteps=0,  # Not used for evaluation
        max_path_length=50,
        test_horizon=50,
        her_ratio=1.0,
        image_obs=image_obs
    )
    
    # Load checkpoint
    model.policy.load_state_dict(torch.load(checkpoint_path))
    model.policy.eval()
    
    # Generate fixed evaluation setups
    eval_setups = []
    for i in range(num_episodes):
        eval_seed = seed + i
        obs, _ = env.reset(seed=eval_seed)
        eval_setups.append({
            'seed': eval_seed,
            'goal': obs['desired_goal'].copy()
        })
    
    # Evaluate
    dis_returns = []
    undis_returns = []
    successes = []
    
    for i in range(num_episodes):
        setup = eval_setups[i]
        obs, _ = env.reset(seed=setup['seed'])
        
        rewards = []
        done = False
        
        for t in range(model.max_path_length):
            if image_obs:
                frame = env.render()
                img = Image.fromarray(frame)
                img = img.resize((84, 84), Image.BILINEAR)
                state = np.array(img)
                state = np.transpose(state, (2, 0, 1)) / 255.
            else:
                state = obs['observation']
                state = model.o_norm.normalize(state)
            
            goal = obs['desired_goal']
            goal = model.g_norm.normalize(goal)
            
            with torch.no_grad():
                input = (
                    torch.from_numpy(np.array(state, dtype=np.float32)).to(device),
                    torch.from_numpy(np.array(goal, dtype=np.float32)).to(device),
                    torch.from_numpy(np.array([model.test_horizon], dtype=np.float32)).to(device)
                )
                action, _ = model.policy(*input)
            
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
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
    success_rate = np.mean(np.array(successes))
    
    print(f"Evaluation Results:")
    print(f"Discounted Return: {dis_return:.3f}")
    print(f"Undiscounted Return: {undis_return:.3f}")
    print(f"Success Rate: {success_rate:.3f}")
    
    return dis_return, undis_return, success_rate

if __name__ == "__main__":
    args = get_args()
    register_envs()

    # override with tuned hyperparameters, uncomment for tuning
    args.her_ratio = HER_RATIO_TUNED[args.task][args.variant]
    args.test_horizon = HORIZON_TUNED[args.task][args.variant]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = build_env(args.task)

    if not args.obstacles:
        dataset_path = "/home/othman/fetchReachExp/data/buffer.pkl" if not args.image_obs else "/home/othman/fetchReachExp/data/image_buffer.pkl"
    else:
        dataset_path = "/home/othman/fetchReachExp/data/buffer_obstacles_env.pkl" if not args.image_obs else "/home/othman/fetchReachExp/data/image_buffer_obstacles_env.pkl"

    # if img dataset doesn't exist, create it
    if not os.path.exists(dataset_path):
        print('Creating image dataset...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = OfflineMerlin(env, dataset_path, device, args.seed, args.max_timesteps, 
                            args.max_path_length, args.test_horizon, args.her_ratio,
                            image_obs=args.image_obs, obstacles=args.obstacles)
    model.train_policy()