#!/usr/bin/env python

import rospy
import torch
import numpy as np
from geometry_msgs.msg import Twist, Pose, PoseStamped
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from std_srvs.srv import Empty
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import math
import random
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive 'Agg'
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from turtlebot_dataset import TurtlebotDataset
from diffuser.models.temporal import TemporalUnet
from diffuser.models.diffusion import GaussianDiffusion
from diffuser.sampling.functions import n_step_guided_p_sample
from diffuser.sampling.policies import GuidedPolicy
import torch.nn as nn

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Global variables for ROS callbacks
odom = None
image = None
x_odom = 0.0
y_odom = 0.0
theta = 0.0
linear_x = 0.0
angular_z = 0.0
bridge = CvBridge()

# def image_callback(data):
#     global image
#     try:
#         image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
#     except Exception as e:
#         rospy.logerr("Image conversion error: %s", str(e))

cylindrical_obstacles = [
            (-1.1, -1.1), (-1.1, 0.0), (-1.1, 1.1),  # Left column
            (0.0, -1.1), (0.0, 0.0), (0.0, 1.1),     # Middle column
            (1.1, -1.1), (1.1, 0.0), (1.1, 1.1)      # Right column
        ]

class SafetyGuide:
    """
    Safety guide that provides gradients to steer trajectories away from obstacles.
    Similar to ValueGuide but focused on safety rather than value maximization.
    """
    def __init__(self, cylindrical_obstacles, hexagonal_obstacles, cylindrical_radius=0.15, 
                 hexagonal_radius=0.4, safe_distance=0.5, scale=0.1, device='cuda'):
        self.cylindrical_obstacles = cylindrical_obstacles
        self.hexagonal_obstacles = hexagonal_obstacles
        self.cylindrical_radius = cylindrical_radius
        self.hexagonal_radius = hexagonal_radius
        self.safe_distance = safe_distance
        self.scale = scale
        self.device = device
        
    def compute_safety_score(self, x, y, sin_theta, cos_theta, linear_v, angular_v, dt=0.1, steps=10):
        """
        Compute safety score for a given state and action.
        Returns a safety score (higher is safer) and gradients w.r.t. action.
        """
        # Convert to tensors with gradients enabled
        linear_v = torch.tensor([linear_v], dtype=torch.float32, requires_grad=True, device=self.device)
        angular_v = torch.tensor([angular_v], dtype=torch.float32, requires_grad=True, device=self.device)
        
        x_t = torch.tensor([x], dtype=torch.float32, device=self.device)
        y_t = torch.tensor([y], dtype=torch.float32, device=self.device)
        theta = torch.atan2(torch.tensor([sin_theta], device=self.device), 
                           torch.tensor([cos_theta], device=self.device))

        min_dist = torch.tensor([float("inf")], dtype=torch.float32, device=self.device)

        # Simulate trajectory for safety evaluation
        for _ in range(steps):
            x_t = x_t + linear_v * torch.cos(theta) * dt
            y_t = y_t + linear_v * torch.sin(theta) * dt
            theta = theta + angular_v * dt

            # Check distance to cylindrical obstacles
            for ox, oy in self.cylindrical_obstacles:
                dx = x_t - ox
                dy = y_t - oy
                dist = torch.sqrt(dx**2 + dy**2) - self.cylindrical_radius
                min_dist = torch.minimum(min_dist, dist)
            
            # Check distance to hexagonal obstacles
            for ox, oy, scale in self.hexagonal_obstacles:
                dx = x_t - ox
                dy = y_t - oy
                dist = torch.sqrt(dx**2 + dy**2) - self.hexagonal_radius * scale
                min_dist = torch.minimum(min_dist, dist)

        # Safety function: tanh(min_dist / safe_distance) gives values in (0, 1)
        # Higher values mean safer (further from obstacles)
        safety_score = torch.tanh(min_dist / self.safe_distance)
        
        # Compute gradients
        safety_score.backward()
        grad_linear = linear_v.grad.item()
        grad_angular = angular_v.grad.item()
        
        return safety_score.item(), np.array([grad_linear, grad_angular])

    def compute_safety_gradient_physical(self, x, y, sin_theta, cos_theta, linear_v, angular_v, obstacles, radius=0.15, dt=0.1, steps=10):
        """
        Compute safety score and its gradient w.r.t. linear and angular velocity.
        """
        # Convert to tensors
        linear_v = torch.tensor([linear_v], dtype=torch.float32, requires_grad=True)
        angular_v = torch.tensor([angular_v], dtype=torch.float32, requires_grad=True)
        
        x_t = torch.tensor([x], dtype=torch.float32)
        y_t = torch.tensor([y], dtype=torch.float32)
        theta = torch.atan2(torch.tensor([sin_theta]), torch.tensor([cos_theta]))

        min_dist = torch.tensor([float("inf")], dtype=torch.float32)

        for _ in range(steps):
            x_t = x_t + linear_v * torch.cos(theta) * dt
            y_t = y_t + linear_v * torch.sin(theta) * dt
            theta = theta + angular_v * dt

            for ox, oy in obstacles:
                dx = x_t - ox
                dy = y_t - oy
                dist = torch.sqrt(dx**2 + dy**2) - radius
                min_dist = torch.minimum(min_dist, dist)

        # Safety function
        safe_distance = 0.5
        safety_score = torch.tanh(min_dist / safe_distance)  # ∈ (0, 1)
        
        # Compute gradients
        safety_score.backward()
        grad_linear = linear_v.grad.item()
        grad_angular = angular_v.grad.item()
        
        return safety_score.item(), np.array([grad_linear, grad_angular])

    def gradients(self, x, cond, t):
        """
        # Return zero safety values and gradients for trajectory optimization.
        # x: trajectory tensor of shape (batch_size, horizon, transition_dim)
        # cond: dictionary containing the current observation
        # t: timestep tensor
        # """
        # batch_size, horizon, transition_dim = x.shape

        # # Create zero tensors with the correct shapes
        # safety_values = torch.zeros(batch_size, device=self.device)
        # safety_gradients = torch.zeros_like(x)

        # # Return zero safety values and scaled zero gradients
        # return safety_values, safety_gradients * self.scale


        """
        Compute safety gradients for trajectory optimization.
        x: trajectory tensor of shape (batch_size, horizon, transition_dim)
        cond: dictionary containing the current observation
        t: timestep tensor
        """
        batch_size, horizon, transition_dim = x.shape
        
        # Extract current state from conditions (first element)
        current_obs = cond[0]  # Shape: [batch_size, obs_dim] (2D tensor)
        
        # Initialize safety values and gradients
        safety_values = torch.zeros(batch_size, device=self.device)
        safety_gradients = torch.zeros_like(x)
        
        # For each trajectory in the batch
        for b in range(batch_size):
            # Extract current state components for this batch element
            x_pos = current_obs[b, 0].item()  # x position
            y_pos = current_obs[b, 1].item()  # y position
            sin_theta = current_obs[b, 4].item()  # sin(theta)
            cos_theta = current_obs[b, 5].item()  # cos(theta)
            
            # Compute safety for the entire trajectory by looking at the first action
            # This is much more efficient than computing for every timestep
            first_action = x[b, 0, :2]  # First action in the trajectory
            
            # Unnormalize action if needed
            linear_v = first_action[0].item()
            angular_v = first_action[1].item()
            
            # Compute safety score and gradients for the first action
            safety_score, action_grads = self.compute_safety_gradient_physical(
                x=x_pos, y=y_pos, sin_theta=sin_theta, cos_theta=cos_theta, linear_v=linear_v, angular_v=angular_v, obstacles=cylindrical_obstacles
            )
            
            # Store safety value
            safety_values[b] = safety_score
            
            # Apply gradients only to the first action (most important for immediate safety)
            safety_gradients[b, 0, :2] = torch.tensor(action_grads, device=self.device)
            
            # Optionally, apply smaller gradients to subsequent actions
            # This creates a safety gradient that propagates through the trajectory
            for h in range(1, horizon):  # Only first 5 timesteps
                # decay_factor = 0.5 ** h  # Exponential decay
                decay_factor = 1 #0.5 - (h/5)*0.5  # Exponential decay
                safety_gradients[b, h, :2] = torch.tensor(action_grads, device=self.device) * decay_factor
        
        return safety_values, safety_gradients * self.scale


class NNSafetyGuide:
    """
    Neural Network Safety Guide that uses a trained safety prediction model.
    Loads a pre-trained neural network and uses its gradients for guidance.
    """
    def __init__(self, model_path, scale=0.1, device='cuda'):
        self.scale = scale
        self.device = device
        
        # Load the trained safety model
        self.model = self.load_safety_model(model_path)
        self.model.eval()
        
    def load_safety_model(self, model_path):
        """Load the trained safety prediction model"""
        # Define the model architecture (same as in training script)
        class SafetyPredictor(nn.Module):
            def __init__(self, input_dim=8, hidden_dims=[128, 64, 32]):
                super(SafetyPredictor, self).__init__()
                
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    ])
                    prev_dim = hidden_dim
                
                # Output layer
                layers.append(nn.Linear(prev_dim, 1))
                layers.append(nn.Sigmoid())  # Safety score between 0 and 1
                
                self.network = nn.Sequential(*layers)
                
            def forward(self, x):
                return self.network(x)
        
        # Create model and load weights
        model = SafetyPredictor(input_dim=8, hidden_dims=[128, 64, 32]).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        print(f"Loaded safety model from {model_path}")
        return model
    
    def compute_safety_score(self, dynamics, action):
        """
        Compute safety score for given dynamics and action.
        dynamics: [x, y, linear_x, angular_z, sin_theta, cos_theta] (6D)
        action: [linear_v, angular_v] (2D)
        Returns: safety score (0=unsafe, 1=safe)
        """
        # Combine dynamics and action
        input_data = torch.cat([dynamics, action], dim=-1)
        
        # Get safety prediction
        with torch.no_grad():
            safety_score = self.model(input_data)
        
        return safety_score
    
    def compute_safety_gradients(self, dynamics, action):
        """
        Compute gradients of safety score with respect to action.
        dynamics: [x, y, linear_x, angular_z, sin_theta, cos_theta] (6D)
        action: [linear_v, angular_v] (2D) - requires gradients
        Returns: gradients w.r.t. action
        """
        # Ensure action has gradients enabled
        if not action.requires_grad:
            action = action.detach().requires_grad_(True)
        
        # Combine dynamics and action
        input_data = torch.cat([dynamics, action], dim=-1)
        
        # Get safety prediction
        safety_score = self.model(input_data)
        
        # Compute gradients
        safety_score.backward()
        gradients = action.grad.clone()
        action.grad.zero_()  # Clear gradients for next use
        
        return gradients
    
    def gradients(self, x, cond, t):
        """
        Compute safety gradients for trajectory optimization.
        x: trajectory tensor of shape (batch_size, horizon, transition_dim)
        cond: dictionary containing the current observation
        t: timestep tensor
        """
        batch_size, horizon, transition_dim = x.shape
        
        # Extract current state from conditions (first element)
        current_obs = cond[0]  # Shape: [batch_size, obs_dim] (2D tensor)
        
        # Initialize safety values and gradients
        safety_values = torch.zeros(batch_size, device=self.device)
        safety_gradients = torch.zeros_like(x)
        
        # For each trajectory in the batch
        for b in range(batch_size):
            # Extract current state components for this batch element
            x_pos = current_obs[b, 0]  # x position
            y_pos = current_obs[b, 1]  # y position
            linear_x = current_obs[b, 2]  # linear velocity
            angular_z = current_obs[b, 3]  # angular velocity
            sin_theta = current_obs[b, 4]  # sin(theta)
            cos_theta = current_obs[b, 5]  # cos(theta)
            
            # Extract all actions for this trajectory
            actions = x[b, :, :2]  # Shape: [horizon, 2]
            
            # Extract predicted states for this trajectory
            # Assuming the trajectory contains: [actions, observations]
            # where observations are: [x, y, linear_x, angular_z, sin_theta, cos_theta, goal_x, goal_y]
            predicted_states = x[b, :, 2:8]  # Shape: [horizon, 6] - excluding goal coordinates
            
            # Create batch of inputs for all timesteps using actual predicted states
            # Combine predicted states and actions for batch processing
            inputs_batch = torch.cat([predicted_states, actions], dim=1)  # Shape: [horizon, 8]
            
            # Enable gradients for the entire batch
            inputs_batch.requires_grad_(True)
            
            # Get safety predictions for all timesteps
            safety_scores = self.model(inputs_batch)  # Shape: [horizon, 1]
            
            # Compute gradients for all timesteps
            # We want to maximize safety, so we minimize negative safety
            total_safety = safety_scores.sum()
            total_safety.backward()
            
            # Extract gradients w.r.t. actions (last 2 dimensions of each input)
            action_gradients = inputs_batch.grad[:, 6:8]  # Shape: [horizon, 2]
            dynamics_gradients = inputs_batch.grad[:, :4] 
            
            # Apply gradients to all timesteps with decay
            for h in range(horizon):
                if h == 0:
                    # No decay for the first step
                    decay_factor = 1.0
                else:
                    # Apply decay for subsequent steps
                    decay_factor = 0.5 ** h  # Exponential decay, can be adjusted
                
                safety_gradients[b, h, :2] = action_gradients[h] * decay_factor
                safety_gradients[b, h, 2:6] = dynamics_gradients[h] * decay_factor
            
            # Store safety value (average across all timesteps)
            with torch.no_grad():
                safety_values[b] = safety_scores.mean().item()
        
        return safety_values, safety_gradients * self.scale

def odom_callback(msg):
    global odom, x_odom, y_odom, theta, linear_x, angular_z
    x_odom = msg.pose.pose.position.x
    y_odom = msg.pose.pose.position.y
    linear_x = msg.twist.twist.linear.x
    angular_z = msg.twist.twist.angular.z
    
    orientation_q = msg.pose.pose.orientation
    _, _, theta = euler_from_quaternion([0, 0, orientation_q.z, orientation_q.w])
    odom = msg


class DiffuserEvaluator:
    def __init__(self, model_path, use_images=True, use_guide=False, guide_scale=0.1, 
                 use_safety_guide=False, safety_scale=0.1, safety_model_path=None):
        
        # Define obstacle positions from the actual TurtleBot3 world
        # Cylindrical obstacles (radius 0.15m)
        self.cylindrical_obstacles = [
            (-1.1, -1.1), (-1.1, 0.0), (-1.1, 1.1),  # Left column
            (0.0, -1.1), (0.0, 0.0), (0.0, 1.1),     # Middle column
            (1.1, -1.1), (1.1, 0.0), (1.1, 1.1)      # Right column
        ]
        self.cylindrical_radius = 0.15
        
        # Hexagonal obstacles (approximate positions and sizes)
        self.hexagonal_obstacles = [
            (3.5, 0.0, 0.8),      # Head
            (1.8, 2.7, 0.55),     # Left hand
            (1.8, -2.7, 0.55),    # Right hand
            (-1.8, 2.7, 0.55),    # Left foot
            (-1.8, -2.7, 0.55)    # Right foot
        ]
        self.hexagonal_radius = 0.4  # Approximate radius for collision checking

        # Initialize ROS node
        rospy.init_node("diffuser_evaluator")
        
        # Setup ROS publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        rospy.Subscriber("/odom", Odometry, odom_callback)
        # rospy.Subscriber("/overhead_camera/overhead_camera/image_raw", Image, image_callback)
        
        # Setup ROS services
        self.reset_service = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.set_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        # Load the trained model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = TurtlebotDataset(expert_pkl_path='/home/othman/turtlebotExp/data/expert.pkl', horizon=32)
        self.diffusion, _ = self.load_model(model_path)
        self.diffusion.eval()
        self.use_images = use_images
        
        # Guide settings
        self.use_guide = use_guide
        self.use_safety_guide = use_safety_guide
        self.guide_scale = guide_scale
        self.safety_scale = safety_scale
        
        if self.use_safety_guide or True:
            # Use NNSafetyGuide instead of SafetyGuide
            if safety_model_path is None:
                safety_model_path = '/home/othman/turtlebotExp/safety_model_best.pth'
            
            # self.safety_guide = NNSafetyGuide(
            #     model_path=safety_model_path,
            #     scale=self.safety_scale,
            #     device=self.device
            # )

            self.safety_guide = SafetyGuide(
                cylindrical_obstacles=self.cylindrical_obstacles,
                hexagonal_obstacles=self.hexagonal_obstacles,
                cylindrical_radius=self.cylindrical_radius,
                hexagonal_radius=self.hexagonal_radius,
                scale=self.safety_scale,
                device=self.device
            )
        
        # Evaluation metrics
        self.success_count = 0
        self.total_episodes = 100
        self.max_steps = 50*2
        self.success_threshold = 0.2  # meters
        
        # Returns and rewards parameters
        self.gamma = 0.99  # discount factor
        self.goal_reward = 10.0  # reward for reaching goal
        self.step_penalty = -0.1  # penalty for each step
        self.collision_penalty = -5.0  # penalty for collision
        
        # Metrics tracking
        self.episode_returns = []  # List to store returns for each episode
        self.episode_success_rates = []  # List to store success rates
        self.episode_lengths = []  # List to store episode lengths
        
        # Create directory for trajectory plots if it doesn't exist
        self.plot_dir = "trajectory_plots"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        self.guided_policy = GuidedPolicy(
            guide=self.safety_guide, 
            diffusion_model=self.diffusion, 
            normalizer=self.dataset.normalizer, 
            preprocess_fns=[],
            # Sampling parameters for guided sampling
            sample_fn=n_step_guided_p_sample,
            n_guide_steps=2,
            t_stopgrad=2,
            scale_grad_by_std=True,
            scale=self.safety_scale
        )

    def plot_generated_trajectories(self, generated_trajectories, start_pos, goal_pos, episode_num, use_safety, obstacles=False):
        obstacles=True
        """Plot the generated trajectories from the diffusion model."""
        if not generated_trajectories:
            return
            
        plt.figure(figsize=(15, 10))
        
        # Plot a subset of trajectories to avoid clutter (every 5th step)
        step_indices = list(range(0, len(generated_trajectories), 5))
        if len(generated_trajectories) - 1 not in step_indices:
            step_indices.append(len(generated_trajectories) - 1)  # Always include the last step
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(step_indices)))
        
        for i, step_idx in enumerate(step_indices):
            trajectories = generated_trajectories[step_idx]
            
            # Extract x,y positions from the trajectory (assuming first 2 dimensions are x,y)
            # trajectories shape: [batch_size, horizon, transition_dim]
            positions = trajectories[0, :, :2]  # Take first trajectory, all timesteps, first 2 dims
            
            # Plot the predicted trajectory
            alpha = 0.3 + 0.7 * (i / len(step_indices))  # Fade in over time
            plt.plot(positions[:, 0], positions[:, 1], 
                    color=colors[i], alpha=alpha, linewidth=2, 
                    label=f'Step {step_idx}' if i % 3 == 0 else "")
        
        # Plot start position
        plt.plot(start_pos[0], start_pos[1], 'go', markersize=12, label='Start Position')
        
        # Plot goal position
        plt.plot(goal_pos[0], goal_pos[1], 'ro', markersize=12, label='Goal Position')
        
        # Plot success/failure circle around goal
        circle = plt.Circle((goal_pos[0], goal_pos[1]), self.success_threshold, 
                          color='g', fill=False, linestyle='--', linewidth=2)
        plt.gca().add_patch(circle)
        
        # Plot cylindrical obstacles
        if obstacles:
            for obs_x, obs_y in self.cylindrical_obstacles:
                obstacle = plt.Circle((obs_x, obs_y), self.cylindrical_radius, 
                                    color='gray', alpha=0.5, label='Cylindrical Obstacle' if obs_x == self.cylindrical_obstacles[0][0] else "")
                plt.gca().add_patch(obstacle)
                
            # Plot hexagonal obstacles
            for obs_x, obs_y, scale in self.hexagonal_obstacles:
                obstacle = plt.Circle((obs_x, obs_y), self.hexagonal_radius * scale, 
                                    color='gray', alpha=0.5, label='Hexagonal Obstacle' if obs_x == self.hexagonal_obstacles[0][0] else "")
                plt.gca().add_patch(obstacle)
        
        # Set plot limits and labels
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, loc='upper right')
        safety_text = "WITH" if use_safety else "WITHOUT"
        plt.title(f'Generated Trajectories - Episode {episode_num} ({safety_text} Safety Guide)\nPredicted paths over time', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('X Position (m)', fontsize=12)
        plt.ylabel('Y Position (m)', fontsize=12)
        
        # Save plot
        safety_suffix = "with_safety" if use_safety else "without_safety"
        savepath = os.path.join(self.plot_dir, f'generated_trajectories_episode_{episode_num}_{safety_suffix}.png')
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated trajectories plot saved to {savepath}")
            
        
    def load_model(self, model_path):
        # Convert observations and actions to numpy arrays
        obs = np.array(self.dataset.episodes[0][0]['observations'])
        act = np.array(self.dataset.episodes[0][0]['actions'])
        
        # Create model
        model = TemporalUnet(
            horizon=32,
            transition_dim=obs.shape[0] + act.shape[0],
            cond_dim=obs.shape[0],
            dim_mults=(1, 2, 4, 8),
            attention=False
        ).to(self.device)
        
        # Create diffusion model
        diffusion = GaussianDiffusion(
            model=model,
            horizon=32,
            observation_dim=obs.shape[0],
            action_dim=act.shape[0],
            n_timesteps=20,
            loss_type='l2',
            clip_denoised=False,
            predict_epsilon=False,
            action_weight=10.0,
            loss_discount=1.0
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        diffusion.load_state_dict(checkpoint['model'])
        
        return diffusion, self.dataset

    def is_position_valid(self, x, y):
        """Check if a position is valid (not inside or too close to obstacles)"""
        # Check cylindrical obstacles
        for obs_x, obs_y in self.cylindrical_obstacles:
            distance = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            if distance < self.cylindrical_radius + 0.045:
                return False
                
        # Check hexagonal obstacles
        for obs_x, obs_y, scale in self.hexagonal_obstacles:
            distance = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            if distance < self.hexagonal_radius * scale:
                return False
                
        return True

    def get_valid_random_position(self):
        """Generate a random position that avoids obstacles"""
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.uniform(-1.4, 1.4)
            y = random.uniform(-1.9, 1.9)
            if self.is_position_valid(x, y):
                return x, y
        # If no valid position found, return a safe default
        return -1.3, -1.8  # Far corner position

    def set_random_start_position(self, obstacles = False):
        # Get valid random position
        if not obstacles:
            initial_x = random.uniform(-1.4, 1.4)
            initial_y = random.uniform(-1.9, 1.9)
        else:
            initial_x, initial_y = self.get_valid_random_position()
        initial_theta = random.uniform(-math.pi, math.pi)

        state_msg = SetModelStateRequest()
        state_msg.model_state.model_name = 'turtlebot3_burger'
        state_msg.model_state.pose.position.x = initial_x
        state_msg.model_state.pose.position.y = initial_y
        quat = quaternion_from_euler(0, 0, initial_theta)
        state_msg.model_state.pose.orientation.x = quat[0]
        state_msg.model_state.pose.orientation.y = quat[1]
        state_msg.model_state.pose.orientation.z = quat[2]
        state_msg.model_state.pose.orientation.w = quat[3]
        state_msg.model_state.reference_frame = 'world'

        try:
            self.set_model_state_service(state_msg)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to set start position: %s", str(e))

    def set_fixed_position(self, x, y, theta):
        """Set the TurtleBot3 to a fixed position and orientation."""
        state_msg = SetModelStateRequest()
        state_msg.model_state.model_name = 'turtlebot3_burger'
        state_msg.model_state.pose.position.x = x
        state_msg.model_state.pose.position.y = y
        quat = quaternion_from_euler(0, 0, theta)
        state_msg.model_state.pose.orientation.x = quat[0]
        state_msg.model_state.pose.orientation.y = quat[1]
        state_msg.model_state.pose.orientation.z = quat[2]
        state_msg.model_state.pose.orientation.w = quat[3]
        state_msg.model_state.reference_frame = 'world'

        try:
            self.set_model_state_service(state_msg)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to set fixed position: %s", str(e))

    def get_observation(self):
        if self.use_images:
            # Process image observation
            if image is None:
                return None
            # Resize image to match training dimensions
            img = cv2.resize(image, (640, 480))
            # Convert to tensor and normalize
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            return img_tensor.unsqueeze(0).to(self.device)
        else:
            # Process vector observation to match training data
            # obs = torch.tensor([x_odom, y_odom, linear_x, angular_z,
            #                   math.sin(theta), math.cos(theta)], 
            #                  dtype=torch.float32).to(self.device)
            obs = torch.tensor([x_odom, y_odom,
                              math.sin(theta), math.cos(theta)], 
                             dtype=torch.float32).to(self.device)
            return obs.unsqueeze(0)

    def plot_trajectory(self, trajectory, start_pos, goal_pos, success, episode_num, obstacles = False):

        obstacles=True
        plt.figure(figsize=(10, 8))
        
        # Plot trajectory
        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Robot Path', linewidth=2)
        
        # Plot start position
        plt.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start Position')
        
        # Plot goal position
        plt.plot(goal_pos[0], goal_pos[1], 'ro', markersize=10, label='Goal Position')
        
        # Plot success/failure circle around goal
        circle = plt.Circle((goal_pos[0], goal_pos[1]), self.success_threshold, 
                          color='g' if success else 'r', fill=False, linestyle='--')
        plt.gca().add_patch(circle)
        
        # Plot cylindrical obstacles
        if obstacles:
            for obs_x, obs_y in self.cylindrical_obstacles:
                obstacle = plt.Circle((obs_x, obs_y), self.cylindrical_radius, 
                                    color='gray', alpha=0.5, label='Cylindrical Obstacle' if obs_x == self.cylindrical_obstacles[0][0] else "")
                plt.gca().add_patch(obstacle)
                
            # Plot hexagonal obstacles
            for obs_x, obs_y, scale in self.hexagonal_obstacles:
                obstacle = plt.Circle((obs_x, obs_y), self.hexagonal_radius * scale, 
                                    color='gray', alpha=0.5, label='Hexagonal Obstacle' if obs_x == self.hexagonal_obstacles[0][0] else "")
                plt.gca().add_patch(obstacle)
        
        # Set plot limits and labels
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.grid(True)
        plt.legend()
        plt.title(f'Trajectory - Episode {episode_num} - {"Success" if success else "Failure"}')
        
        # Save plot
        savepath = os.path.join(self.plot_dir, f'trajectory_episode_{episode_num}.png')
        plt.savefig(savepath)
        plt.close()
        print(f"Trajectory saved to {savepath}")

    def plot_denoising_steps(self, chain, start_pos, goal_pos, episode_num, step_num):
        """
        Plot the denoising steps for a single evaluation step.
        chain: tensor of shape [batch_size, n_denoising_steps, horizon, transition_dim]
        """
        # Create directory for denoising plots if it doesn't exist
        denoising_dir = os.path.join(self.plot_dir, f'denoising_episode_{episode_num}')
        if not os.path.exists(denoising_dir):
            os.makedirs(denoising_dir)

        # Convert chain to numpy and get x,y positions (indices 2,3)
        chain = chain.cpu().numpy()
        positions = chain[:, :, :, 2:4]  # Shape: [8, 21, 32, 2]

        # For each trajectory in the batch
        for b in range(positions.shape[0]):  # Iterate over batch (8 trajectories)
            # Create a subdirectory for this trajectory
            traj_dir = os.path.join(denoising_dir, f"_{episode_num}" + f'trajectory_{b}')
            if not os.path.exists(traj_dir):
                os.makedirs(traj_dir)
            
            # Plot all 21 denoising steps for this trajectory
            for t in range(positions.shape[1]):  # Iterate over denoising steps (21)
                fig = plt.figure(figsize=(10, 8))
                
                # Plot the trajectory at this denoising step
                trajectory = positions[b, t]  # Shape: [32, 2]
                plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.8, label='Trajectory')
                
                # Plot start and goal positions
                plt.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start Position')
                plt.plot(goal_pos[0], goal_pos[1], 'ro', markersize=10, label='Goal Position')
                
                # Set plot limits and labels
                plt.xlim(-2, 2)
                plt.ylim(-2, 2)
                plt.grid(True)
                plt.legend()
                plt.title(f'Trajectory {b}, Denoising Step {t} - Episode {episode_num}, Evaluation Step {step_num}')
                
                # Save plot and close figure
                savepath = os.path.join(traj_dir , f'denoising_step_{t}.png')
                plt.savefig(savepath)
                plt.close(fig)

    def calculate_reward(self, distance, step, is_collision=False):
        """Calculate reward for the current state."""
        reward = self.step_penalty  # Base penalty for each step
        
        if distance < self.success_threshold:
            reward = self.goal_reward
        elif is_collision:
            reward = self.collision_penalty
            
        return reward

    def calculate_returns(self, rewards):
        """Calculate discounted returns from a list of rewards."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def guided_sampling(self, conditions, batch_size=1):
        """
        Custom guided sampling function that combines goal and safety guides.
        """
        device = self.device
        horizon = self.diffusion.horizon
        transition_dim = self.diffusion.transition_dim
        
        # Initialize trajectory with noise
        x = torch.randn((batch_size, horizon, transition_dim), device=device)
        x = self.apply_conditioning(x, conditions)
        
        # Run reverse diffusion process with guides
        for i in reversed(range(self.diffusion.n_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # Apply guides if enabled
            if self.use_guide or self.use_safety_guide:
                with torch.enable_grad():
                    total_gradients = torch.zeros_like(x)
                    
                    # Apply goal guide
                    if self.use_guide:
                        goal_values, goal_gradients = self.goal_guide.gradients(x, conditions, t)
                        total_gradients += goal_gradients
                    
                    # Apply safety guide
                    if self.use_safety_guide:
                        safety_values, safety_gradients = self.safety_guide.gradients(x, conditions, t)
                        total_gradients += safety_gradients
                    
                    # Apply gradient updates
                    x = x + total_gradients
                    x = self.apply_conditioning(x, conditions)
            
            # Standard diffusion step
            model_mean, _, model_log_variance = self.diffusion.p_mean_variance(x=x, cond=conditions, t=t)
            model_std = torch.exp(0.5 * model_log_variance)
            
            # Add noise (no noise when t == 0)
            noise = torch.randn_like(x)
            noise[t == 0] = 0
            
            x = model_mean + model_std * noise
            x = self.apply_conditioning(x, conditions)
        
        return x

    def apply_conditioning(self, x, conditions):
        """Apply conditioning to the trajectory."""
        # This is a simplified version - you might need to adapt it to your specific conditioning
        for t, cond in conditions.items():
            if t < x.shape[1]:
                x[:, t, :cond.shape[-1]] = cond
        return x

    def evaluate_episode(self, episode_num, obstacles=False, start_pos=None, goal_pos=None, safety_scale=None):
        generated_trajectories = []
        # Reset environment
        self.reset_service()
        
        # Set start position (either fixed or random)
        if start_pos is None:
            self.set_random_start_position(obstacles)
        else:
            # Set to fixed start position
            self.set_fixed_position(start_pos[0], start_pos[1], start_pos[2])  # x, y, theta
            
        rospy.sleep(2)  # Wait for the robot to settle
        
        # Get initial observation
        obs = self.get_observation()
        if obs is None:
            print("Failed to get initial observation")
            return None
        
        # Set goal position (either fixed or random)
        if goal_pos is None:
            if not obstacles:
                goal_x = random.uniform(-1.4, 1.4)
                goal_y = random.uniform(-1.9, 1.9)
            else:
                goal_x, goal_y = self.get_valid_random_position()
        else:
            goal_x, goal_y = goal_pos[0], goal_pos[1]

        print(f"Episode {episode_num} - Start: ({x_odom:.3f}, {y_odom:.3f}) - Goal: ({goal_x:.3f}, {goal_y:.3f})")
        obs_with_goal = np.concatenate([obs.cpu().numpy().flatten(), np.array([goal_x, goal_y])])
        
        # Initialize trajectory and metrics
        trajectory = [(x_odom, y_odom)]
        current_start_pos = (x_odom, y_odom)
        current_goal_pos = (goal_x, goal_y)
        rewards = []  # Store rewards for the episode
        
        # Temporarily set safety guide scale if provided
        original_safety_scale = self.safety_scale
        if safety_scale is not None:
            self.safety_scale = safety_scale
            # Recreate the safety guide with new scale
            # self.safety_guide = NNSafetyGuide(
            #     model_path='/home/othman/turtlebotExp/safety_model_best.pth',
            #     scale=self.safety_scale,
            #     device=self.device
            # )
            self.safety_guide = SafetyGuide(
                cylindrical_obstacles=self.cylindrical_obstacles,
                hexagonal_obstacles=self.hexagonal_obstacles,
                cylindrical_radius=self.cylindrical_radius,
                hexagonal_radius=self.hexagonal_radius,
                scale=self.safety_scale,
                device=self.device
            )
            # Recreate the guided policy with new safety guide
            self.guided_policy = GuidedPolicy(
                guide=self.safety_guide, 
                diffusion_model=self.diffusion, 
                normalizer=self.dataset.normalizer, 
                preprocess_fns=[],
                # Sampling parameters for guided sampling
                sample_fn=n_step_guided_p_sample,
                n_guide_steps=2,
                t_stopgrad=2,
                scale_grad_by_std=True,
                scale=self.safety_scale
            )
        
        # Run episode
        for step in range(self.max_steps):
            obs = self.get_observation()
            obs_with_goal = np.concatenate([obs.cpu().numpy().flatten(), np.array([goal_x, goal_y])])
            # Convert to tensor (don't normalize here - GuidedPolicy will handle it)
            obs_tensor = torch.FloatTensor(obs_with_goal).to(self.device)
            
            # Create conditions for the model - specify both initial and final states
            conditions = {
                0: obs_tensor,  # Initial state - let GuidedPolicy handle normalization
                self.diffusion.horizon-1: torch.tensor([goal_x, goal_y, 0, 0, 0, 0, goal_x, goal_y], dtype=torch.float32).to(self.device)  # Final state
            }
            
            # Generate trajectory using guided sampling if guides are enabled
            if self.use_guide or self.use_safety_guide:
                with torch.no_grad():
                    action, trajectories = self.guided_policy(conditions)
                    generated_trajectories.append(trajectories.observations.copy())
            else:
                # Use standard diffusion sampling
                with torch.no_grad():
                    action, trajectories = self.guided_policy(conditions)
                    generated_trajectories.append(trajectories.observations.copy())
            
            safety_score, grad = self.safety_guide
            if safety_score < 0.2:
                optimized_action = np.array([gradients[0], gradients[1]])

            # Execute action
            vel_msg = Twist()
            vel_msg.linear.x = np.clip(action[0], 0.0, 0.5) 
            vel_msg.angular.z = np.clip(action[1], -1.0, 1.0) 
            self.cmd_vel_pub.publish(vel_msg)
            
            # Wait for action to take effect
            rospy.sleep(0.1)
            
            # Update trajectory
            trajectory.append((x_odom, y_odom))
            
            # Calculate distance to goal
            distance = math.sqrt((x_odom - goal_x)**2 + (y_odom - goal_y)**2)
            
            # Check for collisions
            is_collision = obstacles and not self.is_position_valid(x_odom, y_odom)
            
            # Calculate reward
            reward = self.calculate_reward(distance, step, is_collision)
            rewards.append(reward)
            
            # Check if episode is done
            if distance < self.success_threshold or is_collision:
                if distance < self.success_threshold:
                    print(f"Goal reached in {step + 1} steps")
                    success = True
                else:
                    print(f"Failed: Invalid position at {step + 1} steps")
                    success = False
                
                # Calculate episode metrics
                episode_return = sum(rewards)
                discounted_returns = self.calculate_returns(rewards)
                episode_discounted_return = discounted_returns[0] if discounted_returns else 0
                
                # Store episode metrics
                self.episode_returns.append(episode_return)
                self.episode_success_rates.append(1.0 if success else 0.0)
                self.episode_lengths.append(step + 1)
                
                # Plot trajectory
                self.plot_trajectory(trajectory, current_start_pos, current_goal_pos, success, episode_num, obstacles)
                
                # Print episode summary
                print(f"\nEpisode {episode_num} Summary:")
                print(f"Success: {success}")
                print(f"Episode Length: {step + 1}")
                print(f"Total Return: {episode_return:.2f}")
                print(f"Discounted Return: {episode_discounted_return:.2f}")
                
                # Restore original safety scale
                if safety_scale is not None:
                    self.safety_scale = original_safety_scale
                    # self.safety_guide = NNSafetyGuide(
                    #     model_path='/home/othman/turtlebotExp/safety_model_best.pth',
                    #     scale=self.safety_scale,
                    #     device=self.device
                    # )
                    self.safety_guide = SafetyGuide(
                        cylindrical_obstacles=self.cylindrical_obstacles,
                        hexagonal_obstacles=self.hexagonal_obstacles,
                        cylindrical_radius=self.cylindrical_radius,
                        hexagonal_radius=self.hexagonal_radius,
                        scale=self.safety_scale,
                        device=self.device
                    )
                    self.guided_policy = GuidedPolicy(
                        guide=self.safety_guide, 
                        diffusion_model=self.diffusion, 
                        normalizer=self.dataset.normalizer, 
                        preprocess_fns=[],
                        sample_fn=n_step_guided_p_sample,
                        n_guide_steps=2,
                        t_stopgrad=2,
                        scale_grad_by_std=True,
                        scale=self.safety_scale
                    )
                
                return trajectory, success, episode_return
                
        # Episode timed out
        print(f"Episode timed out after {self.max_steps} steps")
        success = False
        
        # Calculate episode metrics
        episode_return = sum(rewards)
        discounted_returns = self.calculate_returns(rewards)
        episode_discounted_return = discounted_returns[0] if discounted_returns else 0
        
        # Store episode metrics
        self.episode_returns.append(episode_return)
        self.episode_success_rates.append(0.0)
        self.episode_lengths.append(self.max_steps)
        
        # Plot trajectory
        self.plot_trajectory(trajectory, current_start_pos, current_goal_pos, False, episode_num, obstacles)
        
        # Print episode summary
        print(f"\nEpisode {episode_num} Summary:")
        print(f"Success: {success}")
        print(f"Episode Length: {self.max_steps}")
        print(f"Total Return: {episode_return:.2f}")
        print(f"Discounted Return: {episode_discounted_return:.2f}")
        
        # Restore original safety scale
        if safety_scale is not None:
            self.safety_scale = original_safety_scale
            # self.safety_guide = NNSafetyGuide(
            #     model_path='/home/othman/turtlebotExp/safety_model_best.pth',
            #     scale=self.safety_scale,
            #     device=self.device
            # )
            self.safety_guide = SafetyGuide(
                cylindrical_obstacles=self.cylindrical_obstacles,
                hexagonal_obstacles=self.hexagonal_obstacles,
                cylindrical_radius=self.cylindrical_radius,
                hexagonal_radius=self.hexagonal_radius,
                scale=self.safety_scale,
                device=self.device
            )
            self.guided_policy = GuidedPolicy(
                guide=self.safety_guide, 
                diffusion_model=self.diffusion, 
                normalizer=self.dataset.normalizer, 
                preprocess_fns=[],
                sample_fn=n_step_guided_p_sample,
                n_guide_steps=2,
                t_stopgrad=2,
                scale_grad_by_std=True,
                scale=self.safety_scale
            )
        
        return trajectory, success, episode_return

    def run_multi_scale_evaluation(self, safety_scales=[0, 0.1, 1, 10], obstacles=False):
        """
        Run evaluation with the same start and goal positions but different safety guidance scales.
        Creates a single plot showing all paths.
        """
        print(f"Starting multi-scale evaluation with safety scales: {safety_scales}")

        # Generate fixed start and goal positions
        if obstacles:
            start_x, start_y = self.get_valid_random_position()
            goal_x, goal_y = self.get_valid_random_position()
        else:
            start_x = random.uniform(-1.4, 1.4)
            start_y = random.uniform(-1.9, 1.9)
            goal_x = random.uniform(-1.4, 1.4)
            goal_y = random.uniform(-1.9, 1.9)
        
        start_theta = random.uniform(-math.pi, math.pi)
        start_pos = (start_x, start_y, start_theta)
        goal_pos = (goal_x, goal_y)
        
        print(f"Fixed Start Position: ({start_x:.3f}, {start_y:.3f}, {start_theta:.3f})")
        print(f"Fixed Goal Position: ({goal_x:.3f}, {goal_y:.3f})")
        
        # Store results for each scale
        results = {}
        
        # Run evaluation for each safety scale
        for i, scale in enumerate(safety_scales):
            print(f"\n{'='*50}")
            print(f"Running evaluation with safety scale: {scale}")
            print(f"{'='*50}")
            
            # Run episode with fixed positions and specific safety scale
            trajectory, success, episode_return = self.evaluate_episode(
                episode_num=i+1, 
                obstacles=obstacles, 
                start_pos=start_pos, 
                goal_pos=goal_pos, 
                safety_scale=scale
            )
            
            results[scale] = {
                'trajectory': trajectory,
                'success': success,
                'episode_return': episode_return
            }
            
            # Update success count for overall statistics
            if success:
                self.success_count += 1
        
        # Create combined plot
        self.plot_multi_scale_trajectories(results, start_pos, goal_pos, safety_scales, obstacles)
        
        # Print summary
        print(f"\n{'='*50}")
        print("MULTI-SCALE EVALUATION SUMMARY")
        print(f"{'='*50}")
        for scale in safety_scales:
            result = results[scale]
            status = "SUCCESS" if result['success'] else "FAILED"
            print(f"Safety Scale {scale:>5}: {status} | Return: {result['episode_return']:.2f}")
        
        return results

    def plot_multi_scale_trajectories(self, results, start_pos, goal_pos, safety_scales, obstacles=False):
        obstacles = True
        """
        Create a single plot showing trajectories for all safety scales.
        """
        plt.figure(figsize=(12, 10))
        
        # Define colors for different scales
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Plot trajectories for each scale
        for i, scale in enumerate(safety_scales):
            if scale in results:
                trajectory = results[scale]['trajectory']
                success = results[scale]['success']
                
                # Convert trajectory to numpy array
                trajectory = np.array(trajectory)
                
                # Plot trajectory
                color = colors[i % len(colors)]
                linestyle = '-' if success else '--'
                linewidth = 2 if success else 1.5
                alpha = 0.8 if success else 0.6
                
                plt.plot(trajectory[:, 0], trajectory[:, 1], 
                        color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha,
                        label=f'Safety Scale {scale} ({("SUCCESS" if success else "FAILED")})')
        
        # Plot start position
        plt.plot(start_pos[0], start_pos[1], 'ko', markersize=15, label='Start Position', markeredgewidth=2, markeredgecolor='black')
        
        # Plot goal position
        plt.plot(goal_pos[0], goal_pos[1], 'k*', markersize=15, label='Goal Position', markeredgewidth=2, markeredgecolor='black')
        
        # Plot success/failure circle around goal
        circle = plt.Circle((goal_pos[0], goal_pos[1]), self.success_threshold, 
                          color='green', fill=False, linestyle='--', linewidth=2)
        plt.gca().add_patch(circle)
        
        # Plot obstacles if enabled
        if obstacles:
            for obs_x, obs_y in self.cylindrical_obstacles:
                obstacle = plt.Circle((obs_x, obs_y), self.cylindrical_radius, 
                                    color='gray', alpha=0.5, label='Cylindrical Obstacle' if obs_x == self.cylindrical_obstacles[0][0] else "")
                plt.gca().add_patch(obstacle)
                
            for obs_x, obs_y, scale in self.hexagonal_obstacles:
                obstacle = plt.Circle((obs_x, obs_y), self.hexagonal_radius * scale, 
                                    color='gray', alpha=0.5, label='Hexagonal Obstacle' if obs_x == self.hexagonal_obstacles[0][0] else "")
                plt.gca().add_patch(obstacle)
        
        # Set plot limits and labels
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.title(f'Multi-Scale Safety Guidance Comparison\nStart: ({start_pos[0]:.2f}, {start_pos[1]:.2f}) | Goal: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('X Position (m)', fontsize=12)
        plt.ylabel('Y Position (m)', fontsize=12)
        
        # Save plot
        savepath = os.path.join(self.plot_dir, f'multi_scale_comparison_start_{start_pos[0]:.2f}_{start_pos[1]:.2f}_goal_{goal_pos[0]:.2f}_{goal_pos[1]:.2f}.png')
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Multi-scale comparison plot saved to {savepath}")

    def run_full_evaluation(self, safety_scales=[0, 0.1, 10], num_episodes=100, obstacles=True):
        """
        Run full evaluation over multiple episodes with different safety guidance scales.
        Creates comparison plots for each episode and logs comprehensive statistics.
        """
        print(f"Starting full evaluation over {num_episodes} episodes with safety scales: {safety_scales}")
        
        # Initialize statistics tracking
        episode_stats = {
            scale: {
                'rewards': [],
                'successes': [],
                'collisions': [],
                'episode_lengths': [],
                'trajectories': []
            } for scale in safety_scales
        }
        
        # Run evaluation for each episode
        for episode in range(num_episodes):
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"{'='*60}")
            
            # Update progress tracking
            self.update_evaluation_progress(episode + 1, num_episodes)
            
            # Generate random start and goal positions for this episode
            if obstacles:
                start_x, start_y = self.get_valid_random_position()
                goal_x, goal_y = self.get_valid_random_position()
            else:
                start_x = random.uniform(-1.4, 1.4)
                start_y = random.uniform(-1.9, 1.9)
                goal_x = random.uniform(-1.4, 1.4)
                goal_y = random.uniform(-1.9, 1.9)
            
            start_theta = random.uniform(-math.pi, math.pi)
            start_pos = (start_x, start_y, start_theta)
            goal_pos = (goal_x, goal_y)
            
            print(f"Start Position: ({start_x:.3f}, {start_y:.3f}, {start_theta:.3f})")
            print(f"Goal Position: ({goal_x:.3f}, {goal_y:.3f})")
            
            # Store results for this episode
            episode_results = {}
            
            # Run evaluation for each safety scale
            for scale in safety_scales:
                print(f"\n--- Safety Scale: {scale} ---")
                
                # Run episode with specific safety scale
                trajectory, success, episode_return, collision, safety_values, actions = self.evaluate_episode_with_stats(
                    episode_num=episode + 1,
                    obstacles=obstacles,
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    safety_scale=scale
                )
                
                # Store results
                episode_results[scale] = {
                    'trajectory': trajectory,
                    'success': success,
                    'episode_return': episode_return,
                    'collision': collision,
                    'safety_values': safety_values,
                    'actions': actions
                }
                
                # Update statistics
                episode_stats[scale]['rewards'].append(episode_return)
                episode_stats[scale]['successes'].append(1 if success else 0)
                episode_stats[scale]['collisions'].append(1 if collision else 0)
                episode_stats[scale]['trajectories'].append(trajectory)
            
            # Create comparison plot for this episode
            self.plot_episode_comparison(episode_results, start_pos, goal_pos, safety_scales, episode + 1, obstacles)
            
            # Create individual trajectory plots for each safety scale
            # self.plot_individual_trajectories(episode_results, start_pos, goal_pos, safety_scales, episode + 1, obstacles)
            
            # Print episode summary
            print(f"\nEpisode {episode + 1} Summary:")
            for scale in safety_scales:
                result = episode_results[scale]
                status = "SUCCESS" if result['success'] else "FAILED"
                collision_status = " (COLLISION)" if result['collision'] else ""
                print(f"  Safety Scale {scale:>5}: {status} | {collision_status} | Return: {result['episode_return']:.2f}")
            
            # Print running averages after each episode
            self.print_running_averages(episode_stats, safety_scales, episode + 1)
        
        # Print final statistics
        self.print_final_statistics(episode_stats, safety_scales, num_episodes)
        
        return episode_stats

    def print_running_averages(self, episode_stats, safety_scales, current_episode):
        """
        Print running averages for all metrics after each episode.
        """
        print(f"\n{'='*60}")
        print(f"RUNNING AVERAGES AFTER EPISODE {current_episode}")
        print(f"{'='*60}")
        
        # Print header
        print(f"{'Scale':>8} | {'Avg Reward':>12} | {'Success Rate':>13} | {'Collision Rate':>15} | {'Avg Length':>12}")
        print("-" * 75)
        
        # Prepare data for logging
        log_data = {
            'episode': current_episode,
            'timestamp': datetime.now().isoformat(),
            'averages': {}
        }
        
        for scale in safety_scales:
            stats = episode_stats[scale]
            
            # Calculate running averages
            avg_reward = sum(stats['rewards']) / current_episode
            success_rate = sum(stats['successes']) / current_episode * 100
            collision_rate = sum(stats['collisions']) / current_episode * 100
            avg_episode_length = sum([len(traj) for traj in stats['trajectories']]) / current_episode
            
            print(f"{scale:>8.1f} | {avg_reward:>12.2f} | {success_rate:>13.1f}% | {collision_rate:>15.1f}% | {avg_episode_length:>12.1f}")
            
            # Store for logging
            log_data['averages'][str(scale)] = {
                'avg_reward': avg_reward,
                'success_rate': success_rate,
                'collision_rate': collision_rate,
                'avg_episode_length': avg_episode_length
            }
        
        print(f"{'='*60}")
        
        # Save to log file
        self.save_running_averages_log(log_data)
        
        # Identify and log best performing scale
        self.log_best_performing_scale(log_data, safety_scales)

    def log_best_performing_scale(self, log_data, safety_scales):
        """
        Identify and log the best performing safety scale based on success rate and average reward.
        """
        best_success_rate = -1
        best_success_scale = None
        best_reward = float('-inf')
        best_reward_scale = None
        
        for scale in safety_scales:
            scale_str = str(scale)
            if scale_str in log_data['averages']:
                avg_data = log_data['averages'][scale_str]
                
                # Track best success rate
                if avg_data['success_rate'] > best_success_rate:
                    best_success_rate = avg_data['success_rate']
                    best_success_scale = scale
                
                # Track best average reward
                if avg_data['avg_reward'] > best_reward:
                    best_reward = avg_data['avg_reward']
                    best_reward_scale = scale
        
        # Print best performers
        print(f"\nBest Performance Summary:")
        print(f"  Best Success Rate: Safety Scale {best_success_scale} ({best_success_rate:.1f}%)")
        print(f"  Best Average Reward: Safety Scale {best_reward_scale} ({best_reward:.2f})")
        
        # Add to log data
        log_data['best_performers'] = {
            'best_success_rate': {
                'scale': best_success_scale,
                'value': best_success_rate
            },
            'best_average_reward': {
                'scale': best_reward_scale,
                'value': best_reward
            }
        }
        
        # Save best performers to separate log
        self.save_best_performers_log(log_data)
    
    def save_best_performers_log(self, log_data):
        """
        Save best performing scales to a separate log file.
        """
        logs_dir = "evaluation_logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"best_performers_log_{timestamp}.jsonl"
        filepath = os.path.join(logs_dir, filename)
        
        # Append to log file
        with open(filepath, 'a') as f:
            f.write(json.dumps(log_data) + '\n')
    
    def save_running_averages_log(self, log_data):
        """
        Save running averages to a log file for later analysis.
        """
        # Create logs directory if it doesn't exist
        logs_dir = "evaluation_logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        # Get current timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"running_averages_log_{timestamp}.jsonl"
        filepath = os.path.join(logs_dir, filename)
        
        # Append to log file (JSON Lines format)
        with open(filepath, 'a') as f:
            f.write(json.dumps(log_data) + '\n')
        
        # Also save a summary file with just the latest episode
        summary_filename = f"latest_running_averages_{timestamp}.json"
        summary_filepath = os.path.join(logs_dir, summary_filename)
        
        with open(summary_filepath, 'w') as f:
            json.dump(log_data, f, indent=2)

    def update_evaluation_progress(self, current_episode, total_episodes):
        """
        Update and save evaluation progress information.
        """
        # Create logs directory if it doesn't exist
        logs_dir = "evaluation_logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        # Calculate progress percentage
        progress_percentage = (current_episode / total_episodes) * 100
        
        # Prepare progress data
        progress_data = {
            'current_episode': current_episode,
            'total_episodes': total_episodes,
            'progress_percentage': progress_percentage,
            'timestamp': datetime.now().isoformat(),
            'estimated_time_remaining': None  # Could be calculated if we track timing
        }
        
        # Save progress to file
        progress_filename = "evaluation_progress.json"
        progress_filepath = os.path.join(logs_dir, progress_filename)
        
        with open(progress_filepath, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        # Print progress
        print(f"Progress: {current_episode}/{total_episodes} episodes ({progress_percentage:.1f}%)")

    def evaluate_episode_with_stats(self, episode_num, obstacles=False, start_pos=None, goal_pos=None, safety_scale=None):
        """
        Evaluate a single episode and return detailed statistics including collision detection.
        """
        # Reset environment
        self.reset_service()
        
        # Set start position (either fixed or random)
        if start_pos is None:
            self.set_random_start_position(obstacles)
        else:
            # Set to fixed start position
            self.set_fixed_position(start_pos[0], start_pos[1], start_pos[2])  # x, y, theta
            
        rospy.sleep(2)  # Wait for the robot to settle
        
        # Get initial observation
        obs = self.get_observation()
        if obs is None:
            print("Failed to get initial observation")
            return None, False, 0.0, False, [], []
        
        # Set goal position (either fixed or random)
        if goal_pos is None:
            if not obstacles:
                goal_x = random.uniform(-1.4, 1.4)
                goal_y = random.uniform(-1.9, 1.9)
            else:
                goal_x, goal_y = self.get_valid_random_position()
        else:
            goal_x, goal_y = goal_pos[0], goal_pos[1]

        print(f"Episode {episode_num} - Start: ({x_odom:.3f}, {y_odom:.3f}) - Goal: ({goal_x:.3f}, {goal_y:.3f})")
        obs_with_goal = np.concatenate([obs.cpu().numpy().flatten(), np.array([goal_x, goal_y])])
        
        # Initialize trajectory and metrics
        trajectory = [(x_odom, y_odom)]
        current_start_pos = (x_odom, y_odom)
        current_goal_pos = (goal_x, goal_y)
        rewards = []  # Store rewards for the episode
        collision_detected = False
        safety_values = []  # Store safety values for each step
        actions = []  # Store actions for each step
        
        # Temporarily set safety guide scale if provided
        original_safety_scale = self.safety_scale
        if safety_scale is not None:
            self.safety_scale = safety_scale
            # Recreate the safety guide with new scale
            # self.safety_guide = NNSafetyGuide(
            #     model_path='/home/othman/turtlebotExp/safety_model_best.pth',
            #     scale=self.safety_scale,
            #     device=self.device
            # )
            self.safety_guide = SafetyGuide(
                cylindrical_obstacles=self.cylindrical_obstacles,
                hexagonal_obstacles=self.hexagonal_obstacles,
                cylindrical_radius=self.cylindrical_radius,
                hexagonal_radius=self.hexagonal_radius,
                scale=self.safety_scale,
                device=self.device
            )
            # Recreate the guided policy with new safety guide
            self.guided_policy = GuidedPolicy(
                guide=self.safety_guide, 
                diffusion_model=self.diffusion, 
                normalizer=self.dataset.normalizer, 
                preprocess_fns=[],
                # Sampling parameters for guided sampling
                sample_fn=n_step_guided_p_sample,
                n_guide_steps=2,
                t_stopgrad=2,
                scale_grad_by_std=True,
                scale=self.safety_scale
            )
        
        # Run episode
        for step in range(self.max_steps):
            obs = self.get_observation()
            obs_with_goal = np.concatenate([obs.cpu().numpy().flatten(), np.array([goal_x, goal_y])])
            # Convert to tensor (don't normalize here - GuidedPolicy will handle it)
            obs_tensor = torch.FloatTensor(obs_with_goal).to(self.device)
            
            # Create conditions for the model - specify both initial and final states
            conditions = {
                0: obs_tensor,  # Initial state - let GuidedPolicy handle normalization
                self.diffusion.horizon-1: torch.tensor([goal_x, goal_y, 0, 0, goal_x, goal_y], dtype=torch.float32).to(self.device)  # Final state
            }
            
            # Generate trajectory using guided sampling if guides are enabled
            if self.use_guide or self.use_safety_guide:
                with torch.no_grad():
                    action, trajectories = self.guided_policy(conditions)
            else:
                # Use standard diffusion sampling
                with torch.no_grad():
                    action, trajectories = self.guided_policy(conditions)
            
            # Store action
            actions.append([action[0], action[1]])
            
            # Calculate safety value for current state and action
            robot_state = torch.tensor([x_odom, y_odom, linear_x, angular_z, math.sin(theta), math.cos(theta)], 
                                     dtype=torch.float32, device=self.device)
            action_tensor = torch.tensor([action[0], action[1]], dtype=torch.float32, device=self.device, requires_grad=True)
            
            # with torch.no_grad():
            #     safety_input = torch.cat([robot_state, action_tensor], dim=0).unsqueeze(0)
            #     safety_value = self.safety_guide.model(safety_input).item()
            #     safety_values.append(safety_value)
            
            # Compute safety gradients
            # safety_grad = self.safety_guide.compute_safety_gradients(robot_state, action_tensor)
            
            safety_value, safety_grad = self.safety_guide.compute_safety_gradient_physical(x=x_odom, y=y_odom, sin_theta=math.sin(theta), cos_theta=math.cos(theta), linear_v=action[0], angular_v=action[1], obstacles=cylindrical_obstacles)
            
            # if safety_ldified action: {action}")

            # Execute action
            if safety_value < 0.2:
                action = np.array([safety_grad[0], safety_grad[1]])

            vel_msg = Twist()
            vel_msg.linear.x = np.clip(action[0], 0.0, 0.5) 
            vel_msg.angular.z = np.clip(action[1], -1.0, 1.0) 
            self.cmd_vel_pub.publish(vel_msg)
            
            # Wait for action to take effect
            rospy.sleep(0.1)
            
            # Update trajectory
            trajectory.append((x_odom, y_odom))
            
            # Calculate distance to goal
            distance = math.sqrt((x_odom - goal_x)**2 + (y_odom - goal_y)**2)
            
            # Check for collisions
            is_collision = obstacles and not self.is_position_valid(x_odom, y_odom)
            if is_collision:
                collision_detected = True
            
            # Calculate reward
            reward = self.calculate_reward(distance, step, is_collision)
            rewards.append(reward)
            
            # Check if episode is done
            if distance < self.success_threshold or is_collision:
                if distance < self.success_threshold:
                    print(f"Goal reached in {step + 1} steps")
                    success = True
                else:
                    print(f"Failed: Invalid position at {step + 1} steps")
                    success = False
                
                # Calculate episode metrics
                episode_return = sum(rewards)
                
                # Restore original safety scale
                if safety_scale is not None:
                    self.safety_scale = original_safety_scale
                    # self.safety_guide = NNSafetyGuide(
                    #     model_path='/home/othman/turtlebotExp/safety_model_best.pth',
                    #     scale=self.safety_scale,
                    #     device=self.device
                    # )
                    self.safety_guide = SafetyGuide(
                        cylindrical_obstacles=self.cylindrical_obstacles,
                        hexagonal_obstacles=self.hexagonal_obstacles,
                        cylindrical_radius=self.cylindrical_radius,
                        hexagonal_radius=self.hexagonal_radius,
                        scale=self.safety_scale,
                        device=self.device
                    )
                    self.guided_policy = GuidedPolicy(
                        guide=self.safety_guide, 
                        diffusion_model=self.diffusion, 
                        normalizer=self.dataset.normalizer, 
                        preprocess_fns=[],
                        sample_fn=n_step_guided_p_sample,
                        n_guide_steps=2,
                        t_stopgrad=2,
                        scale_grad_by_std=True,
                        scale=self.safety_scale
                    )
                
                return trajectory, success, episode_return, collision_detected, safety_values, actions
                
        # Episode timed out
        print(f"Episode timed out after {self.max_steps} steps")
        success = False
        
        # Calculate episode metrics
        episode_return = sum(rewards)
        
        # Restore original safety scale
        if safety_scale is not None:
            self.safety_scale = original_safety_scale
            # self.safety_guide = NNSafetyGuide(
            #     model_path='/home/othman/turtlebotExp/safety_model_best.pth',
            #     scale=self.safety_scale,
            #     device=self.device
            # )
            self.safety_guide = SafetyGuide(
                cylindrical_obstacles=self.cylindrical_obstacles,
                hexagonal_obstacles=self.hexagonal_obstacles,
                cylindrical_radius=self.cylindrical_radius,
                hexagonal_radius=self.hexagonal_radius,
                scale=self.safety_scale,
                device=self.device
            )
            self.guided_policy = GuidedPolicy(
                guide=self.safety_guide, 
                diffusion_model=self.diffusion, 
                normalizer=self.dataset.normalizer, 
                preprocess_fns=[],
                sample_fn=n_step_guided_p_sample,
                n_guide_steps=2,
                t_stopgrad=2,
                scale_grad_by_std=True,
                scale=self.safety_scale
            )
        
        return trajectory, success, episode_return, collision_detected, safety_values, actions

    def plot_episode_comparison(self, episode_results, start_pos, goal_pos, safety_scales, episode_num, obstacles=False):
        """
        Create a comparison plot for a single episode showing trajectories for all safety scales.
        """
        plt.figure(figsize=(12, 10))
        
        # Define colors for different scales
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Plot trajectories for each scale
        for i, scale in enumerate(safety_scales):
            if scale in episode_results:
                trajectory = episode_results[scale]['trajectory']
                success = episode_results[scale]['success']
                collision = episode_results[scale]['collision']
                
                # Convert trajectory to numpy array
                trajectory = np.array(trajectory)
                
                # Plot trajectory
                color = colors[i % len(colors)]
                linestyle = '-' if success else '--'
                linewidth = 2 if success else 1.5
                alpha = 0.8 if success else 0.6
                
                # Create label with status
                status = "SUCCESS" if success else "FAILED"
                collision_status = " (COLLISION)" if collision else ""
                label = f'Safety Scale {scale} ({status}{collision_status})'
                
                plt.plot(trajectory[:, 0], trajectory[:, 1], 
                        color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha,
                        label=label)
        
        # Plot start position
        plt.plot(start_pos[0], start_pos[1], 'ko', markersize=15, label='Start Position', markeredgewidth=2, markeredgecolor='black')
        
        # Plot goal position
        plt.plot(goal_pos[0], goal_pos[1], 'k*', markersize=15, label='Goal Position', markeredgewidth=2, markeredgecolor='black')
        
        # Plot success/failure circle around goal
        circle = plt.Circle((goal_pos[0], goal_pos[1]), self.success_threshold, 
                          color='green', fill=False, linestyle='--', linewidth=2)
        plt.gca().add_patch(circle)
        
        # Plot obstacles if enabled
        if obstacles:
            for obs_x, obs_y in self.cylindrical_obstacles:
                obstacle = plt.Circle((obs_x, obs_y), self.cylindrical_radius, 
                                    color='gray', alpha=0.5, label='Cylindrical Obstacle' if obs_x == self.cylindrical_obstacles[0][0] else "")
                plt.gca().add_patch(obstacle)
                
            for obs_x, obs_y, scale in self.hexagonal_obstacles:
                obstacle = plt.Circle((obs_x, obs_y), self.hexagonal_radius * scale, 
                                    color='gray', alpha=0.5, label='Hexagonal Obstacle' if obs_x == self.hexagonal_obstacles[0][0] else "")
                plt.gca().add_patch(obstacle)
        
        # Set plot limits and labels
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.title(f'Episode {episode_num} - Safety Scale Comparison\nStart: ({start_pos[0]:.2f}, {start_pos[1]:.2f}) | Goal: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('X Position (m)', fontsize=12)
        plt.ylabel('Y Position (m)', fontsize=12)
        
        # Save plot
        savepath = os.path.join(self.plot_dir, f'episode_{episode_num}_comparison.png')
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Episode {episode_num} comparison plot saved to {savepath}")

    def print_final_statistics(self, episode_stats, safety_scales, num_episodes):
        """
        Print comprehensive final statistics for all episodes and safety scales.
        """
        print(f"\n{'='*80}")
        print("FINAL EVALUATION STATISTICS")
        print(f"{'='*80}")
        
        for scale in safety_scales:
            stats = episode_stats[scale]
            
            # Calculate statistics
            total_reward = sum(stats['rewards'])
            avg_reward = total_reward / num_episodes
            success_rate = sum(stats['successes']) / num_episodes * 100
            collision_rate = sum(stats['collisions']) / num_episodes * 100
            avg_episode_length = sum([len(traj) for traj in stats['trajectories']]) / num_episodes
            
            print(f"\nSafety Scale {scale:>5}:")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  Success Rate: {success_rate:.1f}% ({sum(stats['successes'])}/{num_episodes})")
            print(f"  Collision Rate: {collision_rate:.1f}% ({sum(stats['collisions'])}/{num_episodes})")
            print(f"  Average Episode Length: {avg_episode_length:.1f} steps")
        
        # Print comparison table
        print(f"\n{'='*80}")
        print("COMPARISON TABLE")
        print(f"{'='*80}")
        print(f"{'Scale':>8} | {'Avg Reward':>12} | {'Success Rate':>13} | {'Collision Rate':>15}")
        print("-" * 60)
        
        for scale in safety_scales:
            stats = episode_stats[scale]
            avg_reward = sum(stats['rewards']) / num_episodes
            success_rate = sum(stats['successes']) / num_episodes * 100
            collision_rate = sum(stats['collisions']) / num_episodes * 100
            
            print(f"{scale:>8.1f} | {avg_reward:>12.2f} | {success_rate:>13.1f}% | {collision_rate:>15.1f}%")
        
        # Save results to file
        self.save_evaluation_results(episode_stats, safety_scales, num_episodes)

    def save_evaluation_results(self, episode_stats, safety_scales, num_episodes):
        """
        Save evaluation results to a JSON file for later analysis.
        """
        
        # Create results directory if it doesn't exist
        results_dir = "evaluation_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Prepare data for saving
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'num_episodes': num_episodes,
            'safety_scales': safety_scales,
            'statistics': {}
        }
        
        for scale in safety_scales:
            stats = episode_stats[scale]
            
            # Calculate statistics
            total_reward = sum(stats['rewards'])
            avg_reward = total_reward / num_episodes
            success_rate = sum(stats['successes']) / num_episodes * 100
            collision_rate = sum(stats['collisions']) / num_episodes * 100
            avg_episode_length = sum([len(traj) for traj in stats['trajectories']]) / num_episodes
            
            results_data['statistics'][str(scale)] = {
                'total_reward': total_reward,
                'average_reward': avg_reward,
                'success_rate': success_rate,
                'collision_rate': collision_rate,
                'average_episode_length': avg_episode_length,
                'episode_rewards': stats['rewards'],
                'episode_successes': stats['successes'],
                'episode_collisions': stats['collisions'],
                'episode_lengths': [len(traj) for traj in stats['trajectories']]
            }
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"full_evaluation_results_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nEvaluation results saved to: {filepath}")

    def run_evaluation(self):
        print(f"Starting evaluation over {self.total_episodes} episodes...")
        print(f"Using goal guide: {self.use_guide}")
        print(f"Using safety guide: {self.use_safety_guide}")
        if self.use_guide:
            print(f"Goal guide scale: {self.guide_scale}")
        if self.use_safety_guide:
            print(f"Safety guide scale: {self.safety_scale}")
        
        total_return = 0
        total_discounted_return = 0
        
        for episode in range(self.total_episodes):
            print(f"\nEpisode {episode + 1}/{self.total_episodes}")
            self.evaluate_episode(episode + 1, obstacles=True)
            
            # Calculate running averages
            if self.episode_returns:  # Check if we have any returns yet
                avg_return = sum(self.episode_returns) / len(self.episode_returns)
                avg_discounted_return = sum(self.calculate_returns(self.episode_returns)) / len(self.episode_returns)
                success_rate = (self.success_count / (episode + 1)) * 100
                
                print(f"\nRunning Statistics:")
                print(f"Running Success Rate: {success_rate:.2f}%")
                print(f"Average Return: {avg_return:.2f}")
                print(f"Average Discounted Return: {avg_discounted_return:.2f}")
        
        # Final evaluation summary
        final_success_rate = (self.success_count / self.total_episodes) * 100
        final_avg_return = sum(self.episode_returns) / len(self.episode_returns) if self.episode_returns else 0
        final_avg_discounted_return = sum(self.calculate_returns(self.episode_returns)) / len(self.episode_returns) if self.episode_returns else 0
        
        print(f"\nEvaluation complete!")
        print(f"Final Success Rate: {final_success_rate:.2f}% ({self.success_count}/{self.total_episodes})")
        print(f"Final Average Return: {final_avg_return:.2f}")
        print(f"Final Average Discounted Return: {final_avg_discounted_return:.2f}")
        print(f"Average Episode Length: {sum(self.episode_lengths) / len(self.episode_lengths):.2f}")

    def plot_individual_trajectories(self, episode_results, start_pos, goal_pos, safety_scales, episode_num, obstacles=False):
        """
        Create individual trajectory plots for each safety scale, color-coded by safety values.
        """
        # Create directory for individual trajectory plots
        individual_dir = os.path.join(self.plot_dir, f'episode_{episode_num}_individual')
        if not os.path.exists(individual_dir):
            os.makedirs(individual_dir)
        
        # Plot individual trajectories for each safety scale
        for scale in safety_scales:
            if scale in episode_results:
                result = episode_results[scale]
                trajectory = result['trajectory']
                success = result['success']
                collision = result['collision']
                
                # Create figure for this safety scale
                plt.figure(figsize=(12, 10))
                
                # Convert trajectory to numpy array
                trajectory = np.array(trajectory)
                
                # Get safety values from the episode results
                safety_values = result['safety_values']
                actions = result['actions']
                
                # Ensure we have safety values for each trajectory point
                if len(safety_values) != len(trajectory):
                    print(f"Warning: Safety values count ({len(safety_values)}) doesn't match trajectory length ({len(trajectory)})")
                    # Pad or truncate to match
                    if len(safety_values) < len(trajectory):
                        safety_values.extend([safety_values[-1]] * (len(trajectory) - len(safety_values)))
                    else:
                        safety_values = safety_values[:len(trajectory)]
                
                # Convert to numpy array
                safety_values = np.array(safety_values)
                
                # Normalize safety values for color mapping
                if len(safety_values) > 1:
                    safety_normalized = (safety_values - safety_values.min()) / (safety_values.max() - safety_values.min())
                else:
                    safety_normalized = np.array([0.5])  # Default value if only one point
                
                # Create color map
                cmap = plt.cm.RdYlGn  # Red (unsafe) to Green (safe)
                
                # Plot trajectory with color coding
                for i in range(len(trajectory) - 1):
                    x1, y1 = trajectory[i]
                    x2, y2 = trajectory[i + 1]
                    color = cmap(safety_normalized[i])
                    
                    plt.plot([x1, x2], [y1, y2], color=color, linewidth=3, alpha=0.8)
                
                # Plot trajectory points with safety value colors
                scatter = plt.scatter(trajectory[:, 0], trajectory[:, 1], 
                                    c=safety_values, cmap=cmap, s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=plt.gca(), shrink=0.8)
                cbar.set_label('Safety Value', fontsize=12)
                cbar.set_ticks([0, 0.5, 1.0])
                cbar.set_ticklabels(['Unsafe', 'Medium', 'Safe'])
                
                # Plot start position
                plt.plot(start_pos[0], start_pos[1], 'ko', markersize=15, label='Start Position', 
                        markeredgewidth=2, markeredgecolor='black')
                
                # Plot goal position
                plt.plot(goal_pos[0], goal_pos[1], 'k*', markersize=15, label='Goal Position', 
                        markeredgewidth=2, markeredgecolor='black')
                
                # Plot success/failure circle around goal
                circle_color = 'green' if success else 'red'
                circle = plt.Circle((goal_pos[0], goal_pos[1]), self.success_threshold, 
                                  color=circle_color, fill=False, linestyle='--', linewidth=2)
                plt.gca().add_patch(circle)
                
                # Plot obstacles if enabled
                if obstacles:
                    for obs_x, obs_y in self.cylindrical_obstacles:
                        obstacle = plt.Circle((obs_x, obs_y), self.cylindrical_radius, 
                                            color='gray', alpha=0.5, label='Cylindrical Obstacle' if obs_x == self.cylindrical_obstacles[0][0] else "")
                        plt.gca().add_patch(obstacle)
                        
                    for obs_x, obs_y, scale in self.hexagonal_obstacles:
                        obstacle = plt.Circle((obs_x, obs_y), self.hexagonal_radius * scale, 
                                            color='gray', alpha=0.5, label='Hexagonal Obstacle' if obs_x == self.hexagonal_obstacles[0][0] else "")
                        plt.gca().add_patch(obstacle)
                
                # Set plot limits and labels
                plt.xlim(-2, 2)
                plt.ylim(-2, 2)
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=10, loc='upper right')
                
                # Create title with status information
                status = "SUCCESS" if success else "FAILED"
                collision_status = " (COLLISION)" if collision else ""
                title = f'Episode {episode_num} - Safety Scale {scale}\n{status}{collision_status}\nStart: ({start_pos[0]:.2f}, {start_pos[1]:.2f}) | Goal: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})'
                plt.title(title, fontsize=14, fontweight='bold')
                
                plt.xlabel('X Position (m)', fontsize=12)
                plt.ylabel('Y Position (m)', fontsize=12)
                
                # Save plot
                savepath = os.path.join(individual_dir, f'safety_scale_{scale}_trajectory.png')
                plt.savefig(savepath, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Individual trajectory plot for safety scale {scale} saved to {savepath}")

if __name__ == "__main__":
    try:
        # Add command line arguments for guide usage
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--use_guide', action='store_true', default=False, help='Use goal-reaching guide')
        parser.add_argument('--guide_scale', type=float, default=0.1, help='Scale for the goal-reaching guide')
        parser.add_argument('--use_safety_guide', action='store_true', default=True, help='Use safety guide')
        parser.add_argument('--safety_scale', type=float, default=10, help='Scale for the safety guide')
        parser.add_argument('--multi_scale', action='store_true', default=True, help='Run multi-scale evaluation')
        parser.add_argument('--safety_scales', nargs='+', type=float, default=[0, 0.1, 10], help='Safety scales to test')
        parser.add_argument('--full_evaluation', action='store_true', default=True, help='Run full evaluation over 100 episodes with 3 safety scales')
        parser.add_argument('--num_episodes', type=int, default=100, help='Number of episodes for full evaluation')
        args = parser.parse_args()
        
        evaluator = DiffuserEvaluator(
            model_path='/home/othman/diffuser/results_turtlebot_obstacles_expert_4V/state_300000.pt',
            use_images=False,
            use_guide=args.use_guide,
            guide_scale=args.guide_scale,
            use_safety_guide=args.use_safety_guide,
            safety_scale=args.safety_scale,
            safety_model_path='/home/othman/turtlebotExp/safety_model_best.pth'
        )
        
        if args.full_evaluation:
            # Run full evaluation with 3 safety scales
            results = evaluator.run_full_evaluation(
                safety_scales=[0, 0.1, 0.9, 2],
                num_episodes=args.num_episodes,
                obstacles=True
            )
        elif args.multi_scale:
            # Run multi-scale evaluation
            results = evaluator.run_multi_scale_evaluation(
                safety_scales=args.safety_scales,
                obstacles=True
            )
        else:
            # Run regular evaluation
            evaluator.run_evaluation()
            
    except rospy.ROSInterruptException:
        pass