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
from turtlebot_dataset import TurtlebotDataset
from diffuser.models.temporal import TemporalUnet
from diffuser.models.diffusion import GaussianDiffusion
from diffuser.sampling.functions import n_step_guided_p_sample

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

def odom_callback(msg):
    global odom, x_odom, y_odom, theta, linear_x, angular_z
    x_odom = msg.pose.pose.position.x
    y_odom = msg.pose.pose.position.y
    linear_x = msg.twist.twist.linear.x
    angular_z = msg.twist.twist.angular.z
    
    orientation_q = msg.pose.pose.orientation
    _, _, theta = euler_from_quaternion([0, 0, orientation_q.z, orientation_q.w])
    odom = msg

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

    def gradients(self, x, cond, t):
        """
        Compute safety gradients for trajectory optimization.
        x: trajectory tensor of shape (batch_size, horizon, transition_dim)
        cond: dictionary containing the current observation
        t: timestep tensor
        """
        batch_size, horizon, transition_dim = x.shape
        
        # Extract current state from conditions (first element)
        current_obs = cond[0]  # Shape: [obs_dim] (1D tensor)
        
        # Initialize safety values and gradients
        safety_values = torch.zeros(batch_size, device=self.device)
        safety_gradients = torch.zeros_like(x)
        
        # Extract current state components (single observation, not batched)
        x_pos = current_obs[0].item()  # x position
        y_pos = current_obs[1].item()  # y position
        sin_theta = current_obs[4].item()  # sin(theta)
        cos_theta = current_obs[5].item()  # cos(theta)
        
        # For each trajectory in the batch
        for b in range(batch_size):
            # Compute safety for the entire trajectory by looking at the first action
            # This is much more efficient than computing for every timestep
            first_action = x[b, 0, :2]  # First action in the trajectory
            
            # Unnormalize action if needed
            linear_v = first_action[0].item()
            angular_v = first_action[1].item()
            
            # Compute safety score and gradients for the first action
            safety_score, action_grads = self.compute_safety_score(
                x_pos, y_pos, sin_theta, cos_theta, linear_v, angular_v
            )
            
            # Store safety value
            safety_values[b] = safety_score
            
            # Apply gradients only to the first action (most important for immediate safety)
            safety_gradients[b, 0, :2] = torch.tensor(action_grads, device=self.device)
            
            # Optionally, apply smaller gradients to subsequent actions
            # This creates a safety gradient that propagates through the trajectory
            # for h in range(1, min(5, horizon)):  # Only first 5 timesteps
            for h in range(1, horizon):  # Only first 5 timesteps
                # decay_factor = 0.5 ** h  # Exponential decay
                decay_factor = 0.5 - (h/horizon)*0.5  # Exponential decay
                safety_gradients[b, h, :2] = torch.tensor(action_grads, device=self.device) * decay_factor
        
        return safety_values, safety_gradients * self.scale

class GoalGuide:
    def __init__(self, scale=0.1):
        self.scale = scale
        self.values_history = []
        self.gradients_history = []
        self.trajectory_history = []
        
    def gradients(self, x, cond, t):
        """
        Compute the goal-reaching guide value and gradient.
        x: trajectory tensor of shape (batch_size, horizon, transition_dim)
        cond: dictionary containing the goal position
        t: timestep tensor
        """
        # Extract goal from conditions (last two elements of the observation)
        goal = cond[0][-2:]  # Shape: [2] (goal_x, goal_y)
        
        # Extract final positions from trajectories (first 2 dimensions are x,y)
        final_positions = x[:, -1, :2]
        
        # Compute negative distance to goal (so minimizing this maximizes the guide)
        distances = torch.norm(final_positions - goal, dim=-1)
        values = -distances
        
        # Compute gradients
        gradients = torch.zeros_like(x)
        # Only modify the final position
        gradients[:, -1, :2] = (final_positions - goal) / (distances.unsqueeze(-1) + 1e-8)
        
        # Store history for visualization
        self.values_history.append(values.cpu().numpy())
        self.gradients_history.append(gradients[:, -1, :2].cpu().numpy())
        self.trajectory_history.append(x.cpu().numpy())
        
        return values, gradients * self.scale

class DiffuserEvaluator:
    def __init__(self, model_path, use_images=True, use_guide=False, guide_scale=0.1, 
                 use_safety_guide=False, safety_scale=0.1):
        
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
        
        # Initialize guides
        if self.use_guide:
            self.goal_guide = GoalGuide(scale=self.guide_scale)
        
        if self.use_safety_guide:
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
        self.max_steps = 50
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
            print(distance)
            if distance < self.cylindrical_radius+0.04:
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
            obs = torch.tensor([x_odom, y_odom, linear_x, angular_z,
                              math.sin(theta), math.cos(theta)], 
                             dtype=torch.float32).to(self.device)
            return obs.unsqueeze(0)

    def plot_trajectory(self, trajectory, start_pos, goal_pos, success, episode_num, obstacles = False):
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
            traj_dir = os.path.join(denoising_dir, f'trajectory_{b}')
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
                savepath = os.path.join(traj_dir, f'denoising_step_{t}.png')
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
        Efficient guided sampling using the original n_step_guided_p_sample function.
        """
        device = self.device
        horizon = self.diffusion.horizon
        transition_dim = self.diffusion.transition_dim
        
        # Initialize trajectory with noise
        x = torch.randn((batch_size, horizon, transition_dim), device=device)
        x = self.apply_conditioning(x, conditions)
        
        # Create a combined guide if both are enabled
        if self.use_guide and self.use_safety_guide:
            # Create a combined guide that applies both goal and safety gradients
            class CombinedGuide:
                def __init__(self, goal_guide, safety_guide):
                    self.goal_guide = goal_guide
                    self.safety_guide = safety_guide
                
                def gradients(self, x, cond, t):
                    goal_values, goal_gradients = self.goal_guide.gradients(x, cond, t)
                    safety_values, safety_gradients = self.safety_guide.gradients(x, cond, t)
                    
                    # Combine gradients (you can adjust the weighting)
                    combined_gradients = goal_gradients + safety_gradients
                    combined_values = goal_values + safety_values
                    
                    return combined_values, combined_gradients
            
            guide = CombinedGuide(self.goal_guide, self.safety_guide)
        elif self.use_guide:
            guide = self.goal_guide
        elif self.use_safety_guide:
            guide = self.safety_guide
        else:
            # No guide - use standard sampling
            return self.diffusion(conditions).trajectories
        
        # Run reverse diffusion process with guided sampling
        for i in reversed(range(self.diffusion.n_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)

            x = self.apply_conditioning(x, conditions)
            
            # Use the original guided sampling function
            x, values = n_step_guided_p_sample(
                model=self.diffusion,
                x=x,
                cond=conditions,
                t=t,
                guide=guide,
                scale=0.1,  # Combined scale for both guides
                t_stopgrad=2,  # Stop gradients for early timesteps
                n_guide_steps=1,  # Number of gradient steps per diffusion step
                scale_grad_by_std=True
            )
        
        return x

    def apply_conditioning(self, x, conditions):
        """Apply conditioning to the trajectory."""
        # This is a simplified version - you might need to adapt it to your specific conditioning
        for t, cond in conditions.items():
            if t < x.shape[1]:
                x[:, t, :cond.shape[-1]] = cond
        return x

    def evaluate_episode(self, episode_num, obstacles=False, use_safety=False, start_pos=None, goal_pos=None):
        # Reset environment
        self.reset_service()
        rospy.sleep(2)  # Wait for the robot to settle
        
        # Set start position (either provided or random)
        if start_pos is not None:
            # Use provided start position
            initial_x, initial_y, initial_theta = start_pos
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
        else:
            # Use random start position
            self.set_random_start_position(obstacles)
        
        rospy.sleep(2)  # Wait for the robot to settle
        
        # Get initial observation
        obs = self.get_observation()
        if obs is None:
            print("Failed to get initial observation")
            return None, None, None, None
        
        # Set goal position (either provided or random)
        if goal_pos is not None:
            goal_x, goal_y = goal_pos
        else:
            if not obstacles:
                goal_x = random.uniform(-1.4, 1.4)
                goal_y = random.uniform(-1.9, 1.9)
            else:
                goal_x, goal_y = self.get_valid_random_position()

        print(f"Episode {episode_num} - Goal: ({goal_x:.3f}, {goal_y:.3f})")
        obs_with_goal = np.concatenate([obs.cpu().numpy().flatten(), np.array([goal_x, goal_y])])
        
        # Initialize trajectory and metrics
        trajectory = [(x_odom, y_odom)]
        current_start_pos = (x_odom, y_odom)
        current_goal_pos = (goal_x, goal_y)
        rewards = []  # Store rewards for the episode
        
        # Store generated trajectories for plotting
        generated_trajectories = []
        
        # Temporarily set safety guide based on parameter
        original_safety_setting = self.use_safety_guide
        self.use_safety_guide = use_safety
        
        # Run episode
        for step in range(self.max_steps):
            obs = self.get_observation()
            obs_with_goal = np.concatenate([obs.cpu().numpy().flatten(), np.array([goal_x, goal_y])])
            
            # Normalize observation
            normed_obs = self.dataset.normalizer.normalize(obs_with_goal, 'observations')
            normed_obs = torch.FloatTensor(obs_with_goal).to(self.device)
            
            # Create conditions for the model - specify both initial and final states
            conditions = {
                0: normed_obs,  # Initial state
                # self.diffusion.horizon - 20: torch.tensor([goal_x, goal_y, 0, 0, 0, 0, goal_x, goal_y], dtype=torch.float32).to(self.device)  # Final state
            }
            
            # Generate trajectory using guided sampling if guides are enabled
            if self.use_guide or self.use_safety_guide:
                with torch.no_grad():
                    trajectories = self.guided_sampling(conditions, batch_size=1)
                    trajectories = trajectories.cpu().numpy()
                    # Store the generated trajectory for later plotting
                    generated_trajectories.append(trajectories.copy())
            else:
                # Use standard diffusion sampling
                with torch.no_grad():
                    samples = self.diffusion(conditions)
                    trajectories = samples.trajectories.cpu().numpy()
                    # Store the generated trajectory for later plotting
                    generated_trajectories.append(trajectories.copy())
                    
                    # Plot denoising steps if chains are available
                    # if hasattr(samples, 'chains') and samples.chains is not None and episode_num==2 and step>10 or True:
                    #     self.plot_denoising_steps(samples.chains, start_pos, goal_pos, episode_num, step)
            
            # Extract first action
            normed_action = trajectories[0, 0, :self.dataset.action_dim]
            action = self.dataset.normalizer.unnormalize(normed_action, 'actions')
            
            # Execute action
            vel_msg = Twist()
            vel_msg.linear.x = action[0]
            vel_msg.angular.z = action[1]
            self.cmd_vel_pub.publish(vel_msg)
            
            # Wait for action to take effect
            rospy.sleep(0.5)
            
            # Update trajectory
            trajectory.append((x_odom, y_odom))
            
            # Calculate distance to goal
            distance = math.sqrt((x_odom - goal_x)**2 + (y_odom - goal_y)**2)
            
            # Check for collisions
            is_collision = obstacles and not self.is_position_valid(x_odom, y_odom)
            
            # Calculate reward
            reward = self.calculate_reward(distance, step, is_collision)
            rewards.append(reward)
            
            # Check if goal is reached
            print(obs)
            if distance < self.success_threshold or not self.is_position_valid(obs[0][0], obs[0][1]):
                if distance < self.success_threshold:
                    print(f"Goal reached in {step + 1} steps")
                    self.success_count += 1
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
                self.episode_success_rates.append(1.0)
                self.episode_lengths.append(step + 1)
                
                # Print episode summary
                print(f"\nEpisode {episode_num} Summary (Safety: {use_safety}):")
                print(f"Success: {success}")
                print(f"Episode Length: {step + 1}")
                print(f"Total Return: {episode_return:.2f}")
                print(f"Discounted Return: {episode_discounted_return:.2f}")
                
                # Plot generated trajectories at the end of episode
                self.plot_generated_trajectories(generated_trajectories, current_start_pos, current_goal_pos, 
                                               episode_num, use_safety, obstacles)
                
                # Restore original safety setting
                self.use_safety_guide = original_safety_setting
                return trajectory, current_start_pos, current_goal_pos, success
            
            # Check if episode is done
            if step == self.max_steps - 1:
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
                
                # Print episode summary
                print(f"\nEpisode {episode_num} Summary (Safety: {use_safety}):")
                print(f"Success: {success}")
                print(f"Episode Length: {self.max_steps}")
                print(f"Total Return: {episode_return:.2f}")
                print(f"Discounted Return: {episode_discounted_return:.2f}")
                
                # Plot generated trajectories at the end of episode
                self.plot_generated_trajectories(generated_trajectories, current_start_pos, current_goal_pos, 
                                               episode_num, use_safety, obstacles)
                
                # Restore original safety setting
                self.use_safety_guide = original_safety_setting
                return trajectory, current_start_pos, current_goal_pos, success

    def plot_generated_trajectories(self, generated_trajectories, start_pos, goal_pos, episode_num, use_safety, obstacles=False):
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

    def generate_episode_positions(self, obstacles = False):
        """Generate start and goal positions for an episode."""
        # Generate start position
        if not obstacles:
            start_x = random.uniform(-1.4, 1.4)
            start_y = random.uniform(-1.9, 1.9)
        else:
            start_x, start_y = self.get_valid_random_position()
        start_theta = random.uniform(-math.pi, math.pi)
        
        # Generate goal position
        if not obstacles:
            goal_x = random.uniform(-1.4, 1.4)
            goal_y = random.uniform(-1.9, 1.9)
        else:
            goal_x, goal_y = self.get_valid_random_position()
        
        return (start_x, start_y, start_theta), (goal_x, goal_y)

    def plot_comparison_trajectories(self, trajectory_with_safety, trajectory_without_safety, 
                                   start_pos, goal_pos, episode_num, obstacles=False):
        """Plot both trajectories (with and without safety) for comparison."""
        plt.figure(figsize=(12, 10))
        
        # Plot trajectory with safety guidance
        if trajectory_with_safety is not None:
            trajectory_with_safety = np.array(trajectory_with_safety)
            plt.plot(trajectory_with_safety[:, 0], trajectory_with_safety[:, 1], 
                    'b-', label='With Safety Guide', linewidth=3, alpha=0.8)
        
        # Plot trajectory without safety guidance
        if trajectory_without_safety is not None:
            trajectory_without_safety = np.array(trajectory_without_safety)
            plt.plot(trajectory_without_safety[:, 0], trajectory_without_safety[:, 1], 
                    'r-', label='Without Safety Guide', linewidth=3, alpha=0.8)
        
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
        plt.legend(fontsize=12)
        plt.title(f'Trajectory Comparison - Episode {episode_num}\nBlue: With Safety Guide | Red: Without Safety Guide', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('X Position (m)', fontsize=12)
        plt.ylabel('Y Position (m)', fontsize=12)
        
        # Save plot
        savepath = os.path.join(self.plot_dir, f'trajectory_comparison_episode_{episode_num}.png')
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Trajectory comparison saved to {savepath}")

    def run_evaluation(self):
        print(f"Starting evaluation over {self.total_episodes} episodes...")
        print(f"Using goal guide: {self.use_guide}")
        print(f"Using safety guide: {self.use_safety_guide}")
        if self.use_guide:
            print(f"Goal guide scale: {self.guide_scale}")
        if self.use_safety_guide:
            print(f"Safety guide scale: {self.safety_scale}")
        
        # Reset success count for fair comparison
        self.success_count = 0
        self.episode_returns = []
        self.episode_success_rates = []
        self.episode_lengths = []
        
        for episode in range(self.total_episodes):
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{self.total_episodes}")
            print(f"{'='*60}")
            
            # Generate start and goal positions once for this episode
            start_pos, goal_pos = self.generate_episode_positions(obstacles=True)
            print(f"Generated positions - Start: ({start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}), Goal: ({goal_pos[0]:.3f}, {goal_pos[1]:.3f})")
            
            # Run episode with safety guidance
            print(f"\n--- Running WITH Safety Guidance ---")
            trajectory_with_safety, actual_start_pos, actual_goal_pos, success_with_safety = self.evaluate_episode(
                episode + 1, obstacles=True, use_safety=True, start_pos=start_pos, goal_pos=goal_pos
            )
            
            # Run episode without safety guidance (no need to reset, just run with same positions)
            print(f"\n--- Running WITHOUT Safety Guidance ---")
            trajectory_without_safety, _, _, success_without_safety = self.evaluate_episode(
                episode + 1, obstacles=True, use_safety=False, start_pos=start_pos, goal_pos=goal_pos
            )
            
            # Plot comparison
            self.plot_comparison_trajectories(
                trajectory_with_safety, trajectory_without_safety, 
                actual_start_pos, actual_goal_pos, episode + 1, obstacles=True
            )
            
            # Print comparison summary
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1} Comparison Summary:")
            print(f"With Safety Guide: {'SUCCESS' if success_with_safety else 'FAILURE'}")
            print(f"Without Safety Guide: {'SUCCESS' if success_without_safety else 'FAILURE'}")
            print(f"{'='*60}")
            
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
        
        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Final Success Rate: {final_success_rate:.2f}% ({self.success_count}/{self.total_episodes})")
        print(f"Final Average Return: {final_avg_return:.2f}")
        print(f"Final Average Discounted Return: {final_avg_discounted_return:.2f}")
        print(f"Average Episode Length: {sum(self.episode_lengths) / len(self.episode_lengths):.2f}")
        print(f"\nComparison plots saved in: {self.plot_dir}")

if __name__ == "__main__":
    try:
        # Add command line arguments for guide usage
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--use_guide', action='store_true', default=False, help='Use goal-reaching guide')
        parser.add_argument('--guide_scale', type=float, default=0.1, help='Scale for the goal-reaching guide')
        parser.add_argument('--use_safety_guide', action='store_true', default=True, help='Use safety guide')
        parser.add_argument('--safety_scale', type=float, default=1, help='Scale for the safety guide')
        args = parser.parse_args()
        
        evaluator = DiffuserEvaluator(
            model_path='/home/othman/diffuser/results_turtlebot_expert_obstacles/state_60000.pt',
            use_images=False,
            use_guide=args.use_guide,
            guide_scale=args.guide_scale,
            use_safety_guide=args.use_safety_guide,
            safety_scale=args.safety_scale
        )
        evaluator.run_evaluation()
    except rospy.ROSInterruptException:
        pass