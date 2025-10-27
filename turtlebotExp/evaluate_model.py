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
import matplotlib.pyplot as plt
import os
from policy import Policy  # Import from new policy file
from transformers import AutoModel, AutoProcessor
from PIL import Image as PILImage
import time
import pickle
import torch.nn as nn

# Set random seeds for reproducibility
torch.manual_seed(4)
np.random.seed(4)
random.seed(4)

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

# Global variables for ROS callbacks
odom = None
image = None
x_odom = 0.0
y_odom = 0.0
theta = 0.0
linear_x = 0.0
angular_z = 0.0
bridge = CvBridge()

def image_callback(data):
    global image
    try:
        image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
    except Exception as e:
        rospy.logerr("Image conversion error: %s", str(e))

def odom_callback(msg):
    global odom, x_odom, y_odom, theta, linear_x, angular_z
    x_odom = msg.pose.pose.position.x
    y_odom = msg.pose.pose.position.y
    linear_x = msg.twist.twist.linear.x
    angular_z = msg.twist.twist.angular.z
    
    orientation_q = msg.pose.pose.orientation
    _, _, theta = euler_from_quaternion([0, 0, orientation_q.z, orientation_q.w])
    odom = msg

class JEPAProcessor:
    def __init__(self, model_path):
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path).cuda()
        self.model.eval()
        
    def process_single_image(self, image):
        """Process a single image through JEPA model."""
        if image is None:
            return None
            
        try:
            # Convert OpenCV image to PIL
            image_pil = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            with torch.no_grad():
                inputs = self.processor(image_pil, return_tensors="pt")
                inputs = {k: v.cuda() for k, v in inputs.items()}
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)
                return embedding.cpu().numpy()
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

class ModelEvaluator:
    def __init__(self, model_path, use_images=True, jepa_model_path=None, eval_with_gradient=False, safety_model_path=None):
        # Initialize ROS node
        rospy.init_node("model_evaluator")
        
        # Setup ROS publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        rospy.Subscriber("/odom", Odometry, odom_callback)
        rospy.Subscriber("/overhead_camera/overhead_camera/image_raw", Image, image_callback)
        
        # Setup ROS services
        self.reset_service = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.set_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        # Initialize JEPA if model path is provided
        self.jepa_processor = None
        if jepa_model_path:
            print("Loading JEPA model...")
            self.jepa_processor = JEPAProcessor(jepa_model_path)
        
        # Load the trained model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Policy(images=False, use_jepa=jepa_model_path is not None).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.use_images = use_images
        self.use_jepa = jepa_model_path is not None
        
        # Gradient evaluation flag
        self.eval_with_gradient = eval_with_gradient
        
        # Load safety model for gradient optimization
        self.safety_model = None
        if eval_with_gradient and safety_model_path:
            print("Loading safety model for gradient optimization...")
            if os.path.exists(safety_model_path):
                # Load the state dict first
                state_dict = torch.load(safety_model_path, map_location=self.device)
                
                # Create model and load state dict
                self.safety_model = SafetyPredictor(input_dim=8, hidden_dims=[128, 64, 32]).to(self.device)
                self.safety_model.load_state_dict(state_dict)
                
                # Enable gradients for all parameters AFTER loading
                for param in self.safety_model.parameters():
                    param.requires_grad = True
                
                self.safety_model.eval()
                print(f"Safety model loaded from {safety_model_path}")
                
                # Test gradient computation
                self.test_gradient_computation()
            else:
                print(f"Warning: Safety model file {safety_model_path} not found!")
                self.eval_with_gradient = False
        
        # Evaluation metrics
        self.success_count = 0
        self.success_count_gradient = 0  # Separate counter for gradient evaluation λ=0.1
        self.success_count_gradient_lambda2 = 0  # Separate counter for gradient evaluation λ=0.2
        self.total_episodes = 100
        self.max_steps = 50*5
        self.success_threshold = 0.2  # meters
        
        # Returns and rewards parameters
        self.gamma = 0.99  # discount factor
        self.goal_reward = 10.0  # reward for reaching goal
        self.step_penalty = -0.1  # penalty for each step
        self.collision_penalty = -1.0  # penalty for collision
        # self.out_of_bounds_penalty = -5.0  # penalty for going out of bounds
        
        # Metrics tracking - separate for normal and gradient
        self.episode_returns = []  # List to store returns for each episode (normal)
        self.episode_returns_gradient = []  # List to store returns for each episode (gradient λ=0.1)
        self.episode_returns_gradient_lambda2 = []  # List to store returns for each episode (gradient λ=0.2)
        self.episode_discounted_returns = []  # List to store discounted returns for each episode (normal)
        self.episode_discounted_returns_gradient = []  # List to store discounted returns for each episode (gradient λ=0.1)
        self.episode_discounted_returns_gradient_lambda2 = []  # List to store discounted returns for each episode (gradient λ=0.2)
        self.episode_success_rates = []  # List to store success rates (normal)
        self.episode_success_rates_gradient = []  # List to store success rates (gradient λ=0.1)
        self.episode_success_rates_gradient_lambda2 = []  # List to store success rates (gradient λ=0.2)
        self.episode_lengths = []  # List to store episode lengths (normal)
        self.episode_lengths_gradient = []  # List to store episode lengths (gradient λ=0.1)
        self.episode_lengths_gradient_lambda2 = []  # List to store episode lengths (gradient λ=0.2)
        
        # Timing tracking
        self.prediction_times = []  # List to store prediction times for each step (normal)
        self.prediction_times_gradient = []  # List to store prediction times for each step (gradient λ=0.1)
        self.prediction_times_gradient_lambda2 = []  # List to store prediction times for each step (gradient λ=0.2)
        self.total_prediction_time = 0.0  # Total time spent on predictions (normal)
        self.total_prediction_time_gradient = 0.0  # Total time spent on predictions (gradient λ=0.1)
        self.total_prediction_time_gradient_lambda2 = 0.0  # Total time spent on predictions (gradient λ=0.2)
        
        # Create directory for trajectory plots if it doesn't exist
        self.plot_dir = "trajectory_plots_2"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
            
        # Define obstacle positions from the actual TurtleBot3 world
        # Cylindrical obstacles (radius 0.15m)
        self.cylindrical_obstacles = [
            (-1.1, -1.1), (-1.1, 0.0), (-1.1, 1.1),  # Left column
            (0.0, -1.1), (0.0, 0.0), (0.0, 1.1),     # Middle column
            (1.1, -1.1), (1.1, 0.0), (1.1, 1.1)      # Right column
        ]
        self.cylindrical_radius = 0.15  # meters
        
        # Hexagonal obstacles (approximate positions and sizes)
        self.hexagonal_obstacles = [
            (3.5, 0.0, 0.8),      # Head
            (1.8, 2.7, 0.55),     # Left hand
            (1.8, -2.7, 0.55),    # Right hand
            (-1.8, 2.7, 0.55),    # Left foot
            (-1.8, -2.7, 0.55)    # Right foot
        ]
        self.hexagonal_radius = 0.4  # Approximate radius for collision checking

    def compute_safety_physical(self, x, y, sin_theta, cos_theta, linear_v, angular_v, obstacles, radius, dt=0.1, steps=10):
        """
        Compute safety score of taking (linear_v, angular_v) from current pose.

        Returns a scalar safety value between 0 (unsafe) and 1 (safe).
        """
        # Reconstruct angle from sin and cos
        theta = np.arctan2(sin_theta, cos_theta)
        
        positions = []
        for _ in range(steps):
            x += linear_v * np.cos(theta) * dt
            y += linear_v * np.sin(theta) * dt
            theta += angular_v * dt
            positions.append((x, y))
        
        # Compute distance to each obstacle at each step
        min_dist = float("inf")
        for px, py in positions:
            for ox, oy in obstacles:
                dist = np.hypot(px - ox, py - oy) - radius
                if dist < min_dist:
                    min_dist = dist

        # Normalize and clamp
        safe_distance = 0.5  # beyond this distance is considered "safe enough"
        if min_dist <= 0:
            return 0.0  # collision
        else:
            # Use sigmoid or piecewise function
            return float(np.tanh(min_dist / safe_distance))  # ∈ (0, 1)
        
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

    def compute_safety_gradients(self, dynamics, action):
        safety_score, grad = self.compute_safety_gradient_physical(x=dynamics[0], y=dynamics[1], sin_theta=dynamics[4], cos_theta=dynamics[5], obstacles=self.cylindrical_obstacles, linear_v=action[0], angular_v=action[1])
        return grad, safety_score
        
        """Compute gradients of safety with respect to action"""
        if self.safety_model is None:
            return np.array([0.0, 0.0]), 0.0
            
        try:
            # Create tensors with gradients enabled from the start
            dynamics_tensor = torch.tensor(dynamics, dtype=torch.float32, requires_grad=True, device=self.device)
            action_tensor = torch.tensor(action, dtype=torch.float32, requires_grad=True, device=self.device)
            
            # Concatenate and add batch dimension
            input_data = torch.cat([dynamics_tensor, action_tensor]).unsqueeze(0)
            
            # Set model to training mode for gradient computation
            self.safety_model.train()
            
            # Check if model parameters have gradients enabled
            has_gradients = any(param.requires_grad for param in self.safety_model.parameters())
            if not has_gradients:
                print("Warning: Safety model parameters don't have gradients enabled!")
                # Try to enable gradients
                for param in self.safety_model.parameters():
                    param.requires_grad = True
                has_gradients = any(param.requires_grad for param in self.safety_model.parameters())
                if not has_gradients:
                    return np.array([0.0, 0.0]), 0.0
            
            # Forward pass
            safety_score = self.safety_model(input_data)
            
            # Check if safety_score has gradients
            if not safety_score.requires_grad:
                print("Warning: Safety score doesn't require gradients!")
                return np.array([0.0, 0.0]), safety_score.item()
            
            # Backward pass to compute gradients
            safety_score.backward()
            
            # Check if gradients were computed
            if action_tensor.grad is None:
                print("Warning: No gradients computed for action tensor!")
                return np.array([0.0, 0.0]), safety_score.item()
            
            # Extract gradients with respect to action
            action_gradients = action_tensor.grad.cpu().numpy()
            
            # Don't set model back to eval mode here - let calling code manage it
            return action_gradients, safety_score.item()
            
        except Exception as e:
            print(f"Error in compute_safety_gradients: {e}")
            return np.array([0.0, 0.0]), 0.0

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

    def get_valid_random_positions_with_min_distance(self, min_distance=1.2):
        """Generate start and goal positions that avoid obstacles and are at least min_distance apart"""
        max_attempts = 200
        for _ in range(max_attempts):
            # Generate start position
            if random.random() < 0.5:  # 50% chance to use obstacle-aware generation
                start_x, start_y = self.get_valid_random_position()
            else:
                start_x = random.uniform(-1.4, 1.4)
                start_y = random.uniform(-1.9, 1.9)
            
            # Generate goal position
            if random.random() < 0.5:  # 50% chance to use obstacle-aware generation
                goal_x, goal_y = self.get_valid_random_position()
            else:
                goal_x = random.uniform(-1.4, 1.4)
                goal_y = random.uniform(-1.9, 1.9)
            
            # Check if both positions are valid and far enough apart
            distance = math.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
            if distance >= min_distance:
                return (start_x, start_y), (goal_x, goal_y)
        
        # If no valid pair found, return default positions with sufficient distance
        return (-1.3, -1.8), (1.3, 1.8)  # Far corners with distance > 1.2

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
                
            if self.use_jepa:
                # Process image through JEPA
                jepa_embedding = self.jepa_processor.process_single_image(image)
                if jepa_embedding is None:
                    return None
                # Convert to tensor
                return torch.from_numpy(jepa_embedding).float().to(self.device)
            else:
                # Process raw image
                img = cv2.resize(image, (640, 480))
                img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
                return img_tensor.unsqueeze(0).to(self.device)
        else:
            # Process vector observation to match training data
            # obs = torch.tensor( [x_odom, y_odom, math.sin(theta), math.cos(theta)], dtype=torch.float32).to(self.device)
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
        
        # Plot only cylindrical obstacles (no hexagonal obstacles)
        if obstacles:
            for obs_x, obs_y in self.cylindrical_obstacles:
                obstacle = plt.Circle((obs_x, obs_y), self.cylindrical_radius, 
                                    color='gray', alpha=0.5)
                plt.gca().add_patch(obstacle)
        
        # Focus on center area with cylinder grid and margin
        # Cylinder grid spans from -1.1 to 1.1 in both x and y, with 0.15 radius
        # Add margin of 0.3 around the cylinder grid
        margin = 0.3
        plt.autoscale(False)
        plt.xlim(-1.1 - margin, 1.1 + margin)
        plt.ylim(-1.1 - margin, 1.1 + margin)
        
        # Set plot properties
        plt.grid(True)
        plt.axis('equal')
        plt.title(f'Episode {episode_num} - {"Success" if success else "Failure"}')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.legend()
        
        # Save plot
        status = "success" if success else "fail"
        plt.savefig(f'{self.plot_dir}/{status}_{episode_num}.jpg')
        plt.close()

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

    def evaluate_episode(self, episode_num, obstacles=False):
        if self.eval_with_gradient:
            # Run episode twice: once normal, once with gradient optimization
            print(f"\nRunning episode {episode_num + 1} with gradient comparison...")
            
            # Generate start and goal positions once for both runs
            self.reset_service()
            self.set_random_start_position(obstacles)
            rospy.sleep(1)  # Wait for reset to complete
            
            # Generate start and goal positions with minimum distance constraint
            if obstacles:
                start_pos, goal_pos = self.get_valid_random_positions_with_min_distance(min_distance=1.2)
            else:
                # For non-obstacle case, generate positions with minimum distance
                max_attempts = 100
                for _ in range(max_attempts):
                    start_x = random.uniform(-1.4, 1.4)
                    start_y = random.uniform(-1.9, 1.9)
                    goal_x = random.uniform(-1.4, 1.4)
                    goal_y = random.uniform(-1.9, 1.9)
                    
                    distance = math.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
                    if distance >= 1.2:
                        start_pos = (start_x, start_y)
                        goal_pos = (goal_x, goal_y)
                        break
                else:
                    # Fallback to far corners if no valid pair found
                    start_pos = (-1.3, -1.8)
                    goal_pos = (1.3, 1.8)
            
            distance = math.sqrt((goal_pos[0] - start_pos[0])**2 + (goal_pos[1] - start_pos[1])**2)
            print(f"Using same start: {start_pos}, goal: {goal_pos} (distance: {distance:.2f}m) for all three runs")
            
            # Run normal episode with fixed start/goal
            print("Running normal episode...")
            success_normal, episode_return_normal, episode_discounted_return_normal, trajectory_normal = self._run_single_episode_with_fixed_positions(episode_num, obstacles, start_pos, goal_pos, use_gradient=False)
            
            # Run gradient-optimized episode with λ=0.1
            print("Running gradient-optimized episode (λ=0.1)...")
            success_gradient, episode_return_gradient, episode_discounted_return_gradient, trajectory_gradient = self._run_single_episode_with_fixed_positions(episode_num, obstacles, start_pos, goal_pos, use_gradient=True, learning_rate=0.1)
            
            # Run gradient-optimized episode with λ=0.2
            print("Running gradient-optimized episode (λ=0.2)...")
            success_gradient_lambda2, episode_return_gradient_lambda2, episode_discounted_return_gradient_lambda2, trajectory_gradient_lambda2 = self._run_single_episode_with_fixed_positions(episode_num, obstacles, start_pos, goal_pos, use_gradient=True, learning_rate=0.2)
            
            # Plot all three trajectories for comparison
            self.plot_trajectory_comparison_three(trajectory_normal, trajectory_gradient, trajectory_gradient_lambda2, episode_num, 
                                                success_normal, success_gradient, success_gradient_lambda2, obstacles, goal_pos)
            
            # Store metrics separately for normal and both gradient variants
            self.episode_returns.append(episode_return_normal)
            self.episode_returns_gradient.append(episode_return_gradient)
            self.episode_returns_gradient_lambda2.append(episode_return_gradient_lambda2)
            self.episode_discounted_returns.append(episode_discounted_return_normal)
            self.episode_discounted_returns_gradient.append(episode_discounted_return_gradient)
            self.episode_discounted_returns_gradient_lambda2.append(episode_discounted_return_gradient_lambda2)
            self.episode_success_rates.append(1.0 if success_normal else 0.0)
            self.episode_success_rates_gradient.append(1.0 if success_gradient else 0.0)
            self.episode_success_rates_gradient_lambda2.append(1.0 if success_gradient_lambda2 else 0.0)
            self.episode_lengths.append(len(trajectory_normal) if trajectory_normal else 0)
            self.episode_lengths_gradient.append(len(trajectory_gradient) if trajectory_gradient else 0)
            self.episode_lengths_gradient_lambda2.append(len(trajectory_gradient_lambda2) if trajectory_gradient_lambda2 else 0)
            
            # Use normal episode results for metrics (or you could average them)
            success = success_normal
            episode_return = episode_return_normal
            episode_discounted_return = episode_discounted_return_normal
            trajectory = trajectory_normal
            
            # Print comparison
            print(f"\nEpisode {episode_num + 1} Comparison:")
            print(f"Normal - Success: {success_normal}, Return: {episode_return_normal:.2f}")
            print(f"Gradient λ=0.1 - Success: {success_gradient}, Return: {episode_return_gradient:.2f}")
            print(f"Gradient λ=0.2 - Success: {success_gradient_lambda2}, Return: {episode_return_gradient_lambda2:.2f}")
            
        else:
            # Run single episode normally
            success, episode_return, episode_discounted_return, trajectory = self._run_single_episode_with_fixed_positions(episode_num, obstacles, use_gradient=False)
            
            # Plot single trajectory
            start_pos = trajectory[0] if trajectory else (0, 0)
            goal_pos = trajectory[-1] if trajectory else (0, 0)
            self.plot_trajectory(trajectory, start_pos, goal_pos, success, episode_num, obstacles=obstacles)
            
            # Store metrics for normal evaluation only
            self.episode_returns.append(episode_return)
            self.episode_discounted_returns.append(episode_discounted_return)
            self.episode_success_rates.append(1.0 if success else 0.0)
            self.episode_lengths.append(len(trajectory) if trajectory else 0)
        
        # Print episode summary
        print(f"\nEpisode {episode_num + 1} Summary:")
        print(f"Success: {success}")
        print(f"Episode Length: {len(trajectory) if trajectory else 0}")
        print(f"Total Return: {episode_return:.2f}")
        print(f"Discounted Return: {episode_discounted_return:.2f}")
        
        # Print timing summary for this episode
        if self.eval_with_gradient:
            # Get timing for this episode (normal)
            episode_prediction_times = self.prediction_times[-len(trajectory):] if len(trajectory) > 0 and len(self.prediction_times) >= len(trajectory) else []
            episode_prediction_times_gradient = self.prediction_times_gradient[-len(trajectory):] if len(trajectory) > 0 and len(self.prediction_times_gradient) >= len(trajectory) else []
            
            if episode_prediction_times:
                avg_episode_prediction_time = sum(episode_prediction_times) / len(episode_prediction_times)
                print(f"Average Prediction Time (Normal): {avg_episode_prediction_time*1000:.2f} ms")
            
            if episode_prediction_times_gradient:
                avg_episode_prediction_time_gradient = sum(episode_prediction_times_gradient) / len(episode_prediction_times_gradient)
                print(f"Average Prediction Time (Gradient): {avg_episode_prediction_time_gradient*1000:.2f} ms")
        else:
            # Get timing for this episode (normal only)
            episode_prediction_times = self.prediction_times[-len(trajectory):] if len(trajectory) > 0 and len(self.prediction_times) >= len(trajectory) else []
            if episode_prediction_times:
                avg_episode_prediction_time = sum(episode_prediction_times) / len(episode_prediction_times)
                print(f"Average Prediction Time: {avg_episode_prediction_time*1000:.2f} ms")
        
        if self.eval_with_gradient:
            return success, success_gradient, success_gradient_lambda2, episode_return, episode_discounted_return
        else:
            return success, False, False, episode_return, episode_discounted_return

    def _run_single_episode_with_fixed_positions(self, episode_num, obstacles=False, start_pos=None, goal_pos=None, use_gradient=False, learning_rate=0.1):
        """Run a single episode with fixed start and goal positions"""
        # Reset environment and set to specific start position
        self.reset_service()
        
        if start_pos:
            # Set to specific start position
            state_msg = SetModelStateRequest()
            state_msg.model_state.model_name = 'turtlebot3_burger'
            state_msg.model_state.pose.position.x = start_pos[0]
            state_msg.model_state.pose.position.y = start_pos[1]
            initial_theta = random.uniform(-math.pi, math.pi)  # Random orientation
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
            # Use random position if not specified
            self.set_random_start_position(obstacles)
            
        rospy.sleep(1)  # Wait for reset to complete
        
        # Use provided goal position or generate random one with minimum distance constraint
        if goal_pos:
            goal_x, goal_y = goal_pos
        else:
            # Generate goal position with minimum distance from start position
            current_start_pos = (x_odom, y_odom)  # Current robot position after reset
            
            if obstacles:
                # Try to find a valid goal position with minimum distance
                max_attempts = 100
                for _ in range(max_attempts):
                    goal_x, goal_y = self.get_valid_random_position()
                    distance = math.sqrt((goal_x - current_start_pos[0])**2 + (goal_y - current_start_pos[1])**2)
                    if distance >= 1.2:
                        break
                else:
                    # Fallback: use a position far from start
                    goal_x = -current_start_pos[0] if current_start_pos[0] > 0 else 1.3
                    goal_y = -current_start_pos[1] if current_start_pos[1] > 0 else 1.8
            else:
                # For non-obstacle case, generate goal with minimum distance
                max_attempts = 100
                for _ in range(max_attempts):
                    goal_x = random.uniform(-1.4, 1.4)
                    goal_y = random.uniform(-1.9, 1.9)
                    distance = math.sqrt((goal_x - current_start_pos[0])**2 + (goal_y - current_start_pos[1])**2)
                    if distance >= 1.2:
                        break
                else:
                    # Fallback: use opposite corner
                    goal_x = -current_start_pos[0] if current_start_pos[0] > 0 else 1.3
                    goal_y = -current_start_pos[1] if current_start_pos[1] > 0 else 1.8
                
        goal = torch.tensor([goal_x, goal_y], dtype=torch.float32).to(self.device)
        
        # Initialize episode variables
        step = 0
        success = False
        trajectory = [(x_odom, y_odom)]  # Store trajectory points
        rewards = []  # Store rewards for the episode
        
        while step < self.max_steps and not rospy.is_shutdown():
            # Start timing for this prediction step
            step_start_time = time.time()
            
            # Get current observation
            obs = self.get_observation()
            if obs is None:
                continue
                
            # Get model prediction
            # with torch.no_grad():
            obs = obs.unsqueeze(0) if len(obs.shape) == 1 else obs
            goal = goal.unsqueeze(0) if len(goal.shape) == 1 else goal
            t = torch.tensor([1], dtype=torch.float32).to(self.device).unsqueeze(0)
            
            
            mu, sigma = self.model(obs, goal, t)
            dist = torch.distributions.Normal(mu, sigma)
            action = dist.sample()*1.2
            
            # Apply gradient optimization if enabled
            if use_gradient and self.safety_model is not None:
                # Get current dynamics
                dynamics = [x_odom, y_odom, linear_x, angular_z, math.sin(theta), math.cos(theta)]
                initial_action = [float(action[0, 0]), float(action[0, 1])]
                
                # Compute safety gradients
                gradients, safety_score = self.compute_safety_gradients(dynamics, initial_action)

                if safety_score < 0.2:
                    optimized_action = np.array([gradients[0], gradients[1]])

                else:
                    # Apply gradient to action (gradient ascent for safety)
                    action_update = learning_rate * gradients
                    
                    # Apply update
                    optimized_action = np.array([initial_action[0], initial_action[1] + action_update[1]])
                
                # Clamp actions to valid ranges
                optimized_action[0] = np.clip(optimized_action[0], 0.0, 0.5)  # linear velocity
                optimized_action[1] = np.clip(optimized_action[1], -1.0, 1.0)  # angular velocity
                
                # Use optimized action - convert back to tensor
                action[0, 0] = torch.tensor(optimized_action[0], dtype=torch.float32, device=self.device)
                action[0, 1] = torch.tensor(optimized_action[1], dtype=torch.float32, device=self.device)
                
                # Set safety model back to eval mode after gradient computation
                self.safety_model.eval()
                
                print(f"Step {step}: Original action: {initial_action}, Optimized: {optimized_action}, Safety: {safety_score:.3f}")
            
            # End timing for this prediction step
            step_end_time = time.time()
            prediction_time = step_end_time - step_start_time
            
            # Store prediction time
            if use_gradient:
                if learning_rate == 0.1:
                    self.prediction_times_gradient.append(prediction_time)
                    self.total_prediction_time_gradient += prediction_time
                elif learning_rate == 0.2:
                    self.prediction_times_gradient_lambda2.append(prediction_time)
                    self.total_prediction_time_gradient_lambda2 += prediction_time
            else:
                self.prediction_times.append(prediction_time)
                self.total_prediction_time += prediction_time
            
            # Print timing info every 10 steps or for the first few steps
            if step < 5 or step % 10 == 0:
                print(f"Step {step}: Prediction time: {prediction_time*1000:.2f} ms")
                if use_gradient:
                    print(f"Step {step}: Prediction time GRAD: {prediction_time*1000:.2f} ms")
                
            # Execute action
            vel_msg = Twist()
            vel_msg.linear.x = float(action[0, 0])
            vel_msg.angular.z = float(action[0, 1])
            self.cmd_vel_pub.publish(vel_msg)
            
            # Store current position in trajectory
            trajectory.append((x_odom, y_odom))
            
            # Calculate distance to goal
            distance = math.sqrt((goal_x - x_odom)**2 + (goal_y - y_odom)**2)
            
            # Check for collisions and out of bounds
            is_collision = obstacles and not self.is_position_valid(x_odom, y_odom)
            is_out_of_bounds = abs(x_odom) > 1.4 or abs(y_odom) > 1.9
            
            # Calculate reward
            reward = self.calculate_reward(distance, step, is_collision)
            rewards.append(reward)
            
            # Check if goal is reached
            if distance < self.success_threshold:
                success = True
                break

            if is_collision:
                success = False
                break
                
            # Check if robot is out of bounds or collided with obstacle
            # if abs(x_odom) > 1.4 or abs(y_odom) > 1.9:
            # if self.use_jepa and abs(x_odom) > 1.4 or abs(y_odom) > 1.9 or (obstacles and not self.is_position_valid(x_odom, y_odom)):
            #     return -1,-1,-1
                
            step += 1
            rospy.sleep(0.05)  # Control rate
            
        # Stop robot
        vel_msg = Twist()
        vel_msg.linear.x = 0.0
        vel_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(vel_msg)
        
        # Calculate episode metrics
        episode_return = sum(rewards)
        discounted_returns = self.calculate_returns(rewards)
        episode_discounted_return = discounted_returns[0] if discounted_returns else 0
        
        return success, episode_return, episode_discounted_return, trajectory

    def plot_trajectory_comparison(self, trajectory_normal, trajectory_gradient, episode_num, 
                                 success_normal, success_gradient, obstacles=False, goal_pos=None):
        """Plot comparison of normal vs gradient-optimized trajectories"""
        plt.figure(figsize=(15, 10))
        
        # Focus on center area with cylinder grid and margin
        # Cylinder grid spans from -1.1 to 1.1 in both x and y, with 0.15 radius
        # Add margin of 0.3 around the cylinder grid
        margin = 0.3
        plt.xlim(-1.1 - margin, 1.1 + margin)
        plt.ylim(-1.1 - margin, 1.1 + margin)
        
        if obstacles:
            # Plot only cylindrical obstacles (no hexagonal obstacles)
            for obs_x, obs_y in self.cylindrical_obstacles:
                obstacle = plt.Circle((obs_x, obs_y), self.cylindrical_radius, 
                                    color='gray', alpha=0.5)
                plt.gca().add_patch(obstacle)
        
        # Plot goal position if provided
        if goal_pos:
            plt.plot(goal_pos[0], goal_pos[1], 'ro', markersize=12, label='Goal Position')
            # Plot success threshold circle around goal
            circle = plt.Circle((goal_pos[0], goal_pos[1]), self.success_threshold, 
                              color='g', fill=False, linestyle='--', alpha=0.7)
            plt.gca().add_patch(circle)
        
        # Plot trajectories
        if trajectory_normal:
            normal_x, normal_y = zip(*trajectory_normal)
            plt.plot(normal_x, normal_y, 'b-', linewidth=2, label=f'Normal (Success: {success_normal})', alpha=0.8)
            plt.plot(normal_x[0], normal_y[0], 'bo', markersize=8, label='Start')
            plt.plot(normal_x[-1], normal_y[-1], 'bs', markersize=8, label='End Normal')
        
        if trajectory_gradient:
            gradient_x, gradient_y = zip(*trajectory_gradient)
            plt.plot(gradient_x, gradient_y, 'r-', linewidth=2, label=f'Gradient (Success: {success_gradient})', alpha=0.8)
            plt.plot(gradient_x[-1], gradient_y[-1], 'rs', markersize=8, label='End Gradient')
        
        plt.autoscale(False)
        plt.xlim(-1.1 - margin, 1.1 + margin)
        plt.ylim(-1.1 - margin, 1.1 + margin)
        
        # Set plot properties
        plt.grid(True)
        plt.axis('equal')
        plt.title(f'Episode {episode_num} - Trajectory Comparison')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.legend()
        
        # Save plot
        plt.savefig(f'{self.plot_dir}/comparison_{episode_num}.jpg')
        plt.close()

    def plot_trajectory_comparison_three(self, trajectory_normal, trajectory_gradient, trajectory_gradient_lambda2, episode_num, 
                                       success_normal, success_gradient, success_gradient_lambda2, obstacles=False, goal_pos=None):
        """Plot comparison of normal vs gradient-optimized trajectories with two learning rates"""
        plt.figure(figsize=(15, 10))
        
        # Focus on center area with cylinder grid and margin
        # Cylinder grid spans from -1.1 to 1.1 in both x and y, with 0.15 radius
        # Add margin of 0.3 around the cylinder grid
        margin = 0.3
        plt.xlim(-1.1 - margin, 1.1 + margin)
        plt.ylim(-1.1 - margin, 1.1 + margin)
        
        if obstacles:
            # Plot only cylindrical obstacles (no hexagonal obstacles)
            for obs_x, obs_y in self.cylindrical_obstacles:
                obstacle = plt.Circle((obs_x, obs_y), self.cylindrical_radius, 
                                    color='gray', alpha=0.5)
                plt.gca().add_patch(obstacle)
        
        # Plot goal position if provided
        if goal_pos:
            plt.plot(goal_pos[0], goal_pos[1], 'ro', markersize=12, label='Goal Position')
            # Plot success threshold circle around goal
            circle = plt.Circle((goal_pos[0], goal_pos[1]), self.success_threshold, 
                              color='g', fill=False, linestyle='--', alpha=0.7)
            plt.gca().add_patch(circle)
        
        # Plot trajectories
        if trajectory_normal:
            normal_x, normal_y = zip(*trajectory_normal)
            plt.plot(normal_x, normal_y, 'b-', linewidth=2, label=f'Normal (Success: {success_normal})', alpha=0.8)
            plt.plot(normal_x[0], normal_y[0], 'bo', markersize=8, label='Start')
            plt.plot(normal_x[-1], normal_y[-1], 'bs', markersize=8, label='End Normal')
        
        if trajectory_gradient:
            gradient_x, gradient_y = zip(*trajectory_gradient)
            plt.plot(gradient_x, gradient_y, 'r-', linewidth=2, label=f'Gradient λ=0.1 (Success: {success_gradient})', alpha=0.8)
            plt.plot(gradient_x[-1], gradient_y[-1], 'rs', markersize=8, label='End Gradient λ=0.1')
        
        if trajectory_gradient_lambda2:
            gradient_lambda2_x, gradient_lambda2_y = zip(*trajectory_gradient_lambda2)
            plt.plot(gradient_lambda2_x, gradient_lambda2_y, 'g-', linewidth=2, label=f'Gradient λ=0.2 (Success: {success_gradient_lambda2})', alpha=0.8)
            plt.plot(gradient_lambda2_x[-1], gradient_lambda2_y[-1], 'gs', markersize=8, label='End Gradient λ=0.2')
        
        plt.autoscale(False)
        plt.xlim(-1.1 - margin, 1.1 + margin)
        plt.ylim(-1.1 - margin, 1.1 + margin)
        
        # Set plot properties
        plt.grid(True)
        plt.axis('equal')
        plt.title(f'Episode {episode_num} - Trajectory Comparison (λ=0.1 vs λ=0.2)')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.legend()
        
        # Save plot
        plt.savefig(f'{self.plot_dir}/comparison_three_{episode_num}.jpg')
        plt.close()

    def test_gradient_computation(self):
        """Test if gradient computation works with the safety model"""
        print("Testing gradient computation...")
        try:
            dynamics = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # Example dynamics
            action = [0.25, 0.0]  # Example action
            gradients, safety_score = self.compute_safety_gradients(dynamics, action)
            print(f"Test gradients: {gradients}")
            if np.any(gradients != 0):
                print("Gradient computation test passed!")
            else:
                print("Warning: Gradients are zero, this might indicate an issue")
        except Exception as e:
            print(f"Gradient computation test failed: {e}")

    def save_timing_data(self, filename="timing_data.pkl"):
        """Save timing data to a file for later analysis"""
        timing_data = {
            'normal': {
                'prediction_times': self.prediction_times,
                'total_prediction_time': self.total_prediction_time,
                'avg_prediction_time': sum(self.prediction_times) / len(self.prediction_times) if self.prediction_times else 0,
                'min_prediction_time': min(self.prediction_times) if self.prediction_times else 0,
                'max_prediction_time': max(self.prediction_times) if self.prediction_times else 0,
            }
        }
        
        if self.eval_with_gradient:
            timing_data['gradient_lambda1'] = {
                'prediction_times': self.prediction_times_gradient,
                'total_prediction_time': self.total_prediction_time_gradient,
                'avg_prediction_time': sum(self.prediction_times_gradient) / len(self.prediction_times_gradient) if self.prediction_times_gradient else 0,
                'min_prediction_time': min(self.prediction_times_gradient) if self.prediction_times_gradient else 0,
                'max_prediction_time': max(self.prediction_times_gradient) if self.prediction_times_gradient else 0,
            }
            timing_data['gradient_lambda2'] = {
                'prediction_times': self.prediction_times_gradient_lambda2,
                'total_prediction_time': self.total_prediction_time_gradient_lambda2,
                'avg_prediction_time': sum(self.prediction_times_gradient_lambda2) / len(self.prediction_times_gradient_lambda2) if self.prediction_times_gradient_lambda2 else 0,
                'min_prediction_time': min(self.prediction_times_gradient_lambda2) if self.prediction_times_gradient_lambda2 else 0,
                'max_prediction_time': max(self.prediction_times_gradient_lambda2) if self.prediction_times_gradient_lambda2 else 0,
            }
        
        with open(filename, 'wb') as f:
            pickle.dump(timing_data, f)
        print(f"Timing data saved to {filename}")

    def run_evaluation(self, obstacles):
        print("Starting evaluation...")
        total_return = 0
        total_discounted_return = 0
        total_return_gradient = 0
        total_discounted_return_gradient = 0
        
        for episode in range(self.total_episodes):
            success, success_gradient, success_gradient_lambda2, episode_return, episode_discounted_return = self.evaluate_episode(episode, obstacles)
            if success:
                self.success_count += 1
            if success_gradient:
                self.success_count_gradient += 1
            if success_gradient_lambda2:
                self.success_count_gradient_lambda2 += 1
            if not (success==-1 and episode_return==-1 and episode_return==-1):
                total_return += episode_return
                total_discounted_return += episode_discounted_return
                
                # Calculate running averages for normal evaluation
                avg_return = total_return / (episode + 1)
                avg_discounted_return = total_discounted_return / (episode + 1)
                success_rate = (self.success_count / (episode + 1)) * 100
                
                print(f"\nEpisode {episode + 1}/{self.total_episodes}")
                print(f"Running Success Rate: {success_rate:.2f}%")
                print(f"Average Return: {avg_return:.2f}")
                print(f"Average Discounted Return: {avg_discounted_return:.2f}")
                
                # If using gradient evaluation, also show gradient metrics
                if self.eval_with_gradient:
                    # Calculate gradient metrics from stored lists
                    if len(self.episode_returns_gradient) > 0:
                        avg_return_gradient = sum(self.episode_returns_gradient) / len(self.episode_returns_gradient)
                        avg_discounted_return_gradient = sum(self.episode_discounted_returns_gradient) / len(self.episode_discounted_returns_gradient)
                        success_rate_gradient = (self.success_count_gradient / (episode + 1)) * 100
                        
                        print(f"Running Success Rate (Gradient λ=0.1): {success_rate_gradient:.2f}%")
                        print(f"Average Return (Gradient λ=0.1): {avg_return_gradient:.2f}")
                        print(f"Average Discounted Return (Gradient λ=0.1): {avg_discounted_return_gradient:.2f}")
                    
                    if len(self.episode_returns_gradient_lambda2) > 0:
                        avg_return_gradient_lambda2 = sum(self.episode_returns_gradient_lambda2) / len(self.episode_returns_gradient_lambda2)
                        avg_discounted_return_gradient_lambda2 = sum(self.episode_discounted_returns_gradient_lambda2) / len(self.episode_discounted_returns_gradient_lambda2)
                        success_rate_gradient_lambda2 = (self.success_count_gradient_lambda2 / (episode + 1)) * 100
                        
                        print(f"Running Success Rate (Gradient λ=0.2): {success_rate_gradient_lambda2:.2f}%")
                        print(f"Average Return (Gradient λ=0.2): {avg_return_gradient_lambda2:.2f}")
                        print(f"Average Discounted Return (Gradient λ=0.2): {avg_discounted_return_gradient_lambda2:.2f}")
            
        # Final evaluation summary
        final_success_rate = (self.success_count / self.total_episodes) * 100
        final_avg_return = total_return / self.total_episodes
        final_avg_discounted_return = total_discounted_return / self.total_episodes
        final_avg_episode_length = sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0
        
        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE!")
        print(f"{'='*60}")
        
        print(f"\nNORMAL SETTING:")
        print(f"Final Success Rate: {final_success_rate:.2f}% ({self.success_count}/{self.total_episodes})")
        print(f"Final Average Return: {final_avg_return:.2f}")
        print(f"Final Average Discounted Return: {final_avg_discounted_return:.2f}")
        print(f"Average Episode Length: {final_avg_episode_length:.2f}")
        
        # Calculate timing statistics for normal evaluation
        if self.prediction_times:
            avg_prediction_time = sum(self.prediction_times) / len(self.prediction_times)
            min_prediction_time = min(self.prediction_times)
            max_prediction_time = max(self.prediction_times)
            print(f"Average Prediction Time: {avg_prediction_time*1000:.2f} ms")
            print(f"Min Prediction Time: {min_prediction_time*1000:.2f} ms")
            print(f"Max Prediction Time: {max_prediction_time*1000:.2f} ms")
            print(f"Total Prediction Time: {self.total_prediction_time:.2f} s")
        
        if self.eval_with_gradient:
            # Calculate final gradient statistics for λ=0.1
            final_success_rate_gradient = (self.success_count_gradient / self.total_episodes) * 100
            final_avg_return_gradient = sum(self.episode_returns_gradient) / len(self.episode_returns_gradient) if self.episode_returns_gradient else 0
            final_avg_discounted_return_gradient = sum(self.episode_discounted_returns_gradient) / len(self.episode_discounted_returns_gradient) if self.episode_discounted_returns_gradient else 0
            final_avg_episode_length_gradient = sum(self.episode_lengths_gradient) / len(self.episode_lengths_gradient) if self.episode_lengths_gradient else 0
            
            # Calculate final gradient statistics for λ=0.2
            final_success_rate_gradient_lambda2 = (self.success_count_gradient_lambda2 / self.total_episodes) * 100
            final_avg_return_gradient_lambda2 = sum(self.episode_returns_gradient_lambda2) / len(self.episode_returns_gradient_lambda2) if self.episode_returns_gradient_lambda2 else 0
            final_avg_discounted_return_gradient_lambda2 = sum(self.episode_discounted_returns_gradient_lambda2) / len(self.episode_discounted_returns_gradient_lambda2) if self.episode_discounted_returns_gradient_lambda2 else 0
            final_avg_episode_length_gradient_lambda2 = sum(self.episode_lengths_gradient_lambda2) / len(self.episode_lengths_gradient_lambda2) if self.episode_lengths_gradient_lambda2 else 0
            
            print(f"\nWITH GRADIENT λ=0.1:")
            print(f"Final Success Rate: {final_success_rate_gradient:.2f}% ({self.success_count_gradient}/{self.total_episodes})")
            print(f"Final Average Return: {final_avg_return_gradient:.2f}")
            print(f"Final Average Discounted Return: {final_avg_discounted_return_gradient:.2f}")
            print(f"Average Episode Length: {final_avg_episode_length_gradient:.2f}")
            
            print(f"\nWITH GRADIENT λ=0.2:")
            print(f"Final Success Rate: {final_success_rate_gradient_lambda2:.2f}% ({self.success_count_gradient_lambda2}/{self.total_episodes})")
            print(f"Final Average Return: {final_avg_return_gradient_lambda2:.2f}")
            print(f"Final Average Discounted Return: {final_avg_discounted_return_gradient_lambda2:.2f}")
            print(f"Average Episode Length: {final_avg_episode_length_gradient_lambda2:.2f}")
            
            # Calculate timing statistics for gradient evaluation λ=0.1
            if self.prediction_times_gradient:
                avg_prediction_time_gradient = sum(self.prediction_times_gradient) / len(self.prediction_times_gradient)
                min_prediction_time_gradient = min(self.prediction_times_gradient)
                max_prediction_time_gradient = max(self.prediction_times_gradient)
                print(f"Average Prediction Time (λ=0.1): {avg_prediction_time_gradient*1000:.2f} ms")
                print(f"Min Prediction Time (λ=0.1): {min_prediction_time_gradient*1000:.2f} ms")
                print(f"Max Prediction Time (λ=0.1): {max_prediction_time_gradient*1000:.2f} ms")
                print(f"Total Prediction Time (λ=0.1): {self.total_prediction_time_gradient:.2f} s")
            
            # Calculate timing statistics for gradient evaluation λ=0.2
            if self.prediction_times_gradient_lambda2:
                avg_prediction_time_gradient_lambda2 = sum(self.prediction_times_gradient_lambda2) / len(self.prediction_times_gradient_lambda2)
                min_prediction_time_gradient_lambda2 = min(self.prediction_times_gradient_lambda2)
                max_prediction_time_gradient_lambda2 = max(self.prediction_times_gradient_lambda2)
                print(f"Average Prediction Time (λ=0.2): {avg_prediction_time_gradient_lambda2*1000:.2f} ms")
                print(f"Min Prediction Time (λ=0.2): {min_prediction_time_gradient_lambda2*1000:.2f} ms")
                print(f"Max Prediction Time (λ=0.2): {max_prediction_time_gradient_lambda2*1000:.2f} ms")
                print(f"Total Prediction Time (λ=0.2): {self.total_prediction_time_gradient_lambda2:.2f} s")
            
            # Print improvement comparison
            print(f"\nIMPROVEMENT COMPARISON:")
            print(f"\nλ=0.1 vs Normal:")
            success_improvement_lambda1 = final_success_rate_gradient - final_success_rate
            return_improvement_lambda1 = final_avg_return_gradient - final_avg_return
            discounted_return_improvement_lambda1 = final_avg_discounted_return_gradient - final_avg_discounted_return
            length_improvement_lambda1 = final_avg_episode_length_gradient - final_avg_episode_length
            
            print(f"Success Rate Improvement: {success_improvement_lambda1:+.2f}%")
            print(f"Average Return Improvement: {return_improvement_lambda1:+.2f}")
            print(f"Average Discounted Return Improvement: {discounted_return_improvement_lambda1:+.2f}")
            print(f"Average Episode Length Change: {length_improvement_lambda1:+.2f}")
            
            print(f"\nλ=0.2 vs Normal:")
            success_improvement_lambda2 = final_success_rate_gradient_lambda2 - final_success_rate
            return_improvement_lambda2 = final_avg_return_gradient_lambda2 - final_avg_return
            discounted_return_improvement_lambda2 = final_avg_discounted_return_gradient_lambda2 - final_avg_discounted_return
            length_improvement_lambda2 = final_avg_episode_length_gradient_lambda2 - final_avg_episode_length
            
            print(f"Success Rate Improvement: {success_improvement_lambda2:+.2f}%")
            print(f"Average Return Improvement: {return_improvement_lambda2:+.2f}")
            print(f"Average Discounted Return Improvement: {discounted_return_improvement_lambda2:+.2f}")
            print(f"Average Episode Length Change: {length_improvement_lambda2:+.2f}")
            
            print(f"\nλ=0.2 vs λ=0.1:")
            success_improvement_lambda2_vs_lambda1 = final_success_rate_gradient_lambda2 - final_success_rate_gradient
            return_improvement_lambda2_vs_lambda1 = final_avg_return_gradient_lambda2 - final_avg_return_gradient
            discounted_return_improvement_lambda2_vs_lambda1 = final_avg_discounted_return_gradient_lambda2 - final_avg_discounted_return_gradient
            length_improvement_lambda2_vs_lambda1 = final_avg_episode_length_gradient_lambda2 - final_avg_episode_length_gradient
            
            print(f"Success Rate Improvement: {success_improvement_lambda2_vs_lambda1:+.2f}%")
            print(f"Average Return Improvement: {return_improvement_lambda2_vs_lambda1:+.2f}")
            print(f"Average Discounted Return Improvement: {discounted_return_improvement_lambda2_vs_lambda1:+.2f}")
            print(f"Average Episode Length Change: {length_improvement_lambda2_vs_lambda1:+.2f}")
            
            # Print timing comparison
            if self.prediction_times and self.prediction_times_gradient and self.prediction_times_gradient_lambda2:
                avg_prediction_time = sum(self.prediction_times) / len(self.prediction_times)
                avg_prediction_time_gradient = sum(self.prediction_times_gradient) / len(self.prediction_times_gradient)
                avg_prediction_time_gradient_lambda2 = sum(self.prediction_times_gradient_lambda2) / len(self.prediction_times_gradient_lambda2)
                
                timing_overhead_lambda1 = avg_prediction_time_gradient - avg_prediction_time
                timing_overhead_lambda1_percent = (timing_overhead_lambda1 / avg_prediction_time) * 100 if avg_prediction_time > 0 else 0
                print(f"Prediction Time Overhead (λ=0.1): {timing_overhead_lambda1*1000:+.2f} ms ({timing_overhead_lambda1_percent:+.1f}%)")
                
                timing_overhead_lambda2 = avg_prediction_time_gradient_lambda2 - avg_prediction_time
                timing_overhead_lambda2_percent = (timing_overhead_lambda2 / avg_prediction_time) * 100 if avg_prediction_time > 0 else 0
                print(f"Prediction Time Overhead (λ=0.2): {timing_overhead_lambda2*1000:+.2f} ms ({timing_overhead_lambda2_percent:+.1f}%)")
                
                timing_overhead_lambda2_vs_lambda1 = avg_prediction_time_gradient_lambda2 - avg_prediction_time_gradient
                timing_overhead_lambda2_vs_lambda1_percent = (timing_overhead_lambda2_vs_lambda1 / avg_prediction_time_gradient) * 100 if avg_prediction_time_gradient > 0 else 0
                print(f"Prediction Time Overhead (λ=0.2 vs λ=0.1): {timing_overhead_lambda2_vs_lambda1*1000:+.2f} ms ({timing_overhead_lambda2_vs_lambda1_percent:+.1f}%)")
        
        print(f"\n{'='*60}")
        
        # Save timing data for later analysis
        self.save_timing_data()

if __name__ == "__main__":
    try:
        obstacles = True
        eval_with_gradient = True  # Set to True to enable gradient-based safety optimization
        safety_model_path = "/home/othman/turtlebotExp/safety_model_best.pth"  # Path to trained safety model
        
        # Initialize the evaluator with JEPA model
        evaluator = ModelEvaluator(
            model_path="/home/othman/turtlebotExp/models/expert_obstacles/model_9900.pth",
            use_images=False,  # Set to True since we're using JEPA
            # jepa_model_path="/home/othman/merlin/2024_12_29/ijepa/models"  # Path to your trained JEPA model
            eval_with_gradient=eval_with_gradient,
            safety_model_path=safety_model_path
        )
        evaluator.run_evaluation(obstacles)
    except rospy.ROSInterruptException:
        pass 