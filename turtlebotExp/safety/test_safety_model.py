#!/usr/bin/env python

import rospy
import torch
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import Twist, Pose
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from std_srvs.srv import Empty
import math
import random
import os
from train_safety_model import SafetyPredictor

# Global variables for ROS callbacks
odom = None
x_odom = 0.0
y_odom = 0.0
theta = 0.0
linear_x = 0.0
angular_z = 0.0

def odom_callback(msg):
    global odom, x_odom, y_odom, theta, linear_x, angular_z
    x_odom = msg.pose.pose.position.x
    y_odom = msg.pose.pose.position.y
    linear_x = msg.twist.twist.linear.x
    angular_z = msg.twist.twist.angular.z
    
    orientation_q = msg.pose.pose.orientation
    _, _, theta = euler_from_quaternion([0, 0, orientation_q.z, orientation_q.w])
    odom = msg

class SafetyModelTester:
    def __init__(self, model_path='safety_model_best.pth'):
        # Initialize ROS node
        rospy.init_node("safety_model_tester")
        
        # Setup ROS subscribers and services
        rospy.Subscriber("/odom", Odometry, odom_callback)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.reset_service = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.set_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        # Define obstacle positions (same as in data collection)
        self.cylindrical_obstacles = [
            (-1.1, -1.1), (-1.1, 0.0), (-1.1, 1.1),  # Left column
            (0.0, -1.1), (0.0, 0.0), (0.0, 1.1),     # Middle column
            (1.1, -1.1), (1.1, 0.0), (1.1, 1.1)      # Right column
        ]
        self.cylindrical_radius = 0.15  # meters
        
        self.hexagonal_obstacles = [
            (3.5, 0.0, 0.8),      # Head
            (1.8, 2.7, 0.55),     # Left hand
            (1.8, -2.7, 0.55),    # Right hand
            (-1.8, 2.7, 0.55),    # Left foot
            (-1.8, -2.7, 0.55)    # Right foot
        ]
        self.hexagonal_radius = 0.4  # Approximate radius for collision checking
        
        # Load trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SafetyPredictor(input_dim=8, hidden_dims=[128, 64, 32]).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found!")
            return
        
        self.model.eval()
        
        # Create output directory
        self.output_dir = "safety_test_results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def is_position_valid(self, x, y):
        """Check if a position is valid (not inside or too close to obstacles)"""
        # Check cylindrical obstacles
        for obs_x, obs_y in self.cylindrical_obstacles:
            distance = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            if distance < (self.cylindrical_radius + 0.1):
                return False
                
        # Check hexagonal obstacles
        for obs_x, obs_y, scale in self.hexagonal_obstacles:
            distance = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            if distance < (self.hexagonal_radius * scale + 0.2):
                return False
                
        # Check boundaries
        if abs(x) > 1.4 or abs(y) > 1.9:
            return False
                
        return True

    def set_random_position(self):
        """Set robot to a random valid position"""
        max_attempts = 100
        for attempt in range(max_attempts):
            initial_x = random.uniform(-1.4, 1.4)
            initial_y = random.uniform(-1.9, 1.9)
            initial_theta = random.uniform(-math.pi, math.pi)
            
            if self.is_position_valid(initial_x, initial_y):
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
                    rospy.sleep(1.0)  # Wait for position to be set
                    print(f"Set robot to position: ({initial_x:.2f}, {initial_y:.2f}, {initial_theta:.2f})")
                    return True
                except rospy.ServiceException as e:
                    rospy.logerr("Failed to set position: %s", str(e))
        
        print("Failed to find valid position after 100 attempts")
        return False

    def predict_safety(self, dynamics, action):
        """Predict safety score for given dynamics and action"""
        with torch.no_grad():
            # Prepare input
            input_data = torch.cat([
                torch.tensor(dynamics, dtype=torch.float32),
                torch.tensor(action, dtype=torch.float32)
            ]).unsqueeze(0).to(self.device)
            
            # Get prediction
            safety_score = self.model(input_data).item()
            return safety_score

    def compute_safety_gradients(self, dynamics, action):
        """Compute gradients of safety with respect to action"""
        # Prepare input
        input_data = torch.cat([
            torch.tensor(dynamics, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32)
        ]).unsqueeze(0).to(self.device)
        
        # Enable gradient computation
        input_data.requires_grad_(True)
        
        # Forward pass
        safety_score = self.model(input_data)
        
        # Backward pass to compute gradients
        safety_score.backward()
        
        # Extract gradients with respect to action (last 2 elements)
        action_gradients = input_data.grad[0, -2:].cpu().numpy()
        
        return action_gradients

    def test_position(self, test_num):
        """Test safety predictions at current position for various actions"""
        print(f"\nTesting position {test_num}...")
        
        # Wait for odometry to update
        rospy.sleep(2.0)
        
        # Get current robot state
        dynamics = [x_odom, y_odom, linear_x, angular_z, math.sin(theta), math.cos(theta)]
        print(f"Robot state: pos=({x_odom:.2f}, {y_odom:.2f}), theta={theta:.2f}")
        
        # Test different action combinations
        linear_vels = np.linspace(0, 0.5, 11)  # 0 to 0.5 m/s
        angular_vels = np.linspace(-1.0, 1.0, 21)  # -1 to 1 rad/s
        
        safety_grid = np.zeros((len(linear_vels), len(angular_vels)))
        linear_gradient_grid = np.zeros((len(linear_vels), len(angular_vels)))
        angular_gradient_grid = np.zeros((len(linear_vels), len(angular_vels)))
        gradient_magnitude_grid = np.zeros((len(linear_vels), len(angular_vels)))
        
        print("Computing safety predictions and gradients...")
        for i, linear_vel in enumerate(linear_vels):
            for j, angular_vel in enumerate(angular_vels):
                action = [linear_vel, angular_vel]
                safety_score = self.predict_safety(dynamics, action)
                safety_grid[i, j] = safety_score
                
                # Compute gradients
                gradients = self.compute_safety_gradients(dynamics, action)
                linear_gradient_grid[i, j] = gradients[0]  # dS/d(linear_vel)
                angular_gradient_grid[i, j] = gradients[1]  # dS/d(angular_vel)
                gradient_magnitude_grid[i, j] = np.sqrt(gradients[0]**2 + gradients[1]**2)
        
        # Create visualization
        self.plot_safety_heatmap(safety_grid, linear_vels, angular_vels, test_num)
        self.plot_gradient_analysis(linear_gradient_grid, angular_gradient_grid, 
                                  gradient_magnitude_grid, linear_vels, angular_vels, test_num)
        
        return safety_grid, linear_gradient_grid, angular_gradient_grid, gradient_magnitude_grid

    def plot_safety_heatmap(self, safety_grid, linear_vels, angular_vels, test_num):
        """Plot safety predictions as a heatmap"""
        plt.figure(figsize=(15, 10))
        
        # Main safety heatmap
        plt.subplot(2, 3, 1)
        im = plt.imshow(safety_grid, cmap='RdYlGn', aspect='auto', 
                       extent=[angular_vels[0], angular_vels[-1], linear_vels[0], linear_vels[-1]])
        plt.colorbar(im, label='Safety Score')
        plt.xlabel('Angular Velocity (rad/s)')
        plt.ylabel('Linear Velocity (m/s)')
        plt.title(f'Safety Predictions - Test {test_num}')
        plt.grid(True, alpha=0.3)
        
        # Environment plot with robot position
        plt.subplot(2, 3, 2)
        plt.xlim(-1.6, 1.6)
        plt.ylim(-2.1, 2.1)
        
        # Plot obstacles
        for obs_x, obs_y in self.cylindrical_obstacles:
            circle = plt.Circle((obs_x, obs_y), self.cylindrical_radius, 
                              color='gray', alpha=0.5)
            plt.gca().add_patch(circle)
            
        for obs_x, obs_y, scale in self.hexagonal_obstacles:
            circle = plt.Circle((obs_x, obs_y), self.hexagonal_radius * scale, 
                              color='darkgray', alpha=0.5)
            plt.gca().add_patch(circle)
        
        # Plot robot position and heading
        plt.plot(x_odom, y_odom, 'bo', markersize=10, label='Robot')
        arrow_length = 0.2
        dx = arrow_length * math.cos(theta)
        dy = arrow_length * math.sin(theta)
        plt.arrow(x_odom, y_odom, dx, dy, 
                 head_width=0.08, head_length=0.08, fc='b', ec='b')
        
        plt.grid(True)
        plt.axis('equal')
        plt.title(f'Robot Position - Test {test_num}')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        
        # Safety vs Linear Velocity (averaged over angular)
        plt.subplot(2, 3, 3)
        avg_safety_linear = np.mean(safety_grid, axis=1)
        plt.plot(linear_vels, avg_safety_linear, 'b-', linewidth=2)
        plt.xlabel('Linear Velocity (m/s)')
        plt.ylabel('Average Safety Score')
        plt.title('Safety vs Linear Velocity')
        plt.grid(True)
        
        # Safety vs Angular Velocity (averaged over linear)
        plt.subplot(2, 3, 4)
        avg_safety_angular = np.mean(safety_grid, axis=0)
        plt.plot(angular_vels, avg_safety_angular, 'r-', linewidth=2)
        plt.xlabel('Angular Velocity (rad/s)')
        plt.ylabel('Average Safety Score')
        plt.title('Safety vs Angular Velocity')
        plt.grid(True)
        
        # 3D surface plot
        ax = plt.subplot(2, 3, 5, projection='3d')
        X, Y = np.meshgrid(angular_vels, linear_vels)
        surf = ax.plot_surface(X, Y, safety_grid, cmap='RdYlGn', alpha=0.8)
        ax.set_xlabel('Angular Velocity (rad/s)')
        ax.set_ylabel('Linear Velocity (m/s)')
        ax.set_zlabel('Safety Score')
        ax.set_title('3D Safety Surface')
        
        # Statistics
        plt.subplot(2, 3, 6)
        plt.text(0.1, 0.9, f'Test {test_num} Statistics:', fontsize=12, fontweight='bold')
        plt.text(0.1, 0.8, f'Mean Safety: {np.mean(safety_grid):.3f}', fontsize=10)
        plt.text(0.1, 0.7, f'Min Safety: {np.min(safety_grid):.3f}', fontsize=10)
        plt.text(0.1, 0.6, f'Max Safety: {np.max(safety_grid):.3f}', fontsize=10)
        plt.text(0.1, 0.5, f'Std Safety: {np.std(safety_grid):.3f}', fontsize=10)
        plt.text(0.1, 0.4, f'Robot Pos: ({x_odom:.2f}, {y_odom:.2f})', fontsize=10)
        plt.text(0.1, 0.3, f'Robot Theta: {theta:.2f}', fontsize=10)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/safety_test_{test_num}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved safety visualization to {self.output_dir}/safety_test_{test_num}.png")

    def plot_gradient_analysis(self, linear_gradients, angular_gradients, gradient_magnitudes, 
                             linear_vels, angular_vels, test_num):
        """Plot gradient analysis of safety with respect to actions"""
        plt.figure(figsize=(20, 12))
        
        # Linear velocity gradient heatmap
        plt.subplot(3, 4, 1)
        im1 = plt.imshow(linear_gradients, cmap='RdBu_r', aspect='auto',
                        extent=[angular_vels[0], angular_vels[-1], linear_vels[0], linear_vels[-1]])
        plt.colorbar(im1, label='∂S/∂(Linear Vel)')
        plt.xlabel('Angular Velocity (rad/s)')
        plt.ylabel('Linear Velocity (m/s)')
        plt.title(f'∂S/∂(Linear Vel) - Test {test_num}')
        plt.grid(True, alpha=0.3)
        
        # Angular velocity gradient heatmap
        plt.subplot(3, 4, 2)
        im2 = plt.imshow(angular_gradients, cmap='RdBu_r', aspect='auto',
                        extent=[angular_vels[0], angular_vels[-1], linear_vels[0], linear_vels[-1]])
        plt.colorbar(im2, label='∂S/∂(Angular Vel)')
        plt.xlabel('Angular Velocity (rad/s)')
        plt.ylabel('Linear Velocity (m/s)')
        plt.title(f'∂S/∂(Angular Vel) - Test {test_num}')
        plt.grid(True, alpha=0.3)
        
        # Gradient magnitude heatmap
        plt.subplot(3, 4, 3)
        im3 = plt.imshow(gradient_magnitudes, cmap='viridis', aspect='auto',
                        extent=[angular_vels[0], angular_vels[-1], linear_vels[0], linear_vels[-1]])
        plt.colorbar(im3, label='|∇S|')
        plt.xlabel('Angular Velocity (rad/s)')
        plt.ylabel('Linear Velocity (m/s)')
        plt.title(f'Gradient Magnitude |∇S| - Test {test_num}')
        plt.grid(True, alpha=0.3)
        
        # Gradient vector field (sampled)
        plt.subplot(3, 4, 4)
        # Sample every 3rd point for clarity
        sample_step = 3
        X, Y = np.meshgrid(angular_vels[::sample_step], linear_vels[::sample_step])
        U = angular_gradients[::sample_step, ::sample_step]
        V = linear_gradients[::sample_step, ::sample_step]
        
        # Normalize vectors for visualization
        norm = np.sqrt(U**2 + V**2)
        U_norm = U / (norm + 1e-8)
        V_norm = V / (norm + 1e-8)
        
        plt.quiver(X, Y, U_norm, V_norm, norm, cmap='viridis', alpha=0.7)
        plt.colorbar(label='Gradient Magnitude')
        plt.xlabel('Angular Velocity (rad/s)')
        plt.ylabel('Linear Velocity (m/s)')
        plt.title(f'Gradient Vector Field - Test {test_num}')
        plt.grid(True, alpha=0.3)
        
        # Average gradients vs linear velocity
        plt.subplot(3, 4, 5)
        avg_linear_grad = np.mean(linear_gradients, axis=1)
        plt.plot(linear_vels, avg_linear_grad, 'b-', linewidth=2)
        plt.xlabel('Linear Velocity (m/s)')
        plt.ylabel('Avg ∂S/∂(Linear Vel)')
        plt.title('Avg ∂S/∂(Linear Vel) vs Linear Velocity')
        plt.grid(True)
        
        # Average gradients vs angular velocity
        plt.subplot(3, 4, 6)
        avg_angular_grad = np.mean(angular_gradients, axis=0)
        plt.plot(angular_vels, avg_angular_grad, 'r-', linewidth=2)
        plt.xlabel('Angular Velocity (rad/s)')
        plt.ylabel('Avg ∂S/∂(Angular Vel)')
        plt.title('Avg ∂S/∂(Angular Vel) vs Angular Velocity')
        plt.grid(True)
        
        # Gradient magnitude vs linear velocity
        plt.subplot(3, 4, 7)
        avg_grad_mag_linear = np.mean(gradient_magnitudes, axis=1)
        plt.plot(linear_vels, avg_grad_mag_linear, 'g-', linewidth=2)
        plt.xlabel('Linear Velocity (m/s)')
        plt.ylabel('Avg |∇S|')
        plt.title('Avg Gradient Magnitude vs Linear Velocity')
        plt.grid(True)
        
        # Gradient magnitude vs angular velocity
        plt.subplot(3, 4, 8)
        avg_grad_mag_angular = np.mean(gradient_magnitudes, axis=0)
        plt.plot(angular_vels, avg_grad_mag_angular, 'm-', linewidth=2)
        plt.xlabel('Angular Velocity (rad/s)')
        plt.ylabel('Avg |∇S|')
        plt.title('Avg Gradient Magnitude vs Angular Velocity')
        plt.grid(True)
        
        # 3D surface plots
        # Linear gradient surface
        ax1 = plt.subplot(3, 4, 9, projection='3d')
        X, Y = np.meshgrid(angular_vels, linear_vels)
        surf1 = ax1.plot_surface(X, Y, linear_gradients, cmap='RdBu_r', alpha=0.8)
        ax1.set_xlabel('Angular Velocity (rad/s)')
        ax1.set_ylabel('Linear Velocity (m/s)')
        ax1.set_zlabel('∂S/∂(Linear Vel)')
        ax1.set_title('∂S/∂(Linear Vel) Surface')
        
        # Angular gradient surface
        ax2 = plt.subplot(3, 4, 10, projection='3d')
        surf2 = ax2.plot_surface(X, Y, angular_gradients, cmap='RdBu_r', alpha=0.8)
        ax2.set_xlabel('Angular Velocity (rad/s)')
        ax2.set_ylabel('Linear Velocity (m/s)')
        ax2.set_zlabel('∂S/∂(Angular Vel)')
        ax2.set_title('∂S/∂(Angular Vel) Surface')
        
        # Gradient magnitude surface
        ax3 = plt.subplot(3, 4, 11, projection='3d')
        surf3 = ax3.plot_surface(X, Y, gradient_magnitudes, cmap='viridis', alpha=0.8)
        ax3.set_xlabel('Angular Velocity (rad/s)')
        ax3.set_ylabel('Linear Velocity (m/s)')
        ax3.set_zlabel('|∇S|')
        ax3.set_title('Gradient Magnitude Surface')
        
        # Statistics
        plt.subplot(3, 4, 12)
        plt.text(0.1, 0.9, f'Gradient Statistics - Test {test_num}:', fontsize=10, fontweight='bold')
        plt.text(0.1, 0.8, f'Max |∂S/∂(Linear)|: {np.max(np.abs(linear_gradients)):.3f}', fontsize=9)
        plt.text(0.1, 0.7, f'Max |∂S/∂(Angular)|: {np.max(np.abs(angular_gradients)):.3f}', fontsize=9)
        plt.text(0.1, 0.6, f'Max |∇S|: {np.max(gradient_magnitudes):.3f}', fontsize=9)
        plt.text(0.1, 0.5, f'Mean |∇S|: {np.mean(gradient_magnitudes):.3f}', fontsize=9)
        plt.text(0.1, 0.4, f'Std |∇S|: {np.std(gradient_magnitudes):.3f}', fontsize=9)
        plt.text(0.1, 0.3, f'Robot Pos: ({x_odom:.2f}, {y_odom:.2f})', fontsize=9)
        plt.text(0.1, 0.2, f'Robot Theta: {theta:.2f}', fontsize=9)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/gradient_analysis_{test_num}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved gradient analysis to {self.output_dir}/gradient_analysis_{test_num}.png")

    def run_tests(self, num_tests=5):
        """Run multiple safety tests at different positions"""
        print(f"Starting safety model testing with {num_tests} random positions...")
        
        for test_num in range(1, num_tests + 1):
            print(f"\n{'='*50}")
            print(f"TEST {test_num}/{num_tests}")
            print(f"{'='*50}")
            
            # Set random position
            if not self.set_random_position():
                print("Skipping this test due to position setting failure")
                continue
            
            # Test safety predictions
            safety_grid, linear_gradient_grid, angular_gradient_grid, gradient_magnitude_grid = self.test_position(test_num)
            
            # Wait between tests
            rospy.sleep(2.0)
        
        print(f"\nTesting complete! Results saved in {self.output_dir}/")

def main():
    try:
        tester = SafetyModelTester()
        tester.run_tests(num_tests=5)
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main() 