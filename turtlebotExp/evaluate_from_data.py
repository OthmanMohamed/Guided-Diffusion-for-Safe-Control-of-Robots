#!/usr/bin/env python

import rospy
import torch
import numpy as np
from geometry_msgs.msg import Twist, Pose, PoseStamped
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from std_srvs.srv import Empty
import math
import random
import matplotlib.pyplot as plt
import os
from policy import Policy
import pickle
import argparse

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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

class DataModelEvaluator:
    def __init__(self, model_path, data_path, num_trajectories=100):
        # Initialize ROS node
        rospy.init_node("data_model_evaluator")
        
        # Setup ROS publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        rospy.Subscriber("/odom", Odometry, odom_callback)
        
        # Setup ROS services
        self.reset_service = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.set_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        # Load the trained model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Policy(images=False).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Load evaluation data
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # Get trajectories
        self.trajectories = []
        for i in range(min(num_trajectories, len(self.data['o']))):
            traj = {
                'dynamics': self.data['o'][i],  # List of dynamics states
                'actions': self.data['u'][i],   # List of actions
                'goals': self.data['g'][i],     # List of goals (same goal repeated)
                'achieved_goals': self.data['ag'][i]  # List of achieved goals
            }
            self.trajectories.append(traj)
        
        # Evaluation metrics
        self.success_count = 0
        self.total_episodes = len(self.trajectories)
        self.success_threshold = 0.2  # meters
        
        # Create directory for trajectory plots if it doesn't exist
        self.plot_dir = "data_trajectory_plots"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

    def set_robot_state(self, dynamics):
        """Set robot state from dynamics data"""
        state_msg = SetModelStateRequest()
        state_msg.model_state.model_name = 'turtlebot3_burger'
        state_msg.model_state.pose.position.x = dynamics[0]
        state_msg.model_state.pose.position.y = dynamics[1]
        quat = quaternion_from_euler(0, 0, math.atan2(dynamics[4], dynamics[5]))
        state_msg.model_state.pose.orientation.x = quat[0]
        state_msg.model_state.pose.orientation.y = quat[1]
        state_msg.model_state.pose.orientation.z = quat[2]
        state_msg.model_state.pose.orientation.w = quat[3]
        state_msg.model_state.reference_frame = 'world'

        try:
            self.set_model_state_service(state_msg)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to set robot state: %s", str(e))

    def get_observation(self):
        """Get current observation matching training data format"""
        obs = torch.tensor([
            x_odom, y_odom,  # Current position
            linear_x, angular_z,  # Current velocities
            math.sin(theta), math.cos(theta)  # Current orientation
        ], dtype=torch.float32).to(self.device)
        return obs.unsqueeze(0)

    def plot_trajectory(self, trajectory, start_pos, goal_pos, success, episode_num):
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
        
        plt.autoscale(False)
        plt.xlim(-1.6, 1.6)
        plt.ylim(-2.1, 2.1)
        
        # Set plot properties
        plt.grid(True)
        plt.axis('equal')
        plt.title(f'Episode {episode_num} - {"Success" if success else "Failure"}')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.legend()
        
        # Save plot
        status = "success" if success else "fail"
        plt.savefig(f'{self.plot_dir}/{status}_{episode_num}.png')
        plt.close()

    def evaluate_episode(self, episode_idx):
        # Get trajectory data
        traj = self.trajectories[episode_idx]
        dynamics = traj['dynamics'][0]  # Initial state
        goal = traj['goals'][0]  # Goal is the same throughout trajectory
        horizon = len(traj['dynamics'])  # Total steps in trajectory
        
        # Reset environment and set initial state
        self.reset_service()
        self.set_robot_state(dynamics)
        rospy.sleep(2)  # Wait for reset to complete
        
        # Store start position
        start_pos = (dynamics[0], dynamics[1])
        goal_pos = (goal[0], goal[1])
        
        # Initialize episode variables
        step = 0
        success = False
        trajectory = [(dynamics[0], dynamics[1])]  # Store trajectory points
        
        while step < 50 and not rospy.is_shutdown():
            # Get current observation
            obs = self.get_observation()
            
            # Get model prediction
            with torch.no_grad():
                # Calculate remaining steps
                remaining_steps = max(horizon - step, 1)
                t = torch.tensor([10], dtype=torch.float32).to(self.device).unsqueeze(0)
                g = torch.tensor(goal, dtype=torch.float32).to(self.device).unsqueeze(0)
                mu, sigma = self.model(obs, g, t)
                action = mu  # Use mean action
            
            # Apply action
            vel_msg = Twist()
            vel_msg.linear.x = float(action[0, 0])
            vel_msg.angular.z = float(action[0, 1])
            self.cmd_vel_pub.publish(vel_msg)
            
            # Store trajectory point
            trajectory.append((x_odom, y_odom))
            
            # Check for success
            distance = math.sqrt((x_odom - goal[0])**2 + (y_odom - goal[1])**2)
            if distance < self.success_threshold:
                success = True
                break
            
            step += 1
            rospy.sleep(0.5)  # Control rate
        
        # Plot trajectory
        self.plot_trajectory(trajectory, start_pos, goal_pos, success, episode_idx)
        
        return success

    def run_evaluation(self):
        print(f"Starting evaluation of {self.total_episodes} trajectories...")
        
        for episode in range(self.total_episodes):
            success = self.evaluate_episode(episode)
            if success:
                self.success_count += 1
            
            print(f"Episode {episode + 1}/{self.total_episodes} - {'Success' if success else 'Failure'}")
        
        success_rate = (self.success_count / self.total_episodes) * 100
        print(f"\nEvaluation complete!")
        print(f"Success rate: {success_rate:.2f}% ({self.success_count}/{self.total_episodes})")

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model using training data')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the training data pickle file')
    parser.add_argument('--num_trajectories', type=int, default=100,
                      help='Number of trajectories to evaluate')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluator = DataModelEvaluator(args.model_path, args.data_path, args.num_trajectories)
    evaluator.run_evaluation() 