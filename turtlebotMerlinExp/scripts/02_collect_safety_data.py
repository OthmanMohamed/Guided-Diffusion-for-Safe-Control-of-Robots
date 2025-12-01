#!/usr/bin/env python3
"""
Script 02: Collect Safety Data

This script collects safety-labeled data for training the safety model.
It runs episodes and labels actions as safe (1.0) or unsafe (0.0) based on
whether they lead to collisions. Steps before collisions are also labeled as unsafe.

Note: This script requires ROS and Gazebo to be running with the TurtleBot3 simulation.
"""

import rospy
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import Twist, Pose
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from std_srvs.srv import Empty
from sensor_msgs.msg import Imu
import math
import random
import os
import pickle

# ==============================================================================
# GLOBAL STATE VARIABLES
# ==============================================================================
# These store the current state of the robot from ROS callbacks
odom = None
x_odom = 0.0
y_odom = 0.0
theta = 0.0
linear_x = 0.0
angular_z = 0.0
collision_detected = False
last_linear_accel = None
last_collision_time = 0

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================
# Collision detection parameters
COLLISION_ACCEL_THRESHOLD = 15.0      # m/s² - acceleration change threshold
COLLISION_COOLDOWN_TIME = 0.5         # seconds - minimum time between detections
MIN_MOVEMENT_SPEED = 0.1              # m/s - minimum speed for collision detection

# Workspace boundaries
WORKSPACE_X_MIN = -1.4                # meters
WORKSPACE_X_MAX = 1.4                 # meters  
WORKSPACE_Y_MIN = -1.9                # meters
WORKSPACE_Y_MAX = 1.9                 # meters

# Robot starting position range
START_X_MIN = -1.4                     # meters
START_X_MAX = 1.4                      # meters
START_Y_MIN = -1.9                     # meters
START_Y_MAX = 1.9                      # meters

# Safety scoring parameters
SAFETY_HORIZON = 5                     # Number of steps for safety prediction
OBSTACLE_MARGIN = 0.1                  # meters - safety margin around obstacles
HEXAGONAL_MARGIN = 0.2                 # meters - safety margin for hexagonal obstacles

# Velocity limits
MAX_LINEAR_VELOCITY = 0.5              # m/s
MAX_ANGULAR_VELOCITY = 1.0             # rad/s

# ==============================================================================
# ROS CALLBACK FUNCTIONS
# ==============================================================================

def odom_callback(msg):
    """Callback to store latest odometry data (position, velocity, orientation)"""
    global odom, x_odom, y_odom, theta, linear_x, angular_z
    x_odom = msg.pose.pose.position.x
    y_odom = msg.pose.pose.position.y
    linear_x = msg.twist.twist.linear.x
    angular_z = msg.twist.twist.angular.z
    
    orientation_q = msg.pose.pose.orientation
    _, _, theta = euler_from_quaternion([0, 0, orientation_q.z, orientation_q.w])
    odom = msg

def imu_callback(data):
    """Callback to detect collisions using IMU acceleration data
    
    Detects collisions by monitoring sudden changes in acceleration.
    Applies cooldown period and movement check to reduce false positives.
    """
    global collision_detected, last_linear_accel, last_collision_time
    current_accel = data.linear_acceleration
    current_time = rospy.get_time()
    
    if last_linear_accel is not None:
        # Calculate the change in acceleration
        accel_change = math.sqrt(
            (current_accel.x - last_linear_accel.x)**2 +
            (current_accel.y - last_linear_accel.y)**2 +
            (current_accel.z - last_linear_accel.z)**2
        )
        
        # Only consider it a collision if:
        # 1. The acceleration change is above threshold
        # 2. It's been at least cooldown time since the last collision
        # 3. The robot is moving (to avoid false positives when stationary)
        if (accel_change > COLLISION_ACCEL_THRESHOLD and 
            current_time - last_collision_time > COLLISION_COOLDOWN_TIME and
            abs(linear_x) > MIN_MOVEMENT_SPEED):
            print(f"Collision detected! Acceleration change: {accel_change:.2f} m/s²")
            collision_detected = True
            last_collision_time = current_time
    
    last_linear_accel = current_accel

# ==============================================================================
# OBSTACLE DEFINITIONS
# ==============================================================================

CYLINDRICAL_OBSTACLES = [
    (-1.1, -1.1), (-1.1, 0.0), (-1.1, 1.1),  # Left column
    (0.0, -1.1), (0.0, 0.0), (0.0, 1.1),     # Middle column
    (1.1, -1.1), (1.1, 0.0), (1.1, 1.1)      # Right column
]
CYLINDRICAL_RADIUS = 0.15  # meters

HEXAGONAL_OBSTACLES = [
    (3.5, 0.0, 0.8),      # Head
    (1.8, 2.7, 0.55),     # Left hand
    (1.8, -2.7, 0.55),    # Right hand
    (-1.8, 2.7, 0.55),    # Left foot
    (-1.8, -2.7, 0.55)    # Right foot
]
HEXAGONAL_RADIUS = 0.4  # Approximate radius for collision checking


# ==============================================================================
# MAIN DATA COLLECTION CLASS
# ==============================================================================

class SafetyDataCollector:
    """Collects trajectory data with safety labels for training safety models
    
    Generates random trajectories until collisions are detected, then labels all
    trajectory points based on their distance to the collision point using
    binary safety scoring (0 = unsafe, 1 = safe).
    """
    def __init__(self):
        """Initialize the safety data collector"""
        # Initialize ROS node
        rospy.init_node("safety_data_collector", anonymous=True)
        
        # Setup ROS subscribers and services
        rospy.Subscriber("/odom", Odometry, odom_callback)
        rospy.Subscriber("/imu", Imu, imu_callback)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.reset_service = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.set_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        # Use global obstacle definitions
        self.cylindrical_obstacles = CYLINDRICAL_OBSTACLES
        self.cylindrical_radius = CYLINDRICAL_RADIUS
        
        self.hexagonal_obstacles = HEXAGONAL_OBSTACLES
        self.hexagonal_radius = HEXAGONAL_RADIUS
        
        # Create directories for data and plots
        self.data_dir = "safety_data"
        self.plot_dir = "safety_plots_val"
        for directory in [self.data_dir, self.plot_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Data collection parameters
        self.max_steps = 100
        self.dt = 0.1  # time step for prediction
        
        # Collected data
        self.collected_data = []

    def set_random_position(self):
        """Set robot to a random position within the workspace bounds"""
        initial_x = random.uniform(START_X_MIN, START_X_MAX)
        initial_y = random.uniform(START_Y_MIN, START_Y_MAX)
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
            rospy.sleep(1.0)  # Wait for position to be set
        except rospy.ServiceException as e:
            rospy.logerr("Failed to set position: %s", str(e))

    def is_position_valid(self, x, y):
        """Check if a position is valid (not inside or too close to obstacles)
        
        Args:
            x, y: Position coordinates to check
            
        Returns:
            bool: True if position is valid, False if in collision
        """
        # Check cylindrical obstacles
        for obs_x, obs_y in self.cylindrical_obstacles:
            distance = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            if distance < (self.cylindrical_radius + OBSTACLE_MARGIN):
                return False
                
        # Check hexagonal obstacles
        for obs_x, obs_y, scale in self.hexagonal_obstacles:
            distance = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            if distance < (self.hexagonal_radius * scale + HEXAGONAL_MARGIN):
                return False
                
        # Check workspace boundaries
        if abs(x) > abs(WORKSPACE_X_MAX) or abs(y) > abs(WORKSPACE_Y_MAX):
            return False
                
        return True

    def calculate_safety_score(self, distance_to_collision):
        """Calculate safety score based on distance to collision point
        
        Binary safety scoring:
        - 0 (Unsafe): Robot will collide within SAFETY_HORIZON steps
        - 1 (Safe): Robot is more than SAFETY_HORIZON steps away from collision
        
        Args:
            distance_to_collision: Number of steps until collision
            
        Returns:
            float: Safety score (0.0 for unsafe, 1.0 for safe)
        """
        if distance_to_collision <= SAFETY_HORIZON:
            return 0.0  # Unsafe - will collide within horizon
        else:
            return 1.0  # Safe - more than horizon steps away from collision

    def collect_trajectory(self):
        """Collect one trajectory until collision or max steps reached
        
        Generates random actions and collects state-action pairs until
        a collision is detected or max steps is reached. After collision,
        all trajectory points are labeled with safety scores.
        
        Returns:
            list: Trajectory data points with position, dynamics, action, and safety_score
        """
        global collision_detected
        collision_detected = False  # Reset collision flag
        
        self.set_random_position()
        rospy.sleep(1.0)  # Wait for position update
        
        trajectory = []
        step = 0
        
        while step < self.max_steps:
            # Generate random action within velocity limits
            linear_vel = random.uniform(0, MAX_LINEAR_VELOCITY)
            angular_vel = random.uniform(-MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
            
            # Store data point with dynamics matching data_collection.py
            data_point = {
                'position': (x_odom, y_odom, theta),
                'dynamics': [x_odom, y_odom, linear_x, angular_z, math.sin(theta), math.cos(theta)],
                'action': (linear_vel, angular_vel),
                'safety_score': 0.0  # Temporary value
            }
            trajectory.append(data_point)
            
            # Execute action
            vel_msg = Twist()
            vel_msg.linear.x = linear_vel
            vel_msg.angular.z = angular_vel
            self.cmd_vel_pub.publish(vel_msg)
            
            # Check for collision using IMU data
            if collision_detected or not self.is_position_valid(x_odom, y_odom):
                # Collision occurred, update safety scores for all points
                collision_step = len(trajectory) - 1
                for i, point in enumerate(trajectory):
                    distance_to_collision = collision_step - i
                    point['safety_score'] = self.calculate_safety_score(distance_to_collision)
                break
                
            step += 1
            rospy.sleep(0.5)  # Control rate
        
        # Stop robot
        vel_msg = Twist()
        vel_msg.linear.x = 0.0
        vel_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(vel_msg)
        
        # Add trajectory points to collected data
        self.collected_data.extend(trajectory)
        
        return trajectory

    def plot_trajectory(self, trajectory, traj_num):
        """Plot trajectory with color-coded safety scores"""
        plt.figure(figsize=(12, 10))
        
        # Plot environment
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
        
        # Plot trajectory with color-coded safety scores
        positions = np.array([point['position'][:2] for point in trajectory])
        safety_scores = np.array([point['safety_score'] for point in trajectory])
        
        scatter = plt.scatter(positions[:, 0], positions[:, 1], 
                            c=safety_scores, cmap='RdYlGn',
                            s=50, alpha=0.7)
        plt.colorbar(scatter, label='Safety Score')
        
        # Plot robot heading at start
        start_pos = trajectory[0]['position']
        arrow_length = 0.2
        dx = arrow_length * math.cos(start_pos[2])
        dy = arrow_length * math.sin(start_pos[2])
        plt.arrow(start_pos[0], start_pos[1], dx, dy, 
                 head_width=0.08, head_length=0.08, fc='b', ec='b')
        
        plt.grid(True)
        plt.axis('equal')
        plt.title(f'Trajectory {traj_num} with Safety Scores')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        
        # Save plot
        plt.savefig(f'{self.plot_dir}/trajectory_{traj_num}.png')
        plt.close()

    def save_data(self):
        """Save collected data to file"""
        data_file = f'{self.data_dir}/safety_data_val.pkl'
        with open(data_file, 'wb') as f:
            pickle.dump(self.collected_data, f)
        print(f"Saved {len(self.collected_data)} data points to {data_file}")

    def run_collection(self, num_trajectories=100):
        """Run data collection for specified number of trajectories
        
        Args:
            num_trajectories (int): Number of trajectories to collect
        """
        print(f"Starting data collection for {num_trajectories} trajectories...")
        print(f"Safety horizon: {SAFETY_HORIZON} steps")
        print(f"Max steps per trajectory: {self.max_steps}")
        print(f"Workspace: X[{WORKSPACE_X_MIN:.1f}, {WORKSPACE_X_MAX:.1f}], Y[{WORKSPACE_Y_MIN:.1f}, {WORKSPACE_Y_MAX:.1f}]")
        
        for i in range(num_trajectories):
            print(f"\nCollecting trajectory {i+1}/{num_trajectories}")
            trajectory = self.collect_trajectory()
            self.plot_trajectory(trajectory, i+1)
            print(f"Trajectory length: {len(trajectory)} steps, Collected data points: {len(self.collected_data)}")
            
        self.save_data()
        print(f"\nData collection complete! Total data points: {len(self.collected_data)}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    import os
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    parser = argparse.ArgumentParser(description='Collect safety data for training safety model')
    parser.add_argument('--num-trajectories', type=int, default=500,
                      help='Number of trajectories to collect')
    parser.add_argument('--output-dir', type=str, default='../data/safety',
                      help='Directory to save collected data')
    parser.add_argument('--safety-horizon', type=int, default=5,
                      help='Number of steps for safety prediction')
    
    args = parser.parse_args()
    
    # Update global safety horizon
    global SAFETY_HORIZON
    SAFETY_HORIZON = args.safety_horizon
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Update collector data directory
    try:
        collector = SafetyDataCollector()
        collector.data_dir = args.output_dir
        collector.run_collection(num_trajectories=args.num_trajectories)
    except rospy.ROSInterruptException:
        pass 