#!/usr/bin/env python
"""
Data Collection Script for TurtleBot Robot
==========================================
This script collects expert demonstrations or random trajectories for training
guided diffusion policies. It handles obstacle avoidance, collision detection,
and data saving to pickle files.

Usage:
    python data_collection.py
    
Configuration can be modified in the main() function.
"""

import rospy
from geometry_msgs.msg import Twist, Pose, PoseStamped
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math
import random
from std_srvs.srv import Empty 
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Imu
import cv2
import os
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# GLOBAL STATE VARIABLES
# ==============================================================================
# These store the current state of the robot
odom = None
image = None
collision_detected = False
last_linear_accel = None
last_collision_time = 0

# Data dictionaries for storing collected trajectories
data_dict = {}
data_dict['o'] = []      # Observations (with sin/cos orientation)
data_dict['o2'] = []     # Alternative observation representation
data_dict['u'] = []      # Actions (velocity commands)
data_dict['g'] = []      # Goals
data_dict['ag'] = []     # Achieved goals
data_dict['test_vectors'] = []  # Test vectors for evaluation

bridge = CvBridge()     

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================
# Robot control parameters
COLLISION_ACCEL_THRESHOLD = 15.0      # m/s² - acceleration change threshold for collision detection
COLLISION_COOLDOWN_TIME = 0.5         # seconds - minimum time between collision detections
MIN_MOVEMENT_SPEED = 0.1              # m/s - minimum speed to consider robot as moving
MAX_TRIAL_STEPS = 50                  # Maximum number of steps per trial
GOAL_SUCCESS_THRESHOLD = 0.2          # meters - distance to goal for success

# Velocity limits
MAX_LINEAR_VELOCITY = 0.5             # m/s - maximum forward velocity
BASE_LINEAR_VELOCITY = 0.4            # m/s - base forward velocity
MAX_ANGULAR_VELOCITY = 1.0            # rad/s - maximum rotation velocity
ANGULAR_GAIN = 1.0                    # Angular velocity control gain

# Workspace boundaries
WORKSPACE_X_MIN = -1.4                # meters
WORKSPACE_X_MAX = 1.4                 # meters
WORKSPACE_Y_MIN = -1.9                # meters
WORKSPACE_Y_MAX = 1.9                 # meters

# Robot initial position range
START_X_MIN = -1.4                    # meters
START_X_MAX = 1.4                     # meters
START_Y_MIN = -1.9                    # meters
START_Y_MAX = 1.9                     # meters

# Potential field navigation parameters
PF_ATTRACTIVE_GAIN = 0.5              # Attractive force gain
PF_GOAL_DISTANCE_THRESHOLD = 0.3      # Distance threshold for attractive force
PF_REPULSIVE_GAIN = 2                 # Repulsive force gain
PF_OBSTACLE_DISTANCE_THRESHOLD = 0.1  # Distance threshold for repulsive force
PF_ANGULAR_GAIN = 0.8                 # Angular velocity gain for smoother control

# ==============================================================================
# ROS CALLBACK FUNCTIONS
# ==============================================================================

def image_callback(data):
    """Callback to store the latest camera image from overhead camera"""
    global image
    try:
        image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
    except Exception as e:
        rospy.logerr("Image conversion error: %s", str(e))

def odom_callback(msg):
    """Callback to store the latest odometry data (position, velocity, orientation)"""
    global odom, x_odom, y_odom, theta, theta2, linear_x, angular_z
    # Update current position
    x_odom = msg.pose.pose.position.x
    y_odom = msg.pose.pose.position.y
    linear_x = msg.twist.twist.linear.x
    angular_z = msg.twist.twist.angular.z
    
    # Get the orientation in quaternion
    orientation_q = msg.pose.pose.orientation
    _, _, theta = euler_from_quaternion([0, 0, orientation_q.z, orientation_q.w])
    _, _, theta2 = euler_from_quaternion([0, 0, orientation_q.w, orientation_q.z])
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

# Define obstacle positions and parameters
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
# NAVIGATION HELPER FUNCTIONS
# ==============================================================================

def is_position_valid(x, y):
    """Check if a position is valid (not inside or too close to obstacles)"""
    # Check cylindrical obstacles
    for obs_x, obs_y in CYLINDRICAL_OBSTACLES:
        distance = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
        if distance < CYLINDRICAL_RADIUS:
            return False
            
    # Check hexagonal obstacles
    for obs_x, obs_y, scale in HEXAGONAL_OBSTACLES:
        distance = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
        if distance < HEXAGONAL_RADIUS * scale:
            return False
            
    return True

def calculate_forces(x, y, goal_x, goal_y, attraction=True):
    """Calculate attractive and repulsive forces for potential field navigation
    
    Args:
        x, y: Current robot position
        goal_x, goal_y: Goal position
        attraction: Whether to apply attractive force
    
    Returns:
        f_total_x, f_total_y: Total force components (normalized)
    """
    # Attractive force parameters
    k_att = PF_ATTRACTIVE_GAIN if attraction else 0
    d_goal = PF_GOAL_DISTANCE_THRESHOLD
    
    # Repulsive force parameters
    k_rep = PF_REPULSIVE_GAIN
    d_obs = PF_OBSTACLE_DISTANCE_THRESHOLD
    
    # Calculate attractive force
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.sqrt(dx**2 + dy**2)
    
    if dist_to_goal < d_goal:
        f_att_x = k_att * dx
        f_att_y = k_att * dy
    else:
        f_att_x = k_att * d_goal * dx / dist_to_goal
        f_att_y = k_att * d_goal * dy / dist_to_goal
    
    # Initialize repulsive forces
    f_rep_x = 0.0
    f_rep_y = 0.0
    
    # Calculate repulsive forces from cylindrical obstacles
    for obs_x, obs_y in CYLINDRICAL_OBSTACLES:
        dx = x - obs_x
        dy = y - obs_y
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist < d_obs:
            # Smoother repulsive force calculation
            force_mag = k_rep * (1/dist - 1/d_obs)**2
            f_rep_x += force_mag * dx / dist
            f_rep_y += force_mag * dy / dist
    
    # Calculate repulsive forces from hexagonal obstacles
    for obs_x, obs_y, scale in HEXAGONAL_OBSTACLES:
        dx = x - obs_x
        dy = y - obs_y
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist < d_obs * scale:
            # Smoother repulsive force calculation
            force_mag = k_rep * (1/dist - 1/(d_obs * scale))**2
            f_rep_x += force_mag * dx / dist
            f_rep_y += force_mag * dy / dist
    
    # Combine forces
    f_total_x = f_att_x + f_rep_x
    f_total_y = f_att_y + f_rep_y
    
    # Normalize forces
    force_magnitude = math.sqrt(f_total_x**2 + f_total_y**2)
    if force_magnitude > 0:
        f_total_x /= force_magnitude
        f_total_y /= force_magnitude
    
    return f_total_x, f_total_y


# ==============================================================================
# MAIN CONTROLLER CLASS
# ==============================================================================
        
class TurtleBotController:
    """Controller for TurtleBot robot navigation and data collection
    
    Handles robot movement, obstacle avoidance, collision detection,
    and trajectory data collection for training guided diffusion policies.
    """
    def __init__(self, goal_x, goal_y):
        """Initialize the TurtleBot controller
        
        Args:
            goal_x, goal_y: Target goal position coordinates
        """
        # Initialize the node
        rospy.init_node("turtlebot_move_to_goal", anonymous=True)
        
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        
        # Target goal position
        self.goal_x = goal_x
        self.goal_y = goal_y

        self.image_output_path = "./data/expert_with_ims"
        # Create the directory if it doesn't exist
        os.makedirs(self.image_output_path, exist_ok=True)
        self.images_to_save = []
        
        # Initialize ROS service proxies
        self.reset_service = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.set_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    def set_random_start_position(self):
        """Set a random start position for the robot within the workspace bounds"""
        initial_x = random.uniform(START_X_MIN, START_X_MAX)
        initial_y = random.uniform(START_Y_MIN, START_Y_MAX)
        initial_theta = random.uniform(-math.pi, math.pi)  # Random orientation

        # Create the request to set the model state
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

        # Send the request
        try:
            set_model_state_service(state_msg)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to set start position: %s", str(e))

    def save_images_to_disk(self):
        for filename, img in self.images_to_save:
            if img is not None:
                cv2.imwrite(filename, img)
        self.images_to_save.clear()  # Clear after saving

    def plot_trajectory(self, trajectory, start_pos, goal_pos, success, traj_index, plot_dir="trajectory_plots_data_collection"):
        """Plot and save the robot's trajectory"""
        print("Plotting traj")
        plt.figure(figsize=(10, 8))
        
        # Plot trajectory
        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Robot Path', linewidth=2)
        
        # Plot start position
        plt.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start Position')
        
        # Plot goal position
        plt.plot(goal_pos[0], goal_pos[1], 'ro', markersize=10, label='Goal Position')
        
        # Plot success/failure circle around goal
        circle = plt.Circle((goal_pos[0], goal_pos[1]), 0.2, 
                          color='g' if success else 'r', fill=False, linestyle='--')
        plt.gca().add_patch(circle)
        
        # Plot cylindrical obstacles
        for obs_x, obs_y in CYLINDRICAL_OBSTACLES:
            obstacle = plt.Circle((obs_x, obs_y), CYLINDRICAL_RADIUS, 
                                color='gray', alpha=0.5, label='Cylindrical Obstacle' if obs_x == CYLINDRICAL_OBSTACLES[0][0] else "")
            plt.gca().add_patch(obstacle)
            
        # Plot hexagonal obstacles
        for obs_x, obs_y, scale in HEXAGONAL_OBSTACLES:
            obstacle = plt.Circle((obs_x, obs_y), HEXAGONAL_RADIUS * scale, 
                                color='darkgray', alpha=0.5, label='Hexagonal Obstacle' if obs_x == HEXAGONAL_OBSTACLES[0][0] else "")
            plt.gca().add_patch(obstacle)
        
        plt.autoscale(False)
        plt.xlim(-1.6, 1.6)
        plt.ylim(-2.1, 2.1)
        
        # Set plot properties
        plt.grid(True)
        plt.axis('equal')
        plt.title(f'Trajectory {traj_index} - {"Success" if success else "Failure"}')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.legend()
        
        # Create directory if it doesn't exist
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        # Save plot
        status = "success" if success else "fail"
        plt.savefig(f'{plot_dir}/traj_{traj_index}_{status}.png')
        plt.close()

    def move_to_goal(self, traj_index, expert=True, save_images=True, plot_trajs=False, obstacles=True):
        """Move robot to goal and collect trajectory data
        
        Args:
            traj_index: Index of the current trajectory
            expert: Whether this is an expert demonstration (True) or random exploration (False)
            save_images: Whether to save camera images
            plot_trajs: Whether to generate trajectory plots
            obstacles: Whether obstacles are enabled
            
        Returns:
            success_flag: Whether the goal was reached
            o, o2, u, g, ag, test_vectors: Collected trajectory data
        """
        global collision_detected
        collision_detected = False  # Reset collision flag at start
        
        # Set random start position at the beginning of each trial
        vel_msg = Twist()
        vel_msg.linear.x = 0.0
        vel_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(vel_msg)
        rospy.sleep(2)
        
        # Initialize ROS services and subscribers
        reset_service = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_service()
        self.set_random_start_position()
        rospy.sleep(2)
        
        g, o, o2, u, ag, test_vectors = [], [], [], [], [], []
        trajectory = [(x_odom, y_odom)]  # Store trajectory points
        start_pos = (x_odom, y_odom)
        i = 0
        
        while not rospy.is_shutdown():
            if i == MAX_TRIAL_STEPS: 
                if expert:
                    if save_images: self.images_to_save.clear()
                    if plot_trajs:
                        self.plot_trajectory(trajectory, start_pos, (self.goal_x, self.goal_y), False, traj_index)
                    return False, o, o2, u, g, ag, test_vectors
                else: 
                    if save_images: self.save_images_to_disk()
                    if plot_trajs:
                        self.plot_trajectory(trajectory, start_pos, (self.goal_x, self.goal_y), True, traj_index)
                    return True, o, o2, u, g, ag, test_vectors
                
            # Check for collision only if obstacles are enabled
            if obstacles and collision_detected:
                print("Collision detected! Stopping robot.")
                vel_msg.linear.x = 0.0
                vel_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(vel_msg)
                if save_images: self.images_to_save.clear()
                if plot_trajs:
                    self.plot_trajectory(trajectory, start_pos, (self.goal_x, self.goal_y), False, traj_index)
                return False, o, o2, u, g, ag, test_vectors
            
            for j in range(5):
                if image is not None: 
                    image_filename = os.path.join(self.image_output_path, f"image_{traj_index}_{i}_{j}.jpg")
                    if save_images:
                        # Store images in memory
                        self.images_to_save.append((image_filename, image))
                rospy.sleep(0.1)

            # Store current position in trajectory
            trajectory.append((x_odom, y_odom))

            o_item = [x_odom, y_odom, linear_x, angular_z, math.sin(theta), math.cos(theta)]
            o_item2 = [x_odom, y_odom, linear_x, angular_z, math.sin(theta2), math.cos(theta2)]
            ag_item = [x_odom, y_odom]
            im_item = os.path.join(self.image_output_path, f"image_{traj_index}_{i}.jpg")
            g_item = [self.goal_x, self.goal_y]

            if expert:
                # Calculate the distance and angle to the goal
                distance = math.sqrt((self.goal_x - x_odom)**2 + (self.goal_y - y_odom)**2)
                if obstacles:
                    # Use potential field navigation when obstacles are enabled
                    f_x, f_y = calculate_forces(x_odom, y_odom, self.goal_x, self.goal_y)
                    
                    # Calculate desired heading
                    desired_heading = math.atan2(f_y, f_x)
                    
                    # Calculate angular error
                    angular_error = desired_heading - theta
                    
                    # Normalize angular error to [-π, π]
                    while angular_error > math.pi:
                        angular_error -= 2 * math.pi
                    while angular_error < -math.pi:
                        angular_error += 2 * math.pi
                    
                    # Calculate angular velocity with smoother control
                    angular_vel = PF_ANGULAR_GAIN * angular_error  # Reduced gain for smoother turning
                    
                    # Scale angular velocity to be between -MAX_ANGULAR_VELOCITY and MAX_ANGULAR_VELOCITY
                    angular_vel = min(max(angular_vel, -MAX_ANGULAR_VELOCITY), MAX_ANGULAR_VELOCITY)
                    
                    # Adjust linear velocity based on angular error
                    linear_vel = min(BASE_LINEAR_VELOCITY * distance, 0.22) * (1 - 0.5 * abs(angular_error/math.pi))
                else:
                    angle_to_goal = math.atan2(self.goal_y - y_odom, self.goal_x - x_odom)
                    angle_error = math.atan2(math.sin(angle_to_goal - theta), math.cos(angle_to_goal - theta))
                
                # Set velocities
            if expert:
                vel_msg.linear.x = min(BASE_LINEAR_VELOCITY * distance, (0.22 if obstacles else MAX_LINEAR_VELOCITY))
                if obstacles: 
                    vel_msg.angular.z = angular_vel
                    vel_msg.linear.x = linear_vel
                else: 
                    vel_msg.angular.z = min(max(ANGULAR_GAIN * angle_error, -MAX_ANGULAR_VELOCITY), MAX_ANGULAR_VELOCITY)
            else:
                vel_msg.linear.x = random.uniform(0, MAX_LINEAR_VELOCITY)
                vel_msg.angular.z = random.uniform(-MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
            u_item = [vel_msg.linear.x, vel_msg.angular.z]
            
            # Publish the velocity message
            self.cmd_vel_pub.publish(vel_msg)
            
            g.append(g_item)
            u.append(u_item)
            o.append(im_item)
            o2.append(o_item2)
            ag.append(ag_item)
            test_vectors.append(o_item)
            i += 1

            # Check if robot has left the workspace (when obstacles are disabled)
            if expert and not obstacles and (abs(x_odom) > abs(WORKSPACE_X_MAX) or abs(y_odom) > abs(WORKSPACE_Y_MAX)):
                vel_msg.linear.x = 0.0
                vel_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(vel_msg)
                if save_images: self.images_to_save.clear()
                if plot_trajs:
                        self.plot_trajectory(trajectory, start_pos, (self.goal_x, self.goal_y), False, traj_index)
                return False, o, o2, u, g, ag, test_vectors

            # Check if the bot is close enough to the goal
            if (expert and distance < GOAL_SUCCESS_THRESHOLD) or (not expert and abs(x_odom) > 3.0 or abs(y_odom) > 3.0):
                vel_msg.linear.x = 0.0
                vel_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(vel_msg)
                rospy.loginfo("Reached the goal!")
                if expert:
                    # Store the latest reached state in g
                    g = [[x_odom, y_odom] for _ in range(len(g))]
                    if save_images:
                        self.save_images_to_disk()  # Save images at the end of each trial
                    if plot_trajs:
                            self.plot_trajectory(trajectory, start_pos, (self.goal_x, self.goal_y), True, traj_index)
                elif save_images:
                    self.images_to_save.clear()
                return True if expert else False, o, o2, u, g, ag, test_vectors

        # self.save_images_to_disk()  # Save images if trial ends without reaching goal
        return False, o, o2, u, g, ag, test_vectors


# ==============================================================================
# ROS SERVICES AND SUBSCRIBERS INITIALIZATION
# ==============================================================================

def initialize_ros_services_and_subscribers():
    """Initialize ROS service proxies and subscribers"""
    reset_service = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    set_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    
    # Subscribe to topics
    rospy.Subscriber("/odom", Odometry, odom_callback)
    rospy.Subscriber("/overhead_camera/overhead_camera/image_raw", Image, image_callback)
    rospy.Subscriber("/imu", Imu, imu_callback)
    
    return reset_service, set_model_state_service


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    try:
        # =====================================================================
        # CONFIGURATION - MODIFY THESE PARAMETERS AS NEEDED
        # =====================================================================
        NUM_TRIALS = 3000
        SAVE_IMAGES = False         # Set to True to save camera images
        PLOT_TRAJECTORIES = True     # Set to True to save trajectory plots
        USE_OBSTACLES = True         # Set to False to disable obstacle avoidance
        EXPERT_MODE = False          # True for expert demonstrations, False for random exploration
        OUTPUT_FILENAME = "data/obstacles_random_no_collision.pkl"
        # =====================================================================
        
        trial_index = 0
        total_time = 0
        
        # Create second data dictionary for expert.pkl
        data_dict2 = {}
        data_dict2['o'] = []
        data_dict2['u'] = []
        data_dict2['g'] = []
        data_dict2['ag'] = []
        if SAVE_IMAGES: 
            data_dict2['im'] = []
        
        print(f"Starting data collection with {NUM_TRIALS} trials")
        print(f"Expert mode: {EXPERT_MODE}, Obstacles: {USE_OBSTACLES}, Save images: {SAVE_IMAGES}")
        
        while trial_index < NUM_TRIALS:
            tic = time.time()
            
            # Generate random goal position within workspace bounds
            goal_x = random.uniform(WORKSPACE_X_MIN, WORKSPACE_X_MAX)
            goal_y = random.uniform(WORKSPACE_Y_MIN, WORKSPACE_Y_MAX)
            
            print(f"Trial {trial_index}: Goal at ({goal_x:.2f}, {goal_y:.2f})")
            
            controller = TurtleBotController(goal_x, goal_y)
            success_flag, o, o2, u, g, ag, test_vectors = controller.move_to_goal(
                trial_index, 
                expert=EXPERT_MODE, 
                save_images=SAVE_IMAGES, 
                plot_trajs=PLOT_TRAJECTORIES, 
                obstacles=USE_OBSTACLES
            )
            
            if success_flag:
                # Save data for expert2.pkl
                data_dict['o'].append(o)
                data_dict['o2'].append(o2)
                data_dict['u'].append(u)
                data_dict['g'].append(g)
                data_dict['ag'].append(ag)
                data_dict['test_vectors'].append(test_vectors)
                
                # Save data for expert.pkl
                data_dict2['o'].append(test_vectors)  # Using test_vectors as o
                data_dict2['u'].append(u)
                data_dict2['g'].append(g)
                data_dict2['ag'].append(ag)
                if SAVE_IMAGES: 
                    data_dict2['im'].append(o)
                
                trial_index += 1
                
                tac = time.time()
                elapsed = tac - tic
                total_time += elapsed
                avg_time = total_time / trial_index
                
                print(f"Trial completed in {elapsed:.2f}s | Average: {avg_time:.2f}s")
            else:
                tac = time.time()
                print(f"Trial failed after {tac - tic:.2f}s")
            
        # Save collected data to pickle file
        print(f"\nSaving collected data to {OUTPUT_FILENAME}")
        with open(OUTPUT_FILENAME, "wb") as f:
            pickle.dump(data_dict2, f)
        
        print(f"\nData collection complete!")
        print(f"Total successful trials: {trial_index}/{NUM_TRIALS}")
        print(f"Average time per trial: {total_time / trial_index:.2f}s")
        
    except rospy.ROSInterruptException:
        pass
