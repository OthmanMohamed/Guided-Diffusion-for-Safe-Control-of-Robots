#!/usr/bin/env python3
"""
Test script for FetchReach environment with obstacles.
"""

import numpy as np
from envs.fetch_reach_obstacles import create_fetch_reach_obstacles_env, OBSTACLE_CONFIGS


def test_environment():
    """Test the custom environment with different obstacle configurations."""
    
    print("=== Testing FetchReach Environment with Obstacles ===\n")
    
    # Test 1: Single box obstacle
    print("1. Testing with single box obstacle...")
    env1 = create_fetch_reach_obstacles_env(OBSTACLE_CONFIGS['single_box'])
    obs1, info1 = env1.reset()
    print(f"   Goal position: {obs1['desired_goal']}")
    print(f"   Obstacle positions: {env1.get_obstacle_positions()}")
    
    # Test collision detection
    test_pos = np.array([1.25, 0.75, 0.42])
    collision = env1.check_collision_with_obstacles(test_pos)
    print(f"   Collision at {test_pos}: {collision}")
    
    # Test 2: Multiple boxes
    print("\n2. Testing with multiple box obstacles...")
    env2 = create_fetch_reach_obstacles_env(OBSTACLE_CONFIGS['multiple_boxes'])
    obs2, info2 = env2.reset()
    print(f"   Goal position: {obs2['desired_goal']}")
    print(f"   Obstacle positions: {env2.get_obstacle_positions()}")
    
    # Test 3: Wall obstacle
    print("\n3. Testing with wall obstacle...")
    env3 = create_fetch_reach_obstacles_env(OBSTACLE_CONFIGS['wall'])
    obs3, info3 = env3.reset()
    print(f"   Goal position: {obs3['desired_goal']}")
    print(f"   Obstacle positions: {env3.get_obstacle_positions()}")
    
    # Test 4: Custom obstacle configuration
    print("\n4. Testing with custom obstacle configuration...")
    custom_obstacles = {
        'obstacles': [
            {
                'name': 'custom_obstacle1',
                'type': 'box',
                'pos': [1.1, 0.6, 0.42],
                'size': [0.03, 0.03, 0.15],
                'rgba': [0.9, 0.1, 0.1, 0.9],
                'contype': 1,
                'conaffinity': 1
            },
            {
                'name': 'custom_obstacle2',
                'type': 'box',
                'pos': [1.3, 0.4, 0.42],
                'size': [0.02, 0.02, 0.18],
                'rgba': [0.1, 0.9, 0.1, 0.9],
                'contype': 1,
                'conaffinity': 1
            }
        ]
    }
    
    env4 = create_fetch_reach_obstacles_env(custom_obstacles)
    obs4, info4 = env4.reset()
    print(f"   Goal position: {obs4['desired_goal']}")
    print(f"   Obstacle positions: {env4.get_obstacle_positions()}")
    
    # Test collision detection with custom obstacles
    test_positions = [
        np.array([1.1, 0.6, 0.42]),  # Inside first obstacle
        np.array([1.3, 0.4, 0.42]),  # Inside second obstacle
        np.array([1.0, 0.5, 0.42]),  # Outside obstacles
    ]
    
    for pos in test_positions:
        collision = env4.check_collision_with_obstacles(pos)
        print(f"   Collision at {pos}: {collision}")
    
    print("\n=== Environment testing completed successfully! ===")


def test_environment_integration():
    """Test integration with existing training scripts."""
    
    print("\n=== Testing Environment Integration ===\n")
    
    # Test that the environment works with the existing build_env function
    from train_offline import build_env
    
    # Create environment with obstacles
    env = create_fetch_reach_obstacles_env(OBSTACLE_CONFIGS['multiple_boxes'])
    
    # Test basic functionality
    obs, info = env.reset()
    print(f"Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Goal position: {obs['desired_goal']}")
    
    # Test taking actions
    action = np.array([0.1, 0.0, 0.0, 0.0])  # Small movement in x direction
    obs, reward, done, truncated, info = env.step(action)
    print(f"Action taken: {action}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Success: {info.get('is_success', False)}")
    
    print("\n=== Integration testing completed! ===")


if __name__ == "__main__":
    test_environment()
    test_environment_integration() 