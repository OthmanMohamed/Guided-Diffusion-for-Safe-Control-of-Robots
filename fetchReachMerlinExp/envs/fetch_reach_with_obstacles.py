#!/usr/bin/env python3
"""
Custom FetchReach environment with obstacles.

This wrapper adds obstacles to the standard FetchReach environment
by modifying the MuJoCo model directly.
"""

import gymnasium as gym
import numpy as np
from gymnasium.core import Wrapper
import mujoco


class FetchReachWithObstacles(Wrapper):
    def __init__(self, env, obstacles_config=None):
        """
        Initialize FetchReach environment with obstacles.
        
        Args:
            env: Base FetchReach environment
            obstacles_config: Dictionary defining obstacles
                Format: {
                    'obstacles': [
                        {
                            'name': 'obstacle1',
                            'type': 'box',
                            'pos': [x, y, z],
                            'size': [width, height, depth],
                            'rgba': [r, g, b, a],
                            'contype': 1,
                            'conaffinity': 1
                        },
                        ...
                    ]
                }
        """
        super().__init__(env)
        self.env = env
        
        # Default obstacles configuration
        if obstacles_config is None:
            obstacles_config = {
                'obstacles': [
                    {
                        'name': 'obstacle1',
                        'type': 'box',
                        'pos': [1.25, 0.75, 0.42],
                        'size': [0.025, 0.025, 0.2],
                        'rgba': [0.8, 0.2, 0.2, 0.8],
                        'contype': 1,
                        'conaffinity': 1
                    },
                    {
                        'name': 'obstacle2',
                        'type': 'box',
                        'pos': [1.0, 0.5, 0.42],
                        'size': [0.025, 0.025, 0.2],
                        'rgba': [0.2, 0.8, 0.2, 0.8],
                        'contype': 1,
                        'conaffinity': 1
                    }
                ]
            }
        
        self.obstacles_config = obstacles_config
        self.add_obstacles()
    
    def add_obstacles(self):
        """Add obstacles to the MuJoCo model."""
        model = self.env.unwrapped.model
        
        # Get the worldbody
        worldbody = model.find('worldbody')
        if worldbody is None:
            print("Warning: Could not find worldbody in MuJoCo model")
            return
        
        # Add each obstacle
        for obstacle in self.obstacles_config['obstacles']:
            self.add_single_obstacle(worldbody, obstacle)
        
        # Recompile the model
        mujoco.mj_forward(model, self.env.unwrapped.data)
    
    def add_single_obstacle(self, worldbody, obstacle):
        """Add a single obstacle to the worldbody."""
        # Create obstacle body
        obstacle_body = mujoco.mj_addBody(worldbody, obstacle['name'])
        
        # Add geom to the body
        geom = mujoco.mj_addGeom(obstacle_body, obstacle['type'])
        
        # Set geom properties
        geom.pos = obstacle['pos']
        geom.size = obstacle['size']
        geom.rgba = obstacle['rgba']
        geom.contype = obstacle['contype']
        geom.conaffinity = obstacle['conaffinity']
    
    def remove_obstacles(self):
        """Remove all obstacles from the environment."""
        model = self.env.unwrapped.model
        
        # Remove obstacle bodies
        for obstacle in self.obstacles_config['obstacles']:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obstacle['name'])
            if body_id >= 0:
                mujoco.mj_deleteBody(model, body_id)
        
        # Recompile the model
        mujoco.mj_forward(model, self.env.unwrapped.data)
    
    def update_obstacles(self, new_obstacles_config):
        """Update obstacles with new configuration."""
        self.remove_obstacles()
        self.obstacles_config = new_obstacles_config
        self.add_obstacles()
    
    def get_obstacle_positions(self):
        """Get current positions of all obstacles."""
        positions = []
        model = self.env.unwrapped.model
        
        for obstacle in self.obstacles_config['obstacles']:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obstacle['name'])
            if body_id >= 0:
                pos = model.body_pos[body_id].copy()
                positions.append(pos)
        
        return positions


def create_fetch_reach_with_obstacles(obstacles_config=None, render_mode="rgb_array"):
    """
    Create a FetchReach environment with obstacles.
    
    Args:
        obstacles_config: Configuration for obstacles (see FetchReachWithObstacles)
        render_mode: Render mode for the environment
    
    Returns:
        FetchReachWithObstacles: Environment with obstacles
    """
    # Create base environment
    base_env = gym.make("FetchReach-v2", render_mode=render_mode)
    
    # Wrap with obstacles
    env = FetchReachWithObstacles(base_env, obstacles_config)
    
    return env

