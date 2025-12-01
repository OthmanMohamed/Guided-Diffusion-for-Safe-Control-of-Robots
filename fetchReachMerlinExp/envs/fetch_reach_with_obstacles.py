#!/usr/bin/env python3
"""
Custom FetchReach environment with obstacles.

This wrapper provides access to the FetchReach environment with obstacles.
Note: Obstacles must be added directly to the environment source files.
See README.md for instructions on adding obstacles.
"""

import gymnasium as gym
import numpy as np
from gymnasium.core import Wrapper
import mujoco


class FetchReachWithObstacles(Wrapper):
    """
    Wrapper for FetchReach environment with obstacles.
    
    Note: This wrapper assumes obstacles are already defined in the 
    environment's source XML/model files. It does not add obstacles programmatically.
    """
    def __init__(self, env, obstacles_config=None):
        """
        Initialize FetchReach environment wrapper.
        
        Args:
            env: Base FetchReach environment (with obstacles already in source)
            obstacles_config: Optional config (kept for compatibility, not used)
        """
        super().__init__(env)
        self.env = env
    
    def get_obstacle_positions(self):
        """
        Get current positions of all obstacles.
        
        This method attempts to find obstacles by name in the MuJoCo model.
        Adjust obstacle names based on how they are defined in your source files.
        """
        positions = []
        model = self.env.unwrapped.model
        
        # Try to find common obstacle names
        # Adjust these names based on your actual obstacle definitions
        obstacle_names = ['obstacle1', 'obstacle2', 'obstacle1_geom', 'obstacle2_geom']
        
        for name in obstacle_names:
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id >= 0:
                    pos = model.body_pos[body_id].copy()
                    positions.append(pos)
            except:
                # Try as geom instead
                try:
                    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                    if geom_id >= 0:
                        body_id = model.geom_bodyid[geom_id]
                        pos = model.body_pos[body_id].copy()
                        positions.append(pos)
                except:
                    continue
        
        return positions


def create_fetch_reach_with_obstacles(obstacles_config=None, render_mode="rgb_array"):
    """
    Create a FetchReach environment with obstacles.
    
    Note: Obstacles must be added directly to the environment source files.
    This function simply creates the base environment and wraps it.
    
    Args:
        obstacles_config: Optional config (kept for compatibility, not used)
        render_mode: Render mode for the environment
    
    Returns:
        FetchReachWithObstacles: Environment wrapper
    """
    # Create base environment
    # Note: Obstacles should already be defined in the environment source
    base_env = gym.make("FetchReach-v2", render_mode=render_mode)
    
    # Wrap with obstacles wrapper
    env = FetchReachWithObstacles(base_env, obstacles_config)
    
    return env

