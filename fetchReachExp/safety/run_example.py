#!/usr/bin/env python3
"""
Example script showing how to run safety data collection with different parameters.
"""

import os
import sys
from collect_safety_data import collect_safety_data, argparse

def run_example():
    """Run an example data collection with reasonable parameters."""
    
    print("=== Safety Data Collection Example ===")
    print("This example will collect 50 episodes of safety data.")
    print("Each episode will run until collision or max 50 steps.")
    print("Actions will be labeled as safe/unsafe based on 5 steps before collision.\n")
    
    # Create example arguments
    example_args = argparse.Namespace(
        env='FetchReach-v2',
        num_episodes=500,  # Reasonable number for example
        max_steps_per_episode=100,
        unsafe_steps_before_collision=5,  # Updated parameter name
        random_seed=420,
        output_dir='example_safety_data_0.2_full_episodes_eval',
        render=False  # Set to True if you want to see the simulation
    )
    
    # Run data collection
    collected_data = collect_safety_data(example_args)
    
    print("\n=== Example Complete ===")
    print("You can now use this data to train a safety model.")
    print(f"Data saved to: {example_args.output_dir}/safety_data_final.pkl")
    
    return collected_data

if __name__ == "__main__":
    run_example() 