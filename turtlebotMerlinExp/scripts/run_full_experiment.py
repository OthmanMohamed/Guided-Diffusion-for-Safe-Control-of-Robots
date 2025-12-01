#!/usr/bin/env python3
"""
Main Script: Run Full Experiment

This script runs the complete pipeline for the TurtleBot Merlin experiment with safety:
1. Collect demonstration data (requires ROS/Gazebo)
2. Collect safety data (requires ROS/Gazebo)
3. Train safety model
4. Train Merlin policy
5. Evaluate policy (with and without safety, requires ROS/Gazebo)

Note: This experiment uses ROS simulation data. Ensure ROS and Gazebo are running
with the TurtleBot3 simulation before running data collection or evaluation scripts.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with return code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Run full TurtleBot Merlin experiment with safety')
    
    # Data collection arguments
    parser.add_argument('--skip-data-collection', action='store_true',
                      help='Skip demonstration data collection (use existing data)')
    parser.add_argument('--skip-safety-data', action='store_true',
                      help='Skip safety data collection (use existing data)')
    parser.add_argument('--num-demo-trials', type=int, default=3000,
                      help='Number of demonstration trials to collect')
    parser.add_argument('--num-safety-trajectories', type=int, default=500,
                      help='Number of safety data trajectories to collect')
    
    # Training arguments
    parser.add_argument('--skip-safety-training', action='store_true',
                      help='Skip safety model training (use existing model)')
    parser.add_argument('--skip-policy-training', action='store_true',
                      help='Skip policy training (use existing model)')
    parser.add_argument('--epochs', type=int, default=10000,
                      help='Number of training epochs for policy')
    
    # Evaluation arguments
    parser.add_argument('--skip-evaluation', action='store_true',
                      help='Skip evaluation')
    parser.add_argument('--num-eval-episodes', type=int, default=100,
                      help='Number of evaluation episodes')
    
    # Path arguments
    parser.add_argument('--data-dir', type=str, default='data',
                      help='Directory for data files')
    parser.add_argument('--models-dir', type=str, default='models',
                      help='Directory for model files')
    parser.add_argument('--logs-dir', type=str, default='logs',
                      help='Directory for log files')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Setup directories
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / args.data_dir
    models_dir = base_dir / args.models_dir
    logs_dir = base_dir / args.logs_dir
    
    data_dir.mkdir(exist_ok=True, parents=True)
    models_dir.mkdir(exist_ok=True, parents=True)
    logs_dir.mkdir(exist_ok=True, parents=True)
    
    demo_data_dir = data_dir / 'demonstration'
    safety_data_dir = data_dir / 'safety'
    safety_models_dir = models_dir / 'safety'
    policy_models_dir = models_dir / 'policy'
    
    demo_data_dir.mkdir(exist_ok=True, parents=True)
    safety_data_dir.mkdir(exist_ok=True, parents=True)
    safety_models_dir.mkdir(exist_ok=True, parents=True)
    policy_models_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*60)
    print("TurtleBot Merlin Experiment with Safety")
    print("="*60)
    print(f"Base directory: {base_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Models directory: {models_dir}")
    print(f"Logs directory: {logs_dir}")
    print("="*60)
    
    # Step 1: Collect demonstration data
    demo_data_path = demo_data_dir / 'demonstration_data.pkl'
    if not args.skip_data_collection and not demo_data_path.exists():
        print("\nWARNING: Data collection requires ROS and Gazebo to be running!")
        print("Make sure you have:")
        print("  1. ROS installed and sourced")
        print("  2. TurtleBot3 simulation running in Gazebo")
        print("  3. All required ROS topics available")
        response = input("\nContinue with data collection? (y/n): ")
        if response.lower() != 'y':
            print("Skipping data collection.")
        else:
            cmd = [
                sys.executable, 'scripts/01_collect_demonstration_data.py',
                '--num-trials', str(args.num_demo_trials),
                '--output-filename', str(demo_data_path)
            ]
            if not run_command(cmd, "Collecting demonstration data"):
                return
    else:
        print(f"\nSkipping demonstration data collection (file exists: {demo_data_path.exists()})")
    
    # Step 2: Collect safety data
    safety_data_path = safety_data_dir / 'safety_data_val.pkl'
    if not args.skip_safety_data and not safety_data_path.exists():
        print("\nWARNING: Safety data collection requires ROS and Gazebo to be running!")
        response = input("Continue with safety data collection? (y/n): ")
        if response.lower() != 'y':
            print("Skipping safety data collection.")
        else:
            cmd = [
                sys.executable, 'scripts/02_collect_safety_data.py',
                '--num-trajectories', str(args.num_safety_trajectories),
                '--output-dir', str(safety_data_dir)
            ]
            if not run_command(cmd, "Collecting safety data"):
                return
    else:
        print(f"\nSkipping safety data collection (file exists: {safety_data_path.exists()})")
    
    # Step 3: Train safety model
    safety_model_path = None
    if not args.skip_safety_training:
        # Look for existing safety models
        safety_model_dirs = list(safety_models_dir.glob('safety_model_*'))
        if safety_model_dirs:
            safety_model_path = max(safety_model_dirs, key=os.path.getctime) / 'best_model.pt'
            print(f"\nFound existing safety model: {safety_model_path}")
        else:
            # Train new safety model
            # Note: You may need to split your safety data or provide separate train/val files
            train_data = safety_data_path
            val_data = safety_data_dir / 'safety_data_val.pkl'
            
            if not val_data.exists():
                print(f"\nWARNING: Validation safety data not found at {val_data}")
                print("You may need to split your safety data or provide a validation set.")
                print("Skipping safety model training.")
            else:
                cmd = [
                    sys.executable, 'scripts/03_train_safety_model.py',
                    '--train-data', str(train_data),
                    '--val-data', str(val_data),
                    '--output-dir', str(safety_models_dir)
                ]
                if not run_command(cmd, "Training safety model"):
                    return
                
                # Find the newly created model
                safety_model_dirs = list(safety_models_dir.glob('safety_model_*'))
                if safety_model_dirs:
                    safety_model_path = max(safety_model_dirs, key=os.path.getctime) / 'best_model.pt'
    else:
        # Find existing safety model
        safety_model_dirs = list(safety_models_dir.glob('safety_model_*'))
        if safety_model_dirs:
            safety_model_path = max(safety_model_dirs, key=os.path.getctime) / 'best_model.pt'
            print(f"\nUsing existing safety model: {safety_model_path}")
    
    # Step 4: Train Merlin policy
    policy_checkpoint_path = None
    if not args.skip_policy_training:
        cmd = [
            sys.executable, 'scripts/04_train_merlin_policy.py',
            '--train_data_path', str(demo_data_path),
            '--val_data_path', str(demo_data_path),
            '--epochs', str(args.epochs),
            '--model_dir', str(policy_models_dir)
        ]
        if not run_command(cmd, "Training Merlin policy"):
            return
        
        # Find the most recent checkpoint
        if policy_models_dir.exists():
            checkpoints = list(policy_models_dir.rglob('model_*.pth'))
            if checkpoints:
                policy_checkpoint_path = max(checkpoints, key=os.path.getctime)
                print(f"\nPolicy checkpoint saved: {policy_checkpoint_path}")
    else:
        # Find existing checkpoint
        if policy_models_dir.exists():
            checkpoints = list(policy_models_dir.rglob('model_*.pth'))
            if checkpoints:
                policy_checkpoint_path = max(checkpoints, key=os.path.getctime)
                print(f"\nUsing existing policy checkpoint: {policy_checkpoint_path}")
    
    # Step 5: Evaluate policy
    if not args.skip_evaluation and policy_checkpoint_path:
        print("\nWARNING: Evaluation requires ROS and Gazebo to be running!")
        response = input("Continue with evaluation? (y/n): ")
        if response.lower() != 'y':
            print("Skipping evaluation.")
        else:
            # Evaluate without safety
            print(f"\n{'='*60}")
            print("Evaluating policy WITHOUT safety")
            print(f"{'='*60}\n")
            cmd = [
                sys.executable, 'scripts/05_evaluate_policy.py',
                '--model_path', str(policy_checkpoint_path),
                '--num_episodes', str(args.num_eval_episodes)
            ]
            run_command(cmd, "Evaluating policy without safety")
            
            # Evaluate with safety
            if safety_model_path and safety_model_path.exists():
                print(f"\n{'='*60}")
                print("Evaluating policy WITH safety")
                print(f"{'='*60}\n")
                cmd = [
                    sys.executable, 'scripts/06_evaluate_policy_with_safety.py',
                    '--model_path', str(policy_checkpoint_path),
                    '--safety_model_path', str(safety_model_path),
                    '--num_episodes', str(args.num_eval_episodes)
                ]
                run_command(cmd, "Evaluating policy with safety")
            else:
                print(f"\nWARNING: Safety model not found at {safety_model_path}")
                print("Skipping evaluation with safety.")
    
    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)
    print(f"\nResults saved in:")
    print(f"  - Data: {data_dir}")
    print(f"  - Models: {models_dir}")
    print(f"  - Logs: {logs_dir}")


if __name__ == "__main__":
    main()

