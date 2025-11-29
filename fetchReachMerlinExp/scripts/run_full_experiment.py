#!/usr/bin/env python3
"""
Main Script: Run Full Experiment

This script runs the complete pipeline for the FetchReach Merlin experiment with safety:
1. Collect demonstration data
2. Collect safety data
3. Train safety model
4. Train Merlin policy
5. Evaluate policy (with and without safety)
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
    parser = argparse.ArgumentParser(description='Run full FetchReach Merlin experiment with safety')
    
    # Data collection arguments
    parser.add_argument('--skip-data-collection', action='store_true',
                      help='Skip demonstration data collection (use existing data)')
    parser.add_argument('--skip-safety-data', action='store_true',
                      help='Skip safety data collection (use existing data)')
    parser.add_argument('--num-demo-episodes', type=int, default=2000,
                      help='Number of demonstration episodes to collect')
    parser.add_argument('--num-safety-episodes', type=int, default=1000,
                      help='Number of safety data episodes to collect')
    
    # Training arguments
    parser.add_argument('--skip-safety-training', action='store_true',
                      help='Skip safety model training (use existing model)')
    parser.add_argument('--skip-policy-training', action='store_true',
                      help='Skip policy training (use existing model)')
    parser.add_argument('--max-timesteps', type=int, default=50000,
                      help='Maximum training timesteps for policy')
    
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
    print("FetchReach Merlin Experiment with Safety")
    print("="*60)
    print(f"Base directory: {base_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Models directory: {models_dir}")
    print(f"Logs directory: {logs_dir}")
    print("="*60)
    
    # Step 1: Collect demonstration data
    demo_data_path = demo_data_dir / 'buffer.pkl'
    if not args.skip_data_collection and not demo_data_path.exists():
        cmd = [
            sys.executable, 'scripts/01_collect_demonstration_data.py',
            '--num-episodes', str(args.num_demo_episodes),
            '--output-dir', str(demo_data_dir),
            '--seed', str(args.seed)
        ]
        if not run_command(cmd, "Collecting demonstration data"):
            return
    else:
        print(f"\nSkipping demonstration data collection (file exists: {demo_data_path.exists()})")
    
    # Step 2: Collect safety data
    safety_data_path = safety_data_dir / 'safety_data_final.pkl'
    if not args.skip_safety_data and not safety_data_path.exists():
        cmd = [
            sys.executable, 'scripts/02_collect_safety_data.py',
            '--num-episodes', str(args.num_safety_episodes),
            '--output-dir', str(safety_data_dir),
            '--seed', str(args.seed)
        ]
        if not run_command(cmd, "Collecting safety data"):
            return
    else:
        print(f"\nSkipping safety data collection (file exists: {safety_data_path.exists()})")
    
    # Step 3: Train safety model
    # Find the most recent safety model or train a new one
    safety_model_path = None
    if not args.skip_safety_training:
        # Look for existing safety models
        safety_model_dirs = list(safety_models_dir.glob('safety_model_*'))
        if safety_model_dirs:
            # Use the most recent one
            safety_model_path = max(safety_model_dirs, key=os.path.getctime) / 'best_model.pt'
            print(f"\nFound existing safety model: {safety_model_path}")
        else:
            # Train new safety model
            # Note: This assumes you have separate train/val safety data
            # You may need to split the safety data or provide separate files
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
                    '--output-dir', str(safety_models_dir),
                    '--random-seed', str(args.seed)
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
            '--dataset-path', str(demo_data_path),
            '--max-timesteps', str(args.max_timesteps),
            '--seed', str(args.seed),
            '--obstacles'
        ]
        if not run_command(cmd, "Training Merlin policy"):
            return
        
        # Find the most recent checkpoint
        checkpoint_dir = logs_dir / 'checkpoints'
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.rglob('best_policy_*.pt'))
            if checkpoints:
                policy_checkpoint_path = max(checkpoints, key=os.path.getctime)
                print(f"\nPolicy checkpoint saved: {policy_checkpoint_path}")
    else:
        # Find existing checkpoint
        checkpoint_dir = logs_dir / 'checkpoints'
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.rglob('best_policy_*.pt'))
            if checkpoints:
                policy_checkpoint_path = max(checkpoints, key=os.path.getctime)
                print(f"\nUsing existing policy checkpoint: {policy_checkpoint_path}")
    
    # Step 5: Evaluate policy
    if not args.skip_evaluation and policy_checkpoint_path:
        # Extract step number from checkpoint name
        checkpoint_name = policy_checkpoint_path.stem
        step = int(checkpoint_name.split('_')[-1]) if '_' in checkpoint_name else 500
        
        # Evaluate without safety
        print(f"\n{'='*60}")
        print("Evaluating policy WITHOUT safety")
        print(f"{'='*60}\n")
        cmd = [
            sys.executable, 'scripts/05_evaluate_policy.py',
            '--checkpoint', str(policy_checkpoint_path),
            '--num-episodes', str(args.num_eval_episodes),
            '--seed', str(args.seed),
            '--step', str(step)
        ]
        run_command(cmd, "Evaluating policy without safety")
        
        # Evaluate with safety
        if safety_model_path and safety_model_path.exists():
            print(f"\n{'='*60}")
            print("Evaluating policy WITH safety")
            print(f"{'='*60}\n")
            cmd = [
                sys.executable, 'scripts/06_evaluate_policy_with_safety.py',
                '--checkpoint', str(policy_checkpoint_path),
                '--safety-model', str(safety_model_path),
                '--num-episodes', str(args.num_eval_episodes),
                '--seed', str(args.seed),
                '--step', str(step)
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

