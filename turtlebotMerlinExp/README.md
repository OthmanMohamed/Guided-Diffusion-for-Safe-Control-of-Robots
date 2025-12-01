# TurtleBot Merlin Experiment with Safety

This directory contains a complete pipeline for training and evaluating a Merlin policy with safety integration for the TurtleBot3 robot in ROS/Gazebo simulation.

## Overview

This experiment implements a safe robot control system using:
1. **Merlin Policy**: An offline goal-conditioned policy trained on demonstration data from ROS simulation
2. **Safety Model**: A neural network that predicts safety scores for state-action pairs
3. **Integration**: Evaluation of the policy with safety monitoring

The pipeline consists of several stages:
1. **Data Collection**: Collect demonstration data and safety-labeled data from ROS simulation
2. **Safety Model Training**: Train a model to predict safety scores
3. **Policy Training**: Train the Merlin policy on demonstration data
4. **Evaluation**: Evaluate the policy with and without safety integration

## Directory Structure

```
turtlebotMerlinExp/
├── scripts/                    # Main experiment scripts
│   ├── 01_collect_demonstration_data.py
│   ├── 02_collect_safety_data.py
│   ├── 03_train_safety_model.py
│   ├── 04_train_merlin_policy.py
│   ├── 05_evaluate_policy.py
│   ├── 06_evaluate_policy_with_safety.py
│   └── run_full_experiment.py  # Main script to run entire pipeline
├── components/                  # Reusable components
│   ├── policy.py               # Policy network architecture
│   └── buffer.py               # Dataset/buffer for offline RL
├── data/                       # Data storage (created during execution)
│   ├── demonstration/         # Demonstration data
│   └── safety/                # Safety-labeled data
├── models/                     # Model storage (created during execution)
│   ├── safety/                 # Trained safety models
│   └── policy/                 # Trained policy models
├── logs/                       # Training logs and checkpoints
└── README.md                   # This file
```

## Prerequisites

### ROS and Gazebo Setup

This experiment requires ROS and Gazebo to be running with the TurtleBot3 simulation. Before running data collection or evaluation scripts, ensure:

1. **ROS Installation**: ROS Noetic (or compatible version) is installed and sourced
   ```bash
   source /opt/ros/noetic/setup.bash
   ```

2. **TurtleBot3 Setup**: TurtleBot3 packages are installed
   ```bash
   export TURTLEBOT3_MODEL=burger
   ```

3. **Gazebo Simulation**: Launch the TurtleBot3 simulation
   ```bash
   roslaunch turtlebot3_gazebo turtlebot3_world.launch
   ```

4. **Required ROS Topics**: The following topics should be available:
   - `/odom` - Odometry data
   - `/cmd_vel` - Velocity commands
   - `/imu` - IMU data for collision detection
   - `/overhead_camera/overhead_camera/image_raw` - Camera images (optional)

### Python Packages

Install required Python packages:
```bash
pip install torch torchvision numpy matplotlib scikit-learn tensorboard opencv-python tqdm transformers
```

For ROS Python packages:
```bash
pip install rospkg cv-bridge
```

## Pipeline Stages

### Stage 1: Collect Demonstration Data

**Script**: `scripts/01_collect_demonstration_data.py`

Collects demonstration trajectories from the TurtleBot ROS simulation. The data includes:
- Observations (dynamics: x, y, linear_vel, angular_vel, sin(theta), cos(theta))
- Actions (linear velocity, angular velocity)
- Goals
- Achieved goals

**Usage**:
```bash
# Make sure ROS and Gazebo are running first!
python scripts/01_collect_demonstration_data.py \
    --num-trials 3000 \
    --output-filename data/demonstration/demonstration_data.pkl \
    --use-obstacles
```

**Arguments**:
- `--num-trials`: Number of trials to collect (default: 3000)
- `--save-images`: Save camera images (optional)
- `--plot-trajectories`: Save trajectory plots (default: True)
- `--use-obstacles`: Enable obstacle avoidance (default: True)
- `--expert-mode`: Use expert demonstrations (False for random exploration)
- `--output-filename`: Output file path

**Output**: `data/demonstration/demonstration_data.pkl`

### Stage 2: Collect Safety Data

**Script**: `scripts/02_collect_safety_data.py`

Collects safety-labeled data by running episodes and labeling actions as safe (1.0) or unsafe (0.0) based on collisions. Steps before collisions are also labeled as unsafe.

**Usage**:
```bash
# Make sure ROS and Gazebo are running first!
python scripts/02_collect_safety_data.py \
    --num-trajectories 500 \
    --output-dir data/safety \
    --safety-horizon 5
```

**Arguments**:
- `--num-trajectories`: Number of trajectories to collect (default: 500)
- `--output-dir`: Directory to save data (default: data/safety)
- `--safety-horizon`: Number of steps for safety prediction (default: 5)

**Output**: `data/safety/safety_data_val.pkl`

### Stage 3: Train Safety Model

**Script**: `scripts/03_train_safety_model.py`

Trains a neural network to predict safety scores (0-1) given observations and actions. The model uses a binary classification approach.

**Usage**:
```bash
python scripts/03_train_safety_model.py \
    --train-data data/safety/safety_data_val.pkl \
    --val-data data/safety/safety_data_val.pkl \
    --output-dir models/safety \
    --epochs 100 \
    --batch-size 512 \
    --learning-rate 0.001
```

**Arguments**:
- `--train-data`: Path to training safety data
- `--val-data`: Path to validation safety data
- `--output-dir`: Directory to save models
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 512)
- `--learning-rate`: Learning rate (default: 0.001)

**Output**: `models/safety/safety_model_<timestamp>/best_model.pt`

### Stage 4: Train Merlin Policy

**Script**: `scripts/04_train_merlin_policy.py`

Trains the Merlin policy using offline demonstration data. The policy learns to reach goals in the TurtleBot environment.

**Usage**:
```bash
python scripts/04_train_merlin_policy.py \
    --train_data_path data/demonstration/demonstration_data.pkl \
    --val_data_path data/demonstration/demonstration_data.pkl \
    --epochs 10000 \
    --batch_size 2048 \
    --model_dir models/policy
```

**Arguments**:
- `--train_data_path`: Path to training demonstration data
- `--val_data_path`: Path to validation data
- `--epochs`: Number of training epochs (default: 10000)
- `--batch_size`: Batch size (default: 2048)
- `--lr`: Learning rate (default: 1e-4)
- `--model_dir`: Directory to save checkpoints
- `--train_expert`: Use expert data mode
- `--use_jepa`: Use JEPA embeddings (requires JEPA model)

**Output**: 
- Checkpoints: `models/policy/model_<step>.pth`

### Stage 5: Evaluate Policy

**Script**: `scripts/05_evaluate_policy.py`

Evaluates the trained policy without safety integration. Measures:
- Discounted return
- Undiscounted return
- Success rate
- Collision rate

**Usage**:
```bash
# Make sure ROS and Gazebo are running first!
python scripts/05_evaluate_policy.py \
    --model_path models/policy/model_10000.pth \
    --num_episodes 100
```

**Arguments**:
- `--model_path`: Path to trained policy model
- `--num_episodes`: Number of evaluation episodes (default: 100)
- `--use_images`: Use image observations
- `--jepa_model_path`: Path to JEPA model (if using JEPA)

### Stage 6: Evaluate Policy with Safety

**Script**: `scripts/06_evaluate_policy_with_safety.py`

Evaluates the trained policy with safety model integration. Provides additional metrics:
- Average safety score
- Safety analysis
- Correlation between collisions and safety scores

**Usage**:
```bash
# Make sure ROS and Gazebo are running first!
python scripts/06_evaluate_policy_with_safety.py \
    --model_path models/policy/model_10000.pth \
    --safety_model_path models/safety/safety_model_<timestamp>/best_model.pt \
    --num_episodes 100
```

**Arguments**:
- `--model_path`: Path to trained policy model
- `--safety_model_path`: Path to safety model
- `--num_episodes`: Number of evaluation episodes
- `--eval_with_gradient`: Use gradient-based safety optimization
- `--lambda`: Safety gradient weight (default: 0.1)

## Running the Full Pipeline

**Script**: `scripts/run_full_experiment.py`

Runs the complete pipeline from data collection to evaluation.

**Basic Usage**:
```bash
python scripts/run_full_experiment.py
```

**With Custom Arguments**:
```bash
python scripts/run_full_experiment.py \
    --num-demo-trials 3000 \
    --num-safety-trajectories 500 \
    --epochs 10000 \
    --num-eval-episodes 100 \
    --seed 42
```

**Skipping Stages**:
```bash
# Skip data collection (use existing data)
python scripts/run_full_experiment.py --skip-data-collection --skip-safety-data

# Skip training (use existing models)
python scripts/run_full_experiment.py --skip-safety-training --skip-policy-training

# Only evaluate
python scripts/run_full_experiment.py --skip-data-collection --skip-safety-data --skip-safety-training --skip-policy-training
```

## Key Components

### Policy (`components/policy.py`)
- Goal-conditioned policy network
- Uses timestep embeddings
- Supports both image and dynamics observations
- Outputs Gaussian action distributions

### Buffer (`components/buffer.py`)
- Dataset loader for offline learning
- Supports Hindsight Experience Replay (HER)
- Handles expert and random data modes
- Optional JEPA embedding support

### SafetyModel (`scripts/03_train_safety_model.py`)
- Binary classifier for safety prediction
- Takes observation-action pairs as input
- Outputs safety score (0-1)

## Data Format

### Demonstration Data
The demonstration data pickle file contains:
- `o`: List of trajectories, each containing observations (dynamics: [x, y, linear_vel, angular_vel, sin(theta), cos(theta)])
- `u`: List of trajectories, each containing actions ([linear_vel, angular_vel])
- `g`: List of trajectories, each containing goals ([goal_x, goal_y])
- `ag`: List of trajectories, each containing achieved goals ([x, y])

### Safety Data
The safety data pickle file contains a list of dictionaries, each with:
- `dynamics`: Observation vector [x, y, linear_vel, angular_vel, sin(theta), cos(theta)]
- `action`: Action vector [linear_vel, angular_vel]
- `safety_score`: Safety score (0.0 for unsafe, 1.0 for safe)

## Hyperparameters

### Policy Training
- Learning rate: 1e-4
- Batch size: 2048
- Weight decay: 1e-3
- Max path length: 50
- Hidden dimensions: 256

### Safety Model Training
- Learning rate: 0.001
- Batch size: 512
- Hidden dimensions: [128, 64, 32]
- Epochs: 100
- Early stopping patience: 40

## Troubleshooting

### Common Issues

1. **ROS Connection Errors**: 
   - Ensure ROS is sourced: `source /opt/ros/noetic/setup.bash`
   - Check that Gazebo simulation is running
   - Verify ROS topics are available: `rostopic list`

2. **Import Errors**: 
   - Make sure you're running scripts from the correct directory
   - Check that all paths are set correctly
   - Ensure components directory is in Python path

3. **Memory Issues**: 
   - Reduce batch sizes or number of episodes
   - Use smaller datasets for testing

4. **Missing Data**: 
   - Ensure data files exist before training
   - Use the data collection scripts first

5. **Model Loading Errors**: 
   - Check that checkpoint paths are correct
   - Verify model architecture matches saved checkpoint

## Notes

- **ROS Dependency**: Data collection and evaluation scripts require ROS and Gazebo to be running. Training scripts do not require ROS.
- **Simulation Environment**: The scripts assume a TurtleBot3 simulation with obstacles. Adjust obstacle definitions in the scripts if using a different environment.
- **JEPA Support**: Optional JEPA (Joint-Embedding Predictive Architecture) support is available for image-based observations. Requires a pre-trained JEPA model.

## Citation

If you use this code, please cite the original paper:
```
[Add citation when available]
```

## License

[Add license information]

## Contact

[Add contact information]

