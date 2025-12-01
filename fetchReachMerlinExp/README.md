# FetchReach Merlin Experiment with Safety

This directory contains a complete pipeline for training and evaluating a Merlin policy with safety integration for the FetchReach environment with obstacles.

## Overview

This experiment implements a safe robot control system using:
1. **Merlin Policy**: An offline goal-conditioned policy trained on demonstration data
2. **Safety Model**: A neural network that predicts safety scores for state-action pairs
3. **Integration**: Evaluation of the policy with safety monitoring

The pipeline consists of several stages:
1. **Data Collection**: Collect demonstration data and safety-labeled data
2. **Safety Model Training**: Train a model to predict safety scores
3. **Policy Training**: Train the Merlin policy on demonstration data
4. **Evaluation**: Evaluate the policy with and without safety integration

## Directory Structure

```
fetchReachMerlinExp/
├── scripts/                    # Main experiment scripts
│   ├── 01_collect_demonstration_data.py
│   ├── 02_collect_safety_data.py
│   ├── 03_train_safety_model.py
│   ├── 04_train_merlin_policy.py
│   ├── 05_evaluate_policy.py
│   ├── 06_evaluate_policy_with_safety.py
│   └── run_full_experiment.py  # Main script to run entire pipeline
├── components/                  # Reusable components
│   ├── buffer.py               # Replay buffer for offline RL
│   └── normalizer.py           # State normalization utilities
├── envs/                       # Environment wrappers
│   └── fetch_reach_with_obstacles.py
├── data/                       # Data storage (created during execution)
│   ├── demonstration/         # Demonstration data
│   └── safety/                # Safety-labeled data
├── models/                     # Model storage (created during execution)
│   ├── safety/                 # Trained safety models
│   └── policy/                 # Trained policy models
├── logs/                       # Training logs and checkpoints
└── README.md                   # This file
```

## Pipeline Stages

### Stage 1: Collect Demonstration Data

**Script**: `scripts/01_collect_demonstration_data.py`

Collects demonstration trajectories from the FetchReach environment with obstacles. The data includes:
- Observations (states)
- Actions
- Goals
- Achieved goals
- Horizon information (collision indicators)

**Usage**:
```bash
python scripts/01_collect_demonstration_data.py \
    --num-episodes 2000 \
    --output-dir data/demonstration \
    --seed 42
```

**Output**: `data/demonstration/buffer.pkl`

### Stage 2: Collect Safety Data

**Script**: `scripts/02_collect_safety_data.py`

Collects safety-labeled data by running episodes and labeling actions as safe (1.0) or unsafe (0.0) based on collisions. Steps before collisions are also labeled as unsafe.

**Usage**:
```bash
python scripts/02_collect_safety_data.py \
    --num-episodes 1000 \
    --unsafe-steps-before-collision 5 \
    --output-dir data/safety \
    --seed 42
```

**Output**: `data/safety/safety_data_final.pkl`

### Stage 3: Train Safety Model

**Script**: `scripts/03_train_safety_model.py`

Trains a neural network to predict safety scores (0-1) given observations and actions. The model uses a binary classification approach.

**Usage**:
```bash
python scripts/03_train_safety_model.py \
    --train-data data/safety/safety_data_final.pkl \
    --val-data data/safety/safety_data_val.pkl \
    --output-dir models/safety \
    --epochs 500 \
    --batch-size 512 \
    --learning-rate 0.0001
```

**Output**: `models/safety/safety_model_<timestamp>/best_model.pt`

### Stage 4: Train Merlin Policy

**Script**: `scripts/04_train_merlin_policy.py`

Trains the Merlin policy using offline demonstration data. The policy learns to reach goals in the FetchReach environment with obstacles.

**Usage**:
```bash
python scripts/04_train_merlin_policy.py \
    --dataset-path data/demonstration/buffer.pkl \
    --max-timesteps 50000 \
    --seed 42 \
    --obstacles
```

**Output**: 
- Checkpoints: `logs/checkpoints/best_policy_<step>.pt`
- Normalizers: `logs/checkpoints/o_norm_<step>.pt`, `logs/checkpoints/g_norm_<step>.pt`

### Stage 5: Evaluate Policy

**Script**: `scripts/05_evaluate_policy.py`

Evaluates the trained policy without safety integration. Measures:
- Discounted return
- Undiscounted return
- Success rate
- Collision rate

**Usage**:
```bash
python scripts/05_evaluate_policy.py \
    --checkpoint logs/checkpoints/best_policy_500.pt \
    --num-episodes 100 \
    --seed 42 \
    --step 500
```

### Stage 6: Evaluate Policy with Safety

**Script**: `scripts/06_evaluate_policy_with_safety.py`

Evaluates the trained policy with safety model integration. Provides additional metrics:
- Average safety score
- Safety analysis
- Correlation between collisions and safety scores

**Usage**:
```bash
python scripts/06_evaluate_policy_with_safety.py \
    --checkpoint logs/checkpoints/best_policy_500.pt \
    --safety-model models/safety/safety_model_<timestamp>/best_model.pt \
    --num-episodes 100 \
    --seed 42 \
    --step 500
```

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
    --num-demo-episodes 2000 \
    --num-safety-episodes 1000 \
    --max-timesteps 50000 \
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

## Requirements

### Python Packages
- `torch` - PyTorch for deep learning
- `gymnasium` - OpenAI Gym for environments
- `mujoco` - MuJoCo physics simulator
- `numpy` - Numerical computing
- `matplotlib` - Plotting
- `tqdm` - Progress bars
- `tensorboard` - Training visualization (optional)

### Environment Setup
1. Install MuJoCo and the FetchReach environment:
   ```bash
   pip install gymnasium[mujoco]
   pip install mujoco
   ```

2. Ensure you have the FetchReach environment available:
   ```python
   import gymnasium as gym
   env = gym.make("FetchReach-v2")
   ```

3. **Important: Adding Obstacles**
   
   Obstacles must be added directly to the FetchReach environment source files. 
   The `FetchReachWithObstacles` wrapper does not programmatically add obstacles.
   
   To add obstacles:
   - Locate the FetchReach environment XML/model files in your gymnasium installation
   - Modify the XML to include obstacle bodies and geoms in the `<worldbody>` section
   - Example obstacle definition:
     ```xml
     <body name="obstacle1" pos="1.25 0.75 0.42">
       <geom name="obstacle1_geom" type="box" size="0.025 0.025 0.2" 
             rgba="0.8 0.2 0.2 0.8" contype="1" conaffinity="1"/>
     </body>
     ```
   - After modifying the source files, the environment will automatically include obstacles

## Key Components

### ReplayBuffer (`components/buffer.py`)
- Stores trajectories for offline learning
- Supports Hindsight Experience Replay (HER)
- Samples batches for training

### Normalizer (`components/normalizer.py`)
- Normalizes observations and goals
- Maintains running statistics
- Supports saving/loading for evaluation

### FetchReachWithObstacles (`envs/fetch_reach_with_obstacles.py`)
- Wraps the standard FetchReach environment
- Note: Obstacles must be added directly to the environment source XML files
- Provides utilities for accessing obstacle information

### Policy (`scripts/04_train_merlin_policy.py`)
- Goal-conditioned policy network
- Uses timestep embeddings
- Outputs Gaussian action distributions

### SafetyModel (`scripts/03_train_safety_model.py`)
- Binary classifier for safety prediction
- Takes observation-action pairs as input
- Outputs safety score (0-1)

## Hyperparameters

### Policy Training
- Learning rate: 5e-4
- Batch size: 512
- HER ratio: 1.0 (for FetchReach)
- Test horizon: 1
- Max path length: 50

### Safety Model Training
- Learning rate: 0.0001
- Batch size: 512
- Hidden dimensions: [128, 64, 64, 32]
- Epochs: 500
- Early stopping patience: 40

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running scripts from the correct directory and that all paths are set correctly.

2. **MuJoCo Errors**: Ensure MuJoCo is properly installed and the FetchReach environment is available.

3. **Memory Issues**: Reduce batch sizes or number of episodes if you run out of memory.

4. **Missing Data**: Ensure data files exist before training. Use the data collection scripts first.

5. **Checkpoint Loading**: Make sure the step number matches the checkpoint name when loading normalizers.

## Citation

If you use this code, please cite the original paper:
```
[Add citation when available]
```

## License

[Add license information]

## Contact

[Add contact information]

