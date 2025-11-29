#!/usr/bin/env python3
"""
Script 03: Train Safety Model

This script trains a neural network to predict safety scores given observations and actions.
The model takes the current state (observation) and action as input and outputs a safety score (0-1).
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('safety_model_training')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    log_file = os.path.join(log_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class SafetyDataset(Dataset):
    """Dataset for safety model training."""
    
    def __init__(self, data_path, split_ratio=0.8, is_train=True):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the pickle file containing safety data
            split_ratio: Ratio for train/validation split (only used if same file for both)
            is_train: Whether this is training or validation dataset
        """
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        observations = data['observations']
        actions = data['actions']
        safety_scores = data['safety_scores']
        
        # Convert to torch tensors
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)
        self.safety_scores = torch.FloatTensor(safety_scores).unsqueeze(1)  # Add channel dimension
        
        # Only split data if we're using the same file for both train and val
        # If separate files are provided, use all data from each file
        if split_ratio < 1.0:
            num_samples = len(self.observations)
            split_idx = int(num_samples * split_ratio)
            
            if is_train:
                self.observations = self.observations[:split_idx]
                self.actions = self.actions[:split_idx]
                self.safety_scores = self.safety_scores[:split_idx]
            else:
                self.observations = self.observations[split_idx:]
                self.actions = self.actions[split_idx:]
                self.safety_scores = self.safety_scores[split_idx:]
        
        print(f"Dataset {'train' if is_train else 'val'}: {len(self.observations)} samples")
        print(f"Safe actions: {torch.sum(self.safety_scores == 1.0).item()}")
        print(f"Unsafe actions: {torch.sum(self.safety_scores == 0.0).item()}")
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return {
            'observation': self.observations[idx],
            'action': self.actions[idx],
            'safety_score': self.safety_scores[idx]
        }


class SafetyModel(nn.Module):
    """Neural network for predicting safety scores."""
    
    def __init__(self, observation_dim=10, action_dim=4, hidden_dims=[128, 64, 32]):
        """
        Initialize the safety model.
        
        Args:
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super(SafetyModel, self).__init__()
        
        # Input dimensions
        input_dim = observation_dim + action_dim
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Output safety probability between 0 and 1
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, observation, action):
        """
        Forward pass.
        
        Args:
            observation: Observation tensor (batch_size, observation_dim)
            action: Action tensor (batch_size, action_dim)
        
        Returns:
            Safety score tensor (batch_size, 1)
        """
        # Concatenate observation and action
        x = torch.cat([observation, action], dim=1)
        return self.network(x)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, logger, writer):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    correct_predictions = 0
    total_predictions = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} - Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        observation = batch['observation'].to(device)
        action = batch['action'].to(device)
        safety_score = batch['safety_score'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(observation, action)
        loss = criterion(predictions, safety_score)
        
        # Calculate accuracy
        predicted_labels = (predictions > 0.5).float()
        correct_predictions += (predicted_labels == safety_score).sum().item()
        total_predictions += safety_score.size(0)
        
        # Backward pass
        loss.backward()
        
        # Track gradient norms
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Avg Loss': f"{total_loss/(batch_idx+1):.4f}",
            'Acc': f"{100*correct_predictions/total_predictions:.1f}%",
            'Grad': f"{total_norm:.3f}"
        })
        
        # Log to tensorboard
        global_step = epoch * num_batches + batch_idx
        writer.add_scalar('Training/Loss', loss.item(), global_step)
        writer.add_scalar('Training/Gradient_Norm', total_norm, global_step)
        
        # Log every 100 batches
        if batch_idx % 100 == 0:
            logger.info(f"Epoch {epoch+1}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}, Acc: {100*correct_predictions/total_predictions:.1f}%")
    
    avg_loss = total_loss / num_batches
    accuracy = 100 * correct_predictions / total_predictions
    
    # Log epoch-level metrics
    writer.add_scalar('Training/Epoch_Loss', avg_loss, epoch)
    writer.add_scalar('Training/Epoch_Accuracy', accuracy, epoch)
    
    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device, epoch, logger, writer):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    correct_predictions = 0
    total_predictions = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} - Validation")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            observation = batch['observation'].to(device)
            action = batch['action'].to(device)
            safety_score = batch['safety_score'].to(device)
            
            # Forward pass
            predictions = model(observation, action)
            loss = criterion(predictions, safety_score)
            
            # Calculate accuracy (binary classification)
            predicted_labels = (predictions > 0.5).float()
            correct_predictions += (predicted_labels == safety_score).sum().item()
            total_predictions += safety_score.size(0)
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{total_loss/(batch_idx+1):.4f}",
                'Acc': f"{100*correct_predictions/total_predictions:.1f}%"
            })
    
    avg_loss = total_loss / num_batches
    accuracy = 100 * correct_predictions / total_predictions
    
    # Log to tensorboard
    writer.add_scalar('Validation/Loss', avg_loss, epoch)
    writer.add_scalar('Validation/Accuracy', accuracy, epoch)
    
    logger.info(f"Epoch {epoch+1} - Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return avg_loss, accuracy


def plot_training_curves(train_losses, val_losses, save_path):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train safety model')
    parser.add_argument('--train-data', type=str, required=True,
                      help='Path to training data')
    parser.add_argument('--val-data', type=str, required=True,
                      help='Path to validation data')
    parser.add_argument('--output-dir', type=str, default='../models/safety',
                      help='Directory to save models and logs')
    parser.add_argument('--batch-size', type=int, default=512,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=500,
                      help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                      help='Learning rate')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 64, 64, 32],
                      help='Hidden layer dimensions')
    parser.add_argument('--save-interval', type=int, default=10,
                      help='Save model every N epochs')
    parser.add_argument('--random-seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"safety_model_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_dir = os.path.join(output_dir, 'logs')
    logger = setup_logging(log_dir)
    
    # Set up tensorboard
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    writer = SummaryWriter(tensorboard_dir)
    
    # Log training configuration
    logger.info("=== Safety Model Training Configuration ===")
    logger.info(f"Training data: {args.train_data}")
    logger.info(f"Validation data: {args.val_data}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Hidden dimensions: {args.hidden_dims}")
    logger.info(f"Random seed: {args.random_seed}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = SafetyDataset(args.train_data, split_ratio=1.0, is_train=True)
    val_dataset = SafetyDataset(args.val_data, split_ratio=1.0, is_train=False)
    
    # Check if datasets are not empty
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty!")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty!")
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    logger.info("Initializing model...")
    model = SafetyModel(
        observation_dim=10,  # FetchReach observation dimension
        action_dim=4,        # FetchReach action dimension
        hidden_dims=args.hidden_dims
    ).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop
    logger.info("Starting training...")
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 40  # Early stopping patience
    
    for epoch in range(args.epochs):
        logger.info(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        # Train
        train_loss, train_accuracy = train_epoch(model, train_dataloader, criterion, optimizer, device, epoch, logger, writer)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validate
        val_loss, val_accuracy = validate_epoch(model, val_dataloader, criterion, device, epoch, logger, writer)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'best_val_loss': best_val_loss,
            }, best_model_path)
            logger.info(f"New best model saved: {best_model_path} (Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%)")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Log epoch summary
        logger.info(f"Epoch {epoch+1} Summary:")
        logger.info(f"  Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        logger.info(f"  Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        logger.info(f"  Best Val Loss: {best_val_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
    }, final_model_path)
    logger.info(f"Final model saved: {final_model_path}")
    
    # Plot training curves
    plot_path = os.path.join(output_dir, "training_curves.png")
    plot_training_curves(train_losses, val_losses, plot_path)
    logger.info(f"Training curves saved: {plot_path}")
    
    # Log final results
    logger.info("\n=== Training Complete ===")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final training loss: {train_loss:.4f}, accuracy: {train_accuracy:.2f}%")
    logger.info(f"Final validation loss: {val_loss:.4f}, accuracy: {val_accuracy:.2f}%")
    logger.info(f"Tensorboard logs: {tensorboard_dir}")
    logger.info(f"All outputs saved to: {output_dir}")
    
    # Close tensorboard writer
    writer.close()


if __name__ == "__main__":
    main()

