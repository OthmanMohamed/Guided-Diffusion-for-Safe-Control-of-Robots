#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pickle
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

class SafetyDataset(Dataset):
    def __init__(self, data_points):
        self.data_points = data_points
        
    def __len__(self):
        return len(self.data_points)
    
    def __getitem__(self, idx):
        point = self.data_points[idx]
        
        # Input: dynamics + action
        dynamics = torch.tensor(point['dynamics'], dtype=torch.float32)
        action = torch.tensor(point['action'], dtype=torch.float32)
        input_data = torch.cat([dynamics, action])
        
        # Target: safety score
        target = torch.tensor([point['safety_score']], dtype=torch.float32)
        
        return input_data, target

class SafetyPredictor(nn.Module):
    def __init__(self, input_dim=8, hidden_dims=[128, 64, 32]):
        super(SafetyPredictor, self).__init__()
        
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
        layers.append(nn.Sigmoid())  # Safety score between 0 and 1
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def load_data(train_file, val_file):
    """Load training and validation data"""
    print("Loading training data...")
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    
    print("Loading validation data...")
    with open(val_file, 'rb') as f:
        val_data = pickle.load(f)
    
    print(f"Training data points: {len(train_data)}")
    print(f"Validation data points: {len(val_data)}")
    
    return train_data, val_data

def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, device='cuda'):
    """Train the safety prediction model"""
    
    # Setup - use binary cross-entropy for binary classification
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Tensorboard writer
    writer = SummaryWriter('runs/safety_model_training')
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Log to tensorboard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 50)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'safety_model_best.pth')
            print(f'New best model saved! Val Loss: {best_val_loss:.4f}')
    
    writer.close()
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Binary Cross-Entropy)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves (Log Scale)')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    
    return train_losses, val_losses

def evaluate_model(model, val_loader, device='cuda'):
    """Evaluate the trained model"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for inputs, target in val_loader:
            inputs, target = inputs.to(device), target.to(device)
            output = model(inputs)
            predictions.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    
    # Convert predictions to binary (threshold at 0.5)
    binary_predictions = (predictions > 0.5).astype(int)
    binary_targets = targets.astype(int)
    
    # Calculate binary classification metrics
    accuracy = accuracy_score(binary_targets, binary_predictions)
    precision = precision_score(binary_targets, binary_predictions, zero_division=0)
    recall = recall_score(binary_targets, binary_predictions, zero_division=0)
    f1 = f1_score(binary_targets, binary_predictions, zero_division=0)
    
    # Calculate regression metrics for comparison
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    print(f"\nModel Evaluation (Binary Classification):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Mean Prediction: {np.mean(predictions):.4f}")
    print(f"Mean Target: {np.mean(targets):.4f}")
    
    # Print confusion matrix
    cm = confusion_matrix(binary_targets, binary_predictions)
    print(f"\nConfusion Matrix:")
    print(f"[[TN={cm[0,0]}, FP={cm[0,1]}]")
    print(f" [FN={cm[1,0]}, TP={cm[1,1]}]]")
    
    # Plot predictions vs targets
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
    plt.xlabel('True Safety Score')
    plt.ylabel('Predicted Safety Score')
    plt.title('Predictions vs Targets')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.hist(targets, bins=50, alpha=0.7, label='True', density=True)
    plt.hist(predictions, bins=50, alpha=0.7, label='Predicted', density=True)
    plt.xlabel('Safety Score')
    plt.ylabel('Density')
    plt.title('Distribution Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    errors = predictions - targets
    plt.hist(errors, bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Prediction Error Distribution')
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.plot(targets[:100], label='True', alpha=0.7)
    plt.plot(predictions[:100], label='Predicted', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Safety Score')
    plt.title('Sample Predictions')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Safe (0)', 'Unsafe (1)'], rotation=45)
    plt.yticks(tick_marks, ['Safe (0)', 'Unsafe (1)'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.subplot(2, 3, 6)
    # Plot binary predictions vs targets
    plt.scatter(binary_targets, binary_predictions, alpha=0.5)
    plt.xlabel('True Binary Safety')
    plt.ylabel('Predicted Binary Safety')
    plt.title('Binary Predictions vs Targets')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png')
    plt.close()

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_file = '/home/othman/turtlebotExp/safety_data/safety_data.pkl'
    val_file = '/home/othman/turtlebotExp/safety_data/safety_data_val.pkl'
    
    train_data, val_data = load_data(train_file, val_file)
    
    # Create datasets
    train_dataset = SafetyDataset(train_data)
    val_dataset = SafetyDataset(val_data)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create model for binary safety classification
    model = SafetyPredictor(input_dim=8, hidden_dims=[128, 64, 32]).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Training binary safety classifier (0=unsafe, 1=safe)")
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=500, lr=0.001, device=device
    )
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('safety_model_best.pth'))
    evaluate_model(model, val_loader, device)
    
    print("Binary safety classification training complete!")

if __name__ == "__main__":
    main() 