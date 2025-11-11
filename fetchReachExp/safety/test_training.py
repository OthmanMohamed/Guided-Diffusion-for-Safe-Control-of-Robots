#!/usr/bin/env python3
"""
Test script for safety model training.
This script runs a small training session to verify everything works correctly.
"""

import os
import sys
import torch
import numpy as np
from train_safety_model import SafetyModel, SafetyDataset, setup_logging
from torch.utils.data import DataLoader
import tempfile
import shutil

def create_test_data():
    """Create a small test dataset."""
    # Create synthetic data
    num_samples = 100
    observations = np.random.randn(num_samples, 10)  # 10-dim observations
    actions = np.random.randn(num_samples, 4)        # 4-dim actions
    safety_scores = np.random.choice([0.0, 1.0], num_samples)  # Binary safety scores
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    train_file = os.path.join(temp_dir, 'train_data.pkl')
    val_file = os.path.join(temp_dir, 'val_data.pkl')
    
    # Save training data
    import pickle
    train_data = {
        'observations': observations[:80],
        'actions': actions[:80],
        'safety_scores': safety_scores[:80]
    }
    with open(train_file, 'wb') as f:
        pickle.dump(train_data, f)
    
    # Save validation data
    val_data = {
        'observations': observations[80:],
        'actions': actions[80:],
        'safety_scores': safety_scores[80:]
    }
    with open(val_file, 'wb') as f:
        pickle.dump(val_data, f)
    
    return train_file, val_file, temp_dir

def test_training():
    """Test the training functionality."""
    print("Testing safety model training...")
    
    try:
        # Create test data
        train_file, val_file, temp_dir = create_test_data()
        
        # Set up logging
        log_dir = os.path.join(temp_dir, 'logs')
        logger = setup_logging(log_dir)
        
        # Load datasets
        train_dataset = SafetyDataset(train_file, split_ratio=1.0, is_train=True)
        val_dataset = SafetyDataset(val_file, split_ratio=1.0, is_train=False)
        
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Initialize model
        device = torch.device("cpu")  # Use CPU for testing
        model = SafetyModel(observation_dim=10, action_dim=4, hidden_dims=[32, 16]).to(device)
        
        # Loss and optimizer
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Test one training epoch
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_dataloader:
            observation = batch['observation'].to(device)
            action = batch['action'].to(device)
            safety_score = batch['safety_score'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(observation, action)
            loss = criterion(predictions, safety_score)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Test validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                observation = batch['observation'].to(device)
                action = batch['action'].to(device)
                safety_score = batch['safety_score'].to(device)
                
                predictions = model(observation, action)
                loss = criterion(predictions, safety_score)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        print("✅ Training test passed!")
        print(f"   Training loss: {avg_loss:.4f}")
        print(f"   Validation loss: {avg_val_loss:.4f}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test model inference
        test_obs = torch.randn(1, 10)
        test_action = torch.randn(1, 4)
        with torch.no_grad():
            safety_pred = model(test_obs, test_action)
        
        print(f"   Test prediction: {safety_pred.item():.4f}")
        assert 0 <= safety_pred.item() <= 1, "Safety prediction should be between 0 and 1"
        
        # Clean up
        shutil.rmtree(temp_dir)
        print("✅ Cleanup completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training()
    sys.exit(0 if success else 1) 