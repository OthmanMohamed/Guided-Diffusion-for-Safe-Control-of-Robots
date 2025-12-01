import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.buffer import TurtlebotDataset
from torch.distributions import Normal, Independent
from components.policy import Policy
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train offline policy')
    
    # Training Configuration
    parser.add_argument('--train_on_images', action='store_true', default=False, 
                      help='Train on images instead of dynamics')
    parser.add_argument('--epochs', type=int, default=10000, 
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2048, 
                      help='Batch size for training')
    parser.add_argument('--log_interval', type=int, default=10, 
                      help='How often to log training progress')
    
    # Learning Rate Parameters
    parser.add_argument('--lr', type=float, default=1e-4, 
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, 
                      help='Weight decay for AdamW optimizer')
    
    # Data Paths
    parser.add_argument('--train_data_path', type=str, 
                      default="../data/demonstration/demonstration_data.pkl",
                      help='Path to training data pickle file')
    parser.add_argument('--val_data_path', type=str, 
                      default="../data/demonstration/demonstration_data.pkl",
                      help='Path to validation data pickle file')
    
    # Expert Settings
    parser.add_argument('--train_expert', action='store_true', default=False,
                      help='Use expert data for training')
    parser.add_argument('--val_expert', action='store_true', default=False,
                      help='Use expert data for validation')
    
    # JEPA Settings
    parser.add_argument('--use_jepa', action='store_true', default=False,
                      help='Use JEPA embeddings for both training and validation')
    
    # Model Saving
    parser.add_argument('--model_dir', type=str, 
                      default="../models/policy",
                      help='Directory to save model checkpoints')
    parser.add_argument('--save_interval', type=int, default=100, 
                      help='How often to save model checkpoints')
    
    # Evaluation
    parser.add_argument('--eval_interval', type=int, default=20000, 
                      help='How often to run evaluation')
    
    # Data Loading
    parser.add_argument('--num_workers', type=int, default=4, 
                      help='Number of workers for data loading')
    
    # Loss Function Parameters
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, 
                      help='Maximum gradient norm for clipping')
    
    return parser.parse_args()

def timestep_embedding(timesteps, dim=32, max_period=50):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    if len(timesteps.shape) == 1:
        args = timesteps.float() * freqs
    else:
        args = timesteps.float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def linear_lr_schedule(step, total_steps, initial_lr, final_lr, warmup_steps=5000):
    if step < warmup_steps:
        return initial_lr * (step / warmup_steps)  # Linear warmup
    else:
        return initial_lr + (final_lr - initial_lr) * ((step - warmup_steps) / (total_steps - warmup_steps))

from torch.utils.data import DataLoader

def get_dataloader(pkl_path, expert, use_jepa, batch_size=256, num_workers=4):
    dataset = TurtlebotDataset(pkl_path, expert=expert, load_images=False, use_jepa=use_jepa)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

def mu_sigma_loss(mus, sigmas, actions):
    dist = Independent(Normal(mus, sigmas), 1)
    log_p = -dist.log_prob(actions)
    loss = log_p.mean()
    return loss

def mse_loss(mus, actions):
    weights = torch.tensor([1.0, 1.0]).to("cuda")
    per_dim_loss = weights * (mus - actions) ** 2
    loss = per_dim_loss.mean()  # Scalar overall loss
    per_dim_loss_mean = per_dim_loss.mean(dim=0)
    return loss, per_dim_loss_mean

def main():
    args = parse_args()
    
    # Create log directory
    log_dir = "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    # Create model directory
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy(args.train_on_images).to(device)
    # policy.load_state_dict(torch.load("/home/othman/turtlebotExp/models/random_no_obstacles/lr1e-3/continued/model_final.pth"))

    optimizer = optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = get_dataloader(args.train_data_path, expert=args.train_expert, use_jepa=args.use_jepa, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = get_dataloader(args.val_data_path, expert=args.val_expert, use_jepa=args.use_jepa, batch_size=args.batch_size, num_workers=args.num_workers)

    total_steps = (len(train_loader)) * args.epochs

    # Training loop
    global_step = 0 
    for epoch in range(0, args.epochs):
        policy.train()
        running_loss = 0.0
        mse_running_loss = 0.0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch_idx, (image, dynamics, actions, goals, horizons) in enumerate(train_loader):
                if args.train_on_images: 
                    image = image.to(device)
                else: 
                    dynamics = torch.from_numpy(np.array(dynamics, dtype=np.float32)).to(device)
                horizons = torch.from_numpy(np.array(horizons, dtype=np.float32)).to(device)
                actions = torch.from_numpy(np.array(actions, dtype=np.float32)).to(device)
                goals = torch.from_numpy(np.array(goals, dtype=np.float32)).to(device)

                optimizer.zero_grad()
                if args.train_on_images:
                    mus, sigmas = policy(image, goals, horizons)
                else:
                    mus, sigmas = policy(dynamics, goals, horizons)

                loss = mu_sigma_loss(mus, sigmas, actions)
                m_s_e_loss, per_dim_m_s_e_loss = mse_loss(mus, actions)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=args.grad_clip_norm)
                optimizer.step()

                running_loss += loss.item()
                mse_running_loss += m_s_e_loss.item()
                pbar.update(1)
                global_step += 1
                
                if (batch_idx + 1) % args.log_interval == 0:
                    avg_loss = running_loss / args.log_interval
                    mse_avg_loss = mse_running_loss / args.log_interval
                    pbar.set_postfix(loss=avg_loss)
                    writer.add_scalar("Loss/Batch", avg_loss, global_step)
                    writer.add_scalar("Loss_MSE/Batch", mse_avg_loss, global_step)
                    writer.add_scalar("Learning Rate", args.lr, global_step)
                    writer.add_scalar("Loss_MSE/Dimension1", per_dim_m_s_e_loss[0].item(), global_step)
                    writer.add_scalar("Loss_MSE/Dimension2", per_dim_m_s_e_loss[1].item(), global_step)
                    total_norm = 0
                    for p in policy.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.norm(2).item()
                            total_norm += param_norm ** 2
                    total_norm = total_norm ** 0.5  # L2 norm
                    writer.add_scalar("Grad_Norm", total_norm, global_step)
                    print(avg_loss)
                    running_loss = 0.0
                    mse_running_loss = 0.0

            writer.add_scalar("Loss/Epoch", running_loss / len(train_loader), epoch)
            if epoch % args.save_interval == 0:
                checkpoint_path = os.path.join(args.model_dir, f"model_{epoch}.pth")
                torch.save(policy.state_dict(), checkpoint_path)

        policy.eval()
        val_loss = 0.0
        mse_val_loss = 0.0
        val_loss_per_dim = torch.zeros(2, device=device)
        mse_val_loss_per_dim = torch.zeros(2, device=device)
        val_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_idx, (image, dynamics, actions, goals, horizons) in enumerate(val_loader):
                (actions, goals, horizons, dynamics) = actions.to(device), goals.to(device), horizons.to(device), dynamics.to(device)
                if args.train_on_images: 
                    mus, sigmas = policy(image, goals, horizons)
                else: 
                    mus, sigmas = policy(dynamics, goals, horizons)
                loss  = mu_sigma_loss(mus, sigmas, actions)
                m_s_e_loss, per_dim_m_s_e_loss  = mse_loss(mus, actions)
                val_loss += loss.item()
                mse_val_loss += m_s_e_loss.item()
                mse_val_loss_per_dim += per_dim_m_s_e_loss

        avg_val_loss = val_loss / val_batches
        mse_avg_val_loss = mse_val_loss / val_batches
        avg_val_loss_per_dim = val_loss_per_dim / val_batches
        mse_avg_val_loss_per_dim = mse_val_loss_per_dim / val_batches
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Loss_MSE/Validation", mse_avg_val_loss, epoch)
        writer.add_scalar("Loss_MSE/Validation_Dimension1", mse_avg_val_loss_per_dim[0].item(), epoch)
        writer.add_scalar("Loss_MSE/Validation_Dimension2", mse_avg_val_loss_per_dim[1].item(), epoch)

        print(f"Epoch {epoch+1}/{args.epochs} completed.")
        print(f"Final training loss: {running_loss / len(train_loader):.4f}")

        # if epoch % args.eval_interval == 0:
        #     evaluator = ModelEvaluator(
        #         model_path=os.path.join(args.model_dir, f"model_{epoch}.pth"),
        #         use_images=args.train_on_images
        #     )
        #     evaluator.run_evaluation()

    # Save final model
    final_model_path = os.path.join(args.model_dir, "model_final.pth")
    torch.save(policy.state_dict(), final_model_path)
    print(f"Final model saved to '{final_model_path}'")

    writer.close()

if __name__ == "__main__":
    main()