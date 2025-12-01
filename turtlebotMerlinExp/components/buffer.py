from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import numpy as np
from PIL import Image

class TurtlebotDataset(Dataset):
    def __init__(self, pkl_path, expert=True, load_images=False, use_jepa=False):
        self.load_images = load_images
        self.expert = expert
        self.use_jepa = use_jepa
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.dynamics = []
        self.actions = []
        self.goals = []
        self.horizons = []
        self.tei = []
        self.tsi = []

        if self.use_jepa: self.jepa = []
        
        for idx, (traj_dynamics, traj_actions, traj_goals) in enumerate(zip(data['o'], data['u'], data['g'])):
            if self.use_jepa: traj_jepa = data['jepa'][idx]
            traj_length = len(traj_dynamics)
            traj_start_index = len(self.dynamics)
            traj_end_index = len(self.dynamics) + len(traj_dynamics)-1
            for i, (dynamics, actions, goals) in enumerate(zip(traj_dynamics, traj_actions, traj_goals)):
                # Convert dynamics to numpy array if it's a list
                if isinstance(dynamics, list):
                    dynamics = np.array(dynamics)
                # Select specific indices from dynamics
                self.dynamics.append(dynamics[np.array([0, 1, 2, 3, 4, 5])])
                self.actions.append(actions)
                if self.expert:
                    self.goals.append(goals)
                else:
                    self.goals.append(traj_goals[-1])
                if self.use_jepa: self.jepa.append(traj_jepa[i])
                self.horizons.append([traj_length - i])
                self.tei.append(traj_end_index)
                self.tsi.append(traj_start_index)

    def __len__(self):
        return len(self.dynamics)

    def __getitem__(self, idx):
        if self.load_images:
            img_path = self.image_paths[idx].replace("expert_val", "expert_val_original") if "expert_val" in self.image_paths[idx] else self.image_paths[idx].replace("expert", "expert_original")
            image = Image.open(img_path.replace(".jpg", "_4.jpg")).convert("RGB")
            image = self.transform(image)
        else: 
            image = []

        if self.use_jepa:
            dynamics = torch.tensor(self.jepa[idx], dtype=torch.float32)
        else:
            dynamics = torch.tensor(self.dynamics[idx], dtype=torch.float32)
        actions = torch.tensor(self.actions[idx], dtype=torch.float32)

        if self.expert:
            goals = torch.tensor(self.goals[idx], dtype=torch.float32)
            horizons = torch.tensor(self.horizons[idx], dtype=torch.float32)
        else:
            # Hindsight relabeling: select a future state as the goal
            future_idx = min(idx + np.random.randint(1, 10), self.tei[idx])  # Random future state
            goals = torch.tensor(self.dynamics[future_idx][:2], dtype=torch.float32)
            horizons = torch.tensor([future_idx - idx], dtype=torch.float32)

        return image, dynamics, actions, goals, horizons

