import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Tuple, List
import logging
import time
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from typing import Dict, Tuple, List
import os
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging
import time
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
# from trajectory_dataset import TrajectoryDataset


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        max_seq_length: int = 20,
        max_ep_len: int = 1000,
        scale_rewards: bool = True,
        image_size: Tuple[int, int] = (1056, 1056),
        root_dir: str = None,
        action_dim: int = 8,
        action_encoding: str = 'onehot'  # 'onehot' or 'embedding'
    ):
        """
        Initialize the TrajectoryDataset.
        
        Args:
            csv_path (str): Path to the CSV file containing trajectories
            max_seq_length (int): Maximum sequence length for transformer input
            max_ep_len (int): Maximum episode length
            scale_rewards (bool): Whether to scale rewards
            image_size (tuple): Size of input images (height, width)
            root_dir (str): Root directory for image paths if they're relative in CSV
            action_dim (int): Dimension of the action space
            action_encoding (str): Type of action encoding ('onehot' or 'embedding')
        """
        super().__init__()
        self.max_seq_length = max_seq_length
        self.max_ep_len = max_ep_len
        self.scale_rewards = scale_rewards
        self.image_size = image_size
        self.root_dir = root_dir if root_dir else ''
        self.action_dim = action_dim
        self.action_encoding = action_encoding
        
        # Load and process the CSV file
        self.df = pd.read_csv(csv_path)
        
        # Verify required columns exist
        required_columns = ['episode_id', 'image_path', 'action', 'reward']
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")
        
        # Verify actions are integers
        if not np.issubdtype(self.df['action'].dtype, np.integer):
            try:
                self.df['action'] = self.df['action'].astype(int)
            except:
                raise ValueError("Actions must be convertible to integers")
        
        # Verify action values are within valid range
        if (self.df['action'].min() < 0) or (self.df['action'].max() >= action_dim):
            raise ValueError(f"Action values must be in range [0, {action_dim-1}]")
        
        # Group by episode
        self.episodes = list(self.df.groupby('episode_id'))
        
        # Scale rewards if needed
        self.reward_scale = 1.0
        if scale_rewards:
            all_rewards = self.df['reward'].values
            self.reward_scale = 1.0 / (np.std(all_rewards) + 1e-6)
        
        # Setup image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def _encode_action(self, action: int) -> torch.Tensor:
        """
        Encode integer action into appropriate format.
        
        Args:
            action (int): Integer action value
        
        Returns:
            torch.Tensor: Encoded action
        """
        if self.action_encoding == 'onehot':
            encoded = torch.zeros(self.action_dim)
            encoded[action] = 1.0
            return encoded
        elif self.action_encoding == 'embedding':
            # Return the action index to be embedded by the model
            return torch.tensor(action, dtype=torch.long)
        else:
            raise ValueError(f"Unknown action encoding: {self.action_encoding}")
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess an image from the given path."""
        full_path = os.path.join(self.root_dir, image_path)
        try:
            image = Image.open(full_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            raise RuntimeError(f"Error loading image {full_path}: {str(e)}")
    
    def _compute_returns_to_go(self, rewards: np.ndarray, gamma: float = 0.99) -> np.ndarray:
        """Compute discounted returns-to-go for a sequence of rewards."""
        returns = np.zeros_like(rewards)
        returns[-1] = rewards[-1]
        for t in reversed(range(len(rewards)-1)):
            returns[t] = rewards[t] + gamma * returns[t+1]
        return returns
    
    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single episode from the dataset.
        
        Returns:
            tuple: (states, actions, rewards, returns_to_go, timesteps, attention_mask)
        """
        # Get episode data
        episode_id, episode_data = self.episodes[idx]
        
        # Sort by timestep if necessary
        if 'timestep' in episode_data.columns:
            episode_data = episode_data.sort_values('timestep')
        
        # Load images for the episode
        states = torch.stack([
            self._load_image(img_path) 
            for img_path in episode_data['image_path']
        ])
        
        # Convert integer actions to appropriate format
        actions = torch.stack([
            self._encode_action(action) 
            for action in episode_data['action']
        ])
        
        # Convert rewards to tensor
        rewards = torch.tensor(episode_data['reward'].values)
        
        # Scale rewards if needed
        if self.scale_rewards:
            rewards = rewards * self.reward_scale
        
        # Calculate returns-to-go
        returns_to_go = torch.tensor(
            self._compute_returns_to_go(episode_data['reward'].values)
        )
        if self.scale_rewards:
            returns_to_go = returns_to_go * self.reward_scale
        
        # Create timesteps
        timesteps = torch.arange(len(rewards))
        
        # Handle sequences longer than max_seq_length
        if states.shape[0] > self.max_seq_length:
            # Sample random window
            start_idx = np.random.randint(0, states.shape[0] - self.max_seq_length)
            end_idx = start_idx + self.max_seq_length
            
            states = states[start_idx:end_idx]
            actions = actions[start_idx:end_idx]
            rewards = rewards[start_idx:end_idx]
            returns_to_go = returns_to_go[start_idx:end_idx]
            timesteps = timesteps[start_idx:end_idx] - timesteps[start_idx]
        
        # Create attention mask
        attention_mask = torch.ones(self.max_seq_length, dtype=torch.long)
        attention_mask[states.shape[0]:] = 0
        
        # Pad sequences if needed
        if states.shape[0] < self.max_seq_length:
            pad_length = self.max_seq_length - states.shape[0]
            
            # Create padding
            state_padding = torch.zeros((pad_length, *states.shape[1:]), dtype=states.dtype)
            action_padding = torch.zeros((pad_length, *actions.shape[1:]), dtype=actions.dtype)
            reward_padding = torch.zeros(pad_length, dtype=rewards.dtype)
            returns_padding = torch.zeros(pad_length, dtype=returns_to_go.dtype)
            timestep_padding = torch.zeros(pad_length, dtype=timesteps.dtype)
            
            # Concatenate padding
            states = torch.cat([states, state_padding], dim=0)
            actions = torch.cat([actions, action_padding], dim=0)
            rewards = torch.cat([rewards, reward_padding], dim=0)
            returns_to_go = torch.cat([returns_to_go, returns_padding], dim=0)
            timesteps = torch.cat([timesteps, timestep_padding], dim=0)
        
        return states, actions, rewards, returns_to_go, timesteps, attention_mask
    

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_csv: str,
        val_split: float = 0.2,
        optimizer: torch.optim.Optimizer = None,
        batch_size: int = 32,
        num_epochs: int = 100,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        max_seq_length: int = 20,
        action_dim: int = 8,
        action_encoding: str = 'onehot',
        checkpoint_dir: str = 'checkpoints',
        image_root_dir: str = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.action_encoding = action_encoding
        
        # Create full dataset
        self.full_dataset = TrajectoryDataset(
            csv_path=train_csv,
            max_seq_length=max_seq_length,
            action_dim=action_dim,
            action_encoding=action_encoding,
            root_dir=image_root_dir
        )
        
        # Split into train and validation sets
        val_size = int(len(self.full_dataset) * val_split)
        train_size = len(self.full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            self.full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        self.model = model.to(device)
        
        # Create optimizer if not provided
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.checkpoint_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def compute_loss(self, action_preds: torch.Tensor, actions: torch.Tensor, 
                    attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss based on action encoding type.
        """
        if self.action_encoding == 'onehot':
            # For one-hot actions, use MSE loss
            action_mask = attention_mask.unsqueeze(-1).repeat(1, 1, action_preds.shape[-1])
            loss = F.mse_loss(
                action_preds * action_mask,
                actions * action_mask,
                reduction='mean'
            )
        else:  # embedding
            # For embedded actions, use cross entropy loss
            action_mask = attention_mask
            loss = F.cross_entropy(
                action_preds.view(-1, action_preds.size(-1)),
                actions.view(-1),
                reduction='none'
            )
            loss = (loss * action_mask.view(-1)).mean()
        
        return loss
    
    def train_epoch(self) -> float:
        self.model.train()
        epoch_losses = []
        
        for batch in tqdm(self.train_loader, desc="Training"):
            states, actions, rewards, returns_to_go, timesteps, attention_mask = [
                b.to(self.device) for b in batch
            ]
            
            # Forward pass
            action_preds = self.model(
                states, actions, rewards, returns_to_go, timesteps,
                attention_mask=attention_mask
            )
            
            # Calculate loss
            loss = self.compute_loss(action_preds, actions, attention_mask)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
        
        return np.mean(epoch_losses)
    
    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        val_losses = []
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            states, actions, rewards, returns_to_go, timesteps, attention_mask = [
                b.to(self.device) for b in batch
            ]
            
            action_preds = self.model(
                states, actions, rewards, returns_to_go, timesteps,
                attention_mask=attention_mask
            )
            
            loss = self.compute_loss(action_preds, actions, attention_mask)
            val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'action_encoding': self.action_encoding
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.action_encoding = checkpoint['action_encoding']
        return checkpoint['epoch']
    
    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.checkpoint_dir / 'loss_plot.png')
        plt.close()
    
    def train(self):
        self.logger.info(f"Starting training on device: {self.device}")
        self.logger.info(f"Training set size: {len(self.train_dataset)}")
        self.logger.info(f"Validation set size: {len(self.val_dataset)}")
        self.logger.info(f"Action encoding: {self.action_encoding}")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            epoch_time = time.time() - epoch_start_time
            
            # Logging
            self.logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss)
            
            # Plot losses
            if (epoch + 1) % 10 == 0:
                self.plot_losses()
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.plot_losses()

def main():
    """Example usage of the training script."""
    from decision_transformer import DecisionTransformer
    
    # Initialize model
    model = DecisionTransformer(
        state_dim=1056*1056*3,
        action_dim=8,
        hidden_dim=128,
        max_seq_length=20,
        max_ep_len=1000,
        # action_encoding='onehot'  # or 'embedding'
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_csv='/home/jupyter-msiper/bootstrapping-pcgrl/data/loderunner/images/labels.csv',
        val_split=0.2,
        batch_size=32,
        num_epochs=100,
        action_dim=8,
        action_encoding='onehot',
        checkpoint_dir='dt_checkpoints',
        image_root_dir='/home/jupyter-msiper/bootstrapping-pcgrl/data/loderunner/images/'
    )
    
    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()