import torch
import torch.nn as nn
from torch.utils.data import Dataset
import h5py
import numpy as np

class BaseSensorDataset(Dataset):
    """Base dataset class with common error handling and data loading patterns"""
    def __init__(self, h5_file, split='train'):
        self.split = split
        self.h5_file = h5_file
        self._load_data()
    
    def _load_data(self):
        """Load data from HDF5 file with error handling"""
        try:
            with h5py.File(self.h5_file, 'r') as f:
                self._load_specific_data(f)
        except Exception as e:
            raise RuntimeError(f"Error loading {self.split} data from {self.h5_file}: {str(e)}")
    
    def _load_specific_data(self, f):
        """Override this method in subclasses to load specific data fields"""
        raise NotImplementedError("Subclasses must implement _load_specific_data")
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        """Override this method in subclasses to return specific data format"""
        raise NotImplementedError("Subclasses must implement __getitem__")

class SensorDataset(BaseSensorDataset):
    """Dataset for sensor readings and spatial bins"""
    def _load_specific_data(self, f):
        self.x = torch.FloatTensor(f[f'X_{self.split}'][:])  # (N, 30, 9)
        self.spatial_bins = torch.LongTensor(f[f'spatial_bins_{self.split}'][:])  # (N,)
    
    def __getitem__(self, idx):
        return self.x[idx], self.spatial_bins[idx]

class DistanceDataset(BaseSensorDataset):
    """Dataset for sensor readings with distances, optional fungi filter"""
    def __init__(self, h5_file, split='train', fungi_filter=None):
        self.fungi_filter = fungi_filter
        super().__init__(h5_file, split)
    
    def _load_specific_data(self, f):
        x = f[f'X_{self.split}'][:]                # (N, 30, 9)
        distances = f[f'distances_{self.split}'][:] # (N, window_size)
        fungi_labels = f[f'fungi_labels_{self.split}'][:]  # (N,)
        
        if self.fungi_filter is not None:
            mask = (fungi_labels == self.fungi_filter)
            x = x[mask]
            distances = distances[mask]
            fungi_labels = fungi_labels[mask]
            print(f"Filtered for fungi_label={self.fungi_filter}: {np.sum(mask)} samples remain.")
        else:
            print(f"No fungi filter: {len(x)} samples loaded.")
        
        self.x = torch.FloatTensor(x)
        self.distances = torch.FloatTensor(distances)
        self.distance_targets = self.distances[:, -1]
        self.fungi_labels = torch.LongTensor(fungi_labels)
    
    def __getitem__(self, idx):
        return {
            'x': self.x[idx],
            'distance': self.distance_targets[idx],
            'fungi_label': self.fungi_labels[idx]
        }

class FungiDataset(BaseSensorDataset):
    """Dataset for sensor readings and fungi species classification"""
    def _load_specific_data(self, f):
        self.x = torch.FloatTensor(f[f'X_{self.split}'][:])  # (N, 30, 9)
        self.fungi_types = torch.LongTensor(f[f'fungi_labels_{self.split}'][:])  # (N,)
        assert len(self.x) == len(self.fungi_types), "Mismatched data/labels length"
    
    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.fungi_types[idx]
        )

class BinSpecificLoss(nn.Module):
    """Custom loss that applies different weights/penalties for different bins"""
    def __init__(self, bin_difficulties, base_criterion=None, difficulty_mode='weight'):
        super().__init__()
        self.register_buffer('bin_difficulties', torch.tensor(bin_difficulties, dtype=torch.float32))
        self.base_criterion = base_criterion or nn.CrossEntropyLoss(reduction='none')
        self.difficulty_mode = difficulty_mode
        
    def forward(self, logits, targets):
        # Get base loss for each sample
        losses = self.base_criterion(logits, targets)
        
        if self.difficulty_mode == 'weight':
            # Apply higher weights to more difficult bins
            weights = self.bin_difficulties[targets]
            weighted_loss = (losses * weights).mean()
            
            # Add regularization penalty for Bin 1 to prevent overfitting
            bin1_mask = (targets == 1)
            if bin1_mask.sum() > 0:
                # L2 penalty on Bin 1 logits to encourage generalization
                bin1_logits = logits[bin1_mask]
                regularization = 0.1 * torch.mean(torch.norm(bin1_logits, dim=1))
                return weighted_loss + regularization
            
            return weighted_loss
            
        elif self.difficulty_mode == 'margin':
            # Apply margin-based penalty for difficult bins
            predictions = torch.argmax(logits, dim=1)
            correct_mask = (predictions == targets)
            
            # Higher penalty for incorrect predictions on difficult bins
            difficulty_penalty = self.bin_difficulties[targets]
            final_losses = losses * (1 + difficulty_penalty * (~correct_mask).float())
            return final_losses.mean()
            
        else:
            return losses.mean()

class FocalLoss(nn.Module):
    """Focal Loss to handle class imbalance and focus on hard examples"""
    def __init__(self, alpha=None, gamma=2, bin_specific_gamma=None):
        super().__init__()
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None
        self.gamma = gamma
        self.bin_specific_gamma = bin_specific_gamma  # Different gamma for each bin
        
    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Use bin-specific gamma if provided
        if self.bin_specific_gamma is not None:
            gamma_per_sample = torch.tensor([self.bin_specific_gamma[t.item()] for t in targets], 
                                          device=targets.device, dtype=torch.float32)
            focal_weight = (1 - pt) ** gamma_per_sample
        else:
            focal_weight = (1 - pt) ** self.gamma
            
        focal_loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()
