import torch
import torch.nn as nn
from torch.utils.data import Dataset
import h5py
import numpy as np

class BaseTask1Dataset(Dataset):
    """Base dataset class for Task 1 with common error handling and data loading patterns"""
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

class FungiDataset(BaseTask1Dataset):
    """Dataset for fungi location detection - returns binary labels"""
    def _load_specific_data(self, f):
        # Handle both split-specific and general data keys
        if f'X_{self.split}' in f:
            self.x = torch.FloatTensor(f[f'X_{self.split}'][:])
            self.y_bin = torch.LongTensor(f[f'y_bin_{self.split}'][:])
        else:
            # Fallback to general keys (for backward compatibility)
            self.x = torch.FloatTensor(f['X'][:])
            self.y_bin = torch.LongTensor(f['y_bin'][:])
        
        assert len(self.x) == len(self.y_bin), "Mismatched data/labels length"
    
    def __getitem__(self, idx):
        return self.x[idx], self.y_bin[idx]

class SpeciesDataset(BaseTask1Dataset):
    """Dataset for species classification - returns only class labels (y_loc removed)"""
    def _load_specific_data(self, f):
        # Handle both split-specific and general data keys
        if f'X_{self.split}' in f:
            self.x = torch.FloatTensor(f[f'X_{self.split}'][:])
            self.y_class = torch.LongTensor(f[f'y_class_{self.split}'][:])
        else:
            # Fallback to general keys (for backward compatibility)
            self.x = torch.FloatTensor(f['X'][:])
            self.y_class = torch.LongTensor(f['y_class'][:])
        
        assert len(self.x) == len(self.y_class), "Mismatched data/labels length"
    
    def __getitem__(self, idx):
        return self.x[idx], self.y_class[idx]

class FungiLocationDataset(BaseTask1Dataset):
    """Dataset for fungi location detection with spatial information"""
    def _load_specific_data(self, f):
        # Handle both split-specific and general data keys
        if f'X_{self.split}' in f:
            self.x = torch.FloatTensor(f[f'X_{self.split}'][:])
            self.y_loc = torch.FloatTensor(f[f'y_loc_{self.split}'][:])
        else:
            # Fallback to general keys (for backward compatibility)
            self.x = torch.FloatTensor(f['X'][:])
            self.y_loc = torch.FloatTensor(f['y_loc'][:])
        
        assert len(self.x) == len(self.y_loc), "Mismatched data/labels length"
    
    def __getitem__(self, idx):
        return self.x[idx], self.y_loc[idx]

class FungiClassificationDataset(BaseTask1Dataset):
    """Dataset for fungi species classification only"""
    def _load_specific_data(self, f):
        # Handle both split-specific and general data keys
        if f'X_{self.split}' in f:
            self.x = torch.FloatTensor(f[f'X_{self.split}'][:])
            self.y_class = torch.LongTensor(f[f'y_class_{self.split}'][:])
        else:
            # Fallback to general keys (for backward compatibility)
            self.x = torch.FloatTensor(f['X'][:])
            self.y_class = torch.LongTensor(f['y_class'][:])
        
        assert len(self.x) == len(self.y_class), "Mismatched data/labels length"
    
    def __getitem__(self, idx):
        return self.x[idx], self.y_class[idx]

# Legacy dataset classes for backward compatibility
class FungiDatasetLegacy(Dataset):
    """Legacy dataset class for fungi location detection - direct H5 loading"""
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(h5_file, 'r') as f:
            self.X = f['X'][:]
            self.y_bin = f['y_bin'][:]
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y_bin[idx], dtype=torch.long)
        return x, y

class SpeciesDatasetLegacy(Dataset):
    """Legacy dataset class for species classification - direct H5 loading (y_loc removed)"""
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(h5_file, 'r') as f:
            self.X = f['X'][:]
            self.y_class = f['y_class'][:]
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y_class = torch.tensor(self.y_class[idx], dtype=torch.long)
        return x, y_class
