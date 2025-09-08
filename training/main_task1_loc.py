import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import h5py
import numpy as np
import os
import torch.nn.functional as F
import random
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from src.model_task1_loc import MultiBranchFusion
from src.task1_components import FungiDataset

class EarlyStopping:
    """Early stopping utility class"""
    def __init__(self, patience=20, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                print(f"Early stopping triggered! Restored best weights from {self.counter} epochs ago.")
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save the current best model weights"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)
        
        progress_bar.set_postfix({
            'loss': running_loss / total,
            'acc': correct / total * 100
        })
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def plot_training_history(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def train_model(model, train_loader, val_loader, optimizer, criterion,
                num_epochs, device, checkpoint_dir, seed=None, run_index=None,
                early_stop_patience=20):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=early_stop_patience, min_delta=1e-6, restore_best_weights=True)
    
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print(f"Training with early stopping (patience: {early_stop_patience} epochs)")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"Early stopping counter: {early_stopping.counter}/{early_stop_patience}")
        
        # Model checkpoint naming
        run_suffix = f"_run{run_index}" if run_index is not None else ""
        seed_suffix = f"_seed{seed}" if seed is not None else ""
        best_model_path = checkpoint_dir / f'task1Loc_seed_{seed}_run_{run_index}_best.pt'
        last_model_path = checkpoint_dir / f'task1Loc_seed_{seed}_run_{run_index}_last.pt'
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'seed': seed,
                'run_index': run_index,
                'early_stopped': False
            }, best_model_path)
            print(f"New best model saved at {best_model_path}! (Val Acc: {val_acc:.4f})")
        
        # Save last model every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'seed': seed,
            'early_stopped': False
        }, last_model_path)
        
        # Check for early stopping (using validation loss)
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best validation loss was: {early_stopping.best_loss:.6f}")
            break
    
    # Save training history plot and CSV
    plot_path = checkpoint_dir / f'training_plot{seed_suffix}.png'
    plot_training_history(history, save_path=plot_path)
    
    # Save history to CSV
    history_df = pd.DataFrame({
        'epoch': list(range(1, len(history['train_loss']) + 1)),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_acc': history['train_acc'],
        'val_acc': history['val_acc']
    })
    
    csv_path = checkpoint_dir / f'training_log{seed_suffix}.csv'
    history_df.to_csv(csv_path, index=False)
    print(f"Training log saved to: {csv_path}")
    
    return history, early_stopping.counter < early_stop_patience

def main():
    parser = argparse.ArgumentParser(description="Train task1Loc with multi-seed runs")
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="Number of random seeds to run training with (default: 5)"
    )
    args = parser.parse_args()

    # ---- USER SETTINGS ----
    num_seeds = args.num_seeds
    num_epochs = 500
    learning_rate = 1e-5
    batch_size = 64
    validation_split = 0.3
    early_stop_patience = 20  # Number of epochs to wait before early stopping
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Running {num_seeds} seeds with early stopping patience: {early_stop_patience}")
    
    # Set file path for the training data
    h5_file = 'dataset/task1/train_val/preprocessed_data.h5'
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"HDF5 file not found: {h5_file}")
    
    # Load dataset once (same data for all seeds)
    dataset = FungiDataset(h5_file)
    dataset_size = len(dataset)
    
    # Store results from all seeds
    all_results = []
    
    # Generate random seeds for reproducible randomness
    np.random.seed(42)  # Set a fixed seed for generating random seeds
    random_seeds = np.random.randint(0, 100000, size=num_seeds)
    
    print(f"Generated random seeds: {random_seeds}")
    
    # Run training for multiple seeds
    for i, seed in enumerate(random_seeds):
        print(f"\n{'='*60}")
        print(f"STARTING SEED {i+1}/{num_seeds} (seed value: {seed})")
        print(f"{'='*60}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Create train/val split with seed-specific randomness
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)
        
        split = int(np.floor(validation_split * dataset_size))
        val_indices = indices[:split]
        train_indices = indices[split:]
        
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model with fresh weights
        model = MultiBranchFusion().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Create seed-specific checkpoint directory
        checkpoint_dir = f'checkpoints_multiseed_fungi/seed_{i}'
        
        # Train model
        history, completed_all_epochs = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=num_epochs,
            device=device,
            checkpoint_dir=checkpoint_dir,
            seed=seed,
            run_index=i,
            early_stop_patience=early_stop_patience
        )
        
        # Store final results
        final_results = {
            'run_index': i,
            'seed_value': seed,
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'best_val_acc': max(history['val_acc']),
            'best_val_loss': min(history['val_loss']),
            'epochs_completed': len(history['train_acc']),
            'early_stopped': not completed_all_epochs
        }
        all_results.append(final_results)
        
        print(f"Run {i} (seed {seed}) completed:")
        print(f"  Epochs completed: {final_results['epochs_completed']}/{num_epochs}")
        print(f"  Early stopped: {final_results['early_stopped']}")
        print(f"  Final Val Acc: {final_results['final_val_acc']:.4f}")
        print(f"  Best Val Acc: {final_results['best_val_acc']:.4f}")
    
    # Summarize results across all seeds
    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL SEEDS")
    print(f"{'='*60}")
    
    results_df = pd.DataFrame(all_results)
    
    # Calculate statistics
    stats = {
        'val_acc_mean': results_df['final_val_acc'].mean(),
        'val_acc_std': results_df['final_val_acc'].std(),
        'val_loss_mean': results_df['final_val_loss'].mean(),
        'val_loss_std': results_df['final_val_loss'].std(),
        'best_val_acc_mean': results_df['best_val_acc'].mean(),
        'best_val_acc_std': results_df['best_val_acc'].std(),
        'avg_epochs_completed': results_df['epochs_completed'].mean(),
        'early_stop_rate': results_df['early_stopped'].sum() / len(results_df) * 100
    }
    
    print(f"Final Validation Accuracy: {stats['val_acc_mean']:.4f} ± {stats['val_acc_std']:.4f}")
    print(f"Final Validation Loss: {stats['val_loss_mean']:.4f} ± {stats['val_loss_std']:.4f}")
    print(f"Best Validation Accuracy: {stats['best_val_acc_mean']:.4f} ± {stats['best_val_acc_std']:.4f}")
    print(f"Average epochs completed: {stats['avg_epochs_completed']:.1f}/{num_epochs}")
    print(f"Early stopping rate: {stats['early_stop_rate']:.1f}% ({results_df['early_stopped'].sum()}/{len(results_df)} runs)")
    
    # Save summary results
    summary_dir = Path('checkpoints_multiseed_fungi')
    summary_dir.mkdir(exist_ok=True, parents=True)
    
    results_df.to_csv(summary_dir / 'seed_results_summary.csv', index=False)
    
    # Save statistics
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(summary_dir / 'seed_statistics.csv', index=False)
    
    print(f"\nResults saved to {summary_dir}/")
    print("- seed_results_summary.csv: Individual seed results")
    print("- seed_statistics.csv: Summary statistics")

if __name__ == '__main__':
    main()