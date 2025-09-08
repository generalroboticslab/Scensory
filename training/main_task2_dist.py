import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import sys
import argparse

# Add parent directory to Python path to access src module
sys.path.append(str(Path(__file__).parent.parent))
from src.model_task2_dist import DistanceRegressor
from src.task2_components import DistanceDataset

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
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()

def train_epoch(model, train_loader, optimizer, device, grad_clip=None):
    model.train()
    total_loss = 0
    total_dist_error = 0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        x = batch['x'].to(device, non_blocking=True)
        distances = batch['distance'].to(device, non_blocking=True)

        preds = model(x)
        loss = model.compute_loss(preds, distances)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        dist_error = torch.abs(preds - distances).mean().item()
        batch_size = len(distances)
        total_loss += loss.item() * batch_size
        total_dist_error += dist_error * batch_size
        total_samples += batch_size

        progress_bar.set_postfix({
            'loss': total_loss / total_samples,
            'dist_err': total_dist_error / total_samples
        })

    metrics = {
        'loss': total_loss / total_samples,
        'dist_error': total_dist_error / total_samples,
    }
    return metrics

@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_dist_error = 0
    total_samples = 0
    preds_all = []
    targets_all = []

    for batch in val_loader:
        x = batch['x'].to(device, non_blocking=True)
        distances = batch['distance'].to(device, non_blocking=True)

        preds = model(x)
        loss = model.compute_loss(preds, distances)
        dist_error = torch.abs(preds - distances).mean().item()
        batch_size = len(distances)
        total_loss += loss.item() * batch_size
        total_dist_error += dist_error * batch_size
        total_samples += batch_size

        preds_all.append(preds.cpu().numpy())
        targets_all.append(distances.cpu().numpy())

    metrics = {
        'loss': total_loss / total_samples,
        'dist_error': total_dist_error / total_samples,
        'predictions': np.concatenate(preds_all),
        'targets': np.concatenate(targets_all)
    }
    return metrics

def plot_training_history(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax2.plot(history['train_dist_error'], label='Train')
    ax2.plot(history['val_dist_error'], label='Validation')
    ax2.set_title('Distance Error')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Training plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def train_model(model, train_loader, val_loader, optimizer, scheduler, 
                num_epochs, device, checkpoint_dir, model_tag, seed=None, run_index=None,
                early_stop_patience=20):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    early_stopping = EarlyStopping(patience=early_stop_patience, min_delta=1e-6, restore_best_weights=True)
    best_val_dist = float('inf')
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dist_error': [], 'val_dist_error': []
    }
    print(f"Training with early stopping (patience: {early_stop_patience} epochs)")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_metrics = train_epoch(model, train_loader, optimizer, device, grad_clip=1.0)
        val_metrics = validate(model, val_loader, device)
        scheduler.step(val_metrics['dist_error'])

        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_dist_error'].append(train_metrics['dist_error'])
        history['val_dist_error'].append(val_metrics['dist_error'])

        print(f"Train - Loss: {train_metrics['loss']:.4f}, Dist Error: {train_metrics['dist_error']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, Dist Error: {val_metrics['dist_error']:.4f}")
        print(f"Early stopping counter: {early_stopping.counter}/{early_stop_patience}")

        run_suffix = f"_run{run_index}" if run_index is not None else ""
        seed_suffix = f"_seed{seed}" if seed is not None else ""
        best_model_path = checkpoint_dir / f'task2Dist_seed_{seed}_run_{run_index}_best.pt'
        last_model_path = checkpoint_dir / f'task2Dist_seed_{seed}_run_{run_index}_last.pt'

        if val_metrics['dist_error'] < best_val_dist:
            best_val_dist = val_metrics['dist_error']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'seed': seed,
                'run_index': run_index,
                'early_stopped': False
            }, best_model_path)
            print(f"New best model saved at {best_model_path}!")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'seed': seed,
            'early_stopped': False
        }, last_model_path)
        if early_stopping(val_metrics['dist_error'], model):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best validation distance error was: {early_stopping.best_loss:.6f}")
            break

    plot_path = checkpoint_dir / f'training_plot{model_tag}{seed_suffix}.png'
    plot_training_history(history, save_path=plot_path)
    # Save history to CSV
    history_df = pd.DataFrame({
        'epoch': list(range(1, len(history['train_loss']) + 1)),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_dist_error': history['train_dist_error'],
        'val_dist_error': history['val_dist_error'],
    })
    csv_path = checkpoint_dir / f'training_log{model_tag}{seed_suffix}.csv'
    history_df.to_csv(csv_path, index=False)
    print(f"Training log saved to: {csv_path}")
    return history, early_stopping.counter < early_stop_patience

def main():
    parser = argparse.ArgumentParser(description="Train task2Dist with multi-seed runs")
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="Number of random seeds to run training with (default: 5)"
    )
    args = parser.parse_args()

    # ---- USER SETTINGS ----
    fungi_filter = 0    # Set to None to use all, or 0/1/2/3/4 to filter for that label
    num_seeds = args.num_seeds
    num_epochs = 500
    early_stop_patience = 20  # Number of epochs to wait before early stopping
    batch_size = 64
    learning_rate = 1e-3
    data_dir = Path("dataset/task2")
    
    model_tag = f"_fungi{fungi_filter}" if fungi_filter is not None else "_allfungi"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Running {num_seeds} seeds with fungi_filter={fungi_filter}")
    print(f"Early stopping patience: {early_stop_patience} epochs")
    train_dataset = DistanceDataset(data_dir / 'train_data_dist.h5', 'train', fungi_filter=fungi_filter)
    val_dataset = DistanceDataset(data_dir / 'train_data_dist.h5', 'val', fungi_filter=fungi_filter)
    all_results = []
    np.random.seed(42)
    random_seeds = np.random.randint(0, 100000, size=num_seeds)
    print(f"Generated random seeds: {random_seeds}")
    for i, seed in enumerate(random_seeds):
        print(f"\n{'='*60}")
        print(f"STARTING SEED {i+1}/{num_seeds} (seed value: {seed})")
        print(f"{'='*60}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
        )
        # ----- Use DistanceRegressor, not MultiTaskNet -----
        model = DistanceRegressor(
            input_dim=9,
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
            use_bidirectional=True
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        checkpoint_dir = f'pretrained_dist/checkpoints_multiseed{model_tag}/seed_{i}'
        history, completed_all_epochs = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            device=device,
            checkpoint_dir=checkpoint_dir,
            model_tag=model_tag,
            seed=seed,
            run_index=i,
            early_stop_patience=early_stop_patience
        )
        final_results = {
            'run_index': i,
            'seed_value': seed,
            'final_train_dist_error': history['train_dist_error'][-1],
            'final_val_dist_error': history['val_dist_error'][-1],
            'best_val_dist_error': min(history['val_dist_error']),
            'epochs_completed': len(history['train_dist_error']),
            'early_stopped': not completed_all_epochs
        }
        all_results.append(final_results)
        print(f"Run {i} (seed {seed}) completed:")
        print(f"  Epochs completed: {final_results['epochs_completed']}/{num_epochs}")
        print(f"  Early stopped: {final_results['early_stopped']}")
        print(f"  Final Val Dist Error: {final_results['final_val_dist_error']:.4f}")
        print(f"  Best Val Dist Error: {final_results['best_val_dist_error']:.4f}")

    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL SEEDS")
    print(f"{'='*60}")
    results_df = pd.DataFrame(all_results)
    stats = {
        'val_dist_error_mean': results_df['final_val_dist_error'].mean(),
        'val_dist_error_std': results_df['final_val_dist_error'].std(),
        'best_val_dist_error_mean': results_df['best_val_dist_error'].mean(),
        'best_val_dist_error_std': results_df['best_val_dist_error'].std(),
        'avg_epochs_completed': results_df['epochs_completed'].mean(),
        'early_stop_rate': results_df['early_stopped'].sum() / len(results_df) * 100
    }
    print(f"Final Validation Distance Error: {stats['val_dist_error_mean']:.4f} ± {stats['val_dist_error_std']:.4f}")
    print(f"Best Validation Distance Error: {stats['best_val_dist_error_mean']:.4f} ± {stats['best_val_dist_error_std']:.4f}")
    print(f"Average epochs completed: {stats['avg_epochs_completed']:.1f}/{num_epochs}")
    print(f"Early stopping rate: {stats['early_stop_rate']:.1f}% ({results_df['early_stopped'].sum()}/{len(results_df)} runs)")
    summary_dir = Path(f'checkpoints_multiseed{model_tag}')
    summary_dir.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(summary_dir / 'seed_results_summary.csv', index=False)
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(summary_dir / 'seed_statistics.csv', index=False)
    print(f"\nResults saved to {summary_dir}/")
    print("- seed_results_summary.csv: Individual seed results")
    print("- seed_statistics.csv: Summary statistics")

if __name__ == "__main__":
    main()
