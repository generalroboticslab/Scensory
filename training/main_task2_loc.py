import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import pandas as pd
import argparse
from collections import Counter
import sys

# Add parent directory to Python path to access src module
sys.path.append(str(Path(__file__).parent.parent))
from src.model_task2_loc import SpatialBinClassifier
from src.task2_components import SensorDataset, BinSpecificLoss, FocalLoss

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_bin_specific_sampler(dataset, bin_sampling_strategy='balanced'):
    """Create a sampler that handles bin-specific sampling"""
    labels = dataset.spatial_bins.numpy()
    label_counts = Counter(labels)
    
    if bin_sampling_strategy == 'balanced':
        # Standard balanced sampling
        weights = [1.0 / label_counts[label] for label in labels]
        
    elif bin_sampling_strategy == 'difficulty_based':
        # Sample more from difficult bins (based on your confusion matrix)
        bin_difficulties = {0: 1.0, 1: 2.0, 2: 1.5, 3: 1.5}  # Bin 1 most difficult
        base_weights = [1.0 / label_counts[label] for label in labels]
        difficulty_weights = [bin_difficulties[label] for label in labels]
        weights = [b * d for b, d in zip(base_weights, difficulty_weights)]
        
    elif bin_sampling_strategy == 'confusion_based':
        # Weight based on confusion patterns from your matrix
        bin_confusion_scores = {
            0: 0.58,  # 58% accuracy
            1: 0.27,  # 27% accuracy - most confused
            2: 0.46,  # 46% accuracy  
            3: 0.62   # 62% accuracy - best
        }
        # Inverse sampling - sample more from confused bins
        weights = [(1.0 - bin_confusion_scores[label]) / label_counts[label] for label in labels]
        
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def train_epoch_bin_aware(model, train_loader, criterion, optimizer, device, 
                         bin_lr_multipliers=None, grad_clip=None):
    """Training with bin-specific learning rates"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    bin_stats = {i: {'correct': 0, 'total': 0} for i in range(4)}
    
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for x, targets in progress_bar:
        x = x.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass
        logits = model(x)
        loss = criterion(logits, targets)
        
        # Bin-specific learning rate adjustment
        if bin_lr_multipliers is not None:
            # Scale gradients based on bin difficulty
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            
            # Apply bin-specific learning rate multipliers
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Calculate average multiplier for this batch
                    multipliers = torch.tensor([bin_lr_multipliers[t.item()] for t in targets], 
                                             device=device)
                    avg_multiplier = multipliers.mean()
                    param.grad *= avg_multiplier
        else:
            optimizer.zero_grad()
            loss.backward()
            
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        # Calculate accuracy and bin statistics
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == targets).sum().item()
        
        # Update bin-specific statistics
        for i in range(4):
            bin_mask = (targets == i)
            if bin_mask.sum() > 0:
                bin_correct = (predictions[bin_mask] == targets[bin_mask]).sum().item()
                bin_stats[i]['correct'] += bin_correct
                bin_stats[i]['total'] += bin_mask.sum().item()
        
        total_loss += loss.item() * len(targets)
        total_correct += correct
        total_samples += len(targets)
        
        # Calculate bin accuracies for progress bar
        bin_accs = {}
        for i in range(4):
            if bin_stats[i]['total'] > 0:
                bin_accs[f'bin{i}'] = bin_stats[i]['correct'] / bin_stats[i]['total'] * 100
            else:
                bin_accs[f'bin{i}'] = 0
        
        progress_bar.set_postfix({
            'loss': total_loss / total_samples,
            'acc': total_correct / total_samples * 100,
            **bin_accs
        })
    
    return total_loss / total_samples, total_correct / total_samples * 100, bin_stats

@torch.no_grad()
def validate_bin_aware(model, val_loader, criterion, device):
    """Validation with bin-specific metrics"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    bin_stats = {i: {'correct': 0, 'total': 0} for i in range(4)}
    
    all_preds = []
    all_targets = []
    
    for x, targets in val_loader:
        x = x.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        logits = model(x)
        loss = criterion(logits, targets)
        
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == targets).sum().item()
        
        # Update bin statistics
        for i in range(4):
            bin_mask = (targets == i)
            if bin_mask.sum() > 0:
                bin_correct = (predictions[bin_mask] == targets[bin_mask]).sum().item()
                bin_stats[i]['correct'] += bin_correct
                bin_stats[i]['total'] += bin_mask.sum().item()
        
        total_loss += loss.item() * len(targets)
        total_correct += correct
        total_samples += len(targets)
        
        all_preds.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    # Calculate bin accuracies
    bin_accuracies = {}
    for i in range(4):
        if bin_stats[i]['total'] > 0:
            bin_accuracies[f'bin_{i}'] = bin_stats[i]['correct'] / bin_stats[i]['total'] * 100
        else:
            bin_accuracies[f'bin_{i}'] = 0
    
    return (total_loss / total_samples, 
            total_correct / total_samples * 100,
            np.array(all_preds),
            np.array(all_targets),
            bin_accuracies)

def plot_bin_specific_history(history, save_path=None):
    """Plot training history with bin-specific metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Overall metrics
    axes[0,0].plot(history['train_loss'], label='Train')
    axes[0,0].plot(history['val_loss'], label='Validation')
    axes[0,0].set_title('Loss')
    axes[0,0].legend()
    
    axes[0,1].plot(history['train_acc'], label='Train')
    axes[0,1].plot(history['val_acc'], label='Validation')
    axes[0,1].set_title('Overall Accuracy (%)')
    axes[0,1].legend()
    
    # Bin-specific accuracies
    for bin_name, accs in history['bin_accs'].items():
        axes[1,0].plot(accs, label=bin_name)
    axes[1,0].set_title('Validation Accuracy by Bin (%)')
    axes[1,0].legend()
    
    # Focus on difficult bins
    axes[1,1].plot(history['bin_accs']['bin_0'], label='Bin 0', alpha=0.7)
    axes[1,1].plot(history['bin_accs']['bin_1'], label='Bin 1 (Difficult)', linewidth=2)
    axes[1,1].plot(history['bin_accs']['bin_2'], label='Bin 2', alpha=0.7)
    axes[1,1].plot(history['bin_accs']['bin_3'], label='Bin 3', alpha=0.7)
    axes[1,1].set_title('Focus on Difficult Bins')
    axes[1,1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def train_model_bin_specific(model, train_loader, val_loader, criterion, optimizer, 
                            scheduler, num_epochs, device, checkpoint_dir, seed=None, run_id=1,
                            patience=20, bin_lr_multipliers=None):
    """Train the model with bin-specific strategies and early stopping"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    best_val_acc = 0
    best_bin1_acc = 0  # Track improvement on most difficult bin
    epochs_since_improvement = 0
    
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'bin_accs': {f'bin_{i}': [] for i in range(4)},
        'seed': seed
    }
    
    # Create CSV logging list
    csv_logs = []
    
    print(f"\n{'='*60}")
    print(f"BIN-SPECIFIC TRAINING RUN {run_id}/5 | SEED: {seed}")
    print(f"{'='*60}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training with bin awareness
        train_loss, train_acc, train_bin_stats = train_epoch_bin_aware(
            model, train_loader, criterion, optimizer, device, 
            bin_lr_multipliers=bin_lr_multipliers, grad_clip=1.0
        )
        
        # Validation with bin metrics
        val_loss, val_acc, val_preds, val_targets, val_bin_accs = validate_bin_aware(
            model, val_loader, criterion, device
        )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        for bin_name, acc in val_bin_accs.items():
            history['bin_accs'][bin_name].append(acc)
        
        # Log to CSV
        csv_log_entry = {
            'epoch': epoch + 1,
            'seed': seed,
            'run_id': run_id,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': current_lr,
            'epochs_since_improvement': epochs_since_improvement
        }
        # Add bin-specific accuracies to CSV
        csv_log_entry.update(val_bin_accs)
        csv_logs.append(csv_log_entry)
        
        # Print detailed results
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.2e}")
        print("Validation Bin Accuracies:")
        for bin_name, acc in val_bin_accs.items():
            print(f"  {bin_name}: {acc:.2f}%")
        
        # Always save the last model with seed info
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'bin_accuracies': val_bin_accs,
            'val_preds': val_preds,
            'val_targets': val_targets,
            'seed': seed,
            'run_id': run_id,
            'history': history
        }, checkpoint_dir / f'task2Loc_seed_{seed}_run_{run_id}_last.pt')
        
        # Save best model based on overall accuracy or Bin 1 improvement
        current_bin1_acc = val_bin_accs.get('bin_1', 0)
        is_best = False
        
        if val_acc > best_val_acc or current_bin1_acc > best_bin1_acc:
            if current_bin1_acc > best_bin1_acc:
                best_bin1_acc = current_bin1_acc
                print(f"New best Bin 1 accuracy: {current_bin1_acc:.2f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_since_improvement = 0  # reset early stopping counter
                is_best = True
                
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'bin_accuracies': val_bin_accs,
                'val_preds': val_preds,
                'val_targets': val_targets,
                'seed': seed,
                'run_id': run_id,
                'history': history
            }, checkpoint_dir / f'task2Loc_seed_{seed}_run_{run_id}_best.pt')
            
            print(f"[BEST MODEL UPDATED] Epoch {epoch+1} | Val Acc: {val_acc:.2f}%")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement <= 5:  # Only print for first few epochs without improvement
                print(f"No improvement for {epochs_since_improvement} epoch(s)")
        
        # Update CSV log entry with best flag
        csv_logs[-1]['is_best'] = is_best
        
        # Early stopping
        if epochs_since_improvement >= patience:
            print(f"\n[EARLY STOPPING] No improvement in val accuracy for {patience} consecutive epochs.")
            break
    
    # Save CSV log for this run
    csv_df = pd.DataFrame(csv_logs)
    csv_df.to_csv(checkpoint_dir / f'training_log_bin_specific_seed_{seed}_run_{run_id}.csv', index=False)
    
    # Plot bin-specific training history for this run
    plot_bin_specific_history(history, 
                             save_path=checkpoint_dir / f'training_history_bin_specific_seed_{seed}_run_{run_id}.png')
    
    print(f"\nRUN {run_id} COMPLETED | Best Val Acc: {best_val_acc:.2f}% | Best Bin 1 Acc: {best_bin1_acc:.2f}%")
    print(f"Training log saved to: training_log_bin_specific_seed_{seed}_run_{run_id}.csv")
    return history, best_val_acc, csv_logs

def save_summary_results(all_results, checkpoint_dir):
    """Save summary of all runs and create combined CSV"""
    checkpoint_dir = Path(checkpoint_dir)
    
    # Extract key metrics
    summary = {
        'seeds': [result['seed'] for result in all_results],
        'best_val_accs': [result['best_val_acc'] for result in all_results],
        'final_epochs': [len(result['history']['val_acc']) for result in all_results]
    }
    
    # Calculate statistics
    best_accs = summary['best_val_accs']
    mean_acc = np.mean(best_accs)
    std_acc = np.std(best_accs)
    
    print(f"\n{'='*60}")
    print("FINAL SUMMARY ACROSS ALL RUNS")
    print(f"{'='*60}")
    print(f"Seeds used: {summary['seeds']}")
    print(f"Best validation accuracies: {[f'{acc:.2f}%' for acc in best_accs]}")
    print(f"Mean accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Best single run: {max(best_accs):.2f}%")
    print(f"Worst single run: {min(best_accs):.2f}%")
    
    # Save summary to JSON
    summary_data = {
        'seeds': summary['seeds'],
        'best_val_accs': best_accs,
        'final_epochs': summary['final_epochs'],
        'statistics': {
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'best_accuracy': float(max(best_accs)),
            'worst_accuracy': float(min(best_accs))
        }
    }
    
    with open(checkpoint_dir / 'multi_seed_summary_bin_specific.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Combine all CSV logs into one master CSV
    all_csv_logs = []
    for result in all_results:
        all_csv_logs.extend(result['csv_logs'])
    
    if all_csv_logs:
        combined_df = pd.DataFrame(all_csv_logs)
        combined_df.to_csv(checkpoint_dir / 'training_logs_all_seeds_bin_specific.csv', index=False)
        print(f"Combined training logs saved to: training_logs_all_seeds_bin_specific.csv")
    
    print(f"Summary saved to: {checkpoint_dir / 'multi_seed_summary_bin_specific.json'}")

def main():
    """Main function with multi-seed training"""
    parser = argparse.ArgumentParser(description="Train task2Loc with multi-seed runs")
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="Number of random seeds to run training with (default: 5)"
    )
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate N random seeds (reproducible given same numpy RNG seed)
    np.random.seed(42)
    seeds = [np.random.randint(1, 100000) for _ in range(args.num_seeds)]
    print(f"Generated seeds: {seeds}")

    data_dir = Path("dataset/task2")
    all_results = []

    # Train model with each seed
    for run_id, seed in enumerate(seeds, 1):
        print(f"\nStarting run {run_id}/{args.num_seeds} with seed {seed}")

        # Set seed for reproducibility
        set_seed(seed)

        # Datasets & loaders
        train_dataset = SensorDataset(data_dir / 'train_data_species_loc.h5', 'train')
        val_dataset   = SensorDataset(data_dir / 'train_data_species_loc.h5', 'val')

        sampler = create_bin_specific_sampler(train_dataset, 'difficulty_based')
        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            sampler=sampler,
            num_workers=8,
            pin_memory=True
        )
        val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, pin_memory=True)

        # Model
        model = SpatialBinClassifier(
            input_dim=9,
            seq_length=30,
            hidden_dim=64,
            num_layers=3,
            dropout=0.4,
            num_spatial_bins=4
        )
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model = model.to(device)

        # Loss / opt / sched
        bin_difficulties = [1.0, 1.2, 1.1, 1.0]
        criterion = BinSpecificLoss(bin_difficulties, difficulty_mode='weight').to(device)

        bin_lr_multipliers = {0: 1.0, 1: 1.1, 2: 1.0, 3: 1.0}

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

        # Train
        history, best_val_acc, csv_logs = train_model_bin_specific(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=500,
            device=device,
            checkpoint_dir='checkpoints_task2_loc_seed',
            seed=seed,
            run_id=run_id,
            patience=20,
            bin_lr_multipliers=bin_lr_multipliers
        )

        # Aggregate results
        all_results.append({
            'seed': seed,
            'run_id': run_id,
            'history': history,
            'best_val_acc': best_val_acc,
            'csv_logs': csv_logs
        })

        # Cleanup
        del model, train_loader, val_loader, train_dataset, val_dataset
        torch.cuda.empty_cache()

    # Summary across runs
    save_summary_results(all_results, 'checkpoints_task2_loc_seed')
    
if __name__ == "__main__":
    main()