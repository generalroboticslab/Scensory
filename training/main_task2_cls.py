import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import h5py
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import sys
import argparse
import os

# Add parent directory to Python path to access src module
sys.path.append(str(Path(__file__).parent.parent))
from src.model_task2_cls import SingleSensorModel
from src.task2_components import FungiDataset

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_step(model, optimizer, batch, criterion, device, grad_clip=None):
    """Enhanced training step with gradient clipping"""
    model.train()
    sensor_data, fungi_labels = batch
    sensor_data, fungi_labels = sensor_data.to(device), fungi_labels.to(device)

    # Forward pass
    logits = model(sensor_data)
    loss = criterion(logits, fungi_labels)

    # Backward pass with gradient clipping
    optimizer.zero_grad()
    loss.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    # Metrics
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == fungi_labels).float().mean()
    return loss.item(), accuracy.item()

@torch.no_grad()
def validate(model, val_loader, criterion, device, verbose=True):
    """Enhanced validation with full metrics"""
    model.eval()
    total_loss, total_acc = 0, 0
    all_preds, all_labels = [], []

    for batch in val_loader:
        sensor_data, fungi_labels = batch
        sensor_data, fungi_labels = sensor_data.to(device), fungi_labels.to(device)
        
        logits = model(sensor_data)
        loss = criterion(logits, fungi_labels)
        
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == fungi_labels).float().mean()
        
        total_loss += loss.item()
        total_acc += accuracy.item()
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(fungi_labels.cpu().numpy())

    if verbose:
        # Calculate class-wise metrics
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
        
        # Plot confusion matrix
        plot_confusion_matrix(all_labels, all_preds, CLASS_NAMES)
    
    return total_loss / len(val_loader), total_acc / len(val_loader)

def plot_confusion_matrix(labels, preds, class_names):
    """Visualize confusion matrix"""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    # plt.show()
    plt.close()

def train_model(model, train_loader, val_loader, num_epochs, device, 
                learning_rate=0.001, checkpoint_dir='checkpoints', 
                grad_clip=1.0, class_weights=None,
                patience=20, seed=None, run_id=1):
    """Enhanced training loop with early stopping and best model logging"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=1e-4
    )

    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5, 
        patience=3, 
        verbose=True
    )

    best_val_acc = 0
    epochs_since_improvement = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'seed': seed
    }

    print(f"\n{'='*60}")
    print(f"TRAINING RUN {run_id}/5 | SEED: {seed}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        model.train()
        train_loss, train_acc = 0, 0
        
        for batch in tqdm(train_loader, desc="Training"):
            loss, acc = train_step(
                model, optimizer, batch, criterion, 
                device, grad_clip=grad_clip
            )
            train_loss += loss
            train_acc += acc
        
        # Only show detailed validation results for the last epoch or best model
        verbose_val = (epoch == num_epochs - 1) or (epoch % 10 == 0)
        val_loss, val_acc = validate(model, val_loader, criterion, device, verbose=verbose_val)

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc / len(train_loader))
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_acc)

        # Always save the last model with seed info
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'seed': seed,
            'run_id': run_id,
            'history': history
        }, checkpoint_dir / f'task2Cls_seed_{seed}_run_{run_id}_last.pt')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_since_improvement = 0  # reset early stopping counter
            
            # Save best model with seed info
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'seed': seed,
                'run_id': run_id,
                'epoch': epoch,
                'history': history
            }, checkpoint_dir / f'task2Cls_seed_{seed}_run_{run_id}_best.pt')
            
            print(f"[BEST MODEL UPDATED] Epoch {epoch+1} | Val Acc: {val_acc*100:.2f}%")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement <= 5:  # Only print for first few epochs without improvement
                print(f"No improvement for {epochs_since_improvement} epoch(s)")

        print(f"Train Loss: {history['train_loss'][-1]:.4f} | Acc: {history['train_acc'][-1]*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%")

        # Early stopping
        if epochs_since_improvement >= patience:
            print(f"\n[EARLY STOPPING] No improvement in val accuracy for {patience} consecutive epochs.")
            break

    print(f"\nRUN {run_id} COMPLETED | Best Val Acc: {best_val_acc*100:.2f}%")
    return history, best_val_acc

def save_summary_results(all_results, checkpoint_dir):
    """Save summary of all runs"""
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
    print(f"Best validation accuracies: {[f'{acc*100:.2f}%' for acc in best_accs]}")
    print(f"Mean accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"Best single run: {max(best_accs)*100:.2f}%")
    print(f"Worst single run: {min(best_accs)*100:.2f}%")
    
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
    
    with open(checkpoint_dir / 'multi_seed_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Summary saved to: {checkpoint_dir / 'multi_seed_summary.json'}")

def main():
    parser = argparse.ArgumentParser(description="Train task2Cls with multi-seed runs")
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
    learning_rate = 1e-4
    batch_size = 32
    validation_split = 0.2
    early_stop_patience = 20
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Running {num_seeds} seeds with early stopping patience: {early_stop_patience}")
    
    # Set file path for the training data
    h5_file = "dataset/task2/train_data_species_loc.h5"
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"HDF5 file not found: {h5_file}")

    # Load metadata for class weights
    with open("dataset/task2/metadata_dist.json") as f:
        metadata = json.load(f)
        class_counts = metadata['class_distributions']['train']['fungi_labels']
        total_samples = sum(class_counts.values())
        class_weights = torch.tensor([
            total_samples / (5 * count) for count in sorted(class_counts.values())
        ], dtype=torch.float32)

    # Initialize datasets (load once, use for all runs)
    # data_file = Path("dataset/task2/train_data_species_loc.h5") # This line is no longer needed
    
    all_results = []
    
    # Generate N random seeds (reproducible given same numpy RNG seed)
    np.random.seed(42)
    seeds = [np.random.randint(1, 100000) for _ in range(num_seeds)]
    print(f"Generated seeds: {seeds}")
    
    # Train model with each seed
    for run_id, seed in enumerate(seeds, 1):
        print(f"\nStarting run {run_id}/{num_seeds} with seed {seed}")
        
        # Set seed for reproducibility
        set_seed(seed)
        
        # Create fresh datasets and loaders for each run
        train_dataset = FungiDataset(h5_file, 'train')
        val_dataset = FungiDataset(h5_file, 'val')

        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            num_workers=4,
            persistent_workers=True
        )

        # Initialize fresh model for each run
        model = SingleSensorModel(
            input_channels=9,
            num_classes=5,
            hidden_dim=64,
            kernel_size=5,
            dropout=0.3
        ).to(device)

        # Train the model
        history, best_val_acc = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            device=device,
            learning_rate=learning_rate,
            grad_clip=1.0,
            class_weights=class_weights,
            checkpoint_dir='pretrained',
            patience=early_stop_patience,
            seed=seed,
            run_id=run_id
        )
        
        # Store results
        all_results.append({
            'seed': seed,
            'run_id': run_id,
            'history': history,
            'best_val_acc': best_val_acc
        })
        
        # Clean up GPU memory
        del model, train_loader, val_loader, train_dataset, val_dataset
        torch.cuda.empty_cache()
    
    # Save summary of all runs
    save_summary_results(all_results, 'checkpoints_task2_cls_seed')


if __name__ == "__main__":
    # Define your class names
    CLASS_NAMES = ['black', 'green', 'red', 'tri', 'white']  
    main()