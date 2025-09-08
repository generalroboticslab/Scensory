import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import h5py
import json
import logging
import glob
from matplotlib.colors import LinearSegmentedColormap
import sys

# Add parent directory to Python path to access src module
sys.path.append(str(Path(__file__).parent.parent))
from src.task2_components import SensorDataset
from src.model_task2_loc import SpatialBinClassifier

# Modern styling
TITLE_FONT = {'fontsize': 18, 'weight': 'bold'}
LABEL_FONT = {'fontsize': 14, 'weight': 'bold'}
TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12

plt.rcParams.update({
    'axes.labelweight': 'bold',
    'axes.titlesize': TITLE_FONT['fontsize'],
    'axes.titleweight': TITLE_FONT['weight'],
    'axes.labelsize': LABEL_FONT['fontsize'],
    'xtick.labelsize': TICK_FONT_SIZE,
    'ytick.labelsize': TICK_FONT_SIZE,
    'legend.fontsize': LEGEND_FONT_SIZE
})

# Custom colors - using your preferred palette
colors = {
    'primary': '#d1e6ff',      # Light Blue - Base
    'accent': '#ffe6d1',       # Light Orange/Amber - Accent  
    'harmonious1': '#d1ffff',  # Light Teal - Harmonious
    'harmonious2': '#e6d1ff',  # Light Purple - Harmonious
    'contrast': '#aed2f0',     # Slightly deeper Blue
    'border': '#4d545d'        # Muted Dark Blue-Grey
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_model(checkpoint_path, model, device):
    """Load model from checkpoint with handling for DataParallel models"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract state dict from checkpoint
        if isinstance(checkpoint, dict) and ('model_state_dict' in checkpoint or 'state_dict' in checkpoint):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Handle DataParallel/DistributedDataParallel models
        # Check if state_dict has 'module.' prefix (indicating DataParallel training)
        if any(key.startswith('module.') for key in state_dict.keys()):
            logger.info("Detected DataParallel model, removing 'module.' prefix...")
            # Remove 'module.' prefix from all keys
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # Remove 'module.' prefix (7 characters)
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        # Load the corrected state dict
        model.load_state_dict(state_dict)
        return model
        
    except Exception as e:
        raise RuntimeError(f"Error loading model from {checkpoint_path}: {str(e)}")

@torch.no_grad()
def evaluate_model(model, test_loader, device):
    """Evaluate model performance on test set"""
    model.eval()
    all_preds = []
    all_targets = []
    test_correct = 0
    total_samples = 0

    for x, targets in test_loader:
        x = x.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass
        logits = model(x)
        predictions = torch.argmax(logits, dim=1)

        # Calculate accuracy
        correct = (predictions == targets).sum().item()
        test_correct += correct
        total_samples += len(targets)

        # Store predictions and targets
        all_preds.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    test_accuracy = (test_correct / total_samples) * 100
    return np.array(all_preds), np.array(all_targets), test_accuracy

CUSTOM_BLUE = "#1491ff"

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title_suffix=""):
    """Plot confusion matrix with modern styling matching the reference code"""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    
    # Normalize per row (accuracy per class)
    cm_accuracy = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_accuracy = np.nan_to_num(cm_accuracy)  # Handle division by zero
    
    # Formatting the annotations
    cm_display = np.round(cm_accuracy * 100, 2)
    custom_blue_cmap = LinearSegmentedColormap.from_list("custom_blue", ["#ffffff", CUSTOM_BLUE])
    
    # plt.figure(figsize=(10, 8))
    plt.figure(figsize=(9.5, 7.5))
    sns.heatmap(
        cm_accuracy,
        annot=cm_display,
        fmt='.2f',
        cmap=custom_blue_cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        vmin=0,
        vmax=1,
        annot_kws={'color': 'black', 'fontsize': 14}  # Changed from just 'color': 'black'
    )
    
    # title = f'Confusion Matrix{title_suffix}'
    title = 'Confusion Matrix (% Accuracy Per Bin)'
    plt.title(title, **TITLE_FONT)
    plt.xlabel('Predicted Label', **LABEL_FONT)
    plt.ylabel('True Label', **LABEL_FONT)
    plt.xticks(fontsize=14)  # X-axis labels (Predicted)
    plt.yticks(fontsize=14)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def find_model_checkpoints(checkpoint_dir):
    """Find all best model checkpoints from multi-seed training."""
    checkpoint_dir = Path(checkpoint_dir)

    # Primary (new) pattern
    patterns = [
        str(checkpoint_dir / "task2Loc.pt"), # default best model pattern
    ]

    # Fallbacks (old names or single-best variants)
    patterns += [
        str(checkpoint_dir / "best_model_loc.pt"),
        str(checkpoint_dir / "task2Loc_seed_*_best.pt")
    ]

    checkpoint_files = []
    for pat in patterns:
        found = glob.glob(pat)
        if found:
            checkpoint_files.extend(found)

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    # Deduplicate + sort
    checkpoint_files = sorted(set(checkpoint_files))
    return checkpoint_files

def extract_seed_info(checkpoint_path):
    """Extract seed and run information from checkpoint filename"""
    filename = Path(checkpoint_path).name
    try:
        # Handle different filename patterns
        if "bin_specific_seed_" in filename:
            # Extract from "best_model_bin_specific_seed_1184_run_3.pt"
            parts = filename.replace('.pt', '').split('_')
            seed_idx = parts.index('seed') + 1
            run_idx = parts.index('run') + 1
            seed = int(parts[seed_idx])
            run_id = int(parts[run_idx])
            return seed, run_id
        elif "loc_seed_" in filename:
            # Extract from "best_model_loc_seed_1234_run_1.pt"
            parts = filename.replace('.pt', '').split('_')
            seed_idx = parts.index('seed') + 1
            run_idx = parts.index('run') + 1
            seed = int(parts[seed_idx])
            run_id = int(parts[run_idx])
            return seed, run_id
        elif "currentbest" in filename:
            # Handle single best model file
            return "Best", 1
        else:
            # Fallback for other single model files
            return None, None
    except (ValueError, IndexError):
        # Fallback for any parsing errors
        return None, None
    
def evaluate_single_model(model_path, test_loader, device, class_names):
    """Evaluate a single model and return results"""
    # Initialize fresh model
    model = SpatialBinClassifier(
        input_dim=9,
        seq_length=30,
        hidden_dim=64,
        num_layers=3,
        dropout=0.3,
        num_spatial_bins=len(class_names)
    ).to(device)
    
    # Load model weights
    model = load_model(model_path, model, device)
    
    # Evaluate
    y_pred, y_true, accuracy = evaluate_model(model, test_loader, device)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                 digits=4, output_dict=True)
    
    return {
        'y_pred': y_pred,
        'y_true': y_true,
        'accuracy': accuracy,
        'report': report
    }

def extract_detailed_metrics(all_results, class_names):
    """Extract detailed per-class metrics from all models"""
    metrics_data = {
        'accuracy': {class_name: [] for class_name in class_names},  # True per-class accuracy
        'precision': {class_name: [] for class_name in class_names},
        'recall': {class_name: [] for class_name in class_names},
        'f1_score': {class_name: [] for class_name in class_names},
        'overall_accuracy': [],
        'macro_avg_precision': [],
        'macro_avg_recall': [],
        'macro_avg_f1': [],
        'weighted_avg_precision': [],
        'weighted_avg_recall': [],
        'weighted_avg_f1': []
    }
    
    for result in all_results:
        if result.get('seed') == 'Ensemble':
            continue  # Skip ensemble for individual model analysis
            
        report = result['report']
        y_true = result['y_true']
        y_pred = result['y_pred']
        
        # Calculate true per-class accuracy from confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
        
        # Per-class metrics
        for i, class_name in enumerate(class_names):
            # True per-class accuracy = (TP + TN) / Total
            # For multi-class: all correct predictions for all classes / total predictions
            class_accuracy = np.sum(y_true == y_pred) / len(y_true)  # This is overall accuracy
            
            # Better: per-class accuracy = correct predictions for this class / total samples of this class
            class_mask = (y_true == i)
            if np.sum(class_mask) > 0:  # Avoid division by zero
                per_class_accuracy = np.sum((y_true == i) & (y_pred == i)) / np.sum(class_mask)
            else:
                per_class_accuracy = 0.0
            
            metrics_data['accuracy'][class_name].append(per_class_accuracy)
            metrics_data['precision'][class_name].append(report[class_name]['precision'])
            metrics_data['recall'][class_name].append(report[class_name]['recall'])
            metrics_data['f1_score'][class_name].append(report[class_name]['f1-score'])
        
        # Overall metrics
        metrics_data['overall_accuracy'].append(report['accuracy'])
        metrics_data['macro_avg_precision'].append(report['macro avg']['precision'])
        metrics_data['macro_avg_recall'].append(report['macro avg']['recall'])
        metrics_data['macro_avg_f1'].append(report['macro avg']['f1-score'])
        metrics_data['weighted_avg_precision'].append(report['weighted avg']['precision'])
        metrics_data['weighted_avg_recall'].append(report['weighted avg']['recall'])
        metrics_data['weighted_avg_f1'].append(report['weighted avg']['f1-score'])
    
    return metrics_data

def plot_box_metrics(metrics_data, class_names, save_dir):
    """Create box plots for various performance metrics"""
    save_dir = Path(save_dir)
    
    # Create a single figure with 3 subplots side by side (removed accuracy)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define DIFFERENT color palette for spatial bins (distinct from fungi species colors)
    # Using warmer/earth tones instead of the blue-based palette
    spatial_box_colors = [
        '#e6a3a3',      # Light Pink/Rose - Bin 0
        '#a3a3e6',      # Light Peach - Bin 1  
        '#fff2cc',      # Light Yellow - Bin 2
        '#e6f2e6'       # Light Green - Bin 3
    ]
    
    border_color = colors['border']  # Keep same border color
    
    # 1. Precision box plot
    precision_data = [metrics_data['precision'][class_name] for class_name in class_names]
    box_plot = axes[0].boxplot(precision_data, positions=range(len(class_names)), 
                               patch_artist=True, widths=0.6)
    
    for patch, color in zip(box_plot['boxes'], spatial_box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)
        patch.set_edgecolor(border_color)
        patch.set_linewidth(1.5)
    
    for element in ['whiskers', 'fliers', 'caps']:
        plt.setp(box_plot[element], color=border_color, linewidth=1.5)
    plt.setp(box_plot['medians'], color=border_color, linewidth=2.5)
    
    axes[0].set_title('Precision by Group', **TITLE_FONT)
    axes[0].set_ylabel('Precision', **LABEL_FONT)
    axes[0].set_xlabel('Group', **LABEL_FONT)
    axes[0].set_xticks(range(len(class_names)))
    axes[0].set_xticklabels(class_names)
    axes[0].grid(True, alpha=0.3, axis='y', color=border_color)
    
    # 2. Recall box plot
    recall_data = [metrics_data['recall'][class_name] for class_name in class_names]
    box_plot = axes[1].boxplot(recall_data, positions=range(len(class_names)), 
                               patch_artist=True, widths=0.6)
    
    for patch, color in zip(box_plot['boxes'], spatial_box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)
        patch.set_edgecolor(border_color)
        patch.set_linewidth(1.5)
    
    for element in ['whiskers', 'fliers', 'caps']:
        plt.setp(box_plot[element], color=border_color, linewidth=1.5)
    plt.setp(box_plot['medians'], color=border_color, linewidth=2.5)
    
    axes[1].set_title('Recall by Group', **TITLE_FONT)
    axes[1].set_ylabel('Recall', **LABEL_FONT)
    axes[1].set_xlabel('Group', **LABEL_FONT)
    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_xticklabels(class_names)
    axes[1].grid(True, alpha=0.3, axis='y', color=border_color)
    
    # 3. F1-Score box plot
    f1_data = [metrics_data['f1_score'][class_name] for class_name in class_names]
    box_plot = axes[2].boxplot(f1_data, positions=range(len(class_names)), 
                               patch_artist=True, widths=0.6)
    
    for patch, color in zip(box_plot['boxes'], spatial_box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)
        patch.set_edgecolor(border_color)
        patch.set_linewidth(1.5)
    
    for element in ['whiskers', 'fliers', 'caps']:
        plt.setp(box_plot[element], color=border_color, linewidth=1.5)
    plt.setp(box_plot['medians'], color=border_color, linewidth=2.5)
    
    axes[2].set_title('F1-Score by Group', **TITLE_FONT)
    axes[2].set_ylabel('F1-Score', **LABEL_FONT)
    axes[2].set_xlabel('Group', **LABEL_FONT)
    axes[2].set_xticks(range(len(class_names)))
    axes[2].set_xticklabels(class_names)
    axes[2].grid(True, alpha=0.3, axis='y', color=border_color)

    # Set y-limits for all plots
    axes[0].set_ylim(0.0, 1)  # For precision  
    axes[1].set_ylim(0.0, 1)  # For recall
    axes[2].set_ylim(0.0, 1)  # For F1-score

    # Set consistent y-axis limits for all plots
    axes[0].set_ylim(0.0, 1)  # For precision  
    axes[1].set_ylim(0.0, 1)  # For recall
    axes[2].set_ylim(0.0, 1)  # For F1-score
    axes[0].tick_params(axis='x', labelsize=14, labelrotation=0)  # Make x-axis labels larger
    axes[1].tick_params(axis='x', labelsize=14, labelrotation=0)  # Make x-axis labels larger  
    axes[2].tick_params(axis='x', labelsize=14, labelrotation=0)  # Make x-axis labels larger
    axes[0].tick_params(axis='y', labelsize=14, labelrotation=0)  # Make x-axis labels larger
    axes[1].tick_params(axis='y', labelsize=14, labelrotation=0)  # Make x-axis labels larger  
    axes[2].tick_params(axis='y', labelsize=14, labelrotation=0)  # Make x-axis labels larger
    
    
    
    plt.tight_layout()
    plt.savefig(save_dir / 'per_class_metrics_boxplot_spatial.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_accuracy_comparison(all_results, save_path):
    """Plot comparison of accuracies across all seeds"""
    seeds = [result['seed'] for result in all_results]
    accuracies = [result['accuracy'] for result in all_results]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(seeds)), accuracies, 
                   color=colors['primary'], alpha=0.8)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add mean line
    mean_acc = np.mean(accuracies)
    plt.axhline(y=mean_acc, color='red', linestyle='--', alpha=0.8, 
                label=f'Mean: {mean_acc:.2f}%')
    
    plt.title('Test Accuracy Across Different Seeds', **TITLE_FONT)
    plt.xlabel('Model (Seed)', **LABEL_FONT)
    plt.ylabel('Test Accuracy (%)', **LABEL_FONT)
    plt.xticks(range(len(seeds)), [f'Seed {seed}' for seed in seeds], rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def create_metrics_summary_table(metrics_data, class_names, save_path):
    """Create a detailed summary table of all metrics"""
    with open(save_path, 'w') as f:
        f.write("DETAILED METRICS SUMMARY ACROSS ALL MODELS\n")
        f.write("=" * 80 + "\n\n")
        
        # Per-class metrics
        f.write("PER-CLASS METRICS (Mean ± Std)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Class':<15} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}\n")
        f.write("-" * 85 + "\n")
        
        for class_name in class_names:
            accuracy_mean = np.mean(metrics_data['accuracy'][class_name])
            accuracy_std = np.std(metrics_data['accuracy'][class_name])
            precision_mean = np.mean(metrics_data['precision'][class_name])
            precision_std = np.std(metrics_data['precision'][class_name])
            recall_mean = np.mean(metrics_data['recall'][class_name])
            recall_std = np.std(metrics_data['recall'][class_name])
            f1_mean = np.mean(metrics_data['f1_score'][class_name])
            f1_std = np.std(metrics_data['f1_score'][class_name])
            
            f.write(f"{class_name:<15} ")
            f.write(f"{accuracy_mean:.3f}±{accuracy_std:.3f}   ")
            f.write(f"{precision_mean:.3f}±{precision_std:.3f}   ")
            f.write(f"{recall_mean:.3f}±{recall_std:.3f}   ")
            f.write(f"{f1_mean:.3f}±{f1_std:.3f}\n")
        
        f.write("\n\nOVERALL METRICS (Mean ± Std)\n")
        f.write("-" * 40 + "\n")
        
        overall_metrics = [
            ('Overall Accuracy', metrics_data['overall_accuracy']),
            ('Macro Avg Precision', metrics_data['macro_avg_precision']),
            ('Macro Avg Recall', metrics_data['macro_avg_recall']),
            ('Macro Avg F1-Score', metrics_data['macro_avg_f1']),
            ('Weighted Avg Precision', metrics_data['weighted_avg_precision']),
            ('Weighted Avg Recall', metrics_data['weighted_avg_recall']),
            ('Weighted Avg F1-Score', metrics_data['weighted_avg_f1'])
        ]
        
        for metric_name, values in overall_metrics:
            mean_val = np.mean(values)
            std_val = np.std(values)
            f.write(f"{metric_name:<25}: {mean_val:.4f} ± {std_val:.4f}\n")

def save_comprehensive_results(all_results, results_dir, class_names):
    """Save comprehensive evaluation results"""
    results_dir = Path(results_dir)
    
    # Calculate statistics
    accuracies = [result['accuracy'] for result in all_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    # Save summary statistics
    with open(results_dir / 'evaluation_summary_spatial.txt', 'w') as f:
        f.write("Multi-Seed Spatial Model Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Individual Model Results:\n")
        f.write("-" * 30 + "\n")
        for i, result in enumerate(all_results):
            seed = result.get('seed', 'Unknown')
            run_id = result.get('run_id', i+1)
            f.write(f"Run {run_id} (Seed {seed}): {result['accuracy']:.2f}%\n")
        
        f.write(f"\nOverall Statistics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%\n")
        f.write(f"Best Model: {max(accuracies):.2f}%\n")
        f.write(f"Worst Model: {min(accuracies):.2f}%\n")
        f.write(f"Total Models Evaluated: {len(all_results)}\n")
    
    # Save detailed results as JSON
    summary_data = {
        'individual_results': [
            {
                'seed': result.get('seed', 'Unknown'),
                'run_id': result.get('run_id', i+1),
                'accuracy': float(result['accuracy']),
                'classification_report': result['report']
            }
            for i, result in enumerate(all_results)
        ],
        'statistics': {
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'best_accuracy': float(max(accuracies)),
            'worst_accuracy': float(min(accuracies)),
            'num_models': len(all_results)
        }
    }
    
    with open(results_dir / 'detailed_evaluation_results_spatial.json', 'w') as f:
        json.dump(summary_data, f, indent=2)

def main():
    """Main evaluation function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create results directory
    results_dir = Path('Results/task2_localization')
    results_dir.mkdir(exist_ok=True, parents=True)

    # Load test data
    data_dir = Path("dataset/task2")
    test_dataset = SensorDataset(data_dir / 'test_data_final.h5', 'test')  # Update path as needed
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, pin_memory=True)

    # Class names for spatial bins
    class_names = ['Bin 0', 'Bin 1', 'Bin 2', 'Bin 3']

    # Find all model checkpoints
    logger.info("Finding model checkpoints...")
    checkpoint_files = find_model_checkpoints('pretrained')
    logger.info(f"Found {len(checkpoint_files)} model(s) to evaluate")

    # Evaluate all models
    all_results = []
    
    for i, model_path in enumerate(checkpoint_files):
        seed, run_id = extract_seed_info(model_path)
        logger.info(f"Evaluating model {i+1}/{len(checkpoint_files)}: {Path(model_path).name}")
        
        # Evaluate model
        result = evaluate_single_model(model_path, test_loader, device, class_names)
        result['seed'] = seed
        result['run_id'] = run_id
        result['model_path'] = str(model_path)
        
        all_results.append(result)
        
        # Save individual confusion matrix
        if seed is not None:
            cm_path = results_dir / f'confusion_matrix_spatial_seed_{seed}_run_{run_id}.png'
            title_suffix = f' (Seed {seed}, Run {run_id})'
        else:
            cm_path = results_dir / f'confusion_matrix_spatial_model_{i+1}.png'
            title_suffix = f' (Model {i+1})'
        
        plot_confusion_matrix(result['y_true'], result['y_pred'], class_names, 
                            cm_path, title_suffix)
        
        logger.info(f"Model accuracy: {result['accuracy']:.2f}%")

    # Generate aggregate visualizations and reports
    logger.info("Generating aggregate results...")
    
    # Extract detailed metrics for analysis
    individual_results = [r for r in all_results if r.get('seed') != 'Ensemble']
    if len(individual_results) > 1:
        logger.info("Extracting detailed metrics...")
        metrics_data = extract_detailed_metrics(individual_results, class_names)
        
        # Create box plots
        logger.info("Creating box plots...")
        plot_box_metrics(metrics_data, class_names, results_dir)
        
        # Create detailed metrics summary
        logger.info("Creating metrics summary table...")
        create_metrics_summary_table(metrics_data, class_names, 
                                    results_dir / 'detailed_metrics_summary_spatial.txt')
        
        # Plot accuracy comparison
        plot_accuracy_comparison(individual_results, results_dir / 'accuracy_comparison_spatial.png')
    else:
        logger.info("Single model detected - skipping multi-model analysis")
    
    # Create ensemble confusion matrix (majority voting if multiple models)
    if len(all_results) > 1:
        logger.info("Creating ensemble confusion matrix...")
        all_preds = np.array([result['y_pred'] for result in all_results])
        # Majority voting ensemble
        ensemble_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 
                                           axis=0, arr=all_preds)
        ensemble_accuracy = (ensemble_preds == all_results[0]['y_true']).mean() * 100
        
        plot_confusion_matrix(all_results[0]['y_true'], ensemble_preds, class_names,
                            results_dir / 'confusion_matrix_spatial_ensemble.png', 
                            f' (Ensemble - {ensemble_accuracy:.2f}%)')
        
        # Add ensemble results
        ensemble_report = classification_report(all_results[0]['y_true'], ensemble_preds, 
                                              target_names=class_names, digits=4, output_dict=True)
        all_results.append({
            'y_pred': ensemble_preds,
            'y_true': all_results[0]['y_true'],
            'accuracy': ensemble_accuracy,
            'report': ensemble_report,
            'seed': 'Ensemble',
            'run_id': 'Ensemble',
            'model_path': 'Ensemble (Majority Voting)'
        })

    # Save comprehensive results
    save_comprehensive_results(all_results, results_dir, class_names)

    # Print final summary
    logger.info(f"{'='*60}")
    logger.info("EVALUATION COMPLETE")
    logger.info(f"{'='*60}")
    
    accuracies = [r['accuracy'] for r in all_results if r.get('seed') != 'Ensemble']
    if len(accuracies) > 1:
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        logger.info(f"Models evaluated: {len(accuracies)}")
        logger.info(f"Mean accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
        logger.info(f"Best model: {max(accuracies):.2f}%")
        logger.info(f"Worst model: {min(accuracies):.2f}%")
        
        if 'Ensemble' in [r.get('seed') for r in all_results]:
            ensemble_acc = [r['accuracy'] for r in all_results if r.get('seed') == 'Ensemble'][0]
            logger.info(f"Ensemble accuracy: {ensemble_acc:.2f}%")
    else:
        logger.info(f"Single model accuracy: {accuracies[0]:.2f}%")
    
    logger.info(f"Results saved in: {results_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise