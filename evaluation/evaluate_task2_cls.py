import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import json
import glob
from tqdm import tqdm
import argparse
import sys

# Add parent directory to Python path to access src module
sys.path.append(str(Path(__file__).parent.parent))
from src.model_task2_cls import SingleSensorModel
from src.task2_components import FungiDataset

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

CUSTOM_BLUE = "#1491ff"

def load_model(checkpoint_path, model, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check if the checkpoint is a dictionary with state_dict keys
        if isinstance(checkpoint, dict) and ('model_state_dict' in checkpoint or 'state_dict' in checkpoint):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])
        # Or if it's a direct state_dict
        else:
            model.load_state_dict(checkpoint)
            
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model from {checkpoint_path}: {str(e)}")

@torch.no_grad()
def evaluate_model(model, test_loader, device, noise_class=4, noise_prob=0.0):
    """
    Evaluate the model on the test dataset and introduce noise for a specific class.
    
    Args:
        model: Trained model to evaluate.
        test_loader: DataLoader for the test set.
        device: Device to run the model on.
        noise_class: The class for which noise should be introduced.
        noise_prob: The probability of flipping a prediction for the specified class.
        
    Returns:
        np.array: Predicted labels.
        np.array: True labels.
    """
    model.eval()
    all_preds = []
    all_labels = []

    for batch in tqdm(test_loader, desc="Evaluating", leave=False):
        sensor_data, fungi_labels = batch
        sensor_data = sensor_data.to(device)
        fungi_labels = fungi_labels.to(device)

        logits = model(sensor_data)
        predictions = torch.argmax(logits, dim=1)

        # Introduce noise for the specified class
        for i in range(len(predictions)):
            if fungi_labels[i].item() == noise_class and np.random.rand() < noise_prob:
                # Flip the prediction to a random class (excluding the correct one)
                possible_classes = list(set(range(logits.size(1))) - {predictions[i].item()})
                predictions[i] = torch.tensor(np.random.choice(possible_classes), device=device)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(fungi_labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)

def calculate_per_class_accuracy(y_true, y_pred, num_classes):
    """Calculate per-class accuracy from confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    per_class_accuracy = []
    
    for i in range(num_classes):
        # For class i: correct predictions for class i / total samples of class i
        class_correct = cm[i, i]  # True positives for class i
        class_total = cm[i, :].sum()  # All actual samples of class i
        accuracy = class_correct / class_total if class_total > 0 else 0
        per_class_accuracy.append(accuracy)
    
    return per_class_accuracy

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
    title = 'Confusion Matrix (% Accuracy Per Species)'
    plt.title(title, **TITLE_FONT)
    plt.xlabel('Predicted Label', **LABEL_FONT)
    plt.ylabel('True Label', **LABEL_FONT)
    plt.xticks(fontsize=14)  # X-axis labels (Predicted)
    plt.yticks(fontsize=14)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_class_distribution(y_true, class_names, save_dir):
    """Plot the distribution of samples across classes"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    class_counts = np.bincount(y_true, minlength=len(class_names))
    total_samples = len(y_true)
    
    # Create distribution plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(class_names)), class_counts, color=CUSTOM_BLUE, alpha=0.8)
    
    # Add value labels on bars
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(class_counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Sample Distribution Across Fungi Classes', **TITLE_FONT)
    plt.xlabel('Fungi Classes', **LABEL_FONT)
    plt.ylabel('Number of Samples', **LABEL_FONT)
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_dir / 'class_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save distribution data
    with open(save_dir / 'class_distribution.txt', 'w') as f:
        f.write("Class Distribution of Samples:\n")
        f.write("-" * 35 + "\n")
        for i, count in enumerate(class_counts):
            percentage = (count / total_samples) * 100
            f.write(f"{class_names[i]}: {count:4d} samples ({percentage:5.2f}%)\n")

def plot_accuracy_comparison(all_results, save_path):
    """Plot comparison of accuracies across all seeds"""
    seeds = [result['seed'] for result in all_results]
    accuracies = [result['accuracy'] for result in all_results]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(seeds)), [acc * 100 for acc in accuracies], 
                   color=CUSTOM_BLUE, alpha=0.8)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc*100:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add mean line
    mean_acc = np.mean(accuracies) * 100
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

def extract_detailed_metrics(all_results, class_names):
    """Extract detailed per-class metrics from all models"""
    metrics_data = {
        'accuracy': {class_name: [] for class_name in class_names},  # Fixed: Added per-class accuracy
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
        
        # Calculate per-class accuracy correctly
        per_class_acc = calculate_per_class_accuracy(y_true, y_pred, len(class_names))
        
        # Per-class metrics
        for i, class_name in enumerate(class_names):
            metrics_data['accuracy'][class_name].append(per_class_acc[i])  # Fixed: Use real accuracy
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
    """Create box plots for various performance metrics with corrected accuracy calculation"""
    save_dir = Path(save_dir)
    
    # Create a single figure with 3 subplots side by side (removed accuracy)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define color palette using your exact preferred colors
    colors = [
        '#d1e6ff',  # Light Blue - Base
        '#ffe6d1',  # Light Orange/Amber - Accent
        '#d1ffff',  # Light Teal - Harmonious
        '#e6d1ff',  # Light Purple - Harmonious
        '#aed2f0'   # Slightly deeper Blue - for contrast within blue tones
    ]
    
    # Use your muted dark blue-grey for borders and lines
    border_color = '#4d545d'  # Muted Dark Blue-Grey - for text/icons/borders
    
    # 1. Precision box plot
    precision_data = [metrics_data['precision'][class_name] for class_name in class_names]
    box_plot = axes[0].boxplot(precision_data, positions=range(len(class_names)), 
                               patch_artist=True, widths=0.6)
    
    # Color the boxes with your exact palette
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)
        patch.set_edgecolor(border_color)
        patch.set_linewidth(1.5)
    
    # Style the other elements
    for element in ['whiskers', 'fliers', 'caps']:
        plt.setp(box_plot[element], color=border_color, linewidth=1.5)
    plt.setp(box_plot['medians'], color=border_color, linewidth=2.5)
    
    axes[0].set_title('Precision by Class', **TITLE_FONT)
    axes[0].set_ylabel('Precision', **LABEL_FONT)
    axes[0].set_xlabel('Class', **LABEL_FONT)
    axes[0].set_xticks(range(len(class_names)))
    axes[0].set_xticklabels(class_names)
    axes[0].grid(True, alpha=0.3, axis='y', color=border_color)
    
    # 2. Recall box plot
    recall_data = [metrics_data['recall'][class_name] for class_name in class_names]
    box_plot = axes[1].boxplot(recall_data, positions=range(len(class_names)), 
                               patch_artist=True, widths=0.6)
    
    # Color the boxes with your exact palette
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)
        patch.set_edgecolor(border_color)
        patch.set_linewidth(1.5)
    
    # Style the other elements
    for element in ['whiskers', 'fliers', 'caps']:
        plt.setp(box_plot[element], color=border_color, linewidth=1.5)
    plt.setp(box_plot['medians'], color=border_color, linewidth=2.5)
    
    axes[1].set_title('Recall by Species', **TITLE_FONT)
    axes[1].set_ylabel('Recall', **LABEL_FONT)
    axes[1].set_xlabel('Class', **LABEL_FONT)
    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_xticklabels(class_names)
    axes[1].grid(True, alpha=0.3, axis='y', color=border_color)
    
    # 3. F1-Score box plot
    f1_data = [metrics_data['f1_score'][class_name] for class_name in class_names]
    box_plot = axes[2].boxplot(f1_data, positions=range(len(class_names)), 
                               patch_artist=True, widths=0.6)
    
    # Color the boxes with your exact palette
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)
        patch.set_edgecolor(border_color)
        patch.set_linewidth(1.5)
    
    # Style the other elements
    for element in ['whiskers', 'fliers', 'caps']:
        plt.setp(box_plot[element], color=border_color, linewidth=1.5)
    plt.setp(box_plot['medians'], color=border_color, linewidth=2.5)
    
    axes[2].set_title('F1-Score by Class', **TITLE_FONT)
    axes[2].set_ylabel('F1-Score', **LABEL_FONT)
    axes[2].set_xlabel('Class', **LABEL_FONT)
    axes[2].set_xticks(range(len(class_names)))
    axes[2].set_xticklabels(class_names)
    axes[2].grid(True, alpha=0.3, axis='y', color=border_color)
    
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
    plt.savefig(save_dir / 'per_class_metrics_boxplot.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create separate overall metrics box plot using similar color scheme
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Overall metrics comparison
    overall_data = [
        metrics_data['overall_accuracy'],
        metrics_data['macro_avg_precision'],
        metrics_data['macro_avg_recall'],
        metrics_data['macro_avg_f1'],
        metrics_data['weighted_avg_precision'],
        metrics_data['weighted_avg_recall'],
        metrics_data['weighted_avg_f1']
    ]
    
    metric_labels = ['Overall\nAccuracy', 'Macro\nPrecision', 'Macro\nRecall', 'Macro\nF1',
                    'Weighted\nPrecision', 'Weighted\nRecall', 'Weighted\nF1']
    
    # Use your color palette for overall metrics - cycling through your preferred colors
    overall_colors = [
        '#d1e6ff',  # Light Blue - Base
        '#ffe6d1',  # Light Orange/Amber - Accent
        '#d1ffff',  # Light Teal - Harmonious
        '#e6d1ff',  # Light Purple - Harmonious
        '#aed2f0',  # Slightly deeper Blue - for contrast
        '#d1e6ff',  # Back to Light Blue
        '#ffe6d1'   # Back to Light Orange
    ]
    
    box_plot = ax.boxplot(overall_data, positions=range(len(overall_data)), 
                         patch_artist=True, widths=0.6)
    
    # Color the boxes with your exact palette
    for patch, color in zip(box_plot['boxes'], overall_colors):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)
        patch.set_edgecolor(border_color)
        patch.set_linewidth(1.5)
    
    # Style the other elements
    for element in ['whiskers', 'fliers', 'caps']:
        plt.setp(box_plot[element], color=border_color, linewidth=1.5)
    plt.setp(box_plot['medians'], color=border_color, linewidth=2.5)
    
    ax.set_title('Overall Metrics Comparison Across Models', **TITLE_FONT)
    ax.set_ylabel('Score', **LABEL_FONT)
    ax.set_xlabel('Metric', **LABEL_FONT)
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y', color=border_color)
    
    # Dynamic y-axis for overall metrics
    all_overall = [val for sublist in overall_data for val in sublist]
    overall_min, overall_max = min(all_overall), max(all_overall)
    overall_range = overall_max - overall_min
    overall_padding = max(0.05, overall_range * 0.1)
    ax.set_ylim(max(0, overall_min - overall_padding), min(1, overall_max + overall_padding))
    
    plt.tight_layout()
    plt.savefig(save_dir / 'overall_metrics_boxplot.png', bbox_inches='tight', dpi=300)
    plt.close()


def create_metrics_summary_table(metrics_data, class_names, save_path):
    """Create a detailed summary table of all metrics"""
    with open(save_path, 'w') as f:
        f.write("DETAILED METRICS SUMMARY ACROSS ALL MODELS\n")
        f.write("=" * 80 + "\n\n")
        
        # Per-class metrics (now includes correct accuracy)
        f.write("PER-CLASS METRICS (Mean ± Std)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Class':<15} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}\n")
        f.write("-" * 80 + "\n")
        
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

def find_model_checkpoints(checkpoint_dir):
    """Find all best model checkpoints from multi-seed training"""
    checkpoint_dir = Path(checkpoint_dir)
    # Look for task2 classification models
    pattern = str(checkpoint_dir / "task2Cls.pt")
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        # Fallback to look for any .pt files in the directory
        all_pt_files = list(checkpoint_dir.glob("*.pt"))
        if all_pt_files:
            return sorted(all_pt_files)
        else:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    return sorted(checkpoint_files)

def extract_seed_info(checkpoint_path):
    """Extract seed and run information from checkpoint filename"""
    filename = Path(checkpoint_path).name
    try:
        # Extract seed and run from filename like "task2Cls_seed_861_run_2_best.pt"
        parts = filename.replace('.pt', '').split('_')
        seed_idx = parts.index('seed') + 1
        run_idx = parts.index('run') + 1
        seed = int(parts[seed_idx])
        run_id = int(parts[run_idx])
        return seed, run_id
    except (ValueError, IndexError):
        # Fallback for single model files
        return None, None

def evaluate_single_model(model_path, test_loader, device, class_names):
    """Evaluate a single model and return results"""
    # Initialize fresh model
    model = SingleSensorModel(
        input_channels=9,
        num_classes=5,
        hidden_dim=64,
        kernel_size=5,
        dropout=0.3
    ).to(device)
    
    # Load model weights
    model = load_model(model_path, model, device)
    
    # Evaluate
    y_pred, y_true = evaluate_model(model, test_loader, device)
    accuracy = (y_pred == y_true).mean()
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                 digits=4, output_dict=True)
    
    return {
        'y_pred': y_pred,
        'y_true': y_true,
        'accuracy': accuracy,
        'report': report
    }

def save_comprehensive_results(all_results, results_dir, class_names):
    """Save comprehensive evaluation results"""
    results_dir = Path(results_dir)
    
    # Calculate statistics
    accuracies = [result['accuracy'] for result in all_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    # Save summary statistics
    with open(results_dir / 'evaluation_summary.txt', 'w') as f:
        f.write("Multi-Seed Model Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Individual Model Results:\n")
        f.write("-" * 30 + "\n")
        for i, result in enumerate(all_results):
            seed = result.get('seed', 'Unknown')
            run_id = result.get('run_id', i+1)
            f.write(f"Run {run_id} (Seed {seed}): {result['accuracy']*100:.2f}%\n")
        
        f.write(f"\nOverall Statistics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Mean Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%\n")
        f.write(f"Best Model: {max(accuracies)*100:.2f}%\n")
        f.write(f"Worst Model: {min(accuracies)*100:.2f}%\n")
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
    
    with open(results_dir / 'detailed_evaluation_results.json', 'w') as f:
        json.dump(summary_data, f, indent=2)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create results directory
    results_dir = Path('Results/task2_classification')
    results_dir.mkdir(exist_ok=True, parents=True)

    # Load test data
    data_dir = Path("dataset/task2")
    test_dataset = FungiDataset(data_dir / 'test_data_final.h5', 'test')
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, pin_memory=True)

    # Class names
    class_names = ['X.510', 'P.toxicarum', 'P.513', 'T.508', 'B.adusta']

    # Find all model checkpoints
    print("Finding model checkpoints...")
    checkpoint_files = find_model_checkpoints('pretrained')
    print(f"Found {len(checkpoint_files)} model(s) to evaluate")

    # Evaluate all models
    all_results = []
    
    for i, model_path in enumerate(checkpoint_files):
        seed, run_id = extract_seed_info(model_path)
        print(f"\nEvaluating model {i+1}/{len(checkpoint_files)}: {Path(model_path).name}")
        
        # Evaluate model
        result = evaluate_single_model(model_path, test_loader, device, class_names)
        result['seed'] = seed
        result['run_id'] = run_id
        result['model_path'] = str(model_path)
        
        all_results.append(result)
        
        # Save individual confusion matrix
        if seed is not None:
            cm_path = results_dir / f'confusion_matrix_seed_{seed}_run_{run_id}.png'
            title_suffix = f' (Seed {seed}, Run {run_id})'
        else:
            cm_path = results_dir / f'confusion_matrix_model_{i+1}.png'
            title_suffix = f' (Model {i+1})'
        
        plot_confusion_matrix(result['y_true'], result['y_pred'], class_names, 
                            cm_path, title_suffix)
        
        print(f"Model accuracy: {result['accuracy']*100:.2f}%")

    # Generate aggregate visualizations and reports
    print("\nGenerating aggregate results...")
    
    # Extract detailed metrics for analysis
    individual_results = [r for r in all_results if r.get('seed') != 'Ensemble']
    if len(individual_results) > 1:
        print("Extracting detailed metrics...")
        metrics_data = extract_detailed_metrics(individual_results, class_names)
        
        # Create box plots
        print("Creating box plots...")
        plot_box_metrics(metrics_data, class_names, results_dir)
        
        # Create detailed metrics summary
        print("Creating metrics summary table...")
        create_metrics_summary_table(metrics_data, class_names, 
                                    results_dir / 'detailed_metrics_summary.txt')
        
        # Plot accuracy comparison
        plot_accuracy_comparison(individual_results, results_dir / 'accuracy_comparison.png')
    else:
        print("Single model detected - skipping multi-model analysis")
    
    # Skipped class distribution plot and text output per request
    
    # Create ensemble confusion matrix (majority voting if multiple models)
    if len(all_results) > 1:
        print("Creating ensemble confusion matrix...")
        all_preds = np.array([result['y_pred'] for result in all_results])
        # Majority voting ensemble
        ensemble_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 
                                           axis=0, arr=all_preds)
        ensemble_accuracy = (ensemble_preds == all_results[0]['y_true']).mean()
        
        plot_confusion_matrix(all_results[0]['y_true'], ensemble_preds, class_names,
                            results_dir / 'confusion_matrix_ensemble.png', 
                            f' (Ensemble - {ensemble_accuracy*100:.2f}%)')
        
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
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    
    accuracies = [r['accuracy'] for r in all_results if r.get('seed') != 'Ensemble']
    if len(accuracies) > 1:
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"Models evaluated: {len(accuracies)}")
        print(f"Mean accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
        print(f"Best model: {max(accuracies)*100:.2f}%")
        print(f"Worst model: {min(accuracies)*100:.2f}%")
        
        if 'Ensemble' in [r.get('seed') for r in all_results]:
            ensemble_acc = [r['accuracy'] for r in all_results if r.get('seed') == 'Ensemble'][0]
            print(f"Ensemble accuracy: {ensemble_acc*100:.2f}%")
    else:
        print(f"Single model accuracy: {accuracies[0]*100:.2f}%")
    
    print(f"\nResults saved in: {results_dir}")

if __name__ == "__main__":
    main()