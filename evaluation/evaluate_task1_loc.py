import torch
import torch.nn as nn
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from torch.utils.data import DataLoader
import torch.nn.functional as F
import glob
from pathlib import Path
import pandas as pd
import json
import logging
from matplotlib.colors import LinearSegmentedColormap
import sys
from pathlib import Path

# Add parent directory to Python path to access src module
sys.path.append(str(Path(__file__).parent.parent))
from src.model_task1_loc import MultiBranchFusion
from src.task1_components import FungiDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# Custom colors
colors = {
    'primary': '#d1e6ff',      # Light Blue - Base
    'accent': '#ffe6d1',       # Light Orange/Amber - Accent  
    'harmonious1': '#d1ffff',  # Light Teal - Harmonious
    'harmonious2': '#e6d1ff',  # Light Purple - Harmonious
    'contrast': '#aed2f0',     # Slightly deeper Blue
    'border': '#4d545d'        # Muted Dark Blue-Grey
}

def load_model(checkpoint_path, model, device):
    """Load model from checkpoint with handling for different checkpoint formats"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract state dict from checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume the checkpoint is the state dict itself
            state_dict = checkpoint
        
        # Handle DataParallel models
        if any(key.startswith('module.') for key in state_dict.keys()):
            logger.info("Detected DataParallel model, removing 'module.' prefix...")
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # Remove 'module.' prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        return model
        
    except Exception as e:
        raise RuntimeError(f"Error loading model from {checkpoint_path}: {str(e)}")

def find_model_checkpoints(checkpoint_dir):
    """Find all best model checkpoints from multi-seed training"""
    checkpoint_dir = Path(checkpoint_dir)
    
    # Look for the pattern from your multi-seed training
    patterns_to_try = [
        str(checkpoint_dir / "task1Loc.pt"),
        str(checkpoint_dir / "best_fungi_bin_classifier_*.pt")  # Fallback to any fungi classifier
    ]
    
    checkpoint_files = []
    for pattern in patterns_to_try:
        files = glob.glob(pattern, recursive=True)
        if files:
            checkpoint_files = files
            logger.info(f"Found checkpoints using pattern: {pattern}")
            break
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    return sorted(checkpoint_files)

def extract_seed_info(checkpoint_path):
    """Extract seed and run information from checkpoint filename"""
    filename = Path(checkpoint_path).name
    try:
        # Handle pattern: best_fungi_classifier_run0_seed12345.pt
        if "run" in filename and "seed" in filename:
            parts = filename.replace('.pt', '').split('_')
            
            # Find run and seed indices
            run_idx = None
            seed_idx = None
            for i, part in enumerate(parts):
                if part.startswith('run'):
                    run_idx = i
                elif part.startswith('seed'):
                    seed_idx = i
            
            if run_idx is not None and seed_idx is not None:
                run_id = int(parts[run_idx][3:])  # Remove 'run' prefix
                seed = int(parts[seed_idx][4:])   # Remove 'seed' prefix
                return seed, run_id
        
        # Fallback parsing
        return None, None
    except (ValueError, IndexError):
        return None, None

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Evaluate model performance on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    
    return all_preds, all_labels, accuracy

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
        annot_kws={'color': 'black', 'size': 14}  # Increased font size and made bold
    )
    
    # Make the tick labels (class names) larger
    plt.xticks(fontsize=14)  # X-axis labels (Predicted)
    plt.yticks(fontsize=14)  # Y-axis labels (True)
    
    # title = f'Confusion Matrix{title_suffix}'
    title = 'Confusion Matrix (% Accuracy Per Bin)'
    plt.title(title, **TITLE_FONT)
    plt.xlabel('Predicted Label', **LABEL_FONT)
    plt.ylabel('True Label', **LABEL_FONT)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_box_metrics(metrics_data, class_names, save_dir):
    """Create box plots for various performance metrics"""
    save_dir = Path(save_dir)
    
    # Create a single figure with 3 subplots (removed redundant accuracy since it equals recall)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # spatial_box_colors = [
    #     '#ff9999',      # Soft Coral - Bin 0
    #     '#b3d9b3',      # Sage Green - Bin 1  
    #     '#d9b3ff',      # Lavender - Bin 2
    #     '#d9c7a3'       # Soft Tan - Bin 3
    # ]
    
    spatial_box_colors = [
        '#e6a3a3',      # Light Pink/Rose - Bin 0
        '#a3a3e6',      # Light Peach - Bin 1  
        '#fff2cc',      # Light Yellow - Bin 2
        '#e6f2e6'       # Light Green - Bin 3
    ]

    border_color = colors['border']
    
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
    
    # 2. Recall box plot (this is per-class accuracy)
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
    axes[1].set_ylabel('Recall/Accuracy', **LABEL_FONT)
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
    plt.savefig(save_dir / 'per_class_metrics_boxplot_fungi.png', bbox_inches='tight', dpi=300)
    plt.close()

def evaluate_single_model(model_path, test_loader, device, class_names):
    """Evaluate a single model and return results"""
    # Initialize fresh model
    model = MultiBranchFusion().to(device)
    
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
        'accuracy': {class_name: [] for class_name in class_names},
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
        
        # Calculate confusion matrix for manual verification
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
        
        # Per-class metrics - using sklearn's classification_report values (which are correct)
        for i, class_name in enumerate(class_names):
            # For per-class accuracy, we use recall (which is the same as per-class accuracy)
            # Per-class accuracy = TP / (TP + FN) = Recall for that class
            # This is because accuracy for a specific class means:
            # "Of all samples that truly belong to this class, how many did we predict correctly?"
            
            # Get metrics directly from sklearn's classification report (guaranteed correct)
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']  # This IS the per-class accuracy
            f1 = report[class_name]['f1-score']
            
            # Manually verify with confusion matrix if needed
            if i < len(cm):
                tp = cm[i, i]  # True positives for class i
                fn = np.sum(cm[i, :]) - tp  # False negatives for class i
                fp = np.sum(cm[:, i]) - tp  # False positives for class i
                tn = np.sum(cm) - tp - fn - fp  # True negatives for class i
                
                # Manual calculation (should match sklearn)
                manual_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                manual_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                manual_f1 = 2 * (manual_precision * manual_recall) / (manual_precision + manual_recall) if (manual_precision + manual_recall) > 0 else 0.0
                
                # Use manual calculations for verification (should be same as sklearn)
                assert abs(precision - manual_precision) < 1e-6, f"Precision mismatch for {class_name}"
                assert abs(recall - manual_recall) < 1e-6, f"Recall mismatch for {class_name}"
                if manual_f1 > 0:  # Only check F1 if it's not zero
                    assert abs(f1 - manual_f1) < 1e-6, f"F1 mismatch for {class_name}"
            
            # Store the verified metrics
            metrics_data['accuracy'][class_name].append(recall)  # Per-class accuracy = recall
            metrics_data['precision'][class_name].append(precision)
            metrics_data['recall'][class_name].append(recall)
            metrics_data['f1_score'][class_name].append(f1)
        
        # Overall metrics (these are correct from sklearn)
        metrics_data['overall_accuracy'].append(report['accuracy'])
        metrics_data['macro_avg_precision'].append(report['macro avg']['precision'])
        metrics_data['macro_avg_recall'].append(report['macro avg']['recall'])
        metrics_data['macro_avg_f1'].append(report['macro avg']['f1-score'])
        metrics_data['weighted_avg_precision'].append(report['weighted avg']['precision'])
        metrics_data['weighted_avg_recall'].append(report['weighted avg']['recall'])
        metrics_data['weighted_avg_f1'].append(report['weighted avg']['f1-score'])
    
    return metrics_data


def plot_accuracy_comparison(all_results, save_path):
    """Plot comparison of accuracies across all seeds"""
    seeds = [result['seed'] for result in all_results if result.get('seed') != 'Ensemble']
    accuracies = [result['accuracy'] for result in all_results if result.get('seed') != 'Ensemble']
    
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

def save_comprehensive_results(all_results, results_dir, class_names):
    """Save comprehensive evaluation results"""
    results_dir = Path(results_dir)
    
    # Calculate statistics
    individual_results = [r for r in all_results if r.get('seed') != 'Ensemble']
    accuracies = [result['accuracy'] for result in individual_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    # Save summary statistics
    with open(results_dir / 'evaluation_summary_fungi.txt', 'w') as f:
        f.write("Multi-Seed Fungi Classification Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Individual Model Results:\n")
        f.write("-" * 30 + "\n")
        for i, result in enumerate(individual_results):
            seed = result.get('seed', 'Unknown')
            run_id = result.get('run_id', i+1)
            f.write(f"Run {run_id} (Seed {seed}): {result['accuracy']:.2f}%\n")
        
        f.write(f"\nOverall Statistics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%\n")
        f.write(f"Best Model: {max(accuracies):.2f}%\n")
        f.write(f"Worst Model: {min(accuracies):.2f}%\n")
        f.write(f"Total Models Evaluated: {len(individual_results)}\n")
        
        # Add ensemble results if available
        ensemble_results = [r for r in all_results if r.get('seed') == 'Ensemble']
        if ensemble_results:
            f.write(f"Ensemble Accuracy: {ensemble_results[0]['accuracy']:.2f}%\n")
    
    # Save detailed results as JSON
    summary_data = {
        'individual_results': [
            {
                'seed': result.get('seed', 'Unknown'),
                'run_id': result.get('run_id', i+1),
                'accuracy': float(result['accuracy']),
                'classification_report': result['report']
            }
            for i, result in enumerate(individual_results)
        ],
        'statistics': {
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'best_accuracy': float(max(accuracies)),
            'worst_accuracy': float(min(accuracies)),
            'num_models': len(individual_results)
        }
    }
    
    # Add ensemble results to JSON if available
    ensemble_results = [r for r in all_results if r.get('seed') == 'Ensemble']
    if ensemble_results:
        summary_data['ensemble_results'] = {
            'accuracy': float(ensemble_results[0]['accuracy']),
            'classification_report': ensemble_results[0]['report']
        }
    
    with open(results_dir / 'detailed_evaluation_results_fungi.json', 'w') as f:
        json.dump(summary_data, f, indent=2)

def main():
    """Main evaluation function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create results directory
    results_dir = Path('Results/task1_localization')
    results_dir.mkdir(exist_ok=True, parents=True)

    # Load test data
    test_h5_file =  'dataset/task1/test/preprocessed_data.h5'
    if not Path(test_h5_file).exists():
        raise FileNotFoundError(f"Test data file not found: {test_h5_file}")
    
    test_dataset = FungiDataset(test_h5_file)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Class names for fungi classification
    class_names = ['Bin 0', 'Bin 1', 'Bin 2', 'Bin 3']

    # Find all model checkpoints from multi-seed training
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
        result['seed'] = seed if seed is not None else f"Model_{i+1}"
        result['run_id'] = run_id if run_id is not None else i+1
        result['model_path'] = str(model_path)
        
        all_results.append(result)
        
        # Save individual confusion matrix
        if seed is not None:
            cm_path = results_dir / f'confusion_matrix_fungi_seed_{seed}_run_{run_id}.png'
            title_suffix = f' (Seed {seed}, Run {run_id})'
        else:
            cm_path = results_dir / f'confusion_matrix_fungi_model_{i+1}.png'
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
        
        # Plot accuracy comparison
        plot_accuracy_comparison(individual_results, results_dir / 'accuracy_comparison_fungi.png')
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
                            results_dir / 'confusion_matrix_fungi_ensemble.png', 
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
    
    individual_results = [r for r in all_results if r.get('seed') != 'Ensemble']
    accuracies = [r['accuracy'] for r in individual_results]
    
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