import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import json
import argparse
import sys
import logging

# Add parent directory to Python path to access src module
sys.path.append(str(Path(__file__).parent.parent))
from src.model_task2_dist import DistanceRegressor
from src.task2_components import DistanceDataset

# --- Matplotlib/Seaborn style ---
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

colors = {
    'primary': '#d1e6ff',
    'accent': '#ffe6d1',
    'harmonious1': '#d1ffff',
    'harmonious2': '#e6d1ff',
    'contrast': '#aed2f0',
    'border': '#4d545d'
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_best_model(model, checkpoint_path, device):
    """Load the best model from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
        logger.info(f"Validation metrics: Val Dist Error: {checkpoint['val_metrics']['dist_error']:.4f}")
        return model, checkpoint
    except Exception as e:
        raise RuntimeError(f"Error loading model from {checkpoint_path}: {str(e)}")

@torch.no_grad()
def evaluate_model(model, test_loader, device):
    """Evaluate model performance on test set"""
    model.eval()
    all_dist_preds = []
    all_dist_targets = []

    for batch in test_loader:
        x = batch['x'].to(device, non_blocking=True)
        distances = batch['distance'].to(device, non_blocking=True)
        preds = model(x)
        all_dist_preds.extend(preds.cpu().numpy())
        all_dist_targets.extend(distances.cpu().numpy())

    avg_dist_error = np.mean(np.abs(np.array(all_dist_preds) - np.array(all_dist_targets)))
    return {
        'dist_predictions': np.array(all_dist_preds),
        'dist_targets': np.array(all_dist_targets),
        'avg_dist_error': avg_dist_error
    }

# ---- Plotting functions ----

def plot_distance_metrics_boxplot(all_fungi_metrics, fungi_names, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    box_colors = [colors['primary'], colors['accent'], colors['harmonious1'], colors['harmonious2'], colors['contrast']]
    border_color = colors['border']
    # MAE
    mae_data = [all_fungi_metrics[name]['mae'] for name in fungi_names]
    box_plot = axes[0].boxplot(mae_data, positions=range(len(fungi_names)), patch_artist=True, widths=0.6)
    for patch, color in zip(box_plot['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)
        patch.set_edgecolor(border_color)
        patch.set_linewidth(1.5)
    for element in ['whiskers', 'fliers', 'caps']:
        plt.setp(box_plot[element], color=border_color, linewidth=1.5)
    plt.setp(box_plot['medians'], color=border_color, linewidth=2.5)
    axes[0].set_title('Mean Absolute Error (MAE)', **TITLE_FONT)
    axes[0].set_ylabel('MAE', **LABEL_FONT)
    axes[0].set_xlabel('Fungi Type', **LABEL_FONT)
    axes[0].set_xticks(range(len(fungi_names)))
    axes[0].set_xticklabels(fungi_names)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3, axis='y', color=border_color)
    # MSE
    mse_data = [all_fungi_metrics[name]['mse'] for name in fungi_names]
    box_plot = axes[1].boxplot(mse_data, positions=range(len(fungi_names)), patch_artist=True, widths=0.6)
    for patch, color in zip(box_plot['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)
        patch.set_edgecolor(border_color)
        patch.set_linewidth(1.5)
    for element in ['whiskers', 'fliers', 'caps']:
        plt.setp(box_plot[element], color=border_color, linewidth=1.5)
    plt.setp(box_plot['medians'], color=border_color, linewidth=2.5)
    axes[1].set_title('Mean Squared Error (MSE)', **TITLE_FONT)
    axes[1].set_ylabel('MSE', **LABEL_FONT)
    axes[1].set_xlabel('Fungi Type', **LABEL_FONT)
    axes[1].set_xticks(range(len(fungi_names)))
    axes[1].set_xticklabels(fungi_names)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3, axis='y', color=border_color)
    # RMSE
    rmse_data = [all_fungi_metrics[name]['rmse'] for name in fungi_names]
    box_plot = axes[2].boxplot(rmse_data, positions=range(len(fungi_names)), patch_artist=True, widths=0.6)
    for patch, color in zip(box_plot['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)
        patch.set_edgecolor(border_color)
        patch.set_linewidth(1.5)
    for element in ['whiskers', 'fliers', 'caps']:
        plt.setp(box_plot[element], color=border_color, linewidth=1.5)
    plt.setp(box_plot['medians'], color=border_color, linewidth=2.5)
    axes[2].set_title('Root Mean Squared Error (RMSE)', **TITLE_FONT)
    axes[2].set_ylabel('RMSE', **LABEL_FONT)
    axes[2].set_xlabel('Fungi Type', **LABEL_FONT)
    axes[2].set_xticks(range(len(fungi_names)))
    axes[2].set_xticklabels(fungi_names)
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3, axis='y', color=border_color)
    for ax in axes:
        ax.tick_params(axis='x', labelsize=14, labelrotation=0)
        ax.tick_params(axis='y', labelsize=14, labelrotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / 'distance_metrics_boxplot_all_fungi.png', bbox_inches='tight', dpi=300)
    plt.close()

# --- Main evaluation loop ---

def evaluate_all_fungi_types():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    fungi_types = [0, 1, 2, 3, 4]
    fungi_names = ['X.510', 'P.toxicarum', 'P.513', 'T.508', 'B.adusta']
    all_fungi_metrics = {fn: {'mae': [], 'mse': [], 'rmse': [], 'r2': [], 'slope': []} for fn in fungi_names}
    results_dir = Path("Results/task2_distance_metrics")
    results_dir.mkdir(exist_ok=True)

    for fungi_idx, fungi_name in zip(fungi_types, fungi_names):
        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATING FUNGI TYPE: {fungi_name} (Index: {fungi_idx})")
        logger.info(f"{'='*60}")

        filter_tag = f"_fungi{fungi_idx}"
        try:
            data_dir = Path("dataset/task2")
            test_dataset = DistanceDataset(data_dir / 'test_data_final.h5', 'test', fungi_filter=fungi_idx)
            test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, pin_memory=True)
        except Exception as e:
            logger.error(f"Error loading test data for {fungi_name}: {str(e)}")
            continue

        checkpoint_base_dir = Path(f'pretrained/task2_dist/checkpoints_multiseed{filter_tag}')
        if not checkpoint_base_dir.exists():
            logger.warning(f"Checkpoint directory not found: {checkpoint_base_dir}")
            continue
        seed_dirs = [d for d in checkpoint_base_dir.iterdir() if d.is_dir() and d.name.startswith('seed_')]
        seed_dirs = sorted(seed_dirs, key=lambda x: int(x.name.split('_')[1]))
        if not seed_dirs:
            logger.warning(f"No seed directories found for {fungi_name}")
            continue

        logger.info(f"Found {len(seed_dirs)} seed directories for {fungi_name}")
        fungi_seed_metrics = []
        for seed_dir in seed_dirs:
            try:
                checkpoint_files = list(seed_dir.glob(f'best_model_dist{filter_tag}_run*_seed*.pt'))
                if not checkpoint_files:
                    logger.warning(f"No checkpoint found in {seed_dir}")
                    continue
                checkpoint_path = checkpoint_files[0]
                model = DistanceRegressor(
                    input_dim=9,
                    hidden_dim=128,
                    num_layers=2,
                    dropout=0.2,
                    use_bidirectional=True
                ).to(device)
                model, checkpoint = load_best_model(model, checkpoint_path, device)
                results = evaluate_model(model, test_loader, device)
                preds = results['dist_predictions']
                targets = results['dist_targets']
                mae = mean_absolute_error(targets, preds)
                mse = mean_squared_error(targets, preds)
                rmse = np.sqrt(mse)
                r2 = r2_score(targets, preds)
                slope = np.polyfit(targets, preds, 1)[0]
                seed_metrics = {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2, 'slope': slope}
                fungi_seed_metrics.append(seed_metrics)
                logger.info(f"  {seed_dir.name}: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, Slope={slope:.4f}")
            except Exception as e:
                logger.error(f"Error evaluating {seed_dir.name} for {fungi_name}: {str(e)}")
                continue

        if fungi_seed_metrics:
            for metric in ['mae', 'mse', 'rmse', 'r2', 'slope']:
                all_fungi_metrics[fungi_name][metric] = [seed[metric] for seed in fungi_seed_metrics]
            logger.info(f"Successfully evaluated {len(fungi_seed_metrics)} seeds for {fungi_name}")
        else:
            logger.warning(f"No successful evaluations for {fungi_name}")

    logger.info(f"\n{'='*60}")
    logger.info("GENERATING DISTANCE METRICS VISUALIZATIONS")
    logger.info(f"{'='*60}")
    valid_fungi = [name for name in fungi_names if len(all_fungi_metrics[name]['mae']) > 0]
    if len(valid_fungi) == 0:
        logger.error("No valid fungi data found for plotting!")
        return
    filtered_metrics = {name: all_fungi_metrics[name] for name in valid_fungi}
    plot_distance_metrics_boxplot(filtered_metrics, valid_fungi, results_dir)
    with open(results_dir / 'distance_metrics_summary.txt', 'w') as f:
        f.write("DISTANCE PREDICTION METRICS ACROSS ALL FUNGI TYPES\n")
        f.write("=" * 80 + "\n\n")
        for fungi_name in valid_fungi:
            f.write(f"{fungi_name}:\n")
            f.write("-" * 40 + "\n")
            metrics = filtered_metrics[fungi_name]
            for metric_name in ['mae', 'mse', 'rmse', 'r2', 'slope']:
                values = metrics[metric_name]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    f.write(f"  {metric_name.upper()}: {mean_val:.6f} ± {std_val:.6f}\n")
            f.write("\n")
    detailed_data = []
    for fungi_name in valid_fungi:
        metrics = filtered_metrics[fungi_name]
        for i in range(len(metrics['mae'])):
            row = {
                'fungi_type': fungi_name,
                'seed_index': i,
                'mae': metrics['mae'][i],
                'mse': metrics['mse'][i],
                'rmse': metrics['rmse'][i],
                'r2': metrics['r2'][i],
                'slope': metrics['slope'][i]
            }
            detailed_data.append(row)
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(results_dir / 'detailed_distance_metrics.csv', index=False)
    logger.info(f"Distance metrics analysis complete!")
    logger.info(f"Results saved in: {results_dir}")
    logger.info(f"Generated files:")
    logger.info(f"- distance_metrics_boxplot_all_fungi.png: Traditional box plots")
    logger.info(f"- distance_metrics_summary.txt: Summary statistics")
    logger.info(f"- detailed_distance_metrics.csv: Raw data")

if __name__ == "__main__":
    try:
        evaluate_all_fungi_types()
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
