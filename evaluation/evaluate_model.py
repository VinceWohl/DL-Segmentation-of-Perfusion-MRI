# Script to evaluate DL models trained on segmentation of perfusion areas
import os
import json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (safe for servers)
import matplotlib.pyplot as plt



# Modify this path to your results folder root
RESULTS_DIR = "/home/ubuntu/DLSegPerf/data/nnUNet_results/Dataset001_PerfusionTerritories_250522-PerfTerr-06/nnUNetTrainer__nnUNetPlans__2d"
#GT_DIR = "/home/ubuntu/DLSegPerf/data/nnUNet_raw/Dataset001_PerfusionTerritories"


def collect_dice_scores_per_class(results_dir):
    """Collect Dice scores per fold and per class from summary.json files. Automatically detects classes."""
    dice_scores = {}

    for fold in range(5):
        fold_dir = os.path.join(results_dir, f"fold_{fold}", "validation")
        summary_path = os.path.join(fold_dir, "summary.json")
        if not os.path.exists(summary_path):
            print(f"⚠️ Warning: summary.json not found in fold {fold}")
            continue

        with open(summary_path, "r") as f:
            summary = json.load(f)

        for case in summary.get("metric_per_case", []):
            for cls_str, metrics in case["metrics"].items():
                cls = int(cls_str)
                dice = metrics.get("Dice", None)
                if dice is None:
                    continue

                if cls not in dice_scores:
                    dice_scores[cls] = {}
                if f"fold_{fold}" not in dice_scores[cls]:
                    dice_scores[cls][f"fold_{fold}"] = []
                dice_scores[cls][f"fold_{fold}"].append(dice)

    return dice_scores



def plot_dice_violin(dice_scores, save_path=None, title="Validation Dice Scores per Fold"):
    """Generate a violin plot of Dice scores across folds, with enhancements."""
    # Remove empty folds
    filtered_scores = {k: v for k, v in dice_scores.items() if len(v) > 0}
    if len(filtered_scores) == 0:
        print(f"⚠️ Skipping plot: No Dice values to plot for {title}")
        return

    # Combine all dice scores into one for the "All" violin
    all_scores = []
    for fold in sorted(filtered_scores.keys()):
        all_scores.extend(filtered_scores[fold])
    filtered_scores = {"All": all_scores, **filtered_scores}

    fig, ax = plt.subplots(figsize=(10, 5))

    data = [filtered_scores[k] for k in sorted(filtered_scores.keys())]
    positions = np.arange(len(data))

    # Plot the violin
    vp = ax.violinplot(data, positions=positions, showmeans=True, showmedians=False, showextrema=False)

    # X-axis
    ax.set_xticks(positions)
    ax.set_xticklabels(sorted(filtered_scores.keys()))
    ax.set_ylabel("Dice Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Overlay individual points
    for i, d in enumerate(data):
        y = np.array(d)
        x = np.random.normal(loc=positions[i], scale=0.05, size=len(y))  # jitter
        ax.scatter(x, y, alpha=0.5, color='black', s=10)

    # Add mean values
    for i, d in enumerate(data):
        mean_val = np.mean(d)
        ax.text(positions[i] + 0.25, mean_val, f"{mean_val:.3f}", ha='left', va='center', fontsize=10, color='blue')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.tight_layout()


if __name__ == "__main__":
    dice_data_per_class = collect_dice_scores_per_class(RESULTS_DIR)

    output_dir = os.path.join(os.path.dirname(__file__), "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)

    for cls, fold_data in dice_data_per_class.items():
        output_path = os.path.join(output_dir, f"dice_violin_plot_class{cls}.png")
        title = f"Validation Dice Scores per Fold – Class {cls}"
        plot_dice_violin(fold_data, save_path=output_path, title=title)
        total_values = sum(len(v) for v in fold_data.values())
        print(f"✅ Saved plot for class {cls} with {total_values} values.")