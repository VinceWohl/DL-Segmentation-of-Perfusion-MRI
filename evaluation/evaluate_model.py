# Script to evaluate DL models trained on segmentation of perfusion areas
import os
import json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (safe for servers)
import matplotlib.pyplot as plt



# Modify this path to your results folder root
RESULTS_DIR = "/home/ubuntu/DLSegPerf/data/nnUNet_results/Dataset001_PerfusionTerritories/nnUNetTrainer__nnUNetPlans__2d"


def collect_dice_scores_per_class(results_dir):
    """Collect Dice scores per fold and per class from summary.json files."""
    dice_scores = {}
    discovered_class_ids = set()

    # First pass: determine all class IDs across available folds
    for fold in range(5):
        summary_path = os.path.join(results_dir, f"fold_{fold}", "validation", "summary.json")
        if not os.path.exists(summary_path):
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        for case in summary.get("metric_per_case", []):
            discovered_class_ids.update(case["metrics"].keys())

    class_ids = sorted([int(cid) for cid in discovered_class_ids])
    dice_scores = {cls: {} for cls in class_ids}

    # Second pass: collect dice scores
    for fold in range(5):
        fold_dir = os.path.join(results_dir, f"fold_{fold}", "validation")
        summary_path = os.path.join(fold_dir, "summary.json")
        if not os.path.exists(summary_path):
            print(f"⚠️ Warning: summary.json not found in fold {fold}")
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        for cls in class_ids:
            fold_dices = []
            for case in summary.get("metric_per_case", []):
                try:
                    dice = case["metrics"][str(cls)]["Dice"]
                    fold_dices.append(dice)
                except KeyError:
                    continue
            dice_scores[cls][f"fold_{fold}"] = fold_dices

    return dice_scores, class_ids


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
        ax.text(positions[i] + 0.25, mean_val, f"{mean_val:.4f}", ha='left', va='center', fontsize=10, color='blue')


    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)


if __name__ == "__main__":
    dice_data_per_class, class_ids = collect_dice_scores_per_class(RESULTS_DIR)

    output_dir = os.path.join(os.path.dirname(__file__), "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)

    for cls in class_ids:
        print(f"\nClass {cls}:")
        for fold_name, values in dice_data_per_class[cls].items():
            print(f"  {fold_name}: {len(values)} cases")

    for cls, fold_data in dice_data_per_class.items():
        total_values = sum(len(v) for v in fold_data.values())
        if total_values == 0:
            print(f"⚠️ Skipping class {cls} (no values)")
            continue

        output_path = os.path.join(output_dir, f"dice_violin_plot_class{cls}.png")
        title = f"Validation Dice Scores per Fold – Class {cls}"
        plot_dice_violin(fold_data, save_path=output_path, title=title)
        print(f"✅ Saved plot for class {cls} with {total_values} values.")