# Script to evaluate DL models trained on segmentation of perfusion areas
import os
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# Modify this path to your results folder root
RESULTS_DIR = "/mnt/volume/nnUNet_results/Dataset001_PerfusionTerritories_250522-PerfTerr-06/nnUNetTrainer__nnUNetPlans__2d"



def collect_dice_scores(results_dir):
    """Collect Dice scores per fold from summary.json files."""
    dice_scores = {}
    for fold in range(5):
        fold_dir = os.path.join(results_dir, f"fold_{fold}", "validation")
        summary_path = os.path.join(fold_dir, "summary.json")
        if not os.path.exists(summary_path):
            print(f"Warning: summary.json not found in fold {fold}")
            continue

        with open(summary_path, "r") as f:
            summary = json.load(f)
        
        fold_dices = []
        for case in summary.get("metric_per_case", []):
            dice = case["metrics"]["1"]["Dice"]
            fold_dices.append(dice)

        dice_scores[f"fold_{fold}"] = fold_dices
    return dice_scores



def plot_dice_violin(dice_scores, save_path=None):
    """Generate a violin plot of Dice scores across folds, with enhancements."""
    import numpy as np

    # Combine all dice scores into one for the "All" violin
    all_scores = []
    for fold in sorted(dice_scores.keys()):
        all_scores.extend(dice_scores[fold])
    dice_scores = {"All": all_scores, **dice_scores}

    fig, ax = plt.subplots(figsize=(10, 5))

    data = [dice_scores[k] for k in sorted(dice_scores.keys())]
    positions = np.arange(len(data))

    # Plot the violin
    vp = ax.violinplot(data, positions=positions, showmeans=True, showmedians=False, showextrema=False)

    # Set x-axis
    ax.set_xticks(positions)
    ax.set_xticklabels(sorted(dice_scores.keys()))
    ax.set_ylabel("Dice Score", fontsize=12)
    ax.set_title("Validation Dice Scores per Fold", fontsize=14)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Overlay scatter of actual values
    for i, d in enumerate(data):
        y = np.array(d)
        x = np.random.normal(loc=positions[i], scale=0.05, size=len(y))  # jitter
        ax.scatter(x, y, alpha=0.5, color='black', s=10)

    # Add mean value as text
    for i, d in enumerate(data):
        mean_val = np.mean(d)
        ax.text(positions[i] + 0.25, mean_val, f"{mean_val:.3f}", ha='left', va='center', fontsize=10, color='blue')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()




def find_closest_to_mean_case(summary):
    """Return prediction path of the case with Dice closest to the mean."""
    cases = summary.get("metric_per_case", [])
    mean_dice = np.mean([c["metrics"]["1"]["Dice"] for c in cases])
    closest_case = min(cases, key=lambda c: abs(c["metrics"]["1"]["Dice"] - mean_dice))
    return closest_case["prediction_file"], closest_case["metrics"]["1"]["Dice"]



def plot_nii_slices(image_path, dice, output_path, max_slices=16):
    """Plot a grid of slices from a .nii file and save as PNG."""
    img = nib.load(image_path).get_fdata()
    img = np.squeeze(img)
    
    n_slices = img.shape[2]
    step = max(1, n_slices // max_slices)
    slices = range(0, n_slices, step)[:max_slices]
    
    cols = 4
    rows = int(np.ceil(len(slices) / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    fig.suptitle(f"{os.path.basename(image_path)} (Dice: {dice:.3f})", fontsize=14)
    
    for ax, i in zip(axes.flat, slices):
        ax.imshow(img[:, :, i], cmap="gray")
        ax.set_title(f"Slice {i}")
        ax.axis('off')
    
    for ax in axes.flat[len(slices):]:  # Hide unused axes
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_path, dpi=300)
    plt.close()



if __name__ == "__main__":
    dice_data = collect_dice_scores(RESULTS_DIR)

    # Violin plot:
    
    # Ensure output dir exists
    output_dir = os.path.join(os.path.dirname(__file__), "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "dice_violin_plot.png")
    plot_dice_violin(dice_data, save_path=output_path)

    # Sample slices:

    # for fold in range(5):
    #     summary_path = f"{RESULTS_DIR}/fold_{fold}/validation/summary.json"
    #     with open(summary_path, "r") as f:
    #         summary = json.load(f)
    #     pred_file, dice = find_closest_to_mean_case(summary)
    #     output_file = os.path.join("evaluation_results", f"fold_{fold}_closest_sample.png")
    #     plot_nii_slices(pred_file, dice, output_file)
