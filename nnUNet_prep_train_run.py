import subprocess
import os

# ========================================================================================================================================================================
# CODE


task_id = "001" # Numerical ID from folder name: Dataset001_PerfusionTerritories
task_name = f"Dataset{task_id}_PerfusionTerritories"
trainer = "nnU"


# -------------------------------------------------- #
# Prepare Dataset
# -------------------------------------------------- #

def prep_n_train():
 
    print("\n✔ Checking dataset integrity...")
    try:
        subprocess.run(
            ["nnUNetv2_plan_and_preprocess", "-d", task_id, "--verify_dataset_integrity"],
            check=True
        )
        print("✔ Dataset integrity verified.")
    except subprocess.CalledProcessError as e:
        print(f"✘ Dataset verification failed:\n{e}")
        exit(1)

    print("\n✔ Running planner and preprocessing...")
    try:
        subprocess.run(
            ["nnUNetv2_plan_and_preprocess", "-d", task_id],
            check=True
        )
        print("✔ Planning and preprocessing done.")
    except subprocess.CalledProcessError as e:
        print(f"✘ Planning failed:\n{e}")
        exit(1)



# -------------------------------------------------- #
# Train nnUNet
# -------------------------------------------------- #

    trainer = "2d"
    num_folds = 5

    print("\n🚀 Starting nnU-Net 5-fold cross-validation training...")

    for fold in range(num_folds):
        print(f"\n--- Training fold {fold} ---")
        command = [
            "nnUNetv2_train",
            task_id,
            trainer,
            str(fold)
        ]
        try:
            subprocess.run(command, check=True)
            print(f"✔ Fold {fold} completed.")
        except subprocess.CalledProcessError as e:
            print(f"✘ Fold {fold} failed with error:\n{e}")
            break

    print("\n✅ All requested folds have been processed.")



# -------------------------------------------------- #
# Run inference
# -------------------------------------------------- #

def infer():
    fold_to_infer = "4"  # You can change this to the best-performing fold (e.g., "0", "1", ..., "4", or "ensemble")
    input_folder = os.path.join(os.environ["nnUNet_raw"], task_name, "imagesTs")
    output_folder = os.path.join("DLSegPerf", "inference_results")

    print(f"\n🧠 Running inference on imagesTs using fold {fold_to_infer}...")

    command = [
        "nnUNetv2_predict",
        "-d", task_id,
        "-tr", trainer,
        "-f", fold_to_infer,
        "-i", input_folder,
        "-o", output_folder,
        "-p", "2d"
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✔ Inference completed. Results saved to: {output_folder}")
    except subprocess.CalledProcessError as e:
        print(f"✘ Inference failed:\n{e}")

    # The output will be saved under: DLSegPerf/inference_results/


# ========================================================================================================================================================================
# EXECUTION

if __name__ == "__main__":
    run_prep_n_train = True
    run_infer = False

    if run_prep_n_train:
        prep_n_train()

    if run_infer:
        infer()
