import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

def get_model_and_dataset(filename):
    # Example: bert-base-uncased_bc5cdr_training_results.csv
    base = os.path.basename(filename)
    if base.endswith("_training_results.csv"):
        parts = base.replace("_training_results.csv", "").split("_")
        model = "_".join(parts[:-1])
        dataset = parts[-1]
        return model, dataset
    return None, None

def plot_all_losses(results_dir="../output"):
    csv_paths = os.path.join(results_dir, "bert-base-uncased_*_training_results.csv")
    print(csv_paths)
    csv_files = glob.glob(csv_paths)
    if not csv_files:
        print("No *_training_results.csv files found.")
        return

    # Organize files by model
    model_files = {}
    for csv_file in csv_files:
        model, dataset = get_model_and_dataset(csv_file)
        if model is None:
            continue
        model_files.setdefault(model, []).append((dataset, csv_file))

    # Plot for each model
    for model, files in model_files.items():
        plt.figure(figsize=(10, 6))
        datasets = sorted([dataset for dataset, _ in files])
        colors = plt.cm.tab10.colors  # or use any colormap you like
        dataset2color = {ds: colors[i % len(colors)] for i, ds in enumerate(datasets)}
        for dataset, csv_file in sorted(files):
            df = pd.read_csv(csv_file)
            if 'epoch' not in df.columns or 'train_loss' not in df.columns or 'eval_loss' not in df.columns:
                continue
            color = dataset2color[dataset]
            plt.plot(df['epoch'], df['train_loss'], label=f"{dataset} train", color=color)
            plt.plot(df['epoch'], df['eval_loss'], linestyle='--', label=f"{dataset} eval", color=color)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Train/Eval Losses for {model}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{model}_all_losses.png"))
        plt.close()
        print(f"Saved plot for {model}")

if __name__ == "__main__":
    plot_all_losses()