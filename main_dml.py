"""
### Setup (pip, imports)
!pip install -U datasets
!pip install pytorch-metric-learning
!apt install libomp-dev
!pip install faiss-cpu
"""
import argparse, os, json
import pandas as pd
import torch
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from dml.losses_factory import DML_LossesWrapper
from datasets_utils import load_and_preprocess_dataset, unpack_dataset_info, get_dataloaders, SplitInstanceCollate, NerSlidingWindowReconstructor
from dml.test import NERMetricsCalculator

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

def reshape_embeddings_and_labels(embeddings, labels):
    # Make labels 1D
    labels_reshaped = labels.reshape(-1)
    # Mask out -100 labels
    mask = labels_reshaped != -100
    labels_masked = labels_reshaped[mask]
    # Reshape embeddings to [tokens, embedding_dim]
    emb_reshaped = embeddings.reshape(-1, embeddings.shape[-1])
    # Mask embeddings of -100 labels
    emb_reshaped_masked = emb_reshaped[mask]
    assert emb_reshaped_masked.shape[0] == labels_masked.shape[0], \
        f"emb_reshaped_masked.shape: {emb_reshaped_masked.shape}, labels_masked.shape: {labels_masked.shape}"
    return emb_reshaped_masked, labels_masked

def forward_one_epoch(model, train_loader, optimizer, dml_loss, reconstructor, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.set_grad_enabled(train):
        for batch_idx, (data, instance_ids) in enumerate(train_loader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            if train:
                optimizer.zero_grad()
            embeddings = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            recon_batch = reconstructor.reconstruct_sequences(embeddings, instance_ids, data)
            emb_masked, labels_masked = reshape_embeddings_and_labels(recon_batch["embeddings"], recon_batch["labels"])
            loss = dml_loss(emb_masked, labels_masked)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            num_batches += 1
    mean_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return mean_loss

def train(config, model, train_loader, val_loader, optimizer, loss_func, reconstructor, results_dir):
    EPOCHS = config['num_epochs']
    PATIENCE = config['patience']
    print("Starting training...")
    #val_metrics = {"f1_score": [], "accuracy_score": [], "recall_score": []}
    train_loss, val_loss = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    model_save_name = os.path.join(results_dir, "best_model.pth")
    results_save_name = os.path.join(results_dir, "results.csv")

    for epoch in range(1, EPOCHS + 1):
        # Train epoch
        epoch_avg_loss = forward_one_epoch(model, train_loader, optimizer, loss_func, reconstructor)
        train_loss.append(epoch_avg_loss)
        # Validation epoch
        val_avg_loss = forward_one_epoch(model, val_loader, optimizer, loss_func, reconstructor, train=False)
        val_loss.append(val_avg_loss)
        # Calculate metrics
        #metrics = test(train_loader, val_loader, model, accuracy_calculator)
        #val_metrics["f1_score"].append(metrics["f1_score"])
        #val_metrics["accuracy_score"].append(metrics["accuracy_score"])
        #val_metrics["recall_score"].append(metrics["recall_score"])
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}".format(epoch, train_loss[-1], val_loss[-1]))
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_name)
            print(f"Saved best model with val_loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}, no improvement in {PATIENCE} epochs.")
                break
    
    results_df = pd.DataFrame({
        "epoch": list(range(1, len(train_loss) + 1)),
        "train_loss": train_loss,
        "val_loss": val_loss
        #"val_f1_score": val_metrics["f1_score"],
        #"val_accuracy_score": val_metrics["accuracy_score"],
        #"val_recall_score": val_metrics["recall_score"]
    })
    results_df.to_csv(results_save_name, index=False)

def get_all_embeddings(loader: torch.utils.data.DataLoader, model):
    embeddings_lst = []
    labels_lst = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, instance_id) in enumerate(loader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)
            embeddings = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            emb_masked, labels_masked = reshape_embeddings_and_labels(embeddings, labels)
            embeddings_lst.append(emb_masked)
            labels_lst.append(labels_masked)
    stacked_embeddings = torch.cat(embeddings_lst)
    stacked_labels = torch.cat(labels_lst)
    return stacked_embeddings, stacked_labels

### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(train_loader, test_loader, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_loader, model)
    test_embeddings, test_labels = get_all_embeddings(test_loader, model)
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, train_embeddings, train_labels, False
    )
    return accuracies

def run_experiments(config):
    for dataset in config['datasets']:
        for loss_name in config['losses']:
            experiment_name = f"{dataset}_{loss_name}"
            print(f"Running experiment: {experiment_name}")
            results_dir = os.path.join("./dml/results", experiment_name)
            os.makedirs(results_dir, exist_ok=True)
            # Model
            model = AutoModel.from_pretrained(config['model_checkpoint'])
            model = model.to(device)
            # Tokenizer
            hf_tokenizer = AutoTokenizer.from_pretrained(config['model_checkpoint'])
            # Load and preprocess the dataset
            collate_fn = SplitInstanceCollate(hf_tokenizer, max_length=config["max_seq_length"], overlap=config["overlap"])
            ds_info_dict, train_ds_name, valid_ds_name, n_labels = unpack_dataset_info(dataset)
            tokenized_aligned_dataset, labels_list, id2label, label2id = load_and_preprocess_dataset(ds_info_dict, hf_tokenizer, config)
            train_loader, val_loader, test_loader = get_dataloaders(tokenized_aligned_dataset, config, ["train", "val", "test"], 
                                        collate_fn=collate_fn, select=config.get("select", -1))
            reconstructor = NerSlidingWindowReconstructor(hf_tokenizer, config["overlap"])
            # optimizer
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

            # Pytorch Metric Learning setup
            #mining_func = MinerWrapper(miner_name="TripletMarginMiner", )miner_name="TripletMarginMiner",
            loss_func = DML_LossesWrapper(loss_name=loss_name, max_samples=1500)
            K = 3
            accuracy_calculator = NERMetricsCalculator(include=("accuracy_score", "f1_score", "recall_score"), k=K)
            # TRAIN
            train(config, model, train_loader, val_loader, optimizer, loss_func, reconstructor, results_dir)
            # Load best model
            print(f"Loading best model from {results_dir}.pth")
            model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth"), map_location=device))
            # TEST
            test_metrics = test(train_loader, test_loader, model, accuracy_calculator)
            print(f"Final test precision: {test_metrics}")
            # Write config and test_metrics to a .txt file
            summary_path = os.path.join(results_dir, f"summary_K{K}.txt")
            with open(summary_path, "w") as f:
                f.write("CONFIGURATION:\n")
                f.write(json.dumps(config, indent=2))
                f.write("\n\nTEST METRICS:\n")
                for k, v in test_metrics.items():
                    f.write(f"{k}: {v}\n")

def run_configs(configs_paths_lst: list):
    for config_path in configs_paths_lst:
            with open(config_path, 'r') as file:
                config = json.load(file)
            run_experiments(config)

if __name__ == "__main__":
    # Run specific config file if user specifies it
    parser = argparse.ArgumentParser(description="Run Deep Metric Learning fine-tuning experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Specify a config file in the dml/configs directory (e.g., deberta_v3_base_DML_config.json). If not set, runs all configs."
    )
    parser.add_argument(
        "--test",
        action='store_true',
        default=True,
        help="After fine-tuning, loads best model and generate test results."
    )
    args = parser.parse_args()
    config_files_lst = []
    # If config file specified, mount path and send it to run_experiments
    if args.config:
        print(f"Running config file: {args.config}")
        config_files_lst = [os.path.join("dml/configs/", args.config)]
    # If not specified, mount all config files' paths and send em all to run_experiments
    else:
        print("Running all config files in configs/ directory.")
        for f in os.listdir("dml/configs/"):
            if f.endswith(".json"):
                config_files_lst.append(os.path.join("dml/configs/", f))
    
    print(f"configs to run: {config_files_lst}")
    run_configs(config_files_lst)

"""
if __name__ == "__main__":
    ### Dataset
    raw_ds_hf = load_dataset("peluz/lener_br", trust_remote_code=True)
    tokenized_datasets = raw_ds_hf.map(tokenize_and_align_labels, batched=True,
                                    batch_size = BATCH_SIZE)
    torch_ds = tokenized_datasets.with_format("torch")
    train_dataset = torch_ds['train']#.select(range(160))
    eval_dataset = torch_ds['validation']#.select(range(160))
    test_dataset = torch_ds['test']#.select(range(160))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                            collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCH_SIZE,
                                            collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                            collate_fn=collate_fn)

    model = hf_model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    ### pytorch-metric-learning stuff ###
    accuracy_calculator = AccuracyCalculator(avg_of_avgs=True, include=("precision_at_1", "AMI",
    "NMI", "mean_average_precision", "mean_reciprocal_rank"), k=1)
    
    model_to_load = "best_deberta_dml_model.pth"
    model.load_state_dict(torch.load(model_to_load, map_location=device))
    print(f"Loaded model: {model_to_load}")
    test_prec = test(train_dataset, test_dataset, model, accuracy_calculator)
    print(f"Final test precision: {test_prec}")

    import umap
    import matplotlib.pyplot as plt

    # Get embeddings and labels (for example, from the train set)
    embeddings, labels = get_all_embeddings(test_dataset, model)
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Fit UMAP
    umap_2d = umap.UMAP(n_components=2, random_state=42)
    embeddings_umap = umap_2d.fit_transform(embeddings_np)

    # Plot
    print("Plotting UMAP...")
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], c=labels_np, cmap='tab20', s=5, alpha=0.7)
    plt.colorbar(scatter, label='Label')
    plt.title("UMAP projection of test token embeddings")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig("umap_test_embeddings.png", dpi=300)
    plt.close()
    print("Saved UMAP plot to umap_embeddings.png")
    

"""
