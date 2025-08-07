"""
### Setup (pip, imports)
!pip install -U datasets
!pip install pytorch-metric-learning
!apt install libomp-dev
!pip install faiss-cpu
"""
import argparse, os, json
import torch
import torch.optim as optim
from dml.dml_engine import DMLEngine
from transformers import AutoTokenizer, AutoModel
from dml.losses_factory import DML_LossesWrapper
from datasets_utils import load_and_preprocess_dataset, unpack_dataset_info, get_dataloaders, SplitInstanceCollate, NerSlidingWindowReconstructor

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

def init_args():
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
    parser.add_argument(
        "--train",
        action='store_true',
        default=True,
        help="If set, trains the model on the datasets specified in the config file."
    )
    parser.add_argument(
        "--plot_umap",
        action='store_true',
        default=True,
        help="If set, loads model checkpoint and plots UMAP with its embeddings."
    )
    
    args = parser.parse_args()
    return args

def init_experiment(config, dataset, loss_name):
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
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    # Pytorch Metric Learning setup
    MINER_NAME = "TripletMarginMiner"
    loss_func = DML_LossesWrapper(loss_name=loss_name, miner_name=MINER_NAME, max_samples=config['max_samples'])
    # Logging
    experiment_name = f"{dataset}_{loss_name}_miner-{MINER_NAME}_max_samples-{config['max_samples']}"
    print(f"Running experiment: {experiment_name}")
    results_dir = os.path.join("./dml/results/WithMiner", experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    # Create engine
    dml_engine = DMLEngine(config, model, hf_tokenizer, loss_func, optimizer, train_loader, 
            val_loader, test_loader, id2label, reconstructor, device, results_dir)
    return dml_engine

def run_experiments(config, args):
    for dataset_name in config['datasets']:
        for loss_name in config['losses']:
            dml_engine = init_experiment(config, dataset_name, loss_name)
            # TRAIN
            if args.train:
                dml_engine.train()
            if args.test:
                dml_engine.load_checkpoint()
                dml_engine.test()
            if args.plot_umap:
                dml_engine.plot_umap()

def run_configs(configs_paths_lst: list, args):
    for config_path in configs_paths_lst:
            with open(config_path, 'r') as file:
                config = json.load(file)
            run_experiments(config, args)

if __name__ == "__main__":
    args = init_args()
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
    run_configs(config_files_lst, args)
