import os
import json
import torch
import argparse
from models.model_utils import load_model
from data.datasets_utils import load_and_preprocess_dataset, unpack_dataset_info
from engine import NEREngine
from transformers import AutoTokenizer, DataCollatorForTokenClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

def init_experiment(config, dataset_name: str = None):
    # 1. Model
    hf_model = load_model(config, n_labels, id2label, label2id)
    hf_model = hf_model.to(device)
    # 2. Get tokenizer
    if config["model_checkpoint"] == "FacebookAI/roberta-base":
        hf_tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"], add_prefix_space=True)
    else:
        hf_tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
    # 2.1 Data Collator
    hf_data_collator = DataCollatorForTokenClassification(tokenizer=hf_tokenizer)
    # 3. Get the dataset
    # 3.1 Get the dataset info
    ds_info_dict, train_ds_name, valid_ds_name, n_labels = unpack_dataset_info(dataset_name)
    # 3.2 Load and preprocess the dataset
    tokenized_aligned_dataset, labels_list, id2label, label2id = load_and_preprocess_dataset(ds_info_dict, hf_tokenizer, config)
    # 3.3 Get the dataloaders
    #train_loader, val_loader, test_loader = get_dataloaders(tokenized_aligned_dataset, config, ["train", "val", "test"], 
    #                            collate_fn=None, select=config.get("select", -1))
    model_name = config["model_checkpoint"].split("/")[-1]
    experiment_name = f"{model_name}_{dataset_name}"
    results_dir = f"output/{experiment_name}"
    os.makedirs(results_dir, exist_ok=True)
    return NEREngine(config, hf_model, hf_tokenizer, hf_data_collator, train_ds_name, valid_ds_name,
                     tokenized_aligned_dataset, id2label=id2label, reconstructor=None, device=device, 
                     results_dir=results_dir, labels_list=labels_list)

def run_experiment(args, config):
    for dataset in config["datasets"]:
        engine = init_experiment(config, dataset)
        if args.train:
            engine.train()

def run_configs(configs_paths_lst: list):
    for config_path in configs_paths_lst:
            with open(config_path, 'r') as file:
                config = json.load(file)
            run_experiment(config)

if __name__ == "__main__":
    # Run specific config file if user specifies it
    parser = argparse.ArgumentParser(description="Run NER fine-tuning experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Specify a config file in the configs/ directory (e.g., bert_base_uncased_config.json). If not set, runs all configs."
    )
    args = parser.parse_args()
    config_files_lst = []
    # If config file specified, mount path and send it to run_experiments
    if args.config:
        print(f"Running config file: {args.config}")
        config_files_lst = [os.path.join("configs/", args.config)]
    # If not specified, mount all config files' paths and send em all to run_experiments
    else:
        print("Running all config files in configs/ directory.")
        for f in os.listdir("configs/"):
            if f.endswith(".json"):
                config_files_lst.append(os.path.join("configs/", f))
    
    print(f"configs to run: {config_files_lst}")
    run_configs(config_files_lst)
    