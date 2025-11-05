import os
import json
import torch
import argparse
import torch.optim as optim
import torch.nn as nn
from data.datasets_utils import load_and_preprocess_dataset, unpack_dataset_info, get_dataloaders, SplitInstanceCollate
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from ttt.ttt_models import build_model
from ttt.ttt_engine import TTTEngine

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

def init_experiment(model_cfg, data_cfg):
    net, ssh = build_model(model_cfg, data_cfg['n_labels'])
    ds_info_dict, train_ds_name, valid_ds_name, n_labels = unpack_dataset_info('lener')
    # Tokenizer 
    hf_tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_checkpoint"])
    # Data Collator
    collate_fn = SplitInstanceCollate(hf_tokenizer, max_length=model_cfg["max_seq_length"], overlap=model_cfg["overlap"])
    # Dataset
    tokenized_aligned_dataset, labels_list, id2label, label2id = load_and_preprocess_dataset(ds_info_dict, hf_tokenizer, model_cfg)
    # Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(tokenized_aligned_dataset, model_cfg,
                                            ["train", "val", "test"], collate_fn=collate_fn, select=model_cfg.get("select", -1))
    parameters = list(net.parameters())+list(ssh.head.parameters())
    # Optimizer
    optimizer = optim.SGD(parameters, lr=model_cfg['learning_rate'], momentum=0.9, weight_decay=5e-4) 
    # Scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*model_cfg['num_epochs']), 
                                                num_training_steps=model_cfg['num_epochs'])
    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()
    # Pathing for saving results
    experiment_name = model_cfg["experiment_name"]
    results_dir = f"./ttt/results/{experiment_name}"
    os.makedirs(results_dir, exist_ok=True)
    # Return the engine
    return TTTEngine(net, ssh, optimizer, scheduler, criterion, hf_tokenizer, train_loader, 
                     val_loader, test_loader, model_cfg, id2label=id2label, device=device,
                     results_dir=results_dir, labels_list=labels_list)

def run_experiment(args, model_cfg, data_cfg):
    engine = init_experiment(model_cfg, data_cfg)
    if args.train:
        engine.train()
    if args.load_checkpoint:
         engine.load_checkpoint()
    if args.test_std:
        engine.test_std()
    if args.test_ttt_std:
        engine.test_ttt_std()
    if args.test_ttt_online:
        engine.test_ttt_online()

if __name__ == "__main__":
    # Run specific config file if user specifies it
    parser = argparse.ArgumentParser(description="Run NER Test Time Training experiments.")
    parser.add_argument("--model_config", type=str, default=None,
                help="Specify a config file in the ttt/configs/ directory (e.g., bert_base_uncased_config.json)."
    )
    parser.add_argument('--data_set', default='lener', choices=['lener', 'cdjurbr', 'ulysses', 'conll2003'],
                type=str, help='dataset to use')
    parser.add_argument("--train", action="store_true", help="Whether to run training.")
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--test_std", action="store_true", help="Whether to run NER testing.")
    parser.add_argument("--test_ttt_std", action="store_true", help="Whether to run NER testing using standard TTT protocol.")
    parser.add_argument("--test_ttt_online", action="store_true", help="Whether to run NER testing using online TTT protocol.")
    args = parser.parse_args()
    ttt_configs_dir = "./ttt/configs/"
    data_config = None
    with open(os.path.join(f"./data/{args.data_set}",f"{args.data_set}.json"), 'r') as f:
        data_cfg = json.load(f)
    if args.model_config:
        config_file_path = os.path.join(ttt_configs_dir, args.model_config)
        with open(config_file_path, 'r') as file:
                model_cfg = json.load(file)
        run_experiment(args, model_cfg, data_cfg)
    # If not specified, mount all config files' paths and send em all to run_experiments
    else:
        import sys
        print("No config file specified, quitting.")
        sys.exit(0)
    