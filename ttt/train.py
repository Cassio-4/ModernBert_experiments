import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)
import argparse
from src.core.config import get_config
from src.core.criterions_optims import get_criterion, get_optimizer
from src.data.loader import get_dataloader
from models.ttt_models import build_model
from transformers import AutoTokenizer, get_linear_schedule_with_warmup  
from .engine import TTTEngine

# Run specific config file if user specifies it
parser = argparse.ArgumentParser(description="Run NER Test Time Training model training.")
parser.add_argument("--config", type=str, default=None,
            help="Specify a config file in the ../configs/ttt directory (file name only)")
args = parser.parse_args()
config = get_config(args.config, approach='ttt')

# Tokenizer 
hf_tokenizer = AutoTokenizer.from_pretrained(config['model']['hf_checkpoint'])
# Dataloaders
train_loader = get_dataloader(config, split='train', tokenizer=hf_tokenizer)
val_loader = get_dataloader(config, split='val', tokenizer=hf_tokenizer)

# Model
net, ssh = build_model(config)
parameters = list(net.parameters())+list(ssh.head.parameters())
# Optimizer
optimizer = get_optimizer(config, parameters)
# Scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*model_cfg['num_epochs']), 
                                            num_training_steps=model_cfg['num_epochs'])
# Criterion
criterion = get_criterion(config, parameters)
# Pathing for saving results
experiment_name = model_cfg["experiment_name"]
results_dir = f"./ttt/results/{experiment_name}"
os.makedirs(results_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
# Return the engine
TTTEngine(config, model, hf_tokenizer, optimizer, criterion, scheduler, train_loader, val_loader, device=device)

