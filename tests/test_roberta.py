import sys, os
current = os.path.dirname(os.path.realpath(__file__))

parent = os.path.dirname(current)

sys.path.append(parent)

import torch
torch.set_float32_matmul_precision('high')
from models.model_utils import load_model
from data.datasets_utils import load_and_preprocess_dataset, unpack_dataset_info
import json
from transformers import AutoTokenizer, RobertaModel
from datasets import load_dataset


# 1.Load config
with open('../configs/roberta_base_config.json', 'r') as file:
    config = json.load(file)
# 2. Load tokenizer
hf_tokenizer= AutoTokenizer.from_pretrained(config["model_checkpoint"], add_prefix_space=True)
# 3. Load dataset info
ds_info_dict, train_ds_name, valid_ds_name, n_labels = unpack_dataset_info( config["datasets"][0])
# 3.1 Preprocess dataset
tokenized_aligned_dataset, labels_list, id2label, label2id = load_and_preprocess_dataset(ds_info_dict, hf_tokenizer, config)
raw_ds = load_dataset(ds_info_dict['download_reference'], trust_remote_code=True)
# 4. Load model
model = load_model(config, n_labels, id2label, label2id)
# 5. Send model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for i in range(3):
    print("Words:", raw_ds[train_ds_name][i])
    print("Tokens:", hf_tokenizer.convert_ids_to_tokens(tokenized_aligned_dataset[train_ds_name][i]['input_ids']))
    print("Labels:", tokenized_aligned_dataset[train_ds_name][i]['labels'])

texts = ["The capital of France is Paris.", "The capital of Germany is Berlin."]

inputs = hf_tokenizer(
    text=texts,
    add_special_tokens=True,
    padding='max_length',
    truncation=True,
    max_length=512,
    return_attention_mask=True,
    return_tensors='pt' 
)

input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)

outputs = model(input_ids=input_ids, attention_mask=attention_mask)

print(outputs)