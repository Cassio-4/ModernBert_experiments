import os
import numpy as np
import pandas as pd
import torch
import evaluate
import gc
from datasets import load_dataset
from ModernBertWithTanH import convert_ln_to_dyt
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

seqeval = evaluate.load("seqeval")
train_bsz, val_bsz = 32, 32
lr = 8e-5
betas = (0.9, 0.98)
n_epochs = 20
eps = 1e-6
wd = 8e-6
DyT = True

"""
    "lener": {"download_reference": "peluz/lener_br",
    "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
    "n_labels": 13
    },
"""

datasets_dict = {
    "bc5cdr": {
        "download_reference": "tner/bc5cdr",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "n_labels": 5,
        "label2id": {"O": 0, "B-Chemical": 1, "B-Disease": 2, "I-Disease": 3, "I-Chemical": 4},
        "id2label": {0: "O", 1: "B-Chemical", 2: "B-Disease", 3: "I-Disease", 4: "I-Chemical"},
        "labels": ["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"]
    },
    "crossner": {"download_reference": "DFKI-SLT/cross_ner",
        "tasks": ['conll2003', 'politics', 'science', 'music','literature', 'ai'],
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "n_labels": 79
    },
    "ncbi": {
        "download_reference": "ncbi/ncbi_disease",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "n_labels": 3
    }
}

class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.training_history = {"train": [], "eval": []}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:  # Training logs
                self.training_history["train"].append(logs)
            elif "eval_loss" in logs:  # Evaluation logs
                self.training_history["eval"].append(logs)

def compute_metrics_with_labels(label_list):
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    return compute_metrics

def cleanup(things_to_delete: list | None = None):
    if things_to_delete is not None:
        for thing in things_to_delete:
            if thing is not None:
                del thing

    gc.collect()
    torch.cuda.empty_cache()

def get_label_maps(raw_datasets, train_ds_name):
    labels = raw_datasets[train_ds_name].features["ner_tags"].feature

    id2label = {idx: name for idx, name in enumerate(labels.names)} if hasattr(labels, "names") else None
    label2id = {name: idx for idx, name in enumerate(labels.names)} if hasattr(labels, "names") else None

    return id2label, label2id

#https://huggingface.co/docs/transformers/main/en/tasks/token_classification
def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    #print(f'TOKENIZED_INPUTS: {tokenized_inputs}')
    tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][0])
    #print(f'TOKENS: {tokens}')
    labels = []
    if "ner_tags" in examples:
        ner_tags = "ner_tags"
    else:
        ner_tags = "tags"
    for i, label in enumerate(examples[f"{ner_tags}"]):
        #print(f'I: {i}; LABEL: {label}')
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        #print(f'WORD_IDS: {word_ids}')
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    #print(f'TOKENIZED_INPUTS["labels"]: {tokenized_inputs["labels"]}')
    return tokenized_inputs

def finetune_ner_datasets(experiment_name: str = None, dataset: str = None, task: str = None, 
                          checkpoint: str = "answerdotai/ModernBERT-base", do_cleanup: bool = True):
    # 1. Load the task metadata
    dataset_meta = datasets_dict[dataset]
    train_ds_name = dataset_meta["dataset_names"]["train"]
    valid_ds_name = dataset_meta["dataset_names"]["valid"]
    n_labels = dataset_meta["n_labels"]

    # 2. Load the dataset
    if dataset == "crossner" and task is not None:
        raw_datasets = load_dataset(dataset_meta['download_reference'], task, trust_remote_code=True) 
    else:
        raw_datasets = load_dataset(dataset_meta['download_reference'], trust_remote_code=True)
    if dataset == "bc5cdr":
        labels_list = datasets_dict[dataset]['labels']
        id2label, label2id =  datasets_dict[dataset]['id2label'], datasets_dict[dataset]['label2id']
    else: 
        labels_list = raw_datasets["train"].features["ner_tags"].feature.names
        id2label, label2id = get_label_maps(raw_datasets, train_ds_name)
    # 3. Load the tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenized_aligned_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": hf_tokenizer})

    # 4. Define the compute metrics function
    #task_compute_metrics = partial(compute_metrics, task_metrics=task_metrics)

    # 5. Load the model and data collator
    model_additional_kwargs = {"id2label": id2label, "label2id": label2id} if id2label and label2id else {}
    hf_model = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=n_labels, **model_additional_kwargs)
    if DyT:
        hf_model = convert_ln_to_dyt(hf_model)
    # compile=False,

    hf_data_collator = DataCollatorForTokenClassification(tokenizer=hf_tokenizer)

    # 6. Define the training arguments and trainer
    training_args = TrainingArguments(
        output_dir=f"ModernBERT_{experiment_name}",
        learning_rate=lr,
        per_device_train_batch_size=train_bsz,
        per_device_eval_batch_size=val_bsz,
        num_train_epochs=n_epochs,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        adam_beta1=betas[0],
        adam_beta2=betas[1],
        adam_epsilon=eps,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="best",
        load_best_model_at_end=True,
        bf16=True,
        bf16_full_eval=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=hf_model,
        args=training_args,
        train_dataset=tokenized_aligned_datasets[train_ds_name],
        eval_dataset=tokenized_aligned_datasets[valid_ds_name],
        processing_class=hf_tokenizer,
        data_collator=hf_data_collator,
        compute_metrics=compute_metrics_with_labels(labels_list)
    )

    # Add callback to trainer
    metrics_callback = MetricsCallback()
    trainer.add_callback(metrics_callback)

    trainer.train()

    # 7. Get the training results and hyperparameters
    train_history_df = pd.DataFrame(metrics_callback.training_history["train"])
    train_history_df = train_history_df.add_prefix("train_")
    eval_history_df = pd.DataFrame(metrics_callback.training_history["eval"])
    train_res_df = pd.concat([train_history_df, eval_history_df], axis=1)

    args_df = pd.DataFrame([training_args.to_dict()])

    # 8. Cleanup (optional)
    if do_cleanup:
        cleanup(things_to_delete=[trainer, hf_model, hf_tokenizer, tokenized_aligned_datasets, raw_datasets])

    return train_res_df, args_df, hf_model, hf_tokenizer

def training_loop(dataset=None, task=None):
    # Define the experiment name based on the dataset and task
    if task is None:
        experiment_name = dataset
    else:
        experiment_name = f"{dataset}_{task}"
    if DyT:
        experiment_name += "_DyT"
    
    # Call the finetuning function
    print(f"Finetuning {experiment_name}...")
    train_res_df, args_df, hf_model, hf_tokenizer = finetune_ner_datasets(experiment_name, dataset, task)
    print(f"Training results for {experiment_name}:")
    print(train_res_df.head())
    print(f"Training arguments for {experiment_name}:")
    print(args_df.head())
    # Save the model and tokenizer
    model_save_path = f"models/{experiment_name}_model"
    tokenizer_save_path = f"models/{experiment_name}_tokenizer"
    train_res_df_save_path = f"models/{experiment_name}_training_results.csv"
    args_dfs_save_path = f"models/{experiment_name}_training_args.csv"
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(tokenizer_save_path, exist_ok=True)
    hf_model.save_pretrained(model_save_path)
    hf_tokenizer.save_pretrained(tokenizer_save_path)
    train_res_df.to_csv(train_res_df_save_path, index=False)
    args_df.to_csv(args_dfs_save_path, index=False)

if __name__ == "__main__":
    for dataset in datasets_dict.keys():
        if dataset == "crossner":
            for task in datasets_dict[dataset]["tasks"]:
                training_loop(dataset=dataset, task=task)
        else:
            training_loop(dataset=dataset, task=None)