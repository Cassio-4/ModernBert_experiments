import gc
import os
import json
import torch
import evaluate
import numpy as np
import pandas as pd
from models.model_utils import load_model
from datasets_utils import load_and_preprocess_dataset, unpack_dataset_info
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

#seqeval = evaluate.load("seqeval")

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

def finetune_curr_dataset(config, dataset_name: str = None,
                          experiment_name: str = None, do_cleanup: bool = True):
    # 1. Get tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
    # 2. Get the dataset
    # 2.1 Get the dataset info
    ds_info_dict, train_ds_name, valid_ds_name, n_labels = unpack_dataset_info(dataset_name)
    # 2.2 Load and preprocess the dataset
    tokenized_aligned_dataset, labels_list, id2label, label2id = load_and_preprocess_dataset(ds_info_dict, hf_tokenizer)
    # 3. Define the compute metrics function
    #task_compute_metrics = partial(compute_metrics, task_metrics=task_metrics)
    # 4. Load the model and data collator
    hf_model = load_model(config, n_labels, id2label, label2id)
    hf_data_collator = DataCollatorForTokenClassification(tokenizer=hf_tokenizer)

    # 6. Define the training arguments and trainer
    training_args = TrainingArguments(
        output_dir=experiment_name,
        learning_rate=config['learning_rate'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        num_train_epochs=config['num_epochs'],
        lr_scheduler_type="linear",
        optim="adamw_torch",
        adam_beta1=config['adam_betas'][0],
        adam_beta2=config['adam_betas'][1],
        adam_epsilon=config['adam_epsilon'],
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
        train_dataset=tokenized_aligned_dataset[train_ds_name],
        eval_dataset=tokenized_aligned_dataset[valid_ds_name],
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
        cleanup(things_to_delete=[trainer, hf_model, hf_tokenizer, tokenized_aligned_dataset])

    return train_res_df, args_df, hf_model, hf_tokenizer

def get_experiment_name(config, ds_name):
    # Define the experiment name based on the dataset and model
    return f"{config['model_checkpoint']}_{ds_name}"

def do_train(config):
    for dataset in config["datasets"]:
        experiment_name = get_experiment_name(config, dataset)
        
        # Call the finetuning function
        print(f"Finetuning {experiment_name}...")
        train_res_df, args_df, hf_model, hf_tokenizer = finetune_curr_dataset(config, dataset, experiment_name)
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

def run_experiments(config_dir):
    for f in os.listdir("configs/"):
        if f.endswith(".json"):
            with open(os.path.join("configs/", f), 'r') as file:
                config = json.load(file)
            do_train(config)

if __name__ == "__main__":
    run_experiments("configs/")
    