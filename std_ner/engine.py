import gc
import torch
import pandas as pd
import evaluate
import numpy as np
import umap
import matplotlib.pyplot as plt
import os
from transformers import TrainingArguments, Trainer, TrainerCallback

seqeval = evaluate.load("seqeval")

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


class NEREngine:
    def __init__(self, config, model, tokenizer, data_collator, train_ds_name, valid_ds_name,
                  tokenized_aligned_dataset, id2label=None, reconstructor=None, device=None, 
                  results_dir=None, labels_list=None):
        self.config = config
        self.device = device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.train_ds_name = train_ds_name
        self.valid_ds_name = valid_ds_name
        self.labels_list = labels_list
        self.tokenized_aligned_dataset = tokenized_aligned_dataset
        self.id2label = id2label
        self.reconstructor = reconstructor
        self.results_dir = results_dir
        self.loaded_local_checkpoint = False
        self.epochs_run = 0

    def train(self):
        training_args = TrainingArguments(
            output_dir=self.results_dir,
            learning_rate=self.config['learning_rate'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            num_train_epochs=self.config['num_epochs'],
            lr_scheduler_type="linear",
            optim="adamw_torch",
            adam_beta1=self.config['adam_betas'][0],
            adam_beta2=self.config['adam_betas'][1],
            adam_epsilon=self.config['adam_epsilon'],
            logging_strategy="epoch",
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            bf16=True,
            bf16_full_eval=False,
            push_to_hub=False
            )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_aligned_dataset[self.train_ds_name],
            eval_dataset=self.tokenized_aligned_dataset[self.valid_ds_name],
            processing_class=self.hf_tokenizer,
            data_collator=self.hf_data_collator,
            compute_metrics=compute_metrics_with_labels(self.labels_list)
            )

        # Add callback to trainer
        metrics_callback = MetricsCallback()
        trainer.add_callback(metrics_callback)
        print("initializing training...")
        trainer.train()

        # 7. Get the training results and hyperparameters
        train_history_df = pd.DataFrame(metrics_callback.training_history["train"])
        train_history_df = train_history_df.add_prefix("train_")
        eval_history_df = pd.DataFrame(metrics_callback.training_history["eval"])
        train_res_df = pd.concat([train_history_df, eval_history_df], axis=1)
        args_df = pd.DataFrame([training_args.to_dict()])

        train_res_df_save_path = f"{self.results_dir}_loss.csv"
        args_dfs_save_path = f"{self.results_dir}_args.csv"
        train_res_df.to_csv(train_res_df_save_path, index=False)
        args_df.to_csv(args_dfs_save_path, index=False)
        self.model.save_pretrained(f"{self.results_dir}/best_model.pth")

    def test(self):
        if "test" in self.tokenized_aligned_dataset:
            print("Evaluating on test set...")
            test_results = self.trainer.evaluate(eval_dataset=self.tokenized_aligned_dataset["test"])
            test_results_df = pd.DataFrame([test_results])
            if test_results_df is not None:
                test_results_path = f"{self.results_dir}_test_results.csv"
                test_results_df.to_csv(test_results_path, index=False)
    
    def load_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.results_dir, "best_model.pth")

        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.loaded_local_checkpoint = True
        print(f"Loaded best model from {checkpoint_path}")
    
    def freeze_backbone(self):
        print("Freezing backbone model parameters...")
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.encoder.layer[-1].parameters():
            param.requires_grad = True
    
    def plot_umap(self, dataloader_name="test", plot_outside=True):
        embeddings, labels = self.get_all_embeddings(dataloader_name=dataloader_name)
        embeddings_np = embeddings.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # Fit UMAP
        umap_2d = umap.UMAP(n_components=2, random_state=42)
        embeddings_umap = umap_2d.fit_transform(embeddings_np)

        if not plot_outside:
            mask = labels_np != 0
            embeddings_umap = embeddings_umap[mask]
            labels_np = labels_np[mask]

        # Plot
        print("Plotting UMAP...")
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], c=labels_np, cmap='tab20', s=5, alpha=0.7)
        #plt.colorbar(scatter, label='Label')
        import matplotlib.patches as mpatches
        unique_labels = sorted(set(labels_np))
        handles = [
            mpatches.Patch(
                color=plt.cm.tab20(i / max(unique_labels)),
                label=self.id2label.get(int(i), str(i))
            )
            for i in unique_labels
        ]
        plt.legend(handles=handles, title="Label Names", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title("UMAP projection of test token embeddings")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.tight_layout()
        plt.savefig("umap_test_embeddings_no_Outside.png", dpi=300)
        plt.close()
        print("Saved UMAP plot to umap_embeddings.png")
    
