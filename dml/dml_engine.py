import torch
import pandas as pd
import json
import umap
import matplotlib.pyplot as plt
import os
from .test import NERMetricsCalculator
from .losses_factory import DML_LossesWrapper

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

class DMLEngine:
    def __init__(self, config, model, tokenizer, loss_func:DML_LossesWrapper, optimizer, train_loader, 
                 val_loader, test_loader, id2label=None, reconstructor=None, device=None, results_dir=None):
        self.config = config
        self.device = device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.id2label = id2label
        self.reconstructor = reconstructor
        self.results_dir = results_dir
        self.loaded_local_checkpoint = False
        self.epochs_run = 0
        
    def forward_one_epoch(self, train=True):
        if train:
            self.model.train()
            data_loader = self.train_loader
        else:
            self.model.eval()
            data_loader = self.val_loader
        total_loss = 0.0
        num_batches = 0
        with torch.set_grad_enabled(train):
            for batch_idx, (data, instance_ids) in enumerate(data_loader):
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                if train:
                    self.optimizer.zero_grad()
                    self.loss_func.loss_optimizer_zero_grad()
                embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                recon_batch = self.reconstructor.reconstruct_sequences(embeddings, instance_ids, data)
                emb_masked, labels_masked = reshape_embeddings_and_labels(recon_batch["embeddings"], recon_batch["labels"])
                loss = self.loss_func(emb_masked, labels_masked)
                if train:
                    loss.backward()
                    self.optimizer.step()
                    self.loss_func.loss_optimizer_step()
                total_loss += loss.item()
                num_batches += 1
        mean_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return mean_loss

    def train(self):
        EPOCHS = self.config['num_epochs']
        PATIENCE = self.config['patience']
        print("Starting training...")
        train_loss, val_loss = [], []
        best_val_loss = float('inf')
        patience_counter = 0
        model_sv_path = os.path.join(self.results_dir, "best_model.pth")
        losses_sv_path = os.path.join(self.results_dir, "losses.csv")

        for epoch in range(1, EPOCHS + 1):
            # Train epoch
            epoch_avg_loss = self.forward_one_epoch(train=True)
            self.epochs_run = epoch
            train_loss.append(epoch_avg_loss)
            # Validation epoch
            val_avg_loss = self.forward_one_epoch(train=False)
            val_loss.append(val_avg_loss)
            print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}".format(epoch, train_loss[-1], val_loss[-1]))
            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), model_sv_path)
                print(f"Saved best model with val_loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}, no improvement in {PATIENCE} epochs.")
                    break
        
        losses_df = pd.DataFrame({
            "epoch": list(range(1, len(train_loss) + 1)),
            "train_loss": train_loss,
            "val_loss": val_loss
        })
        losses_df.to_csv(losses_sv_path, index=False)

    def test(self):
        print("Encoding train embeddings...")
        train_embeddings, train_labels = self.get_all_embeddings(dataloader_name="train")
        print("Encoding test embeddings...")
        test_embeddings, test_labels = self.get_all_embeddings(dataloader_name="test")
        metrics_dict = {}
        for k in self.config['knn_K']:
            print(f"Calculating metrics for K={k}...")
            accuracy_calculator = NERMetricsCalculator(include=("accuracy_score", "f1_score", "recall_score"), k=k)
            test_metrics = accuracy_calculator.get_accuracy(
                test_embeddings, test_labels, train_embeddings, train_labels, False
            )
            print(f"Test metrics for K={k}: {test_metrics}")
            metrics_dict[k] = test_metrics
        # Write config and test_metrics to a .txt file
        summary_path = os.path.join(self.results_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("CONFIGURATION:\n")
            f.write(json.dumps(self.config, indent=2))
            for k, v in metrics_dict.items():
                f.write(f"\nTEST METRICS K={k}:\n")
                for metric_name, metric_value in v.items():
                    f.write(f"{metric_name}: {metric_value}\n")
    
    def get_all_embeddings(self, dataloader_name=None):
        if dataloader_name == "train":
            loader = self.train_loader
        elif dataloader_name == "test":
            loader = self.test_loader
        elif dataloader_name == "val":
            loader = self.val_loader
        else:
            raise ValueError("No dataloader specified in get_all_embeddings, can't proceed.")
        embeddings_lst = []
        labels_lst = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, instance_ids) in enumerate(loader):
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                recon_batch = self.reconstructor.reconstruct_sequences(embeddings, instance_ids, data)
                emb_masked, labels_masked = reshape_embeddings_and_labels(recon_batch["embeddings"], recon_batch["labels"])            
                embeddings_lst.append(emb_masked)
                labels_lst.append(labels_masked)
        stacked_embeddings = torch.cat(embeddings_lst)
        stacked_labels = torch.cat(labels_lst)
        return stacked_embeddings, stacked_labels
    
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
        plt.title("UMAP projection of test token embeddings DeBERTa SoftTripleLoss")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.tight_layout()
        plt.savefig("umap_test_embeddings_DeBERTa_noO.png", dpi=300)
        plt.close()
        print("Saved UMAP plot to umap_embeddings.png")
    
