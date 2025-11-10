from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import matplotlib as plt

def compute_entity_level_scores(self, preds, labels):
        p = precision_score(labels, preds)
        r = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        report = classification_report(labels, preds, digits=4)
        return {"precision": p, "recall": r, "f1": f1, "report": report},

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