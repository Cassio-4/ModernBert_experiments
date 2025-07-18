from pytorch_metric_learning import losses
import torch

def get_loss_function(loss_name, **kwargs):

    loss_functions = {
        "TripletMarginLoss": losses.TripletMarginLoss,
        "ContrastiveLoss": losses.ContrastiveLoss,
        "MultiSimilarityLoss": losses.MultiSimilarityLoss,
        "SoftTripleLoss": losses.SoftTripleLoss,
        "TripletMarginWithDistanceLoss": losses.TripletMarginWithDistanceLoss,
        "CircleLoss": losses.CircleLoss,
        "AngularLoss": losses.AngularLoss,
        "NormalizedSoftmaxLoss": losses.NormalizedSoftmaxLoss,
        "FocalLoss": losses.FocalLoss,
        "CosineSimilarityLoss": losses.CosineSimilarityLoss,
    }

    if loss_name not in loss_functions:
        raise ValueError(f"Loss function '{loss_name}' is not recognized. Available options are: {list(loss_functions.keys())}")

    return loss_functions[loss_name](**kwargs)
from pytorch_metric_learning import miners
class MinerWrapper:
    def __init__(self, miner_name = "TripletMarginMiner", max_samples=2000, **kwargs):
        self.max_samples = max_samples
        available_miners = {
            "TripletMarginMiner": miners.TripletMarginMiner
        }
        self.miner = available_miners.get(miner_name)(**kwargs)

    def __call__(self, embeddings, labels):
        if len(labels) > self.max_samples:
            sub_embeddings, sub_labels = self.prioritized_subsample(embeddings, labels)
            unique_classes_OG, class_counts_OG = torch.unique(labels, return_counts=True)
            unique_classes, class_counts = torch.unique(sub_labels, return_counts=True)

            tuples = self.miner(sub_embeddings, sub_labels)
        else:
            tuples = self.miner(embeddings, labels)
        return tuples
    
    def prioritized_subsample(self, embeddings: torch.Tensor, labels: torch.Tensor, outside_class: int = 0):
        """
        Subsamples embeddings, prioritizing all non-zero class samples.
        Args:
            embeddings: Tensor of shape [N, emb_dim]
            labels: Tensor of shape [N]
            max_samples: Total samples after subsampling
            outside_class: Class label to deprioritize (default: 0)
        Returns:
            (subsampled_embeddings, subsampled_labels)
        """
        # Count samples per class
        #unique_classes, class_counts = torch.unique(labels, return_counts=True)
        
        # Split into zero and non-zero classes
        #non_zero_classes = unique_classes[unique_classes != outside_class]
        zero_class_mask = (labels == outside_class)
        
        # Collect all non-zero class samples
        non_zero_indices = torch.where(~zero_class_mask)[0]
        
        # Calculate remaining quota for zero class
        non_zero_count = len(non_zero_indices)
        zero_quota = self.max_samples - non_zero_count
        
        if zero_quota > 0:
            # Subsample zero class
            zero_indices = torch.where(zero_class_mask)[0]
            perm = torch.randperm(len(zero_indices))
            selected_zero_indices = zero_indices[perm[:zero_quota]]
            selected_indices = torch.cat([non_zero_indices, selected_zero_indices])
        else:
            # Edge case: Not enough space for all non-zero samples
            selected_indices = non_zero_indices[:self.max_samples]  # Truncate
        
        # Shuffle final selection (optional)
        #selected_indices = selected_indices[torch.randperm(len(selected_indices))]
        
        return embeddings[selected_indices], labels[selected_indices]

    def chunked_mining(self, mining_func, embeddings, labels, chunk_size=512):
        indices = []
        for i in range(0, len(embeddings), chunk_size):
            chunk_emb = embeddings[i:i+chunk_size]
            chunk_labels = labels[i:i+chunk_size]
            indices.append(mining_func(chunk_emb, chunk_labels))
        return indices  # Custom logic to merge tuples