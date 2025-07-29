import torch
from pytorch_metric_learning import losses, miners

def get_loss_function(loss_name, **kwargs):
    loss_functions = {
        "AngularLoss": losses.AngularLoss,
        "ArcFaceLoss": losses.ArcFaceLoss,
        "ContrastiveLoss": losses.ContrastiveLoss,
        "HistogramLoss": losses.HistogramLoss,
        "MarginLoss": losses.MarginLoss,
        "MultiSimilarityLoss": losses.MultiSimilarityLoss,
        "NPairsLoss": losses.NPairsLoss,
        "ProxyNCALoss": losses.ProxyNCALoss,
        "SignalToNoiseRatioContrastiveLoss": losses.SignalToNoiseRatioContrastiveLoss,
        "SoftTripleLoss": losses.SoftTripleLoss,
        "TripletMarginLoss": losses.TripletMarginLoss
    }

    if loss_name not in loss_functions:
        raise ValueError(f"Loss function '{loss_name}' is not recognized. Available options are: {list(loss_functions.keys())}")

    return loss_functions.get(loss_name)(**kwargs)

def get_miner(miner_name, **kwargs):
    available_miners = {
            "TripletMarginMiner": miners.TripletMarginMiner
        }
    if miner_name not in available_miners:
        raise ValueError(f"Miner '{miner_name}' is not recognized. Available options are: {list(available_miners.keys())}")
    return available_miners.get(miner_name)(margin=0.2, type_of_triplets="semihard")#(**kwargs)

def prioritized_subsample(embeddings: torch.Tensor, labels: torch.Tensor, max_samples: int = 1000, outside_class: int = 0):
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
    zero_quota = max_samples - non_zero_count
    
    if zero_quota > 0:
        # Subsample zero class
        zero_indices = torch.where(zero_class_mask)[0]
        perm = torch.randperm(len(zero_indices))
        selected_zero_indices = zero_indices[perm[:zero_quota]]
        selected_indices = torch.cat([non_zero_indices, selected_zero_indices])
    else:
        # Edge case: Not enough space for all non-zero samples
        selected_indices = non_zero_indices[:max_samples]  # Truncate
    
    # Shuffle final selection (optional)
    #selected_indices = selected_indices[torch.randperm(len(selected_indices))]
    
    return embeddings[selected_indices], labels[selected_indices]

class DML_LossesWrapper:
    """
    Wrapper class for Deep Metric Learning (DML) losses and miners.
    """
    def __init__(self, loss_name="TripletMarginLoss", miner_name=None, max_samples=500, **kwargs):
        self.loss_func = get_loss_function(loss_name, **kwargs)
        if miner_name is not None:
            self.miner = get_miner(miner_name, **kwargs)
            print(f"Using miner: {miner_name}")
        else:
            self.miner = None
        self.max_samples = max_samples

    def __call__(self, embeddings, labels):
        # If number of samples is less than or equal to max_samples, no need to subsample
        if len(labels) <= self.max_samples:
            # If miner is specified
            if self.miner is not None:
                tuples = self.miner(embeddings, labels)
                return self.loss_func(embeddings, labels, tuples)
            # If no miner, just compute loss
            else:
                return self.loss_func(embeddings, labels)
        # If number of samples exceeds max_samples, subsample            
        else:
            sub_emb, sub_labels = prioritized_subsample(embeddings, labels, max_samples=self.max_samples)
            #unique_classes_OG, class_counts_OG = torch.unique(labels, return_counts=True)
            #unique_classes, class_counts = torch.unique(sub_labels, return_counts=True)
            if self.miner is not None:
                tuples = self.miner(sub_emb, sub_labels)
                return self.loss_func(sub_emb, sub_labels, tuples)
            else:
                return self.loss_func(sub_emb, sub_labels)