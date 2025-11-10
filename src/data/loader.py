from torch.utils.data import DataLoader
from .dataset_builder import load_dataset
from .collators import get_collator

def get_dataloader(cfg, split, tokenizer):
    """
    Get dataloader for a specified split.
    """
    if split == 'train':
        train_dataset = load_dataset(cfg, split='train', tokenizer=tokenizer)
        collate_fn = get_collator(cfg, tokenizer)
        dataloader =  DataLoader(train_dataset, batch_size=cfg['data']['batch_size'], collate_fn=collate_fn)
    elif split == 'val' or split == 'eval':
        eval_dataset = load_dataset(cfg, split='val', tokenizer=tokenizer)
        collate_fn = get_collator(cfg, tokenizer)
        dataloader = DataLoader(eval_dataset, batch_size=cfg['data']['batch_size'], collate_fn=collate_fn)
    elif split == 'test':
        test_dataset = load_dataset(cfg['dataset_name'], split='test', tokenizer=tokenizer)
        dataloader = DataLoader(test_dataset, batch_size=cfg['data']['batch_size'])
    else:
        raise ValueError(f"Unknown split: {split}")
    return dataloader
