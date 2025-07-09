from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from transformers import DataCollatorForTokenClassification

#, "bc5cdr", "conll2003", "ncbi", "ontonotes"
datasets_dict = {
     "lener": {
        "download_reference": "peluz/lener_br",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "n_labels": 13
    },
    "bc5cdr": {
        "download_reference": "tner/bc5cdr",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "n_labels": 5,
        "label2id": {"O": 0, "B-Chemical": 1, "B-Disease": 2, "I-Disease": 3, "I-Chemical": 4},
        "id2label": {0: "O", 1: "B-Chemical", 2: "B-Disease", 3: "I-Disease", 4: "I-Chemical"},
        "labels": ["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"]
    },
    "conll2003": {
        "download_reference": "tner/conll2003",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "n_labels": 9,
        "label2id": {"O":0, "B-ORG":1, "B-MISC":2, "B-PER":3, "I-PER":4, "B-LOC":5, "I-ORG":6, "I-MISC":7, "I-LOC":8},
        "id2label": {0:"O", 1:"B-ORG",2: "B-MISC", 3:"B-PER", 4:"I-PER", 5:"B-LOC", 6:"I-ORG", 7:"I-MISC", 8:"I-LOC"},
        "labels": ["O", "B-ORG", "B-MISC", "B-PER", "I-PER", "B-LOC", "I-ORG", "I-MISC", "I-LOC"]
    },
    "ncbi": {
        "download_reference": "ncbi/ncbi_disease",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "n_labels": 3
    },
    "ontonotes": {
        "download_reference": "tner/ontonotes5",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "n_labels": 37,
        "label2id": {"O":0, "B-CARDINAL":1, "B-DATE":2, "I-DATE":3, "B-PERSON":4, "I-PERSON":5, "B-NORP":6, "B-GPE":7, "I-GPE":8,
                    "B-LAW":9, "I-LAW": 10, "B-ORG": 11, "I-ORG": 12, "B-PERCENT":13, "I-PERCENT": 14, "B-ORDINAL":15, "B-MONEY":16,
                    "I-MONEY": 17, "B-WORK_OF_ART": 18, "I-WORK_OF_ART": 19, "B-FAC":20, "B-TIME":21, "I-CARDINAL":22, "B-LOC":23,
                    "B-QUANTITY":24, "I-QUANTITY":25, "I-NORP":26, "I-LOC":27, "B-PRODUCT":28, "I-TIME":29, "B-EVENT":30, "I-EVENT":31,
                    "I-FAC":32, "B-LANGUAGE":33, "I-PRODUCT":34, "I-ORDINAL":35, "I-LANGUAGE":36},
        "id2label": {0:"O", 1:"B-CARDINAL", 2:"B-DATE", 3:"I-DATE", 4:"B-PERSON", 5:"I-PERSON", 6:"B-NORP", 7:"B-GPE", 8:"I-GPE",
                    9:"B-LAW", 10:"I-LAW", 11:"B-ORG", 12:"I-ORG", 13:"B-PERCENT", 14:"I-PERCENT", 15:"B-ORDINAL", 16:"B-MONEY",
                    17:"I-MONEY", 18:"B-WORK_OF_ART", 19:"I-WORK_OF_ART", 20:"B-FAC", 21:"B-TIME", 22:"I-CARDINAL", 23:"B-LOC",
                    24:"B-QUANTITY", 25:"I-QUANTITY", 26:"I-NORP", 27:"I-LOC", 28:"B-PRODUCT", 29:"I-TIME", 30:"B-EVENT", 31:"I-EVENT",
                    32:"I-FAC", 33:"B-LANGUAGE", 34:"I-PRODUCT", 35:"I-ORDINAL", 36:"I-LANGUAGE"},
        "labels": ["O", "B-CARDINAL", "B-DATE", "I-DATE", "B-PERSON", "I-PERSON", "B-NORP", "B-GPE", "I-GPE", "B-LAW", "I-LAW", 
                   "B-ORG", "I-ORG", "B-PERCENT", "I-PERCENT", "B-ORDINAL", "B-MONEY", "I-MONEY", "B-WORK_OF_ART", 
                   "I-WORK_OF_ART", "B-FAC", "B-TIME", "I-CARDINAL", "B-LOC", "B-QUANTITY", "I-QUANTITY", "I-NORP", "I-LOC", 
                   "B-PRODUCT", "I-TIME", "B-EVENT", "I-EVENT", "I-FAC", "B-LANGUAGE", "I-PRODUCT", "I-ORDINAL", "I-LANGUAGE"]
    }
}

def unpack_dataset_info(dataset_name):
    dataset_info_dict = datasets_dict[dataset_name]
    train_ds_name = dataset_info_dict["dataset_names"]["train"]
    valid_ds_name = dataset_info_dict["dataset_names"]["valid"]
    n_labels = dataset_info_dict["n_labels"]
    return dataset_info_dict, train_ds_name, valid_ds_name, n_labels    

def get_label_maps(raw_datasets, train_ds_name):
    labels = raw_datasets[train_ds_name].features["ner_tags"].feature

    id2label = {idx: name for idx, name in enumerate(labels.names)} if hasattr(labels, "names") else None
    label2id = {name: idx for idx, name in enumerate(labels.names)} if hasattr(labels, "names") else None

    return id2label, label2id

#https://huggingface.co/docs/transformers/main/en/tasks/token_classification
def tokenize_and_align_labels(examples, tokenizer, max_seq_length):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True)
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

def load_and_preprocess_dataset(ds_info_dict: dict, tokenizer=None, config=None):
    """
    Loads a Hugging Face dataset, retrieves label mappings, and tokenizes the data with aligned labels.

    Args:
        ds_info_dict (dict): Dictionary containing dataset metadata, including download reference and label info.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use for tokenization.
        config (dict): Configuration dictionary, must include 'max_seq_length' for tokenization.

    Returns:
        tokenized_aligned_dataset (datasets.DatasetDict): Tokenized dataset with aligned labels for NER.
        labels_list (list): List of label names.
        id2label (dict): Mapping from label IDs to label names.
        label2id (dict): Mapping from label names to label IDs.
    """
    # Load raw dataset
    raw_ds = load_dataset(ds_info_dict['download_reference'], trust_remote_code=True)
    print(raw_ds)
    # Get id2label and label2id
    if any(d in ds_info_dict["download_reference"] for d in ("bc5cdr", "conll2003", "ontonotes")):
        labels_list = ds_info_dict['labels']
        id2label, label2id =  ds_info_dict['id2label'], ds_info_dict['label2id']
    else: 
        labels_list = raw_ds["train"].features["ner_tags"].feature.names
        id2label, label2id = get_label_maps(raw_ds, ds_info_dict["dataset_names"]["train"])
    # Tokenize and align labels
    tokenized_aligned_dataset = raw_ds.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer, "max_seq_length": config["max_seq_length"]})
    print(tokenized_aligned_dataset)
    return tokenized_aligned_dataset, labels_list, id2label, label2id

def get_dataloaders(tokenized_aligned_dataset, config, splits=[], collate_fn=None, select=None):
    # Convert to PyTorch format
    torch_ds = tokenized_aligned_dataset.with_format("torch")
    if select is None or (select == -1):
        print("No selection limit applied, using full dataset.")
        train_dataset = torch_ds['train']
        eval_dataset = torch_ds['validation']
        test_dataset = torch_ds['test']
    else:
        print(f"Selection limit applied, loading {select} instances of data per split.")
        train_dataset = torch_ds['train'].select(range(select))
        eval_dataset = torch_ds['validation'].select(range(select))
        test_dataset = torch_ds['test'].select(range(select))

    # Create DataLoaders
    bs = config["batch_size"]
    train_loader, val_loader, test_loader = None, None, None
    if "train" in splits:
        train_loader = DataLoader(train_dataset, batch_size=bs, collate_fn=collate_fn)
    if "val" in splits:
        val_loader = DataLoader(eval_dataset, batch_size=bs,collate_fn=collate_fn)
    if "test" in splits:
        test_loader = DataLoader(test_dataset, batch_size=bs, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader

class SplitInstanceCollate:
    def __init__(self, tokenizer, max_length=512, overlap=128):
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForTokenClassification(tokenizer, padding="longest")
        self.max_length = max_length
        if overlap >= max_length:
            raise ValueError(f"Overlap ({overlap}) must be less than max_length ({max_length}).")
        self.overlap = overlap
    
    def calc_special_tokens_offset(self, input_ids: torch.tensor, att_mask, labels) -> int:
        special_tokens_offset = 0
        diff = 0
        add_cls, add_sep = False, False
        seq_len = len(input_ids)
        # If current chunk doesn't start with [CLS]
        if input_ids[0] != self.tokenizer.cls_token_id:
            add_cls = True
            special_tokens_offset += 1
        # If current chunk doesn't end with [SEP]
        if input_ids[-1] != self.tokenizer.sep_token_id:
            add_sep = True
            special_tokens_offset += 1
        # If adding special tokens surpasses max sequence length
        if len(input_ids) + special_tokens_offset > self.max_length:
            diff = len(input_ids) + special_tokens_offset - self.max_length
        assert 0 <= diff <= 2, f"impossible for diff: {diff} to be lower than 0 or larger than 2" 
        # Throw away last diff tokens, next chunk will take them they dont fit here
        input_ids = input_ids[:seq_len - diff]
        att_mask = att_mask[:seq_len - diff]
        labels = labels[:seq_len - diff]
        if add_cls:
            cls_tensor = torch.tensor([self.tokenizer.cls_token_id])
            input_ids = torch.cat((cls_tensor, input_ids))
            att_mask = torch.cat((torch.tensor([1]), att_mask))
            labels = torch.cat((torch.tensor([-100]), labels))
        if add_sep:
            sep_tensor = torch.tensor([self.tokenizer.sep_token_id])
            input_ids = torch.cat((input_ids, sep_tensor))
            att_mask = torch.cat((att_mask, torch.tensor([1])))
            labels = torch.cat((labels, torch.tensor([-100])))

        assert len(input_ids) == len(att_mask) == len(labels), "Input IDs, attention mask, and labels must have the same length."
        assert len(input_ids) <= self.max_length, "Input IDs length exceeds max_length."
        return (input_ids, att_mask, labels), diff

    def __call__(self, batch):
        new_input_ids = []
        new_attention_mask = []
        new_labels = []
        instance_ids = []

        for idx, item in enumerate(batch):
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            labels = item["labels"]
            # Get length of current input_ids
            seq_len = len(input_ids)
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            #print(f"instance({idx}), seq_len: {seq_len}, tokens: {tokens}")
            # Check if current instance is longer than the max allowed
            if seq_len <= self.max_length:
                new_input_ids.append(input_ids)
                new_attention_mask.append(attention_mask)
                new_labels.append(labels)
                instance_ids.append(idx)
            else:
                start = 0
                end = 0
                while end < seq_len:
                    end = min(start + self.max_length, seq_len)
                    #print(f"start: {start}, end: {end}, max_length: {self.max_length}")
                    chunked_tup, special_offset = self.calc_special_tokens_offset(
                                                input_ids[start:end], 
                                                attention_mask[start:end], 
                                                labels[start:end]
                                            )
                    chunk_input_ids, chunk_attention_mask, chunk_labels = chunked_tup 
                    # Append the new chunk to the lists
                    new_input_ids.append(chunk_input_ids)
                    new_attention_mask.append(chunk_attention_mask)
                    new_labels.append(chunk_labels)
                    start += self.max_length - self.overlap - special_offset
                    instance_ids.append(idx)    
                    tokens = self.tokenizer.convert_ids_to_tokens(chunk_input_ids)
                    #print(f"Chunked instance({idx}), tokens: {tokens}")
                    
        # TODO create assert to verify if chunking is correct
        # Pad sequences to the longest in the batch
        features = []
        for i in range(len(new_input_ids)):
            features.append({
                "input_ids": new_input_ids[i],
                "attention_mask": new_attention_mask[i],
                "labels": new_labels[i]
            })
        #print(f"Features: {features}")
        padded_inputs = self.data_collator(features)
        #print(f"Padded Inputs: {padded_inputs}")
        return padded_inputs, instance_ids