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
    "crossner": {"download_reference": "DFKI-SLT/cross_ner",
        "tasks": ['conll2003', 'politics', 'science', 'music','literature', 'ai'],
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "n_labels": 79
    },
    "ncbi": {
        "download_reference": "ncbi/ncbi_disease",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "n_labels": 3
    },
    "ontonotes": {
        "download_reference": "tner/ontonotes5",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "n_labels": 3
    }
}