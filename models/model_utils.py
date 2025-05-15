from transformers import AutoModelForTokenClassification

def load_model(config, n_labels, id2label, label2id):
    model_additional_kwargs = {"id2label": id2label, "label2id": label2id} if id2label and label2id else {}
    hf_model = AutoModelForTokenClassification.from_pretrained(config["model_checkpoint"], 
                                                               num_labels=n_labels, **model_additional_kwargs)
    if config["dyt"]:
        from convert_ln_to_dyt import convert_ln_to_dyt
        hf_model = convert_ln_to_dyt(hf_model)
    # compile=False,
    return hf_model