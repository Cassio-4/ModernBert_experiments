from transformers import AutoTokenizer, ModernBertForTokenClassification
import torch
#from ModernBERT import BertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = ModernBertForTokenClassification.from_pretrained("answerdotai/ModernBERT-base")

inputs = tokenizer(
    "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
)
print(model)

with torch.no_grad():
    logits = model(**inputs).logits

predicted_token_class_ids = logits.argmax(-1)
print(predicted_token_class_ids)

# Note that tokens are classified rather then input words which means that
# there might be more predicted token classes than words.
# Multiple token classes might account for the same word
predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]

labels = predicted_token_class_ids
loss = model(**inputs, labels=labels).loss
print(predicted_tokens_classes)