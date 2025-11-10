import torch
torch.set_float32_matmul_precision('high')
from transformers import AutoTokenizer, ModernBertModel

model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ModernBertModel.from_pretrained(model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

texts = ["The capital of France is Paris.", "The capital of Germany is Berlin."]

inputs = tokenizer(
    text=texts,
    add_special_tokens=True,
    padding='max_length',
    truncation=True,
    max_length=768,
    return_attention_mask=True,
    return_tensors='pt' 
)

input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)

outputs = model(input_ids=input_ids, attention_mask=attention_mask)

print(outputs.last_hidden_state)