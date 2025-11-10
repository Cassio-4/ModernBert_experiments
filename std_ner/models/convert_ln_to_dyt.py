from transformers import ModernBertForTokenClassification
import torch.nn as nn
import torch

#https://github.com/jiachenzhu/DyT/blob/main/dynamic_tanh.py
class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            # Error: The size of tensor a (1) must match the size of tensor b (768) at non-singleton dimension 0
            #x = x * self.weight[:, None, None] + self.bias[:, None, None]
            x = x * self.weight + self.bias
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


def convert_ln_to_dyt(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape, not isinstance(module, nn.LayerNorm))
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    del module
    return module_output

""""
if __name__ == "__main__":
    model = ModernBertForTokenClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=2)
    modules_count = 0
    for name, module in model.named_modules():
        print(name, type(module))
        modules_count += 1
    print(f"Before conversion: layer num = {modules_count}")
    input_ids = torch.tensor([[101, 102, 103]])
    outputs = model(input_ids)
    print(outputs)
    print("..........Converting LayerNorm to DynamicTanh..........")
    modules_count = 0
    convert_ln_to_dyt(model)
    for name, module in model.named_modules():
        modules_count += 1
        print(name, type(module))
    print(f"After conversion: layer num = {modules_count}")
    outputs = model(input_ids)
    print(outputs)
"""