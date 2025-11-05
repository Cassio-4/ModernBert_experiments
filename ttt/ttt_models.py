import torch
from transformers import AutoModelForTokenClassification
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from torch import nn
from typing import Optional

def build_model(model_cfg, num_labels):
	print('Building model...')
	net = AutoModelForTokenClassification.from_pretrained(model_cfg['model_checkpoint'], num_labels=num_labels).cuda()
	if model_cfg['shared'] == 'layer3':
		embeddings, encoder_layers = extractor_from_layer3(net)
		head = BertOnlyMLMHead(net.config) 
	elif model_cfg['shared'] == 'layer2':
		ext = extractor_from_layer2(net)
		head = head_on_layer2(net, args.width, 4)
	ssh = BertSelfSupervisedHead(embeddings, encoder_layers, head).cuda()

	return net, ssh

class BertSelfSupervisedHead(nn.Module):
	def __init__(self, embeddings, encoder_layers, head):
		super(BertSelfSupervisedHead, self).__init__()
		self.embeddings = embeddings
		self.encoder_layers = encoder_layers
		self.head = head

	def forward(self, input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None):
		embeddings = self.embeddings(input_ids)
		hidden_states = embeddings
		
		extended_attention_mask = self.reshape_att_mask(attention_mask, hidden_states)

		# Pass through ext (encoder layers)
		for i, layer_module in enumerate(self.encoder_layers):
			layer_outputs = layer_module(hidden_states, extended_attention_mask)
			hidden_states = layer_outputs[0]

		mlm_result = self.head(hidden_states)
		return mlm_result
	
	def reshape_att_mask(self, attention_mask, hidden_states):
		if attention_mask is None:
			extended_attention_mask = None
		else:
            # [B, S] -> [B, 1, 1, S], match hidden dtype/device
			extended_attention_mask = attention_mask[:, None, None, :].to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
            # convert to additive mask: 1 -> 0.0 (keep), 0 -> large negative (mask)
			extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
		return extended_attention_mask

def extractor_from_layer3(net):
	# Extract first 3 layers
	embeddings = net.bert.embeddings
	encoder_layers =  nn.ModuleList(net.bert.encoder.layer[:3])
    # Optionally, add the embedding and pooler if needed
    #layers = [net.bert.embeddings] + encoder_layers
	#nn.Sequential(*encoder_layers)
	return embeddings, encoder_layers 

def extractor_from_layer2(net):
	layers = [net.conv1, net.layer1, net.layer2]
	return nn.Sequential(*layers)

def head_on_layer2(net, width, classes):
	head = copy.deepcopy([net.layer3, net.bn, net.relu, net.avgpool])
	head.append(ViewFlatten())
	head.append(nn.Linear(64 * width, classes))
	return nn.Sequential(*head)