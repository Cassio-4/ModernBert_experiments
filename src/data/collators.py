import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorForTokenClassification

def get_collator(cfg, tokenizer=None):
    """
    Get data collator based on configuration.
    """
    if tokenizer is None:
        raise ValueError("A tokenizer must be provided to get the collator.")

    if cfg['data']['collator']['name'] == 'SplitInstanceCollate':
        return SplitInstanceCollate(tokenizer, max_length=cfg['data']['max_seq_length'], overlap=cfg['data']['collator']['overlap'])
    else:
        raise ValueError(f"Unknown collator name: {cfg['data']}")

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
            #print(f"labels: {labels}")
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
        padded_inputs = self.data_collator(features)
        return padded_inputs, instance_ids

class NerSlidingWindowReconstructor():
    def __init__(self, tokenizer=None, overlap=-1):
        if tokenizer is None:
            raise ValueError(f"A Tokenizer must be provided in order to get current special token's IDs.")
        self.tokenizer = tokenizer
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        if overlap < 0:
            raise ValueError(f"Invalid Overlap value.")
        self.overlap = overlap
        print("Initialized Reconstructor")
    
    def reconstruct_sequences(self, model_output, instance_ids, batch_data):
        """
        Reconstruct sequences from sliding window outputs.
        
        Args:
            model_output: Tensor outputed by the model [batch_size, num_tokens, emb_dim]
            instance_ids: list of ids that map each instance within a batch
            
        Returns:
            reconstructed_batch: Dictionary containing reconstructed embeddings, input_ids, and labels.
        """
        unique_ids, counts = torch.unique_consecutive(torch.tensor(instance_ids), return_counts=True)
        # Separate the batch into each instance
        chunks = torch.split(model_output, counts.tolist())
        chunks_ids = torch.split(batch_data['input_ids'], counts.tolist())
        chunks_labels = torch.split(batch_data['labels'], counts.tolist())
        reconstructed_batch = {"embeddings": [], "input_ids": [], "labels": []}
        for i in range(len(chunks)):
            curr_chunk = chunks[i]
            if len(curr_chunk) > 1: 
                clean_chunks_emb = self._split_chunks(curr_chunk, chunks_ids[i])
                clean_ids = self._split_chunks(chunks_ids[i], chunks_ids[i])
                clean_labels = self._split_chunks(chunks_labels[i], chunks_ids[i])
                merged_emb = self._merge_chunks_emb_avg(clean_chunks_emb)
                merged_ids = self._merge_input_data(clean_ids)
                merged_labels = self._merge_input_data(clean_labels)
                reconstructed_batch["embeddings"].append(merged_emb)
                reconstructed_batch["input_ids"].append(merged_ids)
                reconstructed_batch["labels"].append(merged_labels)
            else:
                reconstructed_batch["embeddings"].append(curr_chunk.squeeze(0))
                reconstructed_batch["input_ids"].append(chunks_ids[i].squeeze(0))
                reconstructed_batch["labels"].append(chunks_labels[i].squeeze(0))
        
        reconstructed_batch["embeddings"] = pad_sequence(reconstructed_batch["embeddings"], batch_first=True)
        reconstructed_batch["input_ids"] = pad_sequence(reconstructed_batch["input_ids"], batch_first=True)
        reconstructed_batch["labels"] = pad_sequence(reconstructed_batch["labels"], batch_first=True, padding_value=-100)
        return reconstructed_batch
                
    def _split_chunks(self, chunks, ids) -> list:
        """
        Takes tensor of chunks that belong to the same sentence, split each
        into a dictionary and return a list of dictionaries for further processing
        
        Args:
            chunks: Tensor of shape [num_chunks, chunk_size, emb_dim]
            data_batch: Original batch of data coming from dataloader
        
        Returns:
            clean_chunks: List of dictionaries that map each chunk's part
        """
        # we'll first separate each chunk in a dictionary that maps the 
        # special tokens [CLS] and [SEP], the part that overlaps with the 
        # previous chunk, and its own part (stride)
        # chunk composition -> [CLS, ...overlap..., ...stride..., SEP, ...PAD...]
        num_chunks = chunks.shape[0]
        chunk_len = chunks.shape[1]
        
        clean_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_dict = {}
            # Save special tokens' embeddings
            chunk_dict["cls"] = chunk[0].unsqueeze(0)
            sep_index = (ids[i] == self.sep_id).nonzero(as_tuple=True)[0]
            chunk_dict["sep"] = chunk[sep_index]
            # Treat edge cases (first chunk no previous overlap, last chunk padding etc.)
            if i == 0:
                # First chunk doesn't overlap with previous (inexistent) chunk
                chunk_dict["overlap"] = None
                chunk_dict["stride"] = chunk[1:sep_index]
            else:
                # Save overlap part: first overlap tokens after CLS
                chunk_dict["overlap"] = chunk[1:self.overlap+1]
                assert len(chunk_dict["overlap"]) == self.overlap, f"overlap size is different from given overlap"
                # Save stride, between overlap and SEP
                chunk_dict["stride"] = chunk[self.overlap+1: sep_index]
            # If last chunk then may contain padding
            if i == len(chunks)-1:
                # Check if SEP is last index, if not then has PAD
                if sep_index != chunk_len - 1:
                    chunk_dict["padding"] = chunk[sep_index + 1:]
            clean_chunks.append(chunk_dict)
        return clean_chunks

    def _merge_chunks_emb_avg(self, clean_chunks: list):
        """
        Takes a list of pre-processed chunks dictionaries, and
        merge them into a single sequence with normalization.
        
        Args:
            clean_chunks: List of dictionaries that map each chunk's part
                          with keys "cls", "sep", "overlap", "stride" and "padding"
        Returns:
            recon: Tensor of shape [total_length, emb_dim] in the form [CLS, ...embeddings..., SEP, padding...]
                    corresponding to the merged embeddings that form the original data sequence.
        """
        emb_dim = clean_chunks[0]["cls"].shape[-1]
        device = clean_chunks[0]["cls"].device
        
        final_cls = torch.zeros((1, emb_dim), device=device)
        final_sep = torch.zeros((1, emb_dim), device=device)
        recon = None
        for i in range(len(clean_chunks)):
            # Only start concating after first chunk
            if i != 0:
                overlap_mean = (prev_chunk_dict["stride"][-self.overlap:] + clean_chunks[i]["overlap"]) / 2
                if recon is None:
                    recon = torch.cat((prev_chunk_dict["stride"][:-self.overlap], overlap_mean))
                else:
                    recon = torch.cat((recon, prev_chunk_dict["stride"][:-self.overlap], overlap_mean))

            final_cls = final_cls + clean_chunks[i]["cls"]
            final_sep = final_sep + clean_chunks[i]["sep"]
            prev_chunk_dict = clean_chunks[i]

            if i == len(clean_chunks)-1:
                final_cls = final_cls / len(clean_chunks)
                final_sep = final_sep / len(clean_chunks)
                if "padding" in clean_chunks[i]:
                    recon = torch.cat((final_cls, recon, clean_chunks[i]["stride"], final_sep, clean_chunks[i]["padding"]))
                else:
                    recon = torch.cat((final_cls, recon, clean_chunks[i]["stride"], final_sep))
        
        return recon
    
    def _merge_input_data(self, clean_input):
        recon = None
        for i in range(len(clean_input)):
            # Only start concating after first chunk
            if i != 0:
                # Check if overlap is correct, each token_id/label should be the exact same
                is_overlap_correct = torch.equal(prev_chunk_dict["stride"][-self.overlap:], clean_input[i]["overlap"]) 
                assert is_overlap_correct, f"overlap not correct.\n previous stride: {prev_chunk_dict['stride']}\n curr_overlap: {clean_input[i]['overlap']}"
                if recon is None:
                    recon = torch.cat((prev_chunk_dict["stride"], clean_input[i]["stride"]))
                else:
                    recon = torch.cat((recon, clean_input[i]["stride"]))
                # If last part of chunk, treat padding
                if i == len(clean_input)-1:
                    if "padding" in clean_input[i]:
                        recon = torch.cat((clean_input[i]["cls"], recon, clean_input[i]["sep"], clean_input[i]["padding"]))
                    else:
                        recon = torch.cat((clean_input[i]["cls"], recon, clean_input[i]["sep"]))
            prev_chunk_dict = clean_input[i]
        return recon

    def print_clean_chunks_dict(self, clean_chunks):
        for d in clean_chunks:
            for key, value in d.items():
                if value is not None:  # Check if the value is not None
                    print(f"{key}: shape = {value.shape}")
                else:
                    print(f"{key}: value is None")
                if key == "input_ids":
                    print(f"{key}: tokens = {self.tokenizer.convert_ids_to_tokens(value[0])}")
            print("---------------------------")
