from torch.utils.data import Dataset
import torch
from typing import Dict, Optional

class DialogueDataset(Dataset):
    """대화 요약 데이터셋"""
    
    def __init__(self, 
                 encoder_input: Dict[str, torch.Tensor],
                 labels: Optional[Dict[str, torch.Tensor]] = None,
                 decoder_input: Optional[Dict[str, torch.Tensor]] = None,
                 tokenizer = None):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        
        if self.decoder_input is not None:
            decoder_item = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()}
            item['decoder_input_ids'] = decoder_item['input_ids']
            item['decoder_attention_mask'] = decoder_item['attention_mask']
            
        if self.labels is not None:
            labels = self.labels['input_ids'][idx].clone().detach()
            if self.tokenizer is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            item['labels'] = labels
            
        return item
    
    def __len__(self):
        return len(next(iter(self.encoder_input.values()))) 