import torch
import torch.nn as nn
from transformers import AutoModel

class JointPhoBERTModel(nn.Module):
    def __init__(self, model_name, num_intents, num_ner_tags, dropout_prob=0.1):
        super(JointPhoBERTModel, self).__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.phobert.config.hidden_size
        
        self.dropout = nn.Dropout(dropout_prob)
        
        # 1. Head cho bài toán Phân loại ý định (Intent Classification)
        self.intent_classifier = nn.Linear(hidden_size, num_intents)
        
        # 2. Head cho bài toán Nhận dạng thực thể (NER / Token Classification)
        self.ner_classifier = nn.Linear(hidden_size, num_ner_tags)

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Lấy hidden states của tất cả các token
        sequence_output = outputs.last_hidden_state
        
        # --- Intent Classification ---
        cls_output = sequence_output[:, 0, :]
        cls_output = self.dropout(cls_output)
        intent_logits = self.intent_classifier(cls_output)
        
        # --- NER Classification ---
        sequence_output_dropout = self.dropout(sequence_output)
        ner_logits = self.ner_classifier(sequence_output_dropout)
        
        return intent_logits, ner_logits
