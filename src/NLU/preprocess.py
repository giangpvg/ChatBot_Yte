import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class LabelEncoder:
    def __init__(self):
        self.intent2id = {}
        self.id2intent = {}
        # Reserve PAD and O
        self.ner2id = {"PAD": 0, "O": 1}
        self.id2ner = {0: "PAD", 1: "O"}

    def fit(self, data):
        for item in data:
            intent = item.get('sent_label', 'UNKNOWN')
            if intent not in self.intent2id:
                idx = len(self.intent2id)
                self.intent2id[intent] = idx
                self.id2intent[idx] = intent

            seq_label = item.get('seq_label', [])
            for label in seq_label:
                if len(label) >= 3:
                    entity = label[2]
                    b_tag = f"B-{entity}"
                    i_tag = f"I-{entity}"
                    
                    if b_tag not in self.ner2id:
                        idx = len(self.ner2id)
                        self.ner2id[b_tag] = idx
                        self.id2ner[idx] = b_tag
                        
                        idx = len(self.ner2id)
                        self.ner2id[i_tag] = idx
                        self.id2ner[idx] = i_tag
    
    def get_num_intents(self):
        return len(self.intent2id)

    def get_num_ner_tags(self):
        return len(self.ner2id)


class MedicalDataset(Dataset):
    def __init__(self, data_path, tokenizer, label_encoder, max_len=128):
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_len = max_len
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item.get('sentence', '')
        words = sentence.split()
        
        # 1. Chuyển đổi định dạng [start, end, entity] sang mảng BIO tags theo mức từ (word-level)
        word_labels = ["O"] * len(words)
        for label in item.get('seq_label', []):
            if len(label) >= 3:
                start, end, entity = label
                if start < len(words):
                    word_labels[start] = f"B-{entity}"
                for i in range(start + 1, min(end + 1, len(words))):
                    word_labels[i] = f"I-{entity}"

        # 2. Tokenize thủ công và Alignment cho Non-Fast Tokenizer (PhoBERT)
        input_ids = [self.tokenizer.cls_token_id]
        label_ids = [-100]

        for word, tag in zip(words, word_labels):
            # Tokenize từng từ 
            word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
            
            if not word_tokens:
                continue
                
            input_ids.extend(word_tokens)
            
            # Gán nhãn: Subword đầu tiên nhận nhãn gốc, các subwords sau nhận -100
            label_id = self.label_encoder.ner2id.get(tag, self.label_encoder.ner2id["O"])
            label_ids.append(label_id)
            label_ids.extend([-100] * (len(word_tokens) - 1))

        # Thêm token </s> ở cuối
        input_ids.append(self.tokenizer.sep_token_id)
        label_ids.append(-100)

        # Truncation 
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            label_ids = label_ids[:self.max_len]
            input_ids[-1] = self.tokenizer.sep_token_id
            label_ids[-1] = -100

        attention_mask = [1] * len(input_ids)

        # Padding 
        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            label_ids.extend([-100] * padding_length)

        # Lấy nhãn Intent
        intent = item.get('sent_label', 'UNKNOWN')
        intent_id = self.label_encoder.intent2id.get(intent, 0) # Mặc định 0 nếu không tìm thấy (lúc eval)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'intent_label': torch.tensor(intent_id, dtype=torch.long),
            'ner_labels': torch.tensor(label_ids, dtype=torch.long)
        }

def load_data_and_encoders(train_path, dev_path, test_path, model_name="vinai/phobert-base-v2", max_len=128):
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_data)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = MedicalDataset(train_path, tokenizer, label_encoder, max_len)
    dev_dataset = MedicalDataset(dev_path, tokenizer, label_encoder, max_len)
    test_dataset = MedicalDataset(test_path, tokenizer, label_encoder, max_len)
    
    return train_dataset, dev_dataset, test_dataset, label_encoder, tokenizer

if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    train_path = os.path.join(project_root, 'data', 'train.json')
    dev_path = os.path.join(project_root, 'data', 'dev.json')
    test_path = os.path.join(project_root, 'data', 'test.json')
    
    print("Test DataLoader...")
    train_ds, dev_ds, test_ds, encoder, tokenizer = load_data_and_encoders(train_path, dev_path, test_path)
    print(f"Num Intents: {encoder.get_num_intents()}, Num NER Tags: {encoder.get_num_ner_tags()}")
    
    sample = train_ds[0]
    print("Input IDs shape:", sample['input_ids'].shape)
    print("Intent Label:", sample['intent_label'])
    print("NER Labels:", sample['ner_labels'])
