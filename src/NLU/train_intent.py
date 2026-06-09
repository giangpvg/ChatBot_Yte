import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from preprocess import load_data_and_encoders
from model_intent import JointPhoBERTModel
from sklearn.metrics import accuracy_score

def train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    data_dir = os.path.join(project_root, 'data')
    train_path = os.path.join(data_dir, 'train.json')
    dev_path = os.path.join(data_dir, 'dev.json')
    test_path = os.path.join(data_dir, 'test.json')
    
    model_name = "vinai/phobert-base-v2"
    batch_size = 16
    epochs = 10
    max_len = 128
    learning_rate = 2e-5
    
    print("1. Loading Data and Tokenizer...")
    train_dataset, dev_dataset, test_dataset, label_encoder, tokenizer = load_data_and_encoders(
        train_path, dev_path, test_path, model_name, max_len
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    num_intents = label_encoder.get_num_intents()
    num_ner_tags = label_encoder.get_num_ner_tags()
    print(f"   -> Num Intents: {num_intents}, Num NER Tags: {num_ner_tags}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"2. Using device: {device}")
    
    print("3. Initializing Model...")
    model = JointPhoBERTModel(model_name, num_intents, num_ner_tags)
    model.to(device)
    
    intent_criterion = nn.CrossEntropyLoss()
    # Bỏ qua index -100 (padding và các subwords phụ) khi tính loss cho NER
    ner_criterion = nn.CrossEntropyLoss(ignore_index=-100) 
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    best_val_loss = float('inf')
    
    print("4. Starting Training Loop...")
    for epoch in range(epochs):
        print(f"\\n========== Epoch {epoch+1}/{epochs} ==========")
        model.train()
        total_train_loss = 0
        total_intent_loss = 0
        total_ner_loss = 0
        
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            intent_labels = batch['intent_label'].to(device)
            ner_labels = batch['ner_labels'].to(device)
            
            optimizer.zero_grad()
            
            intent_logits, ner_logits = model(input_ids, attention_mask)
            
            loss_intent = intent_criterion(intent_logits, intent_labels)
            
            # Tính Loss NER (reshape lại logits và labels)
            active_logits = ner_logits.view(-1, num_ner_tags)
            active_labels = ner_labels.view(-1)
            loss_ner = ner_criterion(active_logits, active_labels)
            
            # Tổng hợp Loss
            loss = loss_intent + loss_ner
            total_train_loss += loss.item()
            total_intent_loss += loss_intent.item()
            total_ner_loss += loss_ner.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            if (step + 1) % 50 == 0:
                print(f"   Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f} (Intent: {loss_intent.item():.4f}, NER: {loss_ner.item():.4f})")
            
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"-> Average Training Loss: {avg_train_loss:.4f} (Intent: {total_intent_loss/len(train_loader):.4f}, NER: {total_ner_loss/len(train_loader):.4f})")
        
        # Validation
        model.eval()
        total_val_loss = 0
        all_intent_preds = []
        all_intent_labels = []
        
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                intent_labels = batch['intent_label'].to(device)
                ner_labels = batch['ner_labels'].to(device)
                
                intent_logits, ner_logits = model(input_ids, attention_mask)
                
                loss_intent = intent_criterion(intent_logits, intent_labels)
                active_logits = ner_logits.view(-1, num_ner_tags)
                active_labels = ner_labels.view(-1)
                loss_ner = ner_criterion(active_logits, active_labels)
                
                loss = loss_intent + loss_ner
                total_val_loss += loss.item()
                
                intent_preds = torch.argmax(intent_logits, dim=1).cpu().numpy()
                all_intent_preds.extend(intent_preds)
                all_intent_labels.extend(intent_labels.cpu().numpy())
                
        avg_val_loss = total_val_loss / len(dev_loader)
        val_intent_acc = accuracy_score(all_intent_labels, all_intent_preds)
        print(f"-> Validation Loss: {avg_val_loss:.4f} | Intent Accuracy: {val_intent_acc:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_dir = os.path.join(project_root, 'src', 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "best_joint_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"=> Saved new best model to '{ckpt_path}'")

if __name__ == "__main__":
    train()
