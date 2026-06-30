import torch
from transformers import AutoTokenizer
from model_intent import JointPhoBERTModel
from preprocess import LabelEncoder
import json
import warnings
warnings.filterwarnings('ignore')

def load_encoder(train_path):
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    encoder = LabelEncoder()
    encoder.fit(train_data)
    return encoder

def predict(sentence, model, tokenizer, encoder, device, max_len=128):
    model.eval()
    
    # Giả định câu đầu vào đã được tách từ (Word Segmentation) bằng dấu '_'
    words = sentence.split()
    if not words:
        return "UNKNOWN", []
        
    input_ids = [tokenizer.cls_token_id]
    word_to_subword_map = [] # Lưu vị trí subword đầu tiên của mỗi từ
    
    for word in words:
        word_tokens = tokenizer.encode(word, add_special_tokens=False)
        if not word_tokens:
            continue
        word_to_subword_map.append(len(input_ids)) # Vị trí của subword đầu tiên
        input_ids.extend(word_tokens)
        
    input_ids.append(tokenizer.sep_token_id)
    
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        input_ids[-1] = tokenizer.sep_token_id
        
    attention_mask = [1] * len(input_ids)
    
    # Padding
    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids.extend([tokenizer.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long).to(device)
    
    with torch.no_grad():
        intent_logits, ner_logits = model(input_ids_tensor, attention_mask_tensor)
        
    intent_id = torch.argmax(intent_logits, dim=1).item()
    intent_label = encoder.id2intent.get(intent_id, "UNKNOWN")
    
    ner_preds = torch.argmax(ner_logits, dim=2).squeeze(0).cpu().numpy()
    
    extracted_entities = []
    # Trích xuất nhãn NER cho từng từ gốc
    for i, word in enumerate(words):
        if i < len(word_to_subword_map):
            subword_idx = word_to_subword_map[i]
            if subword_idx < max_len:
                tag_id = ner_preds[subword_idx]
                tag_label = encoder.id2ner.get(tag_id, "O")
                extracted_entities.append((word, tag_label))
            else:
                extracted_entities.append((word, "O"))
                
    return intent_label, extracted_entities

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "vinai/phobert-base-v2"
    
    print("Loading Encoder and Tokenizer...")
    encoder = load_encoder('../data/train.json')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading Model Weights...")
    model = JointPhoBERTModel(model_name, encoder.get_num_intents(), encoder.get_num_ner_tags())
    model.load_state_dict(torch.load("checkpoints/best_joint_model.pth", map_location=device))
    model.to(device)
    
    print("\n" + "="*50)
    print("MÔ HÌNH ĐÃ SẴN SÀNG!")
    print("Lưu ý: Từ ghép tiếng Việt cần nối bằng dấu '_' (VD: đau_đầu, bệnh_viện)")
    print("="*50 + "\n")
    
    while True:
        text = input("Nhập câu hỏi (hoặc 'q' để thoát): ")
        if text.strip().lower() == 'q':
            break
        if not text.strip():
            continue
            
        intent, entities = predict(text, model, tokenizer, encoder, device)
        print(f"\n--- KẾT QUẢ DỰ ĐOÁN ---")
        print(f"🔹 Intent: {intent}")
        print(f"🔹 Thực thể (NER):")
        has_entity = False
        for word, tag in entities:
            if tag != "O":
                print(f"   [{tag}] -> {word.replace('_', ' ')}")
                has_entity = True
        if not has_entity:
            print("   (Không tìm thấy thực thể nào)")
        print("-" * 30 + "\n")
