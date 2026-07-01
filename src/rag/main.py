import sys
import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer
from openai import OpenAI
from huggingface_hub import InferenceClient
import os as _os
from pathlib import Path
import warnings

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

sys.path.append(os.path.join(project_root, 'src', 'NLU'))

from predict import predict, load_encoder
from model_intent import JointPhoBERTModel

warnings.filterwarnings('ignore')

# ĐỌC API KEY TỪ FILE .env
def load_env_vars(env_path=None):
    """Đọc các biến môi trường từ file .env."""
    if env_path is None:
        env_path = os.path.join(project_root, 'src', '.env')
    vars_dict = {}
    
    # Fallback cho Google Colab
    if not os.path.exists(env_path):
        colab_path = "/content/ChatBot_Yte/src/.env"
        if os.path.exists(colab_path):
            env_path = colab_path
            
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    vars_dict[k.strip()] = v.strip().strip('"').strip("'")
    except Exception as e:
        print(f"⚠️ Không đọc được file .env: {e}")
    return vars_dict

ENV_VARS = load_env_vars()
OPENAI_API_KEY = ENV_VARS.get("OPENAI_API_KEY")
OPENAI_API_BASE = ENV_VARS.get("OPENAI_API_BASE", "https://models.inference.ai.azure.com")

def load_nlu_model(device):
    model_name = "vinai/phobert-base-v2"
    train_path = os.path.join(project_root, "data", "train.json")
    ckpt_path = os.path.join(project_root, "src", "checkpoints", "best_joint_model.pth")

    # Fallback cho Colab nếu chạy từ xa
    if not os.path.exists(train_path):
        colab_train_path = "/content/ChatBot_Yte/data/train.json"
        if os.path.exists(colab_train_path):
            train_path = colab_train_path
            
    if not os.path.exists(ckpt_path):
        colab_ckpt_path = "/content/ChatBot_Yte/src/checkpoints/best_joint_model.pth"
        if os.path.exists(colab_ckpt_path):
            ckpt_path = colab_ckpt_path

    encoder = load_encoder(train_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = JointPhoBERTModel(model_name, encoder.get_num_intents(), encoder.get_num_ner_tags())
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    return model, tokenizer, encoder

def load_vector_db(device):
    persist_dir = os.path.join(project_root, "src", "rag", "chroma_db")

    # Fallback cho Colab nếu chạy từ xa
    if not os.path.exists(persist_dir):
        colab_persist_dir = "/content/ChatBot_Yte/src/rag/chroma_db"
        if os.path.exists(colab_persist_dir):
            persist_dir = colab_persist_dir

    embeddings = HuggingFaceEmbeddings(
        model_name="keepitreal/vietnamese-sbert",
        model_kwargs={'device': device}
    )
    vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return vector_db

def setup_openai(api_key, api_base):
    """Khởi tạo OpenAI client tương thích với Azure AI Inference."""
    client = OpenAI(
        base_url=api_base,
        api_key=api_key
    )
    return client

def setup_hf_client(hf_token):
    """Khởi tạo Hugging Face InferenceClient."""
    return InferenceClient(token=hf_token)

def build_prompt(query, intent, entity_words, retrieved_docs):
    """Xây dựng Prompt chuẩn y tế cho OpenAI LLM."""

    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        source = os.path.basename(doc.metadata.get('source', 'Tài liệu y tế'))
        context_parts.append(f"[Tài liệu {i+1} - Nguồn: {source}]\n{doc.page_content.strip()}")

    context = "\n\n".join(context_parts)

    # Map intent sang tiếng Việt để ra lệnh cho Gemini
    intent_instruction_map = {
        "treatment":       "Hãy tập trung vào PHƯƠNG PHÁP ĐIỀU TRỊ, thuốc và các bước xử lý.",
        "method_diagnosis":"Hãy tập trung vào TRIỆU CHỨNG, DẤU HIỆU NHẬN BIẾT và phương pháp chẩn đoán.",
        "cause":           "Hãy tập trung vào NGUYÊN NHÂN và các yếu tố nguy cơ gây bệnh.",
        "severity":        "Hãy tập trung vào ĐÁNH GIÁ MỨC ĐỘ NGUY HIỂM và khi nào cần đi khám.",
    }
    intent_instruction = intent_instruction_map.get(intent, "Hãy trả lời một cách toàn diện.")

    prompt = f"""
Bạn là một Trợ lý Y tế AI chuyên nghiệp. Nhiệm vụ của bạn là trả lời câu hỏi y tế của người dùng dựa HOÀN TOÀN vào các tài liệu y khoa chính thức được cung cấp bên dưới.

## NGUYÊN TẮC QUAN TRỌNG:
- CHỈ sử dụng thông tin có trong TÀI LIỆU Y TẾ được cung cấp. KHÔNG tự bịa đặt kiến thức bên ngoài.
- NẾU tài liệu tham khảo CÓ CHỨA thông tin để trả lời (dù chỉ một phần): Hãy tổng hợp và trả lời dựa trên tài liệu. KHÔNG ĐƯỢC chèn thêm câu "Tài liệu hiện có chưa đề cập...".
- NẾU tài liệu tham khảo HOÀN TOÀN KHÔNG chứa bất kỳ thông tin nào liên quan đến câu hỏi: Hãy trả lời DUY NHẤT một câu: "Tài liệu hiện có chưa đề cập đến vấn đề này, vui lòng tham khảo ý kiến bác sĩ." và KHÔNG giải thích gì thêm.
- Luôn kết thúc câu trả lời bằng lời khuyên đi khám bác sĩ (trừ trường hợp dùng câu từ chối ở trên).
- Trả lời bằng tiếng Việt, rõ ràng, có cấu trúc (dùng gạch đầu dòng nếu cần).

## PHÂN TÍCH CÂU HỎI:
- **Ý định người dùng (Intent):** {intent}
- **Thực thể y tế liên quan:** {', '.join(entity_words) if entity_words else 'Không xác định cụ thể'}
- **Hướng dẫn trả lời:** {intent_instruction}

## TÀI LIỆU Y TẾ THAM KHẢO:
{context}

## CÂU HỎI CỦA NGƯỜI DÙNG:
{query}

## CÂU TRẢ LỜI:"""

    return prompt

def generate_answer(openai_client, hf_client, prompt):
    """Gọi OpenAI API trước, nếu lỗi thì chuyển sang Hugging Face (Qwen)."""
    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Lỗi OpenAI: {e}. Đang chuyển sang Hugging Face...")
            
    if hf_client:
        try:
            messages = [{"role": "user", "content": prompt}]
            response = hf_client.chat_completion(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=messages,
                max_tokens=1024,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Lỗi cả 2 API. Lỗi Hugging Face: {e}"
            
    return "Không có kết nối API nào khả dụng."

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("1. Đang khởi tạo NLU Model (PhoBERT)...")
    nlu_model, tokenizer, encoder = load_nlu_model(device)

    print("2. Đang kết nối tới Vector Database (Chroma)...")
    vector_db = load_vector_db(device)

    print("3. Đang kết nối API...")
    HF_TOKEN = ENV_VARS.get("HF_TOKEN")
    openai_client = setup_openai(OPENAI_API_KEY, OPENAI_API_BASE) if OPENAI_API_KEY else None
    hf_client = setup_hf_client(HF_TOKEN) if HF_TOKEN else None

    if not openai_client and not hf_client:
        print("Không tìm thấy OPENAI_API_KEY hay HF_TOKEN!")
        return

    print("\n" + "="*60)
    print("CHATBOT Y TẾ - Powered by PhoBERT + RAG + OpenAI LLM")
    print("="*60)
    print("Nhập câu hỏi y tế của bạn bên dưới.")
    print("Nhập 'q' để thoát.\n")

    while True:
        query = input("Bạn: ")
        if query.strip().lower() == 'q':
            break
        if not query.strip():
            continue

        # Bước 1: NLU - Trích xuất ý định và thực thể
        intent, entities = predict(query, nlu_model, tokenizer, encoder, device)
        entity_words = [word.replace('_', ' ') for word, tag in entities if tag != "O"]

        if entity_words:
            print(f"\n[Phân tích] Intent: {intent} | Thực thể: {entity_words}")
        else:
            print(f"\n[Phân tích] Intent: {intent} | Không có thực thể y tế, dùng toàn bộ câu.")

        # Bước 2: Sử dụng câu truy vấn ở dạng chữ thường và tăng cường (Query Expansion)
        search_query = query.strip().lower()
        if "là gì" in search_query or "thế nào là" in search_query:
            search_query += " đại cương định nghĩa khái niệm"

        # Bước 3: RAG Retrieval
        print(" Đang tra cứu tài liệu y khoa...")
        retrieved_docs = vector_db.similarity_search(search_query, k=10)

        # Bước 4: Xây dựng Prompt và gọi OpenAI
        print("Đang tổng hợp câu trả lời với LLM...")
        prompt = build_prompt(query, intent, entity_words, retrieved_docs)
        answer = generate_answer(openai_client, hf_client, prompt)

        # Bước 5: In câu trả lời cuối cùng
        print("\n" + "─"*60)
        print(f"Trợ lý Y tế:\n")
        print(answer)
        print("─"*60)

if __name__ == "__main__":
    main()
