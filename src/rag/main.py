import sys
import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer
from openai import OpenAI
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

    prompt = f"""Bạn là một Trợ lý Y tế AI chuyên nghiệp. Nhiệm vụ của bạn là trả lời câu hỏi y tế của người dùng dựa HOÀN TOÀN vào các tài liệu y khoa chính thức được cung cấp bên dưới.

## NGUYÊN TẮC QUAN TRỌNG:
- CHỈ sử dụng thông tin có trong TÀI LIỆU Y TẾ được cung cấp. KHÔNG tự bịa đặt.
- Nếu tài liệu tham khảo HOÀN TOÀN không chứa thông tin để trả lời câu hỏi, hãy trả lời duy nhất câu sau và không giải thích thêm: "Tài liệu hiện có chưa đề cập đến vấn đề này, vui lòng tham khảo ý kiến bác sĩ."
- Nếu tài liệu có chứa thông tin nhưng không đầy đủ, hãy trả lời phần thông tin có sẵn trong tài liệu, TUYỆT ĐỐI không tự ý vẽ thêm hoặc bổ sung các chi tiết ngoài tài liệu (như thuốc, phương pháp cấp cứu) từ kiến thức ngoại cảnh của bạn.
- Luôn kết thúc bằng lời khuyên người dùng đến gặp bác sĩ/cơ sở y tế để được tư vấn trực tiếp.
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

def generate_answer(openai_client, prompt):
    """Gọi OpenAI API (Azure AI Inference / GitHub Models) để sinh câu trả lời."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini", # Model mặc định trên GitHub Models
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Lỗi khi gọi OpenAI API: {e}"

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("1. Đang khởi tạo NLU Model (PhoBERT)...")
    nlu_model, tokenizer, encoder = load_nlu_model(device)

    print("2. Đang kết nối tới Vector Database (Chroma)...")
    vector_db = load_vector_db(device)

    print("3. Đang kết nối OpenAI-compatible API...")
    if not OPENAI_API_KEY:
        print("CẢNH BÁO: Không tìm thấy OPENAI_API_KEY!")
        print("   Định dạng: OPENAI_API_KEY = YOUR_KEY_HERE")
        return
    print(f"Đã đọc API Key từ file .env (****{OPENAI_API_KEY[-6:]})")
    openai_client = setup_openai(OPENAI_API_KEY, OPENAI_API_BASE)

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

        # Bước 2: Sử dụng câu truy vấn ở dạng chữ thường (SBERT nhạy cảm với chữ hoa/chữ thường, giúp khớp tốt hơn)
        search_query = query.strip().lower()

        # Bước 3: RAG Retrieval
        print(" Đang tra cứu tài liệu y khoa...")
        retrieved_docs = vector_db.similarity_search(search_query, k=3)

        # Bước 4: Xây dựng Prompt và gọi OpenAI
        print("Đang tổng hợp câu trả lời với LLM...")
        prompt = build_prompt(query, intent, entity_words, retrieved_docs)
        answer = generate_answer(openai_client, prompt)

        # Bước 5: In câu trả lời cuối cùng
        print("\n" + "─"*60)
        print(f"Trợ lý Y tế:\n")
        print(answer)
        print("─"*60)

if __name__ == "__main__":
    main()
