import os
import sys
import torch
import streamlit as st
from transformers import AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'src'))
sys.path.append(os.path.join(ROOT_DIR, 'src', 'NLU'))
sys.path.append(os.path.join(ROOT_DIR, 'src', 'rag'))

from predict import predict, load_encoder
from model_intent import JointPhoBERTModel
from main import build_prompt, generate_answer, setup_openai, load_env_vars
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


st.set_page_config(
    page_title="Trợ lý Y tế AI - Chatbot RAG",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho phong cách Premium Medical
st.markdown("""
<style>
    /* Gradient background cho tiêu đề */
    .medical-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 24px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .medical-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 1px;
    }
    .medical-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 10px;
    }
    /* Thẻ thực thể (Entity Chips) */
    .entity-chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 2px 4px;
        color: white;
    }
    .intent-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        background-color: #e3f2fd;
        color: #0d47a1;
        border: 1px solid #bbdefb;
    }
    .source-card {
        background-color: #f8f9fa;
        border-left: 5px solid #203a43;
        padding: 12px;
        margin: 10px 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_nlu_resources():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "vinai/phobert-base-v2"
    train_path = os.path.join(ROOT_DIR, "data", "train.json")
    ckpt_path = os.path.join(ROOT_DIR, "src", "checkpoints", "best_joint_model.pth")
    
    if not os.path.exists(train_path):
        st.error(f"Không tìm thấy dữ liệu tập train tại `{train_path}`.")
        st.stop()
    if not os.path.exists(ckpt_path):
        st.warning(f"Chưa tìm thấy file checkpoint mô hình NLU tại `{ckpt_path}`. Vui lòng chạy huấn luyện trước bằng cách thực thi `python src/NLU/train_intent.py`.")
        return None, None, None, device

    encoder = load_encoder(train_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = JointPhoBERTModel(model_name, encoder.get_num_intents(), encoder.get_num_ner_tags())
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, tokenizer, encoder, device

@st.cache_resource
def load_vector_store():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    persist_dir = os.path.join(ROOT_DIR, "src", "rag", "chroma_db")
    
    if not os.path.exists(persist_dir):
        st.warning(f"Chưa tìm thấy Cơ sở dữ liệu Vector tại `{persist_dir}`. Vui lòng tạo DB trước bằng cách thực thi `python src/rag/offline_rag.py`.")
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="keepitreal/vietnamese-sbert",
        model_kwargs={'device': device}
    )
    vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return vector_db

# Khởi tạo mô hình và DB
nlu_model, tokenizer, encoder, device = load_nlu_resources()
vector_db = load_vector_store()

# Đọc API Key mặc định từ .env
env_vars = load_env_vars(os.path.join(ROOT_DIR, "src", ".env"))
default_api_key = env_vars.get("OPENAI_API_KEY", "")
default_api_base = env_vars.get("OPENAI_API_BASE", "https://models.inference.ai.azure.com")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/809/809957.png", width=80)
    st.title("Cấu hình hệ thống")
    
    api_key_input = st.text_input(
        "OpenAI / Azure Inference API Key",
        value=default_api_key,
        type="password",
        help="Khóa API để gọi mô hình ngôn ngữ lớn (LLM). Mặc định đọc từ file src/.env"
    )
    
    api_base_input = st.text_input(
        "API Base URL",
        value=default_api_base,
        help="Endpoint API. Mặc định sử dụng GitHub Models / Azure AI Inference."
    )
    
    st.divider()
    st.markdown("### Câu hỏi gợi ý")
    
    sample_queries = [
        "Bé bị sốt cao co giật phải làm sao?",
        "Nguyên nhân gây tăng huyết áp là gì?",
        "Cách phòng ngừa tăng huyết áp thế nào?",
        "Triệu chứng khi trẻ bị viêm phổi?",
        "Khi nào cần đưa trẻ bị tiêu chảy đi khám ngay?"
    ]
    
    for q in sample_queries:
        if st.button(q, use_container_width=True, key=q):
            st.session_state.suggested_query = q
            
    st.divider()
    if st.button("Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown("""
<div class="medical-header">
    <h1 class="medical-title">🏥 TRỢ LÝ Y TẾ AI THÔNG MINH</h1>
    <p class="medical-subtitle">Kết hợp hiểu ý định người dùng (PhoBERT NLU) & Cơ sở dữ liệu tài liệu y khoa chính thống (RAG)</p>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị các tin nhắn trong lịch sử
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Nếu có thông tin phân tích đi kèm (NLU / Retrieval), hiển thị dưới dạng expander
        if "analysis" in message:
            analysis = message["analysis"]
            with st.expander("🔍 Chi tiết phân tích hệ thống"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### Phân tích Ý định & Thực thể (NLU)")
                    st.markdown(f"**Ý định (Intent):** <span class='intent-badge'>{analysis['intent']}</span>", unsafe_allow_html=True)
                    
                    if analysis['entities']:
                        st.markdown("**Thực thể tìm thấy:**")
                        for ent, tag in analysis['entities']:
                            # Tạo màu sắc khác nhau dựa trên nhãn tag
                            color = "#e91e63" if "DISEASE" in tag or "B-" in tag else "#4caf50"
                            st.markdown(f"<span class='entity-chip' style='background-color:{color};'>{ent} ({tag})</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("*Không phát hiện thực thể y tế cụ thể.*")
                        
                with col2:
                    st.markdown("##### Tài liệu tra cứu (RAG)")
                    if analysis['docs']:
                        for idx, doc in enumerate(analysis['docs']):
                            source = os.path.basename(doc.get("source", "Tài liệu y tế"))
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>Tài liệu {idx+1} (Nguồn: {source})</strong><br/>
                                {doc.get("content")[:200]}...
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("*Không tìm thấy tài liệu liên quan.*")

# Nhận câu hỏi mới
query = None
if "suggested_query" in st.session_state and st.session_state.suggested_query:
    query = st.session_state.suggested_query
    st.session_state.suggested_query = None # Reset
else:
    query = st.chat_input("Nhập câu hỏi y tế của bạn tại đây (Ví dụ: bé_bị sốt_cao phải làm_sao)...")

if query:
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Xử lý phản hồi từ chatbot
    with st.chat_message("assistant"):
        with st.spinner("Đang phân tích câu hỏi và tra cứu y văn..."):
            
            # --- Bước 1: Phân tích NLU (Intent + NER) ---
            if nlu_model is not None:
                # Tiền xử lý từ ghép
                # Model predict mong đợi các từ ghép được tách từ bằng dấu gạch dưới "_"
                intent, entities = predict(query, nlu_model, tokenizer, encoder, device)
                entity_words = [word.replace('_', ' ') for word, tag in entities if tag != "O"]
                entities_filtered = [(word.replace('_', ' '), tag) for word, tag in entities if tag != "O"]
            else:
                intent = "Không rõ (Chưa load mô hình)"
                entity_words = []
                entities_filtered = []
            
            # --- Bước 2: RAG Retrieval ---
            retrieved_docs = []
            retrieved_docs_serializable = []
            if vector_db is not None:
                # Dùng phiên bản chữ thường để tìm kiếm khớp tốt hơn với SBERT
                search_query = query.strip().lower()
                retrieved_docs = vector_db.similarity_search(search_query, k=3)
                for doc in retrieved_docs:
                    retrieved_docs_serializable.append({
                        "source": doc.metadata.get("source", "Tài liệu y tế"),
                        "content": doc.page_content
                    })
            
            # --- Bước 3: Sinh câu trả lời với LLM ---
            if not api_key_input:
                answer = "Không tìm thấy API Key. Vui lòng nhập OpenAI / Azure Inference API Key tại bảng cấu hình ở Sidebar bên trái để kích hoạt trả lời từ mô hình ngôn ngữ lớn."
            else:
                openai_client = setup_openai(api_key_input, api_base_input)
                prompt = build_prompt(query, intent, entity_words, retrieved_docs)
                answer = generate_answer(openai_client, prompt)
            
            # Hiển thị câu trả lời
            st.write(answer)
            
            # Lưu phân tích vào session state để hiển thị lại
            analysis_info = {
                "intent": intent,
                "entities": entities_filtered,
                "docs": retrieved_docs_serializable
            }
            
            # Đính kèm phân tích hệ thống vào tin nhắn cuối cùng để kết xuất
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "analysis": analysis_info
            })
            
            st.rerun()

st.markdown("---")
st.caption("**Khuyến cáo y tế**: Thông tin cung cấp bởi chatbot chỉ mang tính chất tham khảo dựa trên tài liệu hướng dẫn y khoa chính thống có sẵn và không thay thế cho chẩn đoán, điều trị của bác sĩ chuyên khoa.")
