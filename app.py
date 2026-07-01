import os
import sys
import torch
import streamlit as st
from transformers import AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'src'))
sys.path.append(os.path.join(ROOT_DIR, 'src', 'NLU'))
sys.path.append(os.path.join(ROOT_DIR, 'src', 'rag'))

from rag.predict import predict, load_encoder
from model_intent import JointPhoBERTModel
from main import build_prompt, generate_answer, setup_openai, setup_hf_client, load_env_vars
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


st.set_page_config(
    page_title="Trợ lý Y tế AI - Chatbot RAG",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Đọc API Key mặc định từ hệ thống hoặc .env
env_vars = load_env_vars(os.path.join(ROOT_DIR, "src", ".env"))
api_key_input = os.environ.get("OPENAI_API_KEY", env_vars.get("OPENAI_API_KEY", ""))
api_base_input = os.environ.get("OPENAI_API_BASE", env_vars.get("OPENAI_API_BASE", "https://models.inference.ai.azure.com"))
hf_token_input = os.environ.get("HF_TOKEN", env_vars.get("HF_TOKEN", ""))

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/809/809957.png", width=80)
    
    st.markdown("### Câu hỏi gợi ý")
    
    sample_queries = [
        "Bé bị sốt cao co giật phải làm sao?",
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
    query = st.chat_input("Nhập câu hỏi y tế của bạn tại đây...")

if query:
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("assistant"):
        with st.spinner("Đang phân tích câu hỏi và tra cứu y văn..."):
            
            # --- Bước 1: Phân tích NLU (Intent + NER) ---
            if nlu_model is not None:
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
                # Dùng phiên bản chữ thường và tăng cường (Query Expansion)
                search_query = query.strip().lower()
                if "là gì" in search_query or "thế nào là" in search_query:
                    search_query += " đại cương định nghĩa khái niệm"
                
                retrieved_docs = vector_db.similarity_search(search_query, k=5)
                for doc in retrieved_docs:
                    retrieved_docs_serializable.append({
                        "source": doc.metadata.get("source", "Tài liệu y tế"),
                        "content": doc.page_content
                    })
            
            # --- Bước 3: Sinh câu trả lời với LLM ---
            if not api_key_input and not hf_token_input:
                answer = "Không tìm thấy API Key nào. Vui lòng thiết lập biến môi trường OPENAI_API_KEY hoặc HF_TOKEN để kích hoạt trả lời từ AI."
            else:
                openai_client = setup_openai(api_key_input, api_base_input) if api_key_input else None
                hf_client = setup_hf_client(hf_token_input) if hf_token_input else None
                prompt = build_prompt(query, intent, entity_words, retrieved_docs)
                answer = generate_answer(openai_client, hf_client, prompt)
            
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
