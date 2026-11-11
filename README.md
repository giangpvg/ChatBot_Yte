---
title: ChatBot Yte
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# ChatBot Y tế (Medical Chatbot) - PhoBERT + Hybrid RAG + FastAPI

Dự án này xây dựng một **Trợ lý Y tế AI (Medical Chatbot)** kết hợp giữa mô hình hiểu ngôn ngữ tự nhiên (NLU), kỹ thuật tạo văn bản tăng cường tra cứu (RAG) và **FastAPI**. Chatbot cung cấp cả **Giao diện Web tương tác trực quan** lẫn **Hệ thống RESTful API chuẩn mực** phục vụ đa nền tảng.

## Tính năng nổi bật

1. **NLU (Natural Language Understanding)**: Sử dụng mô hình **PhoBERT** (`vinai/phobert-base-v2`) huấn luyện đồng thời (Joint Training) cho hai tác vụ:
   - **Intent Classification (Phân loại ý định)**: `treatment` (điều trị / dùng thuốc), `method_diagnosis` (triệu chứng / chẩn đoán), `cause` (nguyên nhân), `severity` (mức độ nguy hiểm).
   - **NER / Token Classification (Nhận diện thực thể)**: Trích xuất thực thể y tế (tên bệnh, triệu chứng, thủ thuật y khoa...) theo chuẩn nhãn BIO.
2. **Hybrid RAG (Retrieval-Augmented Generation)**:
   - Tự động chuyển đổi tài liệu PDF/Word sang Markdown bảo toàn bảng biểu.
   - Kết hợp truy vấn ngữ nghĩa (Dense Retrieval) qua **Vietnamese-SBERT** (`keepitreal/vietnamese-sbert`) lưu trữ trên **ChromaDB** và truy vấn từ khóa (Sparse Retrieval) qua **BM25**.
   - Xếp hạng lại tài liệu với thuật toán **Reciprocal Rank Fusion (RRF)**.
3. **FastAPI Backend & Built-in Web UI**:
   - **Giao diện Web tích hợp (Single-Service)**: Đặt tại thư mục `frontend/` (React + TypeScript + Vite). Sau `npm run build`, FastAPI phục vụ `frontend/dist` tại root URL `/`.
   - **RESTful API**: Endpoints chuẩn hóa với Pydantic Schemas, tự động sinh tài liệu Swagger UI tại `/docs`.

---

## Cấu trúc thư mục (Flat Layout)

```text
ChatBot_Yte/
├── data/                            # TOÀN BỘ DỮ LIỆU ĐẦU VÀO
│   ├── nlu/                         # Dữ liệu gán nhãn NLU (train.json, dev.json, test.json...)
│   └── docs/                        # Tài liệu y khoa thô (.pdf, .docx, .md)
│
├── storage/                         # TOÀN BỘ DỮ LIỆU SINH RA (Gitignore)
│   ├── checkpoints/                 # Trọng số PhoBERT NLU (best_joint_model.pth)
│   └── vector_db/                   # Chroma Vector DB & bm25_retriever.pkl
│
├── frontend/                        # GIAO DIỆN WEB (Vite + React + TypeScript)
│   ├── package.json, vite.config.ts
│   ├── index.html
│   └── src/
│       ├── api/                     # Kiểu dữ liệu & hàm gọi /api/v1/chat
│       ├── hooks/useChat.ts         # State & logic hội thoại
│       ├── components/              # Sidebar, MessageList, MetaPanel, ChatInput
│       └── styles/style.css
│
├── medical_chatbot/                 # PACKAGE MÃ NGUỒN PYTHON DUY NHẤT (Ở ROOT)
│   ├── __init__.py
│   ├── config.py                    # Cấu hình tập trung (đường dẫn, device, keys)
│   ├── cli.py                       # Giao diện dòng lệnh Terminal
│   │
│   ├── api/                         # Phân hệ FastAPI
│   │   ├── __init__.py
│   │   ├── main.py                  # Tạo FastAPI app & mount static frontend
│   │   ├── schemas.py               # Pydantic Schemas (ChatRequest, ChatResponseSchema)
│   │   └── routes/                  # API Endpoints
│   │       ├── chat.py              # POST /api/v1/chat
│   │       └── health.py            # GET /api/v1/health
│   │
│   ├── nlu/                         # Phân hệ NLU (PhoBERT)
│   │   ├── model.py                 # Class JointPhoBERTModel
│   │   ├── dataset.py               # Dataset & LabelEncoder
│   │   ├── predictor.py             # Dự đoán Intent & NER (NLUPredictor)
│   │   └── trainer.py               # Vòng lặp huấn luyện NLU
│   │
│   ├── rag/                         # Phân hệ RAG (Chroma + BM25)
│   │   ├── document_converter.py    # Chuyển đổi PDF/DOCX sang Markdown
│   │   ├── tokenizer.py             # Tách từ tiếng Việt cho BM25
│   │   ├── indexer.py               # Chunking và tạo ChromaDB/BM25
│   │   └── retriever.py             # HybridRetriever & RRF
│   │
│   └── services/                    # Tầng nghiệp vụ điều phối
│       ├── llm_client.py            # Quản lý gọi LLM (OpenAI / Hugging Face fallback)
│       ├── prompt_builder.py        # Xây dựng Prompt y tế có ràng buộc
│       └── chatbot_service.py       # ChatBotService điều phối trọn gói
│
├── tests/                           # BỘ KIỂM THỬ TỰ ĐỘNG
│   ├── __init__.py
│   ├── test_api.py                  # Kiểm thử FastAPI (Health & Index)
│   ├── test_nlu.py                  # Kiểm thử phân loại Intent & NER
│   └── test_retrieval.py            # Kiểm thử tra cứu tài liệu Hybrid
│
├── app.py                           # Điểm khởi chạy Web Server chính
├── .env.example                     # Mẫu cấu hình biến môi trường
├── .env                             # File API Keys (đã gitignore)
├── Dockerfile                       # Cấu hình Docker container
├── requirements.txt                 # Danh sách thư viện phụ thuộc
├── setup.py                         # Cấu hình cài đặt package
└── README.md
```

---

## Hướng dẫn cài đặt

### 1. Cài đặt các thư viện cần thiết

Yêu cầu Python >= 3.10. Dùng môi trường riêng để tránh xung đột với thư viện của dự án khác:

```bash
cd ChatBot_Yte
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Trên Windows, kích hoạt môi trường bằng `.venv\Scripts\activate`.
Với máy chỉ dùng CPU, cài PyTorch trước bước cài `requirements.txt`:

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
```

Bộ thư viện AI được cố định ở `torch==2.9.1`, `transformers==4.57.6` và
`sentence-transformers==5.2.0`. Dự án xử lý văn bản, không cần `torchvision`.
Nếu gặp lỗi `torchvision::nms`, tạo lại môi trường riêng theo hướng dẫn trên.

### 2. Cấu hình biến môi trường

Tạo file `.env` tại thư mục gốc từ file mẫu `.env.example`:

```bash
cp .env.example .env
```

Cấu hình các API Key của bạn trong `.env`:

```env
# 1. Dùng OpenAI API chính thức (Khuyên dùng)
OPENAI_API_KEY="your-openai-api-key"
OPENAI_API_BASE="https://api.openai.com/v1"
MODEL_NAME="gpt-4o-mini"

# 2. Token Hugging Face dự phòng (tùy chọn)
HF_TOKEN="your-huggingface-token"
```

---

## Hướng dẫn sử dụng

### 1. Khởi chạy Ứng dụng Web & API

```bash
python app.py
```
*(Nếu cổng `8000` bận, hệ thống sẽ tự động chuyển sang cổng trống tiếp theo như `8001`)*

- **Giao diện Web Chatbot**: Mở trình duyệt và truy cập: `http://localhost:8000` (hoặc `http://localhost:8001`)
- **Tài liệu Swagger UI tương tác**: Truy cập: `http://localhost:8000/docs`

### 2. Sử dụng REST API

#### Gửi câu hỏi y tế (`POST /api/v1/chat`):
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "Bé bị sốt cao co giật phải làm sao?", "top_n": 5}'
```

#### Kiểm tra tình trạng hệ thống (`GET /api/v1/health`):
```bash
curl "http://localhost:8000/api/v1/health"
```

---

### 3. Các lệnh phụ trợ (CLI & Huấn luyện)

- **Chạy giao diện dòng lệnh (CLI)**:
  ```bash
  python -m medical_chatbot.cli
  ```
- **Huấn luyện lại mô hình NLU**:
  ```bash
  python -m medical_chatbot.nlu.trainer
  ```
- **Tạo lại Vector Database (khi có tài liệu mới)**:
  ```bash
  python -m medical_chatbot.rag.document_converter
  python -m medical_chatbot.rag.indexer
  ```
- **Chạy toàn bộ kiểm thử tự động**:
  ```bash
  pip install -e ".[test]"
  python -m pytest -q tests
  ```

  Các test tích hợp cần frontend đã build, checkpoint trong `storage/checkpoints/`
  và Chroma/BM25 trong `storage/vector_db/`. Model nền được tải ở lần chạy đầu;
  nếu đã có cache, có thể chạy test với `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.


## Phát triển frontend

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173 (proxy /api -> FastAPI :8000)
npm run build    # build ra frontend/dist để FastAPI phục vụ
```
