# ChatBot Y tế (Medical Chatbot) - PhoBERT + RAG + LLM

Dự án này xây dựng một **Trợ lý Y tế AI (Medical Chatbot)** kết hợp giữa mô hình hiểu ngôn ngữ tự nhiên (NLU) và kỹ thuật tạo văn bản tăng cường tra cứu (RAG - Retrieval-Augmented Generation). Chatbot có khả năng hiểu ý định người dùng, nhận diện thực thể y khoa, tra cứu tài liệu hướng dẫn y tế chính thống, và sinh câu trả lời chuẩn xác nhất nhờ LLM.

---

## Tính năng nổi bật

1. **NLU (Natural Language Understanding)**: Sử dụng mô hình **PhoBERT** (`vinai/phobert-base-v2`) được huấn luyện đồng thời (Joint Training) cho hai tác vụ:
   - **Intent Classification (Phân loại ý định)**: Xác định người dùng đang hỏi về phương pháp điều trị (`treatment`), triệu chứng/chẩn đoán (`method_diagnosis`), nguyên nhân (`cause`), hay mức độ nguy hiểm (`severity`).
   - **NER / Token Classification (Nhận diện thực thể)**: Trích xuất các thực thể y tế (tên bệnh, triệu chứng, thuốc...) theo định dạng nhãn BIO.
2. **RAG (Retrieval-Augmented Generation)**:
   - Tự động chuyển đổi tài liệu PDF/Word thành Markdown sạch để bảo toàn cấu trúc bảng biểu.
   - Sử dụng mô hình embedding tiếng Việt **Vietnamese-SBERT** (`keepitreal/vietnamese-sbert`) và lưu trữ vector vào **ChromaDB**.
3. **LLM Integration**: Tích hợp các mô hình ngôn ngữ lớn (như GPT-4o-mini qua API OpenAI/Azure Inference) để sinh câu trả lời tự nhiên, chuyên nghiệp và luôn kiểm soát câu trả lời nằm trong phạm vi tài liệu y tế được cung cấp.

---

## Cấu trúc thư mục

```text
ChatBot_Yte/
├── data/                       # Dữ liệu gán nhãn huấn luyện NLU (intent & NER)
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── data_RAG/                   # Tài liệu y khoa thô đầu vào (.pdf, .docx, .md)
│   ├── huong_dan_chan_doan_dieu_tri_tha.pdf
│   └── HƯỚNG DẪN CHẨN ĐOÁN VÀ ĐIỀU TRỊ MỘT SỐ BỆNH THƯỜNG GẶP Ở TRẺ EM.pdf
└── src/                        # Mã nguồn dự án
    ├── .env                    # Lưu API Key (Cần cấu hình)
    ├── checkpoints/            # Chứa checkpoint mô hình NLU tốt nhất (.pth)
    ├── NLU/                    # Module Hiểu ngôn ngữ tự nhiên
    │   ├── model_intent.py     # Định nghĩa mô hình JointPhoBERTModel
    │   ├── preprocess.py       # Tiền xử lý dữ liệu và tạo Dataset
    │   ├── train_intent.py     # Huấn luyện mô hình NLU
    │   └── predict.py          # Dự đoán ý định & thực thể từ câu hỏi thử nghiệm
    └── rag/                    # Module Tra cứu tài liệu & Sinh câu trả lời
        ├── convert_to_markdown.py # Chuyển PDF/Word sang Markdown
        ├── offline_rag.py      # Chunking, Embeddings & Xây dựng ChromaDB
        ├── main.py             # Pipeline chatbot chạy chính thức
        └── test_retrieval.py   # Kiểm thử tra cứu tài liệu từ ChromaDB
```

---

## Hướng dẫn cài đặt

### 1. Cài đặt các thư viện cần thiết

Yêu cầu Python >= 3.8. Cài đặt các thư viện từ file `requirements.txt`:

```bash
cd ChatBot_Yte
pip install -r requirements.txt
```


### 2. Cấu hình biến môi trường

Tạo hoặc chỉnh sửa file `src/.env` trong thư mục `src/` và cấu hình các  API Key của bạn:

```env
OPENAI_API_KEY = "your-openai-or-azure-inference-api-key"
OPENAI_API_BASE = "https://models.inference.ai.azure.com" # Hoặc endpoint OpenAI chính thức
```

---

## Hướng dẫn sử dụng

### Bước 1: Huấn luyện mô hình NLU (PhoBERT)

Để mô hình có thể nhận diện đúng ý định và các thực thể y khoa trong câu hỏi, tiến hành huấn luyện mô hình Joint PhoBERT:

```bash
cd src/NLU
python train_intent.py
```
*Sau khi huấn luyện thành công, trọng số mô hình tốt nhất sẽ được lưu tại `src/checkpoints/best_joint_model.pth`.*


---

### Bước 2: Xây dựng Cơ sở dữ liệu Vector (ChromaDB)

1. Đặt các tài liệu hướng dẫn y tế dạng PDF hoặc DOCX vào thư mục `data_RAG/`.
2. Chuyển đổi các tài liệu thô này thành Markdown để trích xuất sạch các bảng biểu:
   ```bash
   cd ../rag
   python convert_to_markdown.py
   ```
3. Tạo cơ sở dữ liệu vector ChromaDB từ các tài liệu đã được chuyển đổi:
   ```bash
   python offline_rag.py
   ```
   *Sau khi hoàn thành, thư mục cơ sở dữ liệu `src/rag/chroma_db` sẽ được tạo và sẵn sàng sử dụng.*

---

### Bước 3: Chạy Chatbot Y tế

Sau khi đã hoàn tất huấn luyện NLU và xây dựng Vector DB cho RAG, bạn có thể chạy Chatbot qua hai giao diện:

#### Phương án 1: Giao diện Web Streamlit (Khuyên dùng)
Giao diện Web đẹp mắt, hỗ trợ hiển thị chi tiết phân tích ý định, các thực thể y học trích xuất và các nguồn tài liệu tham khảo:
```bash
streamlit run app.py
```

#### Phương án 2: Giao diện Terminal console
```bash
cd ..
python rag/main.py
```
