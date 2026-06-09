import os
import glob
import torch
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
import warnings

warnings.filterwarnings('ignore')

def load_documents(data_dir):
    documents = []
    
    md_files = glob.glob(os.path.join(data_dir, "*.md"))
    if md_files:
        print(f" -> Tìm thấy {len(md_files)} file Markdown (.md). Tiến hành nạp dữ liệu từ các file này...")
        for md_file in md_files:
            print(f"   => Đang đọc file Markdown: {os.path.basename(md_file)}...")
            try:
                loader = TextLoader(md_file, encoding='utf-8')
                documents.extend(loader.load())
            except Exception as e:
                print(f"Lỗi khi đọc file {md_file}: {e}")
        return documents
        
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    for pdf_file in pdf_files:
        print(f" -> Đang đọc file PDF: {os.path.basename(pdf_file)}...")
        try:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Lỗi khi đọc file {pdf_file}: {e}")

    word_files = glob.glob(os.path.join(data_dir, "*.doc")) + glob.glob(os.path.join(data_dir, "*.docx"))
    for word_file in word_files:
        print(f" -> Đang đọc file Word: {os.path.basename(word_file)}...")
        try:
            loader = UnstructuredWordDocumentLoader(word_file)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Lỗi khi đọc file {word_file}: {e}")
            
    return documents

def build_vector_db():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    data_dir = os.path.join(project_root, "data_RAG")
    persist_dir = os.path.join(project_root, "src", "rag", "chroma_db")
    
    if not os.path.exists(data_dir):
        print(f"Lỗi: Không tìm thấy thư mục {data_dir}")
        return

    # BƯỚC 1: Đọc tài liệu
    print("BƯỚC 1: Đang đọc tài liệu từ thư mục...")
    docs = load_documents(data_dir)
    
    if not docs:
        print("Không có tài liệu nào được đọc thành công. Vui lòng kiểm tra lại data_RAG.")
        return
        
    print(f"   => Tổng số trang/đoạn tài liệu gốc thu được: {len(docs)}")
    
    # BƯỚC 2: Chunking (Phân nhỏ tài liệu)
    print("\nBƯỚC 2: Phân mảnh tài liệu (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"   => Tổng số chunks tạo ra: {len(chunks)}")
    
    # BƯỚC 3: Tải Embedding Model
    print("\nBƯỚC 3: Tải Embedding Model (keepitreal/vietnamese-sbert)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   => Đang sử dụng thiết bị: {device.upper()}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="keepitreal/vietnamese-sbert",
        model_kwargs={'device': device}
    )
    
    # BƯỚC 4: Xây dựng Vector DB
    print("\nBƯỚC 4: Tính toán Vector và lưu vào ChromaDB (Có thể mất vài phút)...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print(f"\n THÀNH CÔNG! Vector Database đã được lưu thành công tại: {persist_dir}")
    print("Hệ thống RAG đã sẵn sàng cho bước truy xuất (Retrieval)!")

if __name__ == "__main__":
    build_vector_db()
