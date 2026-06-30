import os
import warnings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_dir = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.join(current_dir, "src", "rag", "chroma_db")

embeddings = HuggingFaceEmbeddings(
    model_name="keepitreal/vietnamese-sbert",
    model_kwargs={'device': device}
)

vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

queries = [
    "nguyên nhân dẫn đến hôn mê",
    "co giật là gì"
]

for q in queries:
    print(f"\nQUERY: {q}")
    docs = vector_db.similarity_search(q, k=3)
    for i, doc in enumerate(docs):
        print(f"--- DOC {i+1} ---")
        print(doc.page_content[:300] + "...")
