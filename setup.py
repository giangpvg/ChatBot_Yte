from setuptools import setup, find_packages

setup(
    name="medical_chatbot",
    version="1.0.0",
    description="Medical Chatbot Assistant combining PhoBERT NLU, Hybrid RAG, and LLM with FastAPI",
    packages=find_packages(include=["medical_chatbot", "medical_chatbot.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch==2.9.1",
        "transformers==4.57.6",
        "scikit-learn",
        "langchain",
        "langchain-community",
        "langchain-huggingface",
        "langchain-chroma",
        "sentence-transformers==5.2.0",
        "pdfplumber",
        "python-docx",
        "pypdf",
        "openai",
        "python-dotenv",
        "fastapi",
        "uvicorn",
        "pydantic",
        "underthesea",
        "rank_bm25",
    ],
    extras_require={"test": ["pytest", "httpx"]},
    entry_points={
        "console_scripts": [
            "medical-chatbot-cli=medical_chatbot.cli:main",
            "medical-chatbot-api=medical_chatbot.api.main:app",
        ],
    },
)
