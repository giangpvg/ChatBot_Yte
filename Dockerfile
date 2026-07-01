FROM python:3.10-slim

WORKDIR /app

# Cài đặt các thư viện hệ thống cơ bản
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt các thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . .

# Hugging Face Spaces mặc định sử dụng port 7860
EXPOSE 7860

# Chạy ứng dụng Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
