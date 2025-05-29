FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY mm_engine.py .
COPY requirements.txt .

RUN pip3 install --upgrade pip \
    && pip3 install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "mm_engine:app", "--host", "0.0.0.0", "--port", "8000"]
