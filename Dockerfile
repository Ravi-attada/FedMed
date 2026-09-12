FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model/ ./model/
COPY train/ ./train/
COPY data/ ./data/
COPY eval/ ./eval/
COPY server/ ./server/
COPY client/ ./client/
COPY demo/ ./demo/
COPY utils/ ./utils/
RUN mkdir -p logs
ENV PYTHONPATH=/app
CMD ["python", "demo/week1_demo.py"]
