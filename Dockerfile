FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY chatbot ./chatbot
COPY api.py chat.py ./
COPY web ./web

ENV CHATBOT_DATA_DIR=/app/data
VOLUME ["/app/data"]
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
