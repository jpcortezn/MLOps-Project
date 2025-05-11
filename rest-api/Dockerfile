FROM python:3.10-slim

WORKDIR /app

COPY rest-api/main.py .
COPY rest-api/requirements.txt .
COPY models ./models
COPY src ./src

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

