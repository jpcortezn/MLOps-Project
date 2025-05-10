FROM python:3.10-slim

WORKDIR /app

COPY main.py .
COPY ../models ./models
COPY ../src ./src

RUN pip install --no-cache-dir fastapi uvicorn torch torchvision pillow

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
