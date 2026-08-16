FROM python:3.11-slim

WORKDIR /code
COPY requirements.txt requirements.txt
RUN pip3 --no-cache-dir install -r requirements.txt

COPY models ./models
COPY shared ./shared
COPY frontend ./frontend
COPY app ./app

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
