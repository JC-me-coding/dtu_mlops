FROM python:3.9-slim

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn

COPY simple_app.py simple_app.py

CMD exec uvicorn simple_app:app --port $PORT --host 0.0.0.0 --workers 1