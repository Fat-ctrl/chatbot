FROM python:3.11-slim

WORKDIR /app

COPY ./src /app/src
COPY requirements.txt /app/

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8080

CMD ["python", "src/gradio_app.py"]