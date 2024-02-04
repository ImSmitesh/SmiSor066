FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./Model/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./Model /app