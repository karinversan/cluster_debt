FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.streamlit.txt /app/requirements.streamlit.txt
RUN pip install --no-cache-dir -r /app/requirements.streamlit.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
