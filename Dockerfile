FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501 9092
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501"]