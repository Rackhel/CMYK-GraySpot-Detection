FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY gui ./gui

ENV PYTHONUNBUFFERED=1
EXPOSE 8501

# Default: training
# For GUI: docker run -p 8501:8501 grayspot streamlit run gui/app.py
CMD ["python", "src/scripts/train.py"]
