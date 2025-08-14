FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Upgrade pip & set correct index
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

CMD ["streamlit", "run", "app.py"]
