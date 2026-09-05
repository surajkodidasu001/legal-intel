FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
RUN pip install --no-cache-dir -e . || pip install --no-cache-dir ./src

ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "legalintel.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
