FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    RFM_DATA_DIR=/var/data \
    RFM_WATCHLIST=/var/data/watchlist.yaml

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# PORT is provided by the host (Render sets it); defaults to 8000 locally.
CMD ["python", "-m", "rfmonitor", "run"]
