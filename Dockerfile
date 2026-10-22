# Always-on host (Render, Railway, Fly.io, any VPS). Streamlit Community Cloud sleeps, which stops the background
# traders, the data keeper and the research scheduler; a container does not.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TZ=Asia/Kolkata
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Mount a persistent volume here (candle cache, option history, SQLite if DATABASE_URL is not set).
VOLUME /app/data_cache
EXPOSE 8501

# Hosts such as Render and Railway pass the port in $PORT.
CMD ["sh", "-c", "streamlit run quant_intelligence/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
