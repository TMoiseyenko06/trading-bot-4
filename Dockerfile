FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# State persists via mounted volume
VOLUME /app/live_paper_logs

ENTRYPOINT ["python", "run_live_paper.py"]
CMD ["--instruments", "NQ", "ES", "RTY", "YM", "--signal-only", "YM"]
