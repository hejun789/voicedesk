FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
# answer_faq opens these by a path relative to the working directory, so they
# must sit at /app next to where the app runs.
COPY clinic_info.md clinic_info.zh.md ./

ENV PYTHONPATH=/app/src
ENV PORT=7860
# The public demo runs on 8b for its higher free-tier limits; override in the Space to raise quality.
ENV GROQ_MODEL=llama-3.1-8b-instant
EXPOSE 7860

CMD ["python", "-m", "voicedesk.voice"]
