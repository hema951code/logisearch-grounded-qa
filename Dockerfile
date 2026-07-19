FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache-buster: bump this value any time you push code changes and the
# platform's build cache is reusing an old layer instead of picking up the
# new files. Changing this line forces everything below it to rebuild fresh.
ARG CACHEBUST=2

COPY . .
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
