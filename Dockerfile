# BASE IMAGE
FROM python:3.9-slim

# WORKING DIRECTORY
WORKDIR /app

# LOGS
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONBUFFERED=1

# DEPENDENCIES
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# OTHER CODES
COPY . .

# CMD
CMD ["python", "-m", "src.pipelines.training_pipeline"]