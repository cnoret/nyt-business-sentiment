# Image used by CI (Jenkins) to run the test suite and by the batch job.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements-dev.txt pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements-dev.txt && pip install --no-cache-dir --no-deps -e .

COPY tests ./tests
COPY streamlit ./streamlit
COPY airflow ./airflow

CMD ["python", "-m", "pytest"]
