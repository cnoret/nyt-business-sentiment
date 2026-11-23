FROM apache/airflow:3.3.2-python3.12

ARG AIRFLOW_VERSION=3.3.2
# CPU-only PyTorch keeps the image small; Airflow is pinned so pip never upgrades it.
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
      "apache-airflow==${AIRFLOW_VERSION}" \
      prophet==1.5.0 \
      transformers==5.24.0 \
      mlflow-skinny==3.18.0

COPY --chown=airflow:root pyproject.toml README.md /opt/nyt/
COPY --chown=airflow:root src /opt/nyt/src
RUN pip install --no-cache-dir --no-deps /opt/nyt
RUN mkdir -p /opt/airflow/hf-cache
