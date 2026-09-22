ARG version=3.12
FROM python:${version}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The run store is postgres rather than sqlite: sqlite holds one writing
# transaction per file, and parallel benchmark cells (`benchmark.py --workers`)
# run into "database is locked".
#
# psycopg2, not psycopg 3: the MLflow 2.x run store passes experiment_id as a
# string, psycopg 3 binds it as VARCHAR and postgres refuses to compare it with
# an integer column ("operator does not exist: integer = character varying").
# psycopg2 substitutes an untyped literal and the comparison goes through.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /mlflow
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3).read()"]

# The run store is chosen by MLFLOW_BACKEND_STORE_URI and its database is
# created at startup. Artifacts are served by the server itself, so a client
# only needs mlflow-skinny and HTTP - no shared volume with the experiments.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# The recorded benchmark grid of the released model, replayed by the `seed`
# service once the server is healthy, so a fresh server is not empty.
COPY seed_runs.py /mlflow/seed_runs.py
COPY seed /mlflow/seed

CMD ["/usr/local/bin/entrypoint.sh"]
