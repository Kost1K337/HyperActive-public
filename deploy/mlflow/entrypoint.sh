#!/bin/sh
# Start the tracking server.
#
# The run store is chosen by MLFLOW_BACKEND_STORE_URI. The default is postgres:
# sqlite holds one writing transaction per file, and parallel benchmark cells
# run into "database is locked".
set -eu

BACKEND="${MLFLOW_BACKEND_STORE_URI:-sqlite:////mlflow/store/mlflow.db}"
ARTIFACTS="${MLFLOW_ARTIFACTS_DESTINATION:-/mlflow/artifacts}"

case "$BACKEND" in
  postgresql*)
    # The tracking database is created separately from any application
    # database: they may live in one postgres instance, but their schemas are
    # migrated independently.
    echo "tracking: waiting for postgres, creating the database if needed"
    python - "$BACKEND" <<'PY'
import re
import sys
import time
import urllib.parse

import psycopg2

uri = sys.argv[1]
parsed = urllib.parse.urlparse(re.sub(r"^postgresql\+\w+://", "postgresql://", uri))
database = (parsed.path or "/mlflow").lstrip("/") or "mlflow"
# Connect to the maintenance database: the target one may not exist yet.
admin = parsed._replace(path="/postgres").geturl()

deadline = time.time() + 120
last = None
while time.time() < deadline:
    try:
        conn = psycopg2.connect(admin, connect_timeout=5)
        try:
            # CREATE DATABASE does not work inside a transaction.
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
                if cur.fetchone() is None:
                    # The name comes from an environment variable, not from a
                    # user; it is quoted so that its case is preserved.
                    cur.execute(f'CREATE DATABASE "{database}"')
                    print(f"created database {database}")
                else:
                    print(f"database {database} already exists")
        finally:
            conn.close()
        sys.exit(0)
    except Exception as exc:
        last = exc
        time.sleep(3)
print(f"postgres is unavailable: {last}", file=sys.stderr)
sys.exit(1)
PY
    ;;
  *)
    echo "tracking: store $BACKEND"
    ;;
esac

exec mlflow server \
  --host 0.0.0.0 --port 5000 \
  --backend-store-uri "$BACKEND" \
  --artifacts-destination "$ARTIFACTS" \
  --serve-artifacts
