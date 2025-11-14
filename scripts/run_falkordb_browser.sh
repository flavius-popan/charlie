#!/usr/bin/env bash

# Start the FalkorDB container that also bundles the Browser UI, preloading the
# Charlie demo database from data/graphiti-poc.db.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
DATA_DIR="$REPO_ROOT/data"
DEFAULT_DB="$DATA_DIR/graphiti-poc.db"

DB_FILE="${CHARLIE_FALKORDB_DB:-$DEFAULT_DB}"
CONTAINER_NAME="${FALKORDB_BROWSER_CONTAINER:-charlie-falkordb-browser}"
IMAGE="${FALKORDB_BROWSER_IMAGE:-falkordb/falkordb:latest}"
WEB_PORT="${FALKORDB_BROWSER_WEB_PORT:-3000}"
REDIS_PORT="${FALKORDB_BROWSER_REDIS_PORT:-6379}"
LOCAL_HOST_ALIAS="${FALKORDB_BROWSER_HOST_ALIAS:-host.docker.internal}"
HOST_GATEWAY_TARGET="${FALKORDB_BROWSER_HOST_GATEWAY_TARGET:-host-gateway}"
LOCAL_DEBUG_PORT="${FALKORDB_BROWSER_LOCAL_DEBUG_PORT:-6380}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but was not found in PATH." >&2
  exit 1
fi

if [ ! -f "$DB_FILE" ]; then
  echo "Expected FalkorDB RDB dump at '$DB_FILE' but it was not found." >&2
  exit 1
fi

echo "Starting FalkorDB Browser container '${CONTAINER_NAME}' from image '${IMAGE}'."
echo "Browser UI will be available at http://localhost:${WEB_PORT} once the container is ready."
echo "Redis/FalkorDB endpoint exposed on port ${REDIS_PORT}."
echo "Loading database dump from: ${DB_FILE}"
echo "Press Ctrl+C to stop the container (it will be removed automatically)."
if [ -n "$LOCAL_HOST_ALIAS" ] && [ -n "$LOCAL_DEBUG_PORT" ]; then
  echo
  echo "Tip: To connect this Browser to the local Charlie FalkorDB Lite instance,"
  echo "use host '${LOCAL_HOST_ALIAS}' and port ${LOCAL_DEBUG_PORT} in the login form."
  echo "  (The script maps ${LOCAL_HOST_ALIAS} to the host network via ${HOST_GATEWAY_TARGET}.)"
fi

docker_cmd=(
  docker run --rm
  --name "$CONTAINER_NAME"
  -p "${WEB_PORT}:3000"
  -p "${REDIS_PORT}:6379"
  -v "${DB_FILE}:/data/dump.rdb"
)

if [ -n "$LOCAL_HOST_ALIAS" ]; then
  docker_cmd+=(--add-host "${LOCAL_HOST_ALIAS}:${HOST_GATEWAY_TARGET}")
fi

docker_cmd+=("$IMAGE")

exec "${docker_cmd[@]}"
