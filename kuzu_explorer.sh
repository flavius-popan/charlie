#!/bin/bash

# Default to charlie.kuzu if no argument provided
KUZU_FILE=${1:-charlie.kuzu}

docker run -p 8000:8000 \
           -v ./brain:/database \
           -e KUZU_FILE="$KUZU_FILE" \
           --rm kuzudb/explorer:latest
