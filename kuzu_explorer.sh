docker run -p 8000:8000 \
           -v ./brain:/database \
           -e KUZU_FILE=charlie.kuzu \
           --rm kuzudb/explorer:latest
