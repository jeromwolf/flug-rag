# Milvus Vector Store Setup Guide

This guide explains how to use Milvus as the vector store for flux-rag in production environments.

## Prerequisites

1. **Install Milvus Server**

   Using Docker (recommended):
   ```bash
   docker run -d --name milvus-standalone \
     -p 19530:19530 \
     -p 9091:9091 \
     -v $(pwd)/data/milvus:/var/lib/milvus \
     milvusdb/milvus:latest
   ```

   Or follow the [official Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md).

2. **Install Python Dependencies**

   ```bash
   pip install pymilvus
   ```

## Configuration

Add the following environment variables to your `.env` file:

```env
# Vector DB - Milvus (production)
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_INDEX_TYPE=IVF_FLAT
MILVUS_METRIC_TYPE=COSINE
```

## Usage

### Using Milvus in Code

```python
from core.vectorstore import create_vectorstore

# Create Milvus store
vectorstore = create_vectorstore(
    store_type="milvus",
    collection_name="knowledge_base",
    host="localhost",
    port=19530,
)

# Add documents
await vectorstore.add(
    ids=["doc1", "doc2"],
    texts=["Document 1 content", "Document 2 content"],
    embeddings=[[0.1] * 1024, [0.2] * 1024],
    metadatas=[{"source": "file1.pdf"}, {"source": "file2.pdf"}],
)

# Search
results = await vectorstore.search(
    query_embedding=[0.15] * 1024,
    top_k=5,
)

for result in results:
    print(f"ID: {result.id}, Score: {result.score}")
    print(f"Content: {result.content}")
    print(f"Metadata: {result.metadata}")
```

### Migrating from ChromaDB to Milvus

If you have existing data in ChromaDB, use the migration script:

```bash
# Basic migration (uses default settings)
python scripts/migrate_chroma_to_milvus.py

# With custom parameters
python scripts/migrate_chroma_to_milvus.py \
  --chroma-dir ./data/chroma_db \
  --chroma-collection knowledge_base \
  --milvus-host localhost \
  --milvus-port 19530 \
  --milvus-collection knowledge_base \
  --batch-size 100
```

The script will:
1. Connect to both ChromaDB and Milvus
2. Fetch all documents from ChromaDB
3. Migrate them in batches to Milvus
4. Verify the migration was successful
5. Display a summary report

## Features

### Supported Operations

- ✅ `add()` - Add documents with embeddings
- ✅ `search()` - Vector similarity search with COSINE metric
- ✅ `delete()` - Delete documents by IDs
- ✅ `get()` - Retrieve documents by IDs
- ✅ `count()` - Get total document count
- ✅ `clear()` - Clear all documents in collection

### Metadata Support

Metadata is stored as JSON strings in Milvus. Complex nested metadata structures are supported:

```python
metadata = {
    "source": "document.pdf",
    "page": 42,
    "tags": ["important", "review"],
    "nested": {"key": "value"}
}
```

### Index Types

Supported index types (configurable via `MILVUS_INDEX_TYPE`):
- `IVF_FLAT` (default) - Good balance of speed and accuracy
- `IVF_SQ8` - Memory-efficient variant
- `HNSW` - Higher accuracy, more memory
- `FLAT` - Brute force, best for small datasets

### Metric Types

Supported distance metrics (configurable via `MILVUS_METRIC_TYPE`):
- `COSINE` (default) - Cosine similarity (range: 0-1)
- `L2` - Euclidean distance
- `IP` - Inner product

## Performance Tuning

### For Large Datasets (>1M vectors)

```env
MILVUS_INDEX_TYPE=IVF_SQ8
MILVUS_METRIC_TYPE=COSINE
```

### For High Accuracy

```env
MILVUS_INDEX_TYPE=HNSW
MILVUS_METRIC_TYPE=COSINE
```

### Batch Size

When adding many documents, use batching:

```python
batch_size = 100
for i in range(0, len(ids), batch_size):
    batch_ids = ids[i:i+batch_size]
    batch_texts = texts[i:i+batch_size]
    batch_embeddings = embeddings[i:i+batch_size]
    batch_metadatas = metadatas[i:i+batch_size]

    await vectorstore.add(
        ids=batch_ids,
        texts=batch_texts,
        embeddings=batch_embeddings,
        metadatas=batch_metadatas,
    )
```

## Comparison: ChromaDB vs Milvus

| Feature | ChromaDB | Milvus |
|---------|----------|--------|
| **Deployment** | Embedded | Server-based |
| **Scalability** | Small-Medium | Large-scale |
| **Performance** | Good for <100k | Optimized for millions |
| **Use Case** | Development, Testing | Production |
| **Setup** | Simple (no server) | Requires server |
| **Clustering** | No | Yes |
| **High Availability** | No | Yes |

## Troubleshooting

### Connection Failed

```
Error connecting to Milvus: ...
```

**Solution:** Ensure Milvus server is running:
```bash
docker ps | grep milvus
```

If not running, start it:
```bash
docker start milvus-standalone
```

### Collection Already Exists

If you need to recreate a collection:

```python
await vectorstore.clear()  # This drops and recreates the collection
```

### Dimension Mismatch

Ensure `EMBEDDING_DIMENSION` matches your embedding model:
- BGE-M3: 1024
- BGE-Large: 1024
- OpenAI Ada-002: 1536

## Resources

- [Milvus Documentation](https://milvus.io/docs)
- [PyMilvus API Reference](https://milvus.io/api-reference/pymilvus/v2.3.x/About.md)
- [Milvus Performance Tuning](https://milvus.io/docs/performance_faq.md)
