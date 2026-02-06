#!/usr/bin/env python3
"""Migration script from ChromaDB to Milvus vector store.

Usage:
    python scripts/migrate_chroma_to_milvus.py [--chroma-dir ./data/chroma_db] [--milvus-host localhost] [--milvus-port 19530]
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from core.vectorstore.milvus import MilvusStore


async def migrate_chroma_to_milvus(
    chroma_dir: str,
    chroma_collection: str,
    milvus_host: str,
    milvus_port: int,
    milvus_collection: str,
    batch_size: int = 100,
) -> None:
    """Migrate all documents from ChromaDB to Milvus.

    Args:
        chroma_dir: ChromaDB persist directory
        chroma_collection: ChromaDB collection name
        milvus_host: Milvus server host
        milvus_port: Milvus server port
        milvus_collection: Milvus collection name
        batch_size: Number of documents to migrate per batch
    """
    print(f"Starting migration from ChromaDB to Milvus...")
    print(f"  ChromaDB: {chroma_dir} / {chroma_collection}")
    print(f"  Milvus: {milvus_host}:{milvus_port} / {milvus_collection}")

    # Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    try:
        chroma_coll = chroma_client.get_collection(chroma_collection)
    except Exception as e:
        print(f"Error: ChromaDB collection '{chroma_collection}' not found: {e}")
        sys.exit(1)

    # Get all documents from ChromaDB
    print("\nFetching all documents from ChromaDB...")
    try:
        all_data = chroma_coll.get(
            include=["documents", "embeddings", "metadatas"]
        )
    except Exception as e:
        print(f"Error fetching from ChromaDB: {e}")
        sys.exit(1)

    total_docs = len(all_data["ids"])
    print(f"Found {total_docs} documents in ChromaDB")

    if total_docs == 0:
        print("No documents to migrate. Exiting.")
        return

    # Create Milvus store
    print("\nInitializing Milvus store...")
    try:
        milvus_store = MilvusStore(
            host=milvus_host,
            port=milvus_port,
            collection_name=milvus_collection,
            dimension=settings.embedding_dimension,
        )
    except Exception as e:
        print(f"Error connecting to Milvus: {e}")
        print("Make sure Milvus server is running.")
        sys.exit(1)

    # Clear existing data in Milvus (optional, uncomment if needed)
    # print("Clearing existing Milvus collection...")
    # await milvus_store.clear()

    # Migrate in batches
    print(f"\nMigrating documents in batches of {batch_size}...")
    migrated_count = 0

    for i in range(0, total_docs, batch_size):
        batch_end = min(i + batch_size, total_docs)
        batch_ids = all_data["ids"][i:batch_end]
        batch_docs = all_data["documents"][i:batch_end]
        batch_embeddings = all_data["embeddings"][i:batch_end]
        batch_metadatas = all_data["metadatas"][i:batch_end] if all_data["metadatas"] else None

        try:
            await milvus_store.add(
                ids=batch_ids,
                texts=batch_docs,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
            )
            migrated_count += len(batch_ids)
            print(f"  Progress: {migrated_count}/{total_docs} documents migrated")
        except Exception as e:
            print(f"Error migrating batch {i//batch_size + 1}: {e}")
            print(f"  Failed at document index {i}")
            sys.exit(1)

    # Verify counts
    print("\nVerifying migration...")
    milvus_count = await milvus_store.count()
    print(f"  ChromaDB count: {total_docs}")
    print(f"  Milvus count: {milvus_count}")

    if milvus_count >= total_docs:
        print("\n✓ Migration completed successfully!")
        print(f"  Total documents migrated: {migrated_count}")
    else:
        print(f"\n✗ Migration incomplete: expected {total_docs}, got {milvus_count}")
        sys.exit(1)

    # Display collection info
    print("\nMilvus collection info:")
    info = milvus_store.get_collection_info()
    for key, value in info.items():
        print(f"  {key}: {value}")


def main():
    """Parse arguments and run migration."""
    parser = argparse.ArgumentParser(
        description="Migrate documents from ChromaDB to Milvus"
    )
    parser.add_argument(
        "--chroma-dir",
        default=settings.chroma_persist_dir,
        help=f"ChromaDB persist directory (default: {settings.chroma_persist_dir})",
    )
    parser.add_argument(
        "--chroma-collection",
        default=settings.chroma_collection_name,
        help=f"ChromaDB collection name (default: {settings.chroma_collection_name})",
    )
    parser.add_argument(
        "--milvus-host",
        default=settings.milvus_host,
        help=f"Milvus server host (default: {settings.milvus_host})",
    )
    parser.add_argument(
        "--milvus-port",
        type=int,
        default=settings.milvus_port,
        help=f"Milvus server port (default: {settings.milvus_port})",
    )
    parser.add_argument(
        "--milvus-collection",
        default=settings.chroma_collection_name,
        help=f"Milvus collection name (default: {settings.chroma_collection_name})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of documents per batch (default: 100)",
    )

    args = parser.parse_args()

    # Run migration
    asyncio.run(
        migrate_chroma_to_milvus(
            chroma_dir=args.chroma_dir,
            chroma_collection=args.chroma_collection,
            milvus_host=args.milvus_host,
            milvus_port=args.milvus_port,
            milvus_collection=args.milvus_collection,
            batch_size=args.batch_size,
        )
    )


if __name__ == "__main__":
    main()
