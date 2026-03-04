#!/usr/bin/env python3
"""Migration script from ChromaDB to Milvus vector store.

Usage:
    # Using URI (Milvus Lite or Standalone):
    python scripts/migrate_chroma_to_milvus.py --milvus-uri ./data/milvus.db
    python scripts/migrate_chroma_to_milvus.py --milvus-uri http://localhost:19530

    # Legacy host/port (backward compat, constructs http://host:port):
    python scripts/migrate_chroma_to_milvus.py --milvus-host localhost --milvus-port 19530

    # Full example:
    python scripts/migrate_chroma_to_milvus.py \\
        --chroma-dir ./data/chroma_db \\
        --milvus-uri ./data/milvus.db \\
        --milvus-collection knowledge_base
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
    milvus_uri: str,
    milvus_collection: str,
    batch_size: int = 100,
) -> None:
    """Migrate all documents from ChromaDB to Milvus.

    Args:
        chroma_dir: ChromaDB persist directory
        chroma_collection: ChromaDB collection name
        milvus_uri: Milvus URI (file path for Lite, http://host:port for Standalone)
        milvus_collection: Milvus collection name
        batch_size: Number of documents to migrate per batch
    """
    print("Starting migration from ChromaDB to Milvus...")
    print(f"  ChromaDB: {chroma_dir} / {chroma_collection}")
    print(f"  Milvus: {milvus_uri} / {milvus_collection}")

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

    # Create Milvus store with new URI-based constructor
    print("\nInitializing Milvus store...")
    try:
        milvus_store = MilvusStore(
            uri=milvus_uri,
            token=settings.milvus_store_token,
            collection_name=milvus_collection,
            dimension=settings.embedding_dimension,
            metric_type=settings.milvus_metric_type,
        )
    except Exception as e:
        print(f"Error connecting to Milvus: {e}")
        print("Make sure Milvus server is running (or the Lite file path is writable).")
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
        batch_metadatas = all_data["metadatas"][i:batch_end] if all_data["metadatas"] else [{}] * len(batch_ids)

        # Extract source_type per document for proper routing inside MilvusStore.add()
        # MilvusStore.add() handles source_type internally, but we pass full metadata so
        # it can extract it itself. No pre-extraction needed on the caller side.
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
            print(f"Error migrating batch {i // batch_size + 1}: {e}")
            print(f"  Failed at document index {i}")
            sys.exit(1)

    # Verify counts
    print("\nVerifying migration...")
    milvus_count = await milvus_store.count()
    print(f"  ChromaDB count: {total_docs}")
    print(f"  Milvus count: {milvus_count}")

    if milvus_count >= total_docs:
        print("\nMigration completed successfully!")
        print(f"  Total documents migrated: {migrated_count}")
    else:
        print(f"\nMigration incomplete: expected {total_docs}, got {milvus_count}")
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
    # New: URI-based connection (preferred)
    parser.add_argument(
        "--milvus-uri",
        default=None,
        help=(
            f"Milvus URI — file path for Lite (e.g. ./data/milvus.db) or "
            f"http://host:port for Standalone (default: {settings.milvus_store_uri})"
        ),
    )
    # Legacy: host/port for backward compatibility
    parser.add_argument(
        "--milvus-host",
        default=settings.milvus_host,
        help=f"Milvus server host — used only when --milvus-uri is not set (default: {settings.milvus_host})",
    )
    parser.add_argument(
        "--milvus-port",
        type=int,
        default=settings.milvus_port,
        help=f"Milvus server port — used only when --milvus-uri is not set (default: {settings.milvus_port})",
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

    # Resolve the effective Milvus URI:
    #   1. --milvus-uri provided  → use it directly
    #   2. --milvus-host/port provided but not --milvus-uri → construct http://host:port
    #   3. Neither provided → fall back to settings.milvus_store_uri
    if args.milvus_uri is not None:
        effective_uri = args.milvus_uri
    elif args.milvus_host != settings.milvus_host or args.milvus_port != settings.milvus_port:
        # User passed explicit host/port on the command line
        effective_uri = f"http://{args.milvus_host}:{args.milvus_port}"
    else:
        effective_uri = settings.milvus_store_uri

    asyncio.run(
        migrate_chroma_to_milvus(
            chroma_dir=args.chroma_dir,
            chroma_collection=args.chroma_collection,
            milvus_uri=effective_uri,
            milvus_collection=args.milvus_collection,
            batch_size=args.batch_size,
        )
    )


if __name__ == "__main__":
    main()
