"""Patch existing ChromaDB chunks with source_type metadata.

This script adds source_type metadata to all chunks in the vector store
based on their source file path. This enables metadata-based filtering
to solve the retrieval dilution problem.

Usage:
    cd backend
    python scripts/patch_source_type.py [--dry-run]
"""

import argparse
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings


def detect_source_type(source: str, filename: str = "") -> str:
    """Detect document source type from source path and filename."""
    # macOS APFS stores filenames in NFD (decomposed Unicode).
    # Normalize to NFC so Korean keyword matching works correctly.
    combined = unicodedata.normalize("NFC", f"{source} {filename}").lower()

    # Law detection - check both source and filename
    law_keywords = [
        "가스공사법", "시행령", "시행규칙", "법률",
        "가스안전관리법", "도시가스사업법", "액화석유가스법",
        "기업개요",
    ]
    for kw in law_keywords:
        if kw in combined:
            return "법률"

    # Articles of incorporation
    if "정관" in combined:
        return "정관"

    # Internal regulations
    rule_keywords = ["규정", "규칙", "지침", "내규", "요령", "기준"]
    for kw in rule_keywords:
        if kw in combined:
            return "내부규정"

    # Path-based detection (also NFC-normalized)
    nfc_raw = unicodedata.normalize("NFC", f"{source} {filename}")
    if "한국가스공사법" in nfc_raw:
        return "법률"
    if "내부규정" in nfc_raw:
        return "내부규정"

    return "기타"


def main():
    parser = argparse.ArgumentParser(description="Patch ChromaDB chunks with source_type metadata")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without applying")
    args = parser.parse_args()

    import chromadb

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = client.get_collection(name=settings.chroma_collection_name)

    # Get all chunks
    total = collection.count()
    print(f"Total chunks in collection: {total}")

    batch_size = 500
    stats = {"법률": 0, "내부규정": 0, "정관": 0, "기타": 0, "already_set": 0}
    patched = 0

    for offset in range(0, total, batch_size):
        results = collection.get(
            offset=offset,
            limit=batch_size,
            include=["metadatas"],
        )

        ids_to_update = []
        metadatas_to_update = []

        for doc_id, metadata in zip(results["ids"], results["metadatas"]):
            if metadata.get("source_type"):
                stats["already_set"] += 1
                continue

            source = metadata.get("source", "")
            filename = metadata.get("filename", "")
            source_type = detect_source_type(source, filename)

            new_metadata = dict(metadata)
            new_metadata["source_type"] = source_type

            ids_to_update.append(doc_id)
            metadatas_to_update.append(new_metadata)
            stats[source_type] += 1

        if ids_to_update and not args.dry_run:
            collection.update(
                ids=ids_to_update,
                metadatas=metadatas_to_update,
            )
            patched += len(ids_to_update)
        elif ids_to_update:
            patched += len(ids_to_update)

        print(f"  Processed {min(offset + batch_size, total)}/{total} chunks...", end="\r")

    print(f"\n\nResults:")
    print(f"  Already had source_type: {stats['already_set']}")
    print(f"  법률 (law): {stats['법률']}")
    print(f"  내부규정 (rules): {stats['내부규정']}")
    print(f"  정관 (articles): {stats['정관']}")
    print(f"  기타 (other): {stats['기타']}")
    print(f"  Total patched: {patched}")

    if args.dry_run:
        print("\n[DRY RUN] No changes applied. Remove --dry-run to apply.")
    else:
        print(f"\nDone! {patched} chunks patched with source_type metadata.")


if __name__ == "__main__":
    main()
