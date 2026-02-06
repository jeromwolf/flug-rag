"""Plain text document loader."""

import asyncio
from pathlib import Path

from .base import BaseLoader, LoadedDocument


class TextLoader(BaseLoader):
    """Load plain text, markdown, and CSV files."""

    supported_extensions = [".txt", ".md", ".csv"]

    async def load(self, file_path: str | Path) -> LoadedDocument:
        path = self._validate_file(file_path)
        return await asyncio.to_thread(self._load_sync, path)

    def _load_sync(self, path: Path) -> LoadedDocument:
        # Try common encodings
        content = None
        encoding_used = None
        for encoding in ["utf-8", "cp949", "euc-kr", "latin-1"]:
            try:
                content = path.read_text(encoding=encoding)
                encoding_used = encoding
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if content is None:
            raise RuntimeError(f"Could not decode file: {path}")

        return LoadedDocument(
            content=content.strip(),
            metadata={
                "filename": path.name,
                "file_type": path.suffix.lstrip("."),
                "encoding": encoding_used,
                "char_count": len(content),
            },
        )
