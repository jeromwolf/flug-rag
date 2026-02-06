"""HWP document loader using pyhwp."""

import asyncio
import subprocess
import tempfile
from pathlib import Path

from .base import BaseLoader, LoadedDocument


class HWPLoader(BaseLoader):
    """Load HWP files. Tries pyhwp first, falls back to LibreOffice conversion."""

    supported_extensions = [".hwp"]

    async def load(self, file_path: str | Path) -> LoadedDocument:
        path = self._validate_file(file_path)
        return await asyncio.to_thread(self._load_sync, path)

    def _load_sync(self, path: Path) -> LoadedDocument:
        # Try pyhwp first
        try:
            return self._load_with_pyhwp(path)
        except Exception:
            pass

        # Fallback: LibreOffice CLI conversion (hwp → docx → text)
        try:
            return self._load_with_libreoffice(path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load HWP file: {path}. "
                f"Neither pyhwp nor LibreOffice could process it. Error: {e}"
            )

    def _load_with_pyhwp(self, path: Path) -> LoadedDocument:
        """Load HWP using pyhwp library."""
        import olefile

        if not olefile.isOleFile(str(path)):
            raise ValueError("Not a valid OLE file (HWP format)")

        ole = olefile.OleFileIO(str(path))

        # Extract text from HWP bodytext sections
        text_parts = []
        for stream_name in ole.listdir():
            stream_path = "/".join(stream_name)
            if "BodyText" in stream_path or "bodytext" in stream_path.lower():
                try:
                    data = ole.openstream(stream_name).read()
                    # HWP stores text in UTF-16 encoded sections
                    text = self._extract_text_from_bodytext(data)
                    if text.strip():
                        text_parts.append(text.strip())
                except Exception:
                    continue

        ole.close()

        content = "\n\n".join(text_parts)
        if not content.strip():
            raise ValueError("No text extracted from HWP via pyhwp")

        return LoadedDocument(
            content=content,
            metadata={
                "filename": path.name,
                "file_type": "hwp",
                "extraction_method": "pyhwp",
            },
        )

    def _extract_text_from_bodytext(self, data: bytes) -> str:
        """Extract readable text from HWP bodytext binary data."""
        import struct

        text_parts = []
        i = 0
        while i < len(data) - 1:
            # Try to decode as UTF-16LE characters
            try:
                char_code = struct.unpack_from("<H", data, i)[0]
                if 0x20 <= char_code < 0xFFFF and char_code not in (0xFEFF, 0xFFFE):
                    char = chr(char_code)
                    if char.isprintable() or char in ('\n', '\r', '\t'):
                        text_parts.append(char)
                    elif char_code < 0x20:
                        # Control characters - treat some as newlines
                        if char_code in (0x0A, 0x0D, 0x02):
                            text_parts.append('\n')
                i += 2
            except (struct.error, ValueError):
                i += 2

        return "".join(text_parts)

    def _load_with_libreoffice(self, path: Path) -> LoadedDocument:
        """Convert HWP to DOCX using LibreOffice, then extract text."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = subprocess.run(
                [
                    "libreoffice",
                    "--headless",
                    "--convert-to", "docx",
                    "--outdir", tmp_dir,
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                raise RuntimeError(f"LibreOffice conversion failed: {result.stderr}")

            # Find the converted DOCX
            docx_files = list(Path(tmp_dir).glob("*.docx"))
            if not docx_files:
                raise RuntimeError("LibreOffice conversion produced no output")

            # Use python-docx to extract text
            from docx import Document as DocxDocument

            doc = DocxDocument(str(docx_files[0]))
            text_parts = [para.text for para in doc.paragraphs if para.text.strip()]

            return LoadedDocument(
                content="\n\n".join(text_parts),
                metadata={
                    "filename": path.name,
                    "file_type": "hwp",
                    "extraction_method": "libreoffice",
                },
            )
