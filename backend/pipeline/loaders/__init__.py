"""Document loaders for multiple file formats."""

from .base import BaseLoader, LoadedDocument
from .pdf import PDFLoader
from .hwp import HWPLoader
from .docx import DocxLoader
from .excel import ExcelLoader
from .pptx import PPTXLoader
from .text import TextLoader

LOADER_MAP: dict[str, type[BaseLoader]] = {
    ".pdf": PDFLoader,
    ".hwp": HWPLoader,
    ".docx": DocxLoader,
    ".xlsx": ExcelLoader,
    ".xls": ExcelLoader,
    ".pptx": PPTXLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".csv": TextLoader,
}


def get_loader(file_extension: str) -> BaseLoader:
    """Get appropriate loader for file extension."""
    ext = file_extension.lower()
    if ext not in LOADER_MAP:
        raise ValueError(f"Unsupported file format: {ext}. Supported: {list(LOADER_MAP.keys())}")
    return LOADER_MAP[ext]()


__all__ = [
    "BaseLoader", "LoadedDocument", "get_loader",
    "PDFLoader", "HWPLoader", "DocxLoader", "ExcelLoader", "PPTXLoader", "TextLoader",
]
