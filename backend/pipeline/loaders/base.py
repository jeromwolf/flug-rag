"""Base document loader interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LoadedDocument:
    """Represents a loaded document with extracted text and metadata."""
    content: str
    metadata: dict = field(default_factory=dict)
    pages: list[dict] = field(default_factory=list)  # [{page_num, content}]

    @property
    def total_chars(self) -> int:
        return len(self.content)

    @property
    def page_count(self) -> int:
        return len(self.pages) if self.pages else 1


class BaseLoader(ABC):
    """Abstract base for all document loaders."""

    supported_extensions: list[str] = []

    @abstractmethod
    async def load(self, file_path: str | Path) -> LoadedDocument:
        """Load and extract text from a file."""
        ...

    def _validate_file(self, file_path: str | Path) -> Path:
        """Validate file exists and has correct extension."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if self.supported_extensions and path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported extension {path.suffix} for {self.__class__.__name__}")
        return path
