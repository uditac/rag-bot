"""Document loading and chunking. No heavy dependencies, so it is easy to test."""
from pathlib import Path


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into fixed-size, overlapping chunks."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    text = text.strip()
    if not text:
        return []

    step = size - overlap
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
    return chunks


def load_documents(data_dir: str) -> list[tuple[str, str]]:
    """Return (filename, text) for every .txt and .md file in data_dir."""
    folder = Path(data_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")
    docs = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in {".txt", ".md"}:
            docs.append((path.name, path.read_text(encoding="utf-8")))
    if not docs:
        raise ValueError(f"No .txt or .md files found in {data_dir}")
    return docs
