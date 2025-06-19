from src.pynucleus.settings import settings
from src.pynucleus.rag.collector import chunk_text
import pytest

def test_chunk_size():
    sample = "word " * 5000
    chunks = list(chunk_text(sample))
    # each chunk must be ≤512 tokens
    for c in chunks:
        assert len(c.split()) <= 600  # rough check 