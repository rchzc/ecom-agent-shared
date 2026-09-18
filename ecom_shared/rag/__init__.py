"""RAG 层：切分 / 向量化 / 多后端检索 / 混合重排。"""
from .backends import (
    ChromaBackend,
    LexicalBackend,
    NumpyBackend,
    RetrievedChunk,
    VectorBackend,
    build_backend,
)
from .chunking import Chunk, chunk_corpus, chunk_document, load_documents
from .embedder import Embedder
from .rerank import DEFAULT_ALPHA, lexical_score, rerank, tokenize
from .service import IngestReport, RagService

__all__ = [
    "RagService",
    "IngestReport",
    "Chunk",
    "RetrievedChunk",
    "VectorBackend",
    "ChromaBackend",
    "NumpyBackend",
    "LexicalBackend",
    "build_backend",
    "Embedder",
    "chunk_document",
    "chunk_corpus",
    "load_documents",
    "rerank",
    "lexical_score",
    "tokenize",
    "DEFAULT_ALPHA",
]
