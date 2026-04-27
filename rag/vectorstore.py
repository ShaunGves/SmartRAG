"""
rag/vectorstore.py

Handles document ingestion, chunking, embedding, and retrieval
using ChromaDB as the vector store.
"""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load the sentence-transformer embedding model."""
    log.info(f"Loading embedding model: {cfg.model.embedding_model_id}")
    return HuggingFaceEmbeddings(
        model_name=cfg.model.embedding_model_id,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # Cosine similarity ready
    )


def load_documents(source: str) -> List[Document]:
    """
    Load documents from various sources.

    Args:
        source: Path to directory, PDF file, text file, or URL

    Returns:
        List of LangChain Document objects
    """
    source_path = Path(source)

    if source_path.is_dir():
        log.info(f"Loading documents from directory: {source}")
        loader = DirectoryLoader(
            str(source_path),
            glob="**/*.{txt,pdf,md}",
            loader_cls=TextLoader,
            show_progress=True,
        )
    elif source_path.suffix == ".pdf":
        log.info(f"Loading PDF: {source}")
        loader = PyPDFLoader(str(source_path))
    elif source_path.suffix in [".txt", ".md"]:
        log.info(f"Loading text file: {source}")
        loader = TextLoader(str(source_path))
    elif source.startswith("http"):
        log.info(f"Loading URL: {source}")
        loader = WebBaseLoader(source)
    else:
        raise ValueError(f"Unsupported source: {source}")

    docs = loader.load()
    log.info(f"Loaded {len(docs)} documents")
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.rag.chunk_size,
        chunk_overlap=cfg.rag.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    log.info(f"Split into {len(chunks)} chunks (size={cfg.rag.chunk_size}, overlap={cfg.rag.chunk_overlap})")
    return chunks


def build_vectorstore(
    sources: Optional[List[str]] = None,
    docs: Optional[List[Document]] = None,
) -> Chroma:
    """
    Build and persist a ChromaDB vector store from documents.

    Args:
        sources: List of file paths or URLs to index
        docs: Pre-loaded Document objects (alternative to sources)

    Returns:
        Initialized Chroma vector store
    """
    cfg.ensure_dirs()

    if docs is None:
        if sources is None:
            raise ValueError("Provide either 'sources' or 'docs'")
        all_docs = []
        for src in sources:
            all_docs.extend(load_documents(src))
        docs = all_docs

    chunks = chunk_documents(docs)
    embeddings = get_embedding_model()

    log.info(f"Embedding {len(chunks)} chunks into ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=cfg.rag.chroma_persist_dir,
        collection_name=cfg.rag.collection_name,
    )
    vectorstore.persist()
    log.info(f"✅ Vector store saved to: {cfg.rag.chroma_persist_dir}")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    persist_dir = cfg.rag.chroma_persist_dir
    if not Path(persist_dir).exists():
        raise FileNotFoundError(
            f"No vector store found at {persist_dir}. "
            "Run build_vectorstore() first."
        )

    embeddings = get_embedding_model()
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=cfg.rag.collection_name,
    )
    log.info(f"Loaded vector store with {vectorstore._collection.count():,} chunks")
    return vectorstore


def retrieve(query: str, vectorstore: Chroma, top_k: Optional[int] = None) -> List[Document]:
    """
    Retrieve the most relevant document chunks for a query.

    Args:
        query: User's question
        vectorstore: Initialized Chroma vector store
        top_k: Number of chunks to retrieve (defaults to cfg.rag.top_k)

    Returns:
        List of relevant Document chunks with similarity scores
    """
    k = top_k or cfg.rag.top_k

    results = vectorstore.similarity_search_with_relevance_scores(
        query=query,
        k=k,
    )

    # Filter by similarity threshold
    filtered = [
        doc for doc, score in results
        if score >= cfg.rag.similarity_threshold
    ]

    log.debug(f"Retrieved {len(filtered)}/{k} chunks above threshold={cfg.rag.similarity_threshold}")
    return filtered
