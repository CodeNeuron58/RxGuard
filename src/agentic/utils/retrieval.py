"""Retrieval Utilities: Vectorstore and Embedding management."""

from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings
from src.agentic.utils.logging_config import get_logger

logger = get_logger(__name__)

# Global variables for lazy loading
_vectorstore = None
_embedding_model = None


def get_embedding_model():
    """Get or create embedding model (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )
    return _embedding_model


def get_vectorstore():
    """Get or create vectorstore (singleton, lazy loading)."""
    global _vectorstore
    if _vectorstore is None:
        vectorstore_path = Path(settings.VECTOR_STORE_PATH)
        
        if vectorstore_path.exists():
            # Load existing vectorstore
            logger.info("Loading faiss with AVX2 support.")
            try:
                _vectorstore = FAISS.load_local(
                    str(vectorstore_path),
                    get_embedding_model(),
                    allow_dangerous_deserialization=True
                )
                logger.info("Successfully loaded faiss with AVX2 support.")
            except Exception as e:
                logger.error(f"Failed to load vectorstore: {e}")
                raise e
        else:
            logger.warning("Vectorstore not found at path.")
            return None
    
    return _vectorstore
