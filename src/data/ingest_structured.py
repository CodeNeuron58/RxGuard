
"""Ingest Structured JSON Data into FAISS Vector Store."""

import json
import logging
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config.settings import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_structured_data(data_dir: Path) -> List[Document]:
    """Load JSON files and convert to LangChain Documents."""
    documents = []
    
    # Ensure directory exists
    if not data_dir.exists():
        logger.warning(f"Data directory {data_dir} does not exist. Creating it.")
        data_dir.mkdir(parents=True, exist_ok=True)
        return []
    
    json_files = list(data_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in {data_dir}")
    
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Handle list of items or single item
            if isinstance(data, dict):
                data = [data]
                
            for item in data:
                # Create semantic content string for embedding
                # We combine relevant fields to make the vector representation rich
                page_content = (
                    f"Drug: {item.get('drug_name', 'Unknown')}\n"
                    f"Category: {item.get('category', 'General')}\n"
                    f"Population: {item.get('population', 'General')}\n"
                    f"Fact: {item.get('fact', '')}"
                )
                
                # Metadata for retrieval and citation
                metadata = {
                    "source": item.get("source", "Unknown"),
                    "drug_name": item.get("drug_name"),
                    "category": item.get("category"),
                    "id": item.get("id")
                }
                
                doc = Document(page_content=page_content, metadata=metadata)
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            
    return documents

def ingest():
    """Main ingestion function."""
    data_dir = Path("data/structured")
    vectorstore_path = Path(settings.VECTOR_STORE_PATH) # Reuse existing path config
    
    logger.info("Starting ingestion...")
    
    # 1. Load Data
    docs = load_structured_data(data_dir)
    
    if not docs:
        logger.warning("No documents found. Please add JSON files to data/structured/")
        return
        
    logger.info(f"Loaded {len(docs)} structured documents.")
    
    # 2. Create Embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL
    )
    
    # 3. Create/Update Vector Store
    # Note: This overwrites the existing one for now to ensure cleanliness
    vectorstore = FAISS.from_documents(docs, embedding=embedding_model)
    vectorstore.save_local(str(vectorstore_path))
    
    logger.info(f"Vector store saved to {vectorstore_path}")

if __name__ == "__main__":
    ingest()
