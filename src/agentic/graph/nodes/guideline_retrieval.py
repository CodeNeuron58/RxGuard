"""Guideline Retrieval Node: Retrieves relevant clinical guidelines from vectorstore."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from src.agentic.state.schemas import RxGuardState
from src.agentic.utils import get_logger

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


def create_vectorstore():
    """Create vectorstore from PDF guidelines."""
    guideline_dir = Path("data/guidelines")
    vectorstore_path = Path(settings.VECTOR_STORE_PATH)
    
    # Load PDFs
    pdf_files = list(guideline_dir.glob("*.pdf"))
    documents = []
    
    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        docs = loader.load()
        documents.extend(docs)
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create and save vectorstore
    embedding_model = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)
    vectorstore.save_local(str(vectorstore_path))
    
    return vectorstore


def get_vectorstore():
    """Get or create vectorstore (singleton, lazy loading)."""
    global _vectorstore
    if _vectorstore is None:
        vectorstore_path = Path(settings.VECTOR_STORE_PATH)
        
        if vectorstore_path.exists():
            # Load existing vectorstore
            _vectorstore = FAISS.load_local(
                str(vectorstore_path),
                get_embedding_model(),
                allow_dangerous_deserialization=True
            )
        else:
            # Create new vectorstore from PDFs
            _vectorstore = create_vectorstore()
    
    return _vectorstore


def guideline_retrieval_node(state: RxGuardState) -> RxGuardState:
    """Retrieve relevant clinical guidelines based on patient and medication context.
    
    Args:
        state: Current graph state with patient_profile and proposed_medication
        
    Returns:
        Updated state with retrieved_guidelines
    """
    logger.info("--- GUIDELINE RETRIEVAL ---")
    
    # Build search query from patient context and medication
    patient_profile = state["patient_profile"]
    proposed_medication = state["proposed_medication"]
    
    conditions = ", ".join(patient_profile.get("conditions", []))
    drug = proposed_medication.get("drug_name", "")
    
    query = (
        f"Prescribing guidance for {drug} "
        f"in patients with {conditions} "
        "renal impairment contraindications dosing"
    )
    
    # Get vectorstore (load or create)
    vectorstore = get_vectorstore()
    
    # Retrieve top-k similar documents
    results = vectorstore.similarity_search(query, k=settings.TOP_K_RETRIEVAL)
    
    logger.info(f"Retrieved {len(results)} guidelines for query: {query[:50]}...")
    
    # Format retrieved documents
    state["retrieved_guidelines"] = [
        {
            "source": r.metadata.get("source"),
            "page": r.metadata.get("page"),
            "content": r.page_content
        }
        for r in results
    ]
    
    return state