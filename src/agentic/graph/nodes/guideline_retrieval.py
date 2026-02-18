"""Guideline Retrieval Node: Retrieves relevant clinical guidelines from vectorstore."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import settings
from src.agentic.state.schemas import RxGuardState
from src.agentic.utils import get_logger
from src.agentic.agents.base import get_llm

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


# === Dynamic Query Generation ===
query_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a clinical research assistant. Your goal is to generate a specific search query to find relevant medical guidelines."),
    ("human", 
     "Generate a targeted search query to check the safety of {drug} for this patient.\n\n"
     "Patient Profile: {patient_context}\n"
     "Previous Feedback (if any): {feedback}\n\n"
     "If there is feedback, focus solely on addressing that specific gap (e.g., 'Renal dosing for Naproxen').\n"
     "If no feedback, generate a broad safety search query (e.g. 'Contraindications for Naproxen in elderly').\n"
     "Output ONLY the search query string.")
])

query_gen_chain = query_gen_prompt | get_llm() | StrOutputParser()


def guideline_retrieval_node(state: RxGuardState) -> RxGuardState:
    """Retrieve relevant clinical guidelines based on patient and medication context.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with retrieved_guidelines and research_log
    """
    logger.info("--- GUIDELINE RETRIEVAL (Reflexive) ---")
    
    patient_profile = state["patient_profile"]
    proposed_medication = state["proposed_medication"]
    feedback = state.get("critique_feedback")
    
    # 1. Generate Dynamic Query
    query = query_gen_chain.invoke({
        "drug": proposed_medication.get("drug_name"),
        "patient_context": str(patient_profile),
        "feedback": feedback if feedback else "None"
    })
    
    logger.info(f"Generated Research Query: {query}")
    
    # 2. Update Research Log
    new_log_entry = f"Query: {query} | Context: {feedback if feedback else 'Initial Search'}"
    # Use append to extend the list, handling potential None
    current_log = state.get("research_log", []) or []
    current_log.append(new_log_entry)
    state["research_log"] = current_log
    
    # 3. Search Vectorstore
    vectorstore = get_vectorstore()
    # Retrieve top-k similar documents
    results = vectorstore.similarity_search(query, k=settings.TOP_K_RETRIEVAL)
    
    logger.info(f"Retrieved {len(results)} guidelines for query: {query[:50]}...")
    
    # Format retrieved documents
    new_guidelines = [
        {
            "source": r.metadata.get("source"),
            "page": r.metadata.get("page"),
            "content": r.page_content,
            "query_used": query
        }
        for r in results
    ]
    
    # Append to existing guidelines (Agentic Memory)
    current_guidelines = state.get("retrieved_guidelines", []) or []
    state["retrieved_guidelines"] = current_guidelines + new_guidelines
    
    return state