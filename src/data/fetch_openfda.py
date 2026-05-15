"""
Fetch and Structure Drug Data from openFDA.

Usage:
    python src/data/fetch_openfda.py --drug "Lisinopril"
    python src/data/fetch_openfda.py --drug "Metformin"

Prerequisites:
    - Internet access (for openFDA API)
    - GROQ_API_KEY environment variable set (for structuring)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Import project modules

from src.agentic.agents.base import get_llm

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
OPENFDA_API_URL = "https://api.fda.gov/drug/label.json"
DATA_DIR = Path("data/structured")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# === SCHEMAS ===
class StructuredGuideline(BaseModel):
    """Schema for the final JSON output."""
    id: str
    drug_name: str
    source: str = "FDA Label (openFDA)"
    publish_date: str
    facts: List[Dict[str, Any]] = Field(description="List of extracted facts with category, text, population.")

# === FUNCTIONS ===

def fetch_openfda_label(drug_name: str) -> Dict[str, Any]:
    """Fetch raw drug label from openFDA."""
    logger.info(f"🔍 Searching openFDA for: {drug_name}...")
    
    # Query for the specific brand/generic name
    # We use search=openfda.brand_name:X OR openfda.generic_name:X
    query = f'openfda.brand_name:"{drug_name}"+openfda.generic_name:"{drug_name}"'
    params = {
        "search": query,
        "limit": 1
    }
    
    try:
        response = requests.get(OPENFDA_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "results" not in data or not data["results"]:
            logger.error(f"❌ No results found for {drug_name}")
            return None
            
        return data["results"][0]
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"API Error: {e}")
        return None
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return None


def extract_key_sections(raw_data: Dict[str, Any]) -> str:
    """Extract relevant safety sections from raw JSON."""
    relevant_fields = [
        "contraindications",
        "warnings",
        "boxed_warning",
        "warnings_and_cautions",
        "dosage_and_administration",
        "geriatric_use",
        "renal_impairment_specific",
        "drug_interactions"
    ]
    
    extracted_text = []
    
    for field in relevant_fields:
        if field in raw_data:
            # openFDA returns fields as lists of strings
            content = " ".join(raw_data[field])
            extracted_text.append(f"=== {field.upper().replace('_', ' ')} ===\n{content}\n")
            
    return "\n".join(extracted_text)


def structure_with_llm(drug_name: str, raw_text: str) -> Dict[str, Any]:
    """Use LLM to convert raw text into structured JSON."""
    logger.info(f"🧠 Structuring data for {drug_name} using LLM...")
    
    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=StructuredGuideline)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a Clinical Data Specialist. Convert the raw FDA label text into a structured JSON dataset.\n"
         "Extract discrete 'facts' for clinical decision support.\n"
         "Categories: Contraindication, Warning, Dosage, Interaction, Geriatric, Renal.\n"
         "Ensure 'text' is concise but detailed enough for RAG."),
        ("human",
         "Drug: {drug}\n\n"
         "Raw FDA Text:\n{raw_text}\n\n"
         "Format:\n{format_instructions}")
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "drug": drug_name,
            "raw_text": raw_text[:25000], # Trucate text to strictly fit context window if huge
            "format_instructions": parser.get_format_instructions()
        })
        return result
    except Exception as e:
        logger.error(f"LLM Structuring failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Fetch and structure FDA drug labels.")
    parser.add_argument("--drug", required=True, help="Name of the drug (e.g., Lisinopril)")
    
    args = parser.parse_args()
    drug_name = args.drug
    
    # 1. Fetch
    raw_json = fetch_openfda_label(drug_name)
    if not raw_json:
        return
        
    # 2. Extract Text
    relevant_text = extract_key_sections(raw_json)
    if not relevant_text:
        logger.warning("No relevant safety sections found in label.")
        return
        
    logger.info(f"Extracted {len(relevant_text)} characters of safety data.")
    
    # 3. Structure
    structured_data = structure_with_llm(drug_name, relevant_text)
    
    if structured_data:
        # Add metadata manually to ensure it's correct
        structured_data["id"] = f"{drug_name.lower()}-fda-label"
        structured_data["publish_date"] = raw_json.get("effective_time", "Unknown")
        
        # 4. Save
        filename = DATA_DIR / f"{drug_name.lower()}.json"
        with open(filename, "w") as f:
            json.dump(structured_data, f, indent=2)
            
        logger.info(f"✅ Successfully saved structured data to {filename}")
        print(f"\nSuccess! Created {filename}")
        print("Run 'python src/data/ingest_structured.py' to update the vector store.")

if __name__ == "__main__":
    main()
