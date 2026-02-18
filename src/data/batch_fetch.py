"""
Batch Fetcher for RxGuard.

Automates the fetching and structuring of high-priority drugs for the competition.
Target Logic:
- Focus on drugs with high renal/interaction risks (CKD context).
- Focus on drugs with narrow therapeutic indices (Warfarin).
"""

import subprocess
import time
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# List of high-impact drugs for the MedGemma challenge
TARGET_DRUGS = [
    # Renally cleared / Nephrotoxic (CKD Focus)
    "Metformin",
    "Gabapentin",
    "Spironolactone",
    "Ibuprofen",
    "Naproxen",
    
    # Narrow Therapeutic Index / High Interaction
    "Warfarin",
    "Digoxin",
    "Amiodarone",
    "Lithium",
    
    # Common polypharmacy in elderly
    "Atorvastatin",
    "Omeprazole",
    "Levothyroxine",
    "Amlodipine"
]

def main():
    logger.info(f"🚀 Starting Batch Ingestion for {len(TARGET_DRUGS)} drugs...")
    
    for i, drug in enumerate(TARGET_DRUGS, 1):
        logger.info(f"\n[{i}/{len(TARGET_DRUGS)}] Processing {drug}...")
        
        try:
            # Run the fetch command
            # We use subprocess to isolate memory/state between runs
            subprocess.run(
                ["python", "src/data/fetch_openfda.py", "--drug", drug],
                check=True
            )
            
            # Rate limit compliance (be nice to openFDA and Groq)
            time.sleep(2) 
            
        except subprocess.CalledProcessError:
            logger.error(f"❌ Failed to process {drug}")
        except Exception as e:
            logger.error(f"❌ Unexpected error for {drug}: {e}")

    logger.info("\n✅ Batch Processing Complete!")
    logger.info("Now run: python src/data/ingest_structured.py")

if __name__ == "__main__":
    main()
