from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" 
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OLLAMA_MODEL = "llama3.2:3b"
BMZ_URL = "https://www.bmz.de/ajax/filterlist/de/24710-24710"