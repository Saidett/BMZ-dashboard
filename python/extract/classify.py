import json
import random
import ollama
from pathlib import Path
import pdfplumber
import pandas as pd

from python.config import CLASSIFY_PROMPT
# Pre-generate text summaries for each SDG and the BMZ strategy from the chunks?

# load json with chunks
with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    loaded_chunks = json.load(f)

classified_chunks = []

for chunk in loaded_chunks:

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(text=chunk.get("text"))}],
        format="json")
    
    info = json.loads(response["message"]["content"])
    info["pdf_name"] = chunk.get("pdf")
    classified_chunks.append(info)
    print(f"{chunk.get("pdf")}: {info["regions"]}, {info["sdgs"]}")

#pd.DataFrame(classified_chunks).to_csv("data/processed/publikationen_klassifiziert.csv", index=False)
