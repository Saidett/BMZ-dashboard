import json
import ollama
from pathlib import Path
import pdfplumber
import pandas as pd

CLASSIFY_PROMPT = """You are analyzing a German BMZ publication.
Extract structured data in JSON format and in English:

{{
  "countries": ["list of countries MENTIONED, be specific. Exclude Germany"],
  "regions": ["list of regions like Africa, Asia, Latin America, Global, None"],
  "summary": "One sentence summary in German",
  "specific_mentions": ["Kenia", "Klimafonds", ... specific entities mentioned]
}}

First 3000 characters of publication:
{text}
"""

pdf_dir = Path("data/raw/")
results = []

for pdf_path in sorted(pdf_dir.glob("*.pdf")):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages[:5])
    
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(text=text[:3000])}],
        format="json",
    )
    
    info = json.loads(response["message"]["content"])
    info["pdf_name"] = pdf_path.name
    info["pdf_path"] = str(pdf_path)
    results.append(info)
    print(f"{pdf_path.name}: {info['countries']}")

pd.DataFrame(results).to_csv("data/processed/publikationen_klassifiziert.csv", index=False)