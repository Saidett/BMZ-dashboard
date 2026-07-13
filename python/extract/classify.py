import json
import random
import ollama
from pathlib import Path
import pdfplumber
import pandas as pd

# define prompt to classify
CLASSIFY_PROMPT = """You are analyzing a subsection of a German development ministry publication.

Extract:
- regions
- countries
- SDGs

Allowed regions: Africa, Asia, Latin America, Europe, Global

SDGs:
1=No Poverty, 2=Zero Hunger, 3=Good Health and Well-being, 4=Quality Education, 5=Gender Equality
6=Clean Water and Sanitation, 7=Affordable and Clean Energy, 8=Decent Work and Economic Growth, 9=Industry, Innovation and Infrastructure, 10=Reduced Inequalities
11=Sustainable Cities and Communities, 12=Responsible Consumption and Production, 13=Climate Action, 14=Life Below Water
15=Life on Land, 16=Peace, Justice and Strong Institutions, 17=Partnerships for the goals

Rules:
- Use only the allowed region names.
- Use standard English country names (e.g., Kenya, India, Germany).
- Do not include regions or continents as countries.
- Return SDGs as numbers only.
- Include an SDG if the text clearly relates to it, even if the SDG number or name is not mentioned.
- Use [] if no region, SDG or country apply.

Examples:
{{"sdgs": [13], "regions": ["Africa"], "countries": ["Kenya"]}}
{{"sdgs": [1, 2, 10], "regions": ["Asia", "Africa"], "countries": []}}
{{"sdgs": [], "regions": ["Global"], "countries": []}}

Text:
{text}"""

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

models = ["llama3.2:3b", "mistral:7b", "llama3.1:8b", "qwen2.5:7b"]

chunk_sample = random.sample(loaded_chunks, 10)

results = []
for chunk in chunk_sample:
    entry = {"pdf": chunk["pdf"], "text": chunk["text"]}
    for model_name in models:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(text=chunk["text"])}],
            format="json",
        )
        info = json.loads(response["message"]["content"])
        info.setdefault("regions", [])
        info.setdefault("sdgs", [])
        info.setdefault("countries", [])
        entry[f"{model_name}_regions"] = info["regions"]
        entry[f"{model_name}_sdgs"] = info["sdgs"]
        entry[f"{model_name}_countries"] = info["countries"]
    results.append(entry)

pd.DataFrame(results).to_csv("data/processed/model_comparison.csv", index=False)
