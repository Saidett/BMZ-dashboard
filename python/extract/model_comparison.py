import json
import random
import ollama
from pathlib import Path
import pdfplumber
import pandas as pd

CLASSIFY_PROMPT = """You are analyzing a subsection of a German development ministry publication.

Extract:
- regions
- countries
- SDGs
- reasoning

Allowed regions: Africa, Asia, Latin America, Europe, Global

SDGs:
1=No Poverty, 2=Zero Hunger, 3=Good Health and Well-being, 4=Quality Education, 5=Gender Equality
6=Clean Water and Sanitation, 7=Affordable and Clean Energy, 8=Decent Work and Economic Growth, 9=Industry, Innovation and Infrastructure, 10=Reduced Inequalities
11=Sustainable Cities and Communities, 12=Responsible Consumption and Production, 13=Climate Action, 14=Life Below Water
15=Life on Land, 16=Peace, Justice and Strong Institutions, 17=Partnerships for the goals

Rules:
- Use only the allowed region names in English.
- Use standard English country names (e.g., Kenya, India, Germany).
- Do not include regions or continents as countries.
- Return SDGs as numbers only.
- Include an SDG if the text clearly relates to it, even if the SDG number or name is not mentioned.
- Use [] if no region, SDG or country apply.
- "reasoning": a short 1-2 sentence explanation of your choices (keep it brief).

Examples:
{{"sdgs": [13], "regions": ["Africa"], "countries": ["Kenya"], "reasoning": "Mentions renewable energy projects in Kenya, relating to SDG 13 (Climate Action)."}}
{{"sdgs": [1, 2, 10], "regions": ["Asia", "Africa"], "countries": [], "reasoning": "Discusses poverty reduction and hunger programs across Asia and Africa, relating to SDGs 1, 2, and 10."}}
{{"sdgs": [], "regions": ["Global"], "countries": [], "reasoning": "General statement about international cooperation with no specific region or SDG."}}

Text:
{text}"""

# load json with chunks
with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    loaded_chunks = json.load(f)

classified_chunks = []

models = ["llama3.2:3b", "mistral:7b", "llama3.1:8b", "qwen2.5:7b"]

chunk_sample = random.sample(loaded_chunks, 10)

results = []
for chunk in chunk_sample:
    for model_name in models:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(text=chunk["text"])}],
            format="json",
        )
        info = json.loads(response["message"]["content"])
        results.append({
            "pdf": chunk["pdf"],
            "text": chunk["text"],
            "model": model_name,
            "regions": info.get("regions", []),
            "sdgs": info.get("sdgs", []),
            "countries": info.get("countries", []),
            "reasoning": info.get("reasoning", ""),
        })

df = pd.DataFrame(results)
# long format: each row = one model's answer per chunk
print(df.to_string(index=False))
df.to_csv("data/processed/model_comparison.csv", index=False)
