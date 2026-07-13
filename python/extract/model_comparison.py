import json
import random
import ollama
from pathlib import Path
import pdfplumber
import pandas as pd

from python.config import CLASSIFY_PROMPT

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
            "text": text_preview,
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