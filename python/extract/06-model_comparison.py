import json
import random
import ollama
from pathlib import Path
import pdfplumber
import pandas as pd
import pickle
import csv

# load needed files
# chunks
with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    loaded_chunks = json.load(f)

# SDG candidates per chunk
with open("data/processed/top_targets_per_chunk.pkl", "rb") as f:
    top10 = pickle.load(f)

# csv with SDG targets and description
with open("data/SDG-targets.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)
    SDGs = [{"sdg": row[1], "description": row[2]} for row in reader]

# define the prompt
CLASSIFY_PROMPT = """You are analyzing a subsection of a German development ministry publication.
Classify the text into SDG targets, regions, and countries.

Use ONLY the following candidate SDG targets. Do not assign targets outside this list. 

{targets}

Rules:
- Allowed regions: Africa, Asia, Latin America, Europe
- Use standard English country names (e.g., Kenya, India, Germany).
- Do not include regions or continents as countries.
- Return SDG targets as numbers only.
- Include an SDG if the text clearly relates to it, even if the SDG number or name is not mentioned.
- Use [] if no SDG target, region or country apply.
- Provide a one-sentence reason for your choice.

Examples:

Text: "Länderkapitel  |  75 Menschenrechtsaktivistinnen und -aktivisten, die in Landkonflikten Position beziehen oder die Verletzung von Indigenenrechten anprangern, Aus Kreisen evangelikaler Fundamentalisten werden immer wieder Übergriffe auf religiöse und sexuelle Minderheiten bekannt. LGBTIQ+ Personen werden auch aus religiöser Motivation beleidigt und angegriffen. Der zunehmende Einfluss evangelikaler Fundamentalisten erschwert es Anhängerinnen und Anhängern afrobrasilianischer Religionen, ihren Glauben öffentlich auszuüben."
Output: {{"sdgs": [10.2, 10.3, 16.1, 16.10], "regions": ["Latin America"], "countries": ["Brazil"], "reason": [It's about economic and social inclusion but also peace and institutions]}}

Text: "28  |  Nachhaltige Textilien – Eine Frage der Verantwortung! Nationaler Aktionsplan Wirschaft und Menschenrechte Das Lieferkettensorgfaltspflichtengesetz in Deutschland Die Bundesregierung hat im Nationalen Aktionsplan Wirtschaft und Menschenrechte 2016 neben den Pflichten des Staates erstmals auch die Verantwortung von deutschen Unternehmen für die Achtung der Menschenrechte verankert und konkrete Erwartungen an die Umsetzung der Sorgfaltspflichten durch die Privatwirtschaft formuliert."
Output: {{"sdgs": [8.7, 8.8, 12.6], "regions": ["Europe"], "countries": ["Germany"], "reason": [Focuses on economic growth, specifically on labour with a focus on transnational companies]}}

Text: "Die Grundgesamtheit der zu befragenden Vorhaben entstammt einer internen Portfolioanalyse des SV Menschenrechte. Die Befragung erfolgte mittels der Befragungssoftware SurveyXact© zwischen Juli und August 2022 und hatte bei 251 kontaktierten Vorhaben einen Rücklauf von 90 beantwortenden Vorhaben, von denen 85 die Befragung vollständig abschlossen, und wies somit eine Beteiligung von 34% auf. Die Auswertung der Befragungsdaten erfolgte nach der Befragung unter Einsatz statistischer Software in Form von uni-, bi und multivariaten Analysen. Fokusgruppe mit Vorhaben der finanziellen Entwicklungszusammenarbeit"
Output: {{"sdgs": [], "regions": [], "countries": [], "reason": [No SDG mentioned]}}

Text: "{text}"
Output:"""

classified_chunks = []

models = ["llama3.2:3b", "olfh/teuken-7b-instruct-commercial-v0.4:7b", "qwen2.5:7b", "mistral:7b", "llama3.1:8b"]

# take a random sample to compare different models
chunk_sample = random.sample(loaded_chunks, 10)

results = []

for chunk in chunk_sample:

    SDG_candidates = top10[chunk["index"]]
    SDG_targets = [SDGs[i] for i in SDG_candidates]
    
    targets_text = "\n".join(
        f"{i+1} Targets: {target["sdg"]}: {target["description"]}"
        for i, target in enumerate(SDG_targets)
    )

    for model_name in models:
        response = ollama.chat(
            model = model_name,
            messages = [{"role": "user", "content": CLASSIFY_PROMPT.format(text = chunk["text"], targets = targets_text)}],
            format = "json",
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
print(df.to_string(index = False))
df.to_csv("data/processed/model_comparison.csv", index = False)
