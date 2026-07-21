import json
import random
import ollama
from pathlib import Path
import pdfplumber
import pandas as pd

CLASSIFY_PROMPT = """You are analyzing a subsection of a German development ministry publication.
Classify the text into SDG targets, regions, and countries.

Use ONLY the following candidate SDG targets. Do not assign targets outside this list: {targets}

Rules:
- Allowed regions: Africa, Asia, Latin America, Europe
- Use standard English country names (e.g., Kenya, India, Germany).
- Do not include regions or continents as countries.
- Return SDG targets as numbers only.
- Include an SDG if the text clearly relates to it, even if the SDG number or name is not mentioned.
- Use [] if no SDG target, region or country apply.

Examples:

Text: "Länderkapitel  |  75 Menschenrechtsaktivistinnen und -aktivisten, die in Landkonflikten Position beziehen oder die Verletzung von Indigenenrechten anprangern, Aus Kreisen evangelikaler Fundamentalisten werden immer wieder Übergriffe auf religiöse und sexuelle Minderheiten bekannt. LGBTIQ+ Personen werden auch aus religiöser Motivation beleidigt und angegriffen. Der zunehmende Einfluss evangelikaler Fundamentalisten erschwert es Anhängerinnen und Anhängern afrobrasilianischer Religionen, ihren Glauben öffentlich auszuüben."
Output: {{"sdgs": [10.2, 10.3, 16.1, 16.10], "regions": ["Latin America"], "countries": ["Brazil"]}}

Text: "28  |  Nachhaltige Textilien – Eine Frage der Verantwortung! Nationaler Aktionsplan Wirschaft und Menschenrechte Das Lieferkettensorgfaltspflichtengesetz in Deutschland Die Bundesregierung hat im Nationalen Aktionsplan Wirtschaft und Menschenrechte 2016 neben den Pflichten des Staates erstmals auch die Verantwortung von deutschen Unternehmen für die Achtung der Menschenrechte verankert und konkrete Erwartungen an die Umsetzung der Sorgfaltspflichten durch die Privatwirtschaft formuliert."
Output: {{"sdgs": [8.7, 8.8, 12.6], "regions": ["Europe"], "countries": ["Germany"]}}

Text: "Die Grundgesamtheit der zu befragenden Vorhaben entstammt einer internen Portfolioanalyse des SV Menschenrechte. Die Befragung erfolgte mittels der Befragungssoftware SurveyXact© zwischen Juli und August 2022 und hatte bei 251 kontaktierten Vorhaben einen Rücklauf von 90 beantwortenden Vorhaben, von denen 85 die Befragung vollständig abschlossen, und wies somit eine Beteiligung von 34% auf. Die Auswertung der Befragungsdaten erfolgte nach der Befragung unter Einsatz statistischer Software in Form von uni-, bi und multivariaten Analysen. Fokusgruppe mit Vorhaben der finanziellen Entwicklungszusammenarbeit"
Output: {{"sdgs": [], "regions": [], "countries": []}}

Text: "{text}"
Output:"""

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
            messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(text=chunk["text"], )}],
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
