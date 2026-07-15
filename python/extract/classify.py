import json
import random
import ollama
from pathlib import Path
import pdfplumber
import pandas as pd
import pickle 

# classification prompt
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

# define similarity function for target RAG
def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

# Pre-generate text summaries for each SDG and the BMZ strategy from the chunks?

# load json with chunks
with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    loaded_chunks = json.load(f)

# load SDG embeddings
with open("data/processed/SDGs_embedded.pkl", "rb") as f:
    SDG_embedding = pickle.load(f)

classified_chunks = []

for chunk in loaded_chunks:

    chunk_text = chunk.get("text")
    target_list = 

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(text = chunk_text, targets = target_list)}],
        format="json")
    
    info = json.loads(response["message"]["content"])
    info["pdf_name"] = chunk.get("pdf")
    classified_chunks.append(info)
    print(f"{chunk.get("pdf")}: {info["regions"]}, {info["sdgs"]}")

#pd.DataFrame(classified_chunks).to_csv("data/processed/publikationen_klassifiziert.csv", index=False)
