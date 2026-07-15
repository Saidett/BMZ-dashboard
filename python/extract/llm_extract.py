import json
import ollama
SDG = SDG_LIST

EXTRACTION_PROMPT = """You are analyzing German climate policy documents.

Extract all quantitative climate commitments from the text below.
Include emissions targets, renewable energy goals, finance pledges,
sectoral targets, and adaptation goals.

Return ONLY a valid JSON array with this structure:
[
  {{
    "title": "short name",
    "category": "mitigation|adaptation|finance|energy|land_use|cross_cutting",
    "target_value": "e.g. 65% reduction",
    "target_year": 2030,
    "description": "detailed description",
    "source_text": "verbatim quote from document"
  }}
]

Text:
{chunk}
"""


def extract_commitments_from_chunk(chunk: str, model: str = "llama3.2:3b") -> list[dict]:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(chunk=chunk)}],
        format="json",
    )
    raw = response["message"]["content"]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("Failed to parse JSON from LLM response, skipping chunk")
        return []


def extract_all_commitments(chunks: list[str], model: str = "llama3.2:3b") -> list[dict]:
    all_commitments = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)} ...")
        result = extract_commitments_from_chunk(chunk, model)
        all_commitments.extend(result)
    return all_commitments
