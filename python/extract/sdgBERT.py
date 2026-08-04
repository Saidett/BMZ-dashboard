from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import json

# no need for tokenization because of HF pipeline function
# load chunks
with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    loaded_chunks = json.load(f)

# import the classifier
classifier = pipeline("text-classification", model = "MauriceV2021/AuroraSDGsModel")

# truncation to ensure its less than BERT maximum of 512 tokens (altho shouldnt trigger) and batching to reduce calls
results = classifier([chunk["text"] for chunk in loaded_chunks], truncation = True, batch_size = 16, top_k = 17)

sorted(results[1], key=lambda d: d['score'], reverse = True)[:2]
sum(item["score"] for item in results[1])

{k: v for k, v in points.iteritems() if v[0] < 5 and v[1] < 5}


for chunk in results:



