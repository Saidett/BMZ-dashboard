import csv
import ollama
import pickle 

with open("data/SDG-targets.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)
    SDGs = [{"sdg": row[1], "description": row[2]} for row in reader]

EMBEDDING_MODEL = "jina/jina-embeddings-v2-base-de"

SDG_embedding = []

# define embedding function that appends the dict
def embed_SDG(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk["description"])['embeddings'][0]
  
  SDG_embedding.append({
    "sdg": chunk["sdg"],
    "description": chunk["description"],
    "embedding": embedding
  })

# run embedding
for i, target in enumerate(SDGs):
  embed_SDG(target)
  print(f"Added target {i+1}/{len(SDGs)}")

# save
with open("data/processed/SDGs_embedded.pkl", 'wb') as f:
    pickle.dump(SDG_embedding, f)