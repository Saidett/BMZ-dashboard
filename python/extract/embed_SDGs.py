import csv
import ollama
import pickle 

with open("data/SDG-targets.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)
    SDGs = [{"sdg": row[1], "description": row[2]} for row in reader]

EMBEDDING_MODEL = "nomic-embed-text"

SDG_embedding = []

# define embedding function that appends the dict
def add_chunk_to_database(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk["description"])['embeddings'][0]
  
  SDG_embedding.append({
    "sdg": chunk["sdg"],
    "description": chunk["description"],
    "embedding": embedding
  })

# run embedding
for i, chunk in enumerate(SDGs):
  add_chunk_to_database(chunk)
  print(f"Added chunk {i+1}/{len(SDGs)} to the database")

# save
with open("data/processed/SDGs_embedded.pkl", 'wb') as f:
    pickle.dump(SDG_embedding, f)
