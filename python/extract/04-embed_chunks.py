import csv
import ollama
import json
import pickle
import numpy as np

# load json with chunks
with open("data/processed/chunks.json", "r", encoding = "utf-8") as f:
    loaded_chunks = json.load(f)

# load embedded SDG targets
with open("data/processed/SDGs_embedded.pkl", "rb") as f:
    SDG_embedding = pickle.load(f)

EMBEDDING_MODEL = "jina/jina-embeddings-v2-base-de"

# append chunk embedding to dict
for i, chunk in enumerate(loaded_chunks):
  embedding = ollama.embed(model = EMBEDDING_MODEL, input = chunk["text"])['embeddings'][0]
  loaded_chunks[i]["embedding"] = embedding
  print(f"Embedded chunk {i+1}/{len(loaded_chunks)}")

# turn the list of embeddings into a matrix (5205, 768 and 169, 768)
chunk_embedding_array = np.array([c["embedding"] for c in loaded_chunks])
SDG_embedding_array = np.array([c["embedding"] for c in SDG_embedding])

# to normalise the embeddings, computes the distance of the 768 dimensional vector from zero to normalise them (0-1)
chunk_norms = np.linalg.norm(chunk_embedding_array, axis = 1, keepdims=True)
SDG_norms = np.linalg.norm(SDG_embedding_array, axis = 1, keepdims=True)

# dividing the array by the distance from before to normalise
chunk_normed = chunk_embedding_array / chunk_norms
SDG_normed = SDG_embedding_array / SDG_norms

# calculate matrix
similarities = chunk_normed @ SDG_normed.T 

# now select the top 10 per chunk
top10 = np.argsort(similarities, axis = 1)[:, -10:]

# save array with top 10 targets per chunk
with open("data/processed/top_targets_per_chunk.pkl", 'wb') as f:
    pickle.dump(top10, f)