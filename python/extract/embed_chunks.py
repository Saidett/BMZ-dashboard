import csv
import ollama
import json
import pickle
import numpy as np

# load json with chunks
with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    loaded_chunks = json.load(f)

# load embedded SDG targets
with open("data/processed/SDGs_embedded.pkl", "rb") as f:
    SDG_embedding = pickle.load(f)

EMBEDDING_MODEL = "nomic-embed-text"

# append embedding to dict
for i, chunk in enumerate(loaded_chunks):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk["text"])['embeddings'][0]
  loaded_chunks[i]["embedding"] = embedding
  print(f"Embedded chunk {i+1}/{len(loaded_chunks)}")

# find most similar SDG target with cosine similarity funct
def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

chunk_embedding_array = np.array([c["embedding"] for c in loaded_chunks])
SDG_embedding_array = np.array([c["embedding"] for c in SDG_embedding])
similarities = chunk_embedding_array @ SDG_embedding_array.T

for chunk in loaded_chunks:
    
    target_similarity = []
    
    for i, target in SDG_embedding:
        similiarity = cosine_similarity(chunk["embedding"], target["embedding"])
        target_similarity.append(similiarity)
        target_similarity[i]["target"] = 

cosine_similarity(loaded_chunks[1]["embedding"], SDG_embedding[2]["embedding"])

# save
with open("data/processed/SDGs_embedded.pkl", 'wb') as f:
    pickle.dump(SDG_embedding, f)

def cosine_similarity_matrix(chunk_embs, target_embs):
    chunk_embs = np.array(chunk_embs)
    target_embs = np.array(target_embs)
    
    # normalize
    chunk_norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True)
    target_norms = np.linalg.norm(target_embs, axis=1, keepdims=True)
    
    chunk_embs_normed = chunk_embs / chunk_norms
    target_embs_normed = target_embs / target_norms
    
    return chunk_embs_normed @ target_embs_normed.T

# (5000, 40) similarity matrix, all at once
sim_matrix = cosine_similarity_matrix(chunk_embedding_array, SDG_embedding_array)
