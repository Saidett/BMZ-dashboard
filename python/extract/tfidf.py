from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# load json with chunks
with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    loaded_chunks = json.load(f)

# turn into document feature matrix for term frequency inverse document frequency
vec = TfidfVectorizer(max_features=500)
X = vec.fit_transform([c["text"] for c in loaded_chunks])

# select k for number of clusters (examples)
k = 5
km = KMeans(n_clusters=k, random_state=0).fit(X)

centroids = km.cluster_centers_
examples = []
for label in range(k):
    cluster_indices = [i for i, l in enumerate(km.labels_) if l == label]
    # find the chunk closest to this cluster's centroid
    best = min(cluster_indices, key=lambda i: (X[i] - centroids[label]).sum())
    examples.append(loaded_chunks[best])
