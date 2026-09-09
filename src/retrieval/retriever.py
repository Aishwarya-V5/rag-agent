import os
import pickle
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

VECTOR_STORE_PATH = Path("vector_store/vector_index.pkl")
BM25_INDEX_PATH = Path("vector_store/bm25_index.pkl")

with open(VECTOR_STORE_PATH, "rb") as f:
    vector_data = pickle.load(f)

with open(BM25_INDEX_PATH, "rb") as f:
    bm25_data = pickle.load(f)

embeddings_matrix = np.array(vector_data["embeddings"])
documents = vector_data["documents"]
metadatas = vector_data["metadatas"]
bm25 = bm25_data["bm25"]

def get_embedding(text: str):
    response = mistral_client.embeddings.create(
        model="mistral-embed",
        inputs=[text]
    )
    return response.data[0].embedding

def cosine_similarity(query_vec, matrix):
    query_vec = np.array(query_vec)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
    norms[norms == 0] = 1e-10
    return np.dot(matrix, query_vec) / norms


def retrieve(query: str, k=10, group_filter=None):
    query_embedding = get_embedding(query)

    if group_filter and group_filter != "All":
        indices = [i for i, m in enumerate(metadatas) if m["group"] == group_filter]
    else:
        indices = list(range(len(documents)))

    if not indices:
        return []

    filtered_embeddings = embeddings_matrix[indices]
    filtered_documents = [documents[i] for i in indices]
    filtered_metadatas = [metadatas[i] for i in indices]

    sims = cosine_similarity(query_embedding, filtered_embeddings)
    vector_ranks = np.argsort(sims)[::-1]

    tokenized_query = query.lower().split()
    # BM25 needs its own filtered lookup — build a matching subset
    bm25_scores_full = bm25.get_scores(tokenized_query)
    bm25_scores = [bm25_scores_full[i] for i in indices]
    bm25_ranks = np.argsort(bm25_scores)[::-1]

    rrf_scores = {}
    K = 60
    for rank, idx in enumerate(vector_ranks):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (K + rank + 1)
    for rank, idx in enumerate(bm25_ranks):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (K + rank + 1)

    top_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]

    results = [(filtered_documents[i], filtered_metadatas[i]) for i in top_indices]
    return results