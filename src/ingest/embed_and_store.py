import os
import time
import json
import pickle
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral
from rank_bm25 import BM25Okapi

from src.ingest.extract import extract_all
from src.ingest.chunk import chunk_text

load_dotenv()
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

VECTOR_STORE_PATH = Path("vector_store/vector_index.pkl")
BM25_INDEX_PATH = Path("vector_store/bm25_index.pkl")

def get_embedding(text: str):
    response = mistral_client.embeddings.create(
        model="mistral-embed",
        inputs=[text]
    )
    return response.data[0].embedding

def build_index():
    records = extract_all()
    ids, embeddings, documents, metadatas = [], [], [], []
    tokenized_corpus = []
    counter = 0

    for record in records:
        chunks = chunk_text(record["text"])
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            embedding = get_embedding(chunk)
            ids.append(f"{record['source_doc']}_{i}")
            embeddings.append(embedding)
            documents.append(chunk)
            metadatas.append({
                "source_doc": record["source_doc"],
                "group": record["group"],
            })
            tokenized_corpus.append(chunk.lower().split())
            counter += 1
            time.sleep(0.2)

    if ids:
        with open(VECTOR_STORE_PATH, "wb") as f:
            pickle.dump({"ids": ids, "embeddings": embeddings, "documents": documents, "metadatas": metadatas}, f)

        bm25 = BM25Okapi(tokenized_corpus)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump({"bm25": bm25, "documents": documents, "metadatas": metadatas, "ids": ids}, f)

    print(f"Indexed {counter} chunks from {len(records)} documents.")
    print(f"Vector index saved to: {VECTOR_STORE_PATH}")
    print(f"BM25 index saved to: {BM25_INDEX_PATH}")

if __name__ == "__main__":
    build_index()