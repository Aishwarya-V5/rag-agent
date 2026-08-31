import os
import time
import pickle
import json
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
CHECKPOINT_PATH = Path("vector_store/checkpoint.json")

BATCH_SIZE = 20  # chunks per API call

def get_embeddings_batch(texts: list, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = mistral_client.embeddings.create(
                model="mistral-embed",
                inputs=texts
            )
            return [d.embedding for d in response.data]
        except Exception as e:
            if "429" in str(e) or "rate_limited" in str(e):
                wait_time = 2 ** attempt
                print(f"Rate limited, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise
    raise Exception("Max retries exceeded for embedding batch")

def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return {"completed_chunk_ids": [], "ids": [], "embeddings": [], "documents": [], "metadatas": []}

def save_checkpoint(state):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(state, f)

def build_index():
    records = extract_all()

    all_chunks = []  # list of (chunk_id, chunk_text, metadata)
    for record in records:
        chunks = chunk_text(record["text"])
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            chunk_id = f"{record['source_doc']}_p{record.get('page')}_{i}"
            metadata = {
                "source_doc": record["source_doc"],
                "group": record["group"],
                "page": record.get("page"),
            }
            all_chunks.append((chunk_id, chunk, metadata))

    state = load_checkpoint()
    completed_ids = set(state["completed_chunk_ids"])
    remaining = [c for c in all_chunks if c[0] not in completed_ids]

    print(f"Total chunks: {len(all_chunks)} | Already done: {len(completed_ids)} | Remaining: {len(remaining)}")

    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i:i + BATCH_SIZE]
        batch_ids = [c[0] for c in batch]
        batch_texts = [c[1] for c in batch]
        batch_metas = [c[2] for c in batch]

        embeddings = get_embeddings_batch(batch_texts)

        state["ids"].extend(batch_ids)
        state["embeddings"].extend(embeddings)
        state["documents"].extend(batch_texts)
        state["metadatas"].extend(batch_metas)
        state["completed_chunk_ids"].extend(batch_ids)

        save_checkpoint(state)
        print(f"Progress: {len(state['completed_chunk_ids'])}/{len(all_chunks)} chunks embedded")
        time.sleep(1)  # brief pause between batches

    # Final save to actual index files
    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump({
            "ids": state["ids"],
            "embeddings": state["embeddings"],
            "documents": state["documents"],
            "metadatas": state["metadatas"],
        }, f)

    tokenized_corpus = [doc.lower().split() for doc in state["documents"]]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "documents": state["documents"],
            "metadatas": state["metadatas"],
            "ids": state["ids"],
        }, f)

    print(f"\nDone. Indexed {len(state['ids'])} chunks from {len(records)} extracted pages/docs.")
    print(f"Vector index: {VECTOR_STORE_PATH}")
    print(f"BM25 index: {BM25_INDEX_PATH}")

if __name__ == "__main__":
    build_index()