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
CHECKPOINT_PATH = Path("vector_store/checkpoint.jsonl")  # JSONL now, not JSON

BATCH_SIZE = 128

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

def load_completed_ids():
    completed = set()
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    completed.add(entry["id"])
                except json.JSONDecodeError:
                    continue  # skip corrupted last line, if any
    return completed

def append_checkpoint(chunk_id, embedding, text, metadata):
    with open(CHECKPOINT_PATH, "a") as f:
        f.write(json.dumps({
            "id": chunk_id,
            "embedding": embedding,
            "text": text,
            "metadata": metadata
        }) + "\n")

def build_index():
    records = extract_all()

    all_chunks = []
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

    completed_ids = load_completed_ids()
    remaining = [c for c in all_chunks if c[0] not in completed_ids]

    print(f"Total chunks: {len(all_chunks)} | Already done: {len(completed_ids)} | Remaining: {len(remaining)}")

    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i:i + BATCH_SIZE]
        batch_ids = [c[0] for c in batch]
        batch_texts = [c[1] for c in batch]
        batch_metas = [c[2] for c in batch]

        embeddings = get_embeddings_batch(batch_texts)

        for cid, emb, txt, meta in zip(batch_ids, embeddings, batch_texts, batch_metas):
            append_checkpoint(cid, emb, txt, meta)

        done_so_far = len(completed_ids) + i + len(batch)
        print(f"Progress: {done_so_far}/{len(all_chunks)} chunks embedded")

    # Build final index files by reading the full checkpoint
    ids, embeddings, documents, metadatas = [], [], [], []
    with open(CHECKPOINT_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ids.append(entry["id"])
                embeddings.append(entry["embedding"])
                documents.append(entry["text"])
                metadatas.append(entry["metadata"])
            except json.JSONDecodeError:
                continue

    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump({"ids": ids, "embeddings": embeddings, "documents": documents, "metadatas": metadatas}, f)

    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "documents": documents, "metadatas": metadatas, "ids": ids}, f)

    print(f"\nDone. Indexed {len(ids)} chunks total.")
    print(f"Vector index: {VECTOR_STORE_PATH}")
    print(f"BM25 index: {BM25_INDEX_PATH}")

if __name__ == "__main__":
    build_index()