import pickle
import json
from pathlib import Path

VECTOR_STORE_PATH = Path("vector_store/vector_index.pkl")
BM25_INDEX_PATH = Path("vector_store/bm25_index.pkl")
CHECKPOINT_PATH = Path("vector_store/checkpoint.jsonl")


def remove_document(source_doc_name: str):
    # --- Clean checkpoint.jsonl ---
    if CHECKPOINT_PATH.exists():
        kept_lines = []
        removed_count = 0
        skipped_count = 0
        with open(CHECKPOINT_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    # Same defensive handling as embed_and_store.py:
                    # a corrupted/malformed line is dropped rather than
                    # crashing the whole removal.
                    skipped_count += 1
                    continue
                if entry["metadata"]["source_doc"] == source_doc_name:
                    removed_count += 1
                else:
                    kept_lines.append(line)

        with open(CHECKPOINT_PATH, "w") as f:
            for line in kept_lines:
                f.write(line + "\n")

        print(f"Removed {removed_count} chunks from checkpoint.jsonl")
        if skipped_count:
            print(f"Skipped {skipped_count} corrupted/unreadable line(s) in checkpoint.jsonl")

    # --- Rebuild vector_index.pkl and bm25_index.pkl from cleaned checkpoint ---
    ids, embeddings, documents, metadatas = [], [], [], []
    with open(CHECKPOINT_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            ids.append(entry["id"])
            embeddings.append(entry["embedding"])
            documents.append(entry["text"])
            metadatas.append(entry["metadata"])

    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump({"ids": ids, "embeddings": embeddings, "documents": documents, "metadatas": metadatas}, f)

    from rank_bm25 import BM25Okapi
    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "documents": documents, "metadatas": metadatas, "ids": ids}, f)

    print(f"Rebuilt indexes: {len(ids)} chunks remaining total.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest.remove_document <exact_source_doc_filename>")
    else:
        remove_document(sys.argv[1])