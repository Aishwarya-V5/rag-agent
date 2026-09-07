import numpy as np

CHARS_PER_TOKEN = 4
MODEL_CONTEXT_LIMIT_TOKENS = 28000
MAX_CONTEXT_CHARS = MODEL_CONTEXT_LIMIT_TOKENS * CHARS_PER_TOKEN


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def build_context(fetch_results: list, question: str, embed_fn, chunk_fn, k: int = 5):
    successful = [r for r in fetch_results if r["success"]]
    failed_urls = [r["url"] for r in fetch_results if not r["success"]]

    combined_text = "\n\n".join([f"[Source: {r['url']}]\n{r['content']}" for r in successful])
    total_chars = len(combined_text)

    if total_chars <= MAX_CONTEXT_CHARS:
        return combined_text, False, failed_urls

    all_chunks = []
    for r in successful:
        chunks = chunk_fn(r["content"])
        for c in chunks:
            all_chunks.append((r["url"], c))

    chunk_texts = [c[1] for c in all_chunks]
    chunk_embeddings = [embed_fn(c) for c in chunk_texts]
    question_embedding = embed_fn(question)

    embeddings_matrix = np.array(chunk_embeddings)
    query_vec = np.array(question_embedding)
    norms = np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(query_vec)
    norms[norms == 0] = 1e-10
    sims = np.dot(embeddings_matrix, query_vec) / norms

    top_indices = np.argsort(sims)[::-1][:k]
    top_chunks = [all_chunks[i] for i in top_indices]

    context_text = "\n\n".join([f"[Source: {url}]\n{text}" for url, text in top_chunks])

    return context_text, True, failed_urls