import os
import time
import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral

from src.temp_doc.extract_temp import extract_temp_file
from src.ingest.chunk import chunk_text

load_dotenv()
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

CHARS_PER_TOKEN = 4
MODEL_CONTEXT_LIMIT_TOKENS = 28000
MAX_CONTEXT_CHARS = MODEL_CONTEXT_LIMIT_TOKENS * CHARS_PER_TOKEN


def get_embedding(text: str, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(model="mistral-embed", inputs=[text])
            return response.data[0].embedding
        except Exception as e:
            if "429" in str(e) or "rate_limited" in str(e):
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                raise
    raise Exception("Max retries exceeded for embedding request")


def build_temp_context(text: str, question: str, k: int = 5):
    if len(text) <= MAX_CONTEXT_CHARS:
        return text, False

    chunks = chunk_text(text)
    chunk_embeddings = [get_embedding(c) for c in chunks]
    question_embedding = get_embedding(question)

    embeddings_matrix = np.array(chunk_embeddings)
    query_vec = np.array(question_embedding)
    norms = np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(query_vec)
    norms[norms == 0] = 1e-10
    sims = np.dot(embeddings_matrix, query_vec) / norms

    top_indices = np.argsort(sims)[::-1][:k]
    top_chunks = [chunks[i] for i in top_indices]

    return "\n\n".join(top_chunks), True


def answer_from_temp_doc(uploaded_file, question: str):
    text, error = extract_temp_file(uploaded_file)

    if error:
        return {"answer": f"Could not read this document: {error}", "used_chunking": False}

    context_text, used_chunking = build_temp_context(text, question)

    prompt = f"""Answer the question using ONLY the document content below.
Do not guess or use outside knowledge. If the content doesn't answer the question, say so clearly.

Document content:
{context_text}

Question: {question}
"""

    max_retries = 5
    answer = None
    for attempt in range(max_retries):
        try:
            response = client.chat.complete(
                model="ministral-8b-2512",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            answer = response.choices[0].message.content
            break
        except Exception as e:
            if "429" in str(e) or "rate_limited" in str(e):
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                raise

    if answer is None:
        answer = "Failed to get a response after multiple retries due to rate limiting."

    return {"answer": answer, "used_chunking": used_chunking}