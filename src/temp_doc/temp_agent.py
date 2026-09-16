# import hashlib
# import os
# import time
# from typing import Any

# import numpy as np
# import streamlit as st
# from dotenv import load_dotenv
# from mistralai import Mistral

# from src.temp_doc.extract_temp import extract_temp_file
# from src.ingest.chunk import chunk_text


# # ============================================================
# # Configuration
# # ============================================================

# load_dotenv()

# MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# if not MISTRAL_API_KEY:
#     raise RuntimeError(
#         "MISTRAL_API_KEY is not set. "
#         "Please add it to your .env file."
#     )

# client = Mistral(api_key=MISTRAL_API_KEY)

# EMBEDDING_MODEL = "mistral-embed"
# CHAT_MODEL = "ministral-8b-2512"

# CHARS_PER_TOKEN = 4
# MODEL_CONTEXT_LIMIT_TOKENS = 28000
# MAX_CONTEXT_CHARS = MODEL_CONTEXT_LIMIT_TOKENS * CHARS_PER_TOKEN

# # Keep batches reasonably sized. Increase only if your API limits allow it.
# EMBEDDING_BATCH_SIZE = 250
# BATCH_DELAY_SECONDS = 0.5
# MAX_RETRIES = 5

# TEMP_CACHE_KEY = "temp_document_cache"


# # ============================================================
# # Persistent temporary-document cache for this Streamlit session
# # ============================================================

# def _get_temp_document_cache() -> dict[str, dict[str, Any]]:
#     """
#     Return the in-memory temporary-document cache.

#     Streamlit reruns the script for every interaction, but values kept in
#     st.session_state survive those reruns for the current browser session.
#     """
#     if TEMP_CACHE_KEY not in st.session_state:
#         st.session_state[TEMP_CACHE_KEY] = {}

#     return st.session_state[TEMP_CACHE_KEY]


# def _get_chat_id() -> str:
#     """
#     Use an existing chat ID when the application already has one.
#     Fall back to a session-level key when it does not.

#     The common key names below let this work without forcing changes to app.py.
#     """
#     for key in ("current_chat_id", "active_chat_id", "chat_id"):
#         value = st.session_state.get(key)
#         if value:
#             return str(value)

#     return "__streamlit_session__"


# def _get_file_hash(uploaded_file: Any) -> str:
#     """Create a stable hash from the uploaded file bytes."""
#     file_bytes = uploaded_file.getvalue()
#     return hashlib.sha256(file_bytes).hexdigest()


# def _get_cache_key(uploaded_file: Any) -> str:
#     """Create a cache key scoped to chat + uploaded document."""
#     chat_id = _get_chat_id()
#     file_hash = _get_file_hash(uploaded_file)
#     return f"{chat_id}:{file_hash}"


# # ============================================================
# # Rate-limit helpers
# # ============================================================

# def is_rate_limit_error(error: Exception) -> bool:
#     message = str(error).lower()

#     return (
#         "429" in message
#         or "rate_limited" in message
#         or "rate limit" in message
#         or "too many requests" in message
#     )


# # ============================================================
# # Single embedding
# # ============================================================

# def get_embedding(
#     text: str,
#     max_retries: int = MAX_RETRIES,
# ) -> list[float]:
#     """Create one embedding, normally for the user's question."""

#     if not text or not text.strip():
#         raise ValueError("Cannot generate an embedding for empty text.")

#     for attempt in range(max_retries):
#         try:
#             response = client.embeddings.create(
#                 model=EMBEDDING_MODEL,
#                 inputs=[text],
#             )

#             if not response.data:
#                 raise RuntimeError(
#                     "Mistral returned an empty embedding response."
#                 )

#             return response.data[0].embedding

#         except Exception as error:
#             if not is_rate_limit_error(error):
#                 raise

#             wait_time = 2 ** attempt
#             print(
#                 f"Embedding rate limited. "
#                 f"Retry {attempt + 1}/{max_retries} "
#                 f"in {wait_time}s..."
#             )
#             time.sleep(wait_time)

#     raise RuntimeError("Max retries exceeded for embedding request.")


# # ============================================================
# # Batch embeddings
# # ============================================================

# def get_embeddings(
#     texts: list[str],
#     batch_size: int = EMBEDDING_BATCH_SIZE,
#     max_retries: int = MAX_RETRIES,
# ) -> np.ndarray:
#     """Create embeddings for document chunks in batches."""

#     if not texts:
#         return np.empty((0, 0), dtype=np.float32)

#     if batch_size <= 0:
#         raise ValueError("batch_size must be greater than zero.")

#     cleaned_texts = [
#         text for text in texts if text and text.strip()
#     ]

#     if not cleaned_texts:
#         raise ValueError("No valid text chunks available for embedding.")

#     all_embeddings: list[list[float]] = []
#     total_chunks = len(cleaned_texts)

#     for start in range(0, total_chunks, batch_size):
#         batch = cleaned_texts[start:start + batch_size]
#         batch_end = start + len(batch)

#         print(
#             f"Embedding chunks {start + 1}-{batch_end} "
#             f"of {total_chunks}..."
#         )

#         batch_embeddings: list[list[float]] | None = None

#         for attempt in range(max_retries):
#             try:
#                 response = client.embeddings.create(
#                     model=EMBEDDING_MODEL,
#                     inputs=batch,
#                 )

#                 if not response.data:
#                     raise RuntimeError(
#                         "Mistral returned an empty embedding response."
#                     )

#                 ordered = sorted(
#                     response.data,
#                     key=lambda item: item.index,
#                 )

#                 batch_embeddings = [
#                     item.embedding for item in ordered
#                 ]

#                 if len(batch_embeddings) != len(batch):
#                     raise RuntimeError(
#                         "Number of embeddings returned by Mistral "
#                         "does not match number of input chunks."
#                     )

#                 break

#             except Exception as error:
#                 if not is_rate_limit_error(error):
#                     raise

#                 wait_time = 2 ** attempt
#                 print(
#                     f"Embedding batch rate limited. "
#                     f"Retry {attempt + 1}/{max_retries} "
#                     f"in {wait_time}s..."
#                 )
#                 time.sleep(wait_time)

#         if batch_embeddings is None:
#             raise RuntimeError(
#                 "Max retries exceeded for embedding batch."
#             )

#         all_embeddings.extend(batch_embeddings)

#         if batch_end < total_chunks:
#             time.sleep(BATCH_DELAY_SECONDS)

#     return np.asarray(all_embeddings, dtype=np.float32)


# # ============================================================
# # Prepare document ONCE and cache it
# # ============================================================

# def _prepare_temp_document(uploaded_file: Any) -> dict[str, Any]:
#     """
#     Extract, chunk, and embed the temporary document once.

#     The returned object is stored in st.session_state and reused for every
#     later question in the same chat/document session.
#     """

#     text, error = extract_temp_file(uploaded_file)

#     if error:
#         return {
#             "success": False,
#             "error": error,
#         }

#     if not text or not text.strip():
#         return {
#             "success": False,
#             "error": "The document does not contain readable text.",
#         }

#     print(
#         f"Processing temporary document: {len(text):,} characters"
#     )

#     # Small document: no chunking or document embeddings required.
#     if len(text) <= MAX_CONTEXT_CHARS:
#         print("Document fits context limit. No chunking required.")
#         return {
#             "success": True,
#             "text": text,
#             "chunks": [text],
#             "embeddings": None,
#             "used_chunking": False,
#         }

#     # Large document: chunk and embed ONCE.
#     chunks = chunk_text(text)

#     if not chunks:
#         return {
#             "success": False,
#             "error": "No chunks were created from the document.",
#         }

#     print(f"Document contains {len(chunks):,} chunks.")

#     embeddings = get_embeddings(
#         chunks,
#         batch_size=EMBEDDING_BATCH_SIZE,
#     )

#     print(f"Created embedding matrix: {embeddings.shape}")

#     return {
#         "success": True,
#         "text": text,
#         "chunks": chunks,
#         "embeddings": embeddings,
#         "used_chunking": True,
#     }


# def get_or_prepare_temp_document(uploaded_file: Any) -> dict[str, Any]:
#     """
#     Return the cached document for this chat + file.

#     This is the function that prevents chunking and document embedding from
#     happening again when the user asks the next question.
#     """

#     cache = _get_temp_document_cache()
#     cache_key = _get_cache_key(uploaded_file)

#     if cache_key in cache:
#         print("Temporary document loaded from session cache. No re-chunking.")
#         return cache[cache_key]

#     print("Temporary document not in cache. Processing it once...")

#     document_data = _prepare_temp_document(uploaded_file)

#     if document_data.get("success"):
#         cache[cache_key] = document_data

#     return document_data


# # ============================================================
# # Retrieve relevant context
# # ============================================================

# def retrieve_temp_context(
#     chunks: list[str],
#     embeddings: np.ndarray,
#     question: str,
#     k: int = 5,
# ) -> str:
#     """Embed only the question and search the already-cached chunk vectors."""

#     if not chunks or embeddings is None or embeddings.size == 0:
#         return ""

#     # IMPORTANT: the document chunks are NOT embedded here.
#     question_embedding = np.asarray(
#         get_embedding(question),
#         dtype=np.float32,
#     )

#     chunk_norms = np.linalg.norm(embeddings, axis=1)
#     question_norm = np.linalg.norm(question_embedding)

#     denominator = chunk_norms * question_norm
#     denominator[denominator == 0] = 1e-10

#     similarities = (
#         np.dot(embeddings, question_embedding) / denominator
#     )

#     k = min(k, len(chunks))

#     top_indices = np.argsort(similarities)[::-1][:k]

#     top_chunks = [chunks[index] for index in top_indices]

#     return "\n\n".join(top_chunks)


# # ============================================================
# # Answer question using cached document
# # ============================================================

# def answer_from_temp_doc(
#     uploaded_file: Any,
#     question: str,
# ) -> dict[str, Any]:
#     """
#     Main function used by app.py.

#     IMPORTANT BEHAVIOR:
#         First question for a document:
#             extract -> chunk -> embed -> cache -> answer

#         Later questions in the same chat/document:
#             cache -> embed question -> retrieve -> answer

#     Therefore the document is NOT re-chunked or re-embedded for every question.
#     """

#     if not question or not question.strip():
#         return {
#             "answer": "Please enter a question.",
#             "used_chunking": False,
#         }

#     # --------------------------------------------------------
#     # Get document from cache OR process it once
#     # --------------------------------------------------------

#     document_data = get_or_prepare_temp_document(uploaded_file)

#     if not document_data.get("success"):
#         return {
#             "answer": (
#                 f"Could not read this document: "
#                 f"{document_data.get('error', 'Unknown error')}"
#             ),
#             "used_chunking": False,
#         }

#     # --------------------------------------------------------
#     # Retrieve context
#     # --------------------------------------------------------

#     if not document_data["used_chunking"]:
#         context_text = document_data["text"]
#     else:
#         context_text = retrieve_temp_context(
#             chunks=document_data["chunks"],
#             embeddings=document_data["embeddings"],
#             question=question,
#             k=5,
#         )

#     if not context_text:
#         return {
#             "answer": "No relevant content was found in the document.",
#             "used_chunking": document_data["used_chunking"],
#         }

#     # --------------------------------------------------------
#     # Prompt
#     # --------------------------------------------------------

#     prompt = f"""
# Answer the question using ONLY the document content below.

# Rules:
# - Do not guess.
# - Do not use outside knowledge.
# - If the document does not contain enough information,
#   say so clearly.
# - Give a concise and accurate answer.
# - When possible, explain the answer using the document content.

# Document content:
# {context_text}

# Question:
# {question}
# """

#     # --------------------------------------------------------
#     # Generate answer
#     # --------------------------------------------------------

#     for attempt in range(MAX_RETRIES):
#         try:
#             response = client.chat.complete(
#                 model=CHAT_MODEL,
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": prompt,
#                     }
#                 ],
#                 temperature=0.2,
#             )

#             return {
#                 "answer": response.choices[0].message.content,
#                 "used_chunking": document_data["used_chunking"],
#             }

#         except Exception as error:
#             if not is_rate_limit_error(error):
#                 raise

#             wait_time = 2 ** attempt
#             print(
#                 f"Chat API rate limited. "
#                 f"Retry {attempt + 1}/{MAX_RETRIES} "
#                 f"in {wait_time}s..."
#             )
#             time.sleep(wait_time)

#     return {
#         "answer": (
#             "Failed to generate an answer because the Mistral API "
#             "is currently rate limited. Please wait and try again."
#         ),
#         "used_chunking": document_data["used_chunking"],
#     }


# # ============================================================
# # Optional utility: clear temporary document cache
# # ============================================================

# def clear_temp_document_cache() -> None:
#     """Clear cached temporary documents for the current Streamlit session."""
#     st.session_state.pop(TEMP_CACHE_KEY, None)



import hashlib
import os
import time
from typing import Any

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from mistralai import Mistral
from rank_bm25 import BM25Okapi

from src.temp_doc.extract_temp import extract_temp_file
from src.ingest.chunk import chunk_text


# ============================================================
# Configuration
# ============================================================

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY is not set. "
        "Please add it to your .env file."
    )

client = Mistral(api_key=MISTRAL_API_KEY)

EMBEDDING_MODEL = "mistral-embed"
CHAT_MODEL = "ministral-8b-2512"

CHARS_PER_TOKEN = 4
MODEL_CONTEXT_LIMIT_TOKENS = 28000
MAX_CONTEXT_CHARS = MODEL_CONTEXT_LIMIT_TOKENS * CHARS_PER_TOKEN

# Mistral free tier: mistral-embed is limited to ~1 request/second.
EMBEDDING_BATCH_SIZE = 128
BATCH_DELAY_SECONDS = 1.1
MAX_RETRIES = 5

TEMP_CACHE_KEY = "temp_document_cache"


# ============================================================
# Persistent temporary-document cache for this Streamlit session
# ============================================================

def _get_temp_document_cache() -> dict[str, dict[str, Any]]:
    """
    Return the in-memory temporary-document cache.

    Streamlit reruns the script for every interaction, but values kept in
    st.session_state survive those reruns for the current browser session.
    """
    if TEMP_CACHE_KEY not in st.session_state:
        st.session_state[TEMP_CACHE_KEY] = {}

    return st.session_state[TEMP_CACHE_KEY]


def _get_chat_id() -> str:
    """
    Use an existing chat ID when the application already has one.
    Fall back to a session-level key when it does not.
    """
    for key in ("current_chat_id", "active_chat_id", "chat_id"):
        value = st.session_state.get(key)
        if value:
            return str(value)

    return "__streamlit_session__"


def _get_file_hash(uploaded_file: Any) -> str:
    """Create a stable hash from the uploaded file bytes."""
    file_bytes = uploaded_file.getvalue()
    return hashlib.sha256(file_bytes).hexdigest()


def _get_cache_key(uploaded_file: Any) -> str:
    """Create a cache key scoped to chat + uploaded document."""
    chat_id = _get_chat_id()
    file_hash = _get_file_hash(uploaded_file)
    return f"{chat_id}:{file_hash}"


# ============================================================
# Rate-limit helpers
# ============================================================

def is_rate_limit_error(error: Exception) -> bool:
    message = str(error).lower()

    return (
        "429" in message
        or "rate_limited" in message
        or "rate limit" in message
        or "too many requests" in message
    )


# ============================================================
# Single embedding (Mistral)
# ============================================================

def get_embedding(
    text: str,
    max_retries: int = MAX_RETRIES,
) -> list[float]:
    """Create one embedding, normally for the user's question."""

    if not text or not text.strip():
        raise ValueError("Cannot generate an embedding for empty text.")

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                inputs=[text],
            )

            if not response.data:
                raise RuntimeError(
                    "Mistral returned an empty embedding response."
                )

            return response.data[0].embedding

        except Exception as error:
            if not is_rate_limit_error(error):
                raise

            wait_time = 2 ** attempt
            print(
                f"Embedding rate limited. "
                f"Retry {attempt + 1}/{max_retries} "
                f"in {wait_time}s..."
            )
            time.sleep(wait_time)

    raise RuntimeError("Max retries exceeded for embedding request.")


# ============================================================
# Batch embeddings (Mistral)
# ============================================================

def get_embeddings(
    texts: list[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
    max_retries: int = MAX_RETRIES,
) -> np.ndarray:
    """Create embeddings for document chunks in batches."""

    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    cleaned_texts = [
        text for text in texts if text and text.strip()
    ]

    if not cleaned_texts:
        raise ValueError("No valid text chunks available for embedding.")

    all_embeddings: list[list[float]] = []
    total_chunks = len(cleaned_texts)

    for start in range(0, total_chunks, batch_size):
        batch = cleaned_texts[start:start + batch_size]
        batch_end = start + len(batch)

        print(
            f"Embedding chunks {start + 1}-{batch_end} "
            f"of {total_chunks}..."
        )

        batch_embeddings: list[list[float]] | None = None

        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    inputs=batch,
                )

                if not response.data:
                    raise RuntimeError(
                        "Mistral returned an empty embedding response."
                    )

                ordered = sorted(
                    response.data,
                    key=lambda item: item.index,
                )

                batch_embeddings = [
                    item.embedding for item in ordered
                ]

                if len(batch_embeddings) != len(batch):
                    raise RuntimeError(
                        "Number of embeddings returned by Mistral "
                        "does not match number of input chunks."
                    )

                break

            except Exception as error:
                if not is_rate_limit_error(error):
                    raise

                wait_time = 2 ** attempt
                print(
                    f"Embedding batch rate limited. "
                    f"Retry {attempt + 1}/{max_retries} "
                    f"in {wait_time}s..."
                )
                time.sleep(wait_time)

        if batch_embeddings is None:
            raise RuntimeError(
                "Max retries exceeded for embedding batch."
            )

        all_embeddings.extend(batch_embeddings)

        if batch_end < total_chunks:
            time.sleep(BATCH_DELAY_SECONDS)

    return np.asarray(all_embeddings, dtype=np.float32)


# ============================================================
# Prepare document ONCE and cache it (now builds BM25 index too)
# ============================================================

def _prepare_temp_document(uploaded_file: Any) -> dict[str, Any]:
    """
    Extract, chunk, embed, and build a BM25 index for the temporary document
    once. The returned object is stored in st.session_state and reused for
    every later question in the same chat/document session.
    """

    text, error = extract_temp_file(uploaded_file)

    if error:
        return {
            "success": False,
            "error": error,
        }

    if not text or not text.strip():
        return {
            "success": False,
            "error": "The document does not contain readable text.",
        }

    print(
        f"Processing temporary document: {len(text):,} characters"
    )

    # Small document: no chunking, no embeddings, no BM25 required.
    if len(text) <= MAX_CONTEXT_CHARS:
        print("Document fits context limit. No chunking required.")
        return {
            "success": True,
            "text": text,
            "chunks": [text],
            "embeddings": None,
            "bm25": None,
            "used_chunking": False,
        }

    # Large document: chunk, embed, and build BM25 ONCE.
    chunks = chunk_text(text)

    if not chunks:
        return {
            "success": False,
            "error": "No chunks were created from the document.",
        }

    print(f"Document contains {len(chunks):,} chunks.")

    embeddings = get_embeddings(
        chunks,
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    print(f"Created embedding matrix: {embeddings.shape}")

    tokenized_corpus = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    print("Built in-memory BM25 index for hybrid search.")

    return {
        "success": True,
        "text": text,
        "chunks": chunks,
        "embeddings": embeddings,
        "bm25": bm25,
        "used_chunking": True,
    }


def get_or_prepare_temp_document(uploaded_file: Any) -> dict[str, Any]:
    """
    Return the cached document for this chat + file.

    This is the function that prevents chunking, embedding, and BM25
    indexing from happening again when the user asks the next question.
    """

    cache = _get_temp_document_cache()
    cache_key = _get_cache_key(uploaded_file)

    if cache_key in cache:
        print("Temporary document loaded from session cache. No re-processing.")
        return cache[cache_key]

    print("Temporary document not in cache. Processing it once...")

    document_data = _prepare_temp_document(uploaded_file)

    if document_data.get("success"):
        cache[cache_key] = document_data

    return document_data


# ============================================================
# Retrieve relevant context — hybrid: vector + BM25 (RRF merge)
# ============================================================

def retrieve_temp_context(
    chunks: list[str],
    embeddings: np.ndarray,
    bm25: Any,
    question: str,
    k: int = 5,
) -> str:
    """
    Embed only the question and search the already-cached chunk vectors
    (vector similarity) and the already-built BM25 index (keyword search),
    then merge both rankings using Reciprocal Rank Fusion.
    """

    if not chunks or embeddings is None or embeddings.size == 0:
        return ""

    # --- Vector search ---
    question_embedding = np.asarray(
        get_embedding(question),
        dtype=np.float32,
    )

    chunk_norms = np.linalg.norm(embeddings, axis=1)
    question_norm = np.linalg.norm(question_embedding)

    denominator = chunk_norms * question_norm
    denominator[denominator == 0] = 1e-10

    similarities = (
        np.dot(embeddings, question_embedding) / denominator
    )

    vector_ranks = np.argsort(similarities)[::-1]

    # --- BM25 keyword search ---
    if bm25 is not None:
        tokenized_query = question.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_ranks = np.argsort(bm25_scores)[::-1]
    else:
        bm25_ranks = vector_ranks  # fallback: no BM25 available

    # --- Reciprocal Rank Fusion merge ---
    rrf_scores: dict[int, float] = {}
    K = 60

    for rank, idx in enumerate(vector_ranks):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (K + rank + 1)

    for rank, idx in enumerate(bm25_ranks):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (K + rank + 1)

    k = min(k, len(chunks))

    top_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]

    top_chunks = [chunks[index] for index in top_indices]

    return "\n\n".join(top_chunks)


# ============================================================
# Answer question using cached document
# ============================================================

def answer_from_temp_doc(
    uploaded_file: Any,
    question: str,
) -> dict[str, Any]:
    """
    Main function used by app.py.

    IMPORTANT BEHAVIOR:
        First question for a document:
            extract -> chunk -> embed -> build BM25 -> cache -> answer

        Later questions in the same chat/document:
            cache -> embed question -> hybrid retrieve -> answer

    Therefore the document is NOT re-chunked, re-embedded, or re-indexed
    for every question.
    """

    if not question or not question.strip():
        return {
            "answer": "Please enter a question.",
            "used_chunking": False,
        }

    # --------------------------------------------------------
    # Get document from cache OR process it once
    # --------------------------------------------------------

    document_data = get_or_prepare_temp_document(uploaded_file)

    if not document_data.get("success"):
        return {
            "answer": (
                f"Could not read this document: "
                f"{document_data.get('error', 'Unknown error')}"
            ),
            "used_chunking": False,
        }

    # --------------------------------------------------------
    # Retrieve context
    # --------------------------------------------------------

    if not document_data["used_chunking"]:
        context_text = document_data["text"]
    else:
        context_text = retrieve_temp_context(
            chunks=document_data["chunks"],
            embeddings=document_data["embeddings"],
            bm25=document_data["bm25"],
            question=question,
            k=5,
        )

    if not context_text:
        return {
            "answer": "No relevant content was found in the document.",
            "used_chunking": document_data["used_chunking"],
        }

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
Answer the question using ONLY the document content below.

Rules:
- Do not guess.
- Do not use outside knowledge.
- If the document does not contain enough information,
  say so clearly.
- Give a concise and accurate answer.
- When possible, explain the answer using the document content.

Document content:
{context_text}

Question:
{question}
"""

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.complete(
                model=CHAT_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.2,
            )

            return {
                "answer": response.choices[0].message.content,
                "used_chunking": document_data["used_chunking"],
            }

        except Exception as error:
            if not is_rate_limit_error(error):
                raise

            wait_time = 2 ** attempt
            print(
                f"Chat API rate limited. "
                f"Retry {attempt + 1}/{MAX_RETRIES} "
                f"in {wait_time}s..."
            )
            time.sleep(wait_time)

    return {
        "answer": (
            "Failed to generate an answer because the Mistral API "
            "is currently rate limited. Please wait and try again."
        ),
        "used_chunking": document_data["used_chunking"],
    }


# ============================================================
# Optional utility: clear temporary document cache
# ============================================================

def clear_temp_document_cache() -> None:
    """Clear cached temporary documents for the current Streamlit session."""
    st.session_state.pop(TEMP_CACHE_KEY, None)