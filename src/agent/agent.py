import os
import time
from dotenv import load_dotenv
from mistralai import Mistral
from src.retrieval.retriever import retrieve, retrieve_all_in_group

load_dotenv()
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

MAX_RETRIES = 3
MODELS_TO_TRY = ["open-mistral-nemo", "ministral-8b-2512"]


def is_rate_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "429" in message
        or "rate_limited" in message
        or "rate limit" in message
        or "too many requests" in message
        or "backend_out_of_capacity" in message
        or "not enough capacity" in message
    )


def answer_ticket(question: str, history: list = None, group_filter: str = None):
    if history is None:
        history = []

    retrieval_query = question
    if history:
        last_user_msgs = [m["content"] for m in history if m["role"] == "user"]
        if last_user_msgs:
            retrieval_query = last_user_msgs[-1] + " " + question

    list_keywords = [
        "list all",
        "list the",
        "what are all",
        "every",
        "each of the",
    ]

    is_list_query = any(kw in question.lower() for kw in list_keywords)

    if is_list_query and group_filter:
        all_results = retrieve_all_in_group(group_filter)
        if len(all_results) <= 150:
            results = all_results
        else:
            results = retrieve(retrieval_query, k=50, group_filter=group_filter)
    else:
        results = retrieve(retrieval_query, k=10, group_filter=group_filter)

    if not results:
        return {"answer": "No relevant documentation found.", "sources": []}, history

    context = "\n\n".join([f"[{meta['source_doc']}]: {chunk}" for chunk, meta in results])

    prompt = f"""Answer the question using ONLY the context below.
Do not guess or use outside knowledge. If the context doesn't answer the question, say so clearly.

Context:
{context}

Question: {question}
"""

    messages = history + [{"role": "user", "content": prompt}]

    answer = None
    last_error = None

    for model in MODELS_TO_TRY:
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.complete(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                )
                answer = response.choices[0].message.content
                break

            except Exception as error:
                last_error = error
                print(f"[{model}] ERROR: {error}")  # ← add this
                if not is_rate_limit_error(error):
                    raise
                wait_time = 2 ** attempt
                print(f"[{model}] busy/rate limited. Retry {attempt + 1}/{MAX_RETRIES} in {wait_time}s...")
                time.sleep(wait_time)

        if answer is not None:
            break
        else:
            print(f"[{model}] exhausted retries, trying next model if available...")

    if answer is None:
        answer = (
            "The AI service is currently busy or rate limited across all available models. "
            "Please wait a moment and try asking your question again."
        )

    updated_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]

    sources = [{"doc": meta["source_doc"], "group": meta["group"]} for _, meta in results]

    return {"answer": answer, "sources": sources}, updated_history


if __name__ == "__main__":
    result, history = answer_ticket("How do I create a profile on the Broadcom support portal?")
    print(result["answer"])
    print("\nSources:", result["sources"])