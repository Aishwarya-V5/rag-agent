import os
from dotenv import load_dotenv
from mistralai import Mistral
from src.retrieval.retriever import retrieve

load_dotenv()
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

def answer_ticket(question: str, history: list = None):
    if history is None:
        history = []

    results = retrieve(question, k=5)

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

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=messages,
        temperature=0.2,
    )

    answer = response.choices[0].message.content
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