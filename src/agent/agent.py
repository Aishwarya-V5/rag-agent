import os
from dotenv import load_dotenv
from mistralai import Mistral
from src.retrieval.retriever import retrieve

load_dotenv()
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

def answer_ticket(question: str):
    results = retrieve(question, k=5)

    if not results:
        return {"answer": "No relevant documentation found.", "sources": []}

    context = "\n\n".join([f"[{meta['source_doc']}]: {chunk}" for chunk, meta in results])

    prompt = f"""Answer the question using ONLY the context below.
Do not guess or use outside knowledge. If the context doesn't answer the question, say so clearly.

Context:
{context}

Question: {question}
"""

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    sources = [{"doc": meta["source_doc"], "group": meta["group"]} for _, meta in results]

    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
    }

if __name__ == "__main__":
    result = answer_ticket("How do I create a profile on the Broadcom support portal?")
    print(result["answer"])
    print("\nSources:", result["sources"])