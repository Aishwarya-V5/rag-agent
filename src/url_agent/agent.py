import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral
import time

from src.url_agent.fetch import fetch_all
from src.url_agent.context_builder import build_context
from src.ingest.chunk import chunk_text

load_dotenv()
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

SOURCES_PATH = Path(__file__).parent / "sources.yaml"


def load_urls():
    with open(SOURCES_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config.get("urls", [])


def embed_fn(text: str):
    response = client.embeddings.create(model="mistral-embed", inputs=[text])
    return response.data[0].embedding


def answer_from_urls(question: str):
    urls = load_urls()

    fetch_results = fetch_all(urls)

    context_text, used_chunking, failed_urls = build_context(
        fetch_results, question, embed_fn=embed_fn, chunk_fn=chunk_text, k=5
    )

    if not context_text.strip():
        return {
            "answer": "Could not retrieve usable content from any of the configured URLs.",
            "failed_urls": failed_urls,
            "used_chunking": used_chunking,
        }

    prompt = f"""Answer the question using ONLY the content below, which was fetched live from the listed sources.

For every claim, cite the source URL like this: [Source: <url>]

If the fetched content does not contain a clear, specific answer to the question, say so explicitly: "The fetched pages do not contain a specific answer to this question." Do NOT fill the gap with your own general knowledge unless you clearly and separately label it as: "General knowledge (not from the fetched pages):" — never blend it into the main answer.

Content:
{context_text}

Question: {question}
"""
    time.sleep(1.5)
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
                print(f"Rate limited, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise

    if answer is None:
        return {
            "answer": "Failed to get a response after multiple retries due to rate limiting.",
            "failed_urls": failed_urls,
            "used_chunking": used_chunking,
        }

    return {
        "answer": answer,
        "failed_urls": failed_urls,
        "used_chunking": used_chunking,
    }


if __name__ == "__main__":
    result = answer_from_urls("What does NetApp offer for AI workloads?")
    print(result["answer"])
    print(f"\nUsed chunking: {result['used_chunking']}")
    if result["failed_urls"]:
        print(f"Failed URLs: {result['failed_urls']}")