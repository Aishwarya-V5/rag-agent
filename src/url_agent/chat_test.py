from src.url_agent.agent import answer_from_urls

def main():
    print("URL RAG Agent — type 'exit' to quit\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        result = answer_from_urls(question)
        print(f"\nAgent: {result['answer']}")
        print(f"Used chunking: {result['used_chunking']}")
        if result["failed_urls"]:
            print(f"Failed URLs: {result['failed_urls']}")
        print()

if __name__ == "__main__":
    main()