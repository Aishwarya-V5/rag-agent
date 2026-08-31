from src.agent.agent import answer_ticket

def main():
    print("RAG Agent — type 'exit' to quit\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        result = answer_ticket(question)
        print(f"\nAgent: {result['answer']}")
        print(f"Sources: {[s['doc'] for s in result['sources']]}\n")

if __name__ == "__main__":
    main()