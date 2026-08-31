from src.agent.agent import answer_ticket

def main():
    print("RAG Agent — type 'exit' to quit\n")
    history = []
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        result, history = answer_ticket(question, history)
        print(f"\nAgent: {result['answer']}")
        print(f"Sources: {[s['doc'] for s in result['sources']]}\n")

if __name__ == "__main__":
    main()