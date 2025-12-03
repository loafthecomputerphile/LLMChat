from src.chat_model import ChatModel, ModelParams
from src.extractors import *
from src import variables


def initialize_model() -> ChatModel:
    
    router: ExtractionRouter = ExtractionRouter()
    
    model: ChatModel = ChatModel(router)

    params: ModelParams = ModelParams(
        temperature=0.7, context_window=16_000, rag_top_k=4, 
        history_tokens=11_200, long_term_memory=True, long_term_tokens=2048, 
        top_k_memory=4
    ) 
    
    model.load_parameters(params)
    model.load_model(variables.BASE_MODEL)
    
    return model


def main() -> None:
    result: str
    user_input: str
    model: ChatModel = initialize_model()
    
    print("\n🟢 Chat started. Type 'exit' to quit.\n")
    
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting…")
            model.kill()
            break

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            model.kill()
            break

        try:
            result = str(model.prompt(user_input))
            print(f"\nAI: {result}\n")
        except Exception as e:
            print("⚠️ Error:", e)
            model.kill()
            
            
if __name__ == "__main__":
    main()