import asyncio
from pathlib import Path

from llama_index.core.llms import ChatMessage

from src.interfaces.session import BaseChatSession
from src.interfaces.profile import BaseProfile, CHAT_DATA
from src.chat_model import ChatModel, ModelParams
from src.extractors import *


def ensure_profile_dirs(username: str) -> None:
    base: Path = CHAT_DATA / username
    (base / "histories").mkdir(parents=True, exist_ok=True)


def load_or_create_profile(username: str) -> BaseProfile:
    ensure_profile_dirs(username)
    profile_path: Path = CHAT_DATA / username / "profile"

    if profile_path.exists():
        return BaseProfile.load_profile(username)

    profile: BaseProfile = BaseProfile(username)
    profile.save_profile()
    return profile


def print_help() -> None:
    print(
        """
Commands:
  /help                     show this help
  /histories                list saved histories
  /new <name>               create and switch to a new history
  /load <name>              load an existing history
  /save                     save current history
  /stop                     stop generation
  /exit                     exit chat
"""
    )


async def chat_loop(session: BaseChatSession) -> None:
    print("Chat started. Type /help for commands.\n")

    while True:
        try:
            user_input: str = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts: list[str] = user_input.split(maxsplit=1)
            cmd: str = parts[0]

            if cmd == "/help":
                print_help()
            elif cmd == "/histories":
                for name in session.get_histories():
                    print("-", name)
            elif cmd == "/new" and len(parts) == 2:
                session.new_history(parts[1])
                print(f"New history created: {parts[1]}")
            elif cmd == "/load" and len(parts) == 2:
                session.load_history(parts[1])
                print(f"Loaded history: {parts[1]}")
            elif cmd == "/save":
                session.save_history()
                print("History saved")
            elif cmd == "/stop":
                session.stop_prompt()
                print("Generation stopped")
            elif cmd == "/exit":
                break
            else:
                print("Unknown command. Type /help")
                
            continue

        print("\nassistant > ", end="", flush=True)

        try:
            async for token in session.send_message()(user_input):
                print(token, end="", flush=True)
            print("\n")

        except asyncio.CancelledError:
            session.stop_prompt()
            print("\n[stopped]\n")


async def main() -> None:
    username: str = input("Username: ").strip()
    profile: BaseProfile = load_or_create_profile(username)

    model: ChatModel = ChatModel(make_default_router())
    session: BaseChatSession = BaseChatSession(model=model, user=profile)

    params: ModelParams = ModelParams(
        temperature=0.7, context_window=16_000, rag_top_k=4, 
        history_tokens=11_200, long_term_memory=True, long_term_tokens=2048, 
        top_k_memory=4
    )

    session.start_session(params)

    await chat_loop(session)
    
    session.end_session()


if __name__ == "__main__":
    asyncio.run(main())
