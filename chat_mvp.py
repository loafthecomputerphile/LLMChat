import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static
from textual.containers import Vertical
from textual.scroll_view import ScrollView
from textual.binding import Binding
from rich.markdown import Markdown
from rich.panel import Panel

from src.interfaces.session import BaseChatSession
from src.interfaces.profile import BaseProfile
from src.chat_model import ChatModel, ModelParams
from src.extractors import *


class ChatApp(App):
    CSS = """
    ScrollView {
        width: 100%;
        height: 1fr;  /* take available space in Vertical container */
    }

    Input {
        dock: bottom;
        height: 3;    /* fix height so it doesn’t expand */
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+s", "save", "Save history"),
        Binding("ctrl+n", "new", "New history"),
        Binding("ctrl+l", "load", "Load history"),
        Binding("ctrl+x", "stop", "Stop generation"),
    ]

    def __init__(self, session: BaseChatSession):
        super().__init__()
        self.session = session
        self.generating = False
        self.messages = []  # list of (role, text)
        self.panel_widgets = []  # mounted panels in ScrollView

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            self.scroll_view = ScrollView()
            yield self.scroll_view

        self.input = Input(placeholder="Type a message or /command")
        yield self.input
        yield Footer()

    async def on_input_submitted(self, event: Input.Submitted):
        text = event.value.strip()
        self.input.value = ""
        if not text or self.generating:
            return
        if text.startswith("/"):
            await self.handle_command(text)
            return
        await self.send_message(text)

    async def send_message(self, text: str):
        self.generating = True

        # Add user message
        await self.add_message("user", text)

        # Add Gilbert placeholder
        await self.add_message("gilbert", "")

        buffer = ""
        try:
            async for token in self.session.send_message()(text):
                buffer += token
                await self.update_last_gilbert(buffer)
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            await self.update_last_gilbert("[red][stopped][/red]")
        finally:
            self.generating = False

    async def add_message(self, role: str, text: str):
        panel = Panel(
            Markdown(text),
            title="You" if role == "user" else "Gilbert",
            border_style="green" if role == "user" else "magenta"
        )
        widget = Static(panel)  # Wrap panel in Static
        await self.scroll_view.mount(widget)
        self.panel_widgets.append((role, widget))
        self.scroll_view.scroll_end(animate=False)

    async def update_last_gilbert(self, text: str):
        role, widget = self.panel_widgets[-1]
        if role != "gilbert":
            return
        widget.update(Panel(Markdown(text), title="Gilbert", border_style="magenta"))
        await self.scroll_view.scroll_end(animate=False)

    async def handle_command(self, text: str):
        parts = text.split(maxsplit=1)
        cmd = parts[0]
        if cmd == "/help":
            await self.add_message("user", "/help /histories /new <name> /load <name> /save /stop /exit")
        elif cmd == "/exit":
            await self.action_quit()
        elif cmd == "/stop":
            self.session.stop_prompt()
            await self.update_last_gilbert("[red][stopped][/red]")
        else:
            await self.add_message("user", f"[red]Unknown command: {text}[/red]")


async def main():
    username = input("Username: ").strip()
    profile = BaseProfile.load_profile(username)

    model = ChatModel(make_default_router())
    session = BaseChatSession(model=model, user=profile)

    params = ModelParams(
        temperature=0.7,
        context_window=16_000,
        rag_top_k=4,
        history_tokens=11_200,
        long_term_memory=True,
        long_term_tokens=2048,
        top_k_memory=4,
    )

    session.start_session(params)

    app = ChatApp(session)
    await app.run_async()

    session.end_session()


if __name__ == "__main__":
    asyncio.run(main())
