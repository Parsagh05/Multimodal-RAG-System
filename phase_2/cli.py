"""CLI interface for the Phase 2 Agent.

Provides an interactive command-line chat interface for testing
the agent without running the full API server.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

console = Console()
cli_app = typer.Typer(
    name="phase2-agent",
    help="Phase 2: Agentic AI on RAG - CLI Interface",
)


def create_agent(config_path: str = "config.yaml"):
    """Create and return an AgentCore instance."""
    from app.agent.core import AgentCore

    return AgentCore(config_path=config_path)


@cli_app.command()
def chat(
    config: str = typer.Option(
        "config.yaml", "--config", "-c", help="Path to config file"
    ),
    session_id: str = typer.Option(
        None, "--session", "-s", help="Session ID (auto-generated if not set)"
    ),
):
    """Start an interactive chat session with the Agent."""
    console.print(
        Panel(
            "[bold blue]Phase 2: Agentic AI on RAG[/bold blue]\n"
            "Interactive Chat Interface\n\n"
            "[dim]Type 'quit' or 'exit' to end the session.\n"
            "Type 'clear' to reset conversation history.\n"
            "Type 'info' to see session details.[/dim]",
            title="🤖 Product Assistant",
            border_style="blue",
        )
    )

    # Initialize agent
    with console.status("[bold green]Initializing agent..."):
        try:
            agent = create_agent(config_path=config)
        except Exception as e:
            console.print(f"[red]Error initializing agent: {e}[/red]")
            raise typer.Exit(1)

    chat_id = session_id or str(uuid.uuid4())
    console.print(f"[dim]Session ID: {chat_id}[/dim]\n")

    # Main chat loop
    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        user_input = user_input.strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input.lower() == "clear":
            agent.memory.clear_session(chat_id)
            console.print("[yellow]Conversation history cleared.[/yellow]\n")
            continue

        if user_input.lower() == "info":
            _show_session_info(agent, chat_id)
            continue

        # Process the message
        messages = [{"type": "text", "content": user_input}]

        with console.status("[bold cyan]Thinking..."):
            try:
                response = asyncio.run(
                    agent.process_message(chat_id=chat_id, user_messages=messages)
                )
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]\n")
                continue

        # Display response
        console.print()
        if response.message:
            console.print(
                Panel(
                    Markdown(response.message),
                    title="🤖 Assistant",
                    border_style="cyan",
                )
            )

        # Display product keys if present
        if response.base_random_keys:
            table = Table(title="Base Products Suggested")
            table.add_column("Key", style="green")
            for key in response.base_random_keys:
                table.add_row(key)
            console.print(table)

        if response.member_random_keys:
            table = Table(title="Member/Seller Products Suggested")
            table.add_column("Key", style="yellow")
            for key in response.member_random_keys:
                table.add_row(key)
            console.print(table)

        console.print()


@cli_app.command()
def serve(
    config: str = typer.Option(
        "config.yaml", "--config", "-c", help="Path to config file"
    ),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Server host"),
    port: int = typer.Option(8000, "--port", "-p", help="Server port"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Auto-reload"),
):
    """Start the FastAPI server."""
    import uvicorn

    os.environ["PHASE2_CONFIG"] = config
    console.print(
        f"[bold green]Starting server at http://{host}:{port}[/bold green]"
    )
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
    )


def _show_session_info(agent, chat_id: str) -> None:
    """Display current session information."""
    session = agent.memory.get_or_create_session(chat_id)
    table = Table(title="Session Info")
    table.add_column("Property", style="bold")
    table.add_column("Value")
    table.add_row("Chat ID", chat_id)
    table.add_row("Messages", str(len(session.messages)))
    table.add_row("Turn Count", str(session.turn_count))
    table.add_row("Has Summary", str(session.summary is not None))
    table.add_row("Context Keys", str(list(session.context.keys())))
    console.print(table)
    console.print()


if __name__ == "__main__":
    cli_app()
