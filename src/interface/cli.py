"""
AEON CLI - Command-line interface for Project AEON.

Provides slash commands for interacting with the BLACKWALL interface,
verification API, and mode control.

Usage:
    aeon                    # Start interactive mode
    aeon verify "H"         # Verify Hadamard gate is unitary
    aeon mode rigorous      # Switch to rigorous mode
    aeon status             # Show current status
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from .blackwall import BlackwallController, get_controller, ModeConfig
from .enums import InterfaceMode, RequestType
from .mode_detector import ModeDetector, analyze as analyze_input


# =============================================================================
# Console Setup
# =============================================================================

console = Console()


# =============================================================================
# CLI Group
# =============================================================================

@click.group(invoke_without_command=True)
@click.option("--mode", "-m", type=click.Choice(["rigorous", "exploratory", "hybrid"]),
              default="hybrid", help="Initial interface mode")
@click.option("--auto-transition/--no-auto-transition", default=True,
              help="Enable automatic mode transitions")
@click.pass_context
def main(ctx: click.Context, mode: str, auto_transition: bool) -> None:
    """
    AEON - Verified AI Physics Tutor

    A unified cognitive interface combining quantum verification,
    adaptive modes, and cryptographic proofs.
    """
    # Initialize controller
    interface_mode = InterfaceMode(mode)
    controller = BlackwallController(
        default_mode=interface_mode,
        auto_transition=auto_transition,
    )
    ctx.ensure_object(dict)
    ctx.obj["controller"] = controller
    ctx.obj["detector"] = ModeDetector()

    if ctx.invoked_subcommand is None:
        # No subcommand - show welcome and enter interactive mode
        _show_welcome(controller)
        _interactive_mode(controller, ctx.obj["detector"])


# =============================================================================
# Mode Commands
# =============================================================================

@main.command()
@click.argument("mode_name", type=click.Choice(["rigorous", "exploratory", "hybrid"]))
@click.pass_context
def mode(ctx: click.Context, mode_name: str) -> None:
    """
    Switch interface mode.

    \b
    Modes:
      rigorous    - Verified claims only, proof-backed responses
      exploratory - Creative hypotheses, intuition building
      hybrid      - Balanced, adapts based on context
    """
    controller = ctx.obj.get("controller") or get_controller()
    new_mode = InterfaceMode(mode_name)
    controller.set_mode(new_mode)

    console.print(f"[bold green]Mode switched to:[/bold green] {new_mode.value}")
    console.print(f"[dim]{new_mode.description}[/dim]")


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current BLACKWALL status."""
    controller = ctx.obj.get("controller") or get_controller()
    status_info = controller.get_status()

    # Create status table
    table = Table(title="BLACKWALL Status", show_header=False, box=None)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Current Mode", f"[bold]{status_info['current_mode']}[/bold]")
    table.add_row("Description", status_info["mode_description"])
    table.add_row("Auto-Transition", str(status_info["auto_transition"]))
    table.add_row("Transitions", str(status_info["transition_count"]))

    config = status_info["config"]
    table.add_row("", "")  # Spacer
    table.add_row("[bold]Configuration[/bold]", "")
    table.add_row("  Require Proof", str(config["require_proof"]))
    table.add_row("  Allow Speculation", str(config["allow_speculation"]))
    table.add_row("  Min Confidence", f"{config['min_confidence']:.0%}")
    table.add_row("  Format", config["format_preference"])

    console.print(Panel(table, border_style="blue"))


@main.command()
@click.pass_context
def history(ctx: click.Context) -> None:
    """Show mode transition history."""
    controller = ctx.obj.get("controller") or get_controller()
    transitions = controller.history

    if not transitions:
        console.print("[dim]No transitions recorded[/dim]")
        return

    table = Table(title="Mode Transition History")
    table.add_column("#", style="dim")
    table.add_column("From")
    table.add_column("To")
    table.add_column("Reason")
    table.add_column("Time", style="dim")

    for i, t in enumerate(transitions):
        table.add_row(
            str(i),
            t.from_mode.value,
            t.to_mode.value,
            t.reason.value,
            t.timestamp.strftime("%H:%M:%S"),
        )

    console.print(table)


# =============================================================================
# Verification Commands
# =============================================================================

@main.command()
@click.argument("expression")
@click.option("--type", "-t", "check_type",
              type=click.Choice(["gate", "state", "operator", "claim"]),
              default="gate", help="Type of verification")
@click.option("--format", "-f", "output_format",
              type=click.Choice(["plain", "markdown", "latex", "rich"]),
              default="rich", help="Output format")
@click.pass_context
def verify(ctx: click.Context, expression: str, check_type: str, output_format: str) -> None:
    """
    Verify a quantum expression.

    \b
    Examples:
      aeon verify H              # Verify Hadamard is unitary
      aeon verify "|0>" -t state # Verify |0> is normalized
      aeon verify X -t operator  # Verify Pauli-X is Hermitian
    """
    try:
        from src.tutor import TutorVerificationAPI, ExplanationGenerator, OutputFormat

        api = TutorVerificationAPI()
        generator = ExplanationGenerator()

        # Perform verification based on type
        if check_type == "gate":
            result = api.verify_gate(expression)
        elif check_type == "state":
            result = api.verify_state(expression)
        elif check_type == "operator":
            result = api.verify_operator(expression)
        else:  # claim
            result = api.verify_claim(expression)

        # Format output
        fmt = OutputFormat(output_format) if output_format != "rich" else OutputFormat.PLAIN
        if output_format == "rich":
            explanation = generator.to_rich(result, expression)
            console.print(Panel(explanation, title="Verification Result"))
        else:
            explanation = generator.generate(result, fmt, expression)
            console.print(explanation)

    except ImportError:
        console.print("[red]Error:[/red] Verification API not available. Run 'pip install -e .'")
    except Exception as e:
        console.print(f"[red]Verification error:[/red] {e}")


@main.command()
@click.argument("claim")
@click.pass_context
def prove(ctx: click.Context, claim: str) -> None:
    """
    Verify a natural language claim.

    \b
    Examples:
      aeon prove "Hadamard is unitary"
      aeon prove "Bell state is entangled"
    """
    try:
        from src.tutor import TutorVerificationAPI, ClaimParser, ExplanationGenerator

        parser = ClaimParser()
        api = TutorVerificationAPI()
        generator = ExplanationGenerator()

        # Parse the claim
        parsed = parser.parse(claim)
        console.print(f"[dim]Parsed as: {parsed.claim_type.value} claim about '{parsed.subject}'[/dim]")

        # Verify based on parsed type
        result = api.verify_claim(claim)

        # Show result
        explanation = generator.to_rich(result)
        console.print(Panel(explanation, title="Proof Result"))

    except ImportError:
        console.print("[red]Error:[/red] Verification API not available. Run 'pip install -e .'")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


# =============================================================================
# Analysis Commands
# =============================================================================

@main.command()
@click.argument("text")
@click.pass_context
def analyze(ctx: click.Context, text: str) -> None:
    """
    Analyze input text for mode detection.

    Shows what mode BLACKWALL would suggest for the given input.
    """
    detector = ctx.obj.get("detector") or ModeDetector()
    result = detector.analyze(text)

    table = Table(title="Input Analysis", show_header=False, box=None)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Suggested Mode", f"[bold]{result.suggested_mode.value}[/bold]")
    table.add_row("Request Type", result.request_type.value)
    table.add_row("Confidence", f"{result.confidence:.0%}")
    table.add_row("Reasoning", result.reasoning)

    if result.matched_patterns:
        patterns = "\n".join(f"  - {p[:40]}..." if len(p) > 40 else f"  - {p}"
                             for p in result.matched_patterns[:5])
        table.add_row("Matched Patterns", patterns)

    console.print(Panel(table, border_style="blue"))


# =============================================================================
# Help Commands
# =============================================================================

@main.command()
def commands() -> None:
    """List all available commands."""
    help_text = """
# AEON Commands

## Mode Control
- `/mode <rigorous|exploratory|hybrid>` - Switch interface mode
- `/status` - Show current BLACKWALL status
- `/history` - Show mode transition history

## Verification
- `/verify <expression>` - Verify a quantum expression
- `/prove <claim>` - Verify a natural language claim

## Analysis
- `/analyze <text>` - Analyze input for mode detection

## Modes

| Mode        | Description                              |
|-------------|------------------------------------------|
| rigorous    | Verified claims only, proof-backed       |
| exploratory | Creative hypotheses, intuition allowed   |
| hybrid      | Balanced, adapts based on context        |
"""
    console.print(Markdown(help_text))


# =============================================================================
# Interactive Mode
# =============================================================================

def _show_welcome(controller: BlackwallController) -> None:
    """Show welcome message."""
    welcome = f"""
[bold cyan]AEON[/bold cyan] - Adaptive Epistemological Ontology Network
[dim]Verified AI Physics Tutor[/dim]

Current Mode: [bold]{controller.current_mode.value}[/bold]
{controller.current_mode.description}

Type [bold]/help[/bold] for commands, [bold]/mode[/bold] to change mode, or enter a query.
"""
    console.print(Panel(welcome, title="Welcome", border_style="blue"))


def _interactive_mode(controller: BlackwallController, detector: ModeDetector) -> None:
    """Run interactive mode."""
    while True:
        try:
            # Show mode indicator in prompt
            mode_indicator = {
                InterfaceMode.RIGOROUS: "[R]",
                InterfaceMode.EXPLORATORY: "[E]",
                InterfaceMode.HYBRID: "[H]",
                InterfaceMode.SYSTEM: "[S]",
            }[controller.current_mode]

            user_input = console.input(f"[cyan]{mode_indicator}[/cyan] > ")
            user_input = user_input.strip()

            if not user_input:
                continue

            # Handle exit
            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                console.print("[dim]Goodbye![/dim]")
                break

            # Handle commands
            if user_input.startswith("/"):
                _handle_command(user_input, controller, detector)
                continue

            # Analyze input and maybe transition mode
            analysis = detector.analyze(user_input)
            if controller.auto_transition:
                did_transition, new_mode = controller.maybe_transition(analysis.request_type)
                if did_transition:
                    console.print(f"[dim]Auto-transitioned to {new_mode.value} mode[/dim]")

            # For now, just echo the analysis
            # In full implementation, this would dispatch to LLM
            console.print(f"[dim]Detected: {analysis.request_type.value} request[/dim]")
            console.print(f"[dim]Would process in {controller.current_mode.value} mode[/dim]")

        except KeyboardInterrupt:
            console.print("\n[dim]Use /exit to quit[/dim]")
        except EOFError:
            break


def _handle_command(command: str, controller: BlackwallController, detector: ModeDetector) -> None:
    """Handle a slash command in interactive mode."""
    parts = command.split(maxsplit=1)
    cmd = parts[0][1:].lower()  # Remove the slash
    args = parts[1] if len(parts) > 1 else ""

    if cmd == "help" or cmd == "?":
        help_text = """
Commands:
  /mode <mode>     - Switch to rigorous, exploratory, or hybrid
  /status          - Show current status
  /history         - Show transition history
  /verify <expr>   - Verify an expression
  /prove <claim>   - Verify a claim
  /analyze <text>  - Analyze input
  /exit            - Exit interactive mode
"""
        console.print(help_text)

    elif cmd == "mode":
        if args in ("rigorous", "exploratory", "hybrid"):
            controller.set_mode(InterfaceMode(args))
            console.print(f"[green]Mode set to {args}[/green]")
        else:
            console.print("[red]Usage: /mode <rigorous|exploratory|hybrid>[/red]")

    elif cmd == "status":
        status_info = controller.get_status()
        console.print(f"Mode: [bold]{status_info['current_mode']}[/bold]")
        console.print(f"Auto-transition: {status_info['auto_transition']}")
        console.print(f"Transitions: {status_info['transition_count']}")

    elif cmd == "history":
        for i, t in enumerate(controller.history[-5:]):  # Last 5
            console.print(f"  {i}: {t.from_mode.value} -> {t.to_mode.value} ({t.reason.value})")

    elif cmd == "verify":
        if not args:
            console.print("[red]Usage: /verify <expression>[/red]")
        else:
            try:
                from src.tutor import TutorVerificationAPI, ExplanationGenerator
                api = TutorVerificationAPI()
                result = api.verify_gate(args)
                gen = ExplanationGenerator()
                console.print(gen.to_rich(result, args))
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    elif cmd == "prove":
        if not args:
            console.print("[red]Usage: /prove <claim>[/red]")
        else:
            try:
                from src.tutor import TutorVerificationAPI
                api = TutorVerificationAPI()
                result = api.verify_claim(args)
                status = "[green]VERIFIED[/green]" if result.verified else "[red]NOT VERIFIED[/red]"
                console.print(f"{status}: {result.explanation}")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    elif cmd == "analyze":
        if not args:
            console.print("[red]Usage: /analyze <text>[/red]")
        else:
            result = detector.analyze(args)
            console.print(f"Mode: {result.suggested_mode.value}")
            console.print(f"Type: {result.request_type.value}")
            console.print(f"Confidence: {result.confidence:.0%}")

    elif cmd in ("exit", "quit"):
        raise EOFError()

    else:
        console.print(f"[red]Unknown command: /{cmd}[/red]")
        console.print("[dim]Type /help for available commands[/dim]")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
