"""Recon CLI - Command-line interface for research pipelines.

Built with Typer + Rich for a clean terminal experience.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from recon import __version__
from recon.config import Depth, load_plan, create_plan_from_topic

app = typer.Typer(
    name="recon",
    help="Verified research pipelines powered by AI agents.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

TEMPLATES_DIR = Path(__file__).parent / "templates"


@app.command()
def run(
    plan_file: Annotated[
        Optional[Path],
        typer.Argument(help="Path to plan.yaml file"),
    ] = None,
    topic: Annotated[
        Optional[str],
        typer.Option("--topic", "-t", help="Research topic (inline mode)"),
    ] = None,
    depth: Annotated[
        str,
        typer.Option("--depth", "-d", help="Research depth: quick, standard, deep"),
    ] = "standard",
    verify: Annotated[
        bool,
        typer.Option("--verify/--no-verify", help="Enable/disable fact-checking"),
    ] = True,
    provider: Annotated[
        Optional[str],
        typer.Option("--provider", "-p", help="LLM provider"),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Model name"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose output"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Parse and validate plan without executing"),
    ] = False,
) -> None:
    """Run a research pipeline from a plan file or inline topic."""
    # Resolve plan
    if plan_file and topic:
        console.print("[red]Error:[/] Cannot specify both a plan file and --topic. Choose one.")
        raise typer.Exit(code=1)

    if not plan_file and not topic:
        console.print("[red]Error:[/] Provide a plan file or use --topic.")
        raise typer.Exit(code=1)

    try:
        if plan_file:
            plan = load_plan(plan_file)
        else:
            assert topic is not None
            plan = create_plan_from_topic(
                topic=topic,
                depth=depth,
                verify=verify,
                provider=provider,
                model=model,
            )
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[red]Validation error:[/] {e}")
        raise typer.Exit(code=1) from e

    # Display plan summary
    investigations = plan.get_investigations()

    panel_content = (
        f"[bold]Topic:[/] {plan.topic}\n"
        f"[bold]Depth:[/] {plan.depth.value} ({len(investigations)} researchers)\n"
        f"[bold]Model:[/] {plan.model} via {plan.provider}\n"
        f"[bold]Verify:[/] {'yes' if plan.verify else 'no'}\n"
        f"[bold]Search:[/] {plan.search.provider}"
    )
    if plan.focus:
        panel_content += f"\n[bold]Focus:[/] {plan.focus}"

    console.print(Panel(panel_content, title="Recon - Research Pipeline", border_style="blue"))

    if verbose:
        table = Table(title="Investigation Agents")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Questions", style="dim")
        for inv in investigations:
            table.add_row(inv.id, inv.name, str(len(inv.questions)))
        console.print(table)

    if dry_run:
        console.print("\n[yellow]Dry run:[/] Plan is valid. No execution.")
        raise typer.Exit(code=0)

    # Execute pipeline
    console.print("\n[bold]Starting research pipeline...[/]\n")

    try:
        from recon.flow_builder import build_and_run

        build_and_run(plan, verbose=verbose, console=console)
    except Exception as e:
        console.print(f"\n[red]Pipeline failed:[/] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=4) from e

    console.print("\n[green]Research complete.[/]")


@app.command()
def init(
    template: Annotated[
        str,
        typer.Option("--template", "-t", help="Template name"),
    ] = "market-research",
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path"),
    ] = Path("plan.yaml"),
) -> None:
    """Initialize a new plan file from a template."""
    template_file = TEMPLATES_DIR / f"{template}.yaml"

    if not template_file.exists():
        console.print(f"[red]Error:[/] Template '{template}' not found.")
        console.print(f"Available templates: {', '.join(t.stem for t in TEMPLATES_DIR.glob('*.yaml'))}")
        raise typer.Exit(code=1)

    if output.exists():
        overwrite = typer.confirm(f"{output} already exists. Overwrite?")
        if not overwrite:
            raise typer.Exit(code=0)

    shutil.copy(template_file, output)
    console.print(f"[green]Created:[/] {output} (from template '{template}')")
    console.print(f"Edit the file with your topic, then run: [bold]recon run {output}[/]")


@app.command()
def templates() -> None:
    """List available plan templates."""
    table = Table(title="Available Templates")
    table.add_column("Name", style="cyan")
    table.add_column("File", style="dim")

    template_files = sorted(TEMPLATES_DIR.glob("*.yaml"))

    if not template_files:
        console.print("[yellow]No templates found.[/]")
        raise typer.Exit(code=0)

    for t in template_files:
        table.add_row(t.stem, t.name)

    console.print(table)
    console.print("\nUsage: [bold]recon init --template <name>[/]")


@app.command()
def verify(
    research_dir: Annotated[
        Path,
        typer.Argument(help="Directory with research markdown files to verify"),
    ] = Path("./research"),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for verification report"),
    ] = Path("./verification"),
) -> None:
    """Run fact-checking on existing research files."""
    if not research_dir.exists():
        console.print(f"[red]Error:[/] Research directory not found: {research_dir}")
        raise typer.Exit(code=1)

    md_files = list(research_dir.glob("*.md"))
    if not md_files:
        console.print(f"[yellow]No markdown files found in {research_dir}[/]")
        raise typer.Exit(code=0)

    console.print(f"Found {len(md_files)} research files to verify.")

    # TODO: Implement standalone verification (Week 3)
    console.print("[yellow]Verification not yet implemented. Coming in v0.1.0.[/]")


@app.command()
def status(
    plan_file: Annotated[
        Optional[Path],
        typer.Argument(help="Path to plan.yaml file"),
    ] = None,
) -> None:
    """Show status of a previous or running execution."""
    # TODO: Implement status from SQLite state (Week 4)
    console.print("[yellow]Status tracking not yet implemented. Coming in v0.1.0.[/]")


@app.command()
def rerun(
    plan_file: Annotated[
        Path,
        typer.Argument(help="Path to plan.yaml file"),
    ],
    phase: Annotated[
        str,
        typer.Option("--phase", help="Phase to re-run: investigation, verification, synthesis"),
    ] = "verification",
) -> None:
    """Re-run a specific phase of a previous execution."""
    # TODO: Implement rerun from state (Week 4)
    console.print(f"[yellow]Rerun for phase '{phase}' not yet implemented. Coming in v0.1.0.[/]")


def version_callback(value: bool) -> None:
    if value:
        console.print(f"recon {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", callback=version_callback, is_eager=True),
    ] = None,
) -> None:
    """Recon - Verified research pipelines powered by AI agents."""
