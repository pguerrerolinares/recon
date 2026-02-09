"""Recon CLI - Command-line interface for research pipelines.

Built with Typer + Rich for a clean terminal experience.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from recon import __version__
from recon.config import create_plan_from_topic, load_plan

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
        Path | None,
        typer.Argument(help="Path to plan.yaml file"),
    ] = None,
    topic: Annotated[
        str | None,
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
        str | None,
        typer.Option("--provider", "-p", help="LLM provider"),
    ] = None,
    model: Annotated[
        str | None,
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
    memory: Annotated[
        str | None,
        typer.Option("--memory", help="Path to knowledge database for cross-run persistence"),
    ] = None,
    auto_questions: Annotated[
        bool,
        typer.Option(
            "--auto-questions/--no-auto-questions",
            help="Auto-generate sub-questions per investigation angle via LLM",
        ),
    ] = True,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Re-run all phases even if output files exist"),
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

    # Apply --auto-questions override
    plan.auto_questions = auto_questions

    # Apply --memory override (sets knowledge DB path)
    if memory:
        plan.knowledge.enabled = True
        plan.knowledge.db_path = memory

    # Display plan summary
    investigations = plan.get_investigations()

    mode_label = "full" if force else "incremental"
    panel_content = (
        f"[bold]Topic:[/] {plan.topic}\n"
        f"[bold]Depth:[/] {plan.depth.value} ({len(investigations)} researchers)\n"
        f"[bold]Model:[/] {plan.model} via {plan.provider}\n"
        f"[bold]Verify:[/] {'yes' if plan.verify else 'no'}\n"
        f"[bold]Search:[/] {plan.search.provider}\n"
        f"[bold]Mode:[/] {mode_label}"
    )
    if plan.focus:
        panel_content += f"\n[bold]Focus:[/] {plan.focus}"
    if plan.knowledge.enabled:
        panel_content += f"\n[bold]Knowledge:[/] {plan.knowledge.db_path}"

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

        build_and_run(plan, verbose=verbose, console=console, force=force)
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
        console.print(
            f"Available templates: {', '.join(t.stem for t in TEMPLATES_DIR.glob('*.yaml'))}"
        )
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

    output.mkdir(parents=True, exist_ok=True)

    # Build a minimal plan for standalone verification
    plan = create_plan_from_topic(
        topic="Standalone verification",
        verify=True,
    )
    plan.research_dir = str(research_dir)
    plan.verification_dir = str(output)

    try:
        from recon.crews.verification.crew import build_verification_crew
        from recon.providers.llm import create_llm
        from recon.providers.search import create_search_tools

        llm = create_llm(plan)
        search_tools = create_search_tools(plan)

        crew = build_verification_crew(
            plan=plan,
            llm=llm,
            search_tools=search_tools,
            research_dir=str(research_dir),
        )

        if crew is None:
            console.print("[yellow]No claims found to verify.[/]")
            raise typer.Exit(code=0)

        console.print("[bold]Running verification...[/]")
        crew.kickoff()
        console.print(f"\n[green]Verification complete.[/] Report: {output}/report.md")

    except Exception as e:
        console.print(f"\n[red]Verification failed:[/] {e}")
        raise typer.Exit(code=4) from e


@app.command()
def status(
    output_dir: Annotated[
        Path,
        typer.Argument(help="Output directory to read status from"),
    ] = Path("./output"),
) -> None:
    """Show status of a previous or running execution."""
    from recon.callbacks.audit import AuditLogger

    audit_file = output_dir / "audit-log.jsonl"
    if not audit_file.exists():
        console.print(f"[yellow]No audit log found at {audit_file}[/]")
        console.print("Run a pipeline first: [bold]recon run <plan.yaml>[/]")
        raise typer.Exit(code=1)

    logger = AuditLogger(output_dir=str(output_dir))
    entries = logger.read_log()

    if not entries:
        console.print("[yellow]Audit log is empty.[/]")
        raise typer.Exit(code=0)

    # Build status table from audit events.
    table = Table(title="Pipeline Status")
    table.add_column("Phase", style="cyan", width=16)
    table.add_column("Status", width=12)
    table.add_column("Started", style="dim", width=22)
    table.add_column("Finished", style="dim", width=22)
    table.add_column("Output", style="white")

    phases: dict[str, dict[str, str]] = {}
    for entry in entries:
        phase = entry.get("phase", "")
        action = entry.get("action", "")
        ts = entry.get("timestamp", "")[:19]

        if action == "phase_start":
            phases[phase] = {"started": ts, "finished": "", "status": "running", "output": ""}
        elif action == "phase_end" and phase in phases:
            phases[phase]["finished"] = ts
            phases[phase]["status"] = "done"
            files = entry.get("metadata", {}).get("output_files", [])
            if files:
                phases[phase]["output"] = ", ".join(files)
        elif action == "error" and phase in phases:
            phases[phase]["status"] = "error"

    for phase_name, info in phases.items():
        status_text = info["status"]
        if status_text == "done":
            style = "[green]done[/]"
        elif status_text == "error":
            style = "[red]error[/]"
        else:
            style = "[yellow]running[/]"
        table.add_row(phase_name, style, info["started"], info["finished"], info["output"])

    console.print(table)

    # Show total time if pipeline completed.
    phase_list = list(phases.values())
    if phase_list and phase_list[-1]["finished"]:
        first_start = phase_list[0].get("started", "")
        last_end = phase_list[-1].get("finished", "")
        if first_start and last_end:
            console.print(f"\n[dim]Pipeline ran from {first_start} to {last_end}[/]")


@app.command()
def rerun(
    plan_file: Annotated[
        Path,
        typer.Argument(help="Path to plan.yaml file"),
    ],
    phase: Annotated[
        str,
        typer.Option(
            "--phase",
            help="Phase to re-run: investigation, verification, synthesis",
        ),
    ] = "verification",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose output"),
    ] = False,
) -> None:
    """Re-run a specific phase of a previous execution."""
    valid_phases = ("investigation", "verification", "synthesis")
    if phase not in valid_phases:
        console.print(
            f"[red]Error:[/] Invalid phase '{phase}'. Choose from: {', '.join(valid_phases)}"
        )
        raise typer.Exit(code=1)

    try:
        plan = load_plan(plan_file)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[red]Validation error:[/] {e}")
        raise typer.Exit(code=1) from e

    console.print(f"[bold]Re-running phase:[/] {phase}")

    try:
        from recon.providers.llm import create_llm
        from recon.providers.search import create_search_tools

        llm = create_llm(plan)
        search_tools = create_search_tools(plan)

        if phase == "investigation":
            from recon.crews.investigation.crew import build_investigation_crew

            Path(plan.research_dir).mkdir(parents=True, exist_ok=True)
            investigations = plan.get_investigations()
            crew = build_investigation_crew(
                plan=plan,
                investigations=investigations,
                llm=llm,
                search_tools=search_tools,
                verbose=verbose,
            )
            crew.kickoff()
            console.print(f"[green]Investigation complete.[/] Files in {plan.research_dir}/")

        elif phase == "verification":
            from recon.crews.verification.crew import build_verification_crew

            Path(plan.verification_dir).mkdir(parents=True, exist_ok=True)
            crew = build_verification_crew(
                plan=plan,
                llm=llm,
                search_tools=search_tools,
                research_dir=plan.research_dir,
                verbose=verbose,
            )
            if crew is None:
                console.print("[yellow]No research files to verify.[/]")
                raise typer.Exit(code=0)
            crew.kickoff()
            console.print(
                f"[green]Verification complete.[/] Report: {plan.verification_dir}/report.md"
            )

        elif phase == "synthesis":
            from recon.crews.synthesis.crew import build_synthesis_crew

            Path(plan.output_dir).mkdir(parents=True, exist_ok=True)
            crew = build_synthesis_crew(
                plan=plan,
                llm=llm,
                research_dir=plan.research_dir,
                verification_dir=plan.verification_dir if plan.verify else None,
                verbose=verbose,
            )
            crew.kickoff()
            console.print(
                f"[green]Synthesis complete.[/] Report: {plan.output_dir}/final-report.md"
            )

    except Exception as e:
        console.print(f"\n[red]Phase '{phase}' failed:[/] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=4) from e


# --- Memory subcommands ---

memory_app = typer.Typer(
    name="memory",
    help="Manage cross-run knowledge memory (.mv2 files).",
    no_args_is_help=True,
)
app.add_typer(memory_app, name="memory")


@memory_app.command("stats")
def memory_stats(
    path: Annotated[
        Path,
        typer.Argument(help="Path to .mv2 memory file"),
    ] = Path("./memory/recon.mv2"),
) -> None:
    """Show statistics about a memory file."""
    if not path.exists():
        console.print(f"[yellow]Memory file not found:[/] {path}")
        console.print("Memory files are created during pipeline runs with --memory.")
        raise typer.Exit(code=1)

    try:
        from recon.memory.store import MemoryStore

        store = MemoryStore(path=str(path))
        stats = store.stats()
        store.close()

        table = Table(title=f"Memory: {path}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        for key, value in stats.items():
            table.add_row(str(key), str(value))

        console.print(table)

    except ImportError:
        console.print(
            "[red]Error:[/] memvid-sdk is required. "
            "Install with: [bold]pip install recon-ai\\[memory][/]"
        )
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(f"[red]Error reading memory:[/] {e}")
        raise typer.Exit(code=1) from e


@memory_app.command("query")
def memory_query(
    query: Annotated[
        str,
        typer.Argument(help="Search query"),
    ],
    path: Annotated[
        Path,
        typer.Option("--path", "-p", help="Path to .mv2 memory file"),
    ] = Path("./memory/recon.mv2"),
    k: Annotated[
        int,
        typer.Option("--top", "-k", help="Number of results to return"),
    ] = 5,
) -> None:
    """Search for prior research in a memory file."""
    if not path.exists():
        console.print(f"[yellow]Memory file not found:[/] {path}")
        raise typer.Exit(code=1)

    try:
        from recon.memory.store import MemoryStore

        store = MemoryStore(path=str(path))
        results = store.query(topic=query, k=k)
        store.close()

        if not results:
            console.print("[yellow]No results found.[/]")
            raise typer.Exit(code=0)

        for i, hit in enumerate(results, 1):
            title = hit.get("title", "Untitled")
            text = hit.get("text", "")
            score = hit.get("score", 0.0)
            snippet = text[:200] + "..." if len(text) > 200 else text
            console.print(f"\n[bold cyan]{i}.[/] {title} [dim](score: {score:.2f})[/]")
            console.print(f"   {snippet}")

        console.print(f"\n[dim]{len(results)} results found.[/]")

    except ImportError:
        console.print(
            "[red]Error:[/] memvid-sdk is required. "
            "Install with: [bold]pip install recon-ai\\[memory][/]"
        )
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(f"[red]Error querying memory:[/] {e}")
        raise typer.Exit(code=1) from e


def version_callback(value: bool) -> None:
    if value:
        console.print(f"recon {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=version_callback, is_eager=True),
    ] = None,
) -> None:
    """Recon - Verified research pipelines powered by AI agents."""
