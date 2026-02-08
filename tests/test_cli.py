"""Tests for the Recon CLI commands.

Uses typer.testing.CliRunner to invoke commands without subprocess overhead.
All external calls (LLM, search, crews) are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003
from unittest.mock import patch

from typer.testing import CliRunner

from recon.cli import app

runner = CliRunner()


class TestVersion:
    def test_version_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "recon" in result.output
        assert "0.1.0" in result.output

    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "research pipelines" in result.output.lower() or "Usage" in result.output


class TestRun:
    def test_run_requires_plan_or_topic(self) -> None:
        result = runner.invoke(app, ["run"])
        assert result.exit_code == 1
        assert "plan file" in result.output.lower() or "topic" in result.output.lower()

    def test_run_rejects_both_plan_and_topic(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text("topic: test\n")
        result = runner.invoke(app, ["run", str(plan), "--topic", "test"])
        assert result.exit_code == 1
        assert "Cannot specify both" in result.output

    def test_run_nonexistent_plan(self) -> None:
        result = runner.invoke(app, ["run", "/nonexistent/plan.yaml"])
        assert result.exit_code == 1

    def test_run_dry_run(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text(
            "topic: Test topic\nquestions:\n  - What is this?\ndepth: quick\nverify: false\n"
        )
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            result = runner.invoke(app, ["run", str(plan), "--dry-run"])
        assert result.exit_code == 0
        assert "Dry run" in result.output

    def test_run_dry_run_verbose(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text(
            "topic: Test topic\nquestions:\n  - What is this?\ndepth: quick\nverify: false\n"
        )
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            result = runner.invoke(app, ["run", str(plan), "--dry-run", "--verbose"])
        assert result.exit_code == 0
        assert "Investigation Agents" in result.output

    def test_run_inline_topic_dry_run(self) -> None:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            result = runner.invoke(
                app, ["run", "--topic", "AI testing", "--dry-run", "--no-verify"]
            )
        assert result.exit_code == 0
        assert "Dry run" in result.output

    def test_run_dry_run_shows_incremental_mode(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text(
            "topic: Test topic\nquestions:\n  - What is this?\ndepth: quick\nverify: false\n"
        )
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            result = runner.invoke(app, ["run", str(plan), "--dry-run"])
        assert result.exit_code == 0
        assert "incremental" in result.output

    def test_run_dry_run_force_shows_full_mode(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text(
            "topic: Test topic\nquestions:\n  - What is this?\ndepth: quick\nverify: false\n"
        )
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            result = runner.invoke(app, ["run", str(plan), "--dry-run", "--force"])
        assert result.exit_code == 0
        assert "full" in result.output


class TestInit:
    def test_init_default_template(self, tmp_path: Path) -> None:
        output = tmp_path / "plan.yaml"
        result = runner.invoke(app, ["init", "--output", str(output)])
        assert result.exit_code == 0
        assert "Created" in result.output
        assert output.exists()
        content = output.read_text()
        assert "topic:" in content

    def test_init_specific_template(self, tmp_path: Path) -> None:
        output = tmp_path / "plan.yaml"
        result = runner.invoke(
            app, ["init", "--template", "competitive-analysis", "--output", str(output)]
        )
        assert result.exit_code == 0
        assert output.exists()

    def test_init_nonexistent_template(self, tmp_path: Path) -> None:
        output = tmp_path / "plan.yaml"
        result = runner.invoke(app, ["init", "--template", "nonexistent", "--output", str(output)])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_init_overwrite_decline(self, tmp_path: Path) -> None:
        output = tmp_path / "plan.yaml"
        output.write_text("existing content")
        result = runner.invoke(app, ["init", "--output", str(output)], input="n\n")
        assert result.exit_code == 0
        # File should still have original content.
        assert output.read_text() == "existing content"


class TestTemplates:
    def test_templates_lists_all(self) -> None:
        result = runner.invoke(app, ["templates"])
        assert result.exit_code == 0
        assert "market-research" in result.output
        assert "competitive-analysis" in result.output
        assert "technical-landscape" in result.output
        assert "opportunity-finder" in result.output


class TestStatus:
    def test_status_no_audit_log(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["status", str(tmp_path)])
        assert result.exit_code == 1
        assert "No audit log" in result.output

    def test_status_with_audit_log(self, tmp_path: Path) -> None:
        audit_file = tmp_path / "audit-log.jsonl"
        entries = [
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "phase": "investigation",
                "agent": "pipeline",
                "action": "phase_start",
                "detail": "",
            },
            {
                "timestamp": "2026-01-01T00:01:00+00:00",
                "phase": "investigation",
                "agent": "pipeline",
                "action": "phase_end",
                "detail": "",
                "metadata": {"output_files": ["research/test.md"]},
            },
        ]
        audit_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        result = runner.invoke(app, ["status", str(tmp_path)])
        assert result.exit_code == 0
        assert "investigation" in result.output
        assert "done" in result.output


class TestVerify:
    def test_verify_missing_dir(self) -> None:
        result = runner.invoke(app, ["verify", "/nonexistent/dir"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_verify_empty_dir(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["verify", str(tmp_path)])
        assert result.exit_code == 0
        assert "No markdown files" in result.output

    def test_verify_with_files_finds_them(self, tmp_path: Path) -> None:
        """Verify that the command detects markdown files and reports the count."""
        research_dir = tmp_path / "research"
        research_dir.mkdir()
        (research_dir / "test.md").write_text("# Test\nSome research content.\n")
        output_dir = tmp_path / "verification"

        # The command will fail at create_llm (no API key), but it finds files first.
        result = runner.invoke(app, ["verify", str(research_dir), "--output", str(output_dir)])
        assert "1 research file" in result.output


class TestRerun:
    def test_rerun_invalid_phase(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text("topic: test\nquestions:\n  - q1\ndepth: quick\n")
        result = runner.invoke(app, ["rerun", str(plan), "--phase", "invalid"])
        assert result.exit_code == 1
        assert "Invalid phase" in result.output

    def test_rerun_nonexistent_plan(self) -> None:
        result = runner.invoke(app, ["rerun", "/nonexistent/plan.yaml"])
        assert result.exit_code == 1

    def test_rerun_valid_phase_accepted(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text("topic: test\nquestions:\n  - q1\ndepth: quick\nverify: true\n")
        # Will fail at create_llm (no API key) but phase validation should pass.
        result = runner.invoke(app, ["rerun", str(plan), "--phase", "verification"])
        assert result.exit_code != 0
        assert "Re-running phase" in result.output
