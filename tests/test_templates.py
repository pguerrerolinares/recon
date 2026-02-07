"""Tests for YAML plan templates.

Validates that all templates in src/recon/templates/ are valid YAML
and produce valid ReconPlan objects after filling in placeholder fields.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import pytest
import yaml

from recon.config import ReconPlan

TEMPLATES_DIR = Path(__file__).parent.parent / "src" / "recon" / "templates"


def get_template_files() -> list[Path]:
    """Collect all YAML template files."""
    return sorted(TEMPLATES_DIR.glob("*.yaml"))


class TestTemplateFiles:
    def test_templates_directory_exists(self) -> None:
        assert TEMPLATES_DIR.exists()
        assert TEMPLATES_DIR.is_dir()

    def test_four_templates_exist(self) -> None:
        templates = get_template_files()
        assert len(templates) == 4

    def test_expected_template_names(self) -> None:
        names = {t.stem for t in get_template_files()}
        expected = {
            "market-research",
            "competitive-analysis",
            "technical-landscape",
            "opportunity-finder",
        }
        assert names == expected


class TestTemplateValidity:
    @pytest.fixture(params=get_template_files(), ids=lambda p: p.stem)
    def template_path(self, request: pytest.FixtureRequest) -> Path:
        return request.param

    def test_template_is_valid_yaml(self, template_path: Path) -> None:
        content = template_path.read_text()
        data = yaml.safe_load(content)
        assert isinstance(data, dict), f"{template_path.name} does not parse to a dict"

    def test_template_has_required_fields(self, template_path: Path) -> None:
        data = yaml.safe_load(template_path.read_text())
        assert "topic" in data, f"{template_path.name} missing 'topic'"
        assert "depth" in data, f"{template_path.name} missing 'depth'"

    def test_template_produces_valid_plan(self, template_path: Path) -> None:
        """After patching the placeholder topic, the template should create a valid plan."""
        data = yaml.safe_load(template_path.read_text())

        # Templates have placeholder topics like "YOUR TOPIC HERE".
        # Replace with a concrete topic to pass validation.
        data["topic"] = "Test research topic"

        # Remove fields that might reference unset env vars.
        data.pop("provider", None)
        data.pop("model", None)

        plan = ReconPlan(**data)
        assert plan.topic == "Test research topic"
        assert plan.depth in ("quick", "standard", "deep")

    def test_template_has_questions_or_investigations(self, template_path: Path) -> None:
        data = yaml.safe_load(template_path.read_text())
        has_questions = bool(data.get("questions"))
        has_investigations = bool(data.get("investigations"))
        assert has_questions or has_investigations, (
            f"{template_path.name} has neither questions nor investigations"
        )

    def test_template_starts_with_comment(self, template_path: Path) -> None:
        """Templates should have a header comment with usage instructions."""
        first_line = template_path.read_text().split("\n")[0]
        assert first_line.startswith("#"), f"{template_path.name} should start with a comment"
