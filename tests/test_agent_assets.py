"""Validate the campaign agent layer: frontmatter + templates exist & parse."""
from __future__ import annotations

from pathlib import Path

import pytest

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parent.parent
AGENTS = ["experiment-monitor", "experiment-analyst", "experiment-coder"]


def _frontmatter(md_path: Path) -> dict:
    text = md_path.read_text()
    assert text.startswith("---\n"), f"{md_path} missing frontmatter"
    _, fm, _ = text.split("---\n", 2)
    return yaml.safe_load(fm) if yaml else {"raw": fm}


@pytest.mark.skipif(yaml is None, reason="pyyaml not installed")
@pytest.mark.parametrize("name", AGENTS)
def test_agent_frontmatter_has_name_and_description(name):
    fm = _frontmatter(ROOT / ".claude" / "agents" / f"{name}.md")
    assert fm["name"] == name
    assert fm["description"].strip()


@pytest.mark.skipif(yaml is None, reason="pyyaml not installed")
def test_controller_skill_frontmatter():
    fm = _frontmatter(ROOT / ".claude" / "skills" / "run-campaign" / "SKILL.md")
    assert fm["name"] == "run-campaign"
    assert fm["description"].strip()


def test_campaign_templates_exist():
    tmpl = ROOT / "docs" / "campaigns" / "TEMPLATE"
    for fname in ("goal.md", "ledger.md", "frontier.md"):
        assert (tmpl / fname).is_file(), f"missing template {fname}"
