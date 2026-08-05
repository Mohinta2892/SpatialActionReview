"""Checks that must hold before this repository is made public.

These are not about the science; they are about not publishing something
embarrassing or legally unclear. They fail loudly rather than leaving a
placeholder, an unlicensed file tree, or a fabricated-data path in a public
release.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent
WORKSPACE = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))

# sha256 of the canonical GNU AGPL-3.0 text from https://www.gnu.org/licenses/agpl-3.0.txt
AGPL_SHA256 = "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0"


# ------------------------------------------------------------------ licence

def test_license_file_is_the_verbatim_agpl_3():
    """A licence has to be the real text, not a paraphrase of it."""
    path = APP_DIR / "LICENSE"
    assert path.exists(), "LICENSE is missing"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == AGPL_SHA256, (
        "LICENSE does not match the canonical AGPL-3.0 text byte for byte"
    )
    text = raw.decode()
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in text
    assert "Version 3, 19 November 2007" in text


def test_readme_and_citation_agree_on_the_licence():
    readme = (APP_DIR / "README.md").read_text()
    citation = (APP_DIR / "CITATION.cff").read_text()
    assert "AGPL" in readme, "README does not state the licence"
    assert re.search(r"^license:\s*AGPL-3\.0", citation, re.M), (
        "CITATION.cff does not declare the AGPL licence"
    )


# ------------------------------------------------------------------ citation

def test_citation_file_is_parseable_and_complete():
    yaml = pytest.importorskip("yaml", reason="PyYAML is not a runtime dependency")
    data = yaml.safe_load((APP_DIR / "CITATION.cff").read_text())
    assert data["cff-version"].startswith("1.2")
    for field in ("title", "message", "type", "license", "authors", "preferred-citation"):
        assert field in data, f"CITATION.cff is missing {field}"
    assert data["authors"], "CITATION.cff lists no authors"
    for author in data["authors"]:
        assert author.get("family-names") and author.get("given-names")


@pytest.mark.release
def test_no_placeholders_left_in_metadata():
    """REPLACE_ME must be filled in before the repository goes public.

    Marked `release`, so it is excluded from the default run and executed
    deliberately as a pre-publication gate: `pytest -m release`. It is expected
    to fail until the author and repository fields are filled in.
    """
    leftovers = {}
    for name in ("CITATION.cff", "README.md"):
        text = (APP_DIR / name).read_text()
        hits = re.findall(r"REPLACE_ME|TODO|FIXME|XXX", text)
        if hits:
            leftovers[name] = sorted(set(hits))
    assert not leftovers, f"placeholders still present: {leftovers}"


# ------------------------------------------------------- nothing fabricated

def test_no_synthetic_data_path_ships_with_the_app():
    """The app must contain no generator or fallback that could invent numbers."""
    banned = ("make_synthetic", "EMBEDDED_DATA", "percrop_SYNTHETIC", "figs_synthetic")
    offenders = {}
    for path in list(APP_DIR.glob("*.py")) + list((APP_DIR / "sar").glob("*.py")) \
            + list((APP_DIR / "tools").glob("*.py")):
        text = path.read_text()
        hits = [b for b in banned if b in text]
        if hits:
            offenders[path.name] = hits
    assert not offenders, f"synthetic-data code present: {offenders}"


def test_analysis_script_refuses_to_run_without_real_scores():
    """`04_reliability_analysis.py` must not fall back to generated data."""
    script = WORKSPACE / "plan" / "04_reliability_analysis.py"
    if not script.exists():
        pytest.skip("analysis script not present in this checkout")
    text = script.read_text()
    assert "make_synthetic" not in text
    assert "--synthetic" not in text
    assert "--percrop is required" in text, (
        "the script no longer states that real scores are required"
    )


def test_static_prototype_has_no_embedded_fallback():
    page = WORKSPACE / "plan" / "oversight_inspector.html"
    if not page.exists():
        pytest.skip("prototype page not present in this checkout")
    text = page.read_text()
    assert "EMBEDDED_DATA" not in text
    assert "Qwe_Joi" not in text, "synthetic crop ids still embedded in the page"


# --------------------------------------------------- nothing private ships

def test_no_absolute_local_paths_in_shipped_code():
    """Someone else's machine layout should not appear in a public repository."""
    offenders = {}
    for path in list(APP_DIR.glob("*.py")) + list((APP_DIR / "sar").glob("*.py")):
        for line in path.read_text().splitlines():
            if "/Users/" in line or "/mnt/graid" in line:
                offenders.setdefault(path.name, []).append(line.strip()[:80])
    assert not offenders, f"absolute local paths in code: {offenders}"


def test_no_credentials_or_tokens_in_the_tree():
    patterns = [
        re.compile(r"(?i)\b(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}"),
        re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),          # GitHub tokens
        re.compile(r"sk-[A-Za-z0-9]{20,}"),                  # OpenAI-style keys
    ]
    offenders = {}
    for path in APP_DIR.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".md", ".toml", ".cff", ".json"}:
            continue
        if ".venv" in path.parts or "release" in path.parts:
            continue
        text = path.read_text(errors="ignore")
        for pattern in patterns:
            if pattern.search(text):
                offenders.setdefault(str(path.relative_to(APP_DIR)), []).append(pattern.pattern[:30])
    assert not offenders, f"possible secrets: {offenders}"


def test_model_endpoint_config_has_no_baked_in_default():
    """The live-model client must read its endpoint from config, never hardcode one."""
    from sar import serve

    blank = serve.Endpoint()
    assert blank.base_url == "" and blank.model == ""
    assert not blank.configured
    source = (APP_DIR / "sar" / "serve.py").read_text()
    assert not re.search(r"https?://(?!<host>|gpu-host|localhost)[a-z0-9.\-]+\.[a-z]{2,}/v1", source), (
        "a concrete model endpoint appears to be hardcoded"
    )


# ------------------------------------------------------------ scope notice

def test_repository_states_what_it_does_not_include():
    """The model-adaptation code is reported separately and must be scoped out."""
    readme = (APP_DIR / "README.md").read_text().lower()
    assert "scope" in readme, "README has no scope section"
    assert ("training" in readme or "adaptation" in readme), (
        "README does not say whether model training code is included"
    )
