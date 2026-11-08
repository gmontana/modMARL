"""The package version is defined once and repeated consistently where users read it."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import modmarl
from modmarl._version import __version__

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"\d+\.\d+\.\d+")


def test_version_is_semantic_and_exported():
    assert SEMVER.fullmatch(__version__)
    assert modmarl.__version__ == __version__


def test_build_reads_the_single_version_source():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "version" not in project["project"]
    assert "version" in project["project"]["dynamic"]
    attr = project["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    assert attr == "modmarl._version.__version__"


def test_citation_repeats_the_package_version():
    match = re.search(r"^version:\s*\"?([^\"\n]+)\"?$", (ROOT / "CITATION.cff").read_text(), re.M)
    assert match is not None
    assert match.group(1).strip() == __version__


def test_changelog_records_the_current_release():
    releases = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", (ROOT / "CHANGELOG.md").read_text(), re.M)
    assert releases, "CHANGELOG.md has no released versions"
    assert releases[0] == __version__
