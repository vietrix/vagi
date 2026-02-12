from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import vagi_orchestrator.cli as cli_module
from vagi_orchestrator.cli import app


def test_split_paragraphs_handles_blank_lines() -> None:
    text = "First paragraph\n\n\nSecond paragraph\n \nThird paragraph"
    assert cli_module._split_paragraphs(text) == [
        "First paragraph",
        "Second paragraph",
        "Third paragraph",
    ]


def test_memory_ingest_command_uploads_paragraphs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = tmp_path / "knowledge.txt"
    doc.write_text("alpha\n\nbeta\n\n\ngamma", encoding="utf-8")
    ingested: list[str] = []

    class FakeClient:
        def __init__(self, kernel_url: str | None = None, timeout: float = 120.0) -> None:
            self.kernel_url = kernel_url
            self.timeout = timeout

        def add_document(self, text: str) -> bool:
            ingested.append(text)
            return True

    monkeypatch.setattr(cli_module, "MemoryClient", FakeClient)

    runner = CliRunner()
    result = runner.invoke(app, ["memory", "ingest", str(doc)])
    assert result.exit_code == 0
    assert ingested == ["alpha", "beta", "gamma"]
    assert "ingest_total=3 success=3 failed=0" in result.stdout


def test_memory_query_command_prints_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, kernel_url: str | None = None, timeout: float = 120.0) -> None:
            self.kernel_url = kernel_url
            self.timeout = timeout

        def search(self, query: str, top_k: int = 3) -> list[str]:
            return ["doc A", "doc B"]

    monkeypatch.setattr(cli_module, "MemoryClient", FakeClient)

    runner = CliRunner()
    result = runner.invoke(app, ["memory", "query", "what is auth", "--top-k", "2"])
    assert result.exit_code == 0
    assert "1. doc A" in result.stdout
    assert "2. doc B" in result.stdout
