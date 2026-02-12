from __future__ import annotations

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


def test_memory_learn_command_calls_ingest_file(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, kernel_url: str | None = None, timeout: float = 120.0) -> None:
            self.kernel_url = kernel_url
            self.timeout = timeout

        def ingest_file(self, file_path: str) -> dict[str, int]:
            captured["file_path"] = file_path
            return {"total": 3, "success": 3, "failed": 0}

    monkeypatch.setattr(cli_module, "MemoryClient", FakeClient)

    runner = CliRunner()
    result = runner.invoke(app, ["memory", "learn", "knowledge.txt"])
    assert result.exit_code == 0
    assert captured["file_path"] == "knowledge.txt"
    assert "learn_total=3 success=3 failed=0" in result.stdout


def test_memory_ask_command_prints_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, kernel_url: str | None = None, timeout: float = 120.0) -> None:
            self.kernel_url = kernel_url
            self.timeout = timeout

        def retrieve(self, query: str, top_k: int = 3) -> list[str]:
            return ["doc A", "doc B"]

    monkeypatch.setattr(cli_module, "MemoryClient", FakeClient)

    runner = CliRunner()
    result = runner.invoke(app, ["memory", "ask", "what is auth", "--top-k", "2"])
    assert result.exit_code == 0
    assert "1. doc A" in result.stdout
    assert "2. doc B" in result.stdout


def test_memory_alias_commands_still_work(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, kernel_url: str | None = None, timeout: float = 120.0) -> None:
            self.kernel_url = kernel_url
            self.timeout = timeout

        def ingest_file(self, file_path: str) -> dict[str, int]:
            return {"total": 1, "success": 1, "failed": 0}

        def retrieve(self, query: str, top_k: int = 3) -> list[str]:
            return ["doc alias"]

    monkeypatch.setattr(cli_module, "MemoryClient", FakeClient)
    runner = CliRunner()

    ingest_result = runner.invoke(app, ["memory", "ingest", "k.txt"])
    assert ingest_result.exit_code == 0
    assert "learn_total=1 success=1 failed=0" in ingest_result.stdout

    query_result = runner.invoke(app, ["memory", "query", "q"])
    assert query_result.exit_code == 0
    assert "1. doc alias" in query_result.stdout
