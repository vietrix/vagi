from __future__ import annotations

import pytest
from typer.testing import CliRunner

import vagi_orchestrator.cli as cli_module
from vagi_orchestrator.cli import app
from vagi_orchestrator.memory import MemoryAnswer, MemoryHit


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

        def answer(
            self,
            query: str,
            top_k: int = 3,
            min_score: float = 0.2,
            max_sentences: int = 3,
        ) -> MemoryAnswer:
            return MemoryAnswer(
                answer="vAGI la he thong mo phong tu duy.",
                hits=[
                    MemoryHit(text="doc A", score=0.91),
                    MemoryHit(text="doc B", score=0.84),
                ],
            )

    monkeypatch.setattr(cli_module, "MemoryClient", FakeClient)

    runner = CliRunner()
    result = runner.invoke(app, ["memory", "ask", "what is auth", "--top-k", "2"])
    assert result.exit_code == 0
    assert "vAGI la he thong mo phong tu duy." in result.stdout
    assert "1. score=0.910 | doc A" in result.stdout
    assert "2. score=0.840 | doc B" in result.stdout


def test_memory_alias_commands_still_work(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, kernel_url: str | None = None, timeout: float = 120.0) -> None:
            self.kernel_url = kernel_url
            self.timeout = timeout

        def ingest_file(self, file_path: str) -> dict[str, int]:
            return {"total": 1, "success": 1, "failed": 0}

        def answer(
            self,
            query: str,
            top_k: int = 3,
            min_score: float = 0.2,
            max_sentences: int = 3,
        ) -> MemoryAnswer:
            return MemoryAnswer(
                answer="doc alias answer",
                hits=[MemoryHit(text="doc alias", score=0.77)],
            )

    monkeypatch.setattr(cli_module, "MemoryClient", FakeClient)
    runner = CliRunner()

    ingest_result = runner.invoke(app, ["memory", "ingest", "k.txt"])
    assert ingest_result.exit_code == 0
    assert "learn_total=1 success=1 failed=0" in ingest_result.stdout

    query_result = runner.invoke(app, ["memory", "query", "q"])
    assert query_result.exit_code == 0
    assert "doc alias answer" in query_result.stdout
    assert "1. score=0.770 | doc alias" in query_result.stdout
