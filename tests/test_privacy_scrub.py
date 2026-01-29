from __future__ import annotations

from runtime.logging import JsonlWriter
from runtime.privacy import scrub_record, scrub_text


def test_scrub_text_redacts_common_patterns() -> None:
    text = "email me at alice@example.com or call 555-123-4567 from 127.0.0.1 sk-abcdef123456"
    scrubbed = scrub_text(text)
    assert "alice@example.com" not in scrubbed
    assert "555-123-4567" not in scrubbed
    assert "127.0.0.1" not in scrubbed
    assert "<REDACTED_EMAIL>" in scrubbed
    assert "<REDACTED_PHONE>" in scrubbed
    assert "<REDACTED_IP>" in scrubbed
    assert "<REDACTED_API_KEY>" in scrubbed


def test_scrub_record_nested() -> None:
    record = {"msg": "token=abc123", "nested": ["bob@example.com"]}
    scrubbed = scrub_record(record)
    assert "bob@example.com" not in scrubbed["nested"][0]
    assert "<REDACTED_EMAIL>" in scrubbed["nested"][0]


def test_jsonl_writer_scrubs_by_default(tmp_path) -> None:
    path = tmp_path / "log.jsonl"
    writer = JsonlWriter(path)
    writer.write({"msg": "reach me at bob@example.com"})
    writer.close()
    content = path.read_text(encoding="utf-8")
    assert "bob@example.com" not in content
    assert "<REDACTED_EMAIL>" in content
