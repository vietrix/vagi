from core.base import RecurrentState, VAGIConfig, VAGICore


def test_imports() -> None:
    assert VAGICore is not None
    assert VAGIConfig is not None
    assert RecurrentState is not None
