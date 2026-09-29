import pytest

from airport_agent import config


@pytest.fixture(autouse=True)
def _isolated_logs(tmp_path, monkeypatch):
    """Keep test runs from writing telemetry into the repo's logs/ folder."""
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
