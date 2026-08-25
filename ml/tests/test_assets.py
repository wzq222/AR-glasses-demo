from pathlib import Path

import pytest

from crrc_vision.assets import asset_root


def test_asset_root_requires_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRRC_VISION_DATA_ROOT", raising=False)

    with pytest.raises(RuntimeError, match="CRRC_VISION_DATA_ROOT"):
        asset_root()


def test_asset_root_resolves_existing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRRC_VISION_DATA_ROOT", str(tmp_path))

    assert asset_root() == tmp_path.resolve()
