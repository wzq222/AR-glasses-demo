from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path, PurePosixPath, PureWindowsPath


FROZEN_FORMAL_TRUTH_SHA256 = (
    "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"
)
SYNTHETIC_STATES = frozenset({"NORMAL", "SLIGHT_LOOSE", "OBVIOUS_LOOSE"})


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def assert_formal_truth_unchanged(
    path: Path, expected_sha256: str = FROZEN_FORMAL_TRUTH_SHA256
) -> str:
    actual = sha256_file(path)
    if actual.upper() != expected_sha256.upper():
        raise RuntimeError(
            f"formal truth SHA-256 changed: expected {expected_sha256}, got {actual}"
        )
    return actual


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def assert_external_output(output: Path, repository_root: Path) -> Path:
    resolved_output = output.resolve()
    resolved_repo = repository_root.resolve()
    if resolved_output == resolved_repo or _is_relative_to(resolved_output, resolved_repo):
        raise ValueError(f"合成资产必须写入Git外目录: {resolved_output}")
    return resolved_output


@dataclass(frozen=True)
class SyntheticRecord:
    sample_id: str
    source_reference_sha256: str
    source_scene_id: str
    state: str
    image_path: str
    synthetic: bool = field(default=True, init=False)
    eligible_split: str = field(default="train", init=False)

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id不能为空")
        if not self.source_scene_id.strip():
            raise ValueError("source_scene_id不能为空")
        if self.state not in SYNTHETIC_STATES:
            raise ValueError(f"未知合成状态: {self.state}")
        digest = self.source_reference_sha256
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            raise ValueError("source_reference_sha256必须是64位十六进制")
        normalized = self.image_path.replace("\\", "/")
        pure = PurePosixPath(normalized)
        windows_path = PureWindowsPath(self.image_path)
        if pure.is_absolute() or windows_path.is_absolute() or windows_path.drive or ".." in pure.parts:
            raise ValueError("image_path必须是安全相对路径")
