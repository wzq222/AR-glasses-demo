"""Deterministic overlap tiles for full-image small-object inference."""

from __future__ import annotations

from dataclasses import dataclass


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Tile:
    index: int
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def build_tiles(
    width: int,
    height: int,
    grid: int = 2,
    overlap: float = 0.12,
) -> tuple[Tile, ...]:
    """Build the fixed 2x2 overlap layout used by the Phase A pipeline."""

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if grid != 2 or not 0.0 <= overlap < 0.5:
        raise ValueError("supported contract is grid=2 and 0 <= overlap < 0.5")

    middle_x, middle_y = width // 2, height // 2
    expand_x = round(width * overlap / 2)
    expand_y = round(height * overlap / 2)
    x_ranges = (
        (0, min(width, middle_x + expand_x)),
        (max(0, middle_x - expand_x), width),
    )
    y_ranges = (
        (0, min(height, middle_y + expand_y)),
        (max(0, middle_y - expand_y), height),
    )
    return tuple(
        Tile(row * grid + column, x1, y1, x2, y2)
        for row, (y1, y2) in enumerate(y_ranges)
        for column, (x1, x2) in enumerate(x_ranges)
    )


def map_tile_box(
    tile: Tile,
    xyxy: Box,
    image_width: int,
    image_height: int,
) -> Box:
    """Map a tile-local box into image coordinates and clip it to the tile."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not (0 <= tile.x1 < tile.x2 <= image_width):
        raise ValueError("tile is outside the image width")
    if not (0 <= tile.y1 < tile.y2 <= image_height):
        raise ValueError("tile is outside the image height")
    if len(xyxy) != 4:
        raise ValueError("tile box must contain four coordinates")

    local_x1 = max(0.0, min(float(tile.width), xyxy[0]))
    local_y1 = max(0.0, min(float(tile.height), xyxy[1]))
    local_x2 = max(0.0, min(float(tile.width), xyxy[2]))
    local_y2 = max(0.0, min(float(tile.height), xyxy[3]))
    mapped = (
        tile.x1 + local_x1,
        tile.y1 + local_y1,
        tile.x1 + local_x2,
        tile.y1 + local_y2,
    )
    if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
        raise ValueError("empty mapped tile box")
    return mapped
