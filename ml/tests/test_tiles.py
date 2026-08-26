import pytest

from crrc_vision.tiles import build_tiles, map_tile_box


def test_two_by_two_tiles_cover_image_with_overlap() -> None:
    tiles = build_tiles(width=2000, height=1500, grid=2, overlap=0.12)

    assert len(tiles) == 4
    assert min(tile.x1 for tile in tiles) == 0
    assert min(tile.y1 for tile in tiles) == 0
    assert max(tile.x2 for tile in tiles) == 2000
    assert max(tile.y2 for tile in tiles) == 1500
    assert tiles[0].x2 > tiles[1].x1
    assert tiles[0].y2 > tiles[2].y1


def test_tile_box_maps_and_clips_to_original_image() -> None:
    tile = build_tiles(2000, 1500, 2, 0.12)[3]

    mapped = map_tile_box(tile, (-10, -10, 9999, 9999), 2000, 1500)

    assert mapped == (tile.x1, tile.y1, 2000, 1500)


def test_invalid_tile_contract_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_tiles(2000, 1500, grid=3, overlap=0.12)

    with pytest.raises(ValueError):
        build_tiles(2000, 1500, grid=2, overlap=0.5)
