"""Pixel-art builders for tree, mirror ice, and sleigh tile overlays.

Used by ``scripts/generate_icons.sh`` to regenerate packaged PNGs under
``src/procedural_frozenlake/img/``. Not part of the runtime package.
"""

from __future__ import annotations

from typing import Any

TILE_TREE = "T"
TILE_MIRROR = "M"
TILE_SLEIGH = "W"

NATIVE_SIZE = 32

_Color = tuple[int, int, int, int]
_Grid = list[list[_Color]]

_TRANSPARENT: _Color = (0, 0, 0, 0)

# Palette sampled from the original FrozenLake ice.png / hole.png.
_SNOW: _Color = (255, 255, 255, 255)
_NEAR_WHITE: _Color = (248, 252, 255, 255)
_PALE_SOFT: _Color = (222, 238, 252, 255)
_ICE_PALE: _Color = (204, 230, 255, 255)

_GREEN: _Color = (46, 130, 66, 255)
_GREEN_DK: _Color = (24, 84, 44, 255)
_GREEN_LT: _Color = (72, 160, 92, 255)
_TRUNK: _Color = (110, 68, 40, 255)
_TRUNK_DK: _Color = (74, 44, 26, 255)

_RED: _Color = (198, 44, 48, 255)
_RED_DK: _Color = (128, 22, 28, 255)
_RED_LT: _Color = (230, 88, 88, 255)
_GOLD: _Color = (240, 194, 74, 255)
_GOLD_DK: _Color = (176, 128, 36, 255)
_FUR: _Color = (146, 96, 56, 255)
_FUR_DK: _Color = (100, 62, 34, 255)
_FUR_LT: _Color = (190, 144, 96, 255)
_ANTLER: _Color = (226, 196, 148, 255)
_NOSE: _Color = (255, 52, 44, 255)
_EYE: _Color = (30, 22, 18, 255)


def _empty_grid() -> _Grid:
    return [[_TRANSPARENT for _ in range(NATIVE_SIZE)] for _ in range(NATIVE_SIZE)]


def _px(grid: _Grid, x: int, y: int, color: _Color) -> None:
    if 0 <= x < NATIVE_SIZE and 0 <= y < NATIVE_SIZE:
        grid[y][x] = color


def _rect(grid: _Grid, x0: int, y0: int, x1: int, y1: int, color: _Color) -> None:
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            _px(grid, x, y, color)


def _hline(grid: _Grid, x0: int, x1: int, y: int, color: _Color) -> None:
    _rect(grid, x0, y, x1, y, color)


def _vline(grid: _Grid, x: int, y0: int, y1: int, color: _Color) -> None:
    _rect(grid, x, y0, x, y1, color)


def _tree_pixels() -> _Grid:
    """Snowy three-tier pine on a snow mound — impassable land."""
    grid = _empty_grid()
    cx = 16

    _hline(grid, 8, 23, 29, _SNOW)
    _hline(grid, 6, 25, 30, _SNOW)
    _hline(grid, 9, 22, 28, _ICE_PALE)

    _rect(grid, 14, 24, 17, 28, _TRUNK)
    _vline(grid, 14, 24, 28, _TRUNK_DK)

    # Tiers bottom to top: (top_y, bottom_y, top_half_width, bottom_half_width)
    tiers = [(17, 24, 3, 10), (11, 18, 2, 7), (4, 12, 0, 5)]
    for top_y, bot_y, hw0, hw1 in tiers:
        for y in range(top_y, bot_y + 1):
            t = (y - top_y) / max(bot_y - top_y, 1)
            hw = round(hw0 + (hw1 - hw0) * t)
            _hline(grid, cx - hw, cx + hw, y, _GREEN)
            _px(grid, cx - hw, y, _GREEN_DK)
            _px(grid, cx + hw, y, _GREEN_DK)
        _hline(grid, cx - hw1, cx + hw1, bot_y, _GREEN_DK)
        for y in range(top_y, min(top_y + 2, bot_y)):
            t = (y - top_y) / max(bot_y - top_y, 1)
            hw = round(hw0 + (hw1 - hw0) * t)
            _hline(grid, cx - hw, cx + hw, y, _SNOW)
    for top_y, bot_y, hw0, hw1 in tiers:
        for y in range(top_y + 2, bot_y):
            t = (y - top_y) / max(bot_y - top_y, 1)
            hw = round(hw0 + (hw1 - hw0) * t)
            _px(grid, cx - hw + 1, y, _GREEN_LT)

    for x, y in [(10, 22), (21, 23), (12, 16), (20, 17), (14, 10), (18, 11)]:
        _px(grid, x, y, _SNOW)
        _px(grid, x + 1, y, _SNOW)
    return grid


def _mirror_star(
    grid: _Grid, cx: int, cy: int, color: _Color, arm: int, diag_color: _Color | None = None
) -> None:
    for i in range(arm + 1):
        _px(grid, cx - i, cy, color)
        _px(grid, cx + i, cy, color)
        _px(grid, cx, cy - i, color)
        _px(grid, cx, cy + i, color)
    if diag_color is not None:
        _px(grid, cx - 1, cy - 1, diag_color)
        _px(grid, cx + 1, cy - 1, diag_color)
        _px(grid, cx - 1, cy + 1, diag_color)
        _px(grid, cx + 1, cy + 1, diag_color)


def _mirror_pixels() -> _Grid:
    """Almost-white polished ice patch with a star gleam — locally slippery.

    The smooth near-white fill covers the drift texture of the base ice tile,
    so mirror ice tiles read as clear, wiped-smooth ice next to normal
    snow-swirled tiles.
    """
    grid = _empty_grid()
    spans = {
        6: (11, 20), 7: (8, 23), 8: (7, 25), 9: (6, 26),
        10: (5, 27), 11: (4, 27), 12: (4, 28), 13: (3, 28),
        14: (3, 28), 15: (3, 28), 16: (3, 28), 17: (3, 28),
        18: (4, 28), 19: (4, 27), 20: (5, 27), 21: (6, 26),
        22: (7, 25), 23: (8, 23), 24: (11, 20),
    }
    for y, (x0, x1) in spans.items():
        _hline(grid, x0, x1, y, _NEAR_WHITE)
        _px(grid, x0, y, _PALE_SOFT)
        _px(grid, x1, y, _PALE_SOFT)
    _hline(grid, 11, 20, 6, _PALE_SOFT)
    _hline(grid, 11, 20, 24, _PALE_SOFT)

    # One large gleam plus two small echoes.
    _mirror_star(grid, 18, 13, _SNOW, arm=4, diag_color=_ICE_PALE)
    _mirror_star(grid, 10, 20, _PALE_SOFT, arm=2)
    _mirror_star(grid, 23, 20, _PALE_SOFT, arm=1)
    return grid


def _sleigh_pixels() -> _Grid:
    """Red sleigh with a reindeer — warps to the paired sleigh tile.

    Classic side profile facing right: high backrest on the left, seat dip
    in the middle, spiral-curled prow on the right, and gold runners with a
    hooked front tip. The reindeer rides in the sleigh behind the backrest.
    """
    grid = _empty_grid()

    # Reindeer antlers.
    for x, y in [(8, 1), (7, 2), (9, 2), (8, 3), (8, 4),
                 (14, 1), (15, 2), (13, 2), (14, 3), (14, 4),
                 (6, 3), (16, 3)]:
        _px(grid, x, y, _ANTLER)

    # Head with ears, muzzle, red nose, and eye.
    _rect(grid, 8, 5, 14, 10, _FUR)
    _hline(grid, 8, 14, 5, _FUR_DK)
    _vline(grid, 8, 5, 10, _FUR_DK)
    _px(grid, 7, 5, _FUR)
    _px(grid, 15, 5, _FUR)
    _rect(grid, 15, 7, 17, 10, _FUR_LT)
    _rect(grid, 17, 8, 18, 9, _NOSE)
    _px(grid, 13, 7, _EYE)

    # Neck down into the sleigh.
    _rect(grid, 8, 11, 13, 14, _FUR)
    _vline(grid, 8, 11, 14, _FUR_DK)
    _px(grid, 13, 11, _FUR_LT)

    # Body top-edge profile: backrest crest, slope to seat, seat dip, prow rise.
    top = {
        4: 12, 5: 11, 6: 11, 7: 12,
        8: 13, 9: 14, 10: 15, 11: 16,
        12: 17, 13: 17, 14: 17, 15: 17, 16: 17, 17: 17, 18: 17,
        19: 16, 20: 15, 21: 14, 22: 13, 23: 12, 24: 11, 25: 10, 26: 10,
    }
    bottom = 24
    for x, top_y in top.items():
        _vline(grid, x, top_y, bottom, _RED)
        _px(grid, x, top_y, _RED_DK)

    # Prow spiral curl; the transparent center reads as the curl's loop.
    for x, y in [(26, 9), (26, 8), (26, 7), (25, 6), (24, 6),
                 (23, 7), (23, 8), (24, 9), (25, 9)]:
        _px(grid, x, y, _RED)
    _px(grid, 25, 8, _RED)
    _px(grid, 25, 6, _RED_DK)
    _px(grid, 26, 7, _RED_DK)

    # Shading and highlight along the body.
    _hline(grid, 4, 26, bottom, _RED_DK)
    _vline(grid, 4, 12, bottom, _RED_DK)
    _vline(grid, 26, 10, bottom, _RED_DK)
    for x, top_y in top.items():
        if 8 <= x <= 25:
            _px(grid, x, top_y + 1, _RED_LT)
    _px(grid, 5, 12, _RED_LT)
    _px(grid, 6, 12, _RED_LT)

    # Gold trim swoosh following the body profile.
    for x, top_y in top.items():
        if x >= 5:
            _px(grid, x, top_y + 3, _GOLD)

    # Runners with hooked front tip and struts.
    _hline(grid, 5, 27, 27, _GOLD)
    _hline(grid, 5, 27, 28, _GOLD_DK)
    _px(grid, 28, 26, _GOLD)
    _px(grid, 29, 25, _GOLD)
    _px(grid, 29, 24, _GOLD)
    _px(grid, 28, 23, _GOLD)
    _px(grid, 29, 26, _GOLD_DK)
    _px(grid, 4, 26, _GOLD_DK)
    _vline(grid, 8, 25, 26, _GOLD_DK)
    _vline(grid, 23, 25, 26, _GOLD_DK)
    return grid


_PIXEL_BUILDERS = {
    TILE_TREE: _tree_pixels,
    TILE_MIRROR: _mirror_pixels,
    TILE_SLEIGH: _sleigh_pixels,
}


def _surface_from_pixels(grid: _Grid) -> Any:
    import pygame

    surface = pygame.Surface((NATIVE_SIZE, NATIVE_SIZE), pygame.SRCALPHA)
    for y, row in enumerate(grid):
        for x, color in enumerate(row):
            if color[3] > 0:
                surface.set_at((x, y), color)
    return surface


def build_native_tile_icons() -> dict[str, Any]:
    """Return 32x32 pygame surfaces for T/M/W sprites (transparent background)."""
    return {tile: _surface_from_pixels(build()) for tile, build in _PIXEL_BUILDERS.items()}
