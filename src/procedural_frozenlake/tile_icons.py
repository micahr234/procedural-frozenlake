"""Runtime tile-icon helpers: sleigh pair badges and goal reward compositing.

Pixel-art builders for the T/M/W overlays live in ``scripts/tile_icon_pixels.py``
and are used only by ``scripts/generate_icons.sh`` to regenerate packaged PNGs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from procedural_frozenlake.tiles import TILE_MIRROR, TILE_SLEIGH, TILE_TREE

if TYPE_CHECKING:
    import pygame

SPECIAL_TILES = (TILE_TREE, TILE_MIRROR, TILE_SLEIGH)

NATIVE_SIZE = 32

_Color = tuple[int, int, int, int]
_Grid = list[list[_Color]]

_TRANSPARENT: _Color = (0, 0, 0, 0)
_SNOW: _Color = (255, 255, 255, 255)

# (fill, border) colors for sleigh pair badges; cycles when exhausted.
# Ordered for contrast against the red sleigh body and gold runners.
_BADGE_COLORS: tuple[tuple[_Color, _Color], ...] = (
    ((70, 130, 220, 255), (36, 84, 164, 255)),    # blue
    ((52, 168, 120, 255), (26, 112, 76, 255)),    # green
    ((154, 90, 200, 255), (104, 56, 142, 255)),   # purple
    ((230, 100, 160, 255), (168, 58, 110, 255)),  # pink
    ((240, 194, 74, 255), (176, 128, 36, 255)),   # gold
    ((235, 130, 50, 255), (170, 86, 24, 255)),    # orange
)

# 3x5 pixel glyphs for badge text.
_BADGE_GLYPHS: dict[str, tuple[str, ...]] = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    ".": ("000", "000", "000", "000", "010"),
    "-": ("000", "000", "111", "000", "000"),
}


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


def _text_badge_pixels(text: str, fill: _Color, border: _Color) -> _Grid:
    """Small rounded badge with 3x5 pixel text in the bottom-left tile corner."""
    grid = _empty_grid()

    text_w = 4 * len(text) - 1
    badge_w = text_w + 4
    badge_h = 9
    x0 = 1
    y0 = NATIVE_SIZE - 1 - badge_h

    _rect(grid, x0, y0, x0 + badge_w - 1, y0 + badge_h - 1, fill)
    _hline(grid, x0, x0 + badge_w - 1, y0, border)
    _hline(grid, x0, x0 + badge_w - 1, y0 + badge_h - 1, border)
    _vline(grid, x0, y0, y0 + badge_h - 1, border)
    _vline(grid, x0 + badge_w - 1, y0, y0 + badge_h - 1, border)
    # Rounded corners.
    for cx, cy in (
        (x0, y0),
        (x0 + badge_w - 1, y0),
        (x0, y0 + badge_h - 1),
        (x0 + badge_w - 1, y0 + badge_h - 1),
    ):
        _px(grid, cx, cy, _TRANSPARENT)

    dx = x0 + 2
    dy = y0 + 2
    for ch in text:
        for gy, row in enumerate(_BADGE_GLYPHS[ch]):
            for gx, bit in enumerate(row):
                if bit == "1":
                    _px(grid, dx + gx, dy + gy, _SNOW)
        dx += 4
    return grid


def _badge_pixels(pair_index: int) -> _Grid:
    """Numbered badge for sleigh pairs; both sleighs of a pair get the same badge."""
    fill, border = _BADGE_COLORS[pair_index % len(_BADGE_COLORS)]
    return _text_badge_pixels(str(pair_index + 1), fill, border)


def _surface_from_pixels(grid: _Grid) -> Any:
    import pygame

    surface = pygame.Surface((NATIVE_SIZE, NATIVE_SIZE), pygame.SRCALPHA)
    for y, row in enumerate(grid):
        for x, color in enumerate(row):
            if color[3] > 0:
                surface.set_at((x, y), color)
    return surface


def build_sleigh_pair_badges(
    cell_size: tuple[int, int], pair_count: int
) -> list["pygame.Surface"]:
    """Return one badge overlay per sleigh pair, scaled to ``cell_size``."""
    import pygame

    size = (int(cell_size[0]), int(cell_size[1]))
    return [
        pygame.transform.scale(_surface_from_pixels(_badge_pixels(i)), size)
        for i in range(pair_count)
    ]


# Reward color scale for goal presents: yellow (low) → green (high).
_REWARD_LOW: _Color = (235, 205, 50, 255)
_REWARD_HIGH: _Color = (58, 172, 74, 255)

# Badge colors matching the present's box blues.
_GOAL_BADGE_FILL: _Color = (58, 63, 94, 255)
_GOAL_BADGE_BORDER: _Color = (43, 43, 69, 255)

# Brightness of the lightest bow tone in the original goal.png, used to
# preserve the ribbon's shading when re-hueing it.
_BOW_REF_BRIGHTNESS = 240 + 181 + 65


def _reward_color(t: float) -> tuple[int, int, int]:
    t = min(max(t, 0.0), 1.0)
    return (
        int(_REWARD_LOW[0] + (_REWARD_HIGH[0] - _REWARD_LOW[0]) * t),
        int(_REWARD_LOW[1] + (_REWARD_HIGH[1] - _REWARD_LOW[1]) * t),
        int(_REWARD_LOW[2] + (_REWARD_HIGH[2] - _REWARD_LOW[2]) * t),
    )


def _is_bow_pixel(r: int, g: int, b: int) -> bool:
    # The gift bow is the only orange part of goal.png (box is blue).
    return r > 120 and r > b + 50 and g < r


def goal_reward_icon(
    goal_surface: Any, t: float, reward_text: str, cell_size: tuple[int, int]
) -> Any:
    """Return the goal present with its bow tinted by reward and a reward badge.

    ``goal_surface`` is the native (32x32) FrozenLake goal sprite; ``t`` in
    [0, 1] blends the bow from yellow (low reward) to green (high reward).
    """
    import pygame

    icon = goal_surface.copy()
    w, h = icon.get_size()
    target = _reward_color(t)
    for y in range(h):
        for x in range(w):
            r, g, b, a = icon.get_at((x, y))
            if a == 0 or not _is_bow_pixel(r, g, b):
                continue
            factor = min((r + g + b) / _BOW_REF_BRIGHTNESS, 1.0)
            icon.set_at(
                (x, y),
                (
                    int(target[0] * factor),
                    int(target[1] * factor),
                    int(target[2] * factor),
                    a,
                ),
            )
    badge = _surface_from_pixels(
        _text_badge_pixels(reward_text, _GOAL_BADGE_FILL, _GOAL_BADGE_BORDER)
    )
    icon.blit(badge, (0, 0))
    size = (int(cell_size[0]), int(cell_size[1]))
    return pygame.transform.scale(icon, size)
