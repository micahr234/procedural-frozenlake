#!/bin/bash
# Regenerate README tile-legend preview PNGs (development tool).
# Run from the repo root: bash scripts/generate_legend_tiles.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] $PYTHON not found. Run: source scripts/install.sh" >&2
    exit 1
fi

export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"

"$PYTHON" - "$REPO_ROOT" <<'EOF'
import os
import sys

root = sys.argv[1]
sys.path.insert(0, os.path.join(root, "src"))

import pygame
from gymnasium.envs.toy_text import frozen_lake

from procedural_frozenlake.tile_icons import (
    TILE_GLARE,
    TILE_SLEIGH,
    TILE_TREE,
    build_native_tile_icons,
    build_sleigh_pair_badges,
    goal_reward_icon,
)

CELL_SIZE = (32, 32)
gym_img = os.path.join(os.path.dirname(frozen_lake.__file__), "img")
out_dir = os.path.join(root, "docs", "tile_legend")

pygame.init()
pygame.display.set_mode((1, 1))
os.makedirs(out_dir, exist_ok=True)


def load_scaled(name: str) -> pygame.Surface:
    path = os.path.join(gym_img, name)
    return pygame.transform.scale(pygame.image.load(path), CELL_SIZE)


def composite(*layers: pygame.Surface) -> pygame.Surface:
    surface = pygame.Surface(CELL_SIZE, pygame.SRCALPHA)
    for layer in layers:
        surface.blit(layer, (0, 0))
    return surface


ice = load_scaled("ice.png")
special = build_native_tile_icons()
sleigh_badge = build_sleigh_pair_badges(CELL_SIZE, 1)[0]

goal_raw = pygame.image.load(os.path.join(gym_img, "goal.png"))
goal_native = pygame.Surface(goal_raw.get_size(), pygame.SRCALPHA)
goal_native.blit(goal_raw, (0, 0))
goal = goal_reward_icon(goal_native, 0.75, "1.00", CELL_SIZE)

tiles = {
    "s": composite(ice, load_scaled("stool.png")),
    "f": ice.copy(),
    "m": composite(ice, special[TILE_GLARE]),
    "w": composite(ice, special[TILE_SLEIGH], sleigh_badge),
    "h": composite(ice, load_scaled("hole.png")),
    "g": goal,
    "t": composite(ice, special[TILE_TREE]),
}

for key, surface in tiles.items():
    path = os.path.join(out_dir, f"{key}.png")
    pygame.image.save(surface, path)
    print(f"wrote {path}")
EOF
