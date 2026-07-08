#!/bin/bash
# Regenerate tile images under src/procedural_frozenlake/img/.
# Run from the repo root: bash scripts/generate_tile_icons.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] $PYTHON not found. Run: source scripts/install.sh" >&2
    exit 1
fi

# Headless SDL so pygame works without a display.
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
img_dir = os.path.join(root, "src", "procedural_frozenlake", "img")
gym_img = os.path.join(os.path.dirname(frozen_lake.__file__), "img")

pygame.init()
pygame.display.set_mode((1, 1))
os.makedirs(img_dir, exist_ok=True)

overlays = build_native_tile_icons()


def load_scaled(path: str) -> pygame.Surface:
    return pygame.transform.scale(pygame.image.load(path), CELL_SIZE)


def composite(*layers: pygame.Surface) -> pygame.Surface:
    surface = pygame.Surface(CELL_SIZE, pygame.SRCALPHA)
    for layer in layers:
        surface.blit(layer, (0, 0))
    return surface


ice = load_scaled(os.path.join(gym_img, "ice.png"))
sleigh_badge = build_sleigh_pair_badges(CELL_SIZE, 1)[0]

goal_raw = pygame.image.load(os.path.join(gym_img, "goal.png"))
goal_native = pygame.Surface(goal_raw.get_size(), pygame.SRCALPHA)
goal_native.blit(goal_raw, (0, 0))
goal = goal_reward_icon(goal_native, 0.75, "1.00", CELL_SIZE)

# Transparent W-tile sprite — pair badges are composited at render time.
overlay_path = os.path.join(img_dir, "tile_w_overlay.png")
pygame.image.save(overlays[TILE_SLEIGH], overlay_path)
print(f"wrote {overlay_path}")

tiles = {
    "tile_s": composite(ice, load_scaled(os.path.join(gym_img, "stool.png"))),
    "tile_f": ice.copy(),
    "tile_m": composite(ice, overlays[TILE_GLARE]),
    "tile_t": composite(ice, overlays[TILE_TREE]),
    "tile_h": composite(ice, load_scaled(os.path.join(gym_img, "hole.png"))),
    "tile_g": composite(ice, goal),
    "tile_w": composite(ice, overlays[TILE_SLEIGH], sleigh_badge),
}

for name, surface in tiles.items():
    path = os.path.join(img_dir, f"{name}.png")
    pygame.image.save(surface, path)
    print(f"wrote {path}")
EOF
