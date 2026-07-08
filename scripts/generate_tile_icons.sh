#!/bin/bash
# Regenerate the packaged T/M/W tile sprite PNGs (development tool).
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

from procedural_frozenlake.tile_icons import (
    TILE_GLARE,
    TILE_SLEIGH,
    TILE_TREE,
    build_native_tile_icons,
)

out_dir = os.path.join(root, "src", "procedural_frozenlake", "img")
names = {
    TILE_TREE: "tree.png",
    TILE_GLARE: "glare_ice.png",
    TILE_SLEIGH: "sleigh.png",
}

pygame.init()
pygame.display.set_mode((1, 1))
os.makedirs(out_dir, exist_ok=True)
icons = build_native_tile_icons()
for tile, filename in names.items():
    path = os.path.join(out_dir, filename)
    pygame.image.save(icons[tile], path)
    print(f"wrote {path}")
EOF
