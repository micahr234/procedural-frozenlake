#!/usr/bin/env python3
"""Generate PNG tile icons for procedural_frozenlake (run from repo root)."""

from __future__ import annotations

import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

from procedural_frozenlake.tile_icons import (  # noqa: E402
    TILE_GLARE,
    TILE_LAND,
    TILE_SLEIGH,
    build_native_tile_icons,
)

OUT_DIR = os.path.join(ROOT, "src", "procedural_frozenlake", "img")
NAMES = {
    TILE_LAND: "tree.png",
    TILE_GLARE: "glare_ice.png",
    TILE_SLEIGH: "sleigh.png",
}


def main() -> int:
    pygame.init()
    pygame.display.set_mode((1, 1))
    os.makedirs(OUT_DIR, exist_ok=True)
    icons = build_native_tile_icons()
    for tile, filename in NAMES.items():
        path = os.path.join(OUT_DIR, filename)
        pygame.image.save(icons[tile], path)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
