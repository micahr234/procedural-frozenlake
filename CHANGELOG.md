# Changelog

All notable changes to procedural-frozenlake are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Sleigh tiles now show a small numbered color-coded badge in the bottom-left corner in GUI rendering; both sleighs of a warp pair share the same badge, so linked pairs can be identified on the map.
- Goal presents render their reward value in a badge on the tile, and the present's bow is tinted by reward — yellow for low, green for high, normalized over the map's goal reward range (`goal_reward_low`/`goal_reward_high` when they differ, otherwise the rewards present on the map).

### Fixed
- The agent (elf) is now redrawn on top of special tile icons in GUI rendering. Previously it was hidden underneath the tile sprite when standing on an `L`/`R`/`O` cell — most visibly after a sleigh warp, which made warps look like they didn't happen.
- Under fog of war, warping through a sleigh now reveals both sleighs of the pair. Previously only the destination was marked visited, so the sleigh the agent entered stayed hidden as `?` until stepped on again.

### Changed
- Example notebook now generates a map with multiple start and goal tiles (`start_pos_prob` / `goal_pos_prob`) and varied per-goal rewards (`goal_reward_low=0.5`, `goal_reward_high=1.5`), and prints each goal's reward under the text map.
- Renamed ice floes to **sleighs**: `floe_pair_count` constructor parameter is now `sleigh_pair_count`, and the `info["map"]` JSON key `floes` is now `sleighs`. Tile letter (`O`) and warp behavior are unchanged.
- Redrew the special tile icons as 32×32 pixel-art sprites matching the original Gymnasium FrozenLake art style: land (`L`) is a snowy pine tree, glare ice (`R`) is an almost-white polished ice patch with a star gleam, and sleighs (`O`) are a red sleigh with a reindeer. Sprites have transparent backgrounds and are composited over the standard ice tile with nearest-neighbor scaling.
- Q\* value iteration now uses a separate `q_star_step_penalty` (default: match `step_penalty` when set, otherwise `-1e-6`) so optimal labels prefer shorter paths even when the live environment has no per-step cost — e.g. when an episode step limit would otherwise make wall-bumping look optimal.

## [0.3.0] - 2026-07-07

### Added
- Fixed `max_height × max_width` canvas for random maps; playable lake ice carved with jagged land (`L`) shorelines on all four sides.
- Tile-driven dynamics: glare ice (`R`) for per-tile slipperiness; ice floe (`O`) tiles warp between row-major pairs; `glare_prob`, `floe_pair_count`, `land_prob`, and `slippery_success_rate` constructor parameters.
- `land_prob` — optional interior land patches on frozen ice after the lake shoreline is carved.
- `info["map"]` JSON extended with `canvas`, `playable_count`, and `floes.pairs`.
- Distinct illustrated render icons for land (`L`, snowy winter tree with branches on ice), glare ice (`R`, smooth clear ice matching the lake palette without wave texture), and ice floe (`O`, floating chunk with warp portal) in GUI modes; colorized ANSI letters for the same tiles.
- Example notebook prints a bordered text map with row/column labels and tile legend after the first `reset()`.

### Changed
- Random maps always occupy the full `max_width × max_height` canvas; observation index always uses `row * max_width + col`.
- Land (`L`) tiles are hidden under fog-of-war until visited or bumped into.
- Lake shoreline generation uses symmetric edge erosion so trees are not biased toward the top-left.
- Default minimum playable tile count accounts for land margins (`max(4, (min_width - 2) × (min_height - 2))` unless `min_playable_tiles` is set).

### Removed
- `is_slippery` constructor flag — use `R` tiles or `glare_prob=1.0` for slippery behavior.

## [0.2.0] - 2026-07-07

### Added
- `fog_of_war` constructor flag: unvisited tiles render as `?` in `ansi`, `human`, and `rgb_array` modes; revealed tiles stay visible across episode `reset()` calls and clear only when the map regenerates.

### Changed
- Example notebook (`examples/random_rollout.ipynb`): tutorial layout, 20-episode random → Q\* policy ramp, embedded HTML5 replay (no on-disk GIF).

## [0.1.0] - 2026-07-07

### Added
- Initial release: `Procedural-FrozenLake-v1` Gymnasium environment with procedural map generation, variable rectangular grids, flexible start/goal placement, per-goal rewards, optional `info["map"]` and `info["q_star"]`, and configurable step penalties.
- Example notebook: `examples/random_rollout.ipynb`.
