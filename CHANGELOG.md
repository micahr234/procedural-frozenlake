# Changelog

All notable changes to procedural-frozenlake are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- README tile legend uses rendered tile icons from `src/procedural_frozenlake/img/` (same files the env loads for tree and mirror ice). Regenerate with `scripts/generate_tile_icons.sh`.

### Removed
- Runtime fallback that regenerated the `T`/`M`/`W` tile sprites in-process when the packaged PNGs were missing. The icons are static assets shipped with the package; rendering now assumes they exist. Regenerate them during development with `scripts/generate_tile_icons.sh`. The unused `build_special_tile_icons` helper was removed from `tile_icons`.
- `playable_count` field from the `info["map"]` JSON. It counted every non-tree tile (including holes) and nothing consumed it since the `min_playable_tiles`/`max_playable_tiles` rejection logic was removed.

## [0.4.0] - 2026-07-07

### Added
- `permute_obs` and `permute_actions` constructor flags: relabel observations (random permutation of canvas state indices) and actions (random permutation of the four action ids). Permutations are sampled with the map (from `map_seed`), resampled on `regenerate_map`, and exposed in `info["map"]` as `obs_permutation` / `action_permutation`. `info["q_star"]` is reported in external action order.
- Sleigh tiles now show a small numbered color-coded badge in the bottom-left corner in GUI rendering; both sleighs of a warp pair share the same badge, so linked pairs can be identified on the map.
- Goal presents render their reward value in a badge on the tile, and the present's bow is tinted by reward — yellow for low, green for high, normalized over the map's goal reward range (`goal_reward_low`/`goal_reward_high` when they differ, otherwise the rewards present on the map).

### Fixed
- `env.P` now carries the exact rewards the environment pays: per-goal rewards and `step_penalty` are baked into the transition tuples, and `step()` reads its reward straight from `P`. Previously `P` reported 1.0 for every goal and omitted the step penalty, so planning directly from `P` solved a different MDP than the live env.
- Under fog of war, bumping while standing on glare ice now reveals the tile the agent actually slid into. Previously the tile in the *intended* direction was revealed even when the agent slipped sideways — leaking unvisited tiles (including holes) the agent never touched, while the tile it really bumped stayed hidden.
- `info["q_star"]` is now zero at terminal states (goal/hole). Previously value iteration re-applied the goal reward (and step penalty) on the terminal self-loop, so the Q-values emitted on the final step of an episode wrongly suggested further return was obtainable.
- The agent (elf) is now redrawn on top of special tile icons in GUI rendering. Previously it was hidden underneath the tile sprite when standing on a `T`/`M`/`W` cell — most visibly after a sleigh warp, which made warps look like they didn't happen.
- Under fog of war, warping through a sleigh now reveals both sleighs of the pair. Previously only the destination was marked visited, so the sleigh the agent entered stayed hidden as `?` until stepped on again.
- ANSI rendering now colors tiles the same with fog of war on or off. Previously holes, goals, and starts were only colorized in fog mode.

### Changed
- Renamed the special tile letters to match what the tiles are: tree is now `T` (was `L` for "land"), warp sleigh is now `W` (was `O`, a holdover from "ice floe"), and glare/mirror ice is now `M` (was `R`). Affects map strings, `fixed_map` inputs, and `info["map"]` boards; behavior is unchanged.
- Renamed the `land_prob` constructor parameter to `tree_prob` to match the tree (`T`) tile terminology.
- `fog_of_war` now defaults to `True`; pass `fog_of_war=False` to render the full map.
- `info["map"]` is now emitted on every `reset()` and `step()` when `emit_map=True`. Previously it appeared only on the first reset after the map was (re)generated.
- Explicit `start_pos`/`goal_pos` values now raise instead of being silently skipped: positions outside the canvas raise `ValueError` at construction, and positions that can never be placed on playable lake ice surface in the generation error message.
- `min_width`/`max_width` and `min_height`/`max_height` now bound a lake **envelope**: all playable tiles fit inside the sampled `w × h` box, but the jagged shoreline may leave the lake smaller than the envelope. The envelope is placed uniformly at random on the canvas, so land border thickness varies per map (previously the lake was biased toward the canvas center).
- Example notebook now generates a map with multiple start and goal tiles (`start_pos_prob` / `goal_pos_prob`) and varied per-goal rewards (`goal_reward_low=0.5`, `goal_reward_high=1.5`), and prints each goal's reward under the text map.
- Renamed ice floes to **sleighs**: `floe_pair_count` constructor parameter is now `sleigh_pair_count`, and the `info["map"]` JSON key `floes` is now `sleighs`. Warp behavior is unchanged; the tile letter changed from `O` to `W` as part of the tile-letter rename above.
- Redrew the special tile icons as 32×32 pixel-art sprites matching the original Gymnasium FrozenLake art style: tree (`T`) is a snowy pine, glare ice (`M`) is an almost-white polished ice patch with a star gleam, and sleighs (`W`) are a red sleigh with a reindeer. Sprites have transparent backgrounds and are composited over the standard ice tile with nearest-neighbor scaling.
- Q\* value iteration now discounts with a new `q_star_gamma` constructor parameter (default `0.999`) instead of a per-step penalty, solving `env.P` as-is. Since `P` carries the exact env rewards (per-goal reward plus `step_penalty`), `info["q_star"]` is the true optimal value of the live MDP, and discounting keeps it pointed toward shorter paths.
- Constructor parameters are now validated: `hole_prob`, `tree_prob`, `glare_prob`, `slippery_success_rate`, `start_pos_prob`, and `goal_pos_prob` must be in `[0, 1]`; width/height bounds must satisfy `1 <= min <= max`; `sleigh_pair_count` must be non-negative; `q_star_gamma` must be in `(0, 1]`. Invalid values raise `ValueError` at construction instead of corrupting `P` or failing deep inside generation.
- Passing `start_pos`, `start_pos_prob`, `goal_pos`, or `goal_pos_prob` together with `fixed_map` now raises `ValueError`. Previously they were silently ignored.

### Removed
- `min_playable_tiles` and `max_playable_tiles` constructor parameters. They never drove mask generation (only post-hoc rejection); use the width/height bounds and `tree_prob` to shape the playable area.
- `q_star_step_penalty` constructor parameter — superseded by `q_star_gamma` discounting.

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
