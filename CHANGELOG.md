# Changelog

All notable changes to procedural-frozenlake are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.2.0] - 2026-07-07

### Added
- `fog_of_war` constructor flag: unvisited tiles render as `?` in `ansi`, `human`, and `rgb_array` modes; revealed tiles stay visible across episode `reset()` calls and clear only when the map regenerates.

### Changed
- Example notebook (`examples/random_rollout.ipynb`): tutorial layout, 20-episode random → Q\* policy ramp, embedded HTML5 replay (no on-disk GIF).

## [0.1.0] - 2026-07-07

### Added
- Initial release: `Procedural-FrozenLake-v1` Gymnasium environment with procedural map generation, variable rectangular grids, flexible start/goal placement, per-goal rewards, optional `info["map"]` and `info["q_star"]`, and configurable step penalties.
- Example notebook: `examples/random_rollout.ipynb`.
