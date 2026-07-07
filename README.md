# Procedural Frozen Lake

<p align="center"><img src="frozenlake.png" width="400"/></p>

A Gymnasium environment that extends Frozen Lake with **procedurally generated maps**.

`Procedural-FrozenLake-v1` provides:

- **Jagged lake shorelines** — every random map is a rectangular lake on a fixed canvas, bounded by impassable land (`L`) with independently varying edges per row and column.
- **Tile-driven physics** — glare ice (`R`) is locally slippery; sleighs (`O`) warp between paired tiles; no global `is_slippery` flag.
- **Flexible start and goal placement** — fixed positions, lists of positions, or probabilistic placement; multiple starts and goals supported.
- **Per-goal rewards** — sample or specify a different reward for each goal tile.
- **Fresh maps on reset** — pass `options={"regenerate_map": True}` to sample a new valid layout without rebuilding the env.
- **Stable observation space** — always `Discrete(max_width * max_height)`; state index is `row * max_width + col` on the fixed canvas.
- **Optional supervision signals** — `emit_map=True` and `emit_q_star=True` expose the layout and optimal Q-values in `info`.
- **Fog of war rendering** — `fog_of_war=True` hides unvisited tiles as `?`, including trees; bumping a tree reveals it; warping reveals both sleighs of the pair; exploration persists until map regeneration.

## News

- **2026-07-07 — v0.3.0** — Variable map boundaries (land shorelines, glare ice, ice floe warps), fixed `max_width × max_height` canvas, tile-driven slipperiness. See [CHANGELOG.md](CHANGELOG.md).

See [CHANGELOG.md](CHANGELOG.md) for the full release history.

## Install

```bash
pip install procedural-frozenlake
```

For development:

```bash
git clone https://github.com/micahr234/procedural-frozenlake.git
cd procedural-frozenlake
source scripts/install.sh
```

## Quick start

Importing the package registers the environment with Gymnasium:

```python
import gymnasium as gym
import procedural_frozenlake  # registers Procedural-FrozenLake-v1

env = gym.make(
    "Procedural-FrozenLake-v1",
    map_seed=0,
    emit_map=True,
    emit_q_star=True,
    step_penalty=-0.01,
)
obs, info = env.reset(seed=1)
print(info["map"])    # JSON string with board layout and goal rewards
print(info["q_star"]) # optimal Q-values for the current state

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

See [`examples/random_rollout.ipynb`](examples/random_rollout.ipynb) for a tutorial notebook: multi-episode rollout, multiple starts and goals with per-goal rewards, fog-of-war, Q\* labels, and an embedded replay video.

## Environment

**ID:** `Procedural-FrozenLake-v1`

Maps are generated lazily on the first `reset()`, not during construction. **By default, the same map is reused across episodes** — only pass `options={"regenerate_map": True}` when you want a fresh layout. `reset(seed=…)` still controls episode-level randomness (e.g. start sampling); it does not regenerate the map unless you ask.

### Observation space

**Always** `Discrete(max_width * max_height)` — set at construction and unchanged across `reset()` / `regenerate_map`.

Decode any observation as:

```text
row = obs // max_width
col = obs % max_width
```

Random maps use a fixed `max_height × max_width` canvas. Land (`L`) cells surround the lake; the agent only occupies playable tiles (`S`, `F`, `R`, `O`, `G`).

### Tile legend

| Tile | Name | Behavior |
|------|------|----------|
| `S` | Start | Walkable; deterministic movement |
| `F` | Frozen | Normal safe ice; deterministic movement |
| `R` | Glare ice | Slippery ice (stochastic sliding when standing on it) |
| `O` | Sleigh | Warp to paired sleigh on entry (row-major pairing) |
| `H` | Hole | Terminal — fall through |
| `G` | Goal | Terminal — success |
| `L` | Tree | Impassable shoreline and optional interior patches |

In `human` / `rgb_array` rendering, `L` / `R` / `O` appear as pixel-art sprites drawn in the original FrozenLake style (snowy pine tree, almost-white polished ice patch with a star gleam, red sleigh with a reindeer) over the standard ice tile. Each sleigh also carries a small numbered color-coded badge in its bottom-left corner; linked sleighs share the same badge. Goal presents show their reward in a badge, and the bow is tinted from yellow (low reward) to green (high reward) relative to the map's reward range. ANSI mode colorizes those letters (white / cyan / blue).

Glare ice is slippery because of a thin meltwater film on mirror-smooth ice — the dangerous patches on a frozen lake. For a fully slippery map like classic Gymnasium FrozenLake, use `glare_prob=1.0`.

### Constructor parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_hops` | `3` | Minimum shortest-path length from start to goal |
| `min_width`, `max_width` | `3`, `8` | Lake width bounds (canvas width = `max_width`) |
| `min_height`, `max_height` | `3`, `8` | Lake height bounds (canvas height = `max_height`) |
| `land_prob` | `0.0` | Probability a frozen ice tile becomes interior land after the lake is carved |
| `glare_prob` | `0.0` | Probability a playable tile becomes glare ice `R` |
| `sleigh_pair_count` | `0` | Number of sleigh warp pairs (`2 × count` `O` tiles) |
| `slippery_success_rate` | `1/3` | Intended-direction success rate on glare ice `R` |
| `hole_prob` | `0.2` | Probability a tile becomes a hole |
| `start_pos`, `start_pos_prob` | `None` | Fixed start tile(s) or probability of placing starts |
| `goal_pos`, `goal_pos_prob` | `None` | Fixed goal tile(s) or probability of placing goals |
| `fixed_map` | `None` | Fixed layout (list of row strings or dict with `board`/`rewards`) |
| `emit_q_star` | `False` | Inject optimal Q-values in `info["q_star"]` |
| `emit_map` | `False` | Inject map layout in `info["map"]` |
| `goal_reward_low`, `goal_reward_high` | `1.0`, `1.0` | Per-goal reward sampling bounds |
| `step_penalty` | `0.0` | Added to every step reward (e.g. `-0.01`) |
| `q_star_step_penalty` | `None` | Per-step cost in value iteration only; defaults to `step_penalty` when non-zero, otherwise a small epsilon (`-1e-6`) so Q\* prefers shorter paths |
| `map_seed` | `None` | Seed for map generation (independent of reset seed) |
| `fog_of_war` | `False` | Hide unvisited tiles as `?` (trees revealed when visited or bumped) |

### Reset options

Pass `options={"regenerate_map": True}` to `reset()` to generate a new map. Not available when `fixed_map` is set.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
