# Procedural Frozen Lake

<p align="center"><img src="frozenlake.png" width="400"/></p>

A Gymnasium environment that extends Frozen Lake with **procedurally generated maps**.

`Procedural-FrozenLake-v1` provides:

- **Jagged lake shorelines** — every random map is a lake on a fixed canvas, bounded by impassable trees (`T`) with independently varying edges per row and column. A lake envelope sampled within `min_width..max_width` × `min_height..max_height` is placed uniformly at random; all playable tiles fit inside it, though the jagged shoreline may leave the lake smaller than the envelope.
- **Tile-driven physics** — glare ice (`M`, mirror ice) is locally slippery; sleighs (`W`, warp) teleport between paired tiles; no global `is_slippery` flag.
- **Flexible start and goal placement** — fixed positions, lists of positions, or probabilistic placement; multiple starts and goals supported.
- **Per-goal rewards** — sample or specify a different reward for each goal tile.
- **Fresh maps on reset** — pass `options={"regenerate_map": True}` to sample a new valid layout without rebuilding the env.
- **Stable observation space** — always `Discrete(max_width * max_height)`; state index is `row * max_width + col` on the fixed canvas.
- **Optional supervision signals** — `emit_map=True` and `emit_q_star=True` expose the layout and optimal Q-values in `info` on every `reset()` and `step()`.
- **Fog of war rendering** — on by default: unvisited tiles render as `?`, including trees; bumping a tree reveals it; warping reveals both sleighs of the pair; exploration persists until map regeneration. Pass `fog_of_war=False` for a fully visible map.
- **Observation and action relabeling** — `permute_obs=True` / `permute_actions=True` scramble state indices and action ids with permutations sampled alongside the map (and exposed in `info["map"]`), so agents can't rely on the canonical grid numbering.

## News

- **2026-07-07 — v0.4.0 is out!** — Observation and action permutations (`permute_obs` / `permute_actions`): state indices and action ids are relabeled with random permutations sampled alongside the map, so agents can't rely on canonical numbering. Also: new tile letters (`T`/`M`/`W`), fog of war on by default, `env.P` carries exact rewards, Q\* discounted by `q_star_gamma`, constructor validation, lake envelope placement. See [CHANGELOG.md](CHANGELOG.md).
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

### Tile legend

| Tile | Name | Behavior |
|------|------|----------|
| `S` | Start | Walkable; deterministic movement |
| `F` | Frozen | Normal safe ice; deterministic movement |
| `M` | Mirror (glare) ice | Slippery ice (stochastic sliding when standing on it) |
| `W` | Warp sleigh | Warp to paired sleigh on entry (row-major pairing) |
| `H` | Hole | Terminal — fall through |
| `G` | Goal | Terminal — success |
| `T` | Tree | Impassable shoreline and optional interior patches |

In `human` / `rgb_array` rendering, `T` / `M` / `W` appear as pixel-art sprites drawn in the original FrozenLake style (snowy pine tree, almost-white polished ice patch with a star gleam, red sleigh with a reindeer) over the standard ice tile. Each sleigh also carries a small numbered color-coded badge in its bottom-left corner; linked sleighs share the same badge. Goal presents show their reward in a badge, and the bow is tinted from yellow (low reward) to green (high reward) relative to the map's reward range. ANSI mode colorizes tiles the same with fog on or off: trees white, glare ice cyan, sleighs and holes blue, goals green, starts yellow.

Glare ice is slippery because of a thin meltwater film on mirror-smooth ice — the dangerous patches on a frozen lake. `glare_prob=1.0` gives a map similar to the classic slippery Gymnasium FrozenLake (start tiles stay deterministic; only plain frozen tiles become glare).

### Constructor parameters

**Map generation**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `map_seed` | `None` | Seed for map generation (independent of reset seed) |
| `fixed_map` | `None` | Fixed layout (list of row strings or dict with `board`/`rewards`); disables random generation and cannot be combined with `start_pos`/`goal_pos` options |
| `min_width`, `max_width` | `3`, `8` | Width bounds for the sampled lake envelope; all playable tiles fit inside it (canvas width = `max_width`) |
| `min_height`, `max_height` | `3`, `8` | Height bounds for the sampled lake envelope; all playable tiles fit inside it (canvas height = `max_height`) |
| `hole_prob` | `0.2` | Probability a tile becomes a hole `H` |
| `tree_prob` | `0.0` | Probability a frozen ice tile becomes an interior tree `T` after the lake is carved |
| `glare_prob` | `0.0` | Probability a frozen (`F`) tile becomes glare ice `M` |
| `sleigh_pair_count` | `0` | Number of sleigh warp pairs (`2 × count` `W` tiles) |
| `start_pos`, `start_pos_prob` | `None`, `None` | Fixed start tile(s) or probability of placing starts; explicit positions raise an error when they can't be placed on lake ice |
| `goal_pos`, `goal_pos_prob` | `None`, `None` | Fixed goal tile(s) or probability of placing goals; explicit positions raise an error when they can't be placed on lake ice |
| `min_hops` | `3` | Minimum shortest-path length from start to goal |
| `max_tries` | `10_000` | Generation attempts before giving up with an error |

**Dynamics and rewards**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `slippery_success_rate` | `1/3` | Intended-direction success rate on glare ice `M` |
| `step_penalty` | `0.0` | Added to every step reward (e.g. `-0.01`); baked into `env.P` transition rewards |
| `goal_reward_low`, `goal_reward_high` | `1.0`, `1.0` | Per-goal reward sampling bounds |

**Supervision signals in `info`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `emit_map` | `False` | Inject map layout in `info["map"]` on every `reset()` and `step()` |
| `emit_q_star` | `False` | Inject optimal Q-values in `info["q_star"]` (zero at terminal states) |
| `q_star_gamma` | `0.999` | Discount for Q\* value iteration over `env.P` (whose rewards include `step_penalty`), so Q\* is the optimal value of the live MDP and prefers shorter paths |

**Relabeling**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `permute_obs` | `False` | Relabel observations with a random permutation of canvas state indices, sampled with the map |
| `permute_actions` | `False` | Relabel the four actions with a random permutation, sampled with the map |

**Rendering**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fog_of_war` | `True` | Hide unvisited tiles as `?` (trees revealed when visited or bumped); set `False` to render the full map |
| `render_mode` | `None` | Standard Gymnasium render mode: `"ansi"`, `"human"`, or `"rgb_array"` |

### Reset options

Pass `options={"regenerate_map": True}` to `reset()` to generate a new map. Not available when `fixed_map` is set.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
