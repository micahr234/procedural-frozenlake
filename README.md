# Procedural Frozen Lake

<p align="center"><img src="frozenlake.png" width="400"/></p>

A Gymnasium environment that extends Frozen Lake with **procedurally generated maps**.

`Procedural-FrozenLake-v1` optionally supports:

- **Random lake shapes** — jagged shorelines and variable tree borders on a fixed canvas
- **Tree tiles (`T`)** — impassable obstacles around the lake, with optional patches inside
- **Mirror ice tiles (`M`)** — sprinkle slippery tiles where you want them, instead of a global slippery switch
- **Warp sleigh tiles (`W`)** — paired tiles that teleport you between them
- **Multiple starts and goals** — pin exact tiles or place them randomly, with a different reward per goal
- **Fresh maps without rebuilding** — regenerate the layout from `reset()` in the same env instance
- **Supervision signals in `info`** — the map blueprint (`emit_map`) and optimal Q-values for the current state (`emit_q_star`)
- **Fog of war** — unvisited tiles render as `?` until explored (on by default; render-only)
- **Shuffled state / action IDs** — relabel agent-facing observation and action numbers with `permute_obs` / `permute_actions`

Every knob is detailed under [Constructor parameters](#constructor-parameters).

## News

- **2026-07-08 — v1.0.0 is out!** — Stable API: border-based lake generation (`width`/`height`/`min_border`/`max_border`/`shoreline_jaggedness`), `info["map"]` as a dict, `mirror_prob`, and clearer docs. See [CHANGELOG.md](CHANGELOG.md).
- **2026-07-07 — v0.4.0** — Observation and action permutations, tile letter rename (`T`/`M`/`W`), fog of war on by default, exact rewards in `env.P`. See [CHANGELOG.md](CHANGELOG.md).

See [CHANGELOG.md](CHANGELOG.md) for the full release history.

## Install

```bash
pip install procedural-frozenlake
```

For GUI / `rgb_array` rendering you also need pygame (via Gymnasium's toy-text extra):

```bash
pip install "gymnasium[toy-text]"
```

Or install the examples extra: `pip install "procedural-frozenlake[examples]"`.

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
)
obs, info = env.reset(seed=1)
print(info["map"]["board"])  # list of row strings
print(info["q_star"])        # shape (4,) Q* for the current state

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

See [`examples/rollout.ipynb`](examples/rollout.ipynb) for a tutorial notebook: multi-episode rollout, multiple starts and goals with per-goal rewards, fog-of-war, Q\* labels, and an embedded replay video.

## Environment

**ID:** `Procedural-FrozenLake-v1`

This env follows the conventions of Gymnasium's [FrozenLake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/): the same four actions, flat state indices (`row * width + col`), and reward on reaching a goal. What differs:

- **One map per env by default.** The map is generated lazily on the first `reset()` and reused across episodes. Pass `options={"regenerate_map": True}` to `reset()` when you want a fresh layout.
- **Two independent random streams.** `map_seed` fixes the *map* — lake shape, tile placement, goal rewards, and any permutations. `reset(seed=…)` only affects *episode* randomness — which start tile you begin on, which way you slip on mirror ice. A reset seed never draws a new map.

### Tile legend

| Icon | Tile | Name | Behavior |
|:----:|:----:|------|----------|
| <img src="src/procedural_frozenlake/img/tile_s.png" width="40" alt="Start tile"/> | `S` | Start | Walkable; deterministic movement |
| <img src="src/procedural_frozenlake/img/tile_f.png" width="40" alt="Frozen tile"/> | `F` | Frozen | Normal safe ice; deterministic movement |
| <img src="src/procedural_frozenlake/img/tile_m.png" width="40" alt="Mirror ice tile"/> | `M` | Mirror ice | Slippery ice (stochastic sliding when standing on it) |
| <img src="src/procedural_frozenlake/img/tile_w.png" width="40" alt="Warp sleigh tile"/> | `W` | Warp sleigh | Warp to paired sleigh on entry; both tiles in a pair share the same numbered badge |
| <img src="src/procedural_frozenlake/img/tile_h.png" width="40" alt="Hole tile"/> | `H` | Hole | Terminal — fall through; reward `0` |
| <img src="src/procedural_frozenlake/img/tile_g.png" width="40" alt="Goal tile"/> | `G` | Goal | Terminal — success; reward shown in badge, bow tinted yellow (low) to green (high) |
| <img src="src/procedural_frozenlake/img/tile_t.png" width="40" alt="Tree tile"/> | `T` | Tree | Impassable shoreline and optional interior patches |

### Constructor parameters

**Map generation**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `map_seed` | `None` | Seed for map generation (independent of reset seed) |
| `fixed_map` | `None` | Fixed layout: list of row strings, or a blueprint dict (`board` / `rewards` / `sleighs` / permutations — same shape as `info["map"]`). Disables random generation and cannot be combined with `start_pos`/`goal_pos` options |
| `width`, `height` | `8`, `8` | Fixed canvas dimensions; state indices run `0 .. width*height-1` |
| `min_border`, `max_border` | `1`, `2` | Tree margin sampled uniformly on every side; playable lake ice fills the interior inset |
| `shoreline_jaggedness` | `1` | Max tiles of shoreline variation per edge (`0` = smooth rectangle; higher = deeper bays and longer peninsulas into the border band) |
| `hole_prob` | `0.2` | Probability a tile becomes a hole `H` |
| `tree_prob` | `0.0` | Probability a frozen ice tile becomes an interior tree `T` after the lake is carved |
| `mirror_prob` | `0.0` | Probability a frozen (`F`) tile becomes mirror ice `M` |
| `sleigh_pair_count` | `0` | Number of sleigh warp pairs (`2 × count` `W` tiles); pairs linked in row-major order |
| `start_pos`, `start_pos_prob` | `None`, `None` | Fixed start tile(s) as canonical flat index / list of indices, or per-tile Bernoulli probability of placing starts |
| `goal_pos`, `goal_pos_prob` | `None`, `None` | Same for goals |
| `min_hops` | `3` | Minimum shortest-path length from start to goal (deterministic BFS; ignores mirror slip) |
| `max_tries` | `10_000` | Generation attempts before giving up with an error |

**Dynamics and rewards**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `slippery_success_rate` | `1/3` | Intended-direction success rate on mirror ice `M` |
| `goal_reward_low`, `goal_reward_high` | `1.0`, `1.0` | Per-goal reward sampling bounds (`low` must be `<= high`) |

**Supervision signals in `info`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `emit_map` | `False` | Inject map layout dict in `info["map"]` on every `reset()` and `step()` |
| `emit_q_star` | `False` | Inject optimal Q-values for the current state in `info["q_star"]` (shape `(4,)`; zero at terminal states) |
| `q_star_gamma` | `0.999` | Discount for Q\* value iteration over `env.P` |

**Relabeling**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `permute_obs` | `False` | Relabel observations with a random permutation of canvas state indices |
| `permute_actions` | `False` | Relabel the four actions with a random permutation |

Both permutations are sampled with the map (from `map_seed`) and resampled when the map regenerates. Only agent-facing IDs change — the board in `info["map"]` and the position kwargs stay in canonical grid coordinates.

- **`permute_obs`** — the agent no longer sees the canonical flat index. If the agent stands at canonical index `5` and `info["map"]["obs_permutation"][5]` is `17`, then `reset`/`step` return observation `17`. Decode with the permutation, or keep using the canonical board in `info["map"]`.
- **`permute_actions`** — the four movement IDs are shuffled the same way. If `action_permutation` is `[2, 0, 3, 1]`, external action `0` means Right, `1` means Left, and so on. `info["q_star"]` is already ordered by these external action IDs.

**Rendering**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fog_of_war` | `True` | Hide unvisited tiles as `?` in render modes only (trees revealed when visited or bumped); set `False` to render the full map |
| `render_mode` | `None` | Standard Gymnasium render mode: `"ansi"`, `"human"`, or `"rgb_array"` |

### `info["map"]` schema

When `emit_map=True`, `info["map"]` is a blueprint `dict` you can pass back as `fixed_map` to rebuild the same layout:

| Key | Type | Description |
|-----|------|-------------|
| `board` | `list[str]` | Row strings of tile letters (size is implied by the board) |
| `rewards` | `dict[int, float]` | Goal reward by **canonical** flat state index |
| `sleighs` | `dict` | `{"pairs": [[a, b], …]}` canonical state indices (row-major pairing) |
| `obs_permutation` | `list[int]` | Present when observations were relabeled |
| `action_permutation` | `list[int]` | Present when actions were relabeled |

Example:

```python
obs, info = env.reset()
clone = gym.make("Procedural-FrozenLake-v1", fixed_map=info["map"])
```

### Reset options

Pass `options={"regenerate_map": True}` to `reset()` to generate a new map. Not available when `fixed_map` is set. Unknown option keys raise `ValueError`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
