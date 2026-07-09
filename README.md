# Procedural Frozen Lake

<p align="center"><img src="frozenlake.png" width="400"/></p>

A Gymnasium environment that extends Frozen Lake with **procedurally generated maps**.

`Procedural-FrozenLake-v1` optionally supports:

- **Random lake shapes** — Each map is a lake on a fixed canvas with uneven, jagged edges instead of a plain rectangle. Tree border thickness is sampled from your `min_border`/`max_border` bounds on every side; shoreline bays and peninsulas are controlled by `shoreline_jaggedness`.
- **Tree tiles (`T`)** — Impassable tiles that block movement. Every map gets a tree border around the lake; sprinkle more inside with `tree_prob`.
- **Mirror ice tiles (`M`)** — Slippery ice that makes you slide like classic FrozenLake. Sprinkle them with `mirror_prob` instead of flipping a global slippery switch.
- **Warp sleigh tiles (`W`)** — Paired tiles that teleport you to each other when you step on one. Add them with `sleigh_pair_count` (two `W` tiles per pair). Pairs are linked in **row-major scan order**.
- **Multiple start tiles (`S`)** — Pin one spot, pass a list, or sample placement with `start_pos` / `start_pos_prob`. Positions are canonical flat canvas indices (`row * width + col`, top-to-bottom, left-to-right).
- **Multiple goal tiles (`G`)** — Same for goals with `goal_pos` / `goal_pos_prob`.
- **Different rewards per goal** — Each goal tile can pay its own amount, either sampled between bounds or set explicitly. Non-goal transitions pay `0`.
- **Fresh maps without rebuilding** — Pass `options={"regenerate_map": True}` on `reset()` to draw a new layout in the same env instance.
- **Map layout in `info`** — `emit_map=True` puts a Python `dict` in `info["map"]` on every reset and step.
- **Optimal Q-values in `info`** — `emit_q_star=True` puts a length-4 vector in `info["q_star"]` (Q\* for the **current** state, in external action order). Use `env.unwrapped.compute_q_table()` for the full table.
- **Hidden tiles until explored** — Fog of war is on by default: unvisited tiles show as `?` in **render** modes only. The observation is still the full state index. Turn off with `fog_of_war=False`.
- **Shuffled state / action numbers** — `permute_obs` / `permute_actions` relabel agent-facing IDs. The board in `info["map"]` and position kwargs stay in **canonical** grid coordinates.

## News

- **2026-07-08 — v1.0.0 is out!** — Stable API: border-based lake generation (`width`/`height`/`min_border`/`max_border`/`shoreline_jaggedness`), `info["map"]` as a dict, `mirror_prob`, Gymnasium `max_episode_steps=100`, and clearer docs. See [CHANGELOG.md](CHANGELOG.md).
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

Maps are generated lazily on the first `reset()`, not during construction. **By default, the same map is reused across episodes** — only pass `options={"regenerate_map": True}` when you want a fresh layout.

**Two RNG streams:**

| Seed | Controls |
|------|----------|
| `map_seed` | Layout, goal rewards, permutations, border sampling |
| `reset(seed=…)` | Episode randomness (which start tile, mirror-ice slips) |

`reset(seed=…)` does **not** regenerate the map unless you also pass `regenerate_map=True`.

Registered with `max_episode_steps=100` and `nondeterministic=True`, so `truncated` can become `True` under the Gymnasium `TimeLimit` wrapper. Actions are `0=Left, 1=Down, 2=Right, 3=Up` unless `permute_actions=True`.

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
| `fixed_map` | `None` | Fixed layout (list of row strings or dict with `board`/`rewards`); disables random generation and cannot be combined with `start_pos`/`goal_pos` options |
| `width`, `height` | `8`, `8` | Fixed canvas dimensions; state indices are `0 .. width*height-1`, numbered top-to-bottom, left-to-right (`index = row * width + col`) |
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
| `permute_obs` | `False` | Relabel observations with a random permutation of canvas state indices, sampled with the map. The board in `info["map"]` stays canonical. |
| `permute_actions` | `False` | Relabel the four actions with a random permutation, sampled with the map |

**Rendering**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fog_of_war` | `True` | Hide unvisited tiles as `?` in render modes only (trees revealed when visited or bumped); set `False` to render the full map |
| `render_mode` | `None` | Standard Gymnasium render mode: `"ansi"`, `"human"`, or `"rgb_array"` |

### `info["map"]` schema

When `emit_map=True`, `info["map"]` is a `dict` with:

| Key | Type | Description |
|-----|------|-------------|
| `board` | `list[str]` | Row strings of tile letters |
| `rewards` | `dict[int, float]` | Goal reward by **canonical** flat state index |
| `canvas` | `dict` | `{"width": int, "height": int}` |
| `sleighs` | `dict` | `{"pairs": [[a, b], …]}` canonical state indices |
| `border` | `int` | Sampled tree border (random maps only) |
| `obs_permutation` | `list[int]` | Present when `permute_obs=True` |
| `action_permutation` | `list[int]` | Present when `permute_actions=True` |

### Reset options

Pass `options={"regenerate_map": True}` to `reset()` to generate a new map. Not available when `fixed_map` is set. Unknown option keys raise `ValueError`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
