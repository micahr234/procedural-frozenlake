# Procedural Frozen Lake

A Gymnasium environment that extends Frozen Lake with **procedurally generated maps**.

`Procedural-FrozenLake-v1` provides:

- **Variable rectangular grids** — independent `min_width`/`max_width` and `min_height`/`max_height` bounds.
- **Flexible start and goal placement** — fixed positions, lists of positions, or probabilistic placement across the grid; multiple starts and goals are supported.
- **Per-goal rewards** — sample or specify a different reward for each goal tile.
- **Fresh maps on reset** — pass `options={"regenerate_map": True}` to sample a new valid layout without rebuilding the env.
- **Stable observation space** — variable-size maps share a fixed `Discrete(max_width * max_height)` space so it does not change when maps regenerate.
- **Optional supervision signals** — `emit_map=True` and `emit_q_star=True` expose the layout and optimal Q-values in `info`.

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

See [`examples/random_rollout.ipynb`](examples/random_rollout.ipynb) for a worked example.

## Environment

**ID:** `Procedural-FrozenLake-v1`

Maps are generated lazily on the first `reset()`, not during construction. By default, each env instance keeps one generated map across resets.

### Constructor parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `is_slippery` | `False` | Stochastic sliding (standard FrozenLake behavior) |
| `min_hops` | `3` | Minimum shortest-path length from start to goal |
| `min_width`, `max_width` | `3`, `8` | Map width bounds |
| `min_height`, `max_height` | `3`, `8` | Map height bounds |
| `hole_prob` | `0.2` | Probability a tile becomes a hole |
| `start_pos`, `start_pos_prob` | `None` | Fixed start tile(s) or probability of placing starts |
| `goal_pos`, `goal_pos_prob` | `None` | Fixed goal tile(s) or probability of placing goals |
| `fixed_map` | `None` | Fixed layout (list of row strings or dict with `board`/`rewards`) |
| `emit_q_star` | `False` | Inject optimal Q-values in `info["q_star"]` |
| `emit_map` | `False` | Inject map layout in `info["map"]` |
| `goal_reward_low`, `goal_reward_high` | `1.0`, `1.0` | Per-goal reward sampling bounds |
| `step_penalty` | `0.0` | Added to every step reward (e.g. `-0.01`) |
| `map_seed` | `None` | Seed for map generation (independent of reset seed) |

### Reset options

Pass `options={"regenerate_map": True}` to `reset()` to generate a new map. Not available when `fixed_map` is set.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
