"""Procedural Frozen Lake environment with generated maps and optional q_star labels."""

from __future__ import annotations

import random
from collections.abc import Mapping
from os import path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register, registry
from gymnasium.envs.toy_text.frozen_lake import DOWN, LEFT, RIGHT, UP, FrozenLakeEnv
from gymnasium.envs.toy_text.utils import categorical_sample

from procedural_frozenlake.mapgen import (
    build_sleigh_partners,
    generate_valid_map,
    goal_states_from_gridmap as _goal_states_from_gridmap,
    normalize_and_validate_fixed_map,
)
from procedural_frozenlake.tile_icons import (
    SPECIAL_TILES,
    build_sleigh_pair_badges,
    goal_reward_icon,
)
from procedural_frozenlake.tiles import (
    TILE_FROZEN,
    TILE_GOAL,
    TILE_HOLE,
    TILE_MIRROR,
    TILE_SLEIGH,
    TILE_START,
    TILE_TREE,
)
from procedural_frozenlake.value_iteration import solve_tabular_mdp

PROCEDURAL_FROZENLAKE_ENV_ID = "Procedural-FrozenLake-v1"

_TILE_ICON_FILES = {
    TILE_TREE: "tile_t.png",
    TILE_MIRROR: "tile_m.png",
    TILE_SLEIGH: "tile_w_overlay.png",
}


def _validate_prob(name: str, value: float) -> float:
    v = float(value)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}.")
    return v


def _neighbor(
    row: int, col: int, action: int, nrow: int, ncol: int
) -> tuple[int, int]:
    """Apply one cardinal move, clamped to the grid (Gymnasium LEFT/DOWN/RIGHT/UP)."""
    if action == LEFT:
        col = max(col - 1, 0)
    elif action == DOWN:
        row = min(row + 1, nrow - 1)
    elif action == RIGHT:
        col = min(col + 1, ncol - 1)
    elif action == UP:
        row = max(row - 1, 0)
    return row, col


class ProceduralFrozenLakeEnv(FrozenLakeEnv):
    """Procedural Frozen Lake variant with generated valid map and optional q_star info.

    Maps use a fixed canvas of ``height × width`` tiles (random generation) or the
    dimensions of ``fixed_map``. Each random map samples a tree border thickness from
    ``min_border..max_border`` tiles on every side; playable lake ice fills the interior
    inset, though the jagged shoreline may leave the lake smaller than that rectangle
    or push peninsulas of ice into the tree border band. Shoreline variation is
    controlled by ``shoreline_jaggedness`` (0 = smooth rectangle, higher = deeper
    bays and longer peninsulas). Tree (``T``) tiles fill the border band and are
    impassable; mirror ice (``M``) tiles are locally slippery; sleighs (``W``) warp
    between paired tiles.

    Dual seeds: ``map_seed`` controls map generation (and permutations sampled with
    the map); ``reset(seed=...)`` seeds the episode RNG used for stochastic dynamics
    (e.g. mirror slip). They are independent.

    ``goal_reward_low`` / ``goal_reward_high`` sample one reward **per goal tile** when
    the map is generated. Non-goal transitions pay ``0``; goals pay their per-goal
    reward. The transition matrix ``P`` carries those exact rewards, so planning
    directly from ``P`` matches live behavior. When ``emit_q_star`` is True,
    :meth:`compute_q_table` solves ``P`` with discount ``q_star_gamma`` (default
    ``0.999``), so Q\\* is the optimal value of the live MDP and prefers shorter
    paths. Terminal states (goal/hole) and tree states have Q-values of zero.
    ``emit_q_star`` puts a length-4 vector in ``info["q_star"]`` for the current
    state (ordered by external action id when ``permute_actions`` is on).

    When ``emit_map`` is True, every :meth:`reset` and :meth:`step` puts a **dict** in
    ``info["map"]`` with ``board``, ``rewards`` (int keys → float), ``canvas``,
    ``sleighs``, optional ``border``, and — when enabled — ``obs_permutation`` /
    ``action_permutation``.

    Positions (``start_pos`` / ``goal_pos``) and map reward keys use **canonical flat
    canvas indices**: numbered top-to-bottom, left-to-right as
    ``index = row * width + col``.

    ``permute_obs=True`` relabels observations with a random permutation of the canvas
    state indices; ``permute_actions=True`` relabels the four actions. The board and
    position kwargs stay in canonical index space; only agent-facing IDs are
    relabeled. Both permutations are sampled with the map (from ``map_seed``) and
    resampled when the map regenerates.

    Pass ``options={"regenerate_map": True}`` to :meth:`reset` to sample a fresh map.
    That is the only supported reset option.

    With ``fog_of_war=True`` (the default), unvisited tiles render as ``?``, including
    trees (``T``). Fog is render-only and does not change observations or dynamics.
    Warping through a sleigh reveals both sleighs of the pair. Bumping into a blocked
    tile reveals it. Pass ``fog_of_war=False`` to render the full map.

    Episodes may truncate via Gymnasium's ``TimeLimit`` wrapper
    (``max_episode_steps=100`` on :func:`ensure_registered`); this env itself always
    returns ``truncated=False`` from :meth:`step`.
    """

    def __init__(
        self,
        render_mode: str | None = None,
        # Map generation
        map_seed: int | None = None,
        fixed_map: list[str] | tuple[str, ...] | Mapping[str, Any] | None = None,
        width: int = 8,
        height: int = 8,
        min_border: int = 1,
        max_border: int = 2,
        shoreline_jaggedness: int = 1,
        hole_prob: float = 0.2,
        tree_prob: float = 0.0,
        mirror_prob: float = 0.0,
        sleigh_pair_count: int = 0,
        start_pos: int | list[int] | None = None,
        start_pos_prob: float | None = None,
        goal_pos: int | list[int] | None = None,
        goal_pos_prob: float | None = None,
        min_hops: int = 3,
        max_tries: int = 10_000,
        # Dynamics and rewards
        slippery_success_rate: float = 1.0 / 3.0,
        goal_reward_low: float = 1.0,
        goal_reward_high: float = 1.0,
        # Supervision signals in info
        emit_map: bool = False,
        emit_q_star: bool = False,
        q_star_gamma: float = 0.999,
        # Observation/action relabeling
        permute_obs: bool = False,
        permute_actions: bool = False,
        # Rendering
        fog_of_war: bool = True,
    ):
        hole_prob = _validate_prob("hole_prob", hole_prob)
        tree_prob = _validate_prob("tree_prob", tree_prob)
        mirror_prob = _validate_prob("mirror_prob", mirror_prob)
        slippery_success_rate = _validate_prob(
            "slippery_success_rate", slippery_success_rate
        )
        if start_pos_prob is not None:
            _validate_prob("start_pos_prob", start_pos_prob)
        if goal_pos_prob is not None:
            _validate_prob("goal_pos_prob", goal_pos_prob)
        if width < 1 or height < 1:
            raise ValueError(
                f"width and height must be >= 1, got {width} and {height}."
            )
        if min_border < 0 or max_border < 0:
            raise ValueError(
                f"min_border and max_border must be >= 0, got {min_border} and {max_border}."
            )
        if min_border > max_border:
            raise ValueError(
                f"Border bounds must satisfy min <= max, got {min_border}..{max_border}."
            )
        if width - 2 * max_border < 2 or height - 2 * max_border < 2:
            raise ValueError(
                f"Canvas must fit a lake inset after max_border={max_border}: need "
                f"width - 2*max_border >= 2 and height - 2*max_border >= 2, got "
                f"{width}x{height}."
            )
        if shoreline_jaggedness < 0:
            raise ValueError(
                f"shoreline_jaggedness must be >= 0, got {shoreline_jaggedness}."
            )
        if sleigh_pair_count < 0:
            raise ValueError(f"sleigh_pair_count must be >= 0, got {sleigh_pair_count}.")
        if not 0.0 < float(q_star_gamma) <= 1.0:
            raise ValueError(f"q_star_gamma must be in (0, 1], got {q_star_gamma!r}.")
        if fixed_map is not None and any(
            v is not None for v in (start_pos, start_pos_prob, goal_pos, goal_pos_prob)
        ):
            raise ValueError(
                "start_pos, start_pos_prob, goal_pos, and goal_pos_prob are ignored "
                "when fixed_map is set; encode start/goal tiles in the fixed map instead."
            )
        lo, hi = float(goal_reward_low), float(goal_reward_high)
        if lo > hi:
            raise ValueError(
                f"goal_reward_low must be <= goal_reward_high, got {lo} > {hi}."
            )
        self._goal_reward_low = lo
        self._goal_reward_high = hi
        self.reward_range = (min(0.0, lo), max(0.0, hi))
        self.q_star_gamma = float(q_star_gamma)
        self.emit_q_star = bool(emit_q_star)
        self.emit_map = bool(emit_map)
        self._has_fixed_map = fixed_map is not None
        self._render_mode = render_mode
        self._slippery_success_rate = float(slippery_success_rate)
        self._canvas_height = int(height)
        self._canvas_width = int(width)
        self._sampled_border: int | None = None
        self._sleigh_partner_by_state: dict[int, int] = {}
        start_positions = self._normalize_positions(start_pos, "start_pos")
        goal_positions = self._normalize_positions(goal_pos, "goal_pos")
        self._generation_config = {
            "min_hops": int(min_hops),
            "max_tries": int(max_tries),
            "width": int(width),
            "height": int(height),
            "min_border": int(min_border),
            "max_border": int(max_border),
            "shoreline_jaggedness": int(shoreline_jaggedness),
            "hole_prob": float(hole_prob),
            "start_pos": start_positions,
            "start_pos_prob": start_pos_prob,
            "goal_pos": goal_positions,
            "goal_pos_prob": goal_pos_prob,
            "tree_prob": float(tree_prob),
            "mirror_prob": float(mirror_prob),
            "sleigh_pair_count": int(sleigh_pair_count),
        }
        self._map_rng = random.Random(map_seed)
        self._gridmap: list[str] | None = None
        self._goal_rewards_by_state: dict[int, float] = {}
        self._map_info: dict[str, Any] | None = None
        self._q_table: np.ndarray | None = None
        self.fog_of_war = bool(fog_of_war)
        self._visited: np.ndarray | None = None
        self._fog_font_obj: Any = None
        self._tile_icons: dict[str, Any] | None = None
        self._tile_icons_cell_size: tuple[int, int] | None = None
        self._sleigh_badges: list[Any] | None = None
        self._sleigh_badges_key: tuple[tuple[int, int], int] | None = None
        self._goal_icons: dict[int, Any] | None = None
        self._goal_icons_key: Any = None
        self.permute_obs = bool(permute_obs)
        self.permute_actions = bool(permute_actions)
        self._obs_perm: list[int] | None = None
        self._action_perm: list[int] | None = None
        self.action_space = gym.spaces.Discrete(4)
        fixed_rewards: Mapping[int | str, float] | None = None
        if fixed_map is not None:
            gridmap, fixed_rewards = self._parse_fixed_map_spec(fixed_map)
            self._canvas_height = len(gridmap)
            self._canvas_width = len(gridmap[0])
            self._observation_space_n = self._canvas_height * self._canvas_width
            self._initialize_frozenlake_map(gridmap=gridmap, reward_overrides=fixed_rewards)
        else:
            self._observation_space_n = self._canvas_height * self._canvas_width
            self.observation_space = gym.spaces.Discrete(self._observation_space_n)
            # Parent FrozenLakeEnv.__init__ runs on first reset; seed attrs close() needs.
            self.window_surface = None

    def _normalize_positions(
        self,
        pos: int | list[int] | None,
        name: str,
    ) -> list[int] | None:
        """Normalize to a list of canonical flat canvas indices (row-major)."""
        if pos is None:
            return None
        if isinstance(pos, int):
            values = [pos]
        elif isinstance(pos, list) and all(isinstance(p, int) for p in pos):
            values = list(pos)
        else:
            raise ValueError(
                f"{name} must be None, an int, or a list of ints (canonical flat "
                f"canvas indices, row-major: index = row * width + col); got {pos!r}."
            )
        n = self._canvas_height * self._canvas_width
        out: list[int] = []
        for p in values:
            flat = int(p)
            if not 0 <= flat < n:
                raise ValueError(
                    f"{name} {flat} is outside the {self._canvas_height}x"
                    f"{self._canvas_width} canvas (valid state indices are 0..{n - 1}, "
                    f"numbered top-to-bottom, left-to-right)."
                )
            out.append(flat)
        return out

    def _set_observation_space(self) -> None:
        self.observation_space = gym.spaces.Discrete(self._observation_space_n)

    def _obs_to_row_col(self, obs: int) -> tuple[int, int]:
        return divmod(int(obs), int(self._canvas_width))

    def _state_index(self, row: int, col: int) -> int:
        return int(row) * int(self._canvas_width) + int(col)

    @staticmethod
    def _decode_grid_char(cell: Any) -> str:
        if isinstance(cell, bytes):
            return cell.decode("utf-8")
        if isinstance(cell, str):
            return cell
        return bytes(cell).decode("utf-8")

    def _sleigh_pairs_for_info(self) -> list[list[int]]:
        seen: set[int] = set()
        pairs: list[list[int]] = []
        for state, partner in sorted(self._sleigh_partner_by_state.items()):
            if state in seen:
                continue
            seen.add(state)
            seen.add(partner)
            pairs.append([state, partner])
        return pairs

    def _rebuild_transition_matrix(self) -> None:
        nrow, ncol = int(self.nrow), int(self.ncol)
        n_s = nrow * ncol
        n_a = 4
        success_rate = float(self._slippery_success_rate)
        fail_rate = (1.0 - success_rate) / 2.0
        desc = self.desc
        sleigh_partner = self._sleigh_partner_by_state

        def to_s(row: int, col: int) -> int:
            return row * ncol + col

        def resolve_landing(row: int, col: int) -> tuple[int, int, int, bytes, bool]:
            """Return (state, row, col, tile_bytes, terminated). Caller guarantees
            the landing tile is not a tree (tree bumps are handled as self-loops)."""
            tile = bytes(desc[row, col])
            state = to_s(row, col)
            if tile == TILE_SLEIGH.encode():
                state = sleigh_partner.get(state, state)
                row, col = divmod(state, ncol)
                tile = bytes(desc[row, col])
            terminated = tile in {TILE_GOAL.encode(), TILE_HOLE.encode()}
            return state, row, col, tile, terminated

        goal_rewards = self._goal_rewards_by_state

        def transition_reward(state: int, tile: bytes) -> float:
            """Exact reward the env pays for this transition (goal reward, else 0)."""
            if tile == TILE_GOAL.encode():
                return float(goal_rewards[state])
            return 0.0

        self.P = {s: {a: [] for a in range(n_a)} for s in range(n_s)}
        # Movement direction actually taken for each P entry, aligned index-for-index
        # with the transition tuples (None for terminal self-loops). Used by fog of
        # war to reveal the tile the agent really bumped into after a mirror slip.
        self._transition_move_dirs: dict[int, dict[int, list[int | None]]] = {
            s: {a: [] for a in range(n_a)} for s in range(n_s)
        }

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                tile = bytes(desc[row, col])
                if tile in {TILE_GOAL.encode(), TILE_HOLE.encode(), TILE_TREE.encode()}:
                    for a in range(n_a):
                        self.P[s][a] = [(1.0, s, 0.0, True)]
                        self._transition_move_dirs[s][a] = [None]
                    continue

                is_slippery = tile == TILE_MIRROR.encode()
                for a in range(n_a):
                    li = self.P[s][a]
                    dirs = self._transition_move_dirs[s][a]
                    actions = (
                        [(a - 1) % 4, a, (a + 1) % 4]
                        if is_slippery
                        else [a]
                    )
                    for move_a in actions:
                        prob = (
                            success_rate
                            if is_slippery and move_a == a
                            else (fail_rate if is_slippery else 1.0)
                        )
                        new_row, new_col = _neighbor(row, col, move_a, nrow, ncol)
                        landed_tile = bytes(desc[new_row, new_col])
                        if landed_tile == TILE_TREE.encode():
                            new_state = s
                            reward = 0.0
                            terminated = False
                        else:
                            new_state, _, _, landed_tile, terminated = resolve_landing(
                                new_row, new_col
                            )
                            reward = transition_reward(new_state, landed_tile)
                        li.append((prob, new_state, reward, terminated))
                        dirs.append(move_a)

    def _initialize_frozenlake_map(
        self,
        *,
        gridmap: list[str],
        reward_overrides: Mapping[int | str, float] | None,
    ) -> None:
        self._gridmap = gridmap
        self._canvas_height = len(gridmap)
        self._canvas_width = len(gridmap[0])
        self._sleigh_partner_by_state = build_sleigh_partners(
            gridmap, self._canvas_width
        )
        self._goal_rewards_by_state = self._compute_goal_rewards_for_map(
            gridmap=self._gridmap,
            rng=self._map_rng,
            overrides=reward_overrides,
        )
        FrozenLakeEnv.__init__(
            self,
            render_mode=self._render_mode,
            desc=self._gridmap,
            is_slippery=False,
        )
        # Parent sets reward_range=(0, 1); restore our honest range.
        self.reward_range = (
            min(0.0, self._goal_reward_low),
            max(0.0, self._goal_reward_high),
        )
        self._rebuild_transition_matrix()
        self._set_observation_space()
        n_states = self._canvas_height * self._canvas_width
        if self.permute_obs:
            obs_perm = list(range(n_states))
            self._map_rng.shuffle(obs_perm)
            self._obs_perm = obs_perm
        else:
            self._obs_perm = None
        if self.permute_actions:
            action_perm = list(range(4))
            self._map_rng.shuffle(action_perm)
            self._action_perm = action_perm
        else:
            self._action_perm = None
        self._map_info = self._make_map_info_dict()
        self._q_table = None
        self._clear_fog()

    def _regenerate_map(self) -> None:
        gridmap, border = generate_valid_map(
            self._map_rng,
            **self._generation_config,
        )
        self._sampled_border = border
        self._initialize_frozenlake_map(gridmap=gridmap, reward_overrides=None)

    def _ensure_map_initialized(self) -> None:
        if self._gridmap is None:
            self._regenerate_map()

    def _require_map_initialized(self) -> None:
        if self._gridmap is None:
            raise RuntimeError(
                "ProceduralFrozenLakeEnv map is not initialized; call reset() before using the map."
            )

    def _make_map_info_dict(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "board": list(self._gridmap or []),
            "rewards": {
                k: float(v) for k, v in sorted(self._goal_rewards_by_state.items())
            },
            "canvas": {"width": self._canvas_width, "height": self._canvas_height},
            "sleighs": {"pairs": self._sleigh_pairs_for_info()},
        }
        if self._sampled_border is not None:
            info["border"] = int(self._sampled_border)
        if self._obs_perm is not None:
            info["obs_permutation"] = list(self._obs_perm)
        if self._action_perm is not None:
            info["action_permutation"] = list(self._action_perm)
        return info

    @classmethod
    def goal_states_from_gridmap(
        cls, gridmap: list[str], canvas_width: int | None = None
    ) -> list[int]:
        return _goal_states_from_gridmap(gridmap, canvas_width)

    @classmethod
    def _parse_fixed_map_spec(
        cls,
        fixed_map: list[str] | tuple[str, ...] | Mapping[str, Any],
    ) -> tuple[list[str], Mapping[int | str, float] | None]:
        if isinstance(fixed_map, Mapping):
            if "board" not in fixed_map:
                raise ValueError(
                    "fixed_map dict must include 'board' (list of row strings), "
                    "and optionally 'rewards' (goal state index → reward)."
                )
            board_raw = fixed_map["board"]
            if not isinstance(board_raw, (list, tuple)):
                raise ValueError("fixed_map['board'] must be a list or tuple of row strings.")
            gridmap = normalize_and_validate_fixed_map(fixed_map=board_raw)
            rewards_raw = fixed_map.get("rewards")
            if rewards_raw is None:
                return gridmap, None
            if not isinstance(rewards_raw, Mapping):
                raise ValueError(
                    "fixed_map['rewards'] must be a mapping from goal state index to reward."
                )
            return gridmap, rewards_raw
        return normalize_and_validate_fixed_map(fixed_map=fixed_map), None

    def _compute_goal_rewards_for_map(
        self,
        gridmap: list[str],
        rng: random.Random,
        overrides: Mapping[int | str, float] | None,
    ) -> dict[int, float]:
        goal_states = self.goal_states_from_gridmap(gridmap, self._canvas_width)
        if not goal_states:
            raise ValueError("Map must contain at least one 'G' goal tile.")
        goal_set = set(goal_states)
        if overrides is not None:
            norm = {int(k): float(v) for k, v in overrides.items()}
            if set(norm.keys()) != goal_set:
                raise ValueError(
                    "fixed_map['rewards'] must contain exactly one entry per goal state. "
                    f"Expected states {sorted(goal_set)}, got {sorted(norm.keys())}."
                )
            return norm
        lo, hi = self._goal_reward_low, self._goal_reward_high
        return {s: float(rng.uniform(lo, hi)) for s in goal_states}

    def compute_q_table(
        self,
        max_iter: int = 10_000,
        tolerance: float = 1e-10,
    ) -> np.ndarray:
        self._require_map_initialized()
        n_s = int(self.nrow * self.ncol)
        absorbing_tiles = {TILE_GOAL, TILE_HOLE, TILE_TREE}
        terminal_states = {
            self._state_index(r, c)
            for r in range(self.nrow)
            for c in range(self.ncol)
            if self._decode_grid_char(self.desc[r, c]) in absorbing_tiles
        }
        # P carries the exact env rewards (per-goal reward + step penalty), so Q*
        # is the true optimal value of the live MDP, discounted by q_star_gamma.
        return solve_tabular_mdp(
            P=self.P,
            n_states=n_s,
            n_actions=4,
            gamma=float(self.q_star_gamma),
            terminal_states=terminal_states,
            max_iter=max_iter,
            tolerance=tolerance,
        )

    def _q_star_for_obs(self, obs: int) -> np.ndarray:
        if self._q_table is None:
            self._q_table = self.compute_q_table()
        return np.asarray(self._q_table[int(obs)], dtype=np.float64).copy()

    def _clear_fog(self) -> None:
        if not self.fog_of_war:
            self._visited = None
            return
        self._visited = np.zeros((self.nrow, self.ncol), dtype=bool)

    def _ensure_fog_initialized(self) -> None:
        if not self.fog_of_war:
            self._visited = None
            return
        shape = (self.nrow, self.ncol)
        if self._visited is None or self._visited.shape != shape:
            self._visited = np.zeros(shape, dtype=bool)

    def _mark_visited_at(self, row: int, col: int) -> None:
        if not self.fog_of_war or self._visited is None:
            return
        if 0 <= row < self.nrow and 0 <= col < self.ncol:
            self._visited[row, col] = True

    def _mark_visited(self, state: int) -> None:
        if not self.fog_of_war or self._visited is None:
            return
        row, col = self._obs_to_row_col(state)
        self._visited[row, col] = True
        # Standing on a sleigh means the agent warped through its partner (or
        # is about to warp from it), so both ends of the pair are known.
        partner = self._sleigh_partner_by_state.get(state)
        if partner is not None:
            p_row, p_col = self._obs_to_row_col(partner)
            self._visited[p_row, p_col] = True

    def _neighbor_from_action(self, row: int, col: int, action: int) -> tuple[int, int]:
        return _neighbor(row, col, action, int(self.nrow), int(self.ncol))

    def _cell_is_hidden(self, row: int, col: int) -> bool:
        return (
            self.fog_of_war
            and self._visited is not None
            and not self._visited[row, col]
        )

    def _fog_display_char(self, row: int, col: int) -> str:
        if self._cell_is_hidden(row, col):
            return "?"
        return self._decode_grid_char(self.desc[row, col])

    def _fog_desc_for_render(self) -> np.ndarray:
        fog_desc = self.desc.copy()
        if not self.fog_of_war or self._visited is None:
            return fog_desc
        for row in range(self.nrow):
            for col in range(self.ncol):
                if self._cell_is_hidden(row, col):
                    fog_desc[row, col] = TILE_FROZEN.encode()
        return fog_desc

    def _get_fog_font(self) -> Any:
        if self._fog_font_obj is None:
            import pygame  # type: ignore[import-untyped]

            self._fog_font_obj = pygame.font.SysFont(
                "dejavusans", max(16, self.cell_size[1] // 2)
            )
        return self._fog_font_obj

    def _draw_fog_question_marks(self, mode: str) -> np.ndarray | None:
        if not self.fog_of_war or self._visited is None or self.window_surface is None:
            return None
        import pygame  # type: ignore[import-untyped]

        font = self._get_fog_font()
        for row in range(self.nrow):
            for col in range(self.ncol):
                if not self._cell_is_hidden(row, col):
                    continue
                text = font.render("?", True, (40, 40, 40))
                text_rect = text.get_rect(
                    center=(
                        col * self.cell_size[0] + self.cell_size[0] // 2,
                        row * self.cell_size[1] + self.cell_size[1] // 2,
                    )
                )
                self.window_surface.blit(text, text_rect)
        if mode == "human":
            pygame.display.update()
            return None
        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)),
                axes=(1, 0, 2),
            )
        return None

    def _display_char_for_cell(self, row: int, col: int) -> str:
        if self.fog_of_war:
            return self._fog_display_char(row, col)
        return self._decode_grid_char(self.desc[row, col])

    def _ansi_char_for_cell(self, row: int, col: int) -> str:
        from gymnasium import utils

        ch = self._display_char_for_cell(row, col)
        colors = {
            TILE_TREE: "white",
            TILE_MIRROR: "cyan",
            TILE_SLEIGH: "blue",
            TILE_HOLE: "blue",
            TILE_GOAL: "green",
            TILE_START: "yellow",
        }
        color = colors.get(ch)
        if color is None:
            return ch
        return utils.colorize(ch, color, bold=(ch in {TILE_TREE, TILE_SLEIGH}))

    def _ensure_tile_icons(self) -> dict[str, Any]:
        cell_size = (int(self.cell_size[0]), int(self.cell_size[1]))
        if (
            self._tile_icons is not None
            and self._tile_icons_cell_size == cell_size
        ):
            return self._tile_icons
        import pygame  # type: ignore[import-untyped]

        # Icons are static assets shipped with the package; regenerate them with
        # scripts/generate_icons.sh during development.
        img_dir = path.join(path.dirname(__file__), "img")
        icons: dict[str, Any] = {}
        for tile in SPECIAL_TILES:
            icon_path = path.join(img_dir, _TILE_ICON_FILES[tile])
            raw = pygame.image.load(icon_path)
            icons[tile] = pygame.transform.scale(raw, cell_size)
        self._tile_icons = icons
        self._tile_icons_cell_size = cell_size
        return icons

    def _ensure_sleigh_badges(self) -> list[Any]:
        cell_size = (int(self.cell_size[0]), int(self.cell_size[1]))
        pair_count = len(self._sleigh_partner_by_state) // 2
        key = (cell_size, pair_count)
        if self._sleigh_badges is not None and self._sleigh_badges_key == key:
            return self._sleigh_badges
        self._sleigh_badges = build_sleigh_pair_badges(cell_size, pair_count)
        self._sleigh_badges_key = key
        return self._sleigh_badges

    def _sleigh_pair_index_by_state(self) -> dict[int, int]:
        return {
            state: idx
            for idx, pair in enumerate(self._sleigh_pairs_for_info())
            for state in pair
        }

    def _goal_reward_scale_t(self, reward: float) -> float:
        """Normalize a goal reward to [0, 1] for the yellow→green bow tint."""
        lo, hi = self._goal_reward_low, self._goal_reward_high
        if hi <= lo:
            rewards = list(self._goal_rewards_by_state.values())
            lo, hi = min(rewards), max(rewards)
        if hi <= lo:
            return 1.0
        return (reward - lo) / (hi - lo)

    def _ensure_goal_icons(self) -> dict[int, Any]:
        cell_size = (int(self.cell_size[0]), int(self.cell_size[1]))
        key = (cell_size, tuple(sorted(self._goal_rewards_by_state.items())))
        if self._goal_icons is not None and self._goal_icons_key == key:
            return self._goal_icons
        import pygame  # type: ignore[import-untyped]
        from gymnasium.envs.toy_text import frozen_lake

        goal_path = path.join(path.dirname(frozen_lake.__file__), "img", "goal.png")
        raw = pygame.image.load(goal_path)
        # No display may exist in rgb_array mode, so convert to a per-pixel
        # alpha surface manually instead of convert_alpha().
        goal_native = pygame.Surface(raw.get_size(), pygame.SRCALPHA)
        goal_native.blit(raw, (0, 0))
        icons: dict[int, Any] = {}
        for state, reward in self._goal_rewards_by_state.items():
            icons[state] = goal_reward_icon(
                goal_native,
                self._goal_reward_scale_t(reward),
                f"{reward:.2f}",
                cell_size,
            )
        self._goal_icons = icons
        self._goal_icons_key = key
        return icons

    def _draw_special_tiles(self) -> None:
        if self.window_surface is None:
            return

        icons = self._ensure_tile_icons()
        badges = self._ensure_sleigh_badges()
        goal_icons = self._ensure_goal_icons()
        pair_index = self._sleigh_pair_index_by_state()
        for row in range(self.nrow):
            for col in range(self.ncol):
                if self._cell_is_hidden(row, col):
                    continue
                ch = self._decode_grid_char(self.desc[row, col])
                pos = (col * self.cell_size[0], row * self.cell_size[1])
                if ch == TILE_GOAL:
                    goal_icon = goal_icons.get(self._state_index(row, col))
                    if goal_icon is not None:
                        self.window_surface.blit(goal_icon, pos)
                    continue
                if ch not in icons:
                    continue
                self.window_surface.blit(icons[ch], pos)
                if ch == TILE_SLEIGH:
                    idx = pair_index.get(self._state_index(row, col))
                    if idx is not None:
                        self.window_surface.blit(badges[idx], pos)

        # The parent render draws the elf before these icons are blitted, so an
        # agent standing on a special tile (e.g. just warped onto a sleigh)
        # or on a goal would be hidden underneath its overlay. Redraw it on top.
        bot_row, bot_col = self._obs_to_row_col(int(self.s))
        bot_ch = self._decode_grid_char(self.desc[bot_row, bot_col])
        if bot_ch in icons or bot_ch == TILE_GOAL:
            last_action = 1 if self.lastaction is None else int(self.lastaction)
            elf_images = self.elf_images
            assert elf_images is not None
            elf_img = elf_images[last_action]
            pos = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
            self.window_surface.blit(elf_img, pos)

    def _render_gui(self, mode: str) -> np.ndarray | None:
        original_desc = self.desc
        if self.fog_of_war:
            self.desc = self._fog_desc_for_render()
        try:
            super()._render_gui(mode)
            self._draw_special_tiles()
        finally:
            self.desc = original_desc
        if self.fog_of_war:
            return self._draw_fog_question_marks(mode)
        if mode == "rgb_array" and self.window_surface is not None:
            return np.transpose(
                np.array(
                    __import__("pygame").surfarray.pixels3d(self.window_surface)
                ),
                axes=(1, 0, 2),
            )
        return None

    def _render_text(self) -> str:
        from contextlib import closing
        from io import StringIO

        from gymnasium import utils

        outfile = StringIO()
        row, col = self._obs_to_row_col(int(self.s))
        desc = [
            [self._ansi_char_for_cell(y, x) for x in range(self.ncol)]
            for y in range(self.nrow)
        ]
        desc[row][col] = utils.colorize(
            self._display_char_for_cell(row, col), "red", highlight=True
        )
        if self.lastaction is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")
        with closing(outfile):
            return outfile.getvalue()

    def _external_obs(self, state: int) -> int:
        return self._obs_perm[state] if self._obs_perm is not None else state

    def _internal_action(self, action: int) -> int:
        return self._action_perm[action] if self._action_perm is not None else action

    def _emit_q_star_vector(self, state: int) -> np.ndarray:
        """Q-values for the current internal state, ordered by external action id."""
        q = self._q_star_for_obs(state)
        if self._action_perm is not None:
            q = q[np.asarray(self._action_perm)]
        return q

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        reset_options = dict(options or {})
        unknown = set(reset_options) - {"regenerate_map"}
        if unknown:
            raise ValueError(
                f"Unsupported reset option(s): {sorted(unknown)}. "
                "Only 'regenerate_map' is supported."
            )
        regenerate_map = bool(reset_options.pop("regenerate_map", False))
        if regenerate_map and self._has_fixed_map:
            raise ValueError("reset option regenerate_map=True requires fixed_map=None.")
        if regenerate_map:
            self._regenerate_map()
        else:
            self._ensure_map_initialized()
        obs, _ = super().reset(seed=seed, options=None)
        obs = int(obs)
        self._ensure_fog_initialized()
        self._mark_visited(obs)
        info: dict[str, Any] = {}
        if self.emit_map:
            info["map"] = self._map_info
        if self.emit_q_star:
            info["q_star"] = self._emit_q_star_vector(obs)
        return self._external_obs(obs), info

    def step(self, a: Any):
        self._require_map_initialized()
        action = self._internal_action(int(a))
        old_obs = int(self.s)
        old_row, old_col = self._obs_to_row_col(old_obs)

        # Sample the transition ourselves (mirroring FrozenLakeEnv.step) so we know
        # which movement direction was actually taken — needed for fog of war to
        # reveal the tile really bumped into when a mirror slip goes sideways.
        transitions = self.P[old_obs][action]
        idx = categorical_sample([t[0] for t in transitions], self.np_random)
        _, new_state, reward, terminated = transitions[idx]
        self.s = new_state
        self.lastaction = action
        if self.render_mode == "human":
            self.render()

        obs = int(new_state)
        self._mark_visited(obs)
        if obs == old_obs:
            move_dir = self._transition_move_dirs[old_obs][action][idx]
            if move_dir is not None:
                target_row, target_col = self._neighbor_from_action(
                    old_row, old_col, move_dir
                )
                self._mark_visited_at(target_row, target_col)
        # P carries the exact env rewards (per-goal reward + step penalty baked in).
        reward = float(reward)
        info: dict[str, Any] = {}
        if self.emit_map:
            info["map"] = self._map_info
        if self.emit_q_star:
            info["q_star"] = self._emit_q_star_vector(obs)
        # truncated is always False here; TimeLimit (max_episode_steps=100) may wrap.
        return self._external_obs(obs), reward, bool(terminated), False, info

    def close(self) -> None:
        super().close()


def ensure_registered() -> None:
    """Register ``Procedural-FrozenLake-v1`` with Gymnasium exactly once."""
    if PROCEDURAL_FROZENLAKE_ENV_ID in registry:
        return
    register(
        id=PROCEDURAL_FROZENLAKE_ENV_ID,
        entry_point="procedural_frozenlake.env:ProceduralFrozenLakeEnv",
        nondeterministic=True,
        max_episode_steps=100,
    )
