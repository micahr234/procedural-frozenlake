"""Procedural Frozen Lake environment with generated maps and optional q_star labels."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Mapping
from os import path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register, registry
from gymnasium.envs.toy_text.frozen_lake import DOWN, LEFT, RIGHT, UP, FrozenLakeEnv
from gymnasium.envs.toy_text.utils import categorical_sample

from procedural_frozenlake.tile_icons import (
    SPECIAL_TILES,
    build_sleigh_pair_badges,
    build_special_tile_icons,
    goal_reward_icon,
)
from procedural_frozenlake.utils import to_json_str
from procedural_frozenlake.value_iteration import solve_tabular_mdp

PROCEDURAL_FROZENLAKE_ENV_ID = "Procedural-FrozenLake-v1"

TILE_START = "S"
TILE_FROZEN = "F"
TILE_GLARE = "M"  # Mirror ice
TILE_SLEIGH = "W"  # Warp sleigh
TILE_HOLE = "H"
TILE_GOAL = "G"
TILE_TREE = "T"

TERMINAL_TILES = frozenset({TILE_GOAL, TILE_HOLE})
PLAYABLE_TILES = frozenset(
    {TILE_START, TILE_FROZEN, TILE_GLARE, TILE_SLEIGH, TILE_GOAL, TILE_HOLE}
)
ALLOWED_TILES = PLAYABLE_TILES | {TILE_TREE}

_TILE_ICON_FILES = {
    TILE_TREE: "tree.png",
    TILE_GLARE: "glare_ice.png",
    TILE_SLEIGH: "sleigh.png",
}

def _validate_prob(name: str, value: float) -> float:
    v = float(value)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}.")
    return v


class ProceduralFrozenLakeEnv(FrozenLakeEnv):
    """Procedural Frozen Lake variant with generated valid map and optional q_star info.

    Maps use a fixed canvas of ``max_height × max_width`` tiles (random generation) or the
    dimensions of ``fixed_map``. A lake envelope sampled within
    ``min_width..max_width`` × ``min_height..max_height`` is placed uniformly at random on
    the canvas; all playable tiles lie inside it, though the jagged shoreline may leave the
    lake smaller than the envelope. Tree (``T``) tiles surround the lake and are impassable;
    glare ice (``M``, mirror ice) tiles are locally slippery; sleighs (``W``) warp between
    paired tiles.

    ``goal_reward_low`` / ``goal_reward_high`` sample one reward **per goal tile** when the map
    is generated. The transition matrix ``P`` carries the exact env rewards (per-goal reward
    plus ``step_penalty``), so planning directly from ``P`` matches live behavior. When
    ``emit_q_star`` is True, :meth:`compute_q_table` solves ``P`` with the env
    ``step_penalty`` stripped out and discount ``q_star_gamma`` (default ``0.999``), so
    Q\\* is independent of reward shaping and always points toward shorter paths. Terminal
    states (goal/hole) and tree states have Q-values of zero.

    When ``emit_map`` is True, every :meth:`reset` and :meth:`step` puts a JSON string in
    ``info["map"]`` with ``board``, ``rewards``, ``canvas``, ``playable_count``, ``sleighs``,
    and — when enabled — ``obs_permutation`` / ``action_permutation``.

    ``permute_obs=True`` relabels observations with a random permutation of the canvas state
    indices; ``permute_actions=True`` relabels the four actions. Both permutations are sampled
    with the map (from ``map_seed``) and resampled when the map regenerates.

    Pass ``options={"regenerate_map": True}`` to :meth:`reset` to sample a fresh map.

    With ``fog_of_war=True`` (the default), unvisited tiles render as ``?``, including
    trees (``T``). Warping through a sleigh reveals both sleighs of the pair.
    Bumping into a blocked tile reveals it. Pass ``fog_of_war=False`` to render the
    full map.
    """

    def __init__(
        self,
        render_mode: str | None = None,
        # Map generation
        map_seed: int | None = None,
        fixed_map: list[str] | tuple[str, ...] | Mapping[str, Any] | None = None,
        min_width: int = 3,
        max_width: int = 8,
        min_height: int = 3,
        max_height: int = 8,
        hole_prob: float = 0.2,
        tree_prob: float = 0.0,
        glare_prob: float = 0.0,
        sleigh_pair_count: int = 0,
        start_pos: int | list[int] | None = None,
        start_pos_prob: float | None = None,
        goal_pos: int | list[int] | None = None,
        goal_pos_prob: float | None = None,
        min_hops: int = 3,
        max_tries: int = 10_000,
        # Dynamics and rewards
        slippery_success_rate: float = 1.0 / 3.0,
        step_penalty: float = 0.0,
        goal_reward_low: float = 1.0,
        goal_reward_high: float = 1.0,
        # Supervision signals in info
        emit_map: bool = False,
        emit_q_star: bool = False,
        q_star_gamma: float = 0.999,
        # Observation/action relabeling and rendering
        permute_obs: bool = False,
        permute_actions: bool = False,
        fog_of_war: bool = True,
    ):
        hole_prob = _validate_prob("hole_prob", hole_prob)
        tree_prob = _validate_prob("tree_prob", tree_prob)
        glare_prob = _validate_prob("glare_prob", glare_prob)
        slippery_success_rate = _validate_prob(
            "slippery_success_rate", slippery_success_rate
        )
        if start_pos_prob is not None:
            _validate_prob("start_pos_prob", start_pos_prob)
        if goal_pos_prob is not None:
            _validate_prob("goal_pos_prob", goal_pos_prob)
        if min_width < 1 or min_height < 1:
            raise ValueError(
                f"min_width and min_height must be >= 1, got {min_width} and {min_height}."
            )
        if min_width > max_width or min_height > max_height:
            raise ValueError(
                f"Width/height bounds must satisfy min <= max, got width "
                f"{min_width}..{max_width} and height {min_height}..{max_height}."
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
            lo, hi = hi, lo
        self._goal_reward_low = lo
        self._goal_reward_high = hi
        self._step_penalty = float(step_penalty)
        self.q_star_gamma = float(q_star_gamma)
        self.emit_q_star = bool(emit_q_star)
        self.emit_map = bool(emit_map)
        self._has_fixed_map = fixed_map is not None
        self._render_mode = render_mode
        self._slippery_success_rate = float(slippery_success_rate)
        self._canvas_height = int(max_height)
        self._canvas_width = int(max_width)
        self._sleigh_partner_by_state: dict[int, int] = {}
        self._generation_config = {
            "min_hops": int(min_hops),
            "max_tries": int(max_tries),
            "min_width": int(min_width),
            "max_width": int(max_width),
            "min_height": int(min_height),
            "max_height": int(max_height),
            "hole_prob": float(hole_prob),
            "start_pos": start_pos,
            "start_pos_prob": start_pos_prob,
            "goal_pos": goal_pos,
            "goal_pos_prob": goal_pos_prob,
            "tree_prob": float(tree_prob),
            "glare_prob": float(glare_prob),
            "sleigh_pair_count": int(sleigh_pair_count),
        }
        self._map_rng = random.Random(map_seed)
        self._gridmap: list[str] | None = None
        self._goal_rewards_by_state: dict[int, float] = {}
        self._map_info: str | None = None
        self._map_dirty = False
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
            self._validate_positions_in_canvas(start_pos, "start_pos")
            self._validate_positions_in_canvas(goal_pos, "goal_pos")

    def _validate_positions_in_canvas(
        self, pos: int | list[int] | None, name: str
    ) -> None:
        if pos is None:
            return
        values = [pos] if isinstance(pos, int) else list(pos)
        n = self._canvas_height * self._canvas_width
        for p in values:
            if not 0 <= int(p) < n:
                raise ValueError(
                    f"{name} {p} is outside the {self._canvas_height}x{self._canvas_width} "
                    f"canvas (valid state indices are 0..{n - 1})."
                )

    def _restore_max_observation_space(self) -> None:
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

    @classmethod
    def _build_sleigh_partners(cls, gridmap: list[str], canvas_width: int) -> dict[int, int]:
        sleigh_states: list[int] = []
        for r, row in enumerate(gridmap):
            for c, ch in enumerate(row):
                if ch == TILE_SLEIGH:
                    sleigh_states.append(r * canvas_width + c)
        if len(sleigh_states) % 2 != 0:
            raise ValueError(
                f"Map must contain an even number of {TILE_SLEIGH!r} sleigh tiles; got {len(sleigh_states)}."
            )
        partners: dict[int, int] = {}
        for i in range(0, len(sleigh_states), 2):
            a, b = sleigh_states[i], sleigh_states[i + 1]
            partners[a] = b
            partners[b] = a
        return partners

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

    def _playable_tile_count(self, gridmap: list[str]) -> int:
        return sum(ch != TILE_TREE for row in gridmap for ch in row)

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

        def inc(row: int, col: int, action: int) -> tuple[int, int]:
            if action == LEFT:
                col = max(col - 1, 0)
            elif action == DOWN:
                row = min(row + 1, nrow - 1)
            elif action == RIGHT:
                col = min(col + 1, ncol - 1)
            elif action == UP:
                row = max(row - 1, 0)
            return row, col

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
        step_penalty = float(self._step_penalty)

        def transition_reward(state: int, tile: bytes) -> float:
            """Exact reward the env pays for this transition (goal reward + step penalty)."""
            if tile == TILE_GOAL.encode():
                return float(goal_rewards[state]) + step_penalty
            return step_penalty

        self.P = {s: {a: [] for a in range(n_a)} for s in range(n_s)}
        # Movement direction actually taken for each P entry, aligned index-for-index
        # with the transition tuples (None for terminal self-loops). Used by fog of
        # war to reveal the tile the agent really bumped into after a glare slip.
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

                is_slippery = tile == TILE_GLARE.encode()
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
                        new_row, new_col = inc(row, col, move_a)
                        landed_tile = bytes(desc[new_row, new_col])
                        if landed_tile == TILE_TREE.encode():
                            new_state = s
                            reward = step_penalty
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
        self._sleigh_partner_by_state = self._build_sleigh_partners(
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
        self._rebuild_transition_matrix()
        self._restore_max_observation_space()
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
        self._map_info = to_json_str(self._make_map_info_dict())
        self._map_dirty = True
        self._q_table = None
        self._clear_fog()

    def _regenerate_map(self) -> None:
        gridmap = self._generate_valid_map(
            self._map_rng,
            **self._generation_config,
        )
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
                str(k): float(v) for k, v in sorted(self._goal_rewards_by_state.items())
            },
            "canvas": {"width": self._canvas_width, "height": self._canvas_height},
            "playable_count": self._playable_tile_count(self._gridmap or []),
            "sleighs": {"pairs": self._sleigh_pairs_for_info()},
        }
        if self._obs_perm is not None:
            info["obs_permutation"] = list(self._obs_perm)
        if self._action_perm is not None:
            info["action_permutation"] = list(self._action_perm)
        return info

    @staticmethod
    def goal_states_from_gridmap(gridmap: list[str], canvas_width: int | None = None) -> list[int]:
        cols = canvas_width if canvas_width is not None else len(gridmap[0])
        out: list[int] = []
        for r, row in enumerate(gridmap):
            for c, ch in enumerate(row):
                if ch == TILE_GOAL:
                    out.append(r * cols + c)
        return out

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
            gridmap = cls._normalize_and_validate_fixed_map(fixed_map=board_raw)
            rewards_raw = fixed_map.get("rewards")
            if rewards_raw is None:
                return gridmap, None
            if not isinstance(rewards_raw, Mapping):
                raise ValueError(
                    "fixed_map['rewards'] must be a mapping from goal state index to reward."
                )
            return gridmap, rewards_raw
        return cls._normalize_and_validate_fixed_map(fixed_map=fixed_map), None

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

    @classmethod
    def _is_blocked_for_path(cls, ch: str) -> bool:
        return ch in {TILE_HOLE, TILE_TREE}

    @classmethod
    def _find_path_to_goal_bfs(
        cls,
        gridmap: list[str],
        state: int,
    ) -> tuple[list[tuple[int, int]], list[int]] | None:
        rows = len(gridmap)
        cols = len(gridmap[0])
        board = [list(row) for row in gridmap]
        partners = cls._build_sleigh_partners(gridmap, cols)
        start_r, start_c = divmod(state, cols)
        start_pos = (start_r, start_c)
        goals = {
            (i, j)
            for i in range(rows)
            for j in range(cols)
            if board[i][j] == TILE_GOAL
        }
        if not goals:
            return None
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        queue: deque[tuple[tuple[int, int], list[tuple[int, int]], list[int]]] = deque()
        queue.append((start_pos, [], []))
        visited: set[tuple[int, int]] = set()
        while queue:
            (r, c), path, actions = queue.popleft()
            if (r, c) in goals:
                return [start_pos] + path, actions
            if (r, c) in visited:
                continue
            visited.add((r, c))
            next_positions: list[tuple[int, int]] = []
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not cls._is_blocked_for_path(
                    board[nr][nc]
                ):
                    next_positions.append((nr, nc))
            if board[r][c] == TILE_SLEIGH:
                tr, tc = divmod(partners[r * cols + c], cols)
                if not cls._is_blocked_for_path(board[tr][tc]):
                    next_positions.append((tr, tc))
            for nr, nc in next_positions:
                if board[nr][nc] == TILE_SLEIGH:
                    tr, tc = divmod(partners[nr * cols + nc], cols)
                    queue.append(((tr, tc), path + [(tr, tc)], actions + [0]))
                else:
                    queue.append(((nr, nc), path + [(nr, nc)], actions + [0]))
        return None

    @classmethod
    def _map_is_valid(cls, gridmap: list[str], min_hops: int) -> bool:
        board = [list(row) for row in gridmap]
        found_start = False
        cols = len(gridmap[0])
        for r, row in enumerate(board):
            for c, ch in enumerate(row):
                if ch != TILE_START:
                    continue
                found_start = True
                state = r * cols + c
                result = cls._find_path_to_goal_bfs(gridmap=gridmap, state=state)
                if result is None:
                    return False
                _, actions = result
                if len(actions) < min_hops:
                    return False
        return found_start

    @classmethod
    def _normalize_and_validate_fixed_map(
        cls,
        fixed_map: list[str] | tuple[str, ...],
    ) -> list[str]:
        if not fixed_map:
            raise ValueError("fixed_map cannot be empty.")
        gridmap = [str(row) for row in fixed_map]
        row_width = len(gridmap[0])
        if row_width == 0:
            raise ValueError("fixed_map rows cannot be empty.")
        if any(len(row) != row_width for row in gridmap):
            raise ValueError("fixed_map rows must all have the same width.")

        invalid_chars = sorted({ch for row in gridmap for ch in row if ch not in ALLOWED_TILES})
        if invalid_chars:
            raise ValueError(
                f"fixed_map contains unsupported characters {invalid_chars}. "
                f"Allowed: {sorted(ALLOWED_TILES)}."
            )

        num_starts = sum(row.count(TILE_START) for row in gridmap)
        num_goals = sum(row.count(TILE_GOAL) for row in gridmap)
        if num_starts < 1:
            raise ValueError("fixed_map must contain at least one 'S' start tile.")
        if num_goals < 1:
            raise ValueError("fixed_map must contain at least one 'G' goal tile.")

        cls._build_sleigh_partners(gridmap, row_width)
        return gridmap

    @classmethod
    def _largest_connected_component(
        cls, mask: list[list[bool]]
    ) -> set[tuple[int, int]]:
        rows, cols = len(mask), len(mask[0])
        best: set[tuple[int, int]] = set()
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for sr in range(rows):
            for sc in range(cols):
                if not mask[sr][sc]:
                    continue
                component: set[tuple[int, int]] = set()
                queue: deque[tuple[int, int]] = deque([(sr, sc)])
                component.add((sr, sc))
                while queue:
                    r, c = queue.popleft()
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and mask[nr][nc]
                            and (nr, nc) not in component
                        ):
                            component.add((nr, nc))
                            queue.append((nr, nc))
                if len(component) > len(best):
                    best = component
        return best

    @classmethod
    def _edge_erode(
        cls,
        rng: random.Random,
        count: int,
        domain: int,
        min_span: int,
    ) -> list[int]:
        """Random ±1 erosion depth from one edge, keeping span >= min_span."""
        erode = [0] * count
        max_erode = max(0, domain - min_span)
        erode[0] = rng.randint(0, max_erode) if max_erode > 0 else 0
        for i in range(1, count):
            erode[i] = max(
                0,
                min(max_erode, erode[i - 1] + rng.randint(-1, 1)),
            )
        return erode

    @classmethod
    def _mask_is_connected(cls, mask: list[list[bool]]) -> bool:
        rows, cols = len(mask), len(mask[0])
        start: tuple[int, int] | None = None
        total = 0
        for r in range(rows):
            for c in range(cols):
                if mask[r][c]:
                    total += 1
                    if start is None:
                        start = (r, c)
        if total == 0:
            return False
        assert start is not None
        component = cls._largest_connected_component(mask)
        return len(component) == total

    @classmethod
    def _generate_lake_mask(
        cls,
        rng: random.Random,
        *,
        canvas_width: int,
        canvas_height: int,
        min_width: int,
        max_width: int,
        min_height: int,
        max_height: int,
    ) -> list[list[bool]]:
        """Playable lake ice with a jagged land shoreline on all four sides.

        A ``w × h`` envelope with ``w`` sampled from ``min_width..max_width`` and
        ``h`` from ``min_height..max_height`` is placed uniformly at random on the
        canvas, so the land border thickness varies. All playable tiles lie inside
        the envelope; edge random walks erode the shoreline within it, so the lake
        itself may be smaller than the envelope.
        """
        for _ in range(64):
            w = rng.randint(min_width, min(max_width, canvas_width))
            h = rng.randint(min_height, min(max_height, canvas_height))
            top = rng.randint(0, canvas_height - h)
            left = rng.randint(0, canvas_width - w)

            min_span_h = max(2, w - 2)
            min_span_v = max(2, h - 2)

            left_erode = cls._edge_erode(rng, h, w, min_span_h)
            right_erode = cls._edge_erode(rng, h, w, min_span_h)
            top_erode = cls._edge_erode(rng, w, h, min_span_v)
            bottom_erode = cls._edge_erode(rng, w, h, min_span_v)

            mask = [[False] * canvas_width for _ in range(canvas_height)]
            for row_offset in range(h):
                row_idx = top + row_offset
                for col_offset in range(w):
                    col_idx = left + col_offset
                    if (
                        left_erode[row_offset] <= col_offset < w - right_erode[row_offset]
                        and top_erode[col_offset] <= row_offset < h - bottom_erode[col_offset]
                    ):
                        mask[row_idx][col_idx] = True

            if cls._mask_is_connected(mask):
                return mask

        raise RuntimeError("Could not generate a connected lake mask.")

    @classmethod
    def _mask_to_base_gridmap(
        cls, mask: list[list[bool]], canvas_width: int
    ) -> list[str]:
        rows: list[str] = []
        for row in mask:
            rows.append(
                "".join(TILE_FROZEN if playable else TILE_TREE for playable in row)
            )
        return rows

    @staticmethod
    def _playable_indices(gridmap: list[str], canvas_width: int) -> list[int]:
        indices: list[int] = []
        for r, row in enumerate(gridmap):
            for c, ch in enumerate(row):
                if ch != TILE_TREE:
                    indices.append(r * canvas_width + c)
        return indices

    @classmethod
    def _place_sleigh_pairs(
        cls,
        rng: random.Random,
        gridmap: list[str],
        canvas_width: int,
        pair_count: int,
    ) -> None:
        if pair_count <= 0:
            return
        candidates = [
            i
            for i in cls._playable_indices(gridmap, canvas_width)
            if gridmap[i // canvas_width][i % canvas_width]
            in {TILE_FROZEN, TILE_GLARE}
        ]
        rng.shuffle(candidates)
        needed = 2 * pair_count
        if len(candidates) < needed:
            raise RuntimeError("Not enough playable tiles to place sleigh pairs.")
        chosen = candidates[:needed]
        for idx in chosen:
            r, c = divmod(idx, canvas_width)
            row = list(gridmap[r])
            row[c] = TILE_SLEIGH
            gridmap[r] = "".join(row)

    @classmethod
    def _apply_tree_prob(
        cls,
        rng: random.Random,
        gridmap: list[str],
        tree_prob: float,
    ) -> None:
        if tree_prob <= 0.0:
            return
        for r, row in enumerate(gridmap):
            chars = list(row)
            for c, ch in enumerate(chars):
                if ch == TILE_FROZEN and rng.random() < tree_prob:
                    chars[c] = TILE_TREE
            gridmap[r] = "".join(chars)

    @classmethod
    def _apply_glare_prob(
        cls,
        rng: random.Random,
        gridmap: list[str],
        canvas_width: int,
        glare_prob: float,
    ) -> None:
        if glare_prob <= 0.0:
            return
        for r, row in enumerate(gridmap):
            chars = list(row)
            for c, ch in enumerate(chars):
                if ch == TILE_FROZEN and rng.random() < glare_prob:
                    chars[c] = TILE_GLARE
            gridmap[r] = "".join(chars)

    @staticmethod
    def _generate_map(
        rng: random.Random,
        min_width: int,
        max_width: int,
        min_height: int,
        max_height: int,
        hole_prob: float,
        start_pos: int | list[int] | None,
        start_pos_prob: float | None,
        goal_pos: int | list[int] | None,
        goal_pos_prob: float | None,
        tree_prob: float,
        glare_prob: float,
        sleigh_pair_count: int,
    ) -> list[str]:
        canvas_width = max_width
        canvas_height = max_height
        mask = ProceduralFrozenLakeEnv._generate_lake_mask(
            rng,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            min_width=min_width,
            max_width=max_width,
            min_height=min_height,
            max_height=max_height,
        )
        gridmap = ProceduralFrozenLakeEnv._mask_to_base_gridmap(mask, canvas_width)
        ProceduralFrozenLakeEnv._apply_tree_prob(rng, gridmap, tree_prob)

        available_index = ProceduralFrozenLakeEnv._playable_indices(gridmap, canvas_width)
        if len(available_index) < 2:
            raise RuntimeError(
                "Carved lake has fewer than 2 playable tiles (need a start and a goal)."
            )

        def place_tiles(positions: list[int], tile: str, name: str) -> None:
            for p in positions:
                if p not in available_index:
                    raise RuntimeError(
                        f"{name} {p} is not on playable lake ice for this map."
                    )
                r, c = divmod(p, canvas_width)
                row = list(gridmap[r])
                row[c] = tile
                gridmap[r] = "".join(row)
                available_index.remove(p)

        if isinstance(start_pos, int):
            start_positions: list[int] | None = [start_pos]
        elif start_pos is None:
            start_positions = None
        else:
            start_positions = list(start_pos)
        if start_positions is None and start_pos_prob is None:
            start_positions = [rng.choice(available_index)]
        elif start_positions is None and start_pos_prob is not None:
            start_positions = [
                i for i in available_index if rng.random() < start_pos_prob
            ]
        place_tiles(start_positions or [], TILE_START, "start_pos")

        if isinstance(goal_pos, int):
            goal_positions: list[int] | None = [goal_pos]
        elif goal_pos is None:
            goal_positions = None
        else:
            goal_positions = list(goal_pos)
        if goal_positions is None and goal_pos_prob is None:
            goal_positions = [rng.choice(available_index)]
        elif goal_positions is None and goal_pos_prob is not None:
            goal_positions = [
                i for i in available_index if rng.random() < goal_pos_prob
            ]
        place_tiles(goal_positions or [], TILE_GOAL, "goal_pos")

        for i in list(available_index):
            if rng.random() < hole_prob:
                r, c = divmod(i, canvas_width)
                row = list(gridmap[r])
                row[c] = TILE_HOLE
                gridmap[r] = "".join(row)
                available_index.remove(i)

        ProceduralFrozenLakeEnv._place_sleigh_pairs(
            rng, gridmap, canvas_width, sleigh_pair_count
        )
        ProceduralFrozenLakeEnv._apply_glare_prob(
            rng, gridmap, canvas_width, glare_prob
        )
        return gridmap

    @classmethod
    def _generate_valid_map(
        cls,
        rng: random.Random,
        min_hops: int,
        max_tries: int,
        **kwargs: Any,
    ) -> list[str]:
        last_error: str | None = None
        for _ in range(max_tries):
            try:
                gridmap = cls._generate_map(rng=rng, **kwargs)
            except RuntimeError as exc:
                last_error = str(exc)
                continue
            if cls._map_is_valid(gridmap, min_hops=min_hops):
                return gridmap
        message = (
            "Could not generate a valid Procedural Frozen Lake map. "
            "Try lower hole_prob, lower min_hops, or larger dimensions."
        )
        if last_error is not None:
            message += f" Last generation failure: {last_error}"
        raise RuntimeError(message)

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
        # P rewards include the env step penalty; strip it so Q* reflects only the
        # goal rewards, discounted by q_star_gamma to prefer shorter paths.
        return solve_tabular_mdp(
            P=self.P,
            n_states=n_s,
            n_actions=4,
            gamma=float(self.q_star_gamma),
            step_penalty=-float(self._step_penalty),
            terminal_states=terminal_states,
            max_iter=max_iter,
            tolerance=tolerance,
        )

    def _optimal_action_for_obs(self, obs: int) -> int:
        if self._q_table is None:
            return 0
        state = int(obs)
        n_s, _ = self._q_table.shape
        if state < 0 or state >= n_s:
            return 0
        return int(np.argmax(self._q_table[state]))

    def _q_star_for_obs(self, obs: int) -> np.ndarray:
        action_dim = int(getattr(self.action_space, "n", 0))
        if self._q_table is None:
            fallback = np.full((action_dim,), np.nan, dtype=np.float64)
            fallback[self._optimal_action_for_obs(obs)] = 0.0
            return fallback
        state = int(obs)
        n_s, _ = self._q_table.shape
        if state < 0 or state >= n_s:
            fallback = np.full((action_dim,), np.nan, dtype=np.float64)
            fallback[self._optimal_action_for_obs(obs)] = 0.0
            return fallback
        return np.asarray(self._q_table[state], dtype=np.float64).copy()

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
        if action == LEFT:
            col = max(col - 1, 0)
        elif action == DOWN:
            row = min(row + 1, self.nrow - 1)
        elif action == RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif action == UP:
            row = max(row - 1, 0)
        return row, col

    def _decode_cell(self, cell: Any) -> str:
        return self._decode_grid_char(cell)

    def _cell_is_hidden(self, row: int, col: int) -> bool:
        return (
            self.fog_of_war
            and self._visited is not None
            and not self._visited[row, col]
        )

    def _fog_display_char(self, row: int, col: int) -> str:
        if self._cell_is_hidden(row, col):
            return "?"
        return self._decode_cell(self.desc[row, col])

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
        return self._decode_cell(self.desc[row, col])

    def _ansi_char_for_cell(self, row: int, col: int) -> str:
        from gymnasium import utils

        ch = self._display_char_for_cell(row, col)
        colors = {
            TILE_TREE: "white",
            TILE_GLARE: "cyan",
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

        img_dir = path.join(path.dirname(__file__), "img")
        icons: dict[str, Any] = {}
        for tile in SPECIAL_TILES:
            icon_path = path.join(img_dir, _TILE_ICON_FILES[tile])
            if path.isfile(icon_path):
                raw = pygame.image.load(icon_path)
                icons[tile] = pygame.transform.scale(raw, cell_size)
            else:
                if not pygame.get_init():
                    pygame.init()
                icons[tile] = build_special_tile_icons(cell_size)[tile]
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
                ch = self._decode_cell(self.desc[row, col])
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
        bot_ch = self._decode_cell(self.desc[bot_row, bot_col])
        if bot_ch in icons or bot_ch == TILE_GOAL:
            last_action = self.lastaction if self.lastaction is not None else 1
            elf_img = self.elf_images[last_action]
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
        regenerate_map = bool(reset_options.pop("regenerate_map", False))
        if regenerate_map and self._has_fixed_map:
            raise ValueError("reset option regenerate_map=True requires fixed_map=None.")
        if regenerate_map:
            self._regenerate_map()
        else:
            self._ensure_map_initialized()
        obs, info = super().reset(seed=seed, options=reset_options or None)
        obs = int(obs)
        self._ensure_fog_initialized()
        self._mark_visited(obs)
        info = dict[str, Any](info)
        if self._map_dirty:
            if self.emit_q_star:
                self._q_table = self.compute_q_table()
            self._map_dirty = False
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
        # reveal the tile really bumped into when a glare slip goes sideways.
        transitions = self.P[old_obs][action]
        idx = categorical_sample([t[0] for t in transitions], self.np_random)
        prob, new_state, reward, terminated = transitions[idx]
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
        info: dict[str, Any] = {"prob": prob}
        if self.emit_map:
            info["map"] = self._map_info
        if self.emit_q_star:
            info["q_star"] = self._emit_q_star_vector(obs)
        return self._external_obs(obs), reward, bool(terminated), False, info

    def close(self) -> None:
        if self._gridmap is None:
            return
        super().close()


def ensure_registered() -> None:
    """Register ``Procedural-FrozenLake-v1`` with Gymnasium exactly once."""
    if PROCEDURAL_FROZENLAKE_ENV_ID in registry:
        return
    register(
        id=PROCEDURAL_FROZENLAKE_ENV_ID,
        entry_point="procedural_frozenlake.env:ProceduralFrozenLakeEnv",
    )
