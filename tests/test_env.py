"""Tests for Procedural-FrozenLake-v1."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import procedural_frozenlake  # noqa: F401 — registers the env id
from procedural_frozenlake import PROCEDURAL_FROZENLAKE_ENV_ID, ProceduralFrozenLakeEnv


def test_gym_make_step_contract() -> None:
    env = gym.make(
        PROCEDURAL_FROZENLAKE_ENV_ID,
        emit_q_star=True,
        map_seed=0,
    )
    try:
        obs, info = env.reset(seed=1)
        assert isinstance(obs, (int, np.integer))
        assert "q_star" in info
        assert info["q_star"].shape == (4,)
        assert np.all(np.isfinite(info["q_star"]))

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, (int, np.integer))
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "q_star" in info
    finally:
        env.close()


def test_random_map_is_lazy_until_reset() -> None:
    env = ProceduralFrozenLakeEnv(map_seed=1, emit_map=True)
    try:
        assert env._gridmap is None
        _, info = env.reset(seed=2)
        assert env._gridmap is not None
        assert "map" in info
    finally:
        env.close()


def test_can_regenerate_map_on_reset() -> None:
    env = gym.make(
        PROCEDURAL_FROZENLAKE_ENV_ID,
        emit_map=True,
        hole_prob=0.0,
        map_seed=7,
    )
    try:
        _, first_info = env.reset(seed=1)
        env.step(env.action_space.sample())
        _, second_info = env.reset(seed=2, options={"regenerate_map": True})
        assert "map" in first_info
        assert "map" in second_info
        assert first_info["map"] != second_info["map"]
    finally:
        env.close()


def test_map_seed_is_independent_from_reset_seed() -> None:
    env_a = ProceduralFrozenLakeEnv(map_seed=11, emit_map=True)
    env_b = ProceduralFrozenLakeEnv(map_seed=11, emit_map=True)
    try:
        _, info_a = env_a.reset(seed=21)
        _, info_b = env_b.reset(seed=22)
        assert info_a["map"] == info_b["map"]
    finally:
        env_a.close()
        env_b.close()


def test_observation_space_uses_max_map_size() -> None:
    env = gym.make(
        PROCEDURAL_FROZENLAKE_ENV_ID,
        emit_map=True,
        min_width=3,
        max_width=5,
        min_height=3,
        max_height=6,
        map_seed=7,
    )
    try:
        space = env.observation_space
        assert isinstance(space, gym.spaces.Discrete)
        assert space.n == 30

        env.reset(seed=1, options={"regenerate_map": True})
        env.reset(seed=2, options={"regenerate_map": True})

        space = env.observation_space
        assert isinstance(space, gym.spaces.Discrete)
        assert space.n == 30
    finally:
        env.close()


def test_fixed_map() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG",
        ],
        emit_map=True,
    )
    try:
        _, info = env.reset(seed=0)
        assert "map" in info
        with pytest.raises(ValueError, match="regenerate_map=True requires fixed_map=None"):
            env.reset(options={"regenerate_map": True})
    finally:
        env.close()


def test_q_star_matches_compute_q_table() -> None:
    env = ProceduralFrozenLakeEnv(map_seed=3, emit_q_star=True)
    try:
        obs, info = env.reset(seed=0)
        expected = env.compute_q_table()
        assert np.allclose(info["q_star"], expected[obs])
    finally:
        env.close()


def test_q_star_prefers_progress_with_zero_step_penalty() -> None:
    """Default q_star epsilon should beat wall-bumping when step_penalty is zero."""
    env = ProceduralFrozenLakeEnv(
        fixed_map=["SFG"],
        step_penalty=0.0,
        emit_q_star=True,
    )
    try:
        obs, info = env.reset(seed=0)
        assert obs == 0
        q = env.compute_q_table()
        assert q[0, 2] > q[0, 0]
        assert info["q_star"][2] > info["q_star"][0]
    finally:
        env.close()


def test_q_star_step_penalty_decoupled_from_env_step_penalty() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map=["SFG"],
        step_penalty=-0.5,
        q_star_step_penalty=-0.01,
        emit_q_star=True,
    )
    try:
        env.reset(seed=0)
        q = env.compute_q_table()
        assert env._q_star_step_penalty == -0.01
        assert env._step_penalty == -0.5
        q_with_env_penalty = ProceduralFrozenLakeEnv(
            fixed_map=["SFG"],
            step_penalty=-0.5,
            emit_q_star=True,
        )
        q_with_env_penalty.reset(seed=0)
        assert not np.allclose(q, q_with_env_penalty.compute_q_table())
    finally:
        env.close()
        q_with_env_penalty.close()


def test_same_map_is_reused_without_regenerate_option() -> None:
    env = gym.make(
        PROCEDURAL_FROZENLAKE_ENV_ID,
        emit_map=True,
        map_seed=7,
    )
    try:
        env.reset(seed=1)
        first_map = list(env.unwrapped._gridmap or [])
        env.step(env.action_space.sample())
        env.reset(seed=2)
        second_map = list(env.unwrapped._gridmap or [])
        assert first_map
        assert first_map == second_map
    finally:
        env.close()


def test_fog_of_war_hides_all_unvisited_tiles() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "SFG",
            "FFF",
            "FHF",
        ],
        fog_of_war=True,
        render_mode="ansi",
    )
    try:
        env.reset(seed=0)
        assert env._fog_display_char(0, 0) == "S"
        assert env._fog_display_char(0, 1) == "?"
        assert env._fog_display_char(0, 2) == "?"
        assert env._fog_display_char(1, 1) == "?"
        assert env._fog_display_char(2, 1) == "?"
        render_out = env.render()
        assert isinstance(render_out, str)
        assert "?" in render_out
        assert "H" not in render_out
        assert "G" not in render_out
        visited = env._visited
        assert visited is not None
        assert visited[0, 0]
        assert not visited[0, 1]
    finally:
        env.close()


def test_fog_of_war_reveals_cells_when_visited() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "SFG",
            "FFF",
            "FHF",
        ],
        fog_of_war=True,
        render_mode="ansi",
    )
    try:
        env.reset(seed=0)
        env.step(2)  # onto (0, 1)
        assert env._fog_display_char(0, 1) == "F"
        env.step(2)  # onto goal
        visited = env._visited
        assert visited is not None
        assert visited[0, 2]
        assert env._fog_display_char(0, 2) == "G"
        assert env._fog_display_char(2, 1) == "?"
    finally:
        env.close()


def test_fog_of_war_persists_across_episode_reset() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "SFG",
            "FFF",
            "FHF",
        ],
        fog_of_war=True,
        render_mode="ansi",
    )
    try:
        env.reset(seed=0)
        env.step(2)
        env.step(2)
        visited = env._visited
        assert visited is not None
        assert visited[0, 2]
        assert env._fog_display_char(0, 2) == "G"
        env.reset(seed=1)
        visited = env._visited
        assert visited is not None
        assert visited[0, 2]
        assert env._fog_display_char(0, 2) == "G"
        assert env._fog_display_char(2, 1) == "?"
    finally:
        env.close()


def test_fog_of_war_clears_when_map_regenerates() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map=None,
        fog_of_war=True,
        map_seed=7,
        min_width=3,
        max_width=3,
        min_height=3,
        max_height=3,
        hole_prob=0.0,
        min_hops=1,
    )
    try:
        env.reset(seed=0)
        assert env._visited is not None
        env._visited.fill(True)
        env.reset(seed=1, options={"regenerate_map": True})
        visited = env._visited
        assert visited is not None
        assert visited.sum() == 1
    finally:
        env.close()


def test_canvas_is_always_max_size() -> None:
    env = gym.make(
        PROCEDURAL_FROZENLAKE_ENV_ID,
        emit_map=True,
        min_width=3,
        max_width=5,
        min_height=3,
        max_height=6,
        map_seed=7,
    )
    try:
        env.reset(seed=1)
        unwrapped = env.unwrapped
        assert len(unwrapped._gridmap or []) == 6
        assert all(len(row) == 5 for row in unwrapped._gridmap or [])
        env.reset(seed=2, options={"regenerate_map": True})
        assert len(unwrapped._gridmap or []) == 6
        assert all(len(row) == 5 for row in unwrapped._gridmap or [])
    finally:
        env.close()


def test_obs_indexing_uses_max_width_stride() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "LSFFF",
            "LFFGF",
            "LFFFF",
            "LFFFF",
            "LFFFF",
        ],
        emit_map=True,
    )
    try:
        obs, _ = env.reset(seed=0)
        assert obs == 1  # S at (0, 1)
        row, col = env._obs_to_row_col(obs)
        assert row == 0 and col == 1
        assert obs == row * 5 + col
    finally:
        env.close()


def test_land_tiles_are_impassable() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "SLFFF",
            "LFFFF",
            "FFFFG",
        ],
        emit_map=True,
    )
    try:
        obs, _ = env.reset(seed=0)
        assert obs == 0
        new_obs, _, _, _, _ = env.step(2)
        assert new_obs == obs
    finally:
        env.close()


def test_generated_maps_include_land() -> None:
    env = ProceduralFrozenLakeEnv(
        min_width=3,
        max_width=5,
        min_height=3,
        max_height=5,
        hole_prob=0.0,
        min_hops=1,
        map_seed=42,
        emit_map=True,
    )
    try:
        env.reset(seed=0)
        board = "".join(env._gridmap or [])
        assert "L" in board
        assert "S" in board
        assert "G" in board
    finally:
        env.close()


def test_fog_hides_trees_until_revealed() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "SLFG",
            "LFFF",
            "FFFF",
        ],
        fog_of_war=True,
        render_mode="ansi",
    )
    try:
        env.reset(seed=0)
        assert env._fog_display_char(1, 0) == "?"
        assert env._fog_display_char(0, 2) == "?"
    finally:
        env.close()


def test_fog_reveals_tree_when_bumping() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "SLFG",
            "FFFF",
            "FFFF",
        ],
        fog_of_war=True,
        render_mode="ansi",
    )
    try:
        env.reset(seed=0)
        assert env._fog_display_char(0, 1) == "?"
        env.step(2)  # bump into tree at (0, 1) — stay at start
        assert env._fog_display_char(0, 1) == "L"
    finally:
        env.close()


def test_sleigh_warp_pair() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "SOFFG",
            "FFFFF",
            "FFFFO",
        ],
        emit_map=True,
    )
    try:
        env.reset(seed=0)
        obs, _, _, _, _ = env.step(2)
        assert obs == 14
        obs2, _, _, _, _ = env.step(3)  # up off partner sleigh onto frozen ice
        assert obs2 == 9
    finally:
        env.close()


def test_sleigh_pair_count_generation() -> None:
    env = ProceduralFrozenLakeEnv(
        sleigh_pair_count=1,
        hole_prob=0.0,
        min_hops=1,
        min_width=4,
        max_width=4,
        min_height=4,
        max_height=4,
        map_seed=99,
    )
    try:
        env.reset(seed=0)
        assert sum(row.count("O") for row in env._gridmap or []) == 2
    finally:
        env.close()


def test_fixed_map_odd_sleigh_count_raises() -> None:
    with pytest.raises(ValueError, match="even number"):
        ProceduralFrozenLakeEnv(
            fixed_map=[
                "SOFFG",
                "FFFFF",
            ],
        )


def test_glare_ice_is_slippery() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "SRFFG",
            "FFFFF",
            "FFFFF",
        ],
        slippery_success_rate=1.0 / 3.0,
    )
    try:
        env.reset(seed=0)
        outcomes: set[int] = set()
        for _ in range(50):
            env.unwrapped.s = 1
            new_obs, _, _, _, _ = env.step(2)
            outcomes.add(new_obs)
        assert len(outcomes) > 1
    finally:
        env.close()


def test_frozen_tiles_are_deterministic() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "SFFFG",
            "FFFFF",
            "FFFFF",
        ],
    )
    try:
        env.reset(seed=0)
        env.unwrapped.s = 1
        first, _, _, _, _ = env.step(2)
        for _ in range(10):
            env.unwrapped.s = 1
            obs, _, _, _, _ = env.step(2)
            assert obs == first
    finally:
        env.close()


def test_glare_prob_generation() -> None:
    env = ProceduralFrozenLakeEnv(
        glare_prob=1.0,
        hole_prob=0.0,
        min_hops=1,
        min_width=4,
        max_width=4,
        min_height=4,
        max_height=4,
        map_seed=5,
    )
    try:
        env.reset(seed=0)
        board = "".join(env._gridmap or [])
        assert "R" in board
    finally:
        env.close()


def test_land_prob_generates_valid_map() -> None:
    env = ProceduralFrozenLakeEnv(
        land_prob=0.3,
        hole_prob=0.0,
        min_hops=1,
        min_width=6,
        max_width=8,
        min_height=6,
        max_height=8,
        map_seed=12,
    )
    try:
        env.reset(seed=0)
        assert env._gridmap is not None
        assert any("L" in row for row in env._gridmap)
    finally:
        env.close()


def test_fixed_map_with_glare_and_sleighs() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "SOFFG",
            "FRFFF",
            "FFFFO",
        ],
        emit_q_star=True,
    )
    try:
        obs, info = env.reset(seed=0)
        assert np.all(np.isfinite(info["q_star"]))
        q = env.compute_q_table()
        assert np.all(np.isfinite(q[obs]))
    finally:
        env.close()


def test_lake_has_jagged_land_edges() -> None:
    """Generated lakes vary playable span per row and column (not uniform land strips)."""
    env = ProceduralFrozenLakeEnv(
        min_width=5,
        max_width=7,
        min_height=5,
        max_height=7,
        hole_prob=0.0,
        min_hops=1,
        map_seed=123,
    )
    try:
        env.reset(seed=0)
        board = env._gridmap or []
        row_spans: list[tuple[int, int]] = []
        col_spans: list[tuple[int, int]] = []
        for row in board:
            playable = [i for i, ch in enumerate(row) if ch != "L"]
            if playable:
                row_spans.append((playable[0], playable[-1]))
        width = len(board[0]) if board else 0
        for col_idx in range(width):
            playable_rows = [
                r for r, row in enumerate(board) if row[col_idx] != "L"
            ]
            if playable_rows:
                col_spans.append((playable_rows[0], playable_rows[-1]))
        assert len(row_spans) >= 2
        assert len({span for span in row_spans}) > 1
        assert len(col_spans) >= 2
        assert len({span for span in col_spans}) > 1
    finally:
        env.close()


def test_special_tile_icons_render() -> None:
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "SLRFG",
            "FFFFF",
            "FFOOF",
        ],
        render_mode="rgb_array",
    )
    try:
        env.reset(seed=0)
        icons = env._ensure_tile_icons()
        assert set(icons) == {"L", "R", "O"}
        frame = env.render()
        assert frame is not None
        assert frame.shape[2] == 3
    finally:
        env.close()


def test_warp_reveals_both_sleighs_under_fog() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "SOFFG",
            "FFFFF",
            "FFFFO",
        ],
        fog_of_war=True,
        render_mode="ansi",
    )
    try:
        env.reset(seed=0)
        assert env._cell_is_hidden(0, 1)
        assert env._cell_is_hidden(2, 4)
        env.step(2)  # onto sleigh at (0, 1), warp to (2, 4)
        assert not env._cell_is_hidden(0, 1)
        assert not env._cell_is_hidden(2, 4)
    finally:
        env.close()


def test_sleigh_pair_badges_mark_pairs() -> None:
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "SOFOG",
            "FFFFF",
            "FOFOF",
        ],
        render_mode="rgb_array",
    )
    try:
        env.reset(seed=0)
        frame = env.render()
        cell_w, cell_h = env.cell_size

        def cell(row: int, col: int):
            return frame[
                row * cell_h : (row + 1) * cell_h,
                col * cell_w : (col + 1) * cell_w,
            ]

        # Row-major pairing: (0,1)+(0,3) are pair 1, (2,1)+(2,3) are pair 2.
        assert (cell(0, 1) == cell(0, 3)).all()
        assert (cell(2, 1) == cell(2, 3)).all()
        assert (cell(0, 1) != cell(2, 1)).any()
    finally:
        env.close()


def test_goal_reward_tint_and_badge() -> None:
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    env = ProceduralFrozenLakeEnv(
        fixed_map={
            "board": [
                "SFGFG",
                "FFFFF",
            ],
            "rewards": {2: 0.1, 4: 1.5},
        },
        render_mode="rgb_array",
    )
    try:
        env.reset(seed=0)
        frame = env.render()
        cell_w, cell_h = env.cell_size

        low = frame[0:cell_h, 2 * cell_w : 3 * cell_w]
        high = frame[0:cell_h, 4 * cell_w : 5 * cell_w]
        # Different rewards produce different bow tints and badge text.
        assert (low != high).any()
        # Low-reward bow is yellow (red ≈ green); high-reward bow is green.
        assert (low[:, :, 0].astype(int) - low[:, :, 1].astype(int)).max() < 80
        green_dominant = high[:, :, 1].astype(int) - high[:, :, 0].astype(int)
        assert green_dominant.max() > 80
    finally:
        env.close()


def test_elf_visible_on_sleigh_after_warp() -> None:
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "SOFFG",
            "FFFFF",
            "FFFFO",
        ],
        render_mode="rgb_array",
    )
    try:
        env.reset(seed=0)
        cell_w, cell_h = env.cell_size
        partner_row, partner_col = 2, 4
        before = env.render()[
            partner_row * cell_h : (partner_row + 1) * cell_h,
            partner_col * cell_w : (partner_col + 1) * cell_w,
        ].copy()
        obs, _, _, _, _ = env.step(2)  # onto sleigh at state 1, warp to state 14
        assert obs == 14
        after = env.render()[
            partner_row * cell_h : (partner_row + 1) * cell_h,
            partner_col * cell_w : (partner_col + 1) * cell_w,
        ]
        # The elf must be drawn on top of the sleigh icon at the warp target,
        # so the cell's pixels change once the agent arrives.
        assert (before != after).any()
    finally:
        env.close()
