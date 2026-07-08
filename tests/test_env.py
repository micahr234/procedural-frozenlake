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


def test_fog_of_war_defaults_on() -> None:
    env = ProceduralFrozenLakeEnv(map_seed=0)
    try:
        assert env.fog_of_war
    finally:
        env.close()


def test_map_info_emitted_every_reset_and_step() -> None:
    env = ProceduralFrozenLakeEnv(map_seed=1, emit_map=True)
    try:
        _, first = env.reset(seed=1)
        _, _, _, _, step_info = env.step(0)
        _, second = env.reset(seed=2)
        assert first["map"] == step_info["map"] == second["map"]
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
    """Default q_star_gamma discounting should beat wall-bumping when step_penalty is zero."""
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


def test_q_star_gamma_controls_discounting() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map=["SFG"],
        q_star_gamma=0.5,
        emit_q_star=True,
    )
    try:
        env.reset(seed=0)
        q = env.compute_q_table()
        # Two steps to the goal from S: 0.5^1 * 1.0; one step from F: 1.0.
        assert q[0, 2] == pytest.approx(0.5)
        assert q[1, 2] == pytest.approx(1.0)
    finally:
        env.close()


def test_q_star_gamma_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="q_star_gamma"):
        ProceduralFrozenLakeEnv(q_star_gamma=0.0)
    with pytest.raises(ValueError, match="q_star_gamma"):
        ProceduralFrozenLakeEnv(q_star_gamma=1.5)


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
            "TSFFF",
            "TFFGF",
            "TFFFF",
            "TFFFF",
            "TFFFF",
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
            "STFFF",
            "TFFFF",
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
        assert "T" in board
        assert "S" in board
        assert "G" in board
    finally:
        env.close()


def test_fog_hides_trees_until_revealed() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "STFG",
            "TFFF",
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
            "STFG",
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
        assert env._fog_display_char(0, 1) == "T"
    finally:
        env.close()


def test_sleigh_warp_pair() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "SWFFG",
            "FFFFF",
            "FFFFW",
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
        assert sum(row.count("W") for row in env._gridmap or []) == 2
    finally:
        env.close()


def test_fixed_map_odd_sleigh_count_raises() -> None:
    with pytest.raises(ValueError, match="even number"):
        ProceduralFrozenLakeEnv(
            fixed_map=[
                "SWFFG",
                "FFFFF",
            ],
        )


def test_glare_ice_is_slippery() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "SMFFG",
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
        assert "M" in board
    finally:
        env.close()


def test_tree_prob_generates_valid_map() -> None:
    env = ProceduralFrozenLakeEnv(
        tree_prob=0.3,
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
        assert any("T" in row for row in env._gridmap)
    finally:
        env.close()


def test_fixed_map_with_glare_and_sleighs() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "SWFFG",
            "FMFFF",
            "FFFFW",
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
            playable = [i for i, ch in enumerate(row) if ch != "T"]
            if playable:
                row_spans.append((playable[0], playable[-1]))
        width = len(board[0]) if board else 0
        for col_idx in range(width):
            playable_rows = [
                r for r, row in enumerate(board) if row[col_idx] != "T"
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
            "STMFG",
            "FFFFF",
            "FFWWF",
        ],
        fog_of_war=False,
        render_mode="rgb_array",
    )
    try:
        env.reset(seed=0)
        icons = env._ensure_tile_icons()
        assert set(icons) == {"T", "M", "W"}
        frame = env.render()
        assert frame is not None
        assert frame.shape[2] == 3
    finally:
        env.close()


def test_warp_reveals_both_sleighs_under_fog() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "SWFFG",
            "FFFFF",
            "FFFFW",
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
            "SWFWG",
            "FFFFF",
            "FWFWF",
        ],
        fog_of_war=False,
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
        fog_of_war=False,
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
            "SWFFG",
            "FFFFF",
            "FFFFW",
        ],
        fog_of_war=False,
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


def _lake_bounding_box(board: list[str]) -> tuple[int, int]:
    rows = [r for r, row in enumerate(board) if any(ch != "T" for ch in row)]
    cols = [c for c in range(len(board[0])) if any(row[c] != "T" for row in board)]
    return cols[-1] - cols[0] + 1, rows[-1] - rows[0] + 1


def test_lake_fits_inside_width_height_envelope() -> None:
    """All playable tiles fit in a max_width × max_height envelope; the jagged
    lake itself may be smaller than the sampled envelope."""
    for seed in range(50):
        env = ProceduralFrozenLakeEnv(
            min_width=5,
            max_width=7,
            min_height=5,
            max_height=7,
            hole_prob=0.0,
            min_hops=1,
            map_seed=seed,
        )
        try:
            env.reset(seed=0)
            width, height = _lake_bounding_box(env._gridmap or [])
            assert width <= 7
            assert height <= 7
        finally:
            env.close()


def test_q_star_is_zero_at_terminal_states() -> None:
    env = ProceduralFrozenLakeEnv(
        fixed_map={"board": ["SFG"], "rewards": {2: 0.7}},
        step_penalty=-0.1,
        emit_q_star=True,
    )
    try:
        env.reset(seed=0)
        q = env.compute_q_table()
        assert np.allclose(q[2], 0.0)  # goal
        env.step(2)
        _, reward, terminated, _, info = env.step(2)
        assert terminated
        assert reward == pytest.approx(0.7 - 0.1)
        assert np.allclose(info["q_star"], 0.0)
    finally:
        env.close()

    env = ProceduralFrozenLakeEnv(fixed_map=["SFHG"], step_penalty=-0.1, emit_q_star=True)
    try:
        env.reset(seed=0)
        assert np.allclose(env.compute_q_table()[2], 0.0)  # hole
    finally:
        env.close()


def test_fog_slip_bump_reveals_actual_tile_not_intended() -> None:
    """Slipping sideways on glare into a tree reveals the tree, not the intended tile."""
    env = ProceduralFrozenLakeEnv(
        fixed_map=[
            "STFG",
            "FMHF",
            "FFFF",
        ],
        fog_of_war=True,
        render_mode="ansi",
    )
    try:
        found = False
        for seed in range(100):
            env.reset(seed=seed)
            assert env._visited is not None
            env._visited[:] = False
            env.unwrapped.s = 5  # glare at (1, 1)
            env._mark_visited(5)
            obs, _, _, _, _ = env.step(2)  # intend RIGHT; may slip UP into tree
            if obs == 5:
                assert env._visited[0, 1]  # the tree actually bumped
                assert not env._visited[1, 2]  # the untouched hole to the right
                found = True
                break
        assert found, "no slip-into-tree case sampled in 100 seeds"
    finally:
        env.close()


def test_p_matrix_carries_real_rewards() -> None:
    """env.P rewards match exactly what step() pays (per-goal reward + step penalty)."""
    env = ProceduralFrozenLakeEnv(
        fixed_map={"board": ["SFG"], "rewards": {2: 0.7}},
        step_penalty=-0.1,
    )
    try:
        env.reset(seed=0)
        assert env.P[0][2] == [(1.0, 1, pytest.approx(-0.1), False)]
        assert env.P[1][2] == [(1.0, 2, pytest.approx(0.7 - 0.1), True)]
        _, reward, terminated, _, _ = env.step(2)
        assert reward == pytest.approx(-0.1)
        assert not terminated
        _, reward, terminated, _, _ = env.step(2)
        assert reward == pytest.approx(0.7 - 0.1)
        assert terminated
    finally:
        env.close()


def test_q_star_includes_env_step_penalty() -> None:
    """q_star is the optimal value of the live MDP: step_penalty from P is included."""
    env = ProceduralFrozenLakeEnv(
        fixed_map=["SFG"],
        step_penalty=-0.5,
        emit_q_star=True,
    )
    try:
        env.reset(seed=0)
        q = env.compute_q_table()
        # From F: goal reward 1.0 plus step penalty -0.5. From S: another -0.5,
        # with the final +0.5 discounted one step.
        assert q[1, 2] == pytest.approx(0.5)
        assert q[0, 2] == pytest.approx(-0.5 + 0.999 * 0.5)
        assert env.q_star_gamma == pytest.approx(0.999)
    finally:
        env.close()


def test_start_pos_outside_canvas_raises() -> None:
    with pytest.raises(ValueError, match="outside the .* canvas"):
        ProceduralFrozenLakeEnv(start_pos=999, map_seed=0)


def test_probability_params_out_of_range_raise() -> None:
    for kwargs in (
        {"hole_prob": 1.5},
        {"tree_prob": -0.1},
        {"glare_prob": 2.0},
        {"slippery_success_rate": 1.1},
        {"start_pos_prob": -0.5},
        {"goal_pos_prob": 7.0},
    ):
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            ProceduralFrozenLakeEnv(**kwargs)


def test_width_height_bounds_validated() -> None:
    with pytest.raises(ValueError, match="min <= max"):
        ProceduralFrozenLakeEnv(min_width=9, max_width=8)
    with pytest.raises(ValueError, match="min <= max"):
        ProceduralFrozenLakeEnv(min_height=9, max_height=8)
    with pytest.raises(ValueError, match=">= 1"):
        ProceduralFrozenLakeEnv(min_width=0)


def test_positions_with_fixed_map_raise() -> None:
    for kwargs in (
        {"start_pos": 0},
        {"start_pos_prob": 0.5},
        {"goal_pos": 2},
        {"goal_pos_prob": 0.5},
    ):
        with pytest.raises(ValueError, match="ignored"):
            ProceduralFrozenLakeEnv(fixed_map=["SFG"], **kwargs)


def test_unplaceable_explicit_positions_raise_with_reason() -> None:
    env = ProceduralFrozenLakeEnv(start_pos=27, goal_pos=27, map_seed=0, max_tries=50)
    try:
        with pytest.raises(RuntimeError, match="goal_pos 27 is not on playable lake ice"):
            env.reset(seed=0)
    finally:
        env.close()


def test_permutations_relabel_obs_and_actions() -> None:
    import json

    base = ProceduralFrozenLakeEnv(map_seed=5, emit_map=True, emit_q_star=True)
    perm = ProceduralFrozenLakeEnv(
        map_seed=5,
        emit_map=True,
        emit_q_star=True,
        permute_obs=True,
        permute_actions=True,
    )
    try:
        base_obs, base_info = base.reset(seed=3)
        perm_obs, perm_info = perm.reset(seed=3)
        base_map = json.loads(base_info["map"])
        perm_map = json.loads(perm_info["map"])
        assert base_map["board"] == perm_map["board"]
        obs_perm = perm_map["obs_permutation"]
        act_perm = perm_map["action_permutation"]
        assert sorted(obs_perm) == list(range(len(obs_perm)))
        assert sorted(act_perm) == [0, 1, 2, 3]
        assert perm_obs == obs_perm[base_obs]
        assert np.allclose(
            perm_info["q_star"], np.asarray(base_info["q_star"])[act_perm]
        )
        for external_action in range(4):
            base.reset(seed=3)
            perm.reset(seed=3)
            b_obs, b_rew, b_term, _, _ = base.step(act_perm[external_action])
            p_obs, p_rew, p_term, _, _ = perm.step(external_action)
            assert p_obs == obs_perm[b_obs]
            assert p_rew == b_rew
            assert p_term == b_term
    finally:
        base.close()
        perm.close()


def test_permutations_resample_on_map_regeneration() -> None:
    import json

    env = ProceduralFrozenLakeEnv(
        map_seed=5, emit_map=True, permute_obs=True, permute_actions=True
    )
    try:
        _, first = env.reset(seed=0)
        _, second = env.reset(seed=0, options={"regenerate_map": True})
        first_map = json.loads(first["map"])
        second_map = json.loads(second["map"])
        assert (
            first_map["obs_permutation"] != second_map["obs_permutation"]
            or first_map["action_permutation"] != second_map["action_permutation"]
        )
    finally:
        env.close()


def test_map_info_has_no_permutations_when_disabled() -> None:
    import json

    env = ProceduralFrozenLakeEnv(map_seed=5, emit_map=True)
    try:
        _, info = env.reset(seed=0)
        map_info = json.loads(info["map"])
        assert "obs_permutation" not in map_info
        assert "action_permutation" not in map_info
    finally:
        env.close()
