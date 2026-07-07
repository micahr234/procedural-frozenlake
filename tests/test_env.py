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
