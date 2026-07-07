"""Tabular value iteration for Gymnasium toy-text dynamics."""

from __future__ import annotations

from typing import Any

import numpy as np


def solve_tabular_mdp(
    *,
    P: Any,
    n_states: int,
    n_actions: int,
    gamma: float,
    step_penalty: float = 0.0,
    goal_rewards_by_state: dict[int, float] | None = None,
    max_iter: int = 10_000,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Compute an optimal Q-table for a finite tabular MDP from Gymnasium ``P``.

    Args:
        P: Gymnasium ``env.P`` dynamics, indexed by state and action.
        n_states: Number of states.
        n_actions: Number of actions.
        gamma: Discount factor.
        step_penalty: Added to each transition reward before the Bellman backup.
        goal_rewards_by_state: Override terminal rewards at goal states.
        max_iter: Maximum value-iteration sweeps.
        tolerance: Stop when max absolute Q change is below this.

    Returns:
        Optimal Q-table, shape ``(n_states, n_actions)``, ``float64``.
    """
    g = float(gamma)
    goal_rewards = goal_rewards_by_state or {}
    q = np.zeros((n_states, n_actions), dtype=np.float64)
    for _ in range(int(max_iter)):
        v = q.max(axis=1)
        q_new = np.zeros((n_states, n_actions), dtype=np.float64)
        for s in range(n_states):
            for a in range(n_actions):
                acc = 0.0
                for prob, next_s, r, done in P[s][a]:
                    p = float(prob)
                    ns = int(next_s)
                    rr = float(r)
                    if done and ns in goal_rewards:
                        rr = float(goal_rewards[ns])
                    rr += step_penalty
                    if done:
                        acc += p * rr
                    else:
                        acc += p * (rr + g * v[ns])
                q_new[s, a] = acc
        if np.max(np.abs(q_new - q)) <= float(tolerance):
            return q_new
        q = q_new
    return q
