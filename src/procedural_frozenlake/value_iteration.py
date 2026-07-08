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
    terminal_states: set[int] | None = None,
    max_iter: int = 10_000,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Compute an optimal Q-table for a finite tabular MDP from Gymnasium ``P``.

    Args:
        P: Gymnasium ``env.P`` dynamics, indexed by state and action. Transition
            rewards are used as-is.
        n_states: Number of states.
        n_actions: Number of actions.
        gamma: Discount factor.
        terminal_states: States whose Q-values are pinned to zero (no further
            return is obtainable once the episode has ended there).
        max_iter: Maximum value-iteration sweeps.
        tolerance: Stop when max absolute Q change is below this.

    Returns:
        Optimal Q-table, shape ``(n_states, n_actions)``, ``float64``.
    """
    g = float(gamma)
    terminals = frozenset(terminal_states or ())
    q = np.zeros((n_states, n_actions), dtype=np.float64)
    for _ in range(int(max_iter)):
        v = q.max(axis=1)
        q_new = np.zeros((n_states, n_actions), dtype=np.float64)
        for s in range(n_states):
            if s in terminals:
                continue
            for a in range(n_actions):
                acc = 0.0
                for prob, next_s, r, done in P[s][a]:
                    p = float(prob)
                    ns = int(next_s)
                    rr = float(r)
                    if done:
                        acc += p * rr
                    else:
                        acc += p * (rr + g * v[ns])
                q_new[s, a] = acc
        if np.max(np.abs(q_new - q)) <= float(tolerance):
            return q_new
        q = q_new
    return q
