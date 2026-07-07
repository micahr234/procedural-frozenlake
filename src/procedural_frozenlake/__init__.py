"""Procedural Frozen Lake — a Gymnasium environment with generated maps."""

from importlib.metadata import version

from procedural_frozenlake.env import (
    PROCEDURAL_FROZENLAKE_ENV_ID,
    ProceduralFrozenLakeEnv,
    ensure_registered,
)

ensure_registered()

__version__ = version("procedural-frozenlake")

__all__ = [
    "__version__",
    "PROCEDURAL_FROZENLAKE_ENV_ID",
    "ProceduralFrozenLakeEnv",
    "ensure_registered",
]
