# Contributing to Procedural Frozen Lake

Contributions are welcome — bug reports, feature requests, documentation improvements, and pull requests.

## Development setup

```bash
git clone https://github.com/micahr234/procedural-frozenlake.git
cd procedural-frozenlake
source scripts/install.sh
```

This creates a Python 3.12 virtual environment and installs the package in editable mode with dev dependencies.

Notebooks under [`examples/`](examples/) are committed **without** outputs. The workspace sets `"notebook.transientOutputs": true` in [`.vscode/settings.json`](.vscode/settings.json), so Cursor/VS Code does not write cell outputs to disk when you run and save a notebook here.

If you edit notebooks with another tool, clear outputs before committing.

## Pull request workflow

1. Fork the repository and create a branch from `main`.
2. Make your changes. Keep commits focused.
3. Run tests (`.venv/bin/pytest`) and type-check (`pyright src/`) before opening a PR.
4. Open a pull request against `main` with a clear description of what changed and why.

Tests live under [`tests/`](tests/). If you add a feature, add or extend a test and/or update the example notebook.

## Code style

- Python 3.12+, type-annotated throughout.
- Follow existing patterns in `src/procedural_frozenlake/`.
- Avoid silent fallbacks — raise clear errors when preconditions are not met.
- Comments should explain *why*, not *what*.

## Releasing to PyPI

Publishing is automated by [`.github/workflows/publish.yml`](.github/workflows/publish.yml) using [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC).

To publish a release:

1. Bump `version` in `pyproject.toml` on `main`.
2. Update `CHANGELOG.md` — promote `[Unreleased]` to the new version with today's date.
3. Commit, push, and create an annotated tag matching the version (e.g. `v0.1.1`).
4. Push the tag: `git push origin v0.1.1` — the Publish workflow runs on tag push.

## Questions

Open a GitHub Discussion or issue.
