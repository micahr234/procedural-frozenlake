"""Map generation and validation for Procedural Frozen Lake."""

from __future__ import annotations

import random
from collections import deque
from typing import Any

from procedural_frozenlake.tiles import (
    ALLOWED_TILES,
    TILE_FROZEN,
    TILE_GOAL,
    TILE_HOLE,
    TILE_MIRROR,
    TILE_SLEIGH,
    TILE_START,
    TILE_TREE,
)


def build_sleigh_partners(gridmap: list[str], canvas_width: int) -> dict[int, int]:
    """Pair consecutive ``W`` tiles in row-major order into bidirectional warps."""
    sleigh_states: list[int] = []
    for r, row in enumerate(gridmap):
        for c, ch in enumerate(row):
            if ch == TILE_SLEIGH:
                sleigh_states.append(r * canvas_width + c)
    if len(sleigh_states) % 2 != 0:
        raise ValueError(
            f"Map must contain an even number of {TILE_SLEIGH!r} sleigh tiles; "
            f"got {len(sleigh_states)}."
        )
    partners: dict[int, int] = {}
    for i in range(0, len(sleigh_states), 2):
        a, b = sleigh_states[i], sleigh_states[i + 1]
        partners[a] = b
        partners[b] = a
    return partners


def goal_states_from_gridmap(
    gridmap: list[str], canvas_width: int | None = None
) -> list[int]:
    """Return state indices of all ``G`` tiles in row-major order."""
    cols = canvas_width if canvas_width is not None else len(gridmap[0])
    out: list[int] = []
    for r, row in enumerate(gridmap):
        for c, ch in enumerate(row):
            if ch == TILE_GOAL:
                out.append(r * cols + c)
    return out


def normalize_and_validate_fixed_map(
    fixed_map: list[str] | tuple[str, ...],
) -> list[str]:
    """Normalize a fixed board and reject empty, ragged, or invalid maps."""
    if not fixed_map:
        raise ValueError("fixed_map cannot be empty.")
    gridmap = [str(row) for row in fixed_map]
    row_width = len(gridmap[0])
    if row_width == 0:
        raise ValueError("fixed_map rows cannot be empty.")
    if any(len(row) != row_width for row in gridmap):
        raise ValueError("fixed_map rows must all have the same width.")

    invalid_chars = sorted(
        {ch for row in gridmap for ch in row if ch not in ALLOWED_TILES}
    )
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

    build_sleigh_partners(gridmap, row_width)
    return gridmap


def _is_blocked_for_path(ch: str) -> bool:
    return ch in {TILE_HOLE, TILE_TREE}


def _find_path_to_goal_bfs(
    gridmap: list[str],
    state: int,
) -> tuple[list[tuple[int, int]], list[int]] | None:
    rows = len(gridmap)
    cols = len(gridmap[0])
    board = [list(row) for row in gridmap]
    partners = build_sleigh_partners(gridmap, cols)
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
            if 0 <= nr < rows and 0 <= nc < cols and not _is_blocked_for_path(
                board[nr][nc]
            ):
                next_positions.append((nr, nc))
        if board[r][c] == TILE_SLEIGH:
            tr, tc = divmod(partners[r * cols + c], cols)
            if not _is_blocked_for_path(board[tr][tc]):
                next_positions.append((tr, tc))
        for nr, nc in next_positions:
            if board[nr][nc] == TILE_SLEIGH:
                tr, tc = divmod(partners[nr * cols + nc], cols)
                queue.append(((tr, tc), path + [(tr, tc)], actions + [0]))
            else:
                queue.append(((nr, nc), path + [(nr, nc)], actions + [0]))
    return None


def _map_is_valid(gridmap: list[str], min_hops: int) -> bool:
    board = [list(row) for row in gridmap]
    found_start = False
    cols = len(gridmap[0])
    for r, row in enumerate(board):
        for c, ch in enumerate(row):
            if ch != TILE_START:
                continue
            found_start = True
            state = r * cols + c
            result = _find_path_to_goal_bfs(gridmap=gridmap, state=state)
            if result is None:
                return False
            _, actions = result
            if len(actions) < min_hops:
                return False
    return found_start


def _largest_connected_component(mask: list[list[bool]]) -> set[tuple[int, int]]:
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


def _edge_shoreline_shift(
    rng: random.Random,
    count: int,
    domain: int,
    min_span: int,
    jaggedness: int,
    max_protrude: int,
) -> list[int]:
    """Signed per-edge shift from the nominal inset rectangle.

    Positive values carve bays inward (more trees); negative values push
    playable ice outward into the tree border band (peninsulas).
    """
    if jaggedness <= 0:
        return [0] * count
    max_inward = min(jaggedness, max(0, domain - min_span))
    max_outward = min(jaggedness, max_protrude)
    shift = [rng.randint(-max_outward, max_inward)]
    for i in range(1, count):
        delta = rng.randint(-jaggedness, jaggedness)
        shift.append(max(-max_outward, min(max_inward, shift[i - 1] + delta)))
    return shift


def _clamp_inset_index(index: int, span: int) -> int:
    return min(max(index, 0), span - 1)


def _mask_is_connected(mask: list[list[bool]]) -> bool:
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
    component = _largest_connected_component(mask)
    return len(component) == total


def _generate_lake_mask(
    rng: random.Random,
    *,
    canvas_width: int,
    canvas_height: int,
    min_border: int,
    max_border: int,
    shoreline_jaggedness: int,
) -> tuple[list[list[bool]], int]:
    """Playable lake ice with a jagged tree shoreline on all four sides.

    A tree border thickness is sampled from ``min_border..max_border`` on every
    side. Playable tiles start in the inset rectangle, then shoreline shifts
    carve bays inward or push peninsulas outward by up to
    ``shoreline_jaggedness`` tiles.
    """
    for _ in range(64):
        border = rng.randint(min_border, max_border)
        w = canvas_width - 2 * border
        h = canvas_height - 2 * border
        if w < 2 or h < 2:
            continue
        top = border
        left = border

        min_span_h = max(2, w - 2 * shoreline_jaggedness)
        min_span_v = max(2, h - 2 * shoreline_jaggedness)

        left_shift = _edge_shoreline_shift(
            rng, h, w, min_span_h, shoreline_jaggedness, border
        )
        right_shift = _edge_shoreline_shift(
            rng, h, w, min_span_h, shoreline_jaggedness, border
        )
        top_shift = _edge_shoreline_shift(
            rng, w, h, min_span_v, shoreline_jaggedness, border
        )
        bottom_shift = _edge_shoreline_shift(
            rng, w, h, min_span_v, shoreline_jaggedness, border
        )

        mask = [[False] * canvas_width for _ in range(canvas_height)]
        reach = shoreline_jaggedness
        for row_offset in range(-reach, h + reach):
            row_idx = top + row_offset
            if not 0 <= row_idx < canvas_height:
                continue
            for col_offset in range(-reach, w + reach):
                col_idx = left + col_offset
                if not 0 <= col_idx < canvas_width:
                    continue
                ro_for_lr = _clamp_inset_index(row_offset, h)
                co_for_tb = _clamp_inset_index(col_offset, w)
                left_ok = col_offset >= left_shift[ro_for_lr]
                right_ok = col_offset < w - right_shift[ro_for_lr]
                top_ok = row_offset >= top_shift[co_for_tb]
                bottom_ok = row_offset < h - bottom_shift[co_for_tb]
                if left_ok and right_ok and top_ok and bottom_ok:
                    mask[row_idx][col_idx] = True

        if _mask_is_connected(mask):
            return mask, border

    raise RuntimeError("Could not generate a connected lake mask.")


def _mask_to_base_gridmap(mask: list[list[bool]], canvas_width: int) -> list[str]:
    rows: list[str] = []
    for row in mask:
        rows.append(
            "".join(TILE_FROZEN if playable else TILE_TREE for playable in row)
        )
    return rows


def _playable_indices(gridmap: list[str], canvas_width: int) -> list[int]:
    indices: list[int] = []
    for r, row in enumerate(gridmap):
        for c, ch in enumerate(row):
            if ch != TILE_TREE:
                indices.append(r * canvas_width + c)
    return indices


def _place_sleigh_pairs(
    rng: random.Random,
    gridmap: list[str],
    canvas_width: int,
    pair_count: int,
) -> None:
    if pair_count <= 0:
        return
    candidates = [
        i
        for i in _playable_indices(gridmap, canvas_width)
        if gridmap[i // canvas_width][i % canvas_width]
        in {TILE_FROZEN, TILE_MIRROR}
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


def _apply_tree_prob(
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


def _apply_mirror_prob(
    rng: random.Random,
    gridmap: list[str],
    canvas_width: int,
    mirror_prob: float,
) -> None:
    if mirror_prob <= 0.0:
        return
    for r, row in enumerate(gridmap):
        chars = list(row)
        for c, ch in enumerate(chars):
            if ch == TILE_FROZEN and rng.random() < mirror_prob:
                chars[c] = TILE_MIRROR
        gridmap[r] = "".join(chars)


def _generate_map(
    rng: random.Random,
    width: int,
    height: int,
    min_border: int,
    max_border: int,
    shoreline_jaggedness: int,
    hole_prob: float,
    start_pos: list[int] | None,
    start_pos_prob: float | None,
    goal_pos: list[int] | None,
    goal_pos_prob: float | None,
    tree_prob: float,
    mirror_prob: float,
    sleigh_pair_count: int,
) -> tuple[list[str], int]:
    canvas_width = width
    canvas_height = height
    mask, border = _generate_lake_mask(
        rng,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        min_border=min_border,
        max_border=max_border,
        shoreline_jaggedness=shoreline_jaggedness,
    )
    gridmap = _mask_to_base_gridmap(mask, canvas_width)
    _apply_tree_prob(rng, gridmap, tree_prob)

    available_index = _playable_indices(gridmap, canvas_width)
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

    if start_pos is None and start_pos_prob is None:
        start_positions: list[int] = [rng.choice(available_index)]
    elif start_pos is None and start_pos_prob is not None:
        start_positions = [
            i for i in available_index if rng.random() < start_pos_prob
        ]
        if not start_positions:
            raise ValueError(
                "start_pos_prob sampled zero start tiles; "
                "increase start_pos_prob or provide start_pos."
            )
    else:
        start_positions = list(start_pos or [])
    place_tiles(start_positions, TILE_START, "start_pos")

    if goal_pos is None and goal_pos_prob is None:
        goal_positions: list[int] = [rng.choice(available_index)]
    elif goal_pos is None and goal_pos_prob is not None:
        goal_positions = [
            i for i in available_index if rng.random() < goal_pos_prob
        ]
        if not goal_positions:
            raise ValueError(
                "goal_pos_prob sampled zero goal tiles; "
                "increase goal_pos_prob or provide goal_pos."
            )
    else:
        goal_positions = list(goal_pos or [])
    place_tiles(goal_positions, TILE_GOAL, "goal_pos")

    for i in list(available_index):
        if rng.random() < hole_prob:
            r, c = divmod(i, canvas_width)
            row = list(gridmap[r])
            row[c] = TILE_HOLE
            gridmap[r] = "".join(row)
            available_index.remove(i)

    _place_sleigh_pairs(rng, gridmap, canvas_width, sleigh_pair_count)
    _apply_mirror_prob(rng, gridmap, canvas_width, mirror_prob)
    return gridmap, border


def generate_valid_map(
    rng: random.Random,
    min_hops: int,
    max_tries: int,
    **kwargs: Any,
) -> tuple[list[str], int]:
    """Sample maps until one is valid (reachable goals with ``min_hops``), or fail."""
    last_error: str | None = None
    for _ in range(max_tries):
        try:
            gridmap, border = _generate_map(rng=rng, **kwargs)
        except (RuntimeError, ValueError) as exc:
            last_error = str(exc)
            continue
        if _map_is_valid(gridmap, min_hops=min_hops):
            return gridmap, border
    message = (
        "Could not generate a valid Procedural Frozen Lake map. "
        "Try lower hole_prob, lower min_hops, larger dimensions, "
        "or adjust start_pos_prob / goal_pos_prob."
    )
    if last_error is not None:
        message += f" Last generation failure: {last_error}"
    raise RuntimeError(message)
