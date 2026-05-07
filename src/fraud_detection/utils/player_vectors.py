from __future__ import annotations

from collections import Counter
from typing import Any


ROULETTE_POSITIONS = ["0", "00", *[str(number) for number in range(1, 37)]]


def build_position_index(board_positions: list[str] | None = None) -> dict[str, int]:
    positions = board_positions or ROULETTE_POSITIONS
    return {str(pos): index for index, pos in enumerate(positions)}


def player_to_vectors(
    player: dict[str, Any],
    position_index: dict[str, int] | None = None,
    board_size: int = 38,
    *,
    unknown_positions: Counter[str] | None = None,
) -> tuple[bytes, list[float]]:
    """Return Phase-1-compatible coverage bytes and amount vector for one player."""
    from fraud_detection.components.feature_engineering import parse_bets

    index = position_index or build_position_index()
    coverage_bytes = bytearray(board_size)
    amounts = [0.0] * board_size
    for bet in parse_bets(player.get("bets")):
        pos = str(bet.get("number"))
        try:
            amount = float(bet.get("bet_amount", 0) or 0)
        except (TypeError, ValueError):
            continue
        if amount <= 0:
            continue
        idx = index.get(pos)
        if idx is None:
            if unknown_positions is not None:
                unknown_positions[pos] += 1
            continue
        coverage_bytes[idx] = 1
        amounts[idx] += amount
    return bytes(coverage_bytes), amounts

