from __future__ import annotations

from fraud_detection.extraction.candidate_extractor import _build_position_index, _player_to_vectors
from fraud_detection.utils.player_vectors import ROULETTE_POSITIONS, build_position_index, player_to_vectors


def test_shared_player_vector_helper_matches_phase1_behavior() -> None:
    player = {
        "bets": [
            {"number": "0", "bet_amount": 25},
            {"number": "00", "bet_amount": 50},
            {"number": 7, "bet_amount": "100.5"},
            {"number": "bad", "bet_amount": 10},
        ]
    }

    old_coverage, old_amounts = _player_to_vectors(player, _build_position_index(ROULETTE_POSITIONS), 38)
    new_coverage, new_amounts = player_to_vectors(player, build_position_index(ROULETTE_POSITIONS), 38)

    assert new_coverage == old_coverage
    assert new_amounts == old_amounts
