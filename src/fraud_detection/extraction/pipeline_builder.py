from __future__ import annotations

from datetime import datetime


def build_candidate_pipeline(
    chunk_start: datetime,
    chunk_end: datetime,
    game: str,
    min_total_bet_amount: float,
    min_qualifying_players: int,
    *,
    timestamp_field: str = "trans_date",
    match_start: datetime | None = None,
    match_end: datetime | None = None,
) -> list[dict]:
    match_start = match_start or chunk_start
    match_end = match_end or chunk_end
    return [
        {
            "$match": {
                "game": game,
                timestamp_field: {"$gte": match_start, "$lt": match_end},
                "total_bet_amount": {"$gte": min_total_bet_amount},
            }
        },
        {
            "$group": {
                "_id": "$draw_id",
                "qualifying_player_count": {"$sum": 1},
                "trans_date_min": {"$min": f"${timestamp_field}"},
                "trans_date_max": {"$max": f"${timestamp_field}"},
                "players": {
                    "$push": {
                        "member_id": "$member_id",
                        "ccs_id": "$ccs_id",
                        "total_bet_amount": "$total_bet_amount",
                        "win_points": "$win_points",
                        "bets": "$bets",
                    }
                },
            }
        },
        {
            "$match": {
                "qualifying_player_count": {"$gte": min_qualifying_players},
                "trans_date_min": {"$gte": chunk_start, "$lt": chunk_end},
            }
        },
    ]
