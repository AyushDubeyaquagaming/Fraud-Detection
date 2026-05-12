from __future__ import annotations

import pandas as pd

from fraud_detection.components.ccs_features import CCS_FEATURE_COLUMNS, attach_ccs_concentration_features


def _write_profit(tmp_path, rows):
    path = tmp_path / "ccs_daily_profit" / "year=2026" / "month=05"
    path.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(path / "ccs_daily_profit.parquet", index=False)
    return tmp_path / "ccs_daily_profit"


def _member_draw(member_id="A", ccs_id="C1", draw_date="2026-05-07"):
    return pd.DataFrame(
        {
            "member_id": [member_id],
            "ccs_id": [ccs_id],
            "draw_id": [1],
            "draw_date": [pd.Timestamp(draw_date, tz="UTC")],
        }
    )


def test_single_member_ccs_window_sets_solo_not_concentrated(tmp_path):
    profit_path = _write_profit(
        tmp_path,
        [{"ccs_id": "C1", "member_id": "A", "profit_date": pd.Timestamp("2026-05-07").date(), "daily_profit": 100.0}],
    )

    result = attach_ccs_concentration_features(
        _member_draw(),
        ccs_profit_path=profit_path,
        windows_days=[1, 7],
        concentration_threshold=0.7,
    )

    assert result["ccs_profit_share_1d"].iloc[0] == 1.0
    assert result["ccs_solo_member_1d"].iloc[0] == 1.0
    assert result["ccs_high_concentration_1d"].iloc[0] == 0.0


def test_eighty_percent_share_in_multi_member_ccs_is_concentrated(tmp_path):
    profit_path = _write_profit(
        tmp_path,
        [
            {"ccs_id": "C1", "member_id": "A", "profit_date": pd.Timestamp("2026-05-07").date(), "daily_profit": 80.0},
            {"ccs_id": "C1", "member_id": "B", "profit_date": pd.Timestamp("2026-05-07").date(), "daily_profit": 20.0},
        ],
    )

    result = attach_ccs_concentration_features(_member_draw(), ccs_profit_path=profit_path, windows_days=[1], concentration_threshold=0.7)

    assert result["ccs_profit_share_1d"].iloc[0] == 0.8
    assert result["ccs_member_count_1d"].iloc[0] == 2.0
    assert result["ccs_high_concentration_1d"].iloc[0] == 1.0


def test_thirty_percent_share_in_multi_member_ccs_is_not_concentrated(tmp_path):
    profit_path = _write_profit(
        tmp_path,
        [
            {"ccs_id": "C1", "member_id": "A", "profit_date": pd.Timestamp("2026-05-07").date(), "daily_profit": 30.0},
            {"ccs_id": "C1", "member_id": "B", "profit_date": pd.Timestamp("2026-05-07").date(), "daily_profit": 70.0},
        ],
    )

    result = attach_ccs_concentration_features(_member_draw(), ccs_profit_path=profit_path, windows_days=[1], concentration_threshold=0.7)

    assert result["ccs_profit_share_1d"].iloc[0] == 0.3
    assert result["ccs_high_concentration_1d"].iloc[0] == 0.0


def test_non_positive_total_profit_is_neutral(tmp_path):
    profit_path = _write_profit(
        tmp_path,
        [
            {"ccs_id": "C1", "member_id": "A", "profit_date": pd.Timestamp("2026-05-07").date(), "daily_profit": -10.0},
            {"ccs_id": "C1", "member_id": "B", "profit_date": pd.Timestamp("2026-05-07").date(), "daily_profit": 5.0},
        ],
    )

    result = attach_ccs_concentration_features(_member_draw(), ccs_profit_path=profit_path, windows_days=[1], concentration_threshold=0.7)

    assert result["ccs_profit_share_1d"].iloc[0] == 0.0
    assert result["ccs_high_concentration_1d"].iloc[0] == 0.0


def test_profit_share_is_capped_when_other_members_have_losses(tmp_path):
    profit_path = _write_profit(
        tmp_path,
        [
            {"ccs_id": "C1", "member_id": "A", "profit_date": pd.Timestamp("2026-05-07").date(), "daily_profit": 100.0},
            {"ccs_id": "C1", "member_id": "B", "profit_date": pd.Timestamp("2026-05-07").date(), "daily_profit": -90.0},
        ],
    )

    result = attach_ccs_concentration_features(_member_draw(), ccs_profit_path=profit_path, windows_days=[1], concentration_threshold=0.7)

    assert result["ccs_profit_share_1d"].iloc[0] == 1.0
    assert result["ccs_high_concentration_1d"].iloc[0] == 1.0


def test_null_ccs_id_is_neutral(tmp_path):
    profit_path = _write_profit(
        tmp_path,
        [{"ccs_id": "C1", "member_id": "A", "profit_date": pd.Timestamp("2026-05-07").date(), "daily_profit": 100.0}],
    )

    result = attach_ccs_concentration_features(_member_draw(ccs_id=None), ccs_profit_path=profit_path, windows_days=[1], concentration_threshold=0.7)

    assert result["ccs_profit_share_1d"].iloc[0] == 0.0
    assert result["ccs_high_concentration_1d"].iloc[0] == 0.0


def test_null_profit_identifiers_do_not_create_fake_ccs_members(tmp_path):
    profit_path = _write_profit(
        tmp_path,
        [
            {"ccs_id": "C1", "member_id": "A", "profit_date": pd.Timestamp("2026-05-07").date(), "daily_profit": 80.0},
            {"ccs_id": "C1", "member_id": "B", "profit_date": pd.Timestamp("2026-05-07").date(), "daily_profit": 20.0},
            {"ccs_id": None, "member_id": "Z", "profit_date": pd.Timestamp("2026-05-07").date(), "daily_profit": 999.0},
            {"ccs_id": "C1", "member_id": None, "profit_date": pd.Timestamp("2026-05-07").date(), "daily_profit": 999.0},
        ],
    )

    result = attach_ccs_concentration_features(_member_draw(), ccs_profit_path=profit_path, windows_days=[1], concentration_threshold=0.7)

    assert result["ccs_profit_share_1d"].iloc[0] == 0.8
    assert result["ccs_member_count_1d"].iloc[0] == 2.0


def test_window_boundary_is_calendar_day_inclusive(tmp_path):
    profit_path = _write_profit(
        tmp_path,
        [
            {"ccs_id": "C1", "member_id": "A", "profit_date": pd.Timestamp("2026-05-01").date(), "daily_profit": 999.0},
            {"ccs_id": "C1", "member_id": "A", "profit_date": pd.Timestamp("2026-05-02").date(), "daily_profit": 80.0},
            {"ccs_id": "C1", "member_id": "B", "profit_date": pd.Timestamp("2026-05-07").date(), "daily_profit": 20.0},
        ],
    )

    result = attach_ccs_concentration_features(_member_draw(draw_date="2026-05-07"), ccs_profit_path=profit_path, windows_days=[7], concentration_threshold=0.7)

    assert result["ccs_total_profit_7d"].iloc[0] == 1099.0
    assert result["ccs_profit_share_7d"].iloc[0] == (999.0 + 80.0) / 1099.0


def test_column_order_is_stable_for_matched_and_neutral_chunks(tmp_path):
    profit_path = _write_profit(
        tmp_path,
        pd.DataFrame(
            {
                "ccs_id": ["C1"],
                "member_id": ["A"],
                "profit_date": [pd.Timestamp("2026-05-07").date()],
                "daily_profit": [80.0],
                "daily_bet_amount": [100.0],
                "daily_win_points": [180.0],
                "draw_count": [1],
            }
        ),
    )

    matched = attach_ccs_concentration_features(
        _member_draw(),
        ccs_profit_path=profit_path,
        windows_days=[1, 7],
        concentration_threshold=0.7,
    )
    neutral = attach_ccs_concentration_features(
        _member_draw(ccs_id="C2"),
        ccs_profit_path=profit_path,
        windows_days=[1, 7],
        concentration_threshold=0.7,
    )

    assert list(matched.columns) == list(neutral.columns)
    assert list(matched.columns[-len(CCS_FEATURE_COLUMNS) :]) == CCS_FEATURE_COLUMNS


def test_irrelevant_ccs_profit_rows_do_not_affect_features(tmp_path):
    profit_path = _write_profit(
        tmp_path,
        [
            {"ccs_id": "C1", "member_id": "A", "profit_date": pd.Timestamp("2026-05-07").date(), "daily_profit": 80.0},
            {"ccs_id": "C1", "member_id": "B", "profit_date": pd.Timestamp("2026-05-07").date(), "daily_profit": 20.0},
            {"ccs_id": "C2", "member_id": "Z", "profit_date": pd.Timestamp("2026-05-07").date(), "daily_profit": 999999.0},
            {"ccs_id": "C1", "member_id": "OLD", "profit_date": pd.Timestamp("2026-04-01").date(), "daily_profit": 999999.0},
        ],
    )

    result = attach_ccs_concentration_features(_member_draw(), ccs_profit_path=profit_path, windows_days=[1, 7], concentration_threshold=0.7)

    assert result["ccs_profit_share_1d"].iloc[0] == 0.8
    assert result["ccs_total_profit_1d"].iloc[0] == 100.0
    assert result["ccs_total_profit_7d"].iloc[0] == 100.0
