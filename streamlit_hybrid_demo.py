from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from hybrid_inference import (
    HybridInferenceError,
    InvalidMemberIdError,
    MemberNotFoundError,
    ScoreResult,
    ensure_artifacts,
    score_member_id,
    validate_member_id,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_cache"
SCORED_PATH = DATA_DIR / "hybrid_scored_players.parquet"
ALERT_PATH = DATA_DIR / "alert_queue.csv"
EVAL_PATH = DATA_DIR / "hybrid_evaluation.json"

PRIMARY_FEATURES = [
    "draws_played",
    "total_staked",
    "avg_entropy",
    "avg_gini",
    "template_reuse_ratio",
    "avg_tiny_bet_ratio",
    "avg_max_bet_share",
    "avg_nonzero_bets_per_draw",
]

DISPLAY_COLUMNS = [
    "member_id",
    "primary_ccs_id",
    "risk_tier",
    "risk_score",
    "anomaly_score",
    "supervised_score",
    "draws_played",
    "total_staked",
    "avg_entropy",
    "template_reuse_ratio",
    "avg_tiny_bet_ratio",
]

LIVE_RESULT_STATE_KEY = "live_result_payload"


@st.cache_data(show_spinner=False)
def load_assets() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not SCORED_PATH.exists():
        raise FileNotFoundError(f"Missing scored player file: {SCORED_PATH}")
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Missing evaluation file: {EVAL_PATH}")

    scored = pd.read_parquet(SCORED_PATH)
    alert_queue = pd.read_csv(ALERT_PATH) if ALERT_PATH.exists() else pd.DataFrame()
    with open(EVAL_PATH, "r", encoding="utf-8") as handle:
        evaluation = json.load(handle)

    scored = scored.copy()
    scored["member_id"] = scored["member_id"].astype(str).str.upper().str.strip()
    scored["risk_rank"] = scored["risk_score"].rank(ascending=False, method="min").astype(int)
    scored["risk_percentile"] = scored["risk_score"].rank(pct=True)
    scored["anomaly_percentile"] = scored["anomaly_score"].rank(pct=True)
    scored["supervised_percentile"] = scored["supervised_score"].rank(pct=True)
    recommendation_map = {
        "HIGH": "Investigate now",
        "MEDIUM": "Monitor / secondary review",
        "LOW": "Low priority",
    }
    risk_labels = scored["risk_tier"].astype(str)
    review_recommendation = risk_labels.map(recommendation_map).astype(object)
    scored["review_recommendation"] = np.where(review_recommendation.notna(), review_recommendation, "Unassigned")

    return scored, alert_queue, evaluation


@st.cache_resource(show_spinner=False)
def warm_inference_artifacts() -> bool:
    ensure_artifacts()
    return True


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_value(value) -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    return str(value)


def feature_profile(scored: pd.DataFrame, player_row: pd.Series) -> pd.DataFrame:
    rows = []
    for feature in PRIMARY_FEATURES:
        if feature not in scored.columns:
            continue
        series = scored[feature].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        value = float(player_row[feature])
        rows.append(
            {
                "feature": feature,
                "value": value,
                "cohort_median": float(series.median()),
                "percentile": float((series < value).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("percentile", ascending=False)


def nearest_peers(scored: pd.DataFrame, player_row: pd.Series, limit: int = 8) -> pd.DataFrame:
    peer_df = scored[["member_id", "cluster_id", "risk_tier", "risk_score", "style_pc1", "style_pc2"]].copy()
    peer_df["distance"] = np.sqrt(
        (peer_df["style_pc1"] - float(player_row["style_pc1"])) ** 2
        + (peer_df["style_pc2"] - float(player_row["style_pc2"])) ** 2
    )
    peer_df = peer_df[peer_df["member_id"] != player_row["member_id"]]
    return peer_df.sort_values("distance").head(limit)


def pca_chart(scored: pd.DataFrame, player_row: pd.Series):
    plot_df = scored.copy()
    plot_df["selected"] = plot_df["member_id"].eq(player_row["member_id"])
    plot_df["known_fraud"] = plot_df["event_fraud_flag"].astype(int)

    fig = px.scatter(
        plot_df,
        x="style_pc1",
        y="style_pc2",
        color="risk_score",
        color_continuous_scale="YlOrRd",
        hover_data={
            "member_id": True,
            "risk_tier": True,
            "risk_score": ':.3f',
            "anomaly_score": ':.3f',
            "supervised_score": ':.3f',
            "selected": False,
            "known_fraud": False,
        },
        opacity=0.45,
        height=520,
        title="Cohort behaviour map with selected player overlay",
    )

    fraud_df = plot_df[plot_df["known_fraud"] == 1]
    fig.add_scatter(
        x=fraud_df["style_pc1"],
        y=fraud_df["style_pc2"],
        mode="markers",
        name="Known fraud",
        marker={"size": 11, "symbol": "x", "color": "black", "line": {"width": 1}},
        hovertext=fraud_df["member_id"],
        hovertemplate="Known fraud<br>%{hovertext}<extra></extra>",
    )

    fig.add_scatter(
        x=[player_row["style_pc1"]],
        y=[player_row["style_pc2"]],
        mode="markers",
        name="Selected player",
        marker={"size": 18, "symbol": "diamond", "color": "deepskyblue", "line": {"width": 2, "color": "navy"}},
        hovertemplate=f"Selected<br>{player_row['member_id']}<extra></extra>",
    )
    fig.update_layout(margin={"l": 10, "r": 10, "t": 55, "b": 10})
    return fig


def with_review_labels(scored: pd.DataFrame) -> pd.DataFrame:
    df = scored.copy()
    recommendation_map = {
        "HIGH": "Investigate now",
        "MEDIUM": "Monitor / secondary review",
        "LOW": "Low priority",
    }
    if "risk_score" in df.columns:
        computed_rank = df["risk_score"].rank(ascending=False, method="min").astype(int)
        computed_percentile = df["risk_score"].rank(pct=True)
        if "risk_rank" not in df.columns:
            df["risk_rank"] = computed_rank
        else:
            df["risk_rank"] = df["risk_rank"].fillna(computed_rank)
        if "risk_percentile" not in df.columns:
            df["risk_percentile"] = computed_percentile
        else:
            df["risk_percentile"] = df["risk_percentile"].fillna(computed_percentile)
    if "anomaly_score" in df.columns:
        computed_anomaly_percentile = df["anomaly_score"].rank(pct=True)
        if "anomaly_percentile" not in df.columns:
            df["anomaly_percentile"] = computed_anomaly_percentile
        else:
            df["anomaly_percentile"] = df["anomaly_percentile"].fillna(computed_anomaly_percentile)
    if "supervised_score" in df.columns:
        computed_supervised_percentile = df["supervised_score"].rank(pct=True)
        if "supervised_percentile" not in df.columns:
            df["supervised_percentile"] = computed_supervised_percentile
        else:
            df["supervised_percentile"] = df["supervised_percentile"].fillna(computed_supervised_percentile)
    risk_labels = df["risk_tier"].astype(str)
    review_recommendation = risk_labels.map(recommendation_map).astype(object)
    if "score_reliability" in df.columns:
        medium_mask = df["score_reliability"].astype(str).eq("medium")
        low_mask = df["score_reliability"].astype(str).eq("low")
        review_recommendation = pd.Series(
            np.where(
            low_mask,
            "Collect more history before escalation",
            review_recommendation,
            ),
            index=df.index,
        )
        review_recommendation = pd.Series(
            np.where(
            medium_mask & risk_labels.eq("HIGH"),
            "Investigate carefully (limited history)",
            review_recommendation,
            ),
            index=df.index,
        )
    filled_recommendation = np.where(review_recommendation.notna(), review_recommendation, "Unassigned")
    if "review_recommendation" not in df.columns:
        df["review_recommendation"] = filled_recommendation
    else:
        df["review_recommendation"] = df["review_recommendation"].fillna(pd.Series(filled_recommendation, index=df.index))
    return df


def build_augmented_reference(scored: pd.DataFrame, live_row: dict) -> pd.DataFrame:
    combined = pd.concat([scored, pd.DataFrame([live_row])], ignore_index=True, sort=False)
    return with_review_labels(combined)


def get_member_row(scored: pd.DataFrame, member_id: str) -> pd.Series:
    return scored.loc[scored["member_id"] == member_id].iloc[0]


def serialize_live_result(result: ScoreResult) -> dict:
    return {
        "member_id": result.scored_row["member_id"],
        "scored_row": result.scored_row,
        "raw_rows": result.raw_rows,
        "history_rows_used": result.history_rows_used,
        "matched_fraud_rows": result.matched_fraud_rows,
        "reliability": result.reliability,
        "notes": result.notes,
        "source": result.source,
    }


def main() -> None:
    st.set_page_config(page_title="Hybrid Fraud Demo", page_icon="🎯", layout="wide")
    st.title("Hybrid Fraud Detection Demo")
    st.caption(
        "Lookup a real member ID from the scored cohort and inspect the current hybrid risk output. "
        "This demo shows ranking and review priority, not a final legal fraud verdict."
    )

    try:
        scored, alert_queue, evaluation = load_assets()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    scored = with_review_labels(scored)

    if LIVE_RESULT_STATE_KEY not in st.session_state:
        st.session_state[LIVE_RESULT_STATE_KEY] = None

    try:
        warm_inference_artifacts()
        inference_ready = True
        inference_error = None
    except Exception as exc:
        inference_ready = False
        inference_error = str(exc)

    selected_member = None
    live_result: ScoreResult | None = None
    player_row = None
    reference_scored = scored
    selected_source = "precomputed"
    query = ""
    score_live = False

    with st.sidebar:
        st.header("Demo Controls")
        st.metric("Players in cohort", f"{len(scored):,}")
        st.metric("Known fraud in cohort", evaluation.get("fraud_players", 0))
        st.metric("High-risk players", evaluation.get("risk_tier_distribution", {}).get("HIGH", 0))

        query = st.text_input("Search member ID", placeholder="e.g. GK00100019").strip().upper()
        member_ids = scored["member_id"].sort_values().tolist()

        if query:
            matches = [member_id for member_id in member_ids if query in member_id]
            if not matches:
                matches = member_ids[:50]
        else:
            matches = alert_queue["member_id"].astype(str).str.upper().tolist() if not alert_queue.empty else member_ids[:50]

        if query and query not in member_ids:
            if inference_ready:
                st.info("This ID is not in the local scored cohort. Use the button below to score it live from MongoDB.")
            else:
                st.warning(f"Live inference is unavailable: {inference_error}")

        selected_member = st.selectbox("Browse scored cohort", options=matches, index=0 if matches else None)
        score_live = st.button("Score typed ID from MongoDB", type="primary", disabled=not bool(query) or not inference_ready, use_container_width=True)
        validation_mode = st.toggle("Internal validation mode", value=True, help="Show known-fraud labels and evaluation-only fields for internal review.")

    live_payload = st.session_state.get(LIVE_RESULT_STATE_KEY)
    if live_payload and live_payload.get("member_id") == query:
        selected_source = "mongo_live"
        reference_scored = build_augmented_reference(scored, live_payload["scored_row"])
        selected_member = live_payload["member_id"]
        player_row = get_member_row(reference_scored, selected_member)

    if score_live and query:
        try:
            validated_query = validate_member_id(query)
            if validated_query in scored["member_id"].values:
                st.session_state[LIVE_RESULT_STATE_KEY] = None
                live_payload = None
                selected_member = validated_query
                selected_source = "precomputed"
            else:
                with st.spinner("Fetching raw history from MongoDB and scoring the member..."):
                    live_result = score_member_id(validated_query)
                live_payload = serialize_live_result(live_result)
                st.session_state[LIVE_RESULT_STATE_KEY] = live_payload
                reference_scored = build_augmented_reference(scored, live_result.scored_row)
                selected_member = validated_query
                player_row = get_member_row(reference_scored, selected_member)
                selected_source = "mongo_live"
        except (InvalidMemberIdError, MemberNotFoundError, HybridInferenceError) as exc:
            st.error(str(exc))
            st.stop()

    if player_row is None:
        if not selected_member:
            st.warning("No member selected.")
            st.stop()
        player_row = get_member_row(scored, selected_member)

    if selected_source == "precomputed":
        reference_scored = scored

    rank_display = f"#{int(player_row['risk_rank'])} / {len(reference_scored):,}"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Review recommendation", str(player_row["review_recommendation"]))
    c2.metric("Risk tier", str(player_row["risk_tier"]))
    c3.metric("Risk score", f"{float(player_row['risk_score']):.3f}", help="Operational hybrid ranking score.")
    c4.metric("Cohort rank", rank_display, delta=f"Top {format_pct(1 - float(player_row['risk_percentile']))}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Anomaly score", f"{float(player_row['anomaly_score']):.3f}", delta=f"{format_pct(float(player_row['anomaly_percentile']))} percentile")
    c6.metric("Supervised score", f"{float(player_row['supervised_score']):.3f}", delta=f"{format_pct(float(player_row['supervised_percentile']))} percentile")
    c7.metric("Draws played", f"{int(player_row['draws_played']):,}")

    if selected_source == "mongo_live" and live_payload is not None:
        st.success(
            f"Live-scored from MongoDB: {live_payload['raw_rows']} raw rows, {live_payload['history_rows_used']} usable history rows, reliability={live_payload['reliability']}."
        )
        for note in live_payload["notes"]:
            st.warning(note)

    if validation_mode:
        if selected_source == "mongo_live":
            st.info(
                f"Known-fraud label in current extract: {int(player_row['event_fraud_flag'])} | "
                "Evaluation-only scores are not available for one-off live scoring."
            )
        else:
            st.info(
                f"Known-fraud label in current extract: {int(player_row['event_fraud_flag'])} | "
                f"Evaluation-only hybrid score: {format_value(player_row.get('risk_score_eval', np.nan))}"
            )

    tab1, tab2, tab3, tab4 = st.tabs(["Player Summary", "Feature Profile", "Behaviour Map", "Alert Queue"])

    with tab1:
        st.subheader(f"Member {selected_member}")
        summary_cols = DISPLAY_COLUMNS.copy()
        if validation_mode and selected_source != "mongo_live":
            summary_cols += ["event_fraud_flag", "risk_score_eval", "supervised_score_eval"]
        elif validation_mode:
            summary_cols += ["event_fraud_flag"]
        if selected_source == "mongo_live":
            summary_cols += ["raw_history_rows", "history_rows_used", "matched_fraud_rows", "score_reliability", "source"]
        summary_df = pd.DataFrame(
            {
                "field": summary_cols,
                "value": [format_value(player_row[col]) for col in summary_cols],
            }
        )
        st.dataframe(summary_df, width="stretch", hide_index=True)

        peers = nearest_peers(reference_scored, player_row)
        st.subheader("Nearest behavioural peers")
        st.dataframe(peers, width="stretch", hide_index=True)

    with tab2:
        profile_df = feature_profile(reference_scored, player_row)
        st.subheader("Selected feature percentiles vs cohort")
        if profile_df.empty:
            st.warning("No profile features available.")
        else:
            fig = px.bar(
                profile_df.sort_values("percentile"),
                x="percentile",
                y="feature",
                orientation="h",
                text=profile_df.sort_values("percentile")["value"].round(3),
                height=420,
                color="percentile",
                color_continuous_scale="Blues",
                range_x=[0, 1],
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis_tickformat=".0%", margin={"l": 10, "r": 10, "t": 10, "b": 10})
            st.plotly_chart(fig, width="stretch")
            st.dataframe(profile_df, width="stretch", hide_index=True)

    with tab3:
        st.subheader("Player position in behaviour space")
        st.plotly_chart(pca_chart(reference_scored, player_row), width="stretch")
        st.caption(
            "The selected player is highlighted on the same style-PCA space used for the CTO-facing behaviour view. "
            "Known fraud points are marked with black X symbols for internal demos."
        )

    with tab4:
        st.subheader("Current alert queue")
        if alert_queue.empty:
            st.warning("alert_queue.csv not found.")
        else:
            queue_view = alert_queue.copy()
            queue_view["selected"] = queue_view["member_id"].astype(str).str.upper().eq(selected_member)
            st.dataframe(queue_view, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()