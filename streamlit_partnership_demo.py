from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv()

API_BASE_URL = os.getenv("FRAUD_API_BASE_URL", "http://127.0.0.1:8000")
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE")
LIVE_PREDICTIONS_COLLECTION = os.getenv("LIVE_PREDICTIONS_COLLECTION", "live_predictions")


def _risk_label(value: bool) -> str:
    return "HIGH" if value else "LOW"


@st.cache_data(show_spinner=False, ttl=60)
def load_live_predictions(hours: int) -> pd.DataFrame:
    if not MONGODB_URI or not MONGODB_DATABASE:
        return pd.DataFrame()
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
    collection = client[MONGODB_DATABASE][LIVE_PREDICTIONS_COLLECTION]
    docs = list(
        collection.find(
            {"scored_at": {"$gte": since.isoformat()}},
            {"_id": 0},
        ).sort("scored_at", -1).limit(500)
    )
    return pd.DataFrame(docs)


def post_json(path: str, payload: dict) -> dict:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=60)
    if not response.ok:
        raise RuntimeError(f"{response.status_code}: {response.text}")
    return response.json()


def get_json(path: str, params: dict | None = None) -> dict:
    response = requests.get(f"{API_BASE_URL}{path}", params=params or {}, timeout=60)
    if not response.ok:
        raise RuntimeError(f"{response.status_code}: {response.text}")
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="Partnership Collusion Demo", layout="wide")
st.title("Partnership Collusion Demo")

with st.sidebar:
    st.caption("API")
    api_base_url = st.text_input("Base URL", API_BASE_URL)
    API_BASE_URL = api_base_url.rstrip("/")
    hours = st.slider("Prediction history", 1, 168, 24)

tab_predictions, tab_draw, tab_member, tab_ccs, tab_alerts = st.tabs(
    ["Live Predictions", "Draw Score", "Member Score", "CCS Ranking", "Alert Draws"]
)

with tab_predictions:
    predictions = load_live_predictions(hours)
    if predictions.empty:
        st.info("No live prediction documents found for the selected window.")
    else:
        view = predictions.copy()
        view["risk_tier"] = view["requires_review"].map(_risk_label)
        columns = [
            "scored_at",
            "draw_id",
            "risk_tier",
            "n_members_in_draw",
            "max_stage1_score",
            "max_stage2_score",
            "source_run_id",
        ]
        st.dataframe(view[[column for column in columns if column in view.columns]], use_container_width=True)
        selected_draw = st.selectbox("Draw details", view["draw_id"].astype(str).tolist())
        selected = view.loc[view["draw_id"].astype(str).eq(selected_draw)].iloc[0].to_dict()
        st.subheader("Partnerships")
        st.json(selected.get("partnerships", []))
        st.subheader("Flagged members")
        st.json(selected.get("flagged_members", []))

with tab_draw:
    draw_id_raw = st.text_input("Draw ID", placeholder="7297697")
    if st.button("Score draw", type="primary") and draw_id_raw.strip():
        try:
            draw_id = int(draw_id_raw.strip())
            result = get_json(f"/score/draw/{draw_id}")
            st.metric("Review", "HIGH" if result["requires_review"] else "LOW")
            st.write(f"Members in draw: {result['n_members_in_draw']}")
            st.write(f"Max Stage 1: {result['max_stage1_score']:.3f}")
            st.write(f"Max Stage 2: {result['max_stage2_score']:.3f}")
            if result.get("response_details"):
                st.caption(", ".join(result["response_details"]))
            st.subheader("Partnerships")
            st.dataframe(pd.DataFrame(result.get("partnerships", [])), use_container_width=True)
            st.subheader("Flagged members")
            st.dataframe(pd.DataFrame(result.get("flagged_members", [])), use_container_width=True)
        except ValueError:
            st.error("Draw ID must be a number.")
        except Exception as exc:
            st.error(str(exc))

with tab_member:
    member_id = st.text_input("Member ID", placeholder="GK00236424")
    reference_draw_raw = st.text_input("Reference draw ID (optional)", placeholder="7297697")
    lookback_days = st.slider("Lookback days", 1, 30, 30, key="member_lookback")
    if st.button("Score member", type="primary") and member_id.strip():
        try:
            params = {"lookback_days": lookback_days, "max_draws": 500}
            if reference_draw_raw.strip():
                params["draw_id"] = int(reference_draw_raw.strip())
            result = get_json(f"/score/member/{member_id.strip().upper()}", params=params)
            st.metric("Risk", result["risk_tier"])
            st.write(f"Draws scanned: {result['draws_scanned']}")
            evidence = pd.DataFrame(result["evidence_draws"])
            if evidence.empty:
                st.info("No partnership evidence found in the selected window.")
            else:
                st.dataframe(evidence, use_container_width=True)
        except ValueError:
            st.error("Reference draw ID must be a number.")
        except Exception as exc:
            st.error(str(exc))

with tab_ccs:
    ccs_raw = st.text_area("CCS IDs (optional, one per line)")
    lookback_days = st.slider("Lookback days", 1, 30, 7, key="ccs_lookback")
    max_draws = st.slider("Candidate draws to scan", 1000, 50000, 10000, step=1000, key="ccs_max_draws")
    if st.button("Rank CCS IDs", type="primary"):
        payload = {"lookback_days": lookback_days, "max_draws": max_draws}
        ccs_ids = [line.strip() for line in ccs_raw.splitlines() if line.strip()]
        if ccs_ids:
            payload["ccs_ids"] = ccs_ids
        try:
            result = post_json("/score/ccs", payload)
            st.write(f"Draws scanned: {result['draws_scanned']}")
            scores = pd.DataFrame(result["ccs_scores"])
            if scores.empty:
                st.info("No CCS alerts found in the selected window.")
            else:
                st.dataframe(scores, use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

with tab_alerts:
    lookback_days = st.slider("Lookback days", 1, 30, 7, key="alerts_lookback")
    max_draws = st.slider("Candidate draws to scan", 1000, 50000, 10000, step=1000, key="alerts_max_draws")
    limit = st.slider("Alert limit", 10, 1000, 250, step=10)
    if st.button("Load alert draws", type="primary"):
        try:
            result = get_json(
                "/score/alerts",
                params={"lookback_days": lookback_days, "max_draws": max_draws, "limit": limit},
            )
            st.write(f"Draws scanned: {result['draws_scanned']}")
            st.write(f"Alert draws found: {result['alert_draw_count']}")
            alerts = pd.DataFrame(result["alerts"])
            if alerts.empty:
                st.info("No alert draws found in the selected window.")
            else:
                st.dataframe(alerts, use_container_width=True)
        except Exception as exc:
            st.error(str(exc))
