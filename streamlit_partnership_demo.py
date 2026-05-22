from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fraud_detection.orchestration_client import (
    get_flow_run_status,
    trigger_full_cycle_flow,
)


load_dotenv()

API_BASE_URL = os.getenv("FRAUD_API_BASE_URL", "http://127.0.0.1:8000")
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE")
LIVE_PREDICTIONS_COLLECTION = os.getenv("LIVE_PREDICTIONS_COLLECTION", "live_predictions")
MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL") or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
ACTIVE_PREFECT_STATES = {"SCHEDULED", "PENDING", "RUNNING"}


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

tab_predictions, tab_draw, tab_member, tab_ccs, tab_alerts, tab_retrain = st.tabs(
    ["Live Predictions", "Draw Score", "Member Score", "CCS Ranking", "Alert Draws", "Retrain"]
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
            flagged_df = pd.DataFrame(result.get("flagged_members", []))
            st.dataframe(flagged_df, use_container_width=True)
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
                st.caption("CCS results are grouped from the latest promoted weekly alert queue, not from a separate CCS model.")
                summary_columns = [
                    "ccs_id",
                    "risk_tier",
                    "flagged_member_count",
                    "flagged_members",
                    "evidence_draw_ids",
                ]
                st.dataframe(scores[[column for column in summary_columns if column in scores.columns]], use_container_width=True)
                selected_ccs = st.selectbox("CCS evidence details", scores["ccs_id"].astype(str).tolist())
                selected = scores.loc[scores["ccs_id"].astype(str).eq(selected_ccs)].iloc[0].to_dict()
                evidence = pd.DataFrame(selected.get("evidence", []))
                if evidence.empty:
                    st.info("No evidence rows available for the selected CCS ID.")
                else:
                    st.dataframe(evidence, use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

with tab_alerts:
    lookback_days = st.slider("Lookback days", 1, 30, 7, key="alerts_lookback")
    max_draws = st.slider("Candidate draws to scan", 1000, 50000, 10000, step=1000, key="alerts_max_draws")
    limit = st.slider("Alert limit", 10, 1000, 250, step=10)
    min_bet_amount = st.number_input("Minimum flagged bet amount", min_value=0.0, value=0.0, step=1000.0)
    if st.button("Load alert draws", type="primary"):
        try:
            params = {"lookback_days": lookback_days, "max_draws": max_draws, "limit": limit}
            if min_bet_amount > 0:
                params["betAmount"] = min_bet_amount
            result = get_json(
                "/score/alerts",
                params=params,
            )
            st.session_state["alerts_result"] = result
            st.session_state["alerts_params"] = params
            st.session_state.pop("alerts_selected_draw", None)
        except Exception as exc:
            st.error(str(exc))

    result = st.session_state.get("alerts_result")
    if result:
        st.write(f"Draws scanned: {result['draws_scanned']}")
        st.write(f"Alert draws found: {result['alert_draw_count']}")
        alerts = pd.DataFrame(result["alerts"])
        if alerts.empty:
            st.info("No alert draws found in the selected window.")
        else:
            st.caption("Alert draws are sourced from the latest promoted weekly scoring output.")
            columns = [
                "draw_date",
                "draw_id",
                "risk_tier",
                "partnership_count",
                "flagged_member_count",
                "highAmountMemberCount",
                "maxWinAmount",
                "maxBetAmount",
                "ccs_ids",
                "flagged_member_ids",
                "response_details",
            ]
            st.dataframe(alerts[[column for column in columns if column in alerts.columns]], use_container_width=True)
            draw_options = alerts["draw_id"].astype(str).tolist()
            selected_draw = st.session_state.get("alerts_selected_draw")
            if selected_draw not in draw_options:
                st.session_state["alerts_selected_draw"] = draw_options[0]
            selected_draw = st.selectbox(
                "Alert member details",
                draw_options,
                key="alerts_selected_draw",
            )
            selected = alerts.loc[alerts["draw_id"].astype(str).eq(selected_draw)].iloc[0].to_dict()
            members = pd.DataFrame(selected.get("flaggedMembers") or selected.get("flagged_members") or [])
            if members.empty:
                st.info("No flagged member rows available for the selected draw.")
            else:
                st.dataframe(members, use_container_width=True)

with tab_retrain:
    st.subheader("Trigger retrain and rescore")
    st.caption(
        "Submits an on-demand rolling full cycle. Reviewed feedback is read from gk_users.confirmed_fraud during training."
    )
    active_run_id = st.session_state.get("full_cycle_flow_run_id")
    active_state = st.session_state.get("full_cycle_state_type")
    trigger_disabled = bool(active_run_id and str(active_state or "").upper() in ACTIVE_PREFECT_STATES)

    if st.button(
        "Trigger full-cycle retrain and rescore",
        type="primary",
        disabled=trigger_disabled,
    ):
        try:
            run = trigger_full_cycle_flow(
                config_path="configs/config.yaml",
                candidate_config_path="configs/candidate_extraction.yaml",
                ccs_config_path="configs/ccs_profit.yaml",
            )
            st.session_state["full_cycle_flow_run_id"] = run.flow_run_id
            st.session_state["full_cycle_state_type"] = run.state_type
            st.session_state["full_cycle_state_name"] = run.state_name
            st.session_state["full_cycle_ui_url"] = run.ui_url
            st.session_state["full_cycle_window_mode"] = "rolling"
            st.success(f"Triggered full-cycle flow run: {run.flow_run_id}")
        except Exception as exc:
            st.error(
                f"Could not trigger Prefect full-cycle deployment: {exc}. "
                "Confirm Prefect is running and deploy with: "
                "prefect deploy --prefect-file orchestration/prefect.yaml --all"
            )

    if st.session_state.get("full_cycle_flow_run_id"):
        st.subheader("Run status")
        flow_run_id = st.session_state["full_cycle_flow_run_id"]
        st.write(f"Flow run ID: `{flow_run_id}`")
        st.write(f"Requested full-cycle window mode: `{st.session_state.get('full_cycle_window_mode', 'rolling')}`")
        if st.session_state.get("full_cycle_ui_url"):
            st.link_button("Open Prefect flow run", st.session_state["full_cycle_ui_url"])
        try:
            status = get_flow_run_status(flow_run_id)
            st.session_state["full_cycle_state_type"] = status.state_type
            st.session_state["full_cycle_state_name"] = status.state_name
            st.metric("Prefect state", status.state_name or status.state_type or "unknown")
        except Exception as exc:
            st.warning(f"Could not refresh Prefect status: {exc}")

        if str(st.session_state.get("full_cycle_state_type") or "").upper() in ACTIVE_PREFECT_STATES:
            auto_refresh = st.checkbox("Auto-refresh status every 5 seconds", value=False)
            if auto_refresh:
                time.sleep(5)
                st.rerun()
        else:
            st.info(
                "Run is no longer active. Reloading API artifacts is safe, "
                "but serving changes only if the run promoted a model."
            )
            st.link_button("Open MLflow", MLFLOW_UI_URL)
            if st.button("Reload API Artifacts"):
                try:
                    result = post_json("/admin/reload", {})
                    st.success(
                        f"API reloaded. Current source_run_id: {result.get('current_run_id')} "
                        f"(previous: {result.get('previous_run_id')})"
                    )
                except Exception as exc:
                    st.error(f"API reload failed: {exc}")

        if st.button("Clear tracked run"):
            for key in [
                "full_cycle_flow_run_id",
                "full_cycle_state_type",
                "full_cycle_state_name",
                "full_cycle_ui_url",
                "full_cycle_window_mode",
            ]:
                st.session_state.pop(key, None)
            st.rerun()
