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
    widen_candidate_window_for_label,
)
from fraud_detection.utils.mongo_predictions import read_recent_analyst_labels, upsert_analyst_label


load_dotenv()

API_BASE_URL = os.getenv("FRAUD_API_BASE_URL", "http://127.0.0.1:8000")
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE")
LIVE_PREDICTIONS_COLLECTION = os.getenv("LIVE_PREDICTIONS_COLLECTION", "live_predictions")
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


def current_model_version() -> str:
    try:
        return str(get_json("/model-info").get("model_version") or "partnership_v1")
    except Exception:
        return "partnership_v1"


@st.cache_data(show_spinner=False, ttl=20)
def load_recent_labels(limit: int = 25) -> pd.DataFrame:
    try:
        return pd.DataFrame(read_recent_analyst_labels(limit=limit))
    except Exception:
        return pd.DataFrame()


def clear_recent_label_cache() -> None:
    load_recent_labels.clear()


def append_label_audit_csv(row: dict) -> None:
    path = Path("ROULET CHEATING DATA.csv")
    audit_row = pd.DataFrame([row])
    audit_row.to_csv(path, mode="a", header=not path.exists(), index=False)


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
                selected_draw = st.selectbox("Alert member details", alerts["draw_id"].astype(str).tolist())
                selected = alerts.loc[alerts["draw_id"].astype(str).eq(selected_draw)].iloc[0].to_dict()
                members = pd.DataFrame(selected.get("flaggedMembers") or selected.get("flagged_members") or [])
                if not members.empty:
                    st.dataframe(members, use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

with tab_retrain:
    st.subheader("Add confirmed fraud feedback")
    st.caption("Labels are written to the Mongo feedback collection used by candidate-store training.")
    with st.form("confirmed_fraud_label_form"):
        label_date = st.date_input("DATE")
        label_draw_id = st.number_input("DRAW_ID", min_value=0, step=1, format="%d")
        label_member_id = st.text_input("MEMBER_ID", placeholder="GK00236424")
        label_ccs_id = st.text_input("CCS_ID", placeholder="CCS015695")
        mirror_csv = st.checkbox("Also append to legacy CSV audit mirror", value=False)
        submitted = st.form_submit_button("Save confirmed fraud label", type="primary")

    if submitted:
        normalized_member = label_member_id.strip().upper()
        normalized_ccs = label_ccs_id.strip().upper()
        if int(label_draw_id) <= 0 or not normalized_member or not normalized_ccs:
            st.error("DATE, DRAW_ID, MEMBER_ID, and CCS_ID are required.")
        else:
            try:
                label_dt = datetime.combine(label_date, datetime.min.time(), tzinfo=timezone.utc)
                label_id, created = upsert_analyst_label(
                    draw_id=int(label_draw_id),
                    member_id=normalized_member,
                    label="fraud",
                    model_version=current_model_version(),
                    draw_date=label_dt,
                    ccs_id=normalized_ccs,
                )
                if mirror_csv:
                    append_label_audit_csv(
                        {
                            "DATE": label_date.isoformat(),
                            "DRAW_ID": int(label_draw_id),
                            "MEMBER_ID": normalized_member,
                            "CCS_ID": normalized_ccs,
                            "label": "fraud",
                            "mongo_label_id": str(label_id),
                        }
                    )
                st.session_state["latest_label_draw_date"] = label_date.isoformat()
                st.session_state["latest_label_draw_id"] = int(label_draw_id)
                st.session_state["latest_label_member_id"] = normalized_member
                clear_recent_label_cache()
                st.success(
                    "Created new confirmed-fraud label."
                    if created
                    else "Updated existing confirmed-fraud label."
                )
            except Exception as exc:
                st.error(f"Label write failed: {exc}")

    st.subheader("Recent labels")
    recent_labels = load_recent_labels(limit=25)
    if recent_labels.empty:
        st.info("No feedback labels found, or Mongo is unavailable.")
    else:
        columns = ["decided_at", "draw_date", "draw_id", "member_id", "ccs_id", "label"]
        visible_columns = [column for column in columns if column in recent_labels.columns]
        st.dataframe(recent_labels[visible_columns], use_container_width=True)

    st.subheader("Trigger retrain and rescore")
    st.caption(
        "Submits an on-demand full cycle that refreshes candidate data for the needed window, "
        "retrains, evaluates, promotes if gates pass, and batch-scores the promoted model."
    )
    has_session_label = "latest_label_draw_date" in st.session_state
    manual_confirm = st.checkbox("Allow manual trigger without a label written in this session", value=False)
    active_run_id = st.session_state.get("full_cycle_flow_run_id")
    active_state = st.session_state.get("full_cycle_state_type")
    trigger_disabled = bool(active_run_id and str(active_state or "").upper() in ACTIVE_PREFECT_STATES)

    if not has_session_label and not manual_confirm:
        st.info(
            "Save a confirmed-fraud label in this session before triggering the full cycle, "
            "or enable manual trigger."
        )

    if st.button(
        "Trigger full-cycle retrain and rescore",
        type="primary",
        disabled=trigger_disabled or (not has_session_label and not manual_confirm),
    ):
        try:
            draw_date_for_window = st.session_state.get("latest_label_draw_date") or label_date.isoformat()
            start_window, end_window = widen_candidate_window_for_label(
                labeled_draw_date=draw_date_for_window,
                config_path="configs/config.yaml",
            )
            run = trigger_full_cycle_flow(
                start_date=start_window,
                end_date=end_window,
                config_path="configs/config.yaml",
                candidate_config_path="configs/candidate_extraction.yaml",
                ccs_config_path="configs/ccs_profit.yaml",
            )
            st.session_state["full_cycle_flow_run_id"] = run.flow_run_id
            st.session_state["full_cycle_state_type"] = run.state_type
            st.session_state["full_cycle_state_name"] = run.state_name
            st.session_state["full_cycle_ui_url"] = run.ui_url
            st.session_state["full_cycle_start_date"] = start_window.isoformat()
            st.session_state["full_cycle_end_date"] = end_window.isoformat()
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
        st.write(
            f"Requested full-cycle window: `{st.session_state.get('full_cycle_start_date')}` "
            f"to `{st.session_state.get('full_cycle_end_date')}`"
        )
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
            st.link_button("Open MLflow", os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
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
                "full_cycle_start_date",
                "full_cycle_end_date",
            ]:
                st.session_state.pop(key, None)
            st.rerun()
