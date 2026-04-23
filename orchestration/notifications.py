"""Optional Slack webhook notifications for Prefect flows.

SLACK_WEBHOOK_URL must be set in .env to enable notifications.
If unset, warnings are logged and the flow continues normally.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def notify_failure(flow_name: str, error: str) -> None:
    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not set — skipping Slack notification")
        return

    payload = {
        "text": (
            f":x: *{flow_name}* failed at "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"```{error[:400]}```"
        )
    }
    try:
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                logger.warning("Slack notification returned HTTP %s", resp.status)
    except Exception as exc:
        logger.warning("Slack notification failed (non-fatal): %s", exc)


def notify_success(flow_name: str, summary: str) -> None:
    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return

    payload = {
        "text": (
            f":white_check_mark: *{flow_name}* succeeded at "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"{summary[:400]}"
        )
    }
    try:
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception as exc:
        logger.warning("Slack notification failed (non-fatal): %s", exc)
