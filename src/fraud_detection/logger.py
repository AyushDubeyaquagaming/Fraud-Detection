from __future__ import annotations

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/fraud_detection -> src -> repo root
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

_file_handler = TimedRotatingFileHandler(
    filename=str(LOG_DIR / "fraud_detection.log"),
    when="midnight",
    backupCount=10,
    encoding="utf-8",
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
_file_handler.suffix = "%Y%m%d"

_root_logger = logging.getLogger("fraud_detection")
if not _root_logger.handlers:
    _root_logger.setLevel(logging.DEBUG)
    _root_logger.addHandler(_console_handler)
    _root_logger.addHandler(_file_handler)
    _root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"fraud_detection.{name}" if not name.startswith("fraud_detection") else name)
