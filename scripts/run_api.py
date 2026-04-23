from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from fraud_detection.utils.common import read_yaml


def _load_serving_defaults(config_path: str) -> tuple[str, int]:
    config = read_yaml(Path(config_path))
    serving_config = config.get("serving", {})
    host = str(serving_config.get("host", "127.0.0.1"))
    port = int(serving_config.get("port", 8000))
    return host, port


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="configs/config.yaml")
    config_args, remaining = config_parser.parse_known_args()

    default_host, default_port = _load_serving_defaults(config_args.config)

    parser = argparse.ArgumentParser(description="Run fraud detection API", parents=[config_parser])
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(remaining)

    os.environ["FRAUD_DETECTION_CONFIG"] = args.config

    uvicorn.run(
        "fraud_detection.serving.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1,
    )


if __name__ == "__main__":
    main()
