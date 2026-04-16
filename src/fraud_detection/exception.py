from __future__ import annotations

import sys


class FraudDetectionException(Exception):
    def __init__(self, error_message, error_detail: sys = sys):
        super().__init__(str(error_message))
        _, _, tb = error_detail.exc_info()
        if tb is not None:
            self.lineno = tb.tb_lineno
            self.filename = tb.tb_frame.f_code.co_filename
        else:
            self.lineno = "unknown"
            self.filename = "unknown"
        self.error_message = str(error_message)

    def __str__(self) -> str:
        return f"Error in [{self.filename}] line [{self.lineno}]: {self.error_message}"
