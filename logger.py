
import os
import colorlog
import logging

LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO").upper()
NUMERIC_LEVEL = getattr(logging, LOGGING_LEVEL, logging.INFO)

print(f"Logging level set to {LOGGING_LEVEL} ({NUMERIC_LEVEL})")

logging.basicConfig(
    level=NUMERIC_LEVEL,              # minimum level to handle
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

def setup_logging():
    """Configure root logger with colorlog."""
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        log_colors={
            "DEBUG":    "cyan",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "bold_red",
        },
    ))

    root = logging.getLogger()
    root.setLevel(NUMERIC_LEVEL)

    # Avoid adding duplicate handlers if setup_logging() is called twice
    root.handlers.clear()
    root.addHandler(handler)