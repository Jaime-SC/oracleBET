import logging
import os

def setup_logging(level: str | int = None):
    level = level or os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    return logging.getLogger("betcomb")
