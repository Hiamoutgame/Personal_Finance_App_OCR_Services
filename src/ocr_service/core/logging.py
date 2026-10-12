import logging
import sys
from pathlib import Path


def configure_logging(cli_name=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if cli_name:
        try:
            Path("log").mkdir(exist_ok=True)
            handlers.append(logging.FileHandler(Path("log") / f"{cli_name}.log", encoding="utf-8"))
        except OSError:
            logging.getLogger(__name__).warning("Could not open CLI log file")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers, force=True)
