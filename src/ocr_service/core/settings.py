import argparse
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    host: str = "0.0.0.0"
    port: int = 6969

    @property
    def models_dir(self):
        return self.project_root / "models"

    @classmethod
    def from_environment(cls, project_root):
        port = int(os.environ.get("PORT", "6969"))
        if not 1 <= port <= 65535:
            raise ValueError("PORT must be between 1 and 65535")
        return cls(Path(project_root).resolve(), os.environ.get("HOST", "0.0.0.0"), port)


def parse_server_args(settings, argv=None):
    parser = argparse.ArgumentParser(description="Run the OCR FastAPI service")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return Settings(settings.project_root, args.host, args.port)
