from pathlib import Path

from ocr_service.application import create_app
from ocr_service.core.logging import configure_logging
from ocr_service.core.settings import Settings, parse_server_args


PROJECT_ROOT = Path(__file__).resolve().parent
app = create_app(Settings.from_environment(PROJECT_ROOT))


def main(argv=None):
    import logging
    import uvicorn

    settings = parse_server_args(Settings.from_environment(PROJECT_ROOT), argv)
    configure_logging()
    host = "127.0.0.1" if settings.host in {"0.0.0.0", "::"} else settings.host
    logging.getLogger(__name__).info("Swagger UI: http://%s:%s/docs", host, settings.port)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=1)


if __name__ == "__main__":
    main()
