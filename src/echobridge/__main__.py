"""``python -m echobridge`` entry point."""

from __future__ import annotations

import uvicorn

from .app import create_app
from .config import Settings
from .logging import configure_logging


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.bind,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=False,
    )


if __name__ == "__main__":
    main()
