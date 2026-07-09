import logging

import uvicorn

logger = logging.getLogger("gb_ui_backend")


def main():
    config = uvicorn.Config(
        "gb_ui_backend.main:app",
        host="127.0.0.1",
        port=8090,
        reload=False,
    )
    server = uvicorn.Server(config)

    # Fires after uvicorn's own "Application startup complete." / "Uvicorn running
    # on ..." lines, so this ends up as the last, most visible line at boot —
    # reinforcing where the frontend actually lives (gbserver serves it, not this
    # sidecar, which only handles /api/analytics/*).
    original_startup = server.startup

    async def _startup_with_log(*args, **kwargs):
        await original_startup(*args, **kwargs)
        from gb_ui_backend.config import get_config

        # Bold just the URL, matching Uvicorn's own "Uvicorn running on <bold-url>"
        # startup banner so this line carries the same visual weight.
        logger.info(
            "Frontend + API available at \x1b[1m%s\x1b[0m", get_config().gbserver_url
        )

    server.startup = _startup_with_log  # type: ignore[assignment]

    server.run()


if __name__ == "__main__":
    main()
