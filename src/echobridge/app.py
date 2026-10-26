"""FastAPI/uvicorn application for the Phase 1 gateway."""

from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from . import __version__
from .config import Settings
from .logging import get_logger
from .manager import SessionManager

logger = get_logger("echobridge.app")

_PROBE_FILE = Path(__file__).parent / "static" / "probe.html"


def _extract_token(ws: WebSocket) -> str:
    auth = ws.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    protocols = ws.headers.get("sec-websocket-protocol", "")
    for item in protocols.split(","):
        item = item.strip()
        if item.startswith("token."):
            return item[6:].strip()
    return ""


def _has_probe_protocol(ws: WebSocket) -> bool:
    protocols = ws.headers.get("sec-websocket-protocol", "")
    return any(item.strip() == "echobridge" for item in protocols.split(","))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    manager = SessionManager()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await manager.close_active("shutdown")

    app = FastAPI(title="EchoBridge", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.manager = manager

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        active = manager.active
        return JSONResponse(
            {
                "ok": True,
                "version": __version__,
                "backend": "mock",
                "active": None
                if active is None
                else {
                    "session_id": active.session_id,
                    "device_id": active.device_id,
                    "created": active.created,
                    "closed": active.closed,
                },
            }
        )

    @app.get("/")
    async def index() -> JSONResponse:
        return JSONResponse(
            {
                "name": "EchoBridge",
                "version": __version__,
                "ws": "/echobridge/ws",
                "health": "/healthz",
                "probe": "/probe" if settings.enable_probe else None,
            }
        )

    @app.get("/probe")
    async def probe() -> HTMLResponse:
        if not settings.enable_probe:
            return HTMLResponse("probe disabled", status_code=404)
        if not _PROBE_FILE.exists():
            return HTMLResponse("probe asset missing", status_code=500)
        return HTMLResponse(_PROBE_FILE.read_text(encoding="utf-8"))

    @app.websocket("/echobridge/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        token = _extract_token(ws)
        if not token or not hmac.compare_digest(token, settings.token):
            logger.warning("rejecting unauthorized websocket from %s", ws.client)
            await ws.close(code=4401, reason="unauthorized")
            return

        device_id = (
            ws.headers.get("x-device-id")
            or ws.query_params.get("device_id")
            or "unknown"
        )
        asr_text = (
            ws.headers.get("x-mock-asr-text")
            or ws.query_params.get("asr_text")
            or settings.mock_asr_text
        )
        subprotocol = "echobridge" if _has_probe_protocol(ws) else None
        await ws.accept(subprotocol=subprotocol)
        session = await manager.acquire(
            ws,
            device_id,
            settings=settings,
            mock_asr_text=asr_text,
        )
        if session is None:
            await ws.close(code=1013, reason="busy")
            return
        try:
            await session.run()
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("websocket handler failed")
        finally:
            await manager.release(session)

    return app
