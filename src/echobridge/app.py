"""FastAPI/uvicorn application for the EchoBridge gateway."""

from __future__ import annotations

import asyncio
import hmac
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from . import __version__
from .asr import make_asr
from .config import Settings
from .gpu import get_gpu_scheduler
from .logging import get_logger
from .manager import SessionManager
from .tts import make_tts
from .vad import make_vad
from .voices import load_voice_registry

logger = get_logger("echobridge.app")

_PROBE_FILE = Path(__file__).parent / "static" / "probe.html"
_STARTED_AT = time.monotonic()


def _rss_mb() -> float:
    """Resident set size without adding a psutil dependency."""
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


async def _worker_health(
    settings: Settings,
    *,
    asr: object | None = None,
    tts: object | None = None,
) -> dict[str, dict[str, object]]:
    """Probe configured local workers for the readiness endpoint.

    The gateway deliberately starts even when a worker is still loading; the
    watchdog uses this endpoint to wait for (or restart) the full pipeline
    instead of treating a cold model as a dead gateway.  In-process backends
    are reported directly from their loaded state and CUDA allocator counters.
    """
    probes: dict[str, tuple[str, str]] = {}
    results: dict[str, dict[str, object]] = {}
    if settings.asr_backend == "inprocess":
        health = getattr(asr, "health", None)
        results["asr"] = (
            health()
            if callable(health)
            else {"ok": False, "backend": "inprocess", "error": "not constructed"}
        )
    elif settings.asr_backend == "qwen3":
        probes["asr"] = (settings.asr_url + "/healthz", "ok")
    elif settings.asr_backend == "vllm":
        # vLLM answers 200 with an empty body; the status code is the signal.
        probes["asr"] = (settings.asr_url + "/health", "status")

    if settings.tts_backend == "inprocess":
        health = getattr(tts, "health", None)
        results["tts"] = (
            health()
            if callable(health)
            else {"ok": False, "backend": "inprocess", "error": "not constructed"}
        )
    elif settings.tts_backend == "vllm":
        probes["tts"] = (settings.tts_url + "/health", "status")
    elif settings.tts_backend == "qwen3":
        probes["tts"] = (settings.tts_url + "/healthz", "ok")

    if not probes:
        return results
    async with httpx.AsyncClient(timeout=httpx.Timeout(1.5, connect=0.5)) as client:
        for name, (url, field) in probes.items():
            started = time.perf_counter()
            try:
                response = await client.get(url)
                latency_ms = (time.perf_counter() - started) * 1000.0
                healthy = response.status_code == 200
                detail: object = None
                if healthy:
                    try:
                        payload = response.json()
                        detail = payload.get(field)
                        healthy = bool(detail) if field == "ok" else detail in {
                            "healthy",
                            "ok",
                            True,
                        }
                    except ValueError:
                        detail = None
                results[name] = {
                    "ok": healthy,
                    "url": url,
                    "status_code": response.status_code,
                    "latency_ms": round(latency_ms, 1),
                    "detail": detail,
                }
            except Exception as exc:  # noqa: BLE001 - readiness must never raise
                results[name] = {
                    "ok": False,
                    "url": url,
                    "error": type(exc).__name__,
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
                }
    return results


async def _watch_tts_recovery(settings: Settings, tts: object, manager: SessionManager) -> None:
    """Re-warm the TTS worker after it restarts.

    The gateway's PCM caches survive a worker restart, and ``ensure_warm()``
    returns immediately once the voice is marked warm, so a restarted vLLM/TTS
    process would still pay 2-4 s on the first real request.  Probe the worker
    and force one synthesis on the down -> up edge.
    """
    if settings.tts_backend not in {"vllm", "qwen3"}:
        return
    if settings.tts_recovery_probe_s <= 0:
        return
    path = "/health" if settings.tts_backend == "vllm" else "/healthz"
    url = settings.tts_url.rstrip("/") + path
    healthy = True
    down_since: float | None = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(2.0, connect=1.0)) as client:
        while True:
            await asyncio.sleep(max(1.0, settings.tts_recovery_probe_s))
            try:
                response = await client.get(url)
                up = response.status_code == 200
            except Exception:
                up = False
            if not up:
                healthy = False
                if down_since is None:
                    down_since = time.monotonic()
                continue
            if healthy:
                continue
            active = manager.active
            if active is not None and active.busy:
                # Do not steal GPU time from an in-flight answer; retry next tick.
                continue
            try:
                rewarm = getattr(tts, "rewarm", None)
                started = time.perf_counter()
                if callable(rewarm):
                    await rewarm(settings.tts_voice_id)
                else:
                    await tts.ensure_warm(settings.tts_voice_id)
                logger.info(
                    "tts backend recovered after %.1f s; rewarmed voice=%s in %.0f ms",
                    time.monotonic() - (down_since or time.monotonic()),
                    settings.tts_voice_id,
                    (time.perf_counter() - started) * 1000.0,
                )
                healthy = True
                down_since = None
            except Exception as exc:
                logger.warning("tts recovery warmup failed: %s", exc)


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


def _auth_diagnostics(ws: WebSocket) -> str:
    """Describe auth-bearing headers without ever logging their values.

    The gateway sits behind Tailscale Serve in development.  When a 403 is
    reported, the first question is always "did the proxy drop Authorization?".
    Header names, presence and lengths answer that question while keeping the
    token, Origin and Tailscale identity out of the logs.
    """

    headers = ws.headers
    auth = headers.get("authorization", "")
    protocols = headers.get("sec-websocket-protocol", "")
    if auth.lower().startswith("bearer "):
        source = "bearer"
    elif any(
        item.strip().startswith("token.") for item in protocols.split(",")
    ):
        source = "subprotocol"
    else:
        source = "none"

    def present(name: str) -> str:
        return "yes" if headers.get(name) else "no"

    return (
        f"auth={source} auth_len={len(auth)} "
        f"subprotocol_len={len(protocols)} "
        f"origin={present('origin')} ua={present('user-agent')} "
        f"device={present('x-device-id')} "
        f"tailscale_user={present('tailscale-user-login')} "
        f"xff={present('x-forwarded-for')}"
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    manager = SessionManager()
    voice_registry = load_voice_registry(settings)
    shared_vad = make_vad(settings)
    shared_asr = (
        make_asr(settings)
        if settings.asr_backend in {"qwen3", "inprocess", "vllm"}
        else None
    )
    # Keep one TTS client for the whole gateway: its cache holds the fixed
    # greeting and the backchannel fillers warmed at startup.  Previously only
    # the HF qwen3 backend was shared; vLLM sessions each rebuilt an empty
    # cache, which made "prewarmed" greetings and backchannels ineffective.
    shared_tts = make_tts(settings, voice_registry)

    def _inprocess_loaded() -> bool:
        return bool(
            getattr(shared_asr, "loaded", False)
            or getattr(shared_tts, "loaded", False)
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if shared_asr is not None:
            try:
                await shared_asr.warmup()
            except Exception as exc:
                logger.warning("ASR warmup failed: %s", exc)
        if shared_tts is not None:
            try:
                await shared_tts.warmup()
            except Exception as exc:
                logger.warning("TTS warmup failed: %s", exc)
        tts_watch = None
        if shared_tts is not None:
            tts_watch = asyncio.create_task(
                _watch_tts_recovery(settings, shared_tts, manager),
                name="tts-recovery-watch",
            )
        try:
            yield
        finally:
            if tts_watch is not None:
                tts_watch.cancel()
                await asyncio.gather(tts_watch, return_exceptions=True)
            await manager.close_active("shutdown")
            if shared_asr is not None:
                await shared_asr.close()
            if shared_tts is not None:
                await shared_tts.close()

    app = FastAPI(title="EchoBridge", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.manager = manager
    app.state.voice_registry = voice_registry

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        active = manager.active
        return JSONResponse(
            {
                "ok": True,
                "service": "echobridge-gateway",
                "version": __version__,
                "pid": os.getpid(),
                "uptime_s": round(time.monotonic() - _STARTED_AT, 1),
                "rss_mb": _rss_mb(),
                "vad": settings.vad_backend,
                "asr": settings.asr_backend,
                "tts": settings.tts_backend,
                "llm": settings.llm_backend,
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

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        workers = await _worker_health(settings, asr=shared_asr, tts=shared_tts)
        ok = all(bool(item.get("ok")) for item in workers.values())
        scheduler = get_gpu_scheduler()
        payload = {
            "ok": ok,
            "service": "echobridge-gateway",
            "version": __version__,
            "uptime_s": round(time.monotonic() - _STARTED_AT, 1),
            "workers": workers,
            # Additive observability for the single-process/CUDA-context mode.
            # Empty memory until an in-process model actually exists.
            "gpu": scheduler.status(include_memory=_inprocess_loaded()),
        }
        return JSONResponse(payload, status_code=200 if ok else 503)

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        active = 0 if manager.active is None else 1
        lines = [
            "# HELP echobridge_up Gateway process is serving.",
            "# TYPE echobridge_up gauge",
            "echobridge_up 1",
            "# HELP echobridge_uptime_seconds Gateway process uptime.",
            "# TYPE echobridge_uptime_seconds gauge",
            f"echobridge_uptime_seconds {time.monotonic() - _STARTED_AT:.1f}",
            "# HELP echobridge_rss_mb Gateway resident set size.",
            "# TYPE echobridge_rss_mb gauge",
            f"echobridge_rss_mb {_rss_mb()}",
            "# HELP echobridge_active_session Active device session (0/1).",
            "# TYPE echobridge_active_session gauge",
            f"echobridge_active_session {active}",
            "# HELP echobridge_info Static build information.",
            "# TYPE echobridge_info gauge",
            f'echobridge_info{{version="{__version__}"}} 1',
        ]
        memory = (
            get_gpu_scheduler().memory() if _inprocess_loaded() else None
        )
        if memory is not None and memory.device:
            lines.extend(
                [
                    "# HELP echobridge_gpu_memory_allocated_bytes "
                    "CUDA caching-allocator bytes in use by this process.",
                    "# TYPE echobridge_gpu_memory_allocated_bytes gauge",
                    f"echobridge_gpu_memory_allocated_bytes "
                    f"{int(memory.allocated_mb * 1024 * 1024)}",
                    "# HELP echobridge_gpu_memory_reserved_bytes "
                    "CUDA caching-allocator bytes reserved by this process.",
                    "# TYPE echobridge_gpu_memory_reserved_bytes gauge",
                    f"echobridge_gpu_memory_reserved_bytes "
                    f"{int(memory.reserved_mb * 1024 * 1024)}",
                ]
            )
        return "\n".join(lines) + "\n"

    @app.get("/voices")
    async def voices(request: Request) -> JSONResponse:
        auth = request.headers.get("authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        if not token or not hmac.compare_digest(token, settings.token):
            logger.warning(
                "rejecting unauthorized voices request from %s", request.client
            )
            return JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        return JSONResponse(voice_registry.public())

    @app.get("/")
    async def index() -> JSONResponse:
        return JSONResponse(
            {
                "name": "EchoBridge",
                "version": __version__,
                "ws": "/echobridge/ws",
                "voices": "/voices",
                "health": "/healthz",
                "ready": "/readyz",
                "metrics": "/metrics",
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
            logger.warning(
                "rejecting unauthorized websocket from %s path=%s %s",
                ws.client,
                ws.url.path,
                _auth_diagnostics(ws),
            )
            await ws.close(code=4401, reason="unauthorized")
            return
        logger.info(
            "accepted websocket from %s path=%s %s",
            ws.client,
            ws.url.path,
            _auth_diagnostics(ws),
        )

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
        session_asr = shared_asr or make_asr(settings, asr_text)
        session = await manager.acquire(
            ws,
            device_id,
            settings=settings,
            mock_asr_text=asr_text,
            asr=session_asr,
            vad=shared_vad,
            tts=shared_tts,
            voice_registry=voice_registry,
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
