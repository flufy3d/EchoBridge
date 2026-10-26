"""Single-device session manager.

Different device IDs are rejected.  A rapid reconnect from the same device
supersedes the old socket after it has been closed, which prevents a reconnect
storm from triggering the app's Doubao fallback.
"""

from __future__ import annotations

import asyncio

from .logging import get_logger
from .session import Session, WebSocketLike

logger = get_logger("echobridge.manager")


class SessionManager:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._active: Session | None = None

    @property
    def active(self) -> Session | None:
        return self._active

    async def acquire(
        self, ws: WebSocketLike, device_id: str, **session_kwargs
    ) -> Session | None:
        async with self._lock:
            if self._active is not None:
                if self._active.device_id != device_id:
                    logger.warning(
                        "rejecting second device %s (active=%s)",
                        device_id,
                        self._active.device_id,
                    )
                    return None
                logger.info("device %s reconnected; superseding old session", device_id)
                old = self._active
                self._active = None
                await old.close_from_manager("superseded")
            session = Session(ws, device_id=device_id, **session_kwargs)
            self._active = session
            return session

    async def release(self, session: Session) -> None:
        async with self._lock:
            if self._active is session:
                self._active = None

    async def close_active(self, reason: str = "shutdown") -> None:
        async with self._lock:
            active = self._active
            self._active = None
        if active is not None:
            await active.close_from_manager(reason)
