"""One EchoBridge device session.

Concurrency model:
  * one receive loop handles upstream JSON;
  * one response task drives mock LLM/TTS/tool flow;
  * one pacer task emits downstream audio;
  * every downstream send goes through one lock and a generation check.

That is intentionally close to the shape the real ASR/TTS/LLM adapters will use
in later phases.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from typing import Any, Protocol

from . import protocol
from .asr import ASRBackend, MockASR
from .config import Settings
from .llm import MockLLM, MockPlan
from .logging import get_logger
from .pacer import AudioPacer
from .text import split_for_tts
from .tools import available_tool_names, convert_tools
from .tts import MockTTS
from .vad import VADBackend, VADEvent, make_vad

logger = get_logger("echobridge.session")


class WebSocketLike(Protocol):
    async def receive_text(self) -> str: ...

    async def send_text(self, data: str) -> None: ...

    async def close(self, code: int = 1000, reason: str | None = None) -> None: ...


class Session:
    def __init__(
        self,
        ws: WebSocketLike,
        settings: Settings,
        *,
        session_id: str | None = None,
        device_id: str = "unknown",
        mock_asr_text: str | None = None,
        asr: ASRBackend | None = None,
        vad: VADBackend | None = None,
    ) -> None:
        self._ws = ws
        self._settings = settings
        self._session_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        self._device_id = device_id
        self._mock_asr_text = (mock_asr_text or settings.mock_asr_text).strip()

        self._send_lock = asyncio.Lock()
        self._closed = False
        self._created = False
        self._closing = False

        self._model = "unknown"
        self._instructions = ""
        self._voice = ""
        self._tools_volc: list[dict[str, Any]] = []
        self._tools_openai: list[dict[str, Any]] = []
        self._tool_names: set[str] = set()

        self._event_seq = 0
        self._generation = 0
        self._active_ctx: protocol.ResponseContext | None = None
        self._response_task: asyncio.Task[None] | None = None
        self._pacer: AudioPacer | None = None
        self._response_active = False
        self._pending_tools: dict[str, asyncio.Future[str]] = {}
        self._history: list[dict[str, str]] = []

        self._speech_buffers: dict[str, str] = {}
        self._spec_task: asyncio.Task | None = None
        self._spec_item = ""
        self._llm = MockLLM(settings.llm_char_delay_ms)
        self._tts = MockTTS(settings)
        self._asr = asr or MockASR(self._mock_asr_text)
        self._vad = vad or make_vad(settings)

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def created(self) -> bool:
        return self._created

    async def run(self) -> None:
        try:
            first = await asyncio.wait_for(self._ws.receive_text(), timeout=10.0)
            event = protocol.parse_event(first)
            await self._handle_event(event)
            if not self._created and not self._closed:
                await self._send(
                    protocol.error("first event must be session.create", "protocol"),
                    force=True,
                )
                await self.close("protocol")
                return
            while not self._closed:
                raw = await self._ws.receive_text()
                event = protocol.parse_event(raw)
                await self._handle_event(event)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            logger.warning("session %s timed out waiting for session.create", self._session_id)
            await self.close("timeout")
        except Exception as exc:  # includes WebSocketDisconnect
            if not self._closed:
                logger.info("session %s receive loop ended: %s", self._session_id, exc)
        finally:
            await self._shutdown()

    async def close(self, reason: str = "client") -> None:
        if self._closing or self._closed:
            return
        self._closing = True
        self._discard_speculative_asr()
        await self._cancel_active(send_canceled=False, reason=f"close:{reason}")
        self._vad.reset()
        if not self._closed:
            await self._send(protocol.session_closed(reason), force=True)
        self._closed = True
        try:
            await self._ws.close(code=1000, reason="bye")
        except Exception:
            pass

    async def close_from_manager(self, reason: str) -> None:
        await self.close(reason)

    # ------------------------------------------------------------------
    # Upstream dispatch
    # ------------------------------------------------------------------

    async def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if not isinstance(event_type, str):
            return
        if event_type == protocol.SESSION_CREATE:
            await self._handle_session_create(event)
        elif event_type == protocol.AUDIO_APPEND:
            await self._handle_audio(event)
        elif event_type in (protocol.SPEECH_APPEND, protocol.SPEECH_COMMIT):
            await self._handle_speech(event)
        elif event_type == protocol.ITEM_CREATE:
            await self._handle_tool_results(event)
        elif event_type == protocol.RESPONSE_CANCEL:
            await self._cancel_active(send_canceled=True, reason="client")
        elif event_type == protocol.SESSION_CLOSE:
            await self.close("client")
        else:
            logger.debug("unhandled upstream event %s", event_type)

    async def _handle_session_create(self, event: dict[str, Any]) -> None:
        if self._created:
            logger.debug("duplicate session.create ignored")
            return
        session = event.get("session")
        if not isinstance(session, dict):
            await self._send(protocol.error("session.create missing session", "protocol"))
            await self.close("protocol")
            return
        self._model = str(session.get("model") or "unknown")
        self._instructions = str(session.get("instructions") or "")
        audio = session.get("audio")
        if isinstance(audio, dict):
            output = audio.get("output")
            if isinstance(output, dict):
                self._voice = str(output.get("voice") or "")
        tools = session.get("tools")
        self._tools_volc = tools if isinstance(tools, list) else []
        self._tools_openai = convert_tools(self._tools_volc)
        self._tool_names = available_tool_names(self._tools_openai)
        self._created = True
        self._vad.reset()
        await self._send(
            protocol.session_created(self._session_id, self._model, "echobridge-mock")
        )
        await self._send(
            protocol.session_updated(self._session_id, self._model, "echobridge-mock")
        )
        logger.info(
            "session created id=%s device=%s model=%s voice=%s tools=%d filtered=%d",
            self._session_id,
            self._device_id,
            self._model,
            self._voice or "-",
            len(self._tools_openai),
            len(self._tools_volc) - len(self._tools_openai),
        )

    async def _handle_audio(self, event: dict[str, Any]) -> None:
        if not self._created or self._closed:
            return
        if self._response_active:
            # Half duplex: the device sends zero frames while TTS plays.  Do not
            # run turn detection on those frames.
            return
        encoded = protocol.extract_audio(event)
        if not encoded:
            return
        try:
            pcm = base64.b64decode(encoded)
        except Exception:
            logger.debug("bad base64 audio frame ignored")
            return
        if not pcm:
            return
        for vad_event in self._vad.push(pcm):
            await self._handle_vad_event(vad_event)

    async def _handle_vad_event(self, event: VADEvent) -> None:
        if event.kind == "started":
            await self._send(protocol.asr_started(event.item_id))
        elif event.kind == "delta":
            await self._send(protocol.asr_delta(event.item_id, event.text))
        elif event.kind == "early":
            self._start_speculative_asr(event)
        elif event.kind == "resume":
            self._discard_speculative_asr()
        elif event.kind == "final":
            if not event.pcm or event.speech_ms < self._settings.vad_min_speech_ms:
                self._discard_speculative_asr()
                await self._send(protocol.asr_failed(event.item_id, "empty speech"))
                return
            asr_started = time.perf_counter()
            result = await self._take_speculative_asr(event)
            speculative = result is not None
            if result is None:
                try:
                    result = await self._asr.transcribe(event.pcm, 16000)
                except Exception:
                    logger.exception(
                        "asr failed item=%s bytes=%d", event.item_id, len(event.pcm)
                    )
                    await self._send(protocol.asr_failed(event.item_id, "asr_error"))
                    return
            asr_wall_ms = (time.perf_counter() - asr_started) * 1000.0
            text = (result.text or "").strip()
            if not text:
                await self._send(protocol.asr_failed(event.item_id, "empty speech"))
                return
            logger.info(
                "asr final item=%s chars=%d speech_ms=%d worker_ms=%.1f wall_ms=%.1f spec=%s",
                event.item_id,
                len(text),
                event.speech_ms,
                result.latency_ms,
                asr_wall_ms,
                "1" if speculative else "0",
            )
            await self._send(
                protocol.asr_completed(event.item_id, text, result.language or "zh")
            )
            await self._start_turn(text)

    async def _handle_speech(self, event: dict[str, Any]) -> None:
        speech_id = str(event.get("speech_id") or "default")
        text = protocol.extract_text(event)
        if event.get("type") == protocol.SPEECH_APPEND:
            self._speech_buffers[speech_id] = self._speech_buffers.get(speech_id, "") + text
            return
        merged = (self._speech_buffers.pop(speech_id, "") + text).strip()
        if not merged:
            return
        if self._response_active:
            await self._cancel_active(send_canceled=True, reason="greeting")
        await self._start_greeting(merged)

    def _start_speculative_asr(self, event: VADEvent) -> None:
        self._discard_speculative_asr()
        if not event.pcm:
            return
        self._spec_item = event.item_id

        async def run() -> Any:
            return await self._asr.transcribe(event.pcm, 16000)

        task = asyncio.create_task(run(), name=f"spec-asr-{event.item_id}")
        self._spec_task = task
        task.add_done_callback(self._consume_task_exception)

    def _discard_speculative_asr(self) -> None:
        task = self._spec_task
        self._spec_task = None
        self._spec_item = ""
        if task is not None and not task.done():
            task.cancel()

    async def _take_speculative_asr(self, event: VADEvent):
        task = self._spec_task
        if task is None or self._spec_item != event.item_id:
            return None
        self._spec_task = None
        self._spec_item = ""
        try:
            result = await asyncio.wait_for(task, timeout=1.5)
        except Exception:
            if not task.done():
                task.cancel()
            return None
        if result is None or not (result.text or "").strip():
            return None
        return result

    @staticmethod
    def _consume_task_exception(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        try:
            task.exception()
        except BaseException:
            pass

    async def _handle_tool_results(self, event: dict[str, Any]) -> None:
        for result in protocol.extract_tool_results(event):
            future = self._pending_tools.get(result.call_id)
            if future is not None and not future.done():
                future.set_result(result.text)

    # ------------------------------------------------------------------
    # Response lifecycle
    # ------------------------------------------------------------------

    async def _start_turn(self, transcript: str) -> None:
        if self._response_active:
            await self._cancel_active(send_canceled=True, reason="new_turn")
        ctx = self._begin_response()
        self._history.append({"role": "user", "content": transcript})
        self._trim_history()
        self._response_task = asyncio.create_task(
            self._run_turn(ctx, transcript), name=f"turn-{ctx.response_id}"
        )

    async def _start_greeting(self, text: str) -> None:
        ctx = self._begin_response()
        self._response_task = asyncio.create_task(
            self._run_greeting(ctx, text), name=f"greeting-{ctx.response_id}"
        )

    def _begin_response(self) -> protocol.ResponseContext:
        self._generation += 1
        ctx = protocol.ResponseContext(
            response_id=f"resp_{self._generation}", generation=self._generation
        )
        self._active_ctx = ctx
        self._response_active = True
        self._vad.reset()
        return ctx

    async def _run_turn(self, ctx: protocol.ResponseContext, transcript: str) -> None:
        try:
            plan = self._llm.plan(transcript, self._tool_names)
            if plan.tool_calls:
                ok = await self._run_tools(ctx, plan.tool_calls)
                if not ok:
                    return
            text = (plan.text_before + plan.text_after).strip()
            if text:
                await self._speak(ctx, text, emit_text=True)
            else:
                await self._send(protocol.response_done(ctx, "empty"), ctx=ctx)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("turn failed response=%s", ctx.response_id)
            if not self._closed and ctx.generation == self._generation:
                await self._send(protocol.response_done(ctx, "failed"), ctx=ctx)
        finally:
            self._finish_response(ctx)

    async def _run_greeting(self, ctx: protocol.ResponseContext, text: str) -> None:
        try:
            # Greeting is TTS-only: no output_text events.  The app already
            # displays and logs the greeting locally.
            await self._speak(ctx, text, emit_text=False)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("greeting failed response=%s", ctx.response_id)
            if not self._closed and ctx.generation == self._generation:
                await self._send(protocol.response_done(ctx, "failed"), ctx=ctx)
        finally:
            self._finish_response(ctx)

    async def _run_tools(
        self, ctx: protocol.ResponseContext, calls: list[protocol.ToolCall]
    ) -> bool:
        usable = [call for call in calls if call.name in self._tool_names]
        dropped = [call.name for call in calls if call.name not in self._tool_names]
        if dropped:
            logger.info("dropping unavailable/mock tools: %s", ",".join(sorted(set(dropped))))
        if not usable:
            return True
        self._pending_tools = {
            call.call_id: asyncio.get_running_loop().create_future() for call in usable
        }
        ok = await self._send(protocol.tool_call_done(ctx, usable), ctx=ctx)
        if not ok:
            return False
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*self._pending_tools.values()),
                timeout=self._settings.tool_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("tool timeout response=%s calls=%d", ctx.response_id, len(usable))
            if ctx.generation == self._generation and not self._closed:
                self._generation += 1
                await self._send(protocol.response_canceled(ctx), force=True)
            return False
        finally:
            self._pending_tools.clear()
        for call, result in zip(usable, results):
            logger.info(
                "tool result call=%s name=%s chars=%d",
                call.call_id,
                call.name,
                len(result or ""),
            )
            self._history.append({"role": "tool", "name": call.name, "content": result or "ok"})
        self._trim_history()
        return ctx.generation == self._generation

    async def _speak(
        self, ctx: protocol.ResponseContext, text: str, *, emit_text: bool
    ) -> None:
        pacer = AudioPacer(self._send_response_event, ctx)
        self._pacer = pacer
        pacer.start()

        if emit_text:
            buffer = ""
            first_chunk = True
            async for delta in self._llm.stream_text(text):
                if not await self._send(protocol.text_delta(ctx, delta), ctx=ctx):
                    return
                buffer += delta
                limit = 15 if first_chunk else 18
                if self._should_flush(buffer, limit):
                    await self._synthesize_into(pacer, buffer, ctx)
                    buffer = ""
                    first_chunk = False
            if buffer.strip():
                await self._synthesize_into(pacer, buffer, ctx)
        else:
            await self._synthesize_into(pacer, text, ctx)

        await pacer.finish()
        if emit_text:
            if not await self._send(protocol.text_done(ctx, text), ctx=ctx):
                return
        await self._send(protocol.response_done(ctx, "completed"), ctx=ctx)

    async def _synthesize_into(
        self, pacer: AudioPacer, text: str, ctx: protocol.ResponseContext
    ) -> None:
        for chunk in split_for_tts(text) or [text]:
            async for pcm in self._tts.synthesize(chunk, self._voice, ctx.generation):
                await pacer.feed(pcm)

    @staticmethod
    def _should_flush(buffer: str, limit: int) -> bool:
        if not buffer:
            return False
        if any(ch in "。！？!?；;\n" for ch in buffer):
            return True
        return len(buffer) >= limit

    async def _cancel_active(self, *, send_canceled: bool, reason: str) -> None:
        self._discard_speculative_asr()
        old = self._active_ctx
        if old is None and not self._response_active:
            return
        self._generation += 1
        self._response_active = False
        pacer = self._pacer
        task = self._response_task
        if pacer is not None:
            pacer.cancel()
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        wait_for: list[asyncio.Task[Any]] = []
        if task is not None and task is not asyncio.current_task():
            wait_for.append(task)
        if pacer is not None and pacer.task is not None:
            wait_for.append(pacer.task)
        if wait_for:
            await asyncio.gather(*wait_for, return_exceptions=True)
        self._active_ctx = None
        self._response_task = None
        self._pacer = None
        self._pending_tools.clear()
        self._vad.reset()
        logger.info(
            "response canceled response=%s reason=%s",
            old.response_id if old else "-",
            reason,
        )
        if send_canceled and old is not None and not self._closed:
            await self._send(protocol.response_canceled(old), force=True)

    def _finish_response(self, ctx: protocol.ResponseContext) -> None:
        if self._active_ctx is not ctx:
            return
        if ctx.generation == self._generation and not ctx.done:
            ctx.done = True
        pacer = self._pacer
        if pacer is not None:
            pacer.cancel()
        self._response_active = False
        self._active_ctx = None
        self._response_task = None
        self._pacer = None
        self._pending_tools.clear()
        self._vad.reset()

    # ------------------------------------------------------------------
    # Downstream send
    # ------------------------------------------------------------------

    async def _send_response_event(
        self, event: dict[str, Any], ctx: protocol.ResponseContext
    ) -> bool:
        return await self._send(event, ctx=ctx)

    async def _send(
        self,
        event: dict[str, Any],
        *,
        ctx: protocol.ResponseContext | None = None,
        force: bool = False,
    ) -> bool:
        if self._closed and not force:
            return False
        async with self._send_lock:
            if self._closed and not force:
                return False
            if ctx is not None and ctx.generation != self._generation:
                return False
            payload = dict(event)
            if "event_id" not in payload:
                self._event_seq += 1
                payload["event_id"] = f"event_{self._event_seq}"
            try:
                await self._ws.send_text(protocol.dumps(payload))
                return True
            except Exception as exc:
                logger.info("send failed session=%s: %s", self._session_id, exc)
                self._closed = True
                return False

    async def _shutdown(self) -> None:
        self._discard_speculative_asr()
        self._vad.reset()
        if not self._closed:
            await self._cancel_active(send_canceled=False, reason="shutdown")
            self._closed = True

    def _trim_history(self) -> None:
        # Phase 1 mock does not use history, but keep the shape bounded for the
        # DeepSeek adapter.
        if len(self._history) > 40:
            self._history = self._history[-40:]
