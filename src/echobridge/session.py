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
import random
import time
import uuid
from typing import Any, Protocol

from . import protocol
from .asr import ASRBackend, MockASR, is_hotword_echo, parse_hotwords
from .config import Settings, build_asr_context
from .llm import MockLLM
from .llm_deepseek import DeepSeekLLM, build_system_prompt
from .logging import get_logger
from .pacer import AudioPacer
from .text import is_stop_command, split_for_tts
from .tools import available_tool_names, convert_tools
from .tts import TTSBackend
from .vad import VADBackend, VADEvent, make_vad
from .voices import VoiceRegistry, load_voice_registry

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
        tts: TTSBackend | None = None,
        voice_registry: VoiceRegistry | None = None,
    ) -> None:
        self._ws = ws
        self._settings = settings
        self._session_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        self._device_id = device_id
        self._mock_asr_text = (mock_asr_text or settings.mock_asr_text).strip()
        self._voice_registry = voice_registry or load_voice_registry(settings)

        self._send_lock = asyncio.Lock()
        self._closed = False
        self._created = False
        self._closing = False

        self._model = "unknown"
        self._instructions = ""
        default_voice = self._voice_registry.resolve(None)[0]
        self._voice = default_voice.id
        self._voice_name = default_voice.name
        self._voice_fallback = False
        self._voice_warm_task: asyncio.Task[None] | None = None
        self._tools_volc: list[dict[str, Any]] = []
        self._tools_openai: list[dict[str, Any]] = []
        self._tool_names: set[str] = set()
        # None = use the gateway-wide ECHOBRIDGE_ASR_HOTWORDS; a tuple
        # (including an empty one) comes from this session's session.create.
        self._session_hotwords: tuple[str, ...] | None = None
        self._session_asr_context: str | None = None

        self._event_seq = 0
        self._generation = 0
        self._active_ctx: protocol.ResponseContext | None = None
        self._response_task: asyncio.Task[None] | None = None
        self._pacer: AudioPacer | None = None
        self._response_active = False
        self._pending_tools: dict[str, asyncio.Future[str]] = {}
        self._history: list[dict[str, Any]] = []

        self._speech_buffers: dict[str, str] = {}
        self._spec_task: asyncio.Task | None = None
        self._spec_item = ""
        self._backchannel_task: asyncio.Task[None] | None = None
        self._last_backchannel = ""
        self._backchannel_sent_this_turn = False
        self._turn_started_at: float | None = None
        self._first_audio_generation = 0
        self._backchannel_sent_generation = 0
        self._llm = MockLLM(settings.llm_char_delay_ms)
        self._deepseek = DeepSeekLLM(settings) if settings.llm_backend == "deepseek" else None
        from .tts import make_tts

        self._tts = tts or make_tts(settings, self._voice_registry)
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
        self._cancel_voice_warm()
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
        raw_voice: str | None = None
        audio = session.get("audio")
        if isinstance(audio, dict):
            output = audio.get("output")
            if isinstance(output, dict) and output.get("voice") is not None:
                value = output.get("voice")
                raw_voice = value if isinstance(value, str) else str(value)
        spec, fallback = self._voice_registry.resolve(raw_voice)
        self._voice = spec.id
        self._voice_name = spec.name
        self._voice_fallback = fallback
        tools = session.get("tools")
        self._tools_volc = tools if isinstance(tools, list) else []
        self._tools_openai = convert_tools(self._tools_volc)
        self._tool_names = available_tool_names(self._tools_openai)
        self._apply_asr_extension(event, session)
        self._created = True
        self._vad.reset()
        await self._send(
            protocol.session_created(
                self._session_id,
                self._model,
                "echobridge-mock",
                voice=self._voice,
                voice_name=self._voice_name,
                voice_fallback=self._voice_fallback,
            )
        )
        await self._send(
            protocol.session_updated(
                self._session_id,
                self._model,
                "echobridge-mock",
                voice=self._voice,
                voice_name=self._voice_name,
                voice_fallback=self._voice_fallback,
            )
        )
        self._schedule_voice_warm()
        logger.info(
            "session created id=%s device=%s model=%s voice=%s voice_name=%s "
            "fallback=%s tools=%d filtered=%d",
            self._session_id,
            self._device_id,
            self._model,
            self._voice,
            self._voice_name,
            "1" if self._voice_fallback else "0",
            len(self._tools_openai),
            len(self._tools_volc) - len(self._tools_openai),
        )

    def _apply_asr_extension(
        self, event: dict[str, Any], session: dict[str, Any]
    ) -> None:
        """Read per-session ASR hotwords from session.create.extension.

        The device may put ``extension`` at the top level or inside
        ``session``.  ``hotwords`` accepts a JSON array or a delimited string;
        an explicit empty value disables the gateway-wide default for this
        session.  Doubao's ``boosting_table_id`` is ignored here.
        """
        for container in (event, session):
            extension = container.get("extension")
            if not isinstance(extension, dict):
                continue
            asr = extension.get("asr")
            if not isinstance(asr, dict):
                continue
            extra = asr.get("extra")
            if not isinstance(extra, dict):
                continue
            if "hotwords" in extra:
                hotwords = parse_hotwords(extra.get("hotwords"))
                if hotwords is None:
                    return
                self._session_hotwords = hotwords
                self._session_asr_context = build_asr_context(hotwords)
                return
            if "context" in extra:
                context = extra.get("context")
                if isinstance(context, str):
                    self._session_hotwords = ()
                    self._session_asr_context = context.strip()[:400]
                return

    async def _transcribe_pcm(self, pcm: bytes):
        if self._session_asr_context is None:
            return await self._asr.transcribe(pcm, 16000)
        return await self._asr.transcribe(
            pcm, 16000, context=self._session_asr_context
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
                    result = await self._transcribe_pcm(event.pcm)
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
            hotwords = (
                self._settings.asr_hotwords
                if self._session_hotwords is None
                else self._session_hotwords
            )
            if is_hotword_echo(text, hotwords):
                logger.warning(
                    "discarding ASR hotword echo item=%s chars=%d",
                    event.item_id,
                    len(text),
                )
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
            if is_stop_command(text):
                # A barge-in "停一下/不用了" must stop the old answer without
                # turning the stop phrase into a brand-new LLM turn.
                logger.info(
                    "stop command item=%s chars=%d", event.item_id, len(text)
                )
                ctx = self._begin_response()
                self._turn_started_at = None
                await self._send(protocol.response_done(ctx, "stopped"), ctx=ctx)
                self._finish_response(ctx)
                return
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
            return await self._transcribe_pcm(event.pcm)

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

    def _schedule_voice_warm(self) -> None:
        ensure_warm = getattr(self._tts, "ensure_warm", None)
        if ensure_warm is None:
            return
        self._cancel_voice_warm()
        self._voice_warm_task = asyncio.create_task(
            ensure_warm(self._voice), name=f"tts-warm-{self._voice}"
        )
        self._voice_warm_task.add_done_callback(self._consume_task_exception)

    def _cancel_voice_warm(self) -> None:
        task = self._voice_warm_task
        self._voice_warm_task = None
        if task is not None and not task.done():
            task.cancel()

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
        self._turn_started_at = time.perf_counter()
        self._backchannel_sent_this_turn = False
        ctx = self._begin_response()
        self._history.append({"role": "user", "content": transcript})
        self._trim_history()
        self._schedule_backchannel(ctx)
        runner = self._run_deepseek_turn if self._deepseek is not None else self._run_turn
        self._response_task = asyncio.create_task(
            runner(ctx, transcript), name=f"turn-{ctx.response_id}"
        )

    async def _start_greeting(self, text: str) -> None:
        self._turn_started_at = None
        ctx = self._begin_response()
        self._response_task = asyncio.create_task(
            self._run_greeting(ctx, text), name=f"greeting-{ctx.response_id}"
        )

    def _schedule_backchannel(self, ctx: protocol.ResponseContext) -> None:
        self._cancel_backchannel()
        if not self._settings.backchannel_enabled:
            return
        if not self._settings.backchannel_phrases:
            return
        if self._backchannel_sent_this_turn:
            return
        self._backchannel_task = asyncio.create_task(
            self._run_backchannel(ctx), name=f"backchannel-{ctx.response_id}"
        )
        self._backchannel_task.add_done_callback(self._consume_task_exception)

    def _cancel_backchannel(self) -> None:
        task = self._backchannel_task
        self._backchannel_task = None
        if task is not None and not task.done():
            task.cancel()

    async def _run_backchannel(self, ctx: protocol.ResponseContext) -> None:
        try:
            if self._turn_started_at is None:
                return
            delay = max(0, self._settings.backchannel_delay_ms) / 1000.0
            remaining = self._turn_started_at + delay - time.perf_counter()
            if remaining > 0:
                await asyncio.sleep(remaining)
            if (
                self._closed
                or ctx.generation != self._generation
                or not self._response_active
            ):
                return
            if self._first_audio_generation == ctx.generation:
                return
            phrase = self._pick_backchannel()
            pcm = self._tts.cached(phrase, self._voice)
            if not pcm:
                logger.info(
                    "backchannel cache miss response=%s sound=%s",
                    ctx.response_id,
                    phrase,
                )
                return
            # Re-check after the (sync) cache lookup.  TTS may have produced
            # its first chunk while this task was scheduled.
            if self._first_audio_generation == ctx.generation:
                return
            elapsed_ms = (
                (time.perf_counter() - self._turn_started_at) * 1000.0
                if self._turn_started_at is not None
                else -1.0
            )
            if await self._send(protocol.backchannel(ctx, phrase, pcm), ctx=ctx):
                self._backchannel_sent_generation = ctx.generation
                self._backchannel_sent_this_turn = True
                self._last_backchannel = phrase
                logger.info(
                    "backchannel sent response=%s sound=%s after_asr_ms=%.1f bytes=%d",
                    ctx.response_id,
                    phrase,
                    elapsed_ms,
                    len(pcm),
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("backchannel failed response=%s", ctx.response_id)

    def _pick_backchannel(self) -> str:
        phrases = [
            p for p in self._settings.backchannel_phrases if p != self._last_backchannel
        ]
        if not phrases:
            phrases = list(self._settings.backchannel_phrases)
        return random.choice(phrases)

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
                if await self._run_tools(ctx, plan.tool_calls) is None:
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
    ) -> list[tuple[protocol.ToolCall, str]] | None:
        usable = [call for call in calls if call.name in self._tool_names]
        dropped = [call.name for call in calls if call.name not in self._tool_names]
        if dropped:
            logger.info("dropping unavailable/mock tools: %s", ",".join(sorted(set(dropped))))
        if not usable:
            return []
        self._pending_tools = {
            call.call_id: asyncio.get_running_loop().create_future() for call in usable
        }
        ok = await self._send(protocol.tool_call_done(ctx, usable), ctx=ctx)
        if not ok:
            return None
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
            return None
        finally:
            self._pending_tools.clear()
        pairs = list(zip(usable, results))
        for call, result in pairs:
            logger.info(
                "tool result call=%s name=%s chars=%d",
                call.call_id,
                call.name,
                len(result or ""),
            )
            self._history.append(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "name": call.name,
                    "content": result or "ok",
                }
            )
        self._trim_history()
        return pairs

    async def _speak(
        self, ctx: protocol.ResponseContext, text: str, *, emit_text: bool
    ) -> None:
        source = self._llm.stream_text(text) if emit_text else self._single_text(text)
        await self._stream_speech(ctx, source, emit_text=emit_text)
        await self._send(protocol.response_done(ctx, "completed"), ctx=ctx)

    @staticmethod
    async def _single_text(text: str):
        yield text

    async def _stream_speech(self, ctx, source, *, emit_text: bool) -> str:
        pacer = AudioPacer(self._send_response_event, ctx)
        self._pacer = pacer
        pacer.start()
        buffer = ""
        full = ""
        first_chunk = True
        try:
            async for delta in source:
                if not isinstance(delta, str) or not delta:
                    continue
                full += delta
                if emit_text:
                    if not await self._send(protocol.text_delta(ctx, delta), ctx=ctx):
                        return full
                buffer += delta
                limit = 15 if first_chunk else 18
                if self._should_flush(buffer, limit):
                    await self._synthesize_into(pacer, buffer, ctx)
                    buffer = ""
                    first_chunk = False
            if buffer.strip():
                await self._synthesize_into(pacer, buffer, ctx)
            await pacer.finish()
            if emit_text and full:
                await self._send(protocol.text_done(ctx, full), ctx=ctx)
            return full
        finally:
            pacer.cancel()

    async def _run_deepseek_turn(
        self, ctx: protocol.ResponseContext, transcript: str
    ) -> None:
        try:
            if self._deepseek is None:
                raise RuntimeError("DeepSeek backend not initialized")
            for _hop in range(3):
                tool_calls: list[protocol.ToolCall] = []
                text_parts: list[str] = []

                async def text_source():
                    async for delta in self._deepseek.stream(
                        self._build_llm_messages(), self._tools_openai
                    ):
                        if delta.text:
                            text_parts.append(delta.text)
                            yield delta.text
                        if delta.tool_calls:
                            tool_calls.extend(delta.tool_calls)

                text = await self._stream_speech(ctx, text_source(), emit_text=True)
                if tool_calls:
                    self._history.append(
                        {
                            "role": "assistant",
                            "content": text or None,
                            "tool_calls": [
                                {
                                    "id": call.call_id,
                                    "type": "function",
                                    "function": {
                                        "name": call.name,
                                        "arguments": call.arguments,
                                    },
                                }
                                for call in tool_calls
                            ],
                        }
                    )
                    results = await self._run_tools(ctx, tool_calls)
                    if results is None:
                        return
                    if not results:
                        # All calls were filtered (e.g. a hallucinated
                        # ask_expert).  Keep the OpenAI tool-call chain valid.
                        for call in tool_calls:
                            self._history.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call.call_id,
                                    "name": call.name,
                                    "content": "该工具不可用，请直接回答。",
                                }
                            )
                    self._trim_history()
                    await self._send(protocol.response_done(ctx, "tool_calls"), ctx=ctx)
                    ctx = self._begin_response()
                    self._schedule_backchannel(ctx)
                    continue
                if text:
                    self._history.append({"role": "assistant", "content": text})
                    self._trim_history()
                await self._send(
                    protocol.response_done(ctx, "completed" if text else "empty"), ctx=ctx
                )
                return
            await self._send(protocol.response_canceled(ctx), force=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("deepseek turn failed response=%s", ctx.response_id)
            if not self._closed and ctx.generation == self._generation:
                await self._send(protocol.response_done(ctx, "failed"), ctx=ctx)
        finally:
            self._finish_response(ctx)

    def _build_llm_messages(self) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": build_system_prompt(self._instructions)},
            *self._history,
        ]

    async def _synthesize_into(
        self, pacer: AudioPacer, text: str, ctx: protocol.ResponseContext
    ) -> None:
        first_pcm = True
        for chunk in split_for_tts(text) or [text]:
            async for pcm in self._tts.synthesize(chunk, self._voice, ctx.generation):
                if first_pcm:
                    first_pcm = False
                    if self._first_audio_generation != ctx.generation:
                        self._first_audio_generation = ctx.generation
                        self._log_first_audio(ctx)
                await pacer.feed(pcm)

    def _log_first_audio(self, ctx: protocol.ResponseContext) -> None:
        if self._turn_started_at is None:
            return
        elapsed_ms = (time.perf_counter() - self._turn_started_at) * 1000.0
        logger.info(
            "tts first chunk formal response=%s after_asr_ms=%.1f backchannel=%s",
            ctx.response_id,
            elapsed_ms,
            "1" if self._backchannel_sent_generation == ctx.generation else "0",
        )

    @staticmethod
    def _should_flush(buffer: str, limit: int) -> bool:
        if not buffer:
            return False
        if any(ch in "。！？!?；;\n" for ch in buffer):
            return True
        return len(buffer) >= limit

    async def _cancel_active(self, *, send_canceled: bool, reason: str) -> None:
        self._discard_speculative_asr()
        self._cancel_backchannel()
        self._turn_started_at = None
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
        self._cancel_backchannel()
        self._turn_started_at = None
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
        self._cancel_voice_warm()
        self._vad.reset()
        if not self._closed:
            await self._cancel_active(send_canceled=False, reason="shutdown")
            self._closed = True
        if self._deepseek is not None:
            await self._deepseek.close()

    def _trim_history(self) -> None:
        if len(self._history) <= 60:
            return
        trimmed = self._history[-60:]
        while trimmed and trimmed[0].get("role") == "tool":
            trimmed.pop(0)
        self._history = trimmed
