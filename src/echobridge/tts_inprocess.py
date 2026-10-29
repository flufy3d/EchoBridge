"""Experimental all-in-one Qwen3-TTS backend.

This module drives the upstream HF ``Qwen3TTSModel`` talker loop directly so
that codec frames can be decoded while the talker is still generating.  It is
intentionally *not* the default: on an RTX 3070 the pure-PyTorch talker runs at
roughly RTF 3, while vLLM-Omni achieves RTF < 0.5 with CUDA graphs.  The
backend is kept behind ``ECHOBRIDGE_TTS_BACKEND=inprocess`` for experiments and
for hardware where the HF path is fast enough.

What it does provide:

* one process / one CUDA context shared with :class:`InProcessQwen3ASR`;
* the voice-clone prompt tensors live on the GPU, so no 440 KB base64
  ``reference.wav`` is sent per sentence;
* true streaming decode with a 2/4/8/12/25-frame ramp and a 72-frame left
  context (mirroring ``deploy/qwen3_tts_echobridge.yaml``);
* cooperative cancellation between talker steps (barge-in).

No audio is ever played by this module or its benchmarks.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._qwen_compat import ensure_torchaudio_stub
from .audio import trim_silence_pcm16
from .config import Settings
from .gpu import PRIORITY_TTS, get_gpu_scheduler
from .logging import get_logger
from .tts import GREETING, load_backchannel_clips
from .voices import VoiceRegistry, load_voice_registry

logger = get_logger("echobridge.tts_inprocess")

SAMPLE_RATE = 24_000
FRAMES_PER_SECOND = 12.5


@dataclass(slots=True)
class _StreamState:
    voice: str
    device: Any
    mask: Any
    past: Any
    past_hidden: Any
    step: int
    logits: Any
    trailing: Any
    pad: Any
    eos: int
    suppress: tuple[int, ...]
    generated: list[int]
    codes: list[Any]
    cancelled: bool = False


def _pcm16(wav: Any) -> bytes:
    import numpy as np

    audio = np.asarray(wav, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    np.clip(audio, -1.0, 1.0, out=audio)
    return (audio * 32767.0).astype(np.int16).tobytes()


def _blocks(pcm: bytes, block: int = 4_800) -> AsyncIterator[bytes]:
    for offset in range(0, len(pcm), block):
        chunk = pcm[offset : offset + block]
        if chunk:
            yield chunk


class InProcessQwen3TTS:
    """HF Qwen3-TTS with streaming codec decode and cooperative cancel."""

    def __init__(
        self, settings: Settings, registry: VoiceRegistry | None = None
    ) -> None:
        self._settings = settings
        self._registry = registry or load_voice_registry(settings)
        self._voice_id = self._registry.default_id
        self._model_path = str(
            Path(os.path.expanduser(settings.tts_model))
        )
        self._device = settings.tts_device
        self._language = settings.tts_language or "Chinese"
        self._max_new_tokens = max(8, settings.tts_max_new_tokens)
        self._initial = max(1, settings.tts_initial_codec_frames)
        self._chunk = max(1, settings.tts_codec_chunk_frames)
        ramp = tuple(v for v in settings.tts_codec_chunk_ramp if v > 0)
        if not ramp:
            ramp = (self._initial,)
        if ramp[-1] != self._chunk:
            # vLLM-Omni warns that a ramp whose last entry does not match
            # codec_chunk_frames reintroduces the chunk-size cliff.
            ramp = ramp + (self._chunk,)
        self._ramp = ramp
        self._left = max(0, settings.tts_codec_left_context_frames)
        self._scheduler = get_gpu_scheduler()
        self._model: Any = None
        self._tokenizer: Any = None
        self._suppress_cache: tuple[int, ...] = ()
        self._prompts: dict[str, Any] = {}
        self._vcp: dict[str, dict[str, Any]] = {}
        self._load_error = ""
        self._load_s = 0.0
        self._load_lock: asyncio.Lock | None = None
        self._prompt_locks: dict[str, asyncio.Lock] = {}
        self._thread_lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, str], bytes] = OrderedDict()
        self._cache_max = 8
        self._backchannel_cache: dict[tuple[str, str], bytes] = (
            load_backchannel_clips(settings, self._registry)
        )
        self._backchannel_phrases = (
            settings.backchannel_phrases if settings.backchannel_enabled else ()
        )
        self._warmed: set[str] = set()
        self._warm_tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        return "inprocess"

    def _ensure_load_lock(self) -> asyncio.Lock:
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()
        return self._load_lock

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._ensure_load_lock():
            if self._model is not None:
                return
            try:
                await self._scheduler.run(
                    "tts-load",
                    self._load_sync,
                    priority=PRIORITY_TTS,
                    timeout=None,
                )
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                logger.exception("inprocess tts load failed")
                raise
        if self._model is None:
            raise RuntimeError(self._load_error or "TTS model not loaded")

    def _load_sync(self) -> None:
        if self._model is not None:
            return
        with self._thread_lock:
            if self._model is not None:
                return
            ensure_torchaudio_stub()
            import torch
            from qwen_tts import Qwen3TTSModel

            started = time.perf_counter()
            kwargs: dict[str, Any] = {}
            if self._settings.tts_attn_implementation:
                kwargs["attn_implementation"] = self._settings.tts_attn_implementation
            try:
                model = Qwen3TTSModel.from_pretrained(
                    self._model_path,
                    device_map=self._device,
                    dtype=torch.bfloat16,
                    **kwargs,
                )
            except TypeError:
                model = Qwen3TTSModel.from_pretrained(
                    self._model_path,
                    device_map=self._device,
                    dtype=torch.bfloat16,
                )
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            self._model = model
            self._tokenizer = model.model.speech_tokenizer
            if self._settings.tts_compile:
                # The codec decoder is convolutional and has no autoregressive
                # state, so it is the safe half to compile.  The talker's
                # per-step Python loop graph-breaks heavily; compiling it did
                # not change the measured RTF and is intentionally left out.
                try:
                    self._tokenizer.model = torch.compile(
                        self._tokenizer.model,
                        mode="reduce-overhead",
                        dynamic=True,
                    )
                    logger.info("inprocess tts code2wav torch.compile enabled")
                except Exception as exc:
                    logger.warning(
                        "inprocess tts code2wav torch.compile failed: %s", exc
                    )
            tc = model.model.config.talker_config
            self._suppress_cache = tuple(
                i
                for i in range(tc.vocab_size - 1024, tc.vocab_size)
                if i != tc.codec_eos_token_id
            )
            self._load_s = time.perf_counter() - started
            self._load_error = ""
            logger.info(
                "inprocess tts loaded path=%s load_s=%.2f vram_mb=%.1f",
                self._model_path,
                self._load_s,
                torch.cuda.memory_allocated() / (1024 * 1024),
            )

    def _prompt_paths(self, voice_id: str) -> tuple[Path, Path]:
        voice_dir = Path(os.path.expanduser(self._settings.tts_voice_dir))
        if voice_id == self._registry.default_id:
            return voice_dir / "prompt.pt", voice_dir / "prompt.meta.json"
        return (
            voice_dir / f"prompt.{voice_id}.pt",
            voice_dir / f"prompt.{voice_id}.meta.json",
        )

    async def _ensure_prompt(self, voice_id: str) -> None:
        voice_id = self._registry.resolve(voice_id)[0].id
        if voice_id in self._vcp:
            return
        lock = self._prompt_locks.setdefault(voice_id, asyncio.Lock())
        async with lock:
            if voice_id in self._vcp:
                return
            await self._scheduler.run(
                "tts-prompt",
                self._load_prompt_sync,
                voice_id,
                priority=PRIORITY_TTS,
                timeout=None,
            )

    def _load_prompt_sync(self, voice_id: str) -> None:
        if self._model is None:
            raise RuntimeError(self._load_error or "TTS model not loaded")
        import torch

        spec, _ = self._registry.resolve(voice_id)
        prompt_path, _meta_path = self._prompt_paths(spec.id)
        items = None
        if prompt_path.exists():
            try:
                items = torch.load(
                    prompt_path, map_location="cpu", weights_only=False
                )
            except Exception as exc:
                logger.warning(
                    "inprocess tts prompt read failed voice=%s: %s", spec.id, exc
                )
        if not items:
            if not spec.ref_text:
                raise RuntimeError(f"reference text is empty for voice {spec.id}")
            items = self._model.create_voice_clone_prompt(
                ref_audio=str(spec.ref_audio),
                ref_text=spec.ref_text,
                x_vector_only_mode=False,
            )
        # Keep the clone prompt resident on the GPU.  It is tiny (~13 KB) and
        # removes the per-sentence 440 KB reference.wav upload entirely.
        device = torch.device(self._device)
        for item in items:
            for attr in ("ref_code", "ref_spk_embedding"):
                value = getattr(item, attr, None)
                if isinstance(value, torch.Tensor):
                    setattr(item, attr, value.to(device))
        self._prompts[spec.id] = items
        self._vcp[spec.id] = self._model._prompt_items_to_voice_clone_prompt(items)
        logger.info(
            "inprocess tts prompt ready voice=%s ref_frames=%s",
            spec.id,
            int(items[0].ref_code.shape[0]) if items[0].ref_code is not None else 0,
        )

    # ------------------------------------------------------------------
    # Talker prefill / step
    # ------------------------------------------------------------------

    def _prefill_sync(self, text: str, voice_id: str) -> _StreamState:
        import torch

        with torch.no_grad():
            return self._prefill_nograd(text, voice_id)

    def _prefill_nograd(self, text: str, voice_id: str) -> _StreamState:
        import torch

        model = self._model
        if model is None:
            raise RuntimeError(self._load_error or "TTS model not loaded")
        inner = model.model
        talker = inner.talker
        vcp = self._vcp[voice_id]
        prompt_items = self._prompts[voice_id]
        cfg = inner.config
        tc = cfg.talker_config

        input_ids = model._tokenize_texts(
            [model._build_assistant_text(text)]
        )[0]
        ref_text = prompt_items[0].ref_text or ""
        ref_ids = model._tokenize_texts([model._build_ref_text(ref_text)])[0]

        speaker_embed = model.model.generate_speaker_prompt(vcp)[0]
        language_id = tc.codec_language_id[self._language.lower()]

        tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
            talker.get_text_embeddings()(
                torch.tensor(
                    [
                        [
                            cfg.tts_bos_token_id,
                            cfg.tts_eos_token_id,
                            cfg.tts_pad_token_id,
                        ]
                    ],
                    device=model.device,
                    dtype=input_ids.dtype,
                )
            )
        ).chunk(3, dim=1)

        codec_prefill = [
            [
                tc.codec_think_id,
                tc.codec_think_bos_id,
                language_id,
                tc.codec_think_eos_id,
            ]
        ]
        codec_embedding_0 = talker.get_input_embeddings()(
            torch.tensor(
                codec_prefill, device=model.device, dtype=input_ids.dtype
            )
        )
        codec_embedding_1 = talker.get_input_embeddings()(
            torch.tensor(
                [[tc.codec_pad_id, tc.codec_bos_id]],
                device=model.device,
                dtype=input_ids.dtype,
            )
        )
        codec_embedding = torch.cat(
            [
                codec_embedding_0,
                speaker_embed.view(1, 1, -1),
                codec_embedding_1,
            ],
            dim=1,
        )

        role_embed = talker.text_projection(
            talker.get_text_embeddings()(input_ids[:, :3])
        )
        talker_embed = torch.cat(
            (
                tts_pad_embed.expand(
                    -1, codec_embedding.shape[1] - 2, -1
                ),
                tts_bos_embed,
            ),
            dim=1,
        ) + codec_embedding[:, :-1]
        talker_embed = torch.cat((role_embed, talker_embed), dim=1)

        icl_embed, trailing = model.model.generate_icl_prompt(
            text_id=input_ids[:, 3:-5],
            ref_id=ref_ids[:, 3:-2],
            ref_code=vcp["ref_code"][0].to(model.device),
            tts_pad_embed=tts_pad_embed,
            tts_eos_embed=tts_eos_embed,
            non_streaming_mode=False,
        )
        talker_embed = torch.cat([talker_embed, icl_embed], dim=1)
        mask = torch.ones(
            (1, talker_embed.shape[1]),
            dtype=torch.long,
            device=talker_embed.device,
        )

        with torch.no_grad():
            out = talker(
                inputs_embeds=talker_embed,
                attention_mask=mask,
                trailing_text_hidden=trailing,
                tts_pad_embed=tts_pad_embed,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        suppress = tuple(
            i
            for i in range(tc.vocab_size - 1024, tc.vocab_size)
            if i != tc.codec_eos_token_id
        )
        return _StreamState(
            voice=voice_id,
            device=talker_embed.device,
            mask=mask,
            past=out.past_key_values,
            past_hidden=out.past_hidden,
            step=out.generation_step,
            logits=out.logits[:, -1, :],
            trailing=trailing,
            pad=tts_pad_embed,
            eos=tc.codec_eos_token_id,
            suppress=suppress,
            generated=[],
            codes=[],
        )

    def _sample_sync(self, logits: Any, generated: list[int], eos: int) -> int:
        import torch

        values = logits.clone()
        if generated:
            for token in set(generated):
                v = values[0, token]
                values[0, token] = (
                    v / 1.05 if v > 0 else v * 1.05
                )
        values[0, list(self._suppress_cache)] = float("-inf")
        if len(generated) < 2:
            values[0, eos] = float("-inf")
        values = values / 0.9
        top_k = min(50, values.shape[1])
        top_values, top_indices = torch.topk(values[0], top_k)
        filtered = torch.full_like(values[0], float("-inf"))
        filtered[top_indices] = top_values
        probs = torch.softmax(filtered, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    def _step_sync(self, state: _StreamState) -> bool:
        """Advance one talker frame.  Returns True when generation is done."""
        import torch

        with torch.no_grad():
            return self._step_nograd(state)

    def _step_nograd(self, state: _StreamState) -> bool:
        import torch

        if state.cancelled:
            return True
        token = self._sample_sync(state.logits, state.generated, state.eos)
        if token == state.eos:
            return True
        state.generated.append(token)
        state.mask = torch.cat(
            [
                state.mask,
                torch.ones(
                    (1, 1), dtype=state.mask.dtype, device=state.mask.device
                ),
            ],
            dim=1,
        )
        with torch.no_grad():
            out = self._model.model.talker(
                input_ids=torch.tensor(
                    [[token]], device=state.device, dtype=torch.long
                ),
                attention_mask=state.mask,
                cache_position=torch.tensor(
                    [state.mask.shape[1] - 1], device=state.device
                ),
                past_key_values=state.past,
                past_hidden=state.past_hidden,
                generation_step=state.step,
                trailing_text_hidden=state.trailing,
                tts_pad_embed=state.pad,
                subtalker_dosample=True,
                subtalker_top_k=50,
                subtalker_top_p=1.0,
                subtalker_temperature=0.9,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        state.past = out.past_key_values
        state.past_hidden = out.past_hidden
        state.step = out.generation_step
        state.codes.append(out.hidden_states[1][0])
        state.logits = out.logits[:, -1, :]
        return False

    def _decode_sync(
        self,
        codes: list[Any],
        emitted: int,
        end: int,
    ) -> tuple[bytes, int]:
        import torch

        new_frames = end - emitted
        start = max(0, end - (self._left + new_frames))
        window = torch.stack(codes[start:end], dim=0)
        with torch.no_grad():
            wavs, sr = self._tokenizer.decode([{"audio_codes": window}])
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"unexpected codec sample rate {sr}")
        wav = wavs[0]
        skip = int(round((emitted - start) * (SAMPLE_RATE / FRAMES_PER_SECOND)))
        skip = max(0, min(skip, len(wav)))
        pcm = _pcm16(wav[skip:])
        return pcm, end

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

    async def synthesize(
        self, text: str, voice_id: str, generation: int
    ) -> AsyncIterator[bytes]:
        text = text.strip()
        if not text:
            return
        voice = self._registry.resolve(voice_id)[0].id
        key = (voice, text)
        pinned = self._backchannel_cache.get(key)
        if pinned is not None:
            for chunk in _blocks(pinned):
                yield chunk
            return
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            for chunk in _blocks(cached):
                yield chunk
            return

        await self._ensure_loaded()
        await self._ensure_prompt(voice)
        state = await self._scheduler.run(
            "tts-prefill",
            self._prefill_sync,
            text,
            voice,
            priority=PRIORITY_TTS,
            timeout=self._settings.tts_timeout_s,
        )
        collected = bytearray()
        emitted = 0
        canceled = False
        ramp_index = 0
        next_end = min(self._ramp[0], self._max_new_tokens)
        try:
            while state.step < self._max_new_tokens:
                done = await self._scheduler.run(
                    "tts-step",
                    self._step_sync,
                    state,
                    priority=PRIORITY_TTS,
                    timeout=self._settings.tts_timeout_s,
                )
                if state.cancelled:
                    canceled = True
                    break
                frames = len(state.codes)
                if frames >= next_end or (done and frames > emitted):
                    end = frames
                    pcm, emitted = await self._scheduler.run(
                        "tts-decode",
                        self._decode_sync,
                        state.codes,
                        emitted,
                        end,
                        priority=PRIORITY_TTS,
                        timeout=self._settings.tts_timeout_s,
                    )
                    ramp_index += 1
                    size = self._ramp[min(ramp_index, len(self._ramp) - 1)]
                    next_end = min(emitted + size, self._max_new_tokens)
                    if pcm:
                        collected.extend(pcm)
                        for chunk in _blocks(pcm):
                            yield chunk
                if done:
                    break
        except asyncio.CancelledError:
            canceled = True
            state.cancelled = True
            raise
        finally:
            state.cancelled = True
            state.past = None
            state.past_hidden = None
            state.logits = None
        if collected and not canceled:
            pcm = bytes(collected)
            self._cache[key] = pcm
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)

    def cached(self, text: str, voice_id: str) -> bytes | None:
        key = (self._registry.resolve(voice_id)[0].id, text.strip())
        pinned = self._backchannel_cache.get(key)
        if pinned is not None:
            return pinned
        return self._cache.get(key)

    async def _prefetch(self, text: str, voice: str) -> bytes:
        collected = bytearray()
        async for chunk in self.synthesize(text, voice, 0):
            collected.extend(chunk)
        return bytes(collected)

    async def _warm_phrase(self, voice: str, phrase: str) -> None:
        if (voice, phrase) in self._backchannel_cache:
            return
        try:
            pcm = trim_silence_pcm16(await self._prefetch(phrase, voice))
            if pcm:
                self._backchannel_cache[(voice, phrase)] = pcm
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "inprocess tts backchannel prewarm failed voice=%s phrase=%r: %s",
                voice,
                phrase,
                exc,
            )

    def _schedule_phrase_warm(self, voice: str) -> None:
        async def warm_all() -> None:
            for phrase in self._backchannel_phrases:
                if (voice, phrase) in self._backchannel_cache:
                    continue
                await self._warm_phrase(voice, phrase)

        if all(
            (voice, phrase) in self._backchannel_cache
            for phrase in self._backchannel_phrases
        ):
            return
        task = asyncio.create_task(
            warm_all(), name=f"tts-warm-{voice}-{len(self._warm_tasks)}"
        )
        self._warm_tasks.add(task)
        task.add_done_callback(self._warm_tasks.discard)

    async def _warm_voice(self, voice: str) -> None:
        voice = self._registry.resolve(voice)[0].id
        if voice in self._warmed:
            return
        try:
            pcm = await self._prefetch(GREETING, voice)
            if pcm:
                self._backchannel_cache[(voice, GREETING)] = pcm
        except Exception as exc:
            logger.warning(
                "inprocess tts greeting prewarm failed voice=%s: %s", voice, exc
            )
        self._warmed.add(voice)
        # Backchannels are background work: the scheduler gives ASR priority,
        # and the first turn must not wait for five extra HF generations.
        self._schedule_phrase_warm(voice)

    async def warmup(self) -> None:
        await self._ensure_loaded()
        await self._ensure_prompt(self._voice_id)
        await self._warm_voice(self._voice_id)

    async def ensure_warm(self, voice_id: str) -> None:
        await self._ensure_loaded()
        await self._ensure_prompt(voice_id)
        await self._warm_voice(voice_id)

    async def rewarm(self, voice_id: str) -> None:
        """Force one real synthesis after a worker/model reload."""
        voice = self._registry.resolve(voice_id)[0].id
        self._cache.pop((voice, GREETING), None)
        self._backchannel_cache.pop((voice, GREETING), None)
        await self._ensure_loaded()
        await self._ensure_prompt(voice)
        pcm = await self._prefetch(GREETING, voice)
        if pcm:
            self._backchannel_cache[(voice, GREETING)] = pcm
        self._warmed.add(voice)

    def health(self) -> dict[str, object]:
        memory = self._scheduler.memory() if self._model is not None else None
        return {
            "ok": self._model is not None,
            "backend": "inprocess",
            "model": self._model_path,
            "loaded": self._model is not None,
            "load_s": round(self._load_s, 2),
            "load_error": self._load_error,
            "language": self._language,
            "voices": [spec.id for spec in self._registry.voices],
            "prompts_loaded": sorted(self._prompts),
            "cache_entries": len(self._cache),
            "vram_allocated_mb": round(memory.allocated_mb, 1) if memory else 0.0,
            "vram_reserved_mb": round(memory.reserved_mb, 1) if memory else 0.0,
            "device": memory.device if memory else "",
        }

    async def close(self) -> None:
        tasks = list(self._warm_tasks)
        self._warm_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._model = None
        self._tokenizer = None
        self._prompts.clear()
        self._vcp.clear()
        self._cache.clear()
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
