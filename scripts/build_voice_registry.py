#!/usr/bin/env python3
"""Build a ``voices.json`` registry from candidate_*.wav files.

This does not load the TTS model and does not modify the audio files.  It is
safe to run on the production voice directory after ``design_voice.py``:

  python scripts/build_voice_registry.py
  python scripts/build_voice_registry.py --default-candidate 1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_DIR = Path.home() / ".cache" / "echobridge" / "voices" / "mingyue"

# Fixed identities for the three confirmed candidates.
VOICE_IDENTITIES = {
    1: ("mingyue_xixi_v1", "汐汐", "明亮活泼，音调偏高"),
    2: ("mingyue_nuonuo_v1", "糯糯", "软糯温柔，音量偏轻"),
    3: ("mingyue_v1", "月月", "温暖自然，默认声线"),
}


def _relative(target: Path, base: Path) -> str:
    try:
        return os.path.relpath(target, base)
    except ValueError:
        return str(target)


def build(voice_dir: Path, out: Path, default_candidate: int, ref_text: str) -> dict:
    if default_candidate not in VOICE_IDENTITIES:
        raise SystemExit("--default-candidate must be 1, 2, or 3")
    text_path = voice_dir / ref_text
    if not text_path.is_file() or not text_path.read_text(encoding="utf-8").strip():
        raise SystemExit(f"reference text missing or empty: {text_path}")

    voices = []
    order = [default_candidate] + [
        candidate for candidate in sorted(VOICE_IDENTITIES)
        if candidate != default_candidate
    ]
    for candidate in order:
        audio = voice_dir / f"candidate_{candidate}.wav"
        if not audio.is_file():
            print(f"skip candidate_{candidate}: {audio} not found")
            continue
        voice_id, name, description = VOICE_IDENTITIES[candidate]
        voices.append(
            {
                "id": voice_id,
                "name": name,
                "description": description,
                "ref_audio": _relative(audio, out.parent),
                "ref_text_file": _relative(text_path, out.parent),
            }
        )
    if not voices:
        raise SystemExit(f"no candidate_*.wav found in {voice_dir}")
    default_id = VOICE_IDENTITIES[default_candidate][0]
    if default_id not in {voice["id"] for voice in voices}:
        raise SystemExit(
            f"default candidate_{default_candidate}.wav is missing in {voice_dir}"
        )
    return {"default": default_id, "voices": voices}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice-dir", default=str(DEFAULT_DIR))
    parser.add_argument("--out", default="", help="default: <voice-dir>/voices.json")
    parser.add_argument("--default-candidate", type=int, default=3)
    parser.add_argument("--ref-text", default="reference.txt")
    args = parser.parse_args()

    voice_dir = Path(os.path.expanduser(args.voice_dir))
    out = (
        Path(os.path.expanduser(args.out))
        if args.out
        else voice_dir / "voices.json"
    )
    registry = build(voice_dir, out, args.default_candidate, args.ref_text)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {out} default={registry['default']} voices={len(registry['voices'])}"
    )


if __name__ == "__main__":
    main()
