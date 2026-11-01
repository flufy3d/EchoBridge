"""Apply the small WSL2 pin-memory fix to a vLLM-Omni checkout.

Why this exists
---------------
vLLM detects WSL2 and disables ``pin_memory`` by default because the WSL2
pinned-memory pool is small and can fail with ``cudaErrorUnknown``.  Two
vLLM-Omni hot paths bypass that guard:

* ``qwen3_tts/prompt_embeds_builder.py`` builds a fresh pinned tensor for each
  request's text ids;
* ``utils/device_copy.py`` pins every per-step metadata copy.

EchoBridge runs a long-lived TTS worker on WSL2.  Repeating those allocations
for hours makes the pinned pool grow, and the failure surfaces at the next
``torch.tensor(..., pin_memory=True)``.  The fix is to consult vLLM's own
platform guard; on native Linux nothing changes.

Run with the TTS worker's venv:

    python -m echobridge.vllm_omni_patch ~/.cache/echobridge/src/vllm-omni

The operation is idempotent.  ``--check`` reports the action without writing.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MARKER = "ECHOBRIDGE_PIN_MEMORY_PATCH"

PROMPT_BUILDER = Path(
    "vllm_omni/model_executor/models/qwen3_tts/prompt_embeds_builder.py"
)
DEVICE_COPY = Path("vllm_omni/utils/device_copy.py")


class PatchError(RuntimeError):
    pass


def _replace_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    count = text.count(old)
    if count != 1:
        raise PatchError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1), True


def patch_prompt_builder(text: str) -> tuple[str, bool]:
    if MARKER in text:
        return text, False
    text, inserted = _replace_once(
        text,
        "logger = init_logger(__name__)\n",
        (
            "logger = init_logger(__name__)\n"
            "\n"
            f"# {MARKER}: use vLLM's WSL/GPU guard instead of pinning every\n"
            "# request's text ids.  WSL2 has a small pinned-memory pool; a\n"
            "# long-lived worker that keeps allocating from it can fail with\n"
            "# cudaErrorUnknown.  The ids are tiny, so a pageable copy costs\n"
            "# well under 0.01 ms (measured on a 3070/WSL2).\n"
            "def _pin_memory_available() -> bool:\n"
            "    try:\n"
            "        from vllm.utils.platform_utils import is_pin_memory_available\n"
            "    except Exception:\n"
            "        return True\n"
            "    return bool(is_pin_memory_available())\n"
        ),
        "prompt builder marker",
    )
    if not inserted:
        return text, False
    pattern = re.compile(
        r'(?m)^(?P<indent>[ \t]*)use_async_h2d = device\.type == "cuda"[ \t]*$'
    )
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise PatchError(
            "prompt builder use_async_h2d: "
            f"expected one match, found {len(matches)}"
        )
    text = pattern.sub(
        lambda match: (
            f"{match.group('indent')}use_async_h2d = (\n"
            f"{match.group('indent')}    device.type == \"cuda\" "
            "and _pin_memory_available()\n"
            f"{match.group('indent')})"
        ),
        text,
        count=1,
    )
    return text, True


def patch_device_copy(text: str) -> tuple[str, bool]:
    if MARKER in text:
        return text, False
    text, inserted = _replace_once(
        text,
        "from vllm.utils.platform_utils import is_uva_available\n",
        (
            "from vllm.utils.platform_utils import (\n"
            "    is_pin_memory_available,\n"
            "    is_uva_available,\n"
            ")\n"
            "\n"
            f"# {MARKER}: mirror vLLM's WSL2 guard for every helper in this\n"
            "# module.  Non-pinned copies are blocking, but they are a few\n"
            "# metadata elements per step and much safer than exhausting the\n"
            "# WSL2 pinned pool in a worker that runs for days.\n"
            "_PIN_MEMORY_OK = is_pin_memory_available()\n"
        ),
        "device_copy marker",
    )
    if not inserted:
        return text, False

    text, a = _replace_once(
        text,
        "    return tensor.pin_memory().to(device, non_blocking=True)\n",
        (
            "    if not _PIN_MEMORY_OK:\n"
            "        return tensor.to(device)\n"
            "    return tensor.pin_memory().to(device, non_blocking=True)\n"
        ),
        "device_copy to_device_nonblocking",
    )
    text, b = _replace_once(
        text,
        "    return torch.tensor(values, dtype=dtype, pin_memory=True).to(device, non_blocking=True)\n",
        (
            "    if not _PIN_MEMORY_OK:\n"
            "        return torch.tensor(values, dtype=dtype, device=device)\n"
            "    return torch.tensor(values, dtype=dtype, pin_memory=True).to(device, non_blocking=True)\n"
        ),
        "device_copy index_to_device",
    )
    text, c = _replace_once(
        text,
        "        host = torch.empty(capacity, dtype=self.dtype, pin_memory=True)\n",
        "        host = torch.empty(capacity, dtype=self.dtype, pin_memory=_PIN_MEMORY_OK)\n",
        "device_copy DeviceStager",
    )
    return text, a or b or c


def _apply_file(path: Path, patcher) -> str:
    try:
        original = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PatchError(f"{path}: {exc}") from exc
    patched, changed = patcher(original)
    if not changed:
        return "already patched"
    path.write_text(patched, encoding="utf-8")
    return "patched"


def _check_file(path: Path, patcher) -> str:
    try:
        original = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PatchError(f"{path}: {exc}") from exc
    patched, changed = patcher(original)
    if MARKER in patched and not changed and MARKER in original:
        return "already patched"
    return "would patch"


def patch_tree(root: Path, *, check: bool = False) -> list[str]:
    if not root.is_dir():
        raise PatchError(f"vLLM-Omni source not found: {root}")
    results: list[str] = []
    for rel, patcher in (
        (PROMPT_BUILDER, patch_prompt_builder),
        (DEVICE_COPY, patch_device_copy),
    ):
        path = root / rel
        action = _check_file(path, patcher) if check else _apply_file(path, patcher)
        results.append(f"{action}: {rel}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default="~/.cache/echobridge/src/vllm-omni",
        help="vLLM-Omni source checkout (default: %(default)s)",
    )
    parser.add_argument(
        "--check", action="store_true", help="report without modifying files"
    )
    args = parser.parse_args(argv)
    root = Path(args.root).expanduser()
    try:
        for line in patch_tree(root, check=args.check):
            print(line)
    except PatchError as exc:
        print(f"vllm-omni patch failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
