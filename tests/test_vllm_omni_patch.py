from __future__ import annotations

from pathlib import Path

from echobridge.vllm_omni_patch import (
    DEVICE_COPY,
    MARKER,
    PROMPT_BUILDER,
    patch_tree,
)


PROMPT_SOURCE = '''\
from vllm.logger import init_logger

logger = init_logger(__name__)


def preprocess_infos_batch(device):
    use_async_h2d = device.type == "cuda"
    packed_ids_cpu = torch.tensor(flat_ids, dtype=torch.long, pin_memory=use_async_h2d)
'''

DEVICE_COPY_SOURCE = '''\
from vllm.utils.platform_utils import is_uva_available


def to_device_nonblocking(tensor, device):
    if tensor.device.type != "cpu" or device.type not in {"cuda", "npu"}:
        return tensor.to(device)
    return tensor.pin_memory().to(device, non_blocking=True)


def index_to_device(values, device, dtype=torch.long):
    if torch.device(device).type not in {"cuda", "npu"}:
        return torch.tensor(values, dtype=dtype, device=device)
    return torch.tensor(values, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


class DeviceStager:
    def _grow(self, slot, n, device):
        capacity = 16
        host = torch.empty(capacity, dtype=self.dtype, pin_memory=True)
'''


def _make_tree(tmp_path: Path) -> Path:
    root = tmp_path / "vllm-omni"
    prompt = root / PROMPT_BUILDER
    device = root / DEVICE_COPY
    prompt.parent.mkdir(parents=True)
    device.parent.mkdir(parents=True)
    prompt.write_text(PROMPT_SOURCE, encoding="utf-8")
    device.write_text(DEVICE_COPY_SOURCE, encoding="utf-8")
    return root


def test_patch_is_idempotent_and_disables_wsl_pinning(tmp_path):
    root = _make_tree(tmp_path)

    first = patch_tree(root)
    assert all(line.startswith("patched:") for line in first)
    prompt = (root / PROMPT_BUILDER).read_text(encoding="utf-8")
    device = (root / DEVICE_COPY).read_text(encoding="utf-8")
    assert MARKER in prompt
    assert "and _pin_memory_available()" in prompt
    assert MARKER in device
    assert "_PIN_MEMORY_OK = is_pin_memory_available()" in device
    assert "if not _PIN_MEMORY_OK:\n        return tensor.to(device)" in device
    assert "pin_memory=_PIN_MEMORY_OK" in device

    second = patch_tree(root)
    assert all(line.startswith("already patched:") for line in second)


def test_check_mode_does_not_write(tmp_path):
    root = _make_tree(tmp_path)
    before = (root / PROMPT_BUILDER).read_text(encoding="utf-8")

    result = patch_tree(root, check=True)
    assert all(line.startswith("would patch:") for line in result)
    assert (root / PROMPT_BUILDER).read_text(encoding="utf-8") == before
