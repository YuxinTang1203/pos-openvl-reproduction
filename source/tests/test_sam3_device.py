import contextlib
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from pyrualean.hosts import sam3_local


@pytest.fixture
def fake_cuda(monkeypatch, tmp_path):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("PYRUALEAN_SAM3_URL", raising=False)
    monkeypatch.setattr(sam3_local, "_SHARED", {})
    local = threading.local()
    local.current = 2  # Another component has already selected/initialized CUDA.
    events = []

    class Cuda:
        def is_available(self):
            return True

        def current_device(self):
            return getattr(local, "current", 0)

        def set_device(self, index):
            local.current = int(str(index).split(":")[-1])

        @contextlib.contextmanager
        def device(self, index):
            old = self.current_device()
            self.set_device(index)
            try:
                yield
            finally:
                self.set_device(old)

    cuda = Cuda()
    torch = types.ModuleType("torch")
    torch.cuda = cuda
    torch.device = lambda kind, index: f"{kind}:{index}"
    torch.bfloat16 = object()
    torch.autocast = lambda *args, **kwargs: contextlib.nullcontext()
    torch.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()), cudnn=types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", torch)
    for name in ("sam3", "sam3.model", "sam3.model.sam3_image_processor", "sam3.model_builder"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    def build(**kwargs):
        events.append(("build", cuda.current_device()))
        return types.SimpleNamespace(index=cuda.current_device())

    class Processor:
        def __init__(self, model, device, **kwargs):
            assert device == f"cuda:{model.index}"
            self.index = model.index

        def set_image(self, image):
            assert cuda.current_device() == self.index
            events.append(("image", cuda.current_device()))
            return {}

        def set_text_prompt(self, **kwargs):
            assert cuda.current_device() == self.index
            events.append(("text", cuda.current_device()))
            return {"masks": np.ones((1, 3, 4)), "scores": np.array([.9])}

    sys.modules["sam3.model_builder"].build_sam3_image_model = build
    sys.modules["sam3.model.sam3_image_processor"].Sam3Processor = Processor
    checkpoint = tmp_path / "weights.pt"
    checkpoint.touch()
    return cuda, events, checkpoint


def test_initialized_cuda_and_worker_use_requested_device(fake_cuda, monkeypatch):
    cuda, events, checkpoint = fake_cuda
    model = sam3_local.LocalSam3(str(checkpoint), cuda_device=3)
    assert cuda.current_device() == 2  # Constructor does not change caller state.
    assert "CUDA_VISIBLE_DEVICES" not in __import__("os").environ

    def infer():
        assert cuda.current_device() == 0  # New thread's default differs from model.
        result = model.segment(np.zeros((3, 4, 3)), text_prompt="object")
        assert cuda.current_device() == 0
        return result

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(infer).result().found
    assert events == [("build", 3), ("image", 3), ("text", 3)]


@pytest.mark.parametrize("mask, requested, expected", [
    ("7", 7, 0), ("7", 0, 0), ("GPU-uuid", 7, 0),
    ("3,7", 7, 1), ("3,7", None, 0), ("7,3", 7, 0),
])
def test_launcher_visibility_is_preserved(fake_cuda, monkeypatch, mask, requested, expected):
    _, events, checkpoint = fake_cuda
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    sam3_local.LocalSam3(str(checkpoint), cuda_device=requested)
    assert events == [("build", expected)]
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == mask


@pytest.mark.parametrize("mask, requested", [("", 0), ("-1", 0), ("3,7", 2)])
def test_unavailable_mapping_fails_before_loading(fake_cuda, monkeypatch, mask, requested):
    _, events, checkpoint = fake_cuda
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    with pytest.raises((ValueError, RuntimeError)):
        sam3_local.LocalSam3(str(checkpoint), cuda_device=requested)
    assert not events


def test_shared_model_reuses_alias_but_not_another_gpu(fake_cuda, monkeypatch):
    _, events, checkpoint = fake_cuda
    alias = checkpoint.parent / "alias.pt"
    alias.symlink_to(checkpoint)
    first = sam3_local.shared(str(checkpoint), cuda_device=3)
    assert sam3_local.shared(str(alias), cuda_device=3) is first
    assert sam3_local.shared(str(checkpoint), cuda_device=4) is not first
    assert events == [("build", 3), ("build", 4)]
