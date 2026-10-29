"""Real torch CPU reproductions of the upstream BF16 fused MLP failure.

No SAM3 checkpoint/GPU is needed. The tiny MLP uses the same BF16 aten fused
operation followed by an FP32 fc2 as the pinned official SAM3 ViT backbone.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch", reason="Optional real-tensor tests; install a CPU or CUDA PyTorch wheel")

from skyseg.annotate import Sam3Backend
from skyseg.sam3_precision import configure_sam3_fp32


class Bf16FusedMlp(torch.nn.Module):
    def __init__(self, activation=torch.nn.GELU, norm=False):
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 16)
        self.act = activation()
        self.drop1 = torch.nn.Dropout(0.1)
        self.norm = torch.nn.LayerNorm(16) if norm else torch.nn.Identity()
        self.fc2 = torch.nn.Linear(16, 8)
        self.drop2 = torch.nn.Dropout(0.1)

    def forward(self, x):
        # Mirrors sam3/perflib/fused.py's unconditional casts, not torch autocast.
        shape = x.shape
        x = torch.ops.aten._addmm_activation(
            self.fc1.bias.detach().to(torch.bfloat16),
            x.to(torch.bfloat16).reshape(-1, shape[-1]),
            self.fc1.weight.detach().to(torch.bfloat16).t(),
            use_gelu=isinstance(self.act, torch.nn.GELU),
        ).reshape(*shape[:-1], 16)
        return self.drop2(self.fc2(self.norm(self.drop1(x))))


@pytest.mark.parametrize("shape", [(2, 8), (2, 3, 8)])
@pytest.mark.parametrize("activation,norm", [(torch.nn.GELU, False), (torch.nn.ReLU, True)])
def test_reproduces_error_and_fp32_fix_preserves_parameters(shape, activation, norm):
    block = Bf16FusedMlp(activation, norm).eval()
    model = torch.nn.Sequential(block).eval()
    x = torch.randn(shape)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="mat1 and mat2 must have the same dtype"):
            model(x)
        expected = block.drop2(block.fc2(block.norm(block.drop1(block.act(block.fc1(x))))))
        assert configure_sam3_fp32(model, Bf16FusedMlp) == 1
        actual = model(x)
    assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(torch.equal(before[name], tensor) for name, tensor in model.state_dict().items())


def test_fix_is_instance_local_and_idempotent():
    patched = Bf16FusedMlp().eval()
    untouched = Bf16FusedMlp().eval()
    with torch.inference_mode():
        configure_sam3_fp32(patched, Bf16FusedMlp)
        x = torch.randn(1, 8)
        expected = patched(x)
        configure_sam3_fp32(patched, Bf16FusedMlp)
        torch.testing.assert_close(patched(x), expected, rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="mat1 and mat2 must have the same dtype"):
            untouched(x)


def test_fail_clearly_when_upstream_structure_changes():
    with pytest.raises(RuntimeError, match="No SAM3 ViT Mlp"):
        configure_sam3_fp32(torch.nn.Linear(8, 8), Bf16FusedMlp)

    class ChangedMlp(torch.nn.Module):
        pass

    with pytest.raises(RuntimeError, match="structure changed"):
        configure_sam3_fp32(ChangedMlp(), ChangedMlp)


class TinyProcessor:
    def __init__(self, model, device="cpu", confidence_threshold=0.5):
        self.model = model
        self.text_projection = torch.nn.Linear(8, 1).eval()
        self.expected_autocast = False
        self.stages = []

    def set_image(self, image):
        assert torch.is_autocast_enabled("cpu") == self.expected_autocast
        assert torch.is_inference_mode_enabled()
        self.stages.append("image")
        return {"features": self.model(torch.ones(1, 8)), "shape": (image.height, image.width)}

    def set_text_prompt(self, state, prompt):
        assert prompt == "sky"
        assert torch.is_autocast_enabled("cpu") == self.expected_autocast
        assert torch.is_inference_mode_enabled()
        self.stages.append("text")
        return {"masks": torch.ones((1, 1, *state["shape"]), dtype=torch.bool),
                "scores": self.text_projection(state["features"]).sigmoid().reshape(-1)}


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_backend_fp32_and_unsupported_bf16_fallback_under_ambient_autocast(monkeypatch, precision):
    packages = {}
    for name in ("sam3", "sam3.model", "sam3.model_builder", "sam3.model.vitdet",
                 "sam3.model.sam3_image_processor"):
        packages[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, packages[name])

    def builder(**kwargs):
        assert kwargs["device"] == "cpu" and kwargs["eval_mode"] is True
        return torch.nn.Sequential(Bf16FusedMlp())

    packages["sam3.model_builder"].build_sam3_image_model = builder
    packages["sam3.model.vitdet"].Mlp = Bf16FusedMlp
    packages["sam3.model.sam3_image_processor"].Sam3Processor = TinyProcessor
    args = SimpleNamespace(device="cpu", checkpoint=None, bpe_path=None, confidence=0.5, precision=precision)
    backend = Sam3Backend(args)
    assert backend.bf16 is False
    with torch.autocast("cpu", dtype=torch.bfloat16):
        masks, scores = backend.predict(Image.new("RGB", (20, 10)), "sky")
        assert torch.is_autocast_enabled("cpu")  # outer context restored
    assert masks.shape == (1, 1, 10, 20) and masks.dtype == np.bool_
    assert scores.dtype == np.float32 and np.isfinite(scores).all()
    assert backend.processor.stages == ["image", "text"]


def test_bf16_context_covers_image_and_text_with_real_cpu_autocast():
    # CPU autocast checks the same context boundary without pretending to test CUDA.
    backend = Sam3Backend.__new__(Sam3Backend)
    backend.torch = torch
    backend.device = torch.device("cpu")
    backend.bf16 = True
    backend.processor = TinyProcessor(torch.nn.Sequential(Bf16FusedMlp()).eval())
    backend.processor.expected_autocast = True
    masks, scores = backend.predict(Image.new("RGB", (20, 10)), "sky")
    assert masks.dtype == np.bool_ and scores.dtype == np.float32
    assert np.isfinite(scores).all() and backend.processor.stages == ["image", "text"]
