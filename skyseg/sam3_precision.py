"""Instance-local workaround for SAM3's BF16-only fused ViT MLP.

The pinned upstream ``sam3.perflib.fused.addmm_act`` unconditionally casts its
inputs and weights to BF16. Without autocast, ViT Mlp.fc2 then receives a BF16
activation with FP32 weights. Use the existing module layers normally in FP32;
do not change the installed SAM3 source or patch its global functions/classes.
"""

from __future__ import annotations

from types import MethodType


def _unfused_mlp_forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop1(x)
    x = self.norm(x)
    x = self.fc2(x)
    return self.drop2(x)


def configure_sam3_fp32(model, mlp_class=None) -> int:
    """Bypass BF16-only fused ops on this model's ViT Mlp instances.

    ``mlp_class`` can be injected for small regression tests without downloading
    SAM3 or loading its dependencies/checkpoint. No parameters are replaced.
    """
    if mlp_class is None:
        from sam3.model.vitdet import Mlp

        mlp_class = Mlp
    count = 0
    for module in model.modules():
        if isinstance(module, mlp_class):
            required = ("fc1", "act", "drop1", "norm", "fc2", "drop2")
            if not all(hasattr(module, name) for name in required):
                raise RuntimeError("SAM3 ViT Mlp structure changed; use the SAM3 revision in README.")
            module.forward = MethodType(_unfused_mlp_forward, module)
            count += 1
    if not count:
        raise RuntimeError("No SAM3 ViT Mlp modules found for FP32 inference; check the installed SAM3 revision.")
    return count
