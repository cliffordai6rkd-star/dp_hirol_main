# CIVI image + tau_ext phase gate

4,551 parameters. The shared image CNN uses channels 4/8/12, with each camera resized to 64×64. A one-layer GRU (hidden 16) encodes torque history; an MLP (hidden 16) predicts 0=free, 1=precontact, 2=contact. Open rendering when `argmax(logits) != 0`.

Trained on the same episode split as the A/B distillation experiment. Unweighted three-class CE; batch 64; AdamW lr 0.001. Validation early stopping: patience 3, min_delta 0.001. Training stopped at epoch 15 and restored epoch 12 (validation CE 0.11850). Validation free false-open rate 0.99%, precontact open rate 47.62%, contact open rate 100%. Precontact detection remains incomplete.

The saved checkpoint includes torque filtering/normalization, its state and the exact architecture. This experiment follows the source checkpoint: **50 frames at 25 Hz (2 seconds)**, with 7 torque channels and two images (`side`, `wrist`) per observation step. Inputs do not contain offline phase labels, position or velocity. Offline phase labels only supervise CE. The model accepts variable history length, but this saved model was trained on 50 frames.

```python
import torch
from diffusion_policy.model.vision.image_tau_phase_gate import (
    load_phase_gate, load_phase_gate_preprocessor,
)

checkpoint = "model/civi_image_tau_phase_gate/phase_gate_best.pt"
gate = load_phase_gate(checkpoint, "cuda:0")
preprocess = load_phase_gate_preprocessor(checkpoint, "cuda:0")
# obs images: RGB float [0, 1], shape [B, 2, 3, H, W].
# obs["tau_ext"]: raw history [B, 2, 50, 7], same units as training.
with torch.no_grad():
    tau_ext, _, _ = preprocess(obs)
    images = {key: obs[key][:, :2] for key in gate.image_keys}
    logits = gate(images, tau_ext)
    phase = logits.argmax(-1)
    render_open = phase != 0
```

Experiment, training metrics and student loss recalculation: `data/debug/20261120_image_tau_phase_gate_02/`.
