# CIVI phase gate CE search winner

Best trial: `tiny_lr7e4_b32`. Validation CE: 0.107641. Parameters: 4,551.

Architecture and input configuration: `config.json`. The checkpoint includes torque preprocessing/normalization state, architecture and training summary. Full search: `data/debug/20261120_phase_gate_ce_search_01/README.md`.

```python
import torch
from diffusion_policy.model.vision.image_tau_phase_gate import load_phase_gate, load_phase_gate_preprocessor

path = "model/civi_image_tau_phase_gate_ce_best/phase_gate_best.pt"
gate = load_phase_gate(path, "cuda:0")
preprocess = load_phase_gate_preprocessor(path, "cuda:0")
# obs['tau_ext']: [B, 2, 50, 7], raw physical values, same units as training.
# images: RGB [0,1], [B,2,3,H,W] if offsets=(0,).
# For multiple offsets, supply [B,2,len(gate.image_offsets),3,H,W], in offset order.
# Historical image offsets are measured at 25 Hz, using only current/past frames.
# Pad missing image history by repeating the episode's first image.
with torch.no_grad():
    tau, _, _ = preprocess(obs)
    logits = gate(images, tau)
    phase = logits.argmax(-1)
    render_open = phase != 0
```

Original model preserved in `model/civi_image_tau_phase_gate/`. This checkpoint was selected on three validation episodes; it has no new independent test result.
