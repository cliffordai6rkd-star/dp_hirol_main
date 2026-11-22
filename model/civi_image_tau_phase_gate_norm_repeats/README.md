# Normalized CNN phase-gate candidate

Best normalized tiny CNN trial: `tiny_norm_seed43`, 5175 parameters, CE 0.114010, precontact open 61.90%, free false open 1.70%.

The image CNN centers RGB to [-1, 1], uses GroupNorm after each convolution, and preserves a 2x2 spatial grid before projection. This candidate uses current images (offsets=[0]), 50-frame thresholded normalized torque, GRU hidden 16, batch 64, lr 0.001. Load with `load_phase_gate` and `load_phase_gate_preprocessor` as documented in `model/civi_image_tau_phase_gate_ce_best/README.md`.

Across three seeds, the precontact open rates are 63.49%, 61.90%, 65.08%; free false open rates 1.84%, 1.70%, 2.70%. These are measured on the same three validation episodes. The model is a comparison candidate; the minimum-CE checkpoint remains in `model/civi_image_tau_phase_gate_ce_best/`.
