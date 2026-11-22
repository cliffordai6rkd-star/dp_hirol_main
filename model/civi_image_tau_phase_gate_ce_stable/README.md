# Phase gate with the best mean CE across repeated seeds

Configuration: `tiny_spatial2`. 5127 parameters; CNN 4/8/12, 64x64 images, 2x2 spatial pooling, hidden 16, one-layer GRU, batch 64, AdamW lr 0.001 and weight decay 0.0001. History: 50 frames at 25 Hz. Three-class CE, patience 3, min_delta 0.001.

Seeds 42/43/44 mean validation CE 0.117282, population SD 0.001871. Saved checkpoint: best CE seed within this configuration (42), CE 0.115495.

Load with `load_phase_gate` and `load_phase_gate_preprocessor` from `diffusion_policy.model.vision.image_tau_phase_gate`, as documented in `model/civi_image_tau_phase_gate_ce_best/README.md`. Original data split has only three held-out episodes. Full report: `data/debug/20261120_phase_gate_ce_search_01/README.md`.
