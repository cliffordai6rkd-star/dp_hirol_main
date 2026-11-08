# CAD-WM

CAD-WM generates future end-effector wrench from joint position, velocity and
torque histories. This repository contains one world model and the standalone
NEXT free-space torque estimator.

## Architecture

```text
q + dq -> 2-layer motion LSTM -> motion tokens -----------------+
                   |                                         |
                   +-> 2-layer MLP -> current tau-free          |
                                                             v
tau ----> 2-layer torque LSTM -> torque tokens -> dual-condition Flow
                                                             |
Gaussian future wrench noise --------------------------------+
                                                             v
                                                 future wrench [B,8,6]
```

Each Flow block applies future self-attention, independent motion and torque
cross-attention branches, sums their residual updates, then applies an FFN.
Branch embeddings and learned history/future positions identify tokens; a
continuous flow-time embedding conditions the noise-to-wrench transport.
Inference uses Euler or Heun integration. A separate condition-only head
predicts future free/pre-contact/contact phases.

The NEXT auxiliary head reads the current raw motion LSTM output through two
linear layers with a SiLU activation between them. It learns current tau only
on complete, unpadded, all-free histories whose force stays below the free
threshold. Recorded free-space torque serves as the tau-free target. The
auxiliary loss updates the motion LSTM and MLP, and inference skips this head.

```text
loss = flow matching MSE + 0.1 * future phase CE + 0.1 * current free-space tau MSE
```

## Training

Run from this directory:

```bash
python -m train.trainer.cad_wm_train \
  --config config/bench_train_config/libero_drawer_wrench_h5.yaml
```

The LIBERO config directly loads the specified HDF5 `data/demo_*` groups.
Inputs and outputs both use native 20 Hz rows: the latest 10 history rows
predict wrench at t+1 through t+8, corresponding to 0.05–0.40 s. Targets are
read directly without timestamp interpolation or resampling. By default,
`train.validation_enabled: false` uses all demos for training and normalization.
Set it to true to enable the configured 90/10 demo split. `--no-val` overrides
validation settings on the command line.

Contact labels use raw force XYZ L2 norm with thresholds of 8 N for entry
and exit and five consecutive frames. Strict >8 enters and strict <8 exits;
equality holds the state and resets the streak. Offline labels backfill each
confirmed entry/exit window to its first frame. The one second preceding the
first backfilled contact onset is pre-contact. Training
windows are sampled with free/pre-contact/contact weights 1/5/5, using the
maximum existing phase in the future window. Validation is not oversampled.
Importance correction is disabled by default to emphasize pre-contact and
contact; it can be enabled explicitly.

Checkpoints save EMA inference weights, raw training weights, normalization,
the CAD-WM architecture/data contract and sampling settings. Incompatible
contracts are rejected. With `train.checkpoint_eval.enabled: true`, every
scheduled save (including the final save) runs full Flow inference over every
dataset window exactly once, without contact oversampling. Inference uses the
saved EMA weights when enabled, eight Heun steps, and fixed noise per anchor.
Predictions are denormalized and compared with recorded future wrench:

```text
full_dataset_wrench_mse = sum((predicted_wrench - recorded_wrench)^2) / (windows * 8 * 6)
```

The lowest MSE checkpoints are retained up to `train.top_k` (three by default).
`latest.pt` always stores the latest evaluated state, even if it is outside
Top K, and restores the ranking on resume. Force MSE and moment MSE are also
recorded separately. This score measures training-set fit; combining N and
N.m in one MSE lets the numerically larger channels dominate. It does not
measure generalization or the full quality of the Flow distribution.
Disabling checkpoint evaluation restores retention of the most recent step
checkpoints. Optional validation uses fixed source noise per anchor and
reports physical wrench error from integrated predictions.

Full-dataset MSE is also evaluated every `train.val_every` completed epochs,
even when validation splitting is disabled. Checkpoint saves still require a
score for their current weights; evaluations at the same optimizer step are
reused. `val_every` counts epochs, not optimizer steps.

The existing Matplotlib plotting code updates PNGs in `train.output_dir`:
`loss_summary.png` shows epoch train loss, full-dataset wrench MSE, and step
train loss in separate panels. `loss_steps.png` and
`full_dataset_mse_steps.png` provide individual curves. Train loss is recorded
at every optimizer update, averaging the microbatches in an accumulated update.
PNGs refresh every `train.plot_every_steps` optimizer steps (100 by default),
at epoch end, and after full-dataset evaluation. Setting it to 0 keeps only
epoch/evaluation refreshes. Files are replaced atomically so viewers can
refresh them while training runs. Both step and inference histories are
stored in checkpoints and restored on resume.

```bash
# One real-HDF5 CPU update, full-dataset checkpoint inference, and save.
python -m train.trainer.cad_wm_train --smoke-test --output-dir /tmp/cad_wm_smoke

# Resume training.
python -m train.trainer.cad_wm_train --resume outputs/cad_wm/checkpoints/latest.pt
```

## DP inference

```python
from model.wrench_predictor import WrenchPredictor

wm = WrenchPredictor("outputs/cad_wm/checkpoints/latest.pt", device="cuda:0")
result = wm({"q": q_history, "dq": dq_history, "tau": tau_history})
future_wrench = result["future_wrench"]  # [B,8,6], N and N.m
```

Inputs are physical-unit [B,T,7] histories at 20 Hz, ending at the current row.
Outputs are denormalized into the original sensor frame. DP applies its own
normalizer and must use matching future timestamps. Pass source_noise=[B,8,6]
to reproduce a sample; otherwise each call draws fresh Gaussian noise. See
[the training config documentation](config/bench_train_config/README.md) for
timing, phase and data-definition details.

## Standalone NEXT

The existing sequence torque estimator, its LSTM/GRU/TCN options, physics
losses, normalization and inference tools are retained. Its canonical entry
has no numbered suffix:

```bash
python -m train.trainer.tau_free_sequence_train \
  --config config/train_cfg/tau_free_sequence/tau_free_sequence.yaml
```

## Environment and tests

setup.py declares the original Python 3.10/3.11 training dependencies and
optional physics, cache and vision extras. setup.sh prepares the pinn Conda
environment and CUDA PyTorch wheels:

```bash
bash setup.sh
conda activate pinn
python -m pytest -q
```

Installation exposes cad-wm-train and next-train. Shared torque estimation,
contact-gating utilities, data conversion and physics components remain under
data_process, model and physics.
