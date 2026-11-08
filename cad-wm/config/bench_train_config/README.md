# LIBERO CAD-WM training

Run from cad-wm:

```bash
python -m train.trainer.cad_wm_train \
  --config config/bench_train_config/libero_drawer_wrench_h5.yaml

# One real-HDF5 CPU update, full-dataset checkpoint inference, and save.
python -m train.trainer.cad_wm_train --smoke-test --output-dir /tmp/cad_wm_smoke
```

## Data and timing

The config reads all 50 demos directly from the LIBERO HDF5. Inputs contain
the latest 10 native 20 Hz rows of obs/joint_states, obs/dq and obs/tau. The
first seven velocity/torque columns correspond to robot0_joint1 through
robot0_joint7. History ends at the current row. Short histories repeat their
oldest row. Windows stay within one demo. `train.validation_enabled: false`
uses every demo for training, normalization and the phase sampler. It overrides
`val_ratio`, explicit validation demo indices and K-fold splitting. There is
no validation loader or validation-based early stopping. To restore the 90/10
demo split, set `validation_enabled: true`. `--no-val` also disables validation.

The future wrench and phase targets are recorded rows t+1 through t+8, at
0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35 and 0.40 seconds after the anchor.
There is no timestamp interpolation or resampling. high_fps determines timing
and prediction_horizon controls the number of future rows.

## Checkpoint evaluation and selection

`train.checkpoint_eval.enabled: true` runs inference over the full original
dataset every `train.val_every` completed epochs and whenever a checkpoint is
saved, including the final checkpoint. `val_every` counts epochs and remains
active with `validation_enabled: false`. Evaluations at the same optimizer
step are reused, so an epoch evaluation and a save need only one full pass.
Each valid window appears once, with no contact oversampling or dropped final
batch. The currently saved inference weights (EMA when enabled) generate
future wrench from q/dq/tau and fixed per-anchor Gaussian noise. The full
configured Flow integration runs: by default eight Heun steps, or 16 velocity
network calls. No future target enters the prediction.

The score is the sum of squared differences between generated physical wrench
and recorded future wrench, divided by the total number of elements across
all windows, future rows and six channels. It is stored as
`full_dataset_wrench_mse`, alongside separate `full_dataset_force_mse` and
`full_dataset_moment_mse` values. `checkpoint_eval.batch_size` controls memory
use; `checkpoint_eval.seed` fixes sampling noise across checkpoint comparisons.

The lowest-scoring `train.top_k` checkpoints are retained. `latest.pt` always
updates, even when the new model does not enter Top K. Both the score and Top K
history are persisted for resume. The current config saves every 10000
optimizer steps and stops at 100000 steps. If checkpoint evaluation is disabled,
step saves retain the most recent checkpoints instead.

This is a training-set fit score, not a held-out estimate. A single fixed noise
sample does not assess the full conditional distribution. The combined physical
MSE mixes force and moment units, so also inspect their separate errors.

## Live PNG metrics

The existing trainer plotting code writes these files under `train.output_dir`
(default `outputs/cad_wm`):

- `loss_summary.png`: epoch train loss, full-dataset wrench MSE and optimizer
  step train loss in separate panels with independent scales.
- `loss_steps.png`: train loss recorded at every optimizer update.
- `full_dataset_mse_steps.png`: physical wrench MSE at actual evaluation steps.
- `loss_epoch.png`: mean train loss for each epoch.

`train.plot_every_steps: 100` refreshes the PNGs every 100 optimizer updates;
epoch completion and full-dataset evaluation also refresh them. Use a smaller
positive value for more frequent plotting, or 0 for epoch/evaluation updates
only. `val_every` controls full-dataset inference frequency, independently of
PNG refresh frequency. Before the first full evaluation, the MSE panel shows
a waiting message. Missing MSE measurements are not replaced with training
loss or repeated at unevaluated steps. PNGs are replaced atomically in place,
and checkpoints restore both curves' histories when training resumes.

## Dual-condition Flow and NEXT auxiliary task

q and dq are concatenated into 14 channels per row and passed through one
two-layer LSTM. A separate two-layer LSTM reads seven torque channels. Both
retain every temporal output. Tokens receive branch embeddings and learned
recency positions. Each Flow block reads the two memories through independent,
parallel cross-attention branches after future self-attention.

Training uses Gaussian-to-wrench conditional flow matching. Flow time is an
integration coordinate, independent of the 20 Hz physical timeline. Inference
encodes conditions once and integrates future wrench noise with eight Heun
steps by default. model.flow_solver and model.flow_inference_steps configure
the integration method and budget.

A two-layer MLP reads the last raw q/dq LSTM output to predict current tau-free.
Recorded current tau is its target only on complete, unpadded histories with
all-free phase labels and force norms no greater than 8 N. Pre-contact,
contact, padding and unconfirmed high-force streaks cannot qualify. Targets
use tau's Gaussian normalization. The auxiliary task updates only the motion
LSTM and MLP and is omitted at inference. No pretrained NEXT checkpoint is
required.

The loss is flow MSE + 0.1 * phase CE + 0.1 * tau-free MSE. The phase head
reads conditions alone, with no future target as input. Validation uses fixed
noise per anchor and reports physical errors from integrated predictions.

## Contact phases and strengthened sampling

Labels use raw sqrt(Fx²+Fy²+Fz²), with thresholds of 8 N for entry and exit.
Five consecutive native frames confirm a transition. Strict >8 enters,
strict <8 exits, and equality retains state and breaks the streak. Offline
labels backfill both transitions to the first frame of their confirmation
window: all five entry frames become contact, and all five exit frames become
free. Each demo starts free. Free rows in the one second before its first
backfilled contact onset become pre-contact: 0=free, 1=pre-contact, 2=contact.

Each training window takes the maximum phase in its future eight rows. The
replacement sampler weights are 1/5/5 for free/pre-contact/contact, using
training demos only. Validation windows are not oversampled.
train.contact_sampling.importance_correction defaults to false, intentionally
emphasizing strengthened phases; true restores the original window risk.

Five native frames represent 0.25 s of confirmation at 20 Hz. At another raw
rate, matching frame counts does not match the physical debounce duration.
DP also aggregates wrench windows for anchor sampling; CAD-WM's per-frame
phase supervision is distinct from that anchor selection rule.

## DP inference

```python
from model.wrench_predictor import WrenchPredictor

wm = WrenchPredictor("outputs/cad_wm/checkpoints/latest.pt", device="cuda:0")
result = wm({"q": q_history, "dq": dq_history, "tau": tau_history})
future_wrench = result["future_wrench"]  # [B,8,6], Fx/Fy/Fz/Mx/My/Mz, N and N.m
```

The predictor restores EMA weights and normalization, then returns physical
sensor-frame wrench. DP applies its own normalizer. Each call draws fresh
Gaussian noise; pass source_noise for a reproducible sample. History buffers
reset per episode. For [B,n_obs_steps,8,6], retain predictions made at each
observation anchor and stack them along that axis.

A native 20 Hz DP dataset should use eight future wrench rows with
future_wrench_start_offset: 1, future_wrench_stride: 1 and, where needed,
offsets_sec: [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40]. DP's separate runtime
needs to call this predictor. Trained timestamps, channel order and sensor
frame must match CAD-WM.

This recording's tau comes from MuJoCo inverse dynamics. Its velocity and
acceleration use centered differences, so recorded dq/tau can depend on later
rows. Online evaluation needs causal input definitions and units matching
training. Free-space recorded tau is a proxy for tau-free, rather than a
separately measured physical ground truth.
