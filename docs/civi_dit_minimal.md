# Direct-torque CIVI RGB → frozen DiT

This is a new objective and new initialization, based on `d92d1a6`. Historical
response experiments/configurations/checkpoints remain intact. The runnable
configuration is `diffusion_policy/config/train_civi_student_dit_minimal.yaml`.
It does not inherit the historical response configuration or its 180k teacher.

## Data and fixed dependencies

Teacher and student both use
`data/board_25hz_3cam_lerobotv3_clean_ma5_l1_12p5`, not the huanggua data.
The exact teacher is
`data/outputs/2026.11.15/final/teacher/checkpoints/optimizer_step=00150000-val_action_mse=0.00514941.ckpt`.
File SHA256: `33d59938df0092a010a37d20076b1512c6a284f0dd0197460b6f94a8b9d266fa`.
It loads **ema_model**, step150000, tensor-state hash
`2ea7da45c0fe179994c610e8742fbf9582a6e6247dab8c5469b71b00714e76bd`.
The historical pinned EMA container has a different file hash but the identical
parameter-state hash; frozen Adapter/Condition metadata refers to that container.
Both identities are checked, never selected by latest/best filename heuristics.

The matched residual GELU MLP Adapter256→128→256 and Condition snapshot are
strictly loaded and file-hashed. Their existing provenance is retained under
`data/diagnostics/civi_student_iteration/final150k_match/matched/`.
The 56 saved student training episodes contain17740 windows
(F13419/P740/C3581). Validation has1893 windows (F1317/P108/C468).
Internal selection episodes are38/44/53/60. Episodes5/26 are retained reporting
only; they were involved in historical teacher checkpoint evaluation and are not
an untouched final test set. Original saved sampler-index hashes and phase counts
are asserted. The historical uint8 TorchCodec CUDA RGB mmap is pixel-audited;
there is no silent PyAV or dataset switch.

## Training and deployment

Two real causal RGB observations (`side`, then `wrist`) and, for each observation,
its trailing eight raw-unit7D torque samples enter the exact teacher wrench
preprocessor. A base CNN pyramid and skips feed6 image SA blocks, then2 residual
image-query CA blocks whose K/V are **eight direct LSTM tokens**, and an ordinary
3-level RGB decoder/head (patch8, embedding128). This fixed depth is a configuration,
not an experimentally established optimum. Trainable base weights are fresh
(seed42), except the explicitly audited same-teacher direct torque path.
The step6000 response-conditioned renderer is not imported.

The frozen Condition predicts spatial region and alpha. RGB uses historical hard
parameterization: tanh in renderer compute dtype, FP32 residual×gate×alpha,
clamp to ±0.1×gate×alpha, then RGB[0,1]. Alpha0 is exactly identity; physical-zero
input is evaluated separately and need not predict alpha0. No Decoder FiLM,
extra Spatial Modulation, response predictor/projection, predicted-patch K/V,
Oracle, Jacobian generator, or deployment DINO scene branch is enabled.

Stages1/2 reuse hash-verified calibration snapshots. The existing staged workspace
retains Adapter-only and Condition-only pretraining; a new/missing dependency must
be calibrated explicitly there, not silently trained inside stage3. Only the base
RGB renderer is optimized. Teacher/DINO/projection/embeddings/context_norm/Adapter/
Condition/DiT remain frozen and eval; the student visual/DiT path retains input
gradients. Every training microbatch samples new rho and epsilon shared by
teacher/student. KD is an FP32 valid-element MSE between DiT predictions. All
Token/response/direction/feature/demo auxiliary weights are zero.

Checkpoint semantics were inspected: epsilon prediction, joint-position actions,
non-relative horizon9, action_dim7, execution start1 and chunk8, native DDIM8.
Dataset padding follows the existing sampler; it does not emit valid_mask and
native teacher compute_loss rejects one. The shared MSE helper supports an
explicit valid-element mask if a future compatible dataset supplies it.
Single-step denoiser KD is never labelled sampled action MSE. Native complete
sampling is validation only; no sampling chain is differentiated. Time token
enters condition_encoder with context, so its memory is **not cached across rho**.

## Launch and exact recovery

Environment on the four-4090 server:

```bash
cd /home/djhpc/code/dp_hirol_main/data/diagnostics/civi_dit_minimal_v1/source
export PYTHONPATH=.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LD_LIBRARY_PATH=/home/djhpc/miniconda3/envs/fdp/lib/python3.10/site-packages/nvidia/npp/lib:/home/djhpc/miniconda3/envs/fdp/lib:/home/djhpc/miniconda3/envs/fdp/lib/python3.10/site-packages/torch/lib
PY=/home/djhpc/miniconda3/envs/fdp/bin/python
```

Single GPU, using the entire saved training dataset and full internal/retained
validation (default80-step bounded smoke):

```bash
CUDA_VISIBLE_DEVICES=2 "$PY" train_distill_render_net.py \
  --config-name train_civi_student_dit_minimal \
  output_dir=data/outputs/civi_dit_minimal_smoke_single
```

Two GPUs, same global Mixed batch8, microbatch2 per rank, accumulation2:

```bash
CUDA_VISIBLE_DEVICES=2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
  train_distill_render_net.py --config-name train_civi_student_dit_minimal \
  output_dir=data/outputs/civi_dit_minimal_smoke_ddp
```

Four GPUs use `CUDA_VISIBLE_DEVICES=0,1,2,3` and `--nproc_per_node=4`, same global8,
microbatch2, accumulation1. Use only allocated/idle devices. GPU0 was occupied
when engineering checks began; no process is paused, killed or preempted to
obtain four devices. The same code supports1/2/4; measured evidence identifies
the actual tested world sizes.

Exact resume restores the **online** model with its own optimizer, EMA weights
and age, scaler, scheduler (explicitly none), per-rank Python/NumPy/Torch/CUDA RNG,
committed phase/episode/window queues, next batch position, selection and stop
state. It requires unchanged architecture, dependencies, data identity, precision,
optimizer and global-batch semantics, and the same world size for exact RNG replay.

```bash
CUDA_VISIBLE_DEVICES=2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
  train_distill_render_net.py --config-name train_civi_student_dit_minimal \
  output_dir=RUN training.resume_from=RUN/latest.pt training.max_steps=80
```

Explicit compatible warm start copies online or EMA weights and creates a fresh
optimizer, EMA age, RNG/queue plan and selection record. It cannot import a
response-conditioned architecture based solely on matching tensor shapes.

```bash
CUDA_VISIBLE_DEVICES=2 "$PY" train_distill_render_net.py \
  --config-name train_civi_student_dit_minimal output_dir=NEW_RUN \
  training.warm_start_from=OLD_RUN/latest.pt training.warm_start_state=online
```

The default task ledger limits **all** benchmark/replay/smoke optimizer calls
combined to100 and checks the original exploration ledger (19622/20000 before
this task). There is no automatic100k run. The user subsequently authorized100 epochs, early stopping patience5, and
global batch64. The dedicated formal configuration is
`train_civi_student_dit_minimal_100epoch.yaml`. This is a separate authorized
budget and does not alter the historical20k exploration ceiling or counts.

```bash
CUDA_VISIBLE_DEVICES=2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
  train_distill_render_net.py --config-name train_civi_student_dit_minimal_100epoch \
  output_dir=data/outputs/civi_dit_minimal_b64_s42_100epoch
```

Per-card Renderer microbatch8 × accumulation4 ×2 ranks = global64, with
F16/P16/C32. Visual readout remains fixed B2 chunks with input gradients; no
rendered features are cached. One nominal epoch is ceil17740/64=278 updates,
not a promise of visiting each imbalanced window exactly once. Maximum100 epochs
is27800 updates/1779200 sample exposures, including889600 Contact exposures
(~248.42 nominal Contact dataset passes). Validation and complete native action
sampling run each epoch; Contact KD episode macro chooses best, patience5 counts
five completed epochs without a new best (step0 establishes the reference).
A fresh Renderer is initialized for the formal run; it does not continue the80-step
smoke or import response-conditioned weights. Learning rate is unchanged; batch
exposure/update count differences are explicit.

Resume uses the same formal config, same two ranks, and
`training.resume_from=RUN/latest.pt output_dir=RUN`. Its full-data identity,
optimizer and direct-torque architecture guards stay active.

## Validation and outputs

Validation probes derive rho/epsilon from sample_id+probe_id+seed, not batch/rank
or order. FP32/TF32-off reference uses fixed encode batch2 and episode macro means;
Contact KD on internal episodes selects the model. Free and Precontact are always
reported. Complete native8-step action sampling, same-Adapter original RGB
baseline, and fixed donor-torque pairing are recorded at the configured cadence.
Historical BF16 feature curves are not joined to this new FP32 KD curve.

Each run includes config/teacher semantics, dependency/data/init audits, cache
identity and shuffled-read audit, batch semantics, train/update journals,
validation per-sample/per-episode/summary JSON, RGB comparison PNGs, CUDA component
profile, resumable initial/latest/best states and lightweight `civi_rgb.pt`.
Smoke passing demonstrates runnable implementation, not successful distillation
or robot policy success, and does not lower previous acceptance thresholds.

Static caches are built by rank0, others reopen read-only. Keys include teacher,
Normalizer, Condition and Adapter hashes, sample index/split, RGB hashes, camera/
time layout, FP32 dtype and fixed encode shape. Rendered visual features and DiT
outputs are recomputed. Raw uint8 RGB mmap avoids rank/worker image-array copies.
Spawn workers reopen maps rather than serializing them. Compute uses FP32 master
weights/loss, Renderer BF16, frozen visual/DiT FP32; only measured, numerically
checked compile options are enabled. DDP synchronizes trainable renderer params,
with no_sync on accumulated microbatches. Foreach AdamW/EMA is checked against the
existing update order. No activation checkpointing or CUDA prefetch stream is
added because observed memory/data wait does not justify them.

## Lightweight RGB inference

`civi_rgb.pt` contains torque preprocessing/Normalizer, base Renderer and frozen
Condition only. It has no Teacher, DiT, DINO, training MLP Adapter, Predictor,
Oracle, offline phase/alpha or sample-ID input. Condition is recomputed from
current causal inputs. It must receive two **different actual observations**;
do not duplicate the current frame to fake history.

```python
from diffusion_policy.policy.civi_rgb import CiviRGB
model = CiviRGB.load('RUN/civi_rgb.pt', 'cuda:0')
enhanced = model({'side': side_history, 'wrist': wrist_history}, raw_tau_history)
# side/wrist [B,2,3,224,224] float32 in[0,1]; raw_tau [B,2,8,7].
```

For recorded causal windows:

```bash
"$PY" -m diffusion_policy.scripts.civi_rgb_inference \
  --model RUN/civi_rgb.pt --input causal_window.pt --output enhanced.pt --device cuda:0
```

The training/validation enhancedRGB→DiT path is separate and retains the same
frozen visual readout and MLP Adapter. It must not be confused with RGB export.

## Verification commands

```bash
python -m pytest -q tests/test_civi_dit_minimal.py test/test_staged_frozen_dit.py \
  tests/test_render_net_denoiser_distillation.py
python -m diffusion_policy.scripts.benchmark_civi_dit --output BENCH \
  --prepare-updates 2 --iterations 3 --compile
python -m diffusion_policy.scripts.verify_civi_dit_runtime --output VERIFY \
  --weights BENCH/benchmark_weights.pt
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m diffusion_policy.scripts.verify_civi_dit_runtime --ddp --output DDP_VERIFY \
  --weights BENCH/benchmark_weights.pt
```

GPU benchmark/replay updates are charged by the same shared ledger as training.
Do not rerun them without checking the remaining task budget. Measured numbers,
known pre-existing stale tests, and smoke results are under
`docs/reports/civi_dit_minimal_v1/`.
