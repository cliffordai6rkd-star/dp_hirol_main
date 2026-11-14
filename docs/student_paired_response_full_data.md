# Full-data paired-response experiment

This independent experiment uses the unchanged `student_dual_path_c_film_mlp`
architecture at V commit `6482fe3`. It imports the existing renderer and uses
`review_1000/c_mlp_checkpoint_1000.ckpt` for both A and B. Its MLP128 Adapter is
verified against `calibration_mlp_cuda/adapter.pt`. Only `rgb_parameters()` train;
Teacher (including DINO, Context Norm and DiT), Adapter and Condition Network freeze.

The requested teacher is the **final optimizer step 150000 EMA** from
`data/outputs/2026.11.15/03.31.29_teacher_ca_curriculum_b64_150k/teacher`.
`launch_paired_response_after_teacher.py` waits for this actual checkpoint step;
it does not substitute the current latest or best checkpoint. Since this teacher
differs from the historical teacher, its paired target cache must be built once.
Existing decoded image caches and split identities are reused. A/B share the new
cache; the old paired cache and all production files/checkpoints remain untouched.
The historical calibrated Adapter is intentionally retained, without refitting to
the new teacher. Step-zero metrics reveal any baseline mismatch this causes.

## Objective and paired paths

- A: Global Token MSE + original response-weighted MSE, unchanged.
- B: A + lambda_delta times full feature response-difference MSE + L0 + Lfree.
- Physical-zero torque is normalized by the frozen teacher's original normalizer.
  Each student branch independently runs the deployed Condition Network and renderer.
  No training branch forces alpha=0 or substitutes cached visual tokens.
- Explicit alpha=0 is exercised separately to assert pixel-exact RGB identity.
- Dropout is disabled; paired forwards use native deterministic eval LSTM backward.
- Delta weights are detached teacher response RMS, filtered by a TRAIN-Free response
  noise quantile, and divided by the sum of effective weights over the full batch.
  The threshold is never selected using validation.
- L0 (Contact) and Lfree (Free) use actual deployed zero-torque student contexts.
  Their initial RGB gradients and predicted zero alpha are reported. If the real
  gate kills these gradients, those losses remain measurements, not trainable protection.
  The Adapter remains frozen in either case. The baseline A global loss also covers Free.

## Budget, sampling and outputs

The 17740 training windows contain Free=13419, Precontact=740, Contact=3581.
The independent 1893-window validation set contains Free=1317, Precontact=108,
Contact=468. Every training row is visited once per sampler epoch. Within each
phase, episodes are interleaved; phase frequencies match the full dataset.
A/B use identical sampling epochs and minibatches. Batch16, microbatch4, seed42,
RGB lr=0.25 times the source optimizer lr (5e-5), other AdamW settings unchanged.
Microbatch denominators are calculated over the entire batch.

First run 2000 updates and validate every 200. Extend **both** to 5000 only if
neither arm's final validation Contact Token MSE, response-difference MSE or Free
zero-token MSE is worse than its step-zero value. Otherwise stop both at 2000.
Save online/EMA states, optimizer and EMA state in `latest.ckpt`, and select
`best.ckpt` by validation Contact Token MSE, including the step-zero checkpoint.

Full training and full validation metrics are computed at endpoints. Intermediate
train progress uses a fixed all-episode, phase-stratified probe, explicitly labeled
`train_probe`. Reports include phase and episode metrics, full-data generalization
gaps, cache sample/image-hash audits, frozen-state hashes, sampling hashes, loss and
module gradients, contact response maps and RGB previews. `comparison.json` is a
machine-readable artifact; the user-facing final A/B and episode tables are given
directly in the conversation. This does not launch a 150k-step student run.

Example on the GPU server, from an isolated source snapshot:

```bash
python test/run_paired_response_full_suite.py \
  --root /path/to/existing/student_dual_path_ablation \
  --experiment /path/to/new/paired_response_full_final_teacher --gpus 2 3
```

Before this command, the experiment's `inputs/` must contain the pinned final
teacher and the existing reference split/student input snapshots. The local
`launch_paired_response_after_teacher.py` performs that setup and launches the
suite after the teacher ends. Use `--check-only` on the individual runner to audit
paths and calibrate gradients without updating parameters. `--smoke --steps 1`
uses tiny evaluation probes and is labeled nonformal in every manifest.
