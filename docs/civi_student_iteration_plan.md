# CIVI shared RGB student: controlled iteration

Started 2026-11-15 on V, HEAD `009c350bd8676f9cbe4891cd0f0f582be31b1df0`.
Preserve all pre-existing untracked files, production models/configurations and checkpoints.

## Teacher selection

Latest user steering: diagnostic tests may use any saved checkpoint in
`data/outputs/2026.11.15/03.31.29_teacher_ca_curriculum_b64_150k/teacher/checkpoints`.
Continue the matched 70,750-step experiment first, so changing targets does not confound renderer tests.
Final distillation MUST use the exact 150,000-step EMA from that directory. Do not substitute
`best.ckpt` or a still-changing `latest.ckpt`. Pin the final tensors and SHA256 after checking the step.
A teacher change starts a separate experiment: rebuild paired targets, recalibrate the MLP,
retrain/match the frozen response predictor and audit the condition network. Old response targets
must never be reused under a new teacher hash.

## Frozen diagnostic dependencies

- Teacher EMA: `new_teacher70750_adapter_shared_response/inputs/pinned_teacher_ema.ckpt`, SHA256 `e030fcec4f36efc71d0b069dcd862695cbae608bac04d3b31736bab8ada8c87a`.
- Predictor: `scene_torque_response_predictor_quick/best.pt`, SHA256 `e6c5b5ed1488af77ee03567b041d8d907a260fe2d614ebcbe7e46eee4cf31417` (118,912 parameters).
- Adapter: `new_teacher70750_adapter_shared_response/adapter/best_adapter.pt`, SHA256 `3aab75c387fadb3a63fbcdb08993791aba7b776ff65d56bee13817d97b979913`.
- D initialization: `new_teacher70750_adapter_shared_response/d/best.pt`, SHA256 `d5fba6c1494c3b853e83882fa259a4dba737d3c08209b180c7ab82eca0cbe244`.
- Condition state SHA256 `54db83df53aab3bd4e130e27f45cce5743fa090cf5015b703bc3cfcef092fbae`.
- DINO, projection/embeddings, Context Norm, Teacher/DiT, Predictor, Adapter and Condition Network remain frozen.
Paths above are relative to `data/diagnostics/2026.11.15` on the GPU server.

## Data and evaluation fixed before new RGB updates

Full train: 56 episodes / 17,740 windows (Free 13,419; Precontact 740; Contact 3,581).
Validation: 1,893 windows (Free 1,317; Precontact 108; Contact 468).
Internal selection episodes: 38,44,53,60. Episodes 5,26 are repeatedly viewed retained evaluation
subsets, including teacher checkpoint selection; they are not untouched final test data.
No validation examples or validation Oracle RGB may train any module.

First shared-fit panel: 24 Contact windows from 24 evenly spread training episodes, plus 8 Free
windows from those episodes. Select by episode/index/phase only, before inspecting losses.
Train Oracle uses exactly these Contact windows, targets, support and epsilon; its updates are
reported separately from shared RGB optimizer updates.
Interim validation every 200 updates, including zero: fixed 8 Contact / 4 Free / 4 Precontact
windows per internal episode, uniformly spaced by index. Endpoint validation uses all original
validation windows. Train-fit and validation metrics are reported separately for online and EMA.
Expansion uses all training episodes with balanced episode/phase queues; never merges validation.

Readout E is frozen DINO + projection/embeddings + calibrated frozen Adapter + context_norm.
dT = Teacher(real raw torque) - Teacher(physical raw zero torque), identical clear-image preprocessing.
dP = frozen Predictor output; no target alpha rescaling. dS = E(rendered RGB) - E(original RGB),
online at the same encode batch shape. All-patch MSE denominators are fixed.
All cosine/RMS-ratio comparisons use the SAME fixed teacher-active patch mask (existing train-Free
teacher noise threshold); report the threshold and counts. No student mask/alpha metric selection.
Report predictor / realization / true-teacher errors separately, Contact absolute token MSE, per
episode and macro means, Free actual changes, RGB RMS/max and clipping. These three errors are not
an additive error decomposition. Zero-torque deployment recomputes Condition/Predictor; additionally
test forced alpha=0 identity. Frozen-adapter original-image L0 is monitoring only.

## Experiment order and predeclared decisions

1. Audit old initialization and 1,000-step EMA: hard-cap/range clipping fractions, output Jacobian
   support, alpha/FiLM/input/output scaling, layout and CA distribution, freeze/pair consistency.
2. If hard-cap saturation is substantial (>10% of allowed RGB channels), first paired test changes
   only hard-cap parameterization to `epsilon * support * tanh(logits)`; compare to original hard cap
   from identical tensor initialization, optimizer, batches and budget. Same epsilon/support/range.
   The smooth version changes local gain; report this as part of the tested parameterization.
3. Otherwise prioritize input/FiLM alpha removal, retaining identical final support. Do not combine
   this change with clipping or CA changes in the first comparison.
4. First candidates: 600 updates each, seed42; train fit must beat identity by >=10% realization MSE
   and improve internal validation vs its step0 before extension. A useful paired change should
   beat the matched control by >=5% realization MSE without worsening true-teacher response MSE.
5. Only extend candidates improving train fit and internal validation to 3,000–5,000. Repeat a
   winning candidate with seed43. After two consecutive unsuccessful reproducibility rounds on a
   route, stop minor tuning and choose the next most informative test from the user-approved scope.

## Immutable promotion gates / budget

Exploration cumulative RGB optimizer updates <=20,000, including shared-network repeats, new
smoke updates and pixel-Oracle updates. Track shared and per-image optimizer counts separately,
but include both in this conservative budget. Historical runs are reference rows, not silently
counted as new execution.
At most two candidates; use only assigned remote GPUs2/3. Preserve the running local teacher and
all unrelated jobs. Disable only this task's superseded post-teacher experiment queue if needed.

Promotion to full-data distillation requires ALL of:

1. Full internal-validation Contact realization MSE <=75% of identity energy.
2. True-teacher end-to-end response MSE <=90% of identity energy AND better than the old shared decoder.
3. Improvements across episodes, repeated with at least two seeds.
4. Free actual response MSE <=105% of same-protocol step0 (zero baseline requires an absolute tolerance explicitly reported, not a ratio invented after results).
5. Identical per-channel epsilon=0.1*support; overall RGB RMS <=0.015, with saved image comparisons.
6. Correct response condition beats deterministic shuffled and zero conditions.
7. Improvements are measured through enhanced RGB, with deployable causal inputs only.

No gates are relaxed after results. Passing RGB/token checks does not demonstrate downstream
DP/ACT/pi0 success or untouched-data generalization.

After passing and matching to final150k teacher, freeze structure/loss/preprocessing/split/gating;
run only one final candidate on full training episodes, <=100,000 RGB updates, early stopping,
best/latest/optimizer recovery, with internal selection metrics only. If matching to the changed
teacher fails the gates, do not call old-teacher success final-teacher validation.

## Persistence and deliverables

`data/diagnostics/civi_student_iteration/state.json` stores stage, budget, task processes, next
actions and recovery commands; `experiments.csv` is the result ledger. Save actual executed
numbers, manifests, checkpoint/file hashes, per-phase/per-episode/per-seed metrics, plots, RGB
examples, latency/memory and inference dependencies. Summarize findings directly to the user.

## Recorded decisions before round3

Rounds1/2 show <1% change from removing input/FiLM alpha or CA response reading; do not keep
tuning this route. Training-only Oracle24 realizes MSE0.00797979/cos0.70464 under unchanged
support, while shared D train-fit MSE is0.0143/cos0.43. Precision audit shows a BF16 low-amplitude
response floor (RGB RMS2.84e-7 -> response MSE4.84e-5; FP32 3.62e-11), with pixel-gradient cosine
0.9643. Round3 tests two independent changes against the already executed baseline600:

- `fp32_visual`: exactly the original D architecture/initial tensors/loss, FP32 only for the frozen
  visual forward/backward used by training. Evaluation remains the identical BF16 protocol for
  ALL candidates; no precision-dependent metric is substituted to claim success.
- `patch_rgb`: a simpler shared RGB implementation (Conv16-stride16 scene embedding, response
  projection, GELU pixel-block head). It emits 16x16 RGB blocks on the audited14x14 grid, providing
  pixel-phase degrees of freedom missing from a purely convolutional upsampling head. It subtracts
  the same scene's zero-response head output before the tanh; no per-image parameters or IDs.
  Only final RGB uses the same0.1*spatial*alpha support. Small random output initialization,
  seed42, same batch indices/lr/loss/600steps. This tests RGB realization architecture, not teacher,
  loss, target, mask or Adapter changes. No validation Oracle supervision.

Both start with a full step0 validation to make Free/episode baseline comparisons exact. Existing
round1/2 panel metrics remain labeled as panels, distinct from full endpoint results. Neither change
is presumed effective; the original promotion gates still apply. No composite `FP32+patch` candidate
is launched before independent evidence. Train Oracle pixel-optimizer iterations are reported
separately (24 images in12 batches x300), and counted in the cumulative exploration budget.

## Round4/5 predeclared follow-ups

Round4 reuses the existing full-data1000 model/EMA/AdamW state and consumes the next600 deterministic
batches; it is a budget-continuation check (historical train AND internal validation improved fromstep0),
not a new structure. A separate training-only Oracle initialization uses400 RGB pixel-supervision
updates then400 unchanged feature-response updates; this budget includes warm-up optimizer updates.
No validation Oracle is consumed. Both are below promotion gates at completion.

Round5 changes only the patch generator's scene representation: use the frozen DINO/projection/embedding
scene tokens already computed for Predictor, instead of the learned strided RGB scene encoder.
The same spatial response/pixel head, bound, loss,24Contact8Free panel,lr and600updates remain.
Frozen cached scene tokens come from the exact audited same-image DINO pipeline; deployment must
compute them online and cannot read the cache/index. This is explicitly a deployment dependency.
Compare to round3 patch_rgb (samebudget), not as an isolated claim against a different target.
If it fails, stop patch-head microtuning; do not extend failed candidates to force budget expenditure.

## Numerical-path correction before final-teacher RGB

The early diagnostic wrapper cast output logits to FP32 before tanh, whereas the formal wrapper
applies tanh to BF16 logits then casts RGB arithmetic to FP32. All early controlled arms shared
that setting, but they must NOT be described as bit-exact replay of the original AMP renderer.
Their source/config/result snapshots are retained. Restored original tanh order and added both
FP32 and BF16-CPU-autocast exact-forward tests (passed). Final150k experiments use the corrected
path; old/new numerical revisions are not pooled to claim a gain. The in-progress70750 teacher-target
loss check uses its snapshotted earlier path, matched to the earlier predictor-target reference.
No formal model/config/checkpoint was modified.

A final70750 loss-only control (1000steps/fulltrain) changes dP supervision to true dT, preserving
network/initialization/batches/frozen modules and evaluating both realization and Teacher errors.
It tests target bias, not another architecture micro-adjustment. Reserve1000 additional RGB updates;
continue to count all exploration across teacher versions against the same20k limit.

## Final150k short distillation: predeclared loss-only comparison

The exact requested150000-step checkpoint completed naturally. Source checkpoint SHA256
`33d59938df0092a010a37d20076b1512c6a284f0dd0197460b6f94a8b9d266fa`, pinned EMA
`9221fef56a297c4fd8d37bac6491f73ae77e161d4f034c7f13fbf90f76ede799`.
Fresh paired targets, Adapter, Predictor and Condition snapshots are recorded in
`data/diagnostics/civi_student_iteration/final150k_match/dependencies.json`; the old teacher's
targets/Adapter/Condition are not reused. Physical-zero/cache/freeze/actual RGB-gradient smoke
passed on six samples (both splits and all three phases), without optimizer updates.

Two independent1000-update runs use all56 training episodes, seed42, identical batch plan,
corrected original BF16-tanh numeric path, matched frozen snapshots and identical RGB warm-start
from `full_continue_s42_600/best.pt` (RGB parameters only). Configuration:
`test/config/civi_final150k_screen.yaml`. A (`baseline`) optimizes Predictor full response;
B (`teacher_response`) changes ONLY the supervised response vector to true Teacher full response.
Free real-deployment forward is trained toward zero response in both. No L0 gradient is claimed.
Keep lr2e-4, epsilon0.1, final support, loss weights and budget identical. Evaluate every200 updates
including zero, all validation windows at endpoints, fixed internal panels at intermediate points.

Hypothesis: Predictor target bias may limit true Teacher transfer under the new target; direct
Teacher response may improve it without new architecture. A/B is compared within the new teacher,
never by pooling old/new target energies or AMP revisions. Extension requires BOTH training-probe
fit and internal validation improvement; loss B is useful only if true Teacher error improves and
realization/Free limits remain satisfied. The unchanged promotion gates require two seeds before
formal training. No candidate has qualified yet. Reserve2000 RGB updates; the completed70750
loss-only1000 test brings prior shared exploration to6002 updates, so this pair ends at8002 shared
updates if completed. Including3600 train-Oracle pixel updates, cumulative RGB optimization is
11602 at that point; the20000 ceiling applies to this inclusive total.

Conditional follow-up, specified before the1000-step endpoints: if A's training-probe realization
beats identity by at least10%, its full internal-validation realization AND Teacher response MSE
improve from step0, and Free stays within5%, continue the same A optimizer/EMA/batch stream to
3000 total steps (2000 additional). Also repeat the initial1000-step A protocol at seed43 from
the identical warm-start, to test reproducibility. Do not continue B unless it improves upon A
under the same protocol; do not extend either solely because training loss decreased. Stop this
budget-extension route if gains plateau or two-seed results remain below the immutable gates.
Resumption preserves initial full-validation references, frozen hashes, optimizer, RNG, and the
verified sampler prefix. These follow-ups would bring the inclusive RGB budget to14602 updates.

## Reproduction and deployment dependency

The assigned server is `100.97.6.30`; Python is
`/home/djhpc/miniconda3/envs/fdp/bin/python`. Execute in
`/home/djhpc/code/dp_hirol_main/data/diagnostics/2026.11.15/new_teacher_shared_response_source`
or the updated repository with its data/model paths available. Set `CUDA_VISIBLE_DEVICES=2`
or3 only after checking device availability. The server environment requires
`LD_LIBRARY_PATH=/home/djhpc/miniconda3/envs/fdp/lib/python3.10/site-packages/nvidia/npp/lib:/home/djhpc/miniconda3/envs/fdp/lib:/home/djhpc/miniconda3/envs/fdp/lib/python3.10/site-packages/torch/lib`.

```bash
civi_run=data/diagnostics/civi_student_iteration
civi_match=$civi_run/final150k_match
python test/run_civi_rgb_iteration.py \
  --base "$civi_match/matched" --predictor "$civi_match/predictor" \
  --cache "$civi_match/predictor_cache" --dependencies "$civi_match/dependencies.json" \
  --old data/diagnostics/2026.11.15/student_dual_path_ablation \
  --rgb-init "$civi_run/full_continue_s42_600/best.pt" \
  --config test/config/civi_final150k_screen.yaml \
  --variant baseline --scope full --steps 1000 --seed 42 \
  --output "$civi_run/reproduction_baseline_s42_1000"
```

Use `--variant teacher_response` for the loss-only B control, or `--seed 43` for the replicate.
Resume to3000 with the same arguments, a NEW output directory,
`--steps 3000 --resume <previous-output>/latest.pt`; it restores optimizer/EMA/RNG and verifies
the deterministic sampler prefix. Do not repeat the initial1000 updates when accounting budget.
Each run retains its effective configuration, source SHA256 snapshot, paired-index plan,
online/EMA checkpoints and logs. `best.pt` is selected only by the internal-validation panel;
endpoint reports explicitly evaluate the final-budget weights, so do not label them best weights
without separately evaluating that checkpoint.

The inference API is `test/civi_rgb_deploy.py:CIVISharedRGB`. It executes frozen DINO visual
features, the matched Predictor and Condition Network, and a shared renderer; Teacher fusion,
DiT and the evaluation Adapter are not executed to generate RGB. Current D inference uses the
two most recent causal RGB observations and their raw-unit torque histories. In particular,
the Condition Network contains a unidirectional alpha GRU across these observations, so the
previous RGB must be buffered; it cannot silently be replaced with the current frame. This is
an explicit deployment dependency, not evidence for a separately validated single-frame model.
`test/benchmark_civi_rgb_deploy.py --dependencies ... --checkpoint ...` measures this actual
causal RGB path. Benchmark process allocations also hold audit-reference teacher parameters;
active parameter bytes and additional peak allocations are reported separately.

## Closing diagnosis for the changed teacher

After the3000-step endpoint still fails the fixed gates (realization7.8%, Teacher response3.7%
reduction, Free8.7% worse), stop RGB network updates on this route. Before concluding this round,
evaluate the internally selected2800-step EMA with ZERO optimizer updates. Re-evaluation preserves
the original step0 reference and cannot change checkpoint selection.

The70750 Oracle cannot measure reachability of the new150k target. Run one new, training-only
Oracle on12 windows selected uniformly from the already fixed24-window train panel, before
examining any new Oracle results. Use the exact new pinned dependencies, same physical bounds,
same frozen visual readout and raw Predictor target.300 steps per image, two independent images
per batch, consumes1800 pixel optimizer updates; no shared RGB parameters are trained. Evaluate
the selected shared Decoder on these EXACT same images, same batch shape, same gates and same
readout, so the comparison measures realization capacity instead of different samples or targets.
No validation pixels or targets enter this optimization. Save both actual RGB responses and RGB
RMS/max; Oracle success still proves only per-image reachability. Inclusive cumulative budget is
16402 if this diagnosis completes; do not spend the remaining budget on failed microtuning.

## Executed outcome (2026-11-15)

No candidate qualified; formal100k training was not started. Teacher training completed naturally;
none of its processes/checkpoints were stopped or overwritten. Assigned server GPUs2/3 are idle.
Only isolated test modules/configs and this document were added; production student/teacher code
and historical checkpoint/configuration files were preserved.

Full internal-validation Contact episode macro averages (all372 Contact windows from episodes
38/44/53/60), final-budget EMA unless labeled selected:

| Candidate | Updates/seed | Realization MSE | Teacher E2E MSE | Free response change from step0 |
|---|---:|---:|---:|---:|
| Identity RGB | 0 | 0.02460923 | 0.06794662 | reference |
| Predictor target A | 1000/42 | 0.02303641 | 0.06586289 | +4.74% |
| Predictor target A replicate | 1000/43 | 0.02300378 | 0.06583283 | +5.40% |
| Teacher target B | 1000/42 | 0.02371079 | 0.06592207 | +18.15% |
| A budget continuation | 3000/42 | 0.02268426 | 0.06540502 | +8.69% |
| A selected checkpoint | 2800/42 | 0.02270720 | 0.06544461 | +10.44% |

The2800-step checkpoint is selected by the prespecified internal panel, not by endpoint or retained
evaluation. Its7.73% realization and3.68% Teacher response reductions fail25%/10% gates; Free also
fails5%. Two equal-budget seeds reproduce only a small gain, and one already fails Free. Teacher
response supervision increases pixels/response amplitude without better E2E response accuracy.

Selected checkpoint, all per-episode Contact validation windows:

| Episode | Contact windows | Realization MSE | Realization cosine | Teacher E2E MSE |
|---|---:|---:|---:|---:|
| 5 (retained) | 28 | 0.01586372 | 0.20072 | 0.09921249 |
| 26 (retained) | 68 | 0.01860206 | 0.12493 | 0.08308261 |
| 38 | 106 | 0.02105421 | 0.23519 | 0.06312017 |
| 44 | 73 | 0.03025429 | 0.18602 | 0.07272983 |
| 53 | 122 | 0.02364349 | 0.19498 | 0.06759930 |
| 60 | 71 | 0.01587682 | 0.23347 | 0.05832913 |

All validation phases were evaluated (Free1317/Pre108/Contact468). Internal episodes were Teacher
parameter-training episodes;5/26 were used in Teacher checkpoint selection and inspected repeatedly.
There is no claim of untouched test generalization. Stage/episode/seed CSV and raw sample records
are in `report_final150k/` and the individual experiment directories. Train statistics are an
episode-phase midpoint probe, not all17740 training windows. Selected training-probe Contact
realization MSE0.03068195 versus internal0.02270720 (raw gap-0.00797474); target-energy distributions
differ, so this negative raw gap is not proof of absent overfitting.

Train-only12-window Oracle versus the selected shared checkpoint, same images/batch shape/readout:

| RGB generation | Realization MSE | Cosine | Teacher E2E MSE | RGB RMS / max |
|---|---:|---:|---:|---:|
| Shared D | 0.02846434 | 0.29677 | 0.06885901 | 0.004250 / 0.083174 |
| Independent per-image Oracle | 0.01251766 | 0.68961 | 0.05238055 | 0.012826 / 0.088100 |

Both obey the same0.1*support per-channel bound. Oracle consumes more of the allowed RGB RMS;
this is a same-bound comparison, not a matched-RMS experiment. It confirms partial per-image RGB
reachability under the new teacher; it neither solves shared learning nor establishes a global
upper bound. Do not use its optimized pixels for validation or inference. The old70750 Oracle
warm-start and simpler patch-head runs already failed; do not repeat them as if untried. The next
informative diagnosis is a matrix-free local pixel-vs-shared-parameter Jacobian projection on TRAIN
windows, to test controllable response directions/parameter interference without another blind
training sweep. It has not been executed in this round.

Retained diagnostic weights:
`data/diagnostics/civi_student_iteration/final150k_baseline_s42_3000/best.pt`
(step2800, SHA256 `fec1fc56443602efc5be1ff84af00afd787f9363f30dcb3e572e64e84e3683d5`).
Local copies of this checkpoint, matched Adapter/Predictor/Condition and all small reports are saved;
large paired arrays and resumable latest/optimizer files remain on the assigned server. Architecture
remains the D initialization plus frozen Response Predictor and lightweight response projection;
it is a reference, not an accepted final architecture. Actual enhanced RGB inference uses no teacher
target or test-time pixel optimization; downstream DP/ACT/pi0 performance has not been evaluated.

Selected2800-step online RGB inference on RTX4090, batch1: median41.68ms /p9042.01ms;
active parameters89367430 (including frozen DINO), FP32 parameter bytes357469720,
additional peak activation bytes117675008. Cached-vs-online Predictor RGB MSE1.12e-9.
Training process GPU use was approximately4.9GB/card. The benchmark process's431292928 allocated
bytes also includes the audit Teacher, so it is not a minimal standalone inference footprint.

Final inclusive optimization accounting:11002 shared RGB updates +5400 per-image pixel updates
=16402 of20000. Dependencies/cache/Adapter/Predictor/Condition calibration steps are logged separately,
not hidden as RGB updates. No task training processes remain after the final inference benchmark.
