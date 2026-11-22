# CIVI experiment catalog (2026-11-16)

Read-only scan of `data/outputs/2026.11.*/*`, `data/outputs/civi/*`, `data/diagnostics/**`, `data/debug/*`, `docs/*.md`, `docs/reports/**` and `data/outputs/2026.11.16/paper_experiments/*`. I skipped `data/diagnostics/2026.11.16/teacher_torque_reliance` because another job is writing there. Values come from `config.yaml`, `.hydra/*`, `teacher_config.yaml`, `training_summary.json`, `logs.json.txt` (parsed as JSONL), and summary/comparison/README files. Nothing was trained, run for inference or modified, apart from writing this file.

No training process was running locally at scan time. Every `training_summary.json` that still says `"running"` is therefore stale: the process was killed or stopped. In this report those runs are marked **stopped**.

Paths are relative to `data/outputs/` unless stated otherwise.

## 0. Conventions

**NOT** is the normalized open-token MSE: `val_open_token_mse` at the selected (minimum `val_mse`) epoch, divided by its identity value (the zero-RGB-residual baseline). It is the fraction of the teacher-fusion gap that remains, and reduction = 1 − NOT.

The identity value is the first validation epoch whenever that epoch is still gate- or condition-only. In those epochs the RGB residual is zero-initialized and frozen, so the value equals the raw-image baseline. Where that is not the case, the baseline used is stated.

The identity gaps differ by more than 20×, so raw MSE cannot be compared across rows:

| Teacher / readout | Identity open-token MSE |
|---|---:|
| T1105 (LIBERO), 945 val windows | 0.6038 (all tokens: 0.1923) |
| T-civi, no teacher-FFN replay | 1.082 |
| T-civi, FFN replayed (`use_teacher_ffn=true`) | 0.117 |
| T-HG, FFN replayed | 0.0601 |
| T1112 (board), FFN replayed, region tokens | 0.04055 |
| T1112, no FFN replay | ≈0.96 global (adapter baseline) |
| T1115pb | 0.752 |
| T1116L | 0.823 |

The "open" token set is defined differently by family:

- **LIBERO 11.07:** observation steps whose future-wrench switch is on. The open ratio is a fixed 31.7%.
- **11.12–11.13 board:** true precontact/contact labels (`token_region_phases=[1,2]`).
- **11.15/11.16 current students:** the student's **predicted** gate (`p_open≥0.5`), which changes from epoch to epoch.

### Teacher IDs used below

| ID | Teacher |
|---|---|
| T1101 | `2026.11.01/22.57.02_091072…/teacher` (referenced only, not on disk) |
| T1104 | `2026.11.04/13.50.04…/teacher` |
| T1105 | `2026.11.05/22.45.54_375874…/teacher` @200k (the 11.07 reference teacher) |
| T-civi | `civi/teacher` @180k |
| T-HG | `civi/clean_tau_ext_20261111_rounds/huanggua/teacher` @190k |
| T1112 | `2026.11.12/20.05.59_876448_fdp_dis_board/teacher` @180k |
| T0331 | Server `2026.11.15/03.31.29_teacher_ca_curriculum_b64_150k` (local copy: `2026.11.15/final/teacher`); the diagnostics use its @70750 and @150k checkpoints |
| T1115pb | `2026.11.15/23.20.25…/teacher` @150k |
| T1116L | `2026.11.16/06.27.19…/teacher` @150k |

## 1. Teachers

| ID / path | Dataset | Torque input | Fusion / curriculum | Budget, status | Key metrics | Artifacts |
|---|---|---|---|---|---|---|
| T1101 (not on disk) | LIBERO | **future** 6D wrench (+1…+8) | 1 CA, learned token mask; FFN not contact-gated; 7-D action | 200k, b32 | none locally | none |
| T1104 `2026.11.04/13.50.04_dp_distill_gripper_sigmoid/teacher` | LIBERO (8-D action) | future 6D wrench | 1 CA, token mask, not contact-gated | 200k b32, completed | train action MSE 0.154 (norm.); no validation (`val_ratio=0`) | ckpts 160/180/200k; wrench-effect ep0 mp4 |
| **T1105** `2026.11.05/22.45.54_375874…/teacher` | LIBERO, 50 eps | **future** 6D wrench, `observation.wrench` offset +1, 8 steps. The gate also uses the future wrench (L2 of axes 0–2 > 8). | 1 CA + FFN; contact fusion gate; token mask with p=1 until 50k, linear to 0 at 150k, then 0 | 200k b32, completed | train action MSE 0.160 norm. / 0.0065 raw; no validation (`val_ratio=0`) | ckpts 160/180/200k (869 MB) |
| `2026.11.05/10.48.50_local_highres_teacher_distill/teacher` | LIBERO | future | T1105 recipe plus a local high-res wrist branch | 78k/200k, stopped | none | ckpts 20–60k; wrench-effect ep0 mp4 |
| T-civi `civi/teacher` | LIBERO | future wrench | 2 CA + blur σ5→0 (25–75%), `legacy_rollout`; FFN applied to all frames | 200k b32, completed | ckpt action MSE on 2 random **training** episodes per ckpt: best 0.00967 @180k | ckpts 160–200k |
| `civi/fdp_bs128_200k/teacher` | LIBERO | future | 2 CA blur, phase 1:1:1 | 200k b64, completed | 0.00894 @200k (50 training eps; 0.192 norm.) | ckpts 170–200k; dataflow audit |
| `civi/fdp_bs128_200k_phase152_precontact05/teacher` | LIBERO | future | phase 1:5:2 | 40k/200k b128, interrupted | 0.0181 @40k | ckpts 10–40k |
| `civi/teacher_bs128_100k_topk4_20261109` | LIBERO | future | phase 1:7:5 | 153k/200k b128, stopped | best 0.0150 @100k | ckpts 60–140k |
| `civi/dino_dp_bs64_200k/teacher` | LIBERO | **none** (DINO-only baseline policy) | – | 300/200k, paused | – | none |
| T-HG `civi/clean_tau_ext_20261111_rounds/huanggua/teacher` | huanggua clean, 41 eps | 7-joint tau_ext history (−7…0), from **`observation.tau_ext`** (not `_unthresholded`) | 2 CA blur | 200k b64, completed | 0.00373 @190k (all 41 training eps; 0.234 norm.) | ckpts 180–200k; wrench-effect ep0 mp4 |
| `2026.11.12/16.51.13…/teacher` | board ma5_l1_12p5 | tau_ext history | 2 CA blur | 30k b128, interrupted | 0.0206 @10k | ckpts 10k/20k |
| **T1112** `2026.11.12/20.05.59…/teacher` | board, 62 eps | `tau_ext_unthresholded` history (−7…0) | 2 CA + FFN; blur σ5→0 (25–75%), `full_context`; FFN not gated (CA scaled by alpha) | 200k b32, completed | ckpt action MSE on **all 62 (training) eps**: 0.00471 raw / 0.176 norm. @180k (the ckpt used) | ckpts 180/190/200k |
| `2026.11.15/22.41.15…/teacher` | board | tau_ext | phase_binary, b64 | 3.15k/150k, interrupted | val 0.0159 raw @2.8k | ckpt |
| T0331 = `2026.11.15/final/teacher` | board | tau_ext | 2 CA, attention_alpha; blur only on precontact/contact, σ5 for the first 1/3 then → 0 | 150k b64, completed | held-out (5 eps) val action MSE 0.005149 raw / 0.4465 norm.; val denoising loss **rose** 0.134 → 0.637 | ckpts 147–150k |
| **T1115pb** `2026.11.15/23.20.25_board_blur_legacy_fusion_b32_150k_local/teacher` | board | tau_ext history | `phase_binary` full-fusion gate driven by the offline `contact_phase`; 2 CA + FFN; blur on precontact/contact | 150k b32, completed | held-out val action MSE 0.005106 raw / 0.4577 norm. (min 0.456 @107.5k); val denoising loss 0.211 → 0.592 | ckpts 141.5–150k; context-effect previews + mp4 (ep0) |
| **T1116L** `2026.11.16/06.27.19_legacy_teacher_tau_ext_b32_150k_local_5090/teacher` | board, 192×256 | tau_ext history. Gate: L2 of tau axes 0–2 > 8 on any of the past 8 frames, no debounce. | legacy 1 CA + learned token mask (T1105 recipe). Mask total_steps=200k with a 150k budget: p = 1.0 @50k, 0.5 @100k, 0.0 @150k (verified in `04.21.21…/smoke_result.json`), so unlike T1105 there is no mask-free phase. | 150k b32, completed (resumed from 100k after a hang) | none logged (`val_ratio=0`) | ckpts 120/140/150k |
| `2026.11.16/04.21.21…1x4090`, `04.57.34…2x4090`, `2026.11.15/remote_huanggua_single_gpu_150k` | board / huanggua | tau_ext | server copies (1 GPU, DDP-2) / huanggua phase_binary | launch records only | 4.5 it/s (speed file) | metadata only; checkpoints on server |

## 2. Students (main runs)

"Ref" is `2026.11.07/22.23.10_distill_gate_sft_v2/student`. Smoke tests and failed launches are in §3.

| Run | Data / teacher | Torque | Design vs ref | Status (epochs, best ep) | NOT (reduction) | Other metrics | Artifacts |
|---|---|---|---|---|---|---|---|
| `2026.11.02/19.55.45…p90_ablation/unrestricted` | LIBERO / T1101 | future | old transformer + Gaussian-glow renderer; denoiser-MSE objective; no region head | early-stopped 236 (206) | n/a | val denoiser 1.087 → 0.322; token MSE not reduced (1.02 → 1.09) | best/latest; ep0 mp4 + frames |
| `…/context_p90` | same | future | adds binary top-10% context-effect region head (**p90 ablation**) | early-stopped 360 (330) | n/a | 1.055 → 0.478 (0.519 at epoch 230) | same |
| `2026.11.04/01.37.42_distill_patch_context` | LIBERO / T1101 | future | patch_residual + ReLU gate, denoiser | early-stopped 30 (0) | n/a | never improved (1.056); dead-ReLU diagnostic | ckpts; diagnostics json |
| `2026.11.04/13.08.37_distill_patch_context_sigmoid` | same | future | sigmoid gate instead of ReLU | stopped 41 by user (26) | n/a | 1.056 → 0.0742 | ckpts |
| `2026.11.04/13.50.04…/student` | LIBERO / T1104 | future | patch_residual sigmoid, denoiser | early-stopped 96 (66) | n/a | 0.789 → 0.0812 | ckpts; ep0 mp4s; patch-stage PNGs |
| `2026.11.04/21.49.27…/student` | T1104 | future | v2 transformer, gate threshold 0.1, denoiser | early-stopped 75 (45) | n/a | 0.753 → 0.1125 | ckpts; ep1 mp4 |
| `2026.11.05/17.42.05_distill_render_net` | T1101 @180k | future | CNN renderer, 4 CA, gate, denoiser | early-stopped 75 (45) | n/a | 1.044 → 0.0938 | ckpts |
| `2026.11.05/18.59.30_distill_render_net` | T1101 @180k | future | token_mse, no gate, 4 CA | stopped 156 (142) | n/a | token 0.959 → 0.516 (0.538) | ckpts; ep0 mp4 |
| `2026.11.05/22.45.54…/student` | LIBERO / T1105 | future | **no gate**, plain residual CNN (base 16, b32, lr 3e-4) | early-stopped 110 (83) | open-token not logged | **all-token 0.1923 → 0.0878 (0.457)** | ckpts; **ep49 (val)** mp4 + 182 PNGs |
| `2026.11.06/13.48.46…/student` | T1105 | future | soft gate 0.5, **denoiser** objective | interrupted 12 | n/a | denoiser 0.046 → 0.038; token MSE worsened 0.192 → 0.215 | ckpts |
| `2026.11.06/14.03.10_distill_token_mse_gate05/student` | T1105 | future | soft gate (sigmoid, threshold 0.5), token_mse; no separate path / FiLM / stages | early-stopped 85 (58) | not logged | all-token 0.551 | ckpts; ep0 mp4 |
| **Ref** `2026.11.07/22.23.10_distill_gate_sft_v2/student` | LIBERO / T1105, 945 val windows (eps 4/21/31/36/49) | **future 6D** | separate gate path + FiLM/SFT + 5 gate epochs; base 32, b16, lr 2e-4, patience 30 | early-stopped 154 (133) | **0.461 (53.9%)**, from 0.6038 → 0.2782 | all-token 0.1923 → 0.0889 (53.75%); closed 0.00086; mask MSE 0.00365 → 0.00280 | best/latest; ep0 mp4s; FiLM eval CSV/JSON |
| `2026.11.07/render_net_student_stage2_b8/student` | T1105 | future | b8, lr 3e-4, 2 gate epochs, no separate gate path | stopped 28 by user (18) | not logged | all-token 0.815 | ckpts |
| `civi/student` | LIBERO / T-civi | future (+2 Hz filter) | 6 SA / patch 8 / dec 3; `use_teacher_ffn=false` | early-stopped 76 (46) | 0.852 (14.8%) | – | ckpts; 1-window inference smoke |
| `civi/student_teacher_ffn_20261109` | same | future | `use_teacher_ffn=true` (only difference) | early-stopped 52 (25) | 0.721 (27.9%) | – | ckpts; ep0 mp4; FFN alignment check |
| `civi/clean_tau_ext_20261111_rounds/huanggua/student` | huanggua / T-HG | tau_ext | FFN replay, residual_scale 0.1, threshold 0.1 | interrupted 15 (7) | 0.966 (3.4%) | – | ckpts; ep0 mp4 |
| `2026.11.12/20.05.59…/student` | board / T1112 (1893 val windows, eps 5/26/38/44/53/60) | tau_ext | condition net (`phase_prediction`), region tokens, FFN replay, patch 8 / 6 SA / dec 3, **FiLM off** | interrupted 4; resumed at ep1 with b4; val metric redefined at resume | 0.964 (min @4) | phase acc 0.95 | ckpts (+ best_previous_metric) |
| `2026.11.13/12.44.25…_board_film/student` | T1112 | tau_ext | as above with **FiLM on** (only config difference) | interrupted 18 | 0.978 | – | ckpts |
| `2026.11.13/14.00.50…_shared_film/student` | T1112 | tau_ext | **shared condition backbone** (no separate gate), skip_unselected_obs | interrupted 11 (5) | 0.961 | – | ckpts |
| `2026.11.13/15.35.22…_dinolora/student` | T1112 | tau_ext | as shared_film, but **DINO-LoRA** RGB encoder (r8 q/v; SA 0; patch 16) | interrupted 14 (12) | 0.956 | – | ckpts (714 MB) |
| `2026.11.13/cnnvit_ordered_full/student` | T1112 | tau_ext | shared backbone; b16; threshold 0; condition 3 → RGB 3 → joint; detach_for_rgb | stopped 42 (39) | 0.895 (10.5%) | – | ckpts; loss PNGs |
| `2026.11.13/21.23.44_cnnvit_condition_then_render/student` | T1112 | tau_ext | independent compact CNN-ViT condition net; condition (plateau) → RGB only; resumed from the 21.04.17 condition ckpt; b64 | stopped 36 (24) | 0.924 (7.6%) vs identity 0.04055 | see diagnostic in §4 | ckpts; **ep26 (val)** mp4 + no-region-gate inference mp4 |
| `2026.11.13/22.05.38_cnnvit_no_region_gate/student` | T1112 | tau_ext | 21.23.44 with `use_region_gate=false` (contact alpha only) | stopped 45 (23) | 0.913 (8.7%) | – | ckpts |
| `2026.11.14/13.25.56_response_weighted/student` | T1112 | tau_ext | **no FFN replay**; condition_prediction; global + response-weighted token MSE | stopped 68 by user (RGB epoch 62) | identity not logged; open 0.889 (first RGB epoch) → 0.638 | composite 1.805 → 1.537 | ckpts; ep5 inference mp4s |
| `2026.11.14/17.31.30_response_adapter_local_region02/student` | T1112 | tau_ext | + linear zero-torque **context adapter**; region threshold 0.2; RGB head zero-init | early-stopped 75 (RGB epoch 17) | n/a (identity+adapter open-token not logged) | composite 0.4288 (identity+adapter) → 0.4143 (−3.4%); global token 0.1700 → 0.1678 | ckpts (best / adapter / condition) |
| `2026.11.14/21.38.50_response_adapter_random_output_region02` | T1112 | tau_ext | RGB output conv **random init** from 17.31.30's condition ckpt | stopped at RGB epoch 7 | – | composite 0.598 → 0.4162 | ckpts |
| `2026.11.15/00.24.22_response_adapter_restart_region02` | T1112 | tau_ext | restart of 21.38.50, patience 10 | early-stopped 24 (14) | open 0.152 → 0.144 | composite 0.598 → 0.4115 | ckpts; ep5 mp4s |
| **`2026.11.15/23.20.25_board_blur_legacy_fusion_b32_150k_local/student`** (current method) | board / T1115pb (latest EMA) | tau_ext | ref architecture (CNN+LSTM 4/2/4, patch 16, base 32, separate gate, FiLM) + causal CNN+GRU phase predictor (`p_open≥0.5`); 224²; patience 5 counted from RGB start | **stopped manually at 37** (35) | **0.789 (21.1%)** vs 0.752 | phase: acc 0.959, free false-open 0.025, precontact recall 0.69, contact recall 0.977; phase BCE 0.165 → 0.704 (overfits during the RGB stage) | best/latest; **ep5 (val)** mp4; ep0 FiLM on/off mp4 + CSV |
| **`2026.11.16/06.27.19…/distillation/student`** | board / T1116L | tau_ext | identical recipe; gate label = this teacher's tau threshold | early-stopped 42 (37) | **0.697 (30.3%)** vs 0.823 (min 0.693) | acc 0.987, free false-open 0.006, contact recall 0.967 (precontact recall 0 by design) | best/latest; ep0 mp4 + FiLM mp4 |

## 3. Smoke tests, failed launches and engineering runs

- **`2026.11.02/…/{context_p90_smoke,unrestricted_smoke}`:** 2-step GPU checks.
- **`2026.11.04/13.45.31_gripper_pipeline_smoke`:** 3-step teacher and 2-step student.
- **`2026.11.05/10.40.08`, `10.46.36` (local_highres):** teacher failed to start. `2026.11.05/17.41.07` has a config but no logs.
- **11.13 CNN-ViT launches:** `20.42.49` got SIGINT at start, `20.45.22` was killed during cache build, `21.23.34` was a resume launch with no logs. `20.48.16` (b128) and `21.04.17` (b32) only ran the condition stage, for 10 and 7 epochs. The `21.04.17` checkpoint seeds both `21.23.44` and `22.05.38`.
- **`2026.11.13/token_mse_5sample_{cnn_ordered,ordered}_30ep`:** 5-sample, 30-epoch fit tests (NOT ≈ 0.99).
- **`2026.11.14/13.12.44_503186_response_weighted`:** identical config to 13.25.56, aborted in the condition stage. `17.31.30/previous_run` is a copy of 13.25.56 at its stop.
- **`2026.11.12/16.51.13` and `2026.11.15/22.41.15` teachers:** interrupted early; superseded.
- **Records only:**
  - `2026.11.16/civi_phase_integration` (tests, cleanup manifest, queue records)
  - `2026.11.15/23.20.25…{console.log,launch.json}`
  - `civi/training_queue` and `latest_teacher_phase152_queue.txt`
  - `civi/fdp_bs128_200k/teacher/full_mse_migration_*` (5–15k ckpts from before the metric change)
- **`data/debug`:** `civi_clean_preflight_20261111` (2-step teacher, 1-step student) and `render_net_stage_smoke_2026.11.07{,_pyav,_step}`.
- **Diagnostic smokes:**
  - `contact_loss_logging_smoke`, `scene_torque_response_predictor_smoke_fp32`
  - `student_adapter(_compiled)_smoke`, `local_training_acceleration`, `render_differentiability` (gradients flow; analytic vs finite-difference error 0.0024%)
  - `civi_selfattn_action_heads`, `two_stage_implementation`
  - `civi_paired_response_full_final_teacher` (queue superseded)
- **Not scanned** (outside the requested scope): `outputs/cross_attention_queue`, `local_highres_queue`, `usb_dataset_audit_2026-10-18`, `data/remote_runs`, `data/queues`, `data/benchmarks`.

## 4. Diagnostics with quantitative results

| Path | Teacher / student | What | Key numbers | Artifacts |
|---|---|---|---|---|
| `diagnostics/2026.11.15/teacher_fusion_ablation_5k_1829` | board; 3 teachers from the same init, 20 train eps, 5k steps, val eps 5/26/39/45/61 (1547 windows) | CA vs FiLM-only vs Hybrid teacher; zero/shuffled torque at fixed alpha | Val denoising MSE: 0.1777 / 0.1748 / 0.1777. Action MSE: 0.01598 / 0.01558 / 0.01597. Contact token response RMS: 0.0613 / 0.0472 / 0.0610 (free: 0.0031 / 0.0018 / 0.0030). Shuffle penalty (denoising): 0.00087 / 0.00115 / 0.00098. Contact shuffled action MSE is only 1.2–1.5% above normal. **Zero torque lowers contact denoising MSE** (0.2543 → 0.2482 for CA). | PNGs (response maps, interventions, phases, per-episode), per-sample JSON, latest.ckpt |
| `…/teacher_fusion_ablation_020219` | pilot, 1k steps; Hybrid untrained | same | FiLM-only denoising −1.5% but action +0.7%; zero torque lowers contact denoising (0.1392 → 0.1375); shuffle penalty 0.00008 | PNGs, md |
| `…/teacher_best70750_response_study` | T0331 @70750 | stability of the teacher's torque response; ridge; per-image oracle; shared nets; Jacobian | Cross-episode response cosine: condition-matched 0.394, torque-mismatched 0.292, uncontrolled 0.268. Ridge explained energy: torque 0.546, visual 0.517, joint 0.513, joint with shuffled torque 0.355. Shared/RenderNet response cosine 0.05–0.07 vs per-image oracle 0.50–0.53. | PNGs for eps 5/26/38/44/53/60 |
| `…/scene_torque_response_predictor_quick` | T0331 @70750 | ΔZ predictor from scene and torque (119k params) | Val contact explained energy: 0.374 correct, 0.239 shuffled torque, 0.135 shuffled scene, 0.248 train-mean | ckpt, PNG |
| `…/response_conditioned_rgb` | T0331 @70750 | turning the predicted response into RGB | End-to-end cosine on 12 contact samples: per-image oracle 0.517 vs shared decoder 0.155 | PNGs |
| `…/new_teacher70750_adapter_shared_response` | T0331 @70750 | MLP adapter + shared CNN/D renderers | Adapter zero-context MSE 0.643 → 0.112; renderers improve response MSE by only 0.13–0.28% in 1000 steps; stopped by user | PNGs, CSV |
| `…/student_dual_path_ablation/review_1000` | T1112 | RenderNet A (6 SA + 2 CA) / B (interleaved) / C (+ FiLM) and linear vs MLP adapter, at 1000 RGB steps | Contact token MSE 0.1740 / 0.1746 / 0.1739 (C with MLP adapter: 0.1224). Renderer gain over original RGB + same adapter: only +0.5–1.45%. Torque-response error is 1.12–1.37× a no-response student. | render_{free,precontact,contact}.png |
| `diagnostics/civi_student_iteration` | T0331 @70750 and @150k | bounded program of 21 entries within a 20k-update budget | None promoted. Shared renderers reach realization cosine 0.12–0.23; per-image oracle 0.69–0.70. final150k selected: realization −7.7%, teacher response −3.7%, correct torque beats shuffled/zero; 2 seeds for the baseline. | experiments.csv, per_episode_results.csv, visuals |
| `diagnostics/tau_capacity_v2` | T1112 | token-space ΔZ predictors on 144 contact windows, 2 seeds | Test relative MSE: MLP 0.933, **MLP without tau 0.892**, Transformer 0.952, DiT-flow 2.18. 5-sample memorization succeeds. | evidence.png |
| `diagnostics/condition_then_render_validation_gain` | student 21.23.44 best | open tokens vs raw image | 0.04055 → 0.03748 (−7.6%; contact −8.3%; top-10% patches −14.9%; 80.6% of observations improve). Free gets +16% at a level of 3e-4. | JSON |
| `diagnostics/condition_then_render_no_context_gate` | same ckpt | region gate removed at inference | open-token MSE +294% vs gated; 5.2% of observations better | JSON |
| `diagnostics/2026.11.14/adapter_render_baseline` | student 17.31.30 | identity vs adapter vs renderer | Global MSE: 0.962 identity without adapter → 0.170 with adapter → 0.168 with renderer. RGB gain 3.4% on composite. Teacher zero-context: 0.0123. | CSV, PNG |
| `diagnostics/2026.11.14/response_distillation` | T1112 + 13.25.56 snapshot | 12-sample pixel and renderer probes | Raw image vs target ≈ 0.89–0.99 even in free frames (FFN not gated); with FFN replayed the zero-torque gap is ≈ 2e-4; offline adapter brings it to 0.06–0.19 | JSON / pt |
| `2026.11.13/teacher_tau_effect_ep0` (under outputs) | T1112, ep0 | real vs raw-zero tau context effect; 5-window ΔZ probe | Context-effect MSE: contact 0.0408 vs free 0.00056. The probe overfits (eval relative MSE 7.2). | mp4, PNG, JSON |
| `debug/feature_inversion_*` | T1105 + 11.06 gate05 student; 12 samples (6 contact, 6 precontact) | pixel-space upper bound | Raw image 0.671; student ckpt 0.342; student overfit on these samples 0.155–0.206. Pixel optimization: in student mask 0.067, in teacher mask 0.072, with fixed gate 0.032, full image 0.039. | report.json, curves.png |
| `2026.11.07/…/student/film_evaluation` | ref | SFT/FiLM on vs bypassed at inference; 945 windows, 312 active, 3 seeds | Active windows: token 0.294 vs 0.509 (−42.3%); normalized action 0.191 vs 0.219 (−13.1%); position RMSE 58.1 vs 64.7 mm (−10.3%). Episode 21's action error worsened by 2.4%. | CSV, JSON |
| `diagnostics/civi_dit_minimal_v1` + `docs/reports/civi_dit_minimal_v1` | denoiser-KD RGB student, server run | smoke80 + formal launch | Contact KD 4.7% worse, contact sampled action MSE 7.2% better (the README itself says this is not evidence). Formal run is only at step 0/110 locally. | JSON, PNG |
| `diagnostics/2026.11.16/teacher_torque_reliance` | – | **not read** (another job is writing; script `tools/teacher_torque_reliance.py`) | – | – |

`paper_experiments/*` contains only material derived from the ref run (`distillation_1107`, `distillation_visual_evidence`, `temporal_1107`, `legacy_sft_*`). It also has the real-world table (Pi0 31 → 45/60, DP 14 → 29/60, TA-VLA 38/60), which is **user-reported and hard-coded in `build_evidence.py`**, with no local checkpoint provenance.

## 5. Candidate paper evidence

### 5.1 Clean ablation pairs (one design factor, same teacher and data)

| Factor | Runs A → B | Budgets match? | Result | Caveat |
|---|---|---|---|---|
| Soft context gate vs none (LIBERO) | 11.05/22.45.54 student → 11.06/14.03.10 gate05 | Both early-stopped with patience 30 (110 vs 85 epochs) | All-token ratio **0.457 (no gate) vs 0.551 (gate)**; no-gate is better at every common epoch | Open-token not logged; future wrench |
| Region-restricted rendering (p90) vs unrestricted | 11.02 unrestricted → context_p90 | Both early-stopped (236 vs 360) | Val denoiser 0.322 vs 0.478 (p90 at 230 epochs: 0.519) | Old glow renderer and denoiser objective; T1101 not on disk |
| ReLU vs sigmoid gate | 11.04 01.37.42 → 13.08.37 | No (30 epochs early-stopped vs 41 stopped by user) | ReLU never trained; sigmoid reached 0.074 | Engineering-level finding |
| FiLM off vs on (board) | 11.12/20.05.59 → 11.13/12.44.25 (only `spatial_modulation` differs) | **No** (4 vs 18 epochs; A resumed, and its epoch 1 ran at b16) | At matched epochs 2/3/4: 0.973 / 0.968 / 0.964 (off) vs 0.979 / 0.978 / 0.990 (on); best 0.964 vs 0.978 | All reductions only 2–4%; single seed |
| Region gate on vs off | 21.23.44 → 22.05.38 (same condition ckpt) | Partly (36 vs 45 epochs, both stopped) | Best within 36 epochs: 0.924 vs 0.913 | Tiny; contrast with inference-time gate removal (+294%) |
| CNN vs DINO-LoRA encoder | 14.00.50 → 15.35.22 | No (11 vs 14) | 0.961 vs 0.956; within 11 epochs, 0.961 vs 0.959 | Patch size and SA layers change together with the encoder |
| Teacher-FFN replay (target readout) | civi/student → student_teacher_ffn | Both early-stopped (76 vs 52) | 0.852 vs 0.721 | Changes the target definition (identity 1.08 vs 0.117) |
| RGB head zero vs random init | 17.31.30 → 00.24.22 (21.38.50 is partial) | No (patience 30 vs 10) | Composite 0.4143 vs 0.4115; open 0.1451 vs 0.1442 | Composite metric; identity + adapter is 0.4288 |
| Teacher CA vs FiLM-only vs Hybrid | teacher_fusion_ablation_5k | Yes (5k steps each, same init) | See §4 | Single seed, 20 episodes |
| SFT on vs bypassed (inference) | ref film_evaluation | Same ckpt | −42.3% token / −13.1% action / −10.3% position | Not a retrained ablation; future wrench |
| Teacher recipe under a fixed student recipe | 23.20.25 student → 06.27.19 student | ≈ (37 stopped manually vs 42 early-stopped) | NOT 0.789 vs 0.697 | The teachers differ in at least 5 factors |

Pairs that change more than one factor:

- 12.44.25 → 14.00.50: separate path → shared backbone, plus skip_unselected (0.978 → 0.961).
- 14.00.50 → ordered_full: at least 4 factors (0.961 → 0.895).
- 13.25.56 → 17.31.30: adapter plus two thresholds.
- 11.06 13.48.46 → 14.03.10: objective, with A interrupted at 12 epochs.
- gate05 → ref: at least 8 factors. stage2_b8 → ref: 6 factors.
- 11.05 17.42.05 → 18.59.30: objective plus gate.
- Teachers T0331 vs T1115pb: batch size and gate mode; held-out val action 0.005149 vs 0.005106.

### 5.2 Teacher-side evidence

Uses torque in the representation:

- T1112 context effect on ep0: contact 0.0408 vs free 0.00056.
- 5k ablation: contact response RMS 0.061 vs free 0.003.
- T1115pb ep0 context effect: free exactly 0, precontact/contact mean patch L2 ≈ 13.5. This compares full fusion with bypass (gate off), not torque content.

Benefit from torque content is weak:

- At 5k steps, shuffling torque raises contact action MSE by only ≈1.2%.
- Zero torque at fixed alpha **lowers** contact denoising MSE, both at 1k and at 5k steps.
- `tau_capacity_v2`: the MLP without tau predicts ΔZ better than the MLP with tau.
- Ridge (T0331 @70750): explained energy drops from 0.513 to 0.355 when torque is shuffled.
- Scene-torque predictor: explained energy drops from 0.374 to 0.239 when torque is shuffled.

Gaps:

- No completed vision-only teacher exists (the DINO-only baseline was paused at 300 steps).
- No no-blur teacher matched to a blur teacher exists.
- Only T1115pb and T0331 have a held-out teacher val_action_mse (≈0.0051 raw). All other teacher numbers are on training episodes or absent.
- For the ref teacher (T1105), the only evidence is context-effect videos (`debug/episode49_teacher`); there is no torque-content intervention.

### 5.3 Diagnostics already holding quantitative results

All rows in §4. The most paper-ready ones:

- teacher_fusion_ablation_5k (teacher design and torque reliance).
- Ref FiLM evaluation (inference SFT).
- feature_inversion: realizability upper bound; per-image pixels reach 0.03–0.07 vs raw 0.67 vs student 0.34.
- condition_then_render gain / no-gate pair.
- adapter_render_baseline (the renderer adds only 3.4% over the adapter).
- civi_student_iteration and teacher_best70750_response_study: a reachability gap, with per-image oracle cosine ≈0.5–0.7 vs shared renderers ≤0.23.

### 5.4 Qualitative visual assets

| Asset | Checkpoint | Dataset / episode (split) |
|---|---|---|
| Ref `attention_visualize`: comparison/rendered mp4, FiLM on/off mp4 + CSV, previews, stats npz. `paper_experiments` crops (ep0 frame 57) and temporal plot are made from these. | ref best (step 71687) | LIBERO ep0 (**train** episode) |
| 11.05/22.45.54 student render mp4 + 182 PNGs | best (22410), no-gate student | LIBERO ep49 (student val). This is the only student render that pairs with the same-teacher (T1105) ep49 context-effect videos in `debug/episode49_teacher`. |
| gate05 (ep0), 18.59.30 (ep0), p90/unrestricted (ep0 + frames), 11.04 students (ep0/ep1; patch-stage PNGs), T1104 wrench-effect ep0 | respective best | LIBERO |
| 21.23.44 comparison + no-region-gate render + peak preview | best (8611) | board ep26 (val; seen by teacher T1112) |
| 13.25.56 predicted_region_alpha mp4; 00.24.22 initial/epoch-1 renders | best epoch 60 / restart | board ep5 |
| **23.20.25 current student**: comparison/rendered mp4 + peak preview | **latest** (step 39924, epoch 36; best is epoch 35) | board **ep5** (val; likely teacher-held-out) |
| 23.20.25 FiLM on/off mp4 + CSV (peak on/off RGB MAE 0.0129) | same | board ep0 (train) |
| T1115pb context-effect previews (free / precontact / contact) + mp4 | teacher @150k | board ep0 |
| 06.27.19 student render + FiLM mp4 | best (41033) | board ep0 (train) |
| huanggua student ep0, T-HG wrench-effect ep0 (+299 frames), `debug/context_tau_ext_smoke` | respective | huanggua ep0 |
| Diagnostic PNGs: 5k ablation response maps / interventions; dual-path render_*; response-study contact_direction / oracle_rgb; student-iteration visuals; tau_capacity evidence.png | see §4 | board val eps 5/26/38/44/53/60 |
| `docs/figures/method/civi_method.*` + assets (side/wrist original / blur / rendered PNG, mask npy) | **provenance not recorded** in `make_method_figure.py` (uncertain) | – |

### 5.5 Confounds

1. **Future-wrench leakage.** All LIBERO teachers and students (11.02–11.07, `civi/*` LIBERO) feed the *next* 8 wrench samples into the LSTM. For T1105/ref the gate timing also comes from the future wrench. Every `paper_experiments` number inherits this.
2. **Teachers trained on the student's validation episodes.** `val_ratio=0` for T1101, T1104, T1105, T1112, T1116L and the civi teachers. For T1115pb probably only episodes 5 and 26 are held out from both teacher and student (inferred from the T0331 split; not verified for this run).
3. **Teacher "val" metrics are on training data.** `checkpoint_action_mse` is computed on training episodes; for T-civi it uses 2 random episodes per checkpoint. Teacher val denoising loss rises while sampled action MSE falls (T0331, T1115pb).
4. **Unequal or truncated budgets.** Many runs were killed with a stale "running" status. The current student 23.20.25 was stopped manually at epoch 37, not early-stopped. Patience differs (30 / 10 / 5), and the early-stopping counter starts at different points.
5. **Different teachers and target definitions.** Identity gaps range from 0.04 to 1.08. `use_teacher_ffn` changes the target. The open-token set is defined by future-wrench, true-label or predicted-gate rules. The 11.12/20.05.59 val metric changed when it resumed. The adapter runs use a composite val metric.
6. **Image grid and action dims differ:** 192×256 / 768 tokens vs 224² / 784 tokens; action dimension 8 vs 7.
7. **Single seed** almost everywhere. Exceptions: `tau_capacity_v2` and the final150k baseline (2 seeds).
8. **The T-HG teacher uses thresholded `observation.tau_ext`.** The board runs use `_unthresholded`.
9. **Teacher caches live in the Trash** (`data/cache` → `~/.local/share/Trash/files/cache`). Re-evaluation may need recomputation.

### 5.6 Gaps that can be filled without training (inference only)

- Open-token MSE for the no-gate LIBERO student (22.45.54) and for gate05.
- Torque-content interventions (shuffle / zero at a fixed gate) for T1115pb and T1116L, and for the students trained from them.
- Phase-predictor event metrics on the current students.
- Linking the real-world checkpoints to local runs.
