# Current-best teacher response and RGB reachability diagnostics

The user explicitly selected the currently best checkpoint instead of waiting
for step150000. A fixed EMA snapshot was captured at step70750, validation action
MSE0.00530689. `teacher_provenance.json` records the resolved filename and hashes;
later changes to `best.ckpt` cannot alter this experiment. The separate final-teacher
A/B queue and teacher training continue unchanged.

All scripts are independent under `test/`; no production model/config is modified.
The compact FP32-computed teacher cache contains all468 held-out student contact
windows and12 Free windows, plus335 contact/112 Free windows spread over all56
student-training episodes. Targets store FP16 after differencing computations;
context preparation and physical-zero readout share the original teacher path.
The current teacher held out episodes5,26,39,45,61, while the existing student split
held out5,26,38,44,53,60. Only the96 contact windows in5/26 are unseen by both.

1. `prepare_response_study.py` loads pinned EMA, reuses historical CUDA-decoded
   images and exact student split identities, and collects paired contexts,
   pre-adapter visual features, raw/normalized torque histories, posture and alpha.
2. `analyze_teacher_response_stability.py` matches different episodes by pooled
   image descriptors, full torque history direction/magnitude, robot posture and
   contact alpha. It reports full-context cosines, last-observation cosines,
   train-mean-centered directions, camera/region/semantic Patch matches, strict
   matching coverage, six-episode bootstrap contrasts and torque clusters.
   TRAIN-only PCA/ridge predicts camera-pooled responses with visual, torque,
   joint, shuffled and training-mean controls; it does not claim full-Patch prediction.
3. `run_shared_rgb_oracle.py` uses the same frozen old MLP Adapter and deployed
   Condition Network for a tiny shared CNN+torque MLP, D renderer reference and
   individual pixel variables. The RGB bound is0.1 times predicted gate*alpha,
   followed by[0,1] projection. Real/zero torque branches recompute conditions.
   Explicit alpha0 identity is checked separately. Shared generators receive no
   sample IDs or test labels. CNN and D train300 steps; individual pixel fits150
   steps on12 train/12 validation windows. The report compares those exact windows
   and also evaluates both shared networks on all468 contacts. The D renderer
   has a different pretrained initialization and lower lr; this limits architecture
   attribution. The response-only objective is full weighted Delta MSE+10 RGB MSE.
   An earlier200-step mixed-objective diagnostic is preserved separately because
   the old Adapter's absolute context mismatch with the new teacher is substantial.
4. `diagnose_rgb_local_jacobian.py` uses exact forward-mode JVP and reverse VJP
   through preprocessing, frozen DINO projection, MLP Adapter and Context Norm.
   Attention uses the mathematically equivalent eager kernel for derivative support.
   Matrix-free ridge CG uses mean-feature/mean-pixel scaling. JVP/VJP adjoint and
   finite-difference step sensitivity are recorded. Six windows (one per held-out
   student episode) use12 and48 CG iterations, with actual clipped RGB replay
   and deployment-gate bounds. Finite-iteration results are not exact upper bounds.
   A five-scale nonlinear RGB replay checks whether the proposed perturbation
   stays within a useful local region; the best scale is selected per diagnostic
   image using its target, so it is not a generalization estimate.
5. `report_response_study.py` produces tables, episode metrics, curves and response
   maps. Final user conclusions are given directly in the conversation.

Numerical tests check the CG solver against an exact ridge solution and prove
that the small generator starts at RGB identity and propagates gradients through
both its image and torque paths. Actual GPU runs check deployment paths and
unchanged Teacher, Adapter and Condition Network state hashes.

An additional individual Oracle check fixes the zero-torque RGB to the original
image (`--individual-fixed-zero --individual-split validation`). This separates
normal-image reachability from improvements obtained by independently adjusting
the zero-image branch. Both paired and fixed-zero results remain available.
The report compares the exact same12 validation windows and renders RGB residuals.

Dependencies: the project `fdp` environment plus scikit-learn1.7.2 for the CPU
PCA/ridge/torque-mode diagnostic. Run `prepare_response_study.py` first with a
pinned teacher in `STUDY/inputs`, then `analyze_teacher_response_stability.py`.
Run `run_shared_rgb_oracle.py --kind shared --steps 300 --response-only`, then
repeat with `--kind rendernet`; use `--kind individual --steps 150 --response-only`
for the individual mode and repeat that mode
with the fixed-zero flags. Run `diagnose_rgb_local_jacobian.py --iterations 48`
and finish with `report_response_study.py STUDY`. All GPU commands also require
`--root OLD --study STUDY`. Each run refuses to replace an existing output folder.
