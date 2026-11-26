# CIVI CVPR 2027 Supplement

- `main.pdf` is the 14-page English supplement in CVPR two-column review format on US Letter paper.
- `main.tex` is the LaTeX source.
- `cvpr.sty` is the CVPR style file used to compile it.
- `assets/` contains every figure referenced by the source.
- `SOURCES.md` identifies the experiments and exact inputs for the attention, context and simulation results in Sections S10 and S11.

The supplement includes the main paper's success and latency results, method details, architecture ablations, recorded task sequences, renderer residuals across phases, held-out multi-episode render diagnostics, phase-gate and torque-estimator experiments, paired action-response studies, and negative controls. It distinguishes method illustrations from experiment-generated plots and identifies offline readouts and per-image fitting as diagnostics rather than closed-loop success evidence.

Figures S16 and S17 and Table S13 compare attention weights, torque-induced context changes and action error. Figures S18 to S20 and Table S14 cover simulation and real-robot replay, renderer modulation, representation alignment and the K4 simulation evaluation.

Compile from this directory with `tectonic -X compile main.tex`, or run `pdflatex main.tex` twice. Replace `cvprPaperID` in `main.tex` with the assigned submission ID before upload.
