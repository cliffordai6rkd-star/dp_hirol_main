python data_process/tool/visualize_tau_other_inference.py   --checkpoint outputs/tau_other_sequence/lstm_causal_derived/checkpoints/epoch_148_val_loss_0.019611.pt --root data/train_episode/next_bg_data50hz_b_lbv3     --device cuda:0 --tau-ext-filter-mode moving_average --tau-ext-filter-window 20 --tau-ext-lowpass-cutoff-hz 20

降采样:
python data_process/tool/downsample_h5_2_lerobotv3.py \
  --config config/shape_meta/data/next_data_dual_phase.yaml

inference