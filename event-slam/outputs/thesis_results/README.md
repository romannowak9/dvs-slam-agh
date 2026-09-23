# Thesis result artifacts

This directory is the only generated-output tree intended for version control.
It contains the configurations, result trajectories, metric reports and plots
that support the tables and figures in `latex/mgr/chapters/wyniki.tex`.

The artifacts were selected from three completed experiment batches:

- `thesis_experiments/02_ablations/slam_vs_vo` for the historical SLAM/VO
  comparison;
- `thesis_experiments_window12_updated/01_evslam_final` for the seven final
  EvSLAM sequences and official EvSLAM evaluation;
- `thesis_experiments_window24` for BAF, accumulation-window and rotation-source
  comparisons and the final M3ED sequences.

Large logs, map dumps, duplicated plots, temporary completion markers and
intermediate experiment batches are deliberately omitted. The raw estimate,
first-pose-aligned estimate, final configuration and relevant metrics are kept
where they exist, so the plotted result can be traced back to a concrete run.
The `figures` directory contains the final 3D rerenders used by the thesis when
their exact rendered copies were no longer present in the original run tree.

The convenient scripts in `scripts/run_evslam.sh` and `scripts/run_m3ed.sh`
write new results to `outputs/evslam` and `outputs/m3ed`. Those directories are
ignored by Git and do not overwrite this evidence set.
