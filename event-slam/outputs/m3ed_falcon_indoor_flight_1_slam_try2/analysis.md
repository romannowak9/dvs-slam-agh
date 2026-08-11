# Indoor flight 1 — first-pose parameter check

The only changed parameter was `stereo_depth.max_depth`. On the same first
24 seconds, first-pose ATE RMSE was 0.383 m at 100 m, 0.351 m at 20 m, 0.254 m
at 15 m and 0.275 m at 10 m. The 15 m candidate was therefore selected for the
full run without changing RANSAC, tracking or SLAM parameters.

The full result disproved the short-run improvement. First-pose ATE increased
from 0.521 m to 0.595 m and successful frames decreased from 4022 to 4007.
The candidate was better before 30 s, but its ATE was 0.619 m during 30–40 s
and 0.752 m during 40–50 s, versus 0.403 m and 0.519 m for the baseline.
Consequently the default depth was restored to 100 m.

The candidate slightly improved local RPE (translation 0.0333 m, rotation
0.463 deg) and path ratio (98.3%), but these secondary metrics do not compensate
for worse first-pose ATE. No loop candidate or relocalization occurred. The
experiment indicates that distant points are noisy for translation but still
help long-term orientation and map continuity; a fixed global depth cutoff is
not an adequate solution.
