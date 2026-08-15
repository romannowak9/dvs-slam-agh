# M3ED full-run analysis

- Configuration: 12 ms exponential event frames, tau 12 ms, BAF disabled.
- Runtime: 5301.5 s total (781.0 s reader, 4520.5 s processing and SLAM).
- Tracking: 12,928/13,597 successful frames, median 144 PnP inliers; 9,120
  poses came from the local map and 3,807 from VO fallback.
- Map: 2,233 keyframes and 49,209 landmarks. No loop candidate, loop closure
  or relocalization occurred.
- SE(3)-aligned ATE RMSE is 51.66 m and rotation RMSE is 100.55 deg. Sim(3)
  alignment estimates scale 0.453, confirming severe accumulated drift rather
  than a coordinate-axis plotting problem.
- BAF is not recommended: the full BAF run took 178.4 min, increased failures
  from 669 to 931 and slightly worsened SE(3) ATE.
- The M3ED plots display both estimate and GT with axes reordered from x,y,z to
  y,z,x for visual consistency with EvSLAM. Saved poses remain in the official
  left-Prophesee camera convention required by the challenge.
