# Fusion module handoff

This module is hardware-free Swift 5 code. It does not establish camera accuracy,
AirPods performance, or human Webgrid throughput. All benchmark scores below are
synthetic and depend on the declared observer and noise model.

## Design and public API

- `OneEuroFilter` / `OneEuroFilter2D`: first-order low-pass filters with a filtered
  derivative and speed-dependent cutoff, `minCutoff + beta * speed`. The 2D filter
  uses vector speed for the same cutoff on both axes.
- `ExtendedKalmanFilter`: dense state/covariance using the shared `Matrix`; linear
  matrix and nonlinear closure entry points. Measurement updates return innovation,
  innovation covariance, NIS, and whether the optional gate accepted the sample.
  Covariance uses Joseph form and is symmetrized after each step.
- `ConstantVelocityKF`: a small 2D constant-velocity wrapper around the same EKF.
- `EKFCursorFusion: CursorFusing`: six states `[x,y,vx,vy,bx,by]`, gaze observation
  `z = position + bias`, confidence-scaled observation covariance, and head angular
  rate mapped to a velocity observation. Relative head integration and hand deltas
  move position and bias in opposite directions, preserving the gaze-observed sum.
  The gaze channel therefore does not immediately undo a deliberate correction.
- `DirectGazeCursor: CursorFusing`: gaze plus One Euro baseline, with no learned
  cross-fixation calibration correction.
- `ClickArbiter: ClickArbitrating`: per-source confidence, per-button debounce,
  suppression of duplicate intents from another source, and historical onset
  position locking. It emits the shared down/up/click/scroll actions. A held button
  is exposed through `activeButtons` / `isDragging`; the app must emit the actual
  dragged cursor motion because `ClickAction.Kind` has no drag-move case.
- `WebgridTask` / `WebgridScore`: centered square grid, one highlighted target,
  correct/incorrect/outside outcomes and `max(log2(cells * clickTypes - 1) *
  NTPM / 60, 0)`. Outside-canvas clicks are counted separately and do not enter
  NTPM. Simulated sessions currently use left-click targets only; the score helper
  also supports a larger click-type alphabet.
- `WebgridSimulator` / `SimSummary`: seeded sensor/observer simulation through the
  actual cursor and arbiter implementations, individual sessions and aggregates.
- `SimTuner`: random local/global perturbation followed by coordinate search;
  objective is mean training BPS, with a separate held-out-seed evaluation.
- `ReplayEngine` / `ReplayHooks`: recording events in file arrival order, derived
  gaze/head/input streams, display-clock cursor statistics and actions. Optional
  raw-gaze mapper, hand classifier/delta and jaw-detector closures let the app wire
  in its hardware-free mapping/classification logic. Unmapped raw samples are
  counted and skipped.

All stateful objects are intended for one serial queue. Simulator and tuning
sessions run synchronously; there is no background service or persistent queue.

## Fusion behavior and assumptions

Large gaze innovations are initially rejected by NIS and distance gates. A small
cluster of agreeing samples confirms a saccade and resets the gaze-observed sum.
A short window also tests persistent medium shifts that pass the single-sample
gate. On reset, the learned bias decays with jump distance, since a correction at
one screen location need not generalize to another location. Fixation uses the
low-process-noise filter. Confidence zero is an unusable gaze sample.

The filter uses capture timestamps, with a bounded out-of-sequence approximation:
late gaze innovations use historical position and current covariance. This is not
an exact delayed-state EKF. Very old gaze samples are dropped; head samples must
not replace the newest sample used for attitude differentiation. Display calls
extrapolate the decaying head velocity and clamp to the configured screen.

Head mapping uses angular **rate**, a deadband, an isotropic nonlinear gain, a speed
cap and a stale-input timeout. It does not integrate absolute yaw, avoiding direct
attitude-drift pointing. Device axes/signs are configurable and require hardware
verification. Slow gyro bias within the deadband produces no cursor motion.

The user is assumed to keep looking near the target while applying relative head
or hand corrections. The webcam estimator is assumed to compensate head pose
partly; remaining head-to-gaze leakage is represented in the simulator. Neither
gaze alone nor the six-state filter can identify absolute calibration bias without
an independent correction or target observation. The learned bias is an
interaction-based correction, not a separately observed physical ground truth.

Click onset locking defaults to 100 ms before the classifier event timestamp.
Down/up ownership is source-specific: a suppressed duplicate source cannot release
the original source's held button. An owner's release is accepted despite reduced
confidence, so falling gesture confidence does not prevent a matching release.

## Tunables

`OneEuroConfig`, `HeadPointerConfig`,
`EKFCursorFusionConfig`, `DirectGazeCursorConfig`, `ClickArbiterConfig`,
`FusionTuningBundle`, `SimConfig`, its nested sensor/human configuration structs,
`WebgridConfig` and `SimTunerOptions` are Codable. The generic EKF takes the
caller-supplied model matrices/closures rather than imposing a motion model.
`ConstantVelocityKF` is an optional thin example helper; its process-noise value
and per-update measurement noise are caller-supplied model parameters.

The fusion config covers gaze noise/confidence, gaze timeout/reorder allowance,
position/velocity/bias process noise, velocity decay, bias transfer length, saccade
gates and confirmation window, relative-delta noise, modes, clamping margin,
confidence scale and bounded history capacity. Head config covers axes/signs,
rate source, deadband/gain/acceleration/cap, velocity noise and timeout. Arbiter
config covers confidence by source, lookback, debounce, duplicate window and
click-versus-drag duration/travel. `FusionTunables.all` lists the deliberately
limited search ranges; it is not a second processing path.

## Simulation model and evidence level

The defaults use a 1512 by 982 point display and a centered 900 point square grid,
30 by 30 cells, 60 seconds, 30 Hz gaze capture, 25 Hz head samples and 60 Hz display.
The human model uses delayed cursor perception and target geometry, never a
ground-truth click placement. It schedules a click after perceived on-target dwell
or an explicit impatience timeout. A click can miss or be rejected by the arbiter.

Eye motion includes approximately 200 ms target latency, amplitude-dependent
saccade duration, undershoot/correction, OU fixation drift, microsaccades and
blinks. Gaze error includes a smooth spatial calibration-bias field, temporal bias
drift, correlated and white sample noise, head leakage, camera latency/jitter,
dropout and blink-edge artefacts. The head controller sees a delayed displayed
cursor, commands intermittent corrections and includes motor lag, signal-dependent
noise, tremor, excursion/rate limits and noisy delayed IMU samples. The hand
controller uses a delayed relative-motion clutch. Pinch and jaw clicks have
separate latency, miss and false-positive assumptions.

The head controller inverts the configured pointer gain as a trained-user
feed-forward model. It therefore assumes adaptation to each candidate gain; the
search does not measure learnability or novice performance. Gaze-only presets use
delayed compensatory gazing, rather than silently giving them a head controller.
The mouse-like preset is a Fitts-law sanity baseline with endpoint scatter, using
the same click/arbiter path; it is not a measured mouse score.

The three noise levels are engineering scenarios. In particular, the typical
1.5-degree spatial bias, 0.7-degree per-frame noise, 0.008 rad/s gyro noise, sensor
latencies and click detection statistics are **assumptions**, not measurements of
Apple Vision, AirPods 3, or this app. At the assumed viewing geometry, 50 pt/degree
is a convenient approximation, not a display-independent conversion.

## References

1. Casiez, Roussel and Vogel, CHI 2012, *1 Euro Filter: A Simple Speed-based
   Low-pass Filter for Noisy Input in Interactive Systems*. Author-maintained
   [implementation](https://github.com/casiez/OneEuroFilter) and
   [project](https://gery.casiez.net/1euro/). Supports the adaptive-cutoff algorithm.
2. [Webgrid Eval source and formula verification](https://github.com/ofou/webgrid_eval).
   The documented frontend expression is
   `Math.max(Math.log2(t*t-1)*f/60,0)`. No live browser/game was operated.
3. Harris and Wolpert (1998), *Signal-dependent noise determines motor planning*,
   [author paper mirror](https://www.cs.cmu.edu/~jkh/gnhm/signal_dependent_noise.pdf),
   [DOI](https://doi.org/10.1038/29528). Supports increasing motor noise with command
   magnitude, not the simulator's particular motor-noise coefficients.
4. MacKenzie (1992), *Fitts' Law as a Research and Design Tool in Human-Computer
   Interaction*, [author PDF](https://www.yorku.ca/mack/hci1992.pdf). Supports the
   distance/width movement-time relationship; mouse coefficients here are assumed.
5. Papoutsaki et al. (2016), *WebGazer: Scalable Webcam Eye Tracking Using User
   Interactions*, [original paper](https://ijcai.org/proceedings/2016/540).
   Reports webcam regression errors in pixels; it does not validate this model's
   split into 1.5-degree bias and 0.7-degree frame noise.
6. Krafka et al. (2016), *Eye Tracking for Everyone*,
   [CVPR paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Krafka_Eye_Tracking_for_CVPR_2016_paper.html).
   Consumer-camera gaze-estimation context; its mobile CNN is not Apple Vision
   landmarks and is not a direct accuracy guarantee for this implementation.
7. Harwood, Mezey and Harris (1999), *The Spectral Main Sequence of Human Saccades*,
   [original research](https://pmc.ncbi.nlm.nih.gov/articles/PMC6782780/). Reports
   duration/amplitude slopes of 2.22-3.37 ms/degree and intercepts of 20.3-30.4 ms
   for 4-20 degree horizontal saccades in ten participants. The simulator's
   `21 ms + 2.2 ms/degree` is an approximate prior within this family, not an exact
   universal eye-motion law.
8. Trottier and Pratt (2005), *Visual processing of targets can reduce saccadic
   latencies*, [paper](https://www.cis.rit.edu/pelz/lab/papers/trottier_pratt_saccadic_latencies.pdf).
   Discusses the conventional roughly 200 ms reaction time and task-dependent
   reductions. Reaction latency is separate from the shorter saccade transit time.
9. Engbert and Mergenthaler (2006), *Microsaccades are triggered by low retinal
   image slip*, [paper record](https://pubmed.ncbi.nlm.nih.gov/16632611/). Supports
   microsaccades during fixation; the chosen OU model and amplitudes are tunable
   simplifications of that behavior.

## Verification and benchmark results

The full preset/noise matrix is in [benchmarks.md](../benchmarks.md). CLI examples:

```sh
swift run --disable-sandbox -c release gazefuse-bench sim --preset gazeHeadEKF --noise typical --grid 30 --seconds 60 --seeds 10
swift run --disable-sandbox -c release gazefuse-bench tune --preset gazeHeadEKF --noise typical --train-seeds 6 --test-seeds 6 --iterations 20 --passes 1 --output /tmp/fusion.json
swift run --disable-sandbox -c release gazefuse-bench all --config /tmp/fusion.json --seed-start 101 --seeds 10
swift run --disable-sandbox gazefuse-bench replay session.jsonl
```

In this restricted worktree, the default global Clang module cache is unwritable.
Build/test commands use these environment overrides, without changing Package.swift:

```sh
export CLANG_MODULE_CACHE_PATH=/private/tmp/gazefuse-fusion-module-cache
export SWIFTPM_MODULECACHE_OVERRIDE=/private/tmp/gazefuse-fusion-module-cache
swift build --disable-sandbox
swift test --disable-sandbox
```

The installed SwiftPM defaults to the Swift Build engine. Release linking succeeded
but its dSYM generation was denied by the sandbox. `swift build --disable-sandbox
-c release -debug-info-format none` completed successfully and supplies the optimized
binary used for benchmarks. This is an environment flag, not a source workaround.

Deterministic numerical filter results:

| Measurement | Observed value |
|---|---:|
| One Euro rest input RMS | 3.034129 |
| One Euro rest output RMS | 0.803206 (73.5% reduction) |
| One Euro fast-move mean absolute error | 4.848813 |
| Linear EKF position RMSE | 0.213642 |
| Linear EKF velocity RMSE | 0.427482 |
| Nonlinear range/bearing final position error | 0.061850 |
| Mean NIS for a consistent scalar system (expected 1) | 0.987220 |
| NIS fraction inside 95% chi-square gate | 95.125% |

The first parameter search evaluated 55 candidates with the initial implementation:
training and held-out BPS both remained zero. This exposed the clipped-score plateau.
The tuner now breaks equal-BPS ties using NTPM, preserving BPS as the primary
objective and accepting no BPS decrease. All searches use disjoint training and
held-out seeds. Model repairs and parameter-search effects are reported separately
in the benchmark artifact.

## Integration limitations

- No camera/AirPods/CGEvent behavior was exercised. Verify rate axes and signs,
  timestamp convention and actual display dimensions before enabling cursor output.
- The app must call `cursor(at:)` at display cadence to populate useful onset
  history and emit dragged cursor movement while arbiter buttons are held.
- `Recording.swift` has no relative-delta event. Simulated hand deltas cannot be
  reproduced from the simulator's optional recording alone. The current replay
  hook can derive them from raw hand frames. Suggested shared-contract extension:
  a timestamped screen-point relative-delta event, if exact hand replay is needed.
- The simplified square-grid simulator does not reproduce browser layout,
  distractors, target-recognition effort or every Webgrid click-type mode.
- The benchmark validates this software/model path only. Human trials and captured
  device recordings are the next evidence needed to set real defaults.
- Files are left uncommitted for the integrator, per the worktree sandbox instruction.

## Integrator changes (after the module hand-off)

* `SimHuman`: continuous head servo (oscillated at ~0.35 s delay) replaced by intermittent minimum-jerk
  submovements + gain adaptation; hand clutch uses take-off release as the click; clicks only after the
  perceived cursor settles inside the inner 60% of the cell.
* `EKFCursorFusion`: `saccadeMinConfidence` (0.6) — blink-edge samples can't reset; `GazeHeadLeakCompensator`
  (fixation-differenced forgetting LS, clamped to 0.9 × pointer gain); innovation-adaptive gaze σ
  (α 0.02, max 120 pt).
* `DefaultTuning` from `gazefuse-bench tune --write-defaults` (train 4.956→5.238, held-out 4.783→5.129 BPS).
* Simulated gazeHeadEKF/typical/30×30: 0 → 2.02 → 2.67 → 3.38 → 3.79 → 5.32 BPS (20 seeds from 3000). See docs/benchmarks.md.
* Regression gate test: `FusionLeakAdaptiveTests.testSimulatorRegressionGate` (typical > 4 BPS, acc > 0.9).
* Saccade `confirmSamples` 1 beats 2 by ~0.7 BPS in simulation; 2 may be more robust to real outliers.
