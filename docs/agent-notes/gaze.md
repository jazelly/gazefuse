# Gaze estimation and calibration

## Scope and integration

This module implements a native, landmark-based personal calibration path. It
does not include pretrained neural weights or claim a measured human accuracy.
The shared interfaces, package manifest, recording code, and matrix implementation
are unchanged. Camera capture and the camera loop remain the integrator's work.

The camera image must be unmirrored, `.up`, with the size passed to `extract`
matching its pixel buffer. Vision image coordinates have a bottom-left origin;
gaze predictions and calibration targets have the global CG top-left origin, in
display points rather than Retina pixels.

```swift
let extractor = FaceGazeExtractor()
let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up, options: [:])
try handler.perform(extractor.requests + otherRequests)
let sample = extractor.extract(t: t, imageSize: CGSize(width: w, height: h))
```

`FaceGazeExtractor.Config(refinePupil: true)` enables optional local Y-plane
refinement; call `extractor.prepare(pixelBuffer: pb)` before performing the
requests for that frame. The normal API works without the additional call.
Refinement is a dark-pixel centroid seeded by Vision, not an implementation of
Timm and Barth's gradient-voting algorithm. It is disabled by default pending
camera validation under glasses, shadows, eyelashes, and reflections.

`CalibrationPlan(screen:layout:config:)` generates the five, nine, thirteen, or
sixteen target geometry and a separate validation lattice. Its seeded shuffle
only changes presentation order. Use the actual CG rectangle of the selected
display; `CalibrationWindowController` verifies/converts the AppKit coordinates.
The controller's API is `init(screen:mapper:plan:)`, `start()`, `cancel()`,
`feed(_:)`, and `onFinish`. `feed` can be called from any queue; mapper calls
inside calibration execute on main. Pause the ordinary mapper consumer during
calibration, then resume on its single serial queue after `onFinish`; the mapper
does not add locks or background workers.

## Algorithm

`RidgeGazeCalibrator` implements `GazeMapper`. Binocular pupil positions, head
yaw/pitch/roll, face center/size, and a bounded stable prefix of extractor extras
form a small polynomial feature vector. Fitting uses `Matrix.choleskySolve` for
two screen-coordinate outputs. The intercept is not ridge-penalized.

The first 300 ms of each fixation, blink frames, low-confidence frames, and
within-target robust outliers are excluded. Short temporal median aggregation
reduces landmark noise while retaining head-motion variation within a target.
All observations from a target are held out together when selecting lambda, so
repeated frames of the same fixation cannot leak into its training fold. The
reported errors are target-level errors of median predictions; they are not
single-frame jitter or an independent estimate of hyperparameter-selection bias.

`addCorrection(sample:truePoint:)` maintains a bounded window of click residuals
and updates the coordinate-wise median output bias. This intentionally learns
systematic offset, not a new nonlinear gaze mapping. The caller must supply a
confirmed target that the user was actually looking at; every click is not
necessarily a valid gaze label. Invalid or blink corrections are ignored.

The fitted `RidgeGazeCalibrator.Model` is Codable. Encode `mapper.model` with
`JSONEncoder`, decode `Model`, then construct `RidgeGazeCalibrator(model:)`.
Weights, feature-basis choices, selected lambda, and learned correction bias
travel with the model. The saved bias seeds subsequent click correction; the
entire recent-residual history is not persisted. Preserve the runtime `Config` alongside it when using
nondefault blink/confidence thresholds. Calibration is specific to the user,
camera arrangement, and display placement.

## Research and applicability

- [WebEyeTrack (Davalos et al., 2025)](https://arxiv.org/html/2508.19544v1)
  combines metric head pose, eye patches, a lightweight BlazeGaze CNN, and
  on-device few-shot adaptation. It reports 2.32 cm on GazeCapture and 2.4 ms
  inference on iPhone 14. The transferable ideas here are pose-aware features,
  personal calibration, blink suppression, and explicit click corrections. Its
  pretrained representation and benchmark results are not reproduced by this
  landmark-only model.
- [MediaPipe Iris (Google Research)](https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/)
  estimates iris/eye contours and camera distance from RGB images. Google
  explicitly distinguishes iris tracking from inferring where a person looks.
  Vision landmarks serve as the available native input here; personal screen
  mapping remains necessary.
- [ETH-XGaze (Zhang et al., ECCV 2020)](https://arxiv.org/abs/2007.15837)
  supplies over one million images spanning large head/gaze variation. Its
  appearance-based baseline and normalization motivate collecting pose diversity
  and testing on held-out poses. This module does not download its dataset or
  weights.
- [L2CS-Net (Abdelrahman et al.)](https://arxiv.org/abs/2203.03339)
  uses separate classification/regression objectives for gaze angles. Its
  original abstract reports 3.92 degrees on MPIIGaze and 10.41 degrees on
  Gaze360, illustrating that a single 2–4 degree claim cannot be generalized
  across datasets or hardware.
- [Timm and Barth, VISAPP 2011](https://doi.org/10.5220/0003326101250130)
  localize the eye center by agreement of image gradients with candidate-center
  directions. The current optional centroid is cheaper and simpler; it does not
  inherit that paper's robustness claims.
- [Apple Vision landmark coordinates](https://developer.apple.com/documentation/vision/vnfacelandmarks2d)
  are relative to the detected face bounding box. The face-landmark request
  already performs face detection, so no second rectangle request is needed.

A future CoreML path should first choose appropriately licensed model weights
and reproduce their exact face/eye crop normalization. Run a small appearance
encoder on the camera queue, place its fixed-length gaze angles or embedding in
`GazeSample.extra`, and implement a separate `GazeMapper` with a personal
calibration head. The current `GazeMapper` receives features rather than images,
so an image CNN cannot be placed inside it without a camera-side encoder or a
shared-contract change. Validate held-out people, glasses, illumination, head
motion, physical angular error, and end-to-end latency before adopting it.

## Validation results

On Apple Silicon with Swift 6.4.2, Swift 5 language mode, macOS 14 deployment
target, and the installed macOS 27.2 SDK, the complete XCTest run passed **10
tests, 0 failures** (8 gaze tests and 2 existing matrix/recording tests), in
44.84 seconds. The app target also builds, including Vision and AppKit code.

The deterministic fixture projects a spherical pupil through `sin(eye rotation)`
after subtracting head pose; screen angles come from viewing geometry, not from
the mapper's fitted polynomial. It uses a 1440 × 900 point display at a nonzero
global origin, 0.22 mm/point, nominal 650 mm viewing distance, head translation,
depth changes, roll, Gaussian landmark noise (0.0007 eye widths), 30 Hz samples,
and 267 ms target-acquisition latency. Each of 16 targets has 45 frames. The
contaminated version adds blinks, one-eye dropouts, and gross pupil outliers.
Evaluation uses 240 independent, continuously distributed screen targets and
unseen motion phases; the head-pose ablation increases yaw/pitch amplitude from
0.055 to 0.09 radians.

| Measurement | Result |
| --- | ---: |
| Clean held-out mean / p95 | 11.364 / 23.776 pt |
| Contaminated held-out mean / p95 | 11.238 / 23.865 pt |
| Clean target-grouped CV mean | 6.642 pt |
| Contaminated target-grouped CV mean | 6.460 pt |
| Accepted contaminated frames | 471 / 720 |
| Larger unseen head motion, pose features on / off | 14.499 / 237.307 pt mean |
| New systematic bias, before / after 32 corrections | 87.863 / 11.397 pt mean |
| CV for lambda 0.00001 / 0.01 / 100 | 6.832 / 8.287 / 521.321 pt |
| Lambda selected from those three candidates | 0.00001 |
| Debug inference mean, 10,000 calls | 16.330 microseconds |
| Round-trip model JSON, head features off with nonzero correction | 1,322 bytes |

Tests also verify blink/missing-eye confidence, ignoring invalid click
corrections, rejection of onset-only calibration clips, preserving feature-basis
choices and correction bias through Codable, seeded target order, requested
layout counts, margins, negative display origins, and disjoint validation points.

These are **synthetic accuracy and mapper-only timing**, not human accuracy or
camera-pipeline latency. Optional pupil-refinement timing, its requested sub-2-ms
budget, the visible AppKit interaction, and physical display geometry remain
unmeasured. Debug fitting currently blocks its caller for several seconds during
CV; the calibration UI deliberately performs this small V1 workflow synchronously.

The sandbox denies the default `~/.cache/clang` location, so validation used:

```sh
CLANG_MODULE_CACHE_PATH=/tmp/gazefuse-gaze-module-cache \
SWIFT_MODULECACHE_PATH=/tmp/gazefuse-gaze-module-cache \
swift test --disable-sandbox --cache-path /tmp/gazefuse-gaze-swiftpm-cache

CLANG_MODULE_CACHE_PATH=/tmp/gazefuse-gaze-module-cache \
SWIFT_MODULECACHE_PATH=/tmp/gazefuse-gaze-module-cache \
swift build --disable-sandbox --cache-path /tmp/gazefuse-gaze-swiftpm-cache
```

## Tunables

All tunables are Codable in the owning type's `Config`; constants that define
the feature basis are part of the current implementation.

| Type | Defaults |
| --- | --- |
| `RidgeGazeCalibrator.Config` | 0.30 s onset exclusion; 0.16 blink openness; 0.10 minimum confidence; 5 MAD outlier cutoff; 240 frames/target; median bins of 3; first 12 extras; lambda candidates 1e-5 through 10; 0.01 if candidate list is empty; pose and polynomial features enabled; last 32 click residuals |
| `FaceGazeExtractor.Config` | refinement off; radius 0.34 eye widths; pixel stride 2; 18 luminance-unit contrast; minimum dark mass 8; minimum face confidence 0.05 |
| `CalibrationPlan.Config` | 8% margins; seeded shuffled order; 1.5 s/target; 5 validation points at 0.8 s each; 0.3 s validation settling; 2 s summary; angle assumptions 344 mm display width and 600 mm viewing distance |

The extractor emits 24 extras: eye-axis pupil coordinates (0–3), normalized eye
centers (4–7), widths (8–9), openness (10–11), normalized interocular distance
(12), interocular x/y displacement (13–14), face center/width (15–17), final
image-axis pupil offsets (18–21), and pupil-presence flags (22–23). The default
mapper uses only the first twelve; increase `maxExtraFeatures` explicitly to
evaluate the remainder. Missing values are zero-filled, so keep the schema
stable across calibration and use.

## Delivery status

All changes are limited to the gaze-owned source/test paths and this note. No
push was attempted. The requested commit could not be created: `git add` failed
with `Operation not permitted` creating
the worktree `index.lock`.
That Git directory is outside the writable worktree. The source remains
available for the integrator to stage and commit on `agent/gaze`.

## Known limitations and next steps

- A landmark-only personal model cannot recover eye appearance or gaze depth
  that Vision did not measure. A narrow head-pose calibration does not establish
  accuracy outside that pose range.
- The default blink threshold must be checked against actual Vision contours;
  an individual squint or eye morphology can require tuning.
- One visible eye reduces confidence; missing both eyes or a bilateral blink
  yields no estimate, as specified by `GazeMapper`.
- Online residual correction changes only output bias and does not certify a
  saved model after camera/display rearrangement.
- Screen-point validation and a viewing-distance-based degree estimate serve
  different purposes. The displayed degrees remain approximate until the actual
  viewing distance and panel dimensions are measured.
- Next real run: integrate the camera loop, perform a sixteen-point calibration
  with small natural head movements, inspect the independent validation pass,
  then measure Webgrid with the fusion/click modules. No Webgrid BPS result is
  claimed by this module.

## Iteration 2: pupil refinement + appearance model

- `PupilLocator.timmBarth` (Timm & Barth 2011, gradient dot-product objective, weighted by inverted gaussian intensity)
  on the eye crop from the camera Y plane, masked by the Vision eye contour. Synthetic error (eye-widths, mean):
  48 px clean 0.0082 vs dark-centroid 0.0157; with glint+lashes+noise 0.0139 vs 0.0357; 32 px 0.0219 vs 0.0452.
  17.85 ms/eye (debug). **On a real low-res clip (median eye 19.5 px) it increased jitter 0.0120 → 0.0456 eye-widths**, so
  `FaceGazeExtractor` gates it (eye ≥ 32 px, distinctness ≥ 1.6, inside contour, shift ≤ 0.15); gated jitter 0.0120 → 0.0120.
- `GazeSample.appearanceGaze` (yaw, pitch rad) from `AppearanceGazeEstimator` (CoreML, L2CS-style 90×4° bins → softmax
  expectation, converted by `scripts/convert_gaze_model.py`; weights Gaze360 non-commercial → not shipped).
  2.55 ms/frame on the real clip; noisy on small faces (median frame-to-frame change 7.53°).
- `RidgeGazeCalibrator` appends `[ax, ay, ax², ay², ax·ay]` and keeps the appearance head only if leave-one-target-out
  CV beats landmark-only (needs ≥60% of samples with appearance). Synthetic held-out: landmark σ0.05 266.6 pt →
  +appearance 2.5° 161.6 pt; TB pupils 70.3 pt (appearance then dropped by CV, 15.3 vs 14.9).
- Self-test: `GazeFuse --gaze-video FILE [--appearance-model FILE]`, env `GF_DUMP_DIR`, `GF_MIN_EYE_PX`.
