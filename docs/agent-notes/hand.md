# Hand gestures

## Scope and design

The module converts one unmirrored Vision hand observation into left/right button
events and relative pinch motion. It does not move the cursor, choose a gaze target,
or implement the click arbiter. Shared contracts and package configuration are unchanged.
All classifier/extractor calls belong on the camera's single serial queue.

`HandPoseExtractor` owns a `VNDetectHumanHandPoseRequest`. The default maximum hand
count is one. With multiple observations, selection combines mean joint confidence,
a preferred-chirality bonus and proximity to the last selected palm. This is a small
selection heuristic, not persistent hand identity tracking. Missing detections clear
the extractor's previous palm. Every output contains the 21 joint names; unavailable
or non-finite points have zero confidence. Mean confidence uses all 21 entries so
coverage affects the score. Candidates need usable, noncollapsed palm geometry.

`HandGestureFeatures` uses

```text
handSize = max(distance(wrist, middleMCP), distance(indexMCP, littleMCP) / 0.72)
pinchDistance = distance(thumbTip, fingerTip) / handSize
curl = distance(MCP, tip) / sum(projected phalanx lengths)
```

Either palm axis can supply scale when the other is unavailable. Translation,
uniform scale and in-plane rotation do not change these idealized 2D features.
Taking the larger palm axis reduces foreshortening sensitivity but does not make
the classifier invariant to arbitrary 3D poses. Pair confidence includes both tips
and the mean palm confidence. Finger states are extended/curled/unknown; open palm
requires extended fingers and thumb, with the thumb away from both candidate tips.

`PinchGestureClassifier` smooths distances with a speed-adaptive One-Euro-style
low-pass filter. A distance jump above `maxDistanceRate` is ignored for one frame;
if it persists it is accepted. On/off hysteresis and consecutive-frame debounce
produce `.down`/`.up`. Index means left; middle means right. Raw contact distances
must resolve close-pair ambiguity before filtered distances and geometry can select
a button. Only one button can be active until release. Curled-finger and nearest-tip
checks reject a thumb wrapped over a fist. Normalized distances reject rigid
translation as pinch evidence; excessive palm speed also blocks onset under blur.

New pinches require a previously observed open pair. A hand entering already
pinched cannot click. Up to two missing/low-confidence frames preserve an active
drag but emit no motion. Longer loss emits one release, clears tracking state and
requires opening before another click. A missing frame clears incomplete onset
and release debounce. Brief gaps do not restart the presence warmup.

Pinch motion is a smoothed tip-midpoint delta in normalized image coordinates;
the palm center is an optional reference. Onset has no motion delta. The default
path immediately emits down after debounce and supports ordinary tap or hold/drag.

Optional scroll mode defers left down: sufficiently vertical palm movement becomes
scroll; horizontal movement or a 0.4-second still hold becomes a drag. A short still
pinch emits down/up together on release. Once a drag has started, vertical movement
remains a drag. Right pinches always use button events. Scroll mode defaults off.

## Research and algorithm references

- [Apple gestures HIG](https://developer.apple.com/design/human-interface-guidelines/gestures)
  and [visionOS interaction overview](https://developer.apple.com/visionos/get-started/):
  gaze plus pinch motivates separating target selection from button intent. This
  webcam implementation does not claim visionOS tracking quality or access to its
  private recognition algorithms.
- [Apple WWDC20: Detect Body and Hand Pose with Vision](https://developer.apple.com/videos/play/wwdc2020/10653/):
  low-confidence landmark rejection and multi-frame pinch evidence; higher requested
  hand counts cost latency. Edge occlusion, edge-on hands, gloves and occasional
  false hand detections remain upstream limitations. This module adds normalized
  scale, hysteresis, button arbitration and explicit loss/re-entry behavior.
- [Vision hand observations](https://developer.apple.com/documentation/vision/vnhumanhandposeobservation)
  and [maximumHandCount](https://developer.apple.com/documentation/vision/vndetecthumanhandposerequest/maximumhandcount):
  the platform API supplies 21 joints, joint confidences and chirality. We override
  its two-hand default with one for the requested camera-loop integration.
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker):
  a contemporary landmark-based pipeline also separates hand detection, landmark
  estimation and handedness. GazeFuse uses Vision without adding a model/dependency;
  this classifier only receives projected 2D landmarks, not MediaPipe world landmarks.
- [Casiez, Roussel and Vogel, CHI 2012, 1€ Filter](https://gery.casiez.net/1euro/):
  cutoff increases with filtered speed to balance resting jitter against movement
  lag. The 2D helper uses one speed magnitude/cutoff for both axes.

## Public API and integration

```swift
let hand = HandPoseExtractor() // Or init(maximumHandCount: 2, preferredChirality: "right")
let classifier = PinchGestureClassifier()

try handler.perform(face.requests + hand.requests)
let sample: HandSample? = hand.extract(t: t)
let events = classifier.process(sample, at: t)
let imageDelta: Point2? = classifier.pinchMotionDelta
let pauseSignal = classifier.isOpenPalm
```

- `HandPoseExtractor.Config`: `Codable, Equatable`; request/selection tunables.
  `requests`, `extract(t:)`, `reset()`, `lastObservationCount` support the camera loop.
- `PinchGestureClassifier`: conforms to `HandGestureClassifying`; `Config` is
  `Codable, Equatable`. Additional accessors are `pinchButton`, `isOpenPalm`,
  `lastFeatures`, `filteredIndexDistance`, `filteredMiddleDistance`, `handSpeed`,
  `activeThresholds` and smoothed curl ratios.
- `flush(at:)` emits an up for a held button, cancels pending scroll/tap intent and
  forgets tracking state. Deliver its events before stopping camera processing.
  `reset()` silently clears state and must not replace a required button release.
- `HandGestureFeatures(sample:config:)` returns nil without usable palm scale.
  It exposes the two pinch pairs, palm size/center/confidence and finger states/curls.
  `HandFinger` and `HandFingerState` name the finger vocabulary.
- `HandOneEuroFilter` and `HandOneEuroFilter2D` are allocation-free value-type helpers.

Feed nil detections on every missing camera frame. Use monotonic seconds for `t`.
All events use `.hand`; down timestamps refer to the first filtered candidate frame
of the confirmed pinch, whereas up/scroll use the current frame. Down timestamps
are inferred onset times, not measured contact times. Integrators should distinguish
event emission latency from this retrospective timestamp.

Image motion has origin bottom-left and +y up. The fusion/app integrator must choose
the screen gain, invert y as appropriate and apply any intended camera mirroring.
Scroll deltas are points, with positive y for upward image motion. Never apply
`pinchMotionDelta` during scroll mode; the classifier supplies nil in that phase.
Open palm is only an accessor; this module does not stop tracking or emit a pause
event. Continue feeding frames if using it as a clutch. Change configuration between
gestures or flush before replacement. Frequencies must be positive, on must be below
off, and rates/gains must be nonnegative.

## Main defaults and tunables

| Config group | Defaults / units |
| --- | --- |
| Extractor | maximum hands 1; preferred chirality nil; preference bonus 0.25; consistency bonus 0.5/radius 0.25 image units; mean confidence 0.2; palm joint confidence 0.1 |
| Features | joint confidence 0.3; palm width/length 0.72; extended curl >= 0.85; curled <= 0.55; open-palm thumb separation >= 0.6 hand sizes |
| Pinch | on 0.28, off 0.42, button margin 0.06 hand sizes; onset confidence 0.45; 2 on / 2 off / 3 initial presence frames; 2 missing frames tolerated; 0.08 s refractory period |
| Motion gates | onset palm speed <= 3 hand sizes/s; speed EMA 0.5/frame; distance jump limit 30 hand sizes/s; curl veto and nearest-tip check enabled; curl EMA 0.35/frame |
| Filters | distance cutoff 4 Hz/beta 1; motion cutoff 1.5 Hz/beta 5; derivative cutoff 1 Hz; pinch-point motion reference |
| Scroll | off; activation 0.15 hand sizes; vertical/horizontal ratio 1.5; deferred hold 0.4 s; gain 400 points/hand size |
| Clutch | 3 consecutive open-palm frames |

## Known limitations and next steps

These are synthetic classifier results, not webcam accuracy, accessibility efficacy,
Webgrid throughput or a medical-device evaluation. Vision confidence is not a
calibrated probability of intentional pinching. No hardware, AirPods, real cursor
output or app camera loop was exercised by these tests.

2D overlap can resemble physical contact; severe foreshortening or unusual pinch
postures can defeat curl/nearest-tip checks. Distances use normalized image axes;
camera aspect ratio is not present in `HandSample`, so aspect distortion is not
corrected here. Short gaps can conceal a real release/re-pinch, and a gap longer
than the configured tolerance intentionally interrupts dragging. Frame-based
debounce/dropout/EMA parameters have different durations at 30 versus 60 fps.

Multiple overlapping hands can exchange identity; preference/proximity scoring is
not an identity tracker. `maximumHandCount = 1` cannot choose a preferred hand that
Vision did not return. Optional scroll adds intentional press latency and cannot
be chosen after a drag is already active.

Next: record consented webcam samples with the final camera orientation/aspect
ratio; check small/large hands, lighting, backgrounds, left/right handedness and
clutch ergonomics. Tune on those recordings before claiming real-world precision
or Webgrid BPS. No shared-contract change is required for the current integration.

Files are left uncommitted for the integrator as requested by the resumed-task
instructions; shared git metadata was not modified.

## Iteration 2: near-miss learning for adaptive thresholds

The old adaptive mode only learned the pinched distance from pinches it had already detected
(chicken-and-egg), so a "wide contact" user (camera angle makes touching tips look 0.32 hand-sizes
apart) went from recall 0.183 (fixed) to only 0.317. Near-miss learning treats an idle dip of the
thumb–index distance below 0.5 × open distance, with tip-nearest / not-curled pose at its minimum,
lasting 0.05–1.0 s and returning to open without a trigger, as an attempted pinch; the median of
the last ≤5 such minima pulls the pinched estimate up (after ≥2 near misses).

* wide contact 0.32, 60 taps: fixed recall 0.183 → adaptive 0.317 (old) → **0.933** (precision 1.000).
* 3 contact gaps (0.30/0.34/0.38) × 3 seeds × 40 taps, each followed by 12 rounds of non-pinch
  motion in the same stream: fixed recall 0.250 → adaptive **0.947**, adaptive FP 3 (all in "wave"
  segments whose finger aperture only opens to 0.45 — inherently close to a 0.38 "contact").
* Normal users with adaptive on: recall 1.000, precision 1.000; non-pinch suite 0 FP.
* Still default **off**; menu: Input Channels → "Adaptive pinch thresholds".
