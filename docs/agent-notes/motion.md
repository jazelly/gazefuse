# Motion module

## Scope and integration

This branch implements the requested headphone adapter, experimental jaw-click
detector, and relative head-pose utility. The shared interfaces and package are
unchanged. There are no dependencies or hardware access in `GazeFuseCore`.
Each mechanism below serves an explicit motion-module acceptance criterion;
there is no background retry, recording, scheduling, or persistence machinery.

Delivery state: implementation and verification are complete, but the agent's
sandbox denied `git add` because the linked worktree's Git metadata is outside
its writable roots. The exact failure was inability to create
`gazefuse/.git/worktrees/gazefuse-wt-motion/index.lock` (Operation not permitted).
No commit or push was made. From a shell with access to that metadata, stage only
these six files and commit on `agent/motion`:

```sh
git add Sources/GazeFuseCore/Motion/HeadPoseTracker.swift \
  Sources/GazeFuseCore/Motion/JawClenchDetector.swift \
  Sources/GazeFuse/Motion/HeadphoneMotionSource.swift \
  Tests/GazeFuseCoreTests/MotionHeadPoseTests.swift \
  Tests/GazeFuseCoreTests/MotionJawClenchTests.swift \
  docs/agent-notes/motion.md
git commit -m "Add headphone motion source and calibrated jaw click detection"
```

The adapter must be used from a bundled macOS app with
`NSMotionUsageDescription` set by the integrator. It does not request camera or
Accessibility permission and does not emit CGEvents. The app must route samples
to the fusion module and the detector; it must route returned jaw events through
the existing click arbiter. This branch does not wire the app's `main.swift`.

## Algorithms and public API

### HeadphoneMotionSource (app target)

`HeadphoneMotionSource: NSObject, CMHeadphoneMotionManagerDelegate` exposes
`start()`, `stop()`, `onSample`, `onStatusChange`, static `authorizationStatus`,
and `isDeviceMotionAvailable`. Samples and status callbacks share a dedicated
serial `OperationQueue`. Set callbacks before the first start and keep them stable
for the source lifetime; consumers that update
AppKit UI must dispatch to the main queue. Core stateful consumers must likewise
receive all their calls on one serial queue.

Every `HeadMotionSample` includes Euler attitude, quaternion in x/y/z/w order,
rotation rate (rad/s), gravity and user acceleration (g). The source copies
`CMDeviceMotion.timestamp` unchanged. Apple documents `CMLogItem.timestamp` as
seconds since boot; it is the uptime clock used by the shared contract, not wall
time or handler arrival time. No offset is added, and delivery latency is retained.
Actual AirPods timing/latency across connection and system sleep needs a device run.
The public headphone API does not offer a configurable sample interval; this
module handles observed timestamps rather than assuming exactly 25 Hz.
Connection-status observation remains active while waiting or disconnected;
connect restarts motion delivery. Apple does not expose separate availability
flags for host support and absent headphones. An explicit not-available error
can therefore report `.unavailable` while still listening for a later connection.
This status distinction needs a hardware check.

### JawClenchDetector (core target)

`JawClenchDetector(config:)` conforms to `JawClenchDetecting`:

- `process(_:) -> [InputEvent]` emits `.click` from `.jaw`.
- `reset()` cancels temporal evidence and pending clicks, preserving calibration.
- `calibrate(restSamples:clenchSamples:) throws -> MotionJawCalibrationReport`
  measures rest and isolated guided bursts and updates `config.minimumRMS`.
- Read-only `config`, `currentRMS`, and `currentThresholdRMS` support setup UI.

Acceleration has already had gravity removed by Core Motion. A time-aware
one-pole high pass additionally removes slow movement, followed by an exact
elapsed-time-weighted RMS window. The threshold is the greater of an absolute
floor and median + `thresholdZ` times robust noise scale (1.4826 * MAD, with a
small scale floor). Quiet samples update a bounded-in-time noise history;
median/MAD are refreshed at 5 Hz using reused scratch storage.

A candidate must have sufficient duration and multiple samples supporting a
real acceleration excursion. Raw support includes samples preceding the RMS
threshold crossing, because the energy window delays that crossing. The maximum
duration bounds the raw excursion rather than counting the filter's decay tail.
It is emitted only after energy falls below a
hysteretic release threshold and remains quiet. Gross gyro/acceleration motion
cancels a candidate and starts a quiet cooldown. Excessively long activity and
closely repeated candidates are rejected. Nonincreasing timestamps are ignored;
long sample gaps discard unfinished gestures and restart noise warmup.

Double-clench mode is opt-in. Two accepted onsets within 500 ms emit one right
click. A single candidate waits for that deadline and emits one left click; it
does not leak an initial left click before the right click. Samples must continue
arriving to settle pending events; reset/disconnection discards them. Event time
is gesture onset, not delivery time, so the arbiter can use preceding cursor history.
Confidence represents threshold evidence strength in 0...1, not a learned
probability that a user intended a click.

Calibration applies the same feature filter to guided rest and clench streams.
It measures rest median/MAD and the lower quartile of separated burst peaks,
then chooses a threshold above resting noise and below the representative peak.
Insufficient samples or a signal that cannot separate from noise throw a clear
`MotionJawCalibrationError`; failed calibration leaves runtime settings unchanged.
After successful calibration, feed fresh quiet samples for runtime warmup.

### HeadPoseTracker (core target)

`HeadPoseTracker(config:)` offers `process(_:) -> HeadPoseEstimate`, `recenter()`,
and `reset()`. The first valid sample establishes the yaw/pitch zero. Wrapped
angular differences and an exponential smoother handle irregular intervals;
a soft deadband suppresses tiny changes. The estimate includes relative angles,
per-sample deltas, validity and gap status. A long gap reports a new endpoint
without inventing a delta for the unseen path. Recenter uses the latest raw pose.

Slow zero drift correction requires sustained stillness and proximity to zero,
so large deliberate held offsets are preserved. Small deliberate held offsets
inside the capture window can still be absorbed: disable drift correction when
absolute relative posture matters. Euler angles have the usual singularity near
vertical pitch. Fusion should continue to use the sample's rotation rate directly.
Optional nod/shake gestures are deferred; they are not needed for cursor input.

## Default tunables

Both detectors expose `Codable` Config structs for the integrator to save.

| Jaw parameter | Default |
| --- | --- |
| High-pass time constant / energy window | 0.12 s / 0.12 s |
| Noise history / warmup / statistics update | 6 s / 0.8 s / 0.2 s |
| Robust scale floor / minimum RMS / threshold z | 0.00015 g / 0.006 g / 6 |
| Release excess ratio | 0.55 |
| Minimum energy span / maximum raw support span | 0.06 s / 0.42 s |
| Supporting samples / support ratio | 2 / 0.65 |
| Quiet before / after | 0.12 s / 0.10 s |
| Rotation / acceleration veto | 0.35 rad/s / 0.12 g |
| Motion cooldown / refractory / maximum sample gap | 0.24 s / 0.28 s / 0.16 s |
| Double-clench enabled / onset interval | false / 0.5 s |
| Calibration separation / peak threshold ratio | 3 robust z / 0.45 |

| Head-pose parameter | Default |
| --- | --- |
| Smoothing / deadband / maximum gap | 0.12 s / 0.012 rad / 0.75 s |
| Drift enabled / time constant / capture window | true / 45 s / 0.10 rad |
| Minimum still duration | 1 s |
| Still gyro / acceleration limits | 0.12 rad/s / 0.06 g |

## Evidence and limitations

### Validation results (2026-11-13)

`swift test --disable-sandbox` passes all 11 tests: 9 motion tests and the 2
unchanged matrix/recording tests. `swift build --disable-sandbox` compiles the
macOS app adapter and core. The host uses Apple Swift 6.4.2 in the package's
Swift 5 language mode and a macOS 14 deployment target. Commands used writable
temporary module caches because the sandbox cannot write user Library caches:

```sh
CLANG_MODULE_CACHE_PATH=/tmp/gazefuse-modulecache \
SWIFT_MODULECACHE_PATH=/tmp/gazefuse-swiftcache \
swift test --disable-sandbox

CLANG_MODULE_CACHE_PATH=/tmp/gazefuse-modulecache \
SWIFT_MODULECACHE_PATH=/tmp/gazefuse-swiftcache \
swift build --disable-sandbox
```

SwiftPM emits warnings that user-level manifest caches are read-only; the build
and tests complete successfully using the workspace and temporary caches.

The jaw benchmark uses a fixed SplitMix64 seed, Gaussian acceleration noise
(0.00065 g per axis), slow baseline drift, 35–45 ms intervals, one missing frame
every 127 frames, and a single 0.08 g impulse below the gross-motion veto. It
contains 23 intended half-sine clenches (0.10–0.29 s, 0.014–0.040 g), some
overlapping nuisance activity. There are 3,718 delivered samples over 150 s;
matching uses one-to-one onset matches, not repeated credit for the same target.

| Mixed-stream metric | Result |
| --- | --- |
| Correct detections / intended clenches | 20 / 23 |
| Total clicks / false clicks | 21 / 1 |
| Precision / recall / F1 | 0.952 / 0.870 / 0.909 |
| Mean delivery latency from estimated onset | 0.440 s |
| Whole-run false clicks per minute | 0.40 |

| Nuisance segment | Duration | False clicks | False clicks/min |
| --- | ---: | ---: | ---: |
| Head turn | 2.5 s | 0 | 0 |
| Nod | 2.8 s | 0 | 0 |
| Walking-like periodic motion | 10.0 s | 0 | 0 |
| Talking-like rhythmic bursts | 11.0 s | 0 | 0 |
| Chewing-like rhythmic bursts | 5.5 s | 1 | 10.91 |

These short segment rates are descriptive counts, not statistical error-rate
estimates. Missed onsets are 83.40 s (walking overlap), 98.20 s (talking overlap),
and 123.00 s (after chewing). Adaptive noise estimation can remain conservative
after nuisance activity. The chewing false click is a material remaining
limitation; this detector is an experimental alternative input, not validated
for unattended real cursor clicking.

Additional deterministic results:

- A pair of 160 ms clenches separated by 440 ms yields exactly one right click
  and zero left clicks. A lone clench in double mode yields exactly one left
  click after the 500 ms deadline. Faster or overlapping pairs can merge into
  one burst and are not guaranteed to be recognized.
- A single 0.08 g impulse yields zero clicks. Interrupting a real clench candidate
  with a >160 ms gap cancels it and yields zero clicks after resumption.
- Guided calibration uses 3 s of rest and two 0.016 g training bursts. Measured
  rest median = 0.00075 g, MAD = 0.00014 g, representative clench peak = 0.00684 g,
  learned threshold = 0.00308 g (default absolute threshold = 0.006 g). One
  independent, weaker 0.010 g clench changes from 0/1 detected to 1/1 detected.
  This is evidence of the calibration path, not a population recall estimate;
  reducing the threshold can also increase nuisance false positives.
- The head-pose utility's mean radial yaw/pitch error after settling under seeded
  attitude noise and irregular intervals is 0.003918 rad (~0.225 degrees), with
  drift correction disabled for that measurement. Other tests verify recenter,
  deadband, near-zero drift correction, preserved large held offsets, wrapped
  angle continuity, stale timestamps, and zero deltas across gaps.

### Hardware boundary

Synthetic data validates the implemented decision rules and timing only. The
amplitudes and noise distributions are stylized, not measured AirPods recordings.
No AirPods, camera, macOS permission prompt, reconnect, physical cursor or Webgrid
trial was exercised. A click-like isolated jaw motion during talking or chewing
can be indistinguishable from an intentional clench with this sensor stream.
No software threshold can resolve identical observations. Head motion gating
also deliberately loses valid clenches performed during gross movement.

Apple provides fused, low-rate motion here, not raw high-rate IMU or bone/audio
vibration data. The approximately 25 Hz rate is the task's expected operating
rate, not an Apple-guaranteed value. Nyquist is only about 12.5 Hz at that rate;
sub-frame teeth taps may be missed or aliased. Gentle guided clench validation
on the actual AirPods fit is the next step before enabling real cursor clicks.

## References

- [Apple: CMHeadphoneMotionManager](https://developer.apple.com/documentation/coremotion/cmheadphonemotionmanager): macOS 14+, motion fields, connection delegates, and mandatory usage description.
- [Apple: CMLogItem.timestamp](https://developer.apple.com/documentation/coremotion/cmlogitem/timestamp): boot-relative sample timestamps.
- [Apple WWDC23: What's new in Core Motion](https://developer.apple.com/videos/play/wwdc2023/10179/): headphone motion on macOS and supported headphones.
- [Prakash et al., EarSense, MobiCom 2020](https://sinrg.csl.illinois.edu/papers/earsense_mobicom20.pdf): teeth vibrations sensed using earphone speaker/audio circuitry, not the AirPods motion API.
- [TeethTap, 2021](https://arxiv.org/abs/2102.12548): custom earpiece IMUs plus contact microphones for discrete teeth gestures. Its hardware and recognition results do not transfer to this detector.
- [Rupavatharam and Gruteser, Towards In-Ear Inertial Jaw Clenching Detection, EarComp 2019](https://www.esense.io/earcomp/EarComp-PreACM.pdf): eSense gyro-magnitude peaks/dips establish an inertial sensing precedent; this branch instead uses acceleration energy with gyro veto.
- [Earables for Detection of Bruxism: a Feasibility Study, 2021](https://arxiv.org/abs/2108.04144): eSense inertial data with nuisance activities; a different classification problem and protocol, not evidence of short AirPods click reliability.

## Iteration 2: early commit

`JawClenchDetector.Config.earlyCommitEnabled` (default on): commit after 60 ms of burst at ≥1.25× threshold if the prior
0.6 s was quiet and |ω| ≤ 0.08 rad/s before and during the burst. Mixed 150 s stream: F1 0.909 → 0.909, mean latency
0.440 → 0.120 s, p90 0.552 → 0.251 s; 8-seed F1 0.905 → 0.905. Without the head-still gate F1 dropped to 0.870.
Off automatically when double-clench is enabled. Sweep: `GF_SWEEP=1 swift test --filter testZZJawSweep`.
