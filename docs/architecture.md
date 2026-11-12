# GazeFuse architecture

```
 ┌───────────── built-in camera (AVCaptureSession, 720p, 420f) ─────────────┐
 │  one VNImageRequestHandler per frame                                       │
 │   ├─ VNDetectFaceLandmarksRequest ──► FaceGazeExtractor ──► GazeSample      │
 │   └─ VNDetectHumanHandPoseRequest ──► HandPoseExtractor ──► HandSample      │
 └────────────────────────────────────────────────────────────────────────────┘
 ┌── AirPods (CMHeadphoneMotionManager, ~25 Hz) ──► HeadphoneMotionSource ──► HeadMotionSample
 │
 │              GazeFuseCore (pure Swift, hardware-free, unit-tested)
 │   GazeSample ──► RidgeGazeCalibrator (GazeMapper) ──► GazeEstimate ─┐
 │   HeadMotionSample ────────────────────────────────────────────────►├─► EKFCursorFusion (CursorFusing) ─► CursorState
 │   HandSample ─► PinchGestureClassifier ─► pinchMotionDelta ────────►┘           │ position(at:) history
 │                        │                                                         ▼
 │                        └─► InputEvent(.hand down/up) ─┐                ClickArbiter (ClickArbitrating)
 │   HeadMotionSample ─► JawClenchDetector ─► InputEvent(.jaw click) ─┘──────────►   │
 │                                                                                   ▼
 │                                                                           ClickAction
 └──────────────────────────────────────────────────────────────────────────────────────┘
                     App: CursorOutput (CGEvent move / down / up / drag / scroll)
                     SessionRecorder: every raw + derived stream → JSONL (replayable)
```

## Targets

| Target | Contents |
| --- | --- |
| `GazeFuseCore` | `Interfaces.swift` (shared contract), `Math/Matrix`, `Filters/` (One Euro, generic EKF, CV-KF), `Gaze/` (ridge calibration, calibration plans), `Hand/` (pinch classifier, hand features), `Motion/` (jaw-clench detector, head-pose tracker), `Fusion/` (EKF cursor fusion, baseline, click arbiter, tuning), `Recording/` (JSONL format), `Sim/` (Webgrid simulator, simulated user, tuner, replay). No AppKit/AVFoundation/Vision/CoreMotion. |
| `GazeFuse` (app) | Menu-bar app: `App/` (AppDelegate, Engine, CameraPipeline, CursorOutput, OverlayHUD), `Gaze/` (Vision face extractor, calibration window), `Hand/` (Vision hand-pose extractor), `Motion/` (CMHeadphoneMotionManager source). |
| `gazefuse-bench` | CLI: Webgrid simulation, parameter tuning, replay of recordings. |
| `GazeFuseCoreTests` | XCTest suites for every core module. |

## Threading

- Camera frames + Vision run on `gazefuse.camera` (serial). Results are posted to the engine queue.
- AirPods samples arrive on the motion source's serial OperationQueue → engine queue.
- `Engine.queue` (`gazefuse.engine`, serial) owns every algorithm object (mapper, classifiers, fusion, arbiter) and a 120 Hz timer that advances the fusion and moves the cursor.
- Calibration UI runs on main; while it runs, the engine routes raw gaze samples to it instead of the mapper.

## Coordinate systems

- Screen: CG global display points, top-left origin, +y down (CGEvent space). The calibration window converts from AppKit (bottom-left).
- Image: Vision normalized, bottom-left origin, un-mirrored camera image. Hand pinch motion is flipped on both axes in `Engine` before it is applied as a cursor delta.
- Time: host uptime seconds (`CACurrentMediaTime`); camera PTS is converted from the capture session clock; CoreMotion timestamps are already uptime-based.

## Interaction model (why these three channels)

- **Gaze = coarse, fast pointing** (MAGIC pointing, Zhai et al. 1999; Apple Vision Pro "look"). Webcam gaze is accurate to a few degrees at best, i.e. tens of points on a laptop, so it cannot hit a 30×30 Webgrid cell (~30 pt) alone.
- **Head motion = fine, precise relative correction.** AirPods attitude is low-noise and relative; the EKF treats it as a velocity input and absorbs the gaze/cursor disagreement into a slowly varying gaze-bias state, so the eyes can stay on the target while the head nudges the cursor.
- **Pinch = click** (Vision Pro "look and tap"). The arbiter places the click at the fused cursor position from just *before* gesture onset, cancelling the gaze/cursor jitter that the gesture itself causes. Jaw clench is an alternative hands-free click.
- Touching the physical mouse/trackpad pauses control (synthetic events are tagged via `eventSourceUserData`).

## Recording / replay

`SessionRecorder` writes JSON Lines (`Recording.swift`): header, raw `gaze`/`hand`/`head` samples, `gazeEstimate`, `input`, `cursor` (30 Hz), `action`, calibration targets and free-text markers (e.g. `webgrid-start`). `gazefuse-bench replay <file>` re-runs the algorithms offline, so a live session can be re-tuned without the user.
