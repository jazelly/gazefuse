# GazeFuse

**Multimodal, software-only cursor control for macOS: eye gaze (built-in camera) + finger pinches (camera) + AirPods head motion, fused with an EKF.**

Look roughly at a target → the cursor jumps there; nudge it with a small head movement (AirPods) or a pinch-and-move of your fingers; click with a thumb–index pinch (or a jaw clench). No extra hardware beyond a Mac with a camera and (optionally) AirPods 3 / Pro / Max.

> Not a medical device. Research / hobby project, Apache-2.0.

## Channels

| Channel | Sensor | Role |
|---|---|---|
| Eye gaze | FaceTime camera → Vision `VNDetectFaceLandmarksRequest` (pupils + head pose) → personal ridge-regression calibration | coarse pointing (saccades / "warp") |
| Hand | camera → Vision `VNDetectHumanHandPoseRequest` → pinch classifier (hysteresis, One Euro, velocity/curl vetoes) | thumb–index = left click, thumb–middle = right click, pinch-hold + move = fine adjust (take-off click) or drag; optional pinch + vertical move = scroll |
| Head | AirPods `CMHeadphoneMotionManager` (~25 Hz attitude/rotation rate) | fine correction (rate → cursor velocity), online gaze-bias correction |
| Jaw | AirPods IMU micro-vibration (high-pass, RMS energy, median+MAD adaptive threshold, head-motion gate) | alternative hands-free click |

## Architecture (short)

* **EKF fusion** — state `[x, y, vx, vy, bx, by]`. Gaze measures `x + b` (b = slowly varying calibration bias); head rate is a velocity measurement; head/hand corrections move `x` and `b` in opposite directions, so every correction also *teaches* the gaze bias. Saccades are detected with an NIS gate + windowed test and reset position; low-confidence samples (blinks) can't trigger resets.
* **Head→gaze leak compensator** — head rotation leaks into webcam gaze estimates; a fixation-differenced forgetting least-squares estimator removes it.
* **Innovation-adaptive gaze noise** (Mehra / Sage–Husa style).
* **ClickArbiter** — clicks are placed at the fused position from just *before* gesture onset (the pinch itself jiggles the face/hand), de-duplicates hand vs jaw.
* **CGEvent output** — move / down / up / drag / scroll, synthetic events tagged; a physical mouse/trackpad movement pauses GazeFuse for 1 s.
* **Recording / replay** — every raw and derived stream to JSONL; `gazefuse-bench replay` re-runs the pipeline offline.
* **Offline Webgrid-like simulator** — simulated operator (saccades, fixational jitter, webcam bias field + drift, intermittent minimum-jerk corrective submovements, visual-motor delay, settle-before-click) on a Neuralink-Webgrid task with the same BPS formula `max(log2(N²−1)·NTPM/60, 0)`; plus a random-search tuner validated on held-out seeds.

Details: [docs/architecture.md](docs/architecture.md) · module notes in [docs/agent-notes/](docs/agent-notes/).

## Results so far (simulated — not a live human)

30×30 grid, 60 s, 20 held-out seeds (full table and caveats: [docs/benchmarks.md](docs/benchmarks.md)):

| Preset | optimistic | typical | pessimistic |
|---|---:|---:|---:|
| gaze only (raw / EKF) | 0.00 | 0.00 | 0.00 |
| gaze + AirPods head (EKF) | 6.44 | **5.32** (acc 98.9%) | 3.70 |
| gaze + hand fine-adjust | 6.53 | 5.33 | 2.73 |
| gaze + head + hand | 6.48 | 5.15 | 2.77 |
| simulator ceiling (`mouseLike`) | 6.85 | 6.54 | 6.12 |

Component tests on synthetic data: gaze calibration held-out error 11.4 pt mean / 23.8 pt p95; pinch detector precision 1.000 / recall 0.997 over 300 taps, 0 false positives over ~22k non-pinch frames; jaw-clench F1 0.91 (precision 0.95, recall 0.87, ~440 ms latency). **Live Webgrid scores have not been measured yet.**

## Build & run

Requirements: macOS 14+, Xcode 16+/Swift 6 toolchain, Apple-silicon Mac with a camera. AirPods 3/Pro/Max optional.

```bash
swift build -c release --disable-sandbox     # (--disable-sandbox only needed inside sandboxed shells)
swift test --disable-sandbox                 # unit tests + simulator regression gate
scripts/bundle.sh                            # → dist/GazeFuse.app (ad-hoc signed); --install copies to ~/Applications
open dist/GazeFuse.app
```

Permissions (System Settings → Privacy & Security): **Camera**, **Accessibility** (to post mouse events), **Motion & Fitness** (AirPods). Ad-hoc re-signing after a rebuild can invalidate the Accessibility grant — toggle it off/on again.

Menu bar `◉ GF`: start/stop, calibrate, channel toggles, pinch mode, Webgrid helpers, recording. Global hotkeys: `⌃⌥G` toggle tracking, `⌃⌥C` calibrate, `⌃⌥R` record, `⌃⌥H` HUD.

Simulator / tuning:

```bash
B=.build/out/Products/Release/gazefuse-bench   # path of SwiftPM's new build system; older toolchains: .build/release
$B sim --preset gazeHeadEKF --noise typical --grid 30 --seconds 60 --seeds 20
$B all --seeds 20 --output docs/benchmarks.md
$B tune --preset gazeHeadEKF --noise typical          # random search, held-out validation
$B replay ~/Library/Application\ Support/GazeFuse/recordings/<file>.jsonl
```

Live Webgrid protocol (Chinese): [docs/webgrid-guide-zh.md](docs/webgrid-guide-zh.md).

## Known limitations / next steps

* All BPS numbers are from the simulator; the operator model is a model. Tuning on real recordings is the next step.
* AirPods yaw/pitch sign conventions and hand-delta axes need verification on hardware (one flag each in `Engine`).
* Gaze uses landmark-based pupil features; an appearance-based CoreML gaze model would cut the ~1–1.5° bias substantially.
* Saccade confirmation = 1 sample (best in simulation) trades robustness to outliers for latency.
* Jaw-clench latency ~440 ms; adaptive hand thresholds are experimental.

---

## 中文简介

GazeFuse 是一个纯软件、开源（Apache-2.0）的 macOS 多模态光标控制项目：**眼动（内置摄像头）负责粗定位，AirPods 头部运动 / 手指捏合负责微调，捏合（或咬合）负责点击**，三路信号用 EKF 融合。不是医疗器械。

- 眼动：Vision 人脸关键点 → 瞳孔 + 头姿特征 → 个人 ridge 回归校准；扫视（saccade）时光标直接跳到注视点。
- 手指：拇指+食指捏合 = 左键，拇指+中指 = 右键；默认 *take-off* 模式——捏住不放移动手微调光标，松开即点击；也可切换为按下/拖拽模式。
- AirPods：头部转动速度 → 光标速度做微调，同时在线修正眼动的系统偏差；咬合（jaw clench）可作为免手点击。
- 离线 Webgrid 仿真器（与 Neuralink Webgrid 相同的 BPS 公式）：典型噪声下 眼动+头部 ≈ **5.3 BPS**（仿真值，非真人实测），纯眼动因摄像头校准偏差大于 30×30 格子而为 0。
- 实测 Webgrid 步骤见 [docs/webgrid-guide-zh.md](docs/webgrid-guide-zh.md)。

## License

Apache License 2.0 — see [LICENSE](LICENSE).
