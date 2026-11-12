# GazeFuse

**Multimodal, software-only cursor control for macOS: eye gaze (built-in camera) + finger pinches (camera) + AirPods head motion, fused with an EKF.**

Look roughly at a target → the cursor jumps there; nudge it with a small head movement (AirPods) or a pinch-and-move of your fingers; click with a thumb–index pinch (or a jaw clench). No extra hardware beyond a Mac with a camera and (optionally) AirPods 3 / Pro / Max.

> Not a medical device. Research / hobby project, Apache-2.0.

## Channels

| Channel | Sensor | Role |
|---|---|---|
| Eye gaze | FaceTime camera → Vision `VNDetectFaceLandmarksRequest` (pupils + head pose), optional Timm–Barth pupil refinement, optional appearance-based CoreML gaze model (L2CS-style) → personal ridge-regression calibration | coarse pointing (saccades / "warp") |
| Hand | camera → Vision `VNDetectHumanHandPoseRequest` → pinch classifier (hysteresis, One Euro, velocity/curl vetoes) | thumb–index = left click, thumb–middle = right click, pinch-hold + move = fine adjust (take-off click) or drag; optional pinch + vertical move = scroll |
| Head | AirPods `CMHeadphoneMotionManager` (~25 Hz attitude/rotation rate) | fine correction (rate → cursor velocity), online gaze-bias correction |
| Jaw | AirPods IMU micro-vibration (high-pass, RMS energy, median+MAD adaptive threshold, head-motion gate, early commit during the burst) | alternative hands-free click |

## Architecture (short)

* **EKF fusion** — state `[x, y, vx, vy, bx, by]`. Gaze measures `x + b` (b = slowly varying calibration bias); head rate is a velocity measurement; head/hand corrections move `x` and `b` in opposite directions, so every correction also *teaches* the gaze bias. Saccades are detected with an NIS gate + windowed test and reset position; low-confidence samples (blinks) can't trigger resets.
* **Head→gaze leak compensator** — head rotation leaks into webcam gaze estimates; a fixation-differenced forgetting least-squares estimator removes it.
* **Innovation-adaptive gaze noise** (Mehra / Sage–Husa style).
* **Gaze-only mode** — with no head/hand correction channel the bias is unobservable: it becomes a Schmidt-Kalman *consider* state, its process noise is frozen and the position process noise is raised so deliberate "look-beside-the-target" compensation works.
* **ClickArbiter** — clicks are placed at the fused position from just *before* gesture onset (the pinch itself jiggles the face/hand), de-duplicates hand vs jaw.
* **CGEvent output** — move / down / up / drag / scroll, synthetic events tagged; a physical mouse/trackpad movement pauses GazeFuse for 1 s.
* **Recording / replay** — every raw and derived stream to JSONL; `gazefuse-bench replay` re-runs the pipeline offline.
* **Offline Webgrid-like simulator** — simulated operator (saccades, fixational jitter, webcam bias field + drift, intermittent minimum-jerk corrective submovements, visual-motor delay, settle-before-click) on a Neuralink-Webgrid task with the same BPS formula `max(log2(N²−1)·NTPM/60, 0)`; plus a random-search tuner validated on held-out seeds.

Details: [docs/architecture.md](docs/architecture.md) · module notes in [docs/agent-notes/](docs/agent-notes/).

## Results so far (simulated — not a live human)

30×30 grid, 60 s, 20 held-out seeds (full table and caveats: [docs/benchmarks.md](docs/benchmarks.md)):

| Preset | optimistic | typical | pessimistic |
|---|---:|---:|---:|
| gaze only (raw / EKF) | 0.00 / 1.10 | 0.00 / 0.15 | 0.00 / 0.00 |
| gaze + AirPods head (EKF) | 6.50 | **5.30** (acc 98.8%) | 3.67 |
| gaze + hand fine-adjust | 6.54 | 5.36 | 2.67 |
| gaze + head + hand | 6.47 | 5.08 | 2.76 |
| simulator ceiling (`mouseLike`) | 6.85 | 6.54 | 6.12 |

Gaze-only on coarser grids (typical): 6×6 raw 2.84 / EKF 3.27 BPS, 10×10 1.03 / 3.08, 15×15 0.04 / 2.53.

Component tests on synthetic data: gaze calibration held-out error 11.4 pt mean / 23.8 pt p95; pinch detector precision 1.000 / recall 0.997 over 300 taps, 0 false positives over ~22k non-pinch frames; adaptive pinch thresholds recall 0.947 for wide-contact users (fixed: 0.250); jaw-clench F1 0.91 (precision 0.95, recall 0.87) with mean latency 0.12 s (was 0.44 s); Timm–Barth pupil error 0.008–0.022 eye-widths vs 0.016–0.045 for dark centroid. **Live Webgrid scores have not been measured yet.**

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

### Optional appearance-based gaze model

```bash
python3 -m venv .venv && .venv/bin/pip install torch torchvision coremltools
mkdir -p ~/Library/Application\ Support/GazeFuse
.venv/bin/python scripts/convert_gaze_model.py --arch mobileone_s0 \
    --out ~/Library/Application\ Support/GazeFuse/GazeAppearance.mlpackage
```

The converter uses [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation) (code MIT). **Its pretrained
weights are trained on Gaze360, whose licence is non-commercial research only, so no weights are shipped in this
Apache-2.0 repo** — you convert them yourself and are responsible for complying with that licence. Without the model
the app runs landmark-only. Recalibrate after installing; calibration keeps the appearance features only if
leave-one-target-out CV improves. Menu toggles: *Pupil refinement (Timm–Barth)*, *Appearance gaze model*.

Offline self-test of the real camera pipeline on a video file:
`dist/GazeFuse.app/Contents/MacOS/GazeFuse --gaze-video clip.mov [--appearance-model GazeAppearance.mlpackage]`.

Live Webgrid protocol (Chinese): [docs/webgrid-guide-zh.md](docs/webgrid-guide-zh.md).

## Known limitations / next steps

* All BPS numbers are from the simulator; the operator model is a model. Tuning on real recordings is the next step.
* AirPods yaw/pitch sign conventions and hand-delta axes need verification on hardware (one flag each in `Engine`).
* Timm–Barth refinement and the appearance model were validated on synthetic data and one real low-resolution clip only; on small eyes (<32 px) Timm–Barth is gated off because it increased jitter there.
* Saccade confirmation = 1 sample (best in simulation) trades robustness to outliers for latency.
* Adaptive hand thresholds are off by default (≈1 false positive per ambiguous "wave" segment in simulation).

---

## 中文简介

GazeFuse 是一个纯软件、开源（Apache-2.0）的 macOS 多模态光标控制项目：**眼动（内置摄像头）负责粗定位，AirPods 头部运动 / 手指捏合负责微调，捏合（或咬合）负责点击**，三路信号用 EKF 融合。不是医疗器械。

- 眼动：Vision 人脸关键点 → 瞳孔 + 头姿特征 → 个人 ridge 回归校准；扫视（saccade）时光标直接跳到注视点。
- 手指：拇指+食指捏合 = 左键，拇指+中指 = 右键；默认 *take-off* 模式——捏住不放移动手微调光标，松开即点击；也可切换为按下/拖拽模式。
- AirPods：头部转动速度 → 光标速度做微调，同时在线修正眼动的系统偏差；咬合（jaw clench）可作为免手点击。
- 离线 Webgrid 仿真器（与 Neuralink Webgrid 相同的 BPS 公式）：典型噪声下 眼动+头部 ≈ **5.3 BPS**（仿真值，非真人实测），纯眼动在 30×30 格子上仍接近 0（摄像头校准偏差大于格子），但在 6×6 / 10×10 粗格子上可用（EKF 3.27 / 3.08 BPS）。
- 可选：Timm–Barth 瞳孔精定位 + 外观模型（CoreML，需用 `scripts/convert_gaze_model.py` 本地转换，权重因 Gaze360 非商用许可不随仓库分发）。
- 实测 Webgrid 步骤见 [docs/webgrid-guide-zh.md](docs/webgrid-guide-zh.md)。

## License

Apache License 2.0 — see [LICENSE](LICENSE).
