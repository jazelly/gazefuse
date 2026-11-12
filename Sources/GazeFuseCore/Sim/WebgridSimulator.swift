// WebgridSimulator.swift — drives the REAL cursor pipeline (EKFCursorFusion /
// DirectGazeCursor + ClickArbiter) with simulated GazeEstimate / HeadMotionSample /
// InputEvent / relative-delta streams produced from a SimHuman, scores the session
// with WebgridTask, and can emit a RecordedEvent stream for replay tests.
//
// Timeline: a fixed tick (default 240 Hz) advances the human; sensors sample at
// their own rates (camera 30 fps, AirPods 25 Hz) and deliver through a latency queue
// so the fusion sees capture timestamps that arrive late and out of order, exactly
// like the app. The display runs at 60 Hz: `cursor(at:)` is what the human sees.
import Foundation

/// Spatially varying gaze calibration bias: a constant plus Gaussian bumps,
/// normalised to a given RMS over the screen.
public struct SimBiasField {
    public var constant: Point2
    public var centers: [Point2]
    public var weights: [Point2]
    public var lengthScale: Double

    public init(rng: SimRNG, screen: ScreenRect, rmsPoints: Double, lengthScale: Double,
                bumps: Int = 6, constantFraction: Double = 0.7) {
        self.lengthScale = max(lengthScale, 1)
        constant = rng.gaussian2D(sd: constantFraction)
        var c: [Point2] = [], w: [Point2] = []
        for _ in 0..<bumps {
            c.append(Point2(rng.uniform(screen.origin.x - lengthScale, screen.origin.x + screen.width + lengthScale),
                            rng.uniform(screen.origin.y - lengthScale, screen.origin.y + screen.height + lengthScale)))
            w.append(rng.gaussian2D(sd: 1))
        }
        centers = c; weights = w
        // Normalise to the requested RMS over a grid of screen points.
        var sum = 0.0, n = 0
        for i in 0..<12 {
            for j in 0..<8 {
                let p = Point2(screen.origin.x + (Double(i) + 0.5) / 12 * screen.width,
                               screen.origin.y + (Double(j) + 0.5) / 8 * screen.height)
                let b = bias(at: p); sum += b.x * b.x + b.y * b.y; n += 1
            }
        }
        let rms = (sum / Double(n)).squareRoot()
        let s = rms > 1e-9 ? rmsPoints / rms : 0
        constant = constant * s
        weights = weights.map { $0 * s }
    }

    public func bias(at p: Point2) -> Point2 {
        var b = constant
        let l2 = 2 * lengthScale * lengthScale
        for (c, w) in zip(centers, weights) {
            let d = p - c
            b = b + w * exp(-(d.x * d.x + d.y * d.y) / l2)
        }
        return b
    }
}

/// Ideal pointer for the `mouseLike` baseline (implements CursorFusing so the arbiter
/// path is identical).
public final class SimIdealPointer: CursorFusing {
    public var screen: ScreenRect
    public var position: Point2
    private var history: FusionCursorHistory
    public init(screen: ScreenRect, position: Point2) {
        self.screen = screen; self.position = position; history = FusionCursorHistory(capacity: 512)
    }
    public func ingest(gaze: GazeEstimate) {}
    public func ingest(head: HeadMotionSample) {}
    public func ingest(relativeDelta: Point2, at t: TimeSec) { position = position + relativeDelta }
    public func cursor(at t: TimeSec) -> CursorState {
        history.append(FusionCursorHistory.Entry(t: t, state: position, display: position))
        return CursorState(t: t, position: position, velocity: .zero, mode: .fixation, confidence: 1)
    }
    public func position(at t: TimeSec) -> Point2? { history.lookup(at: t)?.display }
    public func reset() { history.removeAll() }
}

public struct SimResult: Codable, Equatable {
    public var preset: SimPreset
    public var noise: SimNoiseLevel
    public var seed: UInt64
    public var seconds: Double
    public var score: WebgridScore
    public var bps: Double
    public var ntpm: Double
    public var accuracy: Double
    public var correct: Int
    public var incorrect: Int
    public var outsideClicks: Int
    public var meanTimePerTarget: Double
    public var medianTimePerTarget: Double
    public var clickAttempts: Int
    public var patienceClicks: Int
    public var falsePositiveClicks: Int
    /// Mean distance (pt) between the click position and the target centre.
    public var meanClickError: Double
    /// RMS frame-to-frame cursor displacement (pt) while fixating with a quiet head.
    public var fixationJitter: Double
    public var fusionStats: EKFCursorFusion.Stats?
    public var targetTimes: [Double]
}

/// Min-heap of deliveries ordered by (time, sequence).
struct SimDeliveryQueue<T> {
    private var heap: [(t: TimeSec, seq: Int, item: T)] = []
    private var seq = 0
    var isEmpty: Bool { heap.isEmpty }
    var nextTime: TimeSec? { heap.first?.t }
    mutating func push(_ item: T, at t: TimeSec) {
        heap.append((t, seq, item)); seq += 1
        var i = heap.count - 1
        while i > 0 {
            let p = (i - 1) / 2
            if less(i, p) { heap.swapAt(i, p); i = p } else { break }
        }
    }
    mutating func pop() -> (t: TimeSec, item: T)? {
        guard let top = heap.first else { return nil }
        let last = heap.removeLast()
        if !heap.isEmpty {
            heap[0] = last
            var i = 0
            while true {
                let l = 2 * i + 1, r = l + 1
                var m = i
                if l < heap.count && less(l, m) { m = l }
                if r < heap.count && less(r, m) { m = r }
                if m == i { break }
                heap.swapAt(i, m); i = m
            }
        }
        return (top.t, top.item)
    }
    private func less(_ a: Int, _ b: Int) -> Bool {
        heap[a].t < heap[b].t || (heap[a].t == heap[b].t && heap[a].seq < heap[b].seq)
    }
}

public final class WebgridSimulator {
    enum Delivery {
        case gaze(GazeEstimate)
        case head(HeadMotionSample)
        case input(InputEvent)
        case hand(Point2, TimeSec)
    }

    public let config: SimConfig
    public let seed: UInt64
    public private(set) var recording: [RecordedEvent] = []

    public init(config: SimConfig, seed: UInt64) { self.config = config; self.seed = seed }

    /// Runs one session and returns its metrics. With `record`, `recording` holds the
    /// sensor/event stream in arrival order (plus cursor/action/marker events).
    public func run(record: Bool = false) -> SimResult {
        let cfg = config
        let root = SimRNG(seed: seed)
        let rngTask = root.fork(), rngHuman = root.fork(), rngGaze = root.fork(), rngHead = root.fork(), rngClick = root.fork()
        let screen = cfg.webgrid.screen
        let ppd = cfg.pointsPerDegree
        let task = WebgridTask(config: cfg.webgrid, rng: rngTask)

        // Cursor pipeline under test.
        let ekf: EKFCursorFusion? = cfg.cursorMode == .ekf ? EKFCursorFusion(config: cfg.tuning.fusion, screen: screen) : nil
        let direct: DirectGazeCursor? = cfg.cursorMode == .direct ? DirectGazeCursor(config: cfg.tuning.direct, screen: screen) : nil
        let ideal: SimIdealPointer? = cfg.cursorMode == .ideal ? SimIdealPointer(screen: screen, position: screen.center) : nil
        let cursor: CursorFusing
        if let e = ekf { cursor = e } else if let d = direct { cursor = d } else { cursor = ideal! }
        let arbiter = ClickArbiter(config: cfg.tuning.arbiter)
        let headGain = cfg.cursorMode == .direct ? cfg.tuning.direct.head : cfg.tuning.fusion.head

        let human = SimHuman(config: cfg, rng: rngHuman, headGain: headGain, startGaze: screen.center)
        let biasField = SimBiasField(rng: rngGaze, screen: screen, rmsPoints: cfg.gazeSensor.biasRMSDeg * ppd,
                                     lengthScale: cfg.gazeSensor.biasFieldLengthScale,
                                     bumps: cfg.gazeSensor.biasFieldBumps,
                                     constantFraction: cfg.gazeSensor.biasConstantFraction)
        var biasDrift = Point2.zero, corrNoise = Point2.zero, gyroBias = Point2.zero
        var lastGazePoint = screen.center
        var attitudeYawDrift = 0.0
        var queue = SimDeliveryQueue<Delivery>()

        // Ideal pointer motion plan.
        struct PointerMove { var from: Point2; var to: Point2; var t0: TimeSec; var duration: TimeSec }
        var pointerMove: PointerMove?

        let dt = 1 / cfg.tickHz
        let cameraPeriod = 1 / cfg.gazeSensor.frameRate
        let headPeriod = 1 / cfg.head.sampleRate
        let displayPeriod = 1 / cfg.displayHz
        var nextCamera = 0.0, nextHead = 0.0, nextDisplay = 0.0
        var nextFalsePositive = rngClick.exponential(rate: (cfg.clickSource == .pinch ? cfg.click.pinchFalsePositivesPerMin : cfg.click.jawFalsePositivesPerMin) / 60)
        var falsePositives = 0
        var clickErrors: [Double] = []
        var jitterSum = 0.0, jitterCount = 0
        var lastDisplayPos: Point2?
        var lastDisplayMode = CursorMode.idle

        if record {
            recording.append(.header(RecordingHeader(createdAt: "simulated", screen: screen,
                                                     notes: "WebgridSimulator preset=\(cfg.preset.rawValue) noise=\(cfg.noise.rawValue) seed=\(seed)")))
        }

        func scheduleIdealMove(at t: TimeSec) {
            guard let p = ideal else { return }
            let m = cfg.mouse
            let w = cfg.webgrid.cellSize
            let d = max((task.targetCenter - p.position).length, 1)
            let mt = m.fittsA + m.fittsB * log2(d / w + 1)
            pointerMove = PointerMove(from: p.position, to: task.targetCenter + rngHuman.gaussian2D(sd: w * m.endpointSigmaFraction),
                                      t0: t + m.reactionTime, duration: mt)
        }

        func presentTarget(at t: TimeSec) {
            human.newTarget(center: task.targetCenter, rect: task.targetRect, at: t)
            if record { recording.append(.marker(t: t, text: "target \(task.targetRow),\(task.targetCol)")) }
            scheduleIdealMove(at: t)
        }

        func scheduleClick(at t: TimeSec, falsePositive: Bool) {
            let c = cfg.click
            let conf = rngClick.uniform(c.confidenceMin, 1)
            switch cfg.clickSource {
            case .pinch:
                if !falsePositive && rngClick.bernoulli(c.pinchMissProbability) { return }
                let lat = rngClick.lognormal(mean: c.pinchDetectLatencyMean, cv: c.pinchDetectLatencyCV)
                queue.push(.input(InputEvent(t: t + lat, source: .hand, kind: .down, button: .left, confidence: conf)), at: t + lat)
                let up = t + c.pinchHoldDuration + rngClick.lognormal(mean: c.pinchDetectLatencyMean, cv: c.pinchDetectLatencyCV)
                queue.push(.input(InputEvent(t: up, source: .hand, kind: .up, button: .left, confidence: conf)), at: up)
            case .jaw:
                if !falsePositive && rngClick.bernoulli(c.jawMissProbability) { return }
                let lat = rngClick.lognormal(mean: c.jawDetectLatencyMean, cv: c.jawDetectLatencyCV)
                queue.push(.input(InputEvent(t: t + lat, source: .jaw, kind: .click, button: .left, confidence: conf)), at: t + lat)
            }
        }

        func sampleGaze(at t: TimeSec) {
            let g = cfg.gazeSensor
            if rngGaze.bernoulli(g.dropoutProbability) { return }
            let deliver = t + g.latency + rngGaze.uniform(-g.latencyJitter, g.latencyJitter)
            if human.isBlinking(at: t) {
                queue.push(.gaze(GazeEstimate(t: t, point: lastGazePoint, confidence: 0)), at: deliver)
                return
            }
            let eye = human.eyePosition
            var p = eye + biasField.bias(at: eye) + biasDrift + corrNoise
                + human.headAngle * (g.headLeakage * g.headLeakPointsPerRadian)
                + rngGaze.gaussian2D(sd: g.noiseSigmaDeg * ppd)
            var conf = rngGaze.uniform(g.confidenceMin, 1)
            if human.isBlinkEdge(at: t, window: g.blinkEdgeWindow) {
                p.y += g.blinkEdgeArtifactDeg * ppd
                conf = g.blinkEdgeConfidence
            }
            lastGazePoint = p
            queue.push(.gaze(GazeEstimate(t: t, point: p, confidence: conf)), at: deliver)
        }

        func sampleHead(at t: TimeSec) {
            let h = cfg.head
            let trueRate = human.headTrueRate(at: t)
            let measured = trueRate + gyroBias + rngHead.gaussian2D(sd: h.gyroNoiseSigma)
            // Convert screen-convention (yaw→x, pitch→y) rates into the configured sensor axes;
            // the third (roll) axis carries only noise.
            var comps = [rngHead.gaussian(mean: 0, sd: h.gyroNoiseSigma), rngHead.gaussian(mean: 0, sd: h.gyroNoiseSigma), rngHead.gaussian(mean: 0, sd: h.gyroNoiseSigma)]
            comps[min(max(headGain.yawAxis, 0), 2)] = measured.x / (headGain.yawSign == 0 ? 1 : headGain.yawSign)
            comps[min(max(headGain.pitchAxis, 0), 2)] = measured.y / (headGain.pitchSign == 0 ? 1 : headGain.pitchSign)
            let yaw = human.headAngle.x / (headGain.yawSign == 0 ? 1 : headGain.yawSign) + attitudeYawDrift
            let pitch = human.headAngle.y / (headGain.pitchSign == 0 ? 1 : headGain.pitchSign)
            // Quaternion from yaw (about y) then pitch (about x), roll 0.
            let cy = cos(yaw / 2), sy = sin(yaw / 2), cp = cos(pitch / 2), sp = sin(pitch / 2)
            let q = [sp * cy, cp * sy, -sp * sy, cp * cy]
            let s = HeadMotionSample(t: t, yaw: yaw, pitch: pitch, roll: 0, quaternion: q,
                                     rotationRate: Vector3(comps[0], comps[1], comps[2]),
                                     userAcceleration: Vector3(rngHead.gaussian(mean: 0, sd: h.userAccelerationNoiseSigma), rngHead.gaussian(mean: 0, sd: h.userAccelerationNoiseSigma), rngHead.gaussian(mean: 0, sd: h.userAccelerationNoiseSigma)),
                                     gravity: Vector3(0, -1, 0))
            queue.push(.head(s), at: t + h.latency)
        }

        func handleAction(_ a: ClickAction, at t: TimeSec) {
            if record { recording.append(.action(a)) }
            guard a.button == .left, a.kind == .down || a.kind == .click else { return }
            clickErrors.append((a.position - task.targetCenter).length)
            let outcome = task.click(at: a.position, t: t)
            human.noteOutcome(outcome, at: t)
            if outcome == .correct {
                presentTarget(at: t)
            } else if cfg.cursorMode == .ideal {
                // A mouse user sees the failed click and makes a corrective
                // submovement toward the still-lit target before retrying.
                scheduleIdealMove(at: t)
            }
        }

        func deliver(_ d: Delivery, at t: TimeSec) {
            switch d {
            case .gaze(let g):
                if record { recording.append(.gazeEstimate(g)) }
                cursor.ingest(gaze: g)
            case .head(let h):
                if record { recording.append(.head(h)) }
                cursor.ingest(head: h)
            case .hand(let delta, let tc):
                cursor.ingest(relativeDelta: delta, at: tc)
            case .input(let e):
                if record { recording.append(.input(e)) }
                for a in arbiter.ingest(e, cursor: cursor) { handleAction(a, at: t) }
            }
        }

        presentTarget(at: 0)
        let session = cfg.webgrid.sessionSeconds
        var t = 0.0
        var tick = 0
        while t < session {
            // Slow sensor-error processes.
            biasDrift = simOUStep(biasDrift, sigma: cfg.gazeSensor.biasDriftSigmaDeg * ppd, tau: cfg.gazeSensor.biasDriftTau, dt: dt, rng: rngGaze)
            corrNoise = simOUStep(corrNoise, sigma: cfg.gazeSensor.noiseCorrelatedSigmaDeg * ppd, tau: cfg.gazeSensor.noiseCorrelatedTau, dt: dt, rng: rngGaze)
            gyroBias = simOUStep(gyroBias, sigma: cfg.head.gyroBiasSigma, tau: cfg.head.gyroBiasTau, dt: dt, rng: rngHead)
            attitudeYawDrift += cfg.head.attitudeDriftRate * dt

            // Human.
            for action in human.step(t: t, dt: dt) {
                switch action {
                case .clickOnset:
                    scheduleClick(at: t, falsePositive: false)
                case .handDelta(let delta):
                    let noisy = delta + rngGaze.gaussian2D(sd: cfg.hand.trackingNoise)
                    queue.push(.hand(noisy, t), at: t + cfg.gazeSensor.latency)
                }
            }
            if t >= nextFalsePositive {
                falsePositives += 1
                scheduleClick(at: t, falsePositive: true)
                nextFalsePositive = t + rngClick.exponential(rate: (cfg.clickSource == .pinch ? cfg.click.pinchFalsePositivesPerMin : cfg.click.jawFalsePositivesPerMin) / 60)
            }
            // Ideal pointer.
            if let p = ideal, let m = pointerMove, t >= m.t0 {
                let tau = min((t - m.t0) / max(m.duration, 1e-3), 1)
                let sj = tau * tau * tau * (10 - 15 * tau + 6 * tau * tau)
                p.position = m.from + (m.to - m.from) * sj
                if tau >= 1 { pointerMove = nil }
            }
            // Sensors.
            if cfg.cursorMode != .ideal && t >= nextCamera - 1e-9 { sampleGaze(at: t); nextCamera += cameraPeriod }
            if cfg.headEnabled && t >= nextHead - 1e-9 { sampleHead(at: t); nextHead += headPeriod }
            // Deliveries.
            while let nt = queue.nextTime, nt <= t, let item = queue.pop() { deliver(item.item, at: t) }
            // Display.
            if t >= nextDisplay - 1e-9 {
                let cs = cursor.cursor(at: t)
                human.perceive(cursor: cs.position, at: t)
                if record { recording.append(.cursor(cs)) }
                if let lp = lastDisplayPos, cs.mode == .fixation, lastDisplayMode == .fixation,
                   human.hasLanded, !human.isSaccading, human.headRate.length < 0.01 {
                    let d = (cs.position - lp).length
                    jitterSum += d * d; jitterCount += 1
                }
                lastDisplayPos = cs.position; lastDisplayMode = cs.mode
                nextDisplay += displayPeriod
            }
            tick += 1
            t = Double(tick) * dt
        }

        let score = task.score(elapsed: session)
        let times = task.targetTimes
        let sorted = times.sorted()
        let median = sorted.isEmpty ? 0 : sorted[sorted.count / 2]
        return SimResult(
            preset: cfg.preset, noise: cfg.noise, seed: seed, seconds: session, score: score,
            bps: score.bitsPerSecond, ntpm: score.netTargetsPerMinute, accuracy: score.accuracy,
            correct: task.correct, incorrect: task.incorrect, outsideClicks: task.outsideClicks,
            meanTimePerTarget: times.isEmpty ? 0 : times.reduce(0, +) / Double(times.count),
            medianTimePerTarget: median,
            clickAttempts: human.clickAttempts, patienceClicks: human.patienceClicks, falsePositiveClicks: falsePositives,
            meanClickError: clickErrors.isEmpty ? 0 : clickErrors.reduce(0, +) / Double(clickErrors.count),
            fixationJitter: jitterCount > 0 ? (jitterSum / Double(jitterCount)).squareRoot() : 0,
            fusionStats: ekf?.stats, targetTimes: times)
    }
}

/// Aggregate of several sessions.
public struct SimSummary: Codable, Equatable {
    public var preset: SimPreset
    public var noise: SimNoiseLevel
    public var sessions: Int
    public var bpsMean: Double, bpsSD: Double
    public var ntpmMean: Double
    public var accuracyMean: Double
    public var timePerTargetMean: Double
    public var correctMean: Double, incorrectMean: Double
    public var fixationJitterMean: Double
    public var results: [SimResult]

    public init(results: [SimResult]) {
        precondition(!results.isEmpty)
        preset = results[0].preset; noise = results[0].noise; sessions = results.count
        func mean(_ xs: [Double]) -> Double { xs.reduce(0, +) / Double(xs.count) }
        let b = results.map { $0.bps }
        let bm = mean(b)
        bpsMean = bm
        bpsSD = b.count > 1 ? (b.map { ($0 - bm) * ($0 - bm) }.reduce(0, +) / Double(b.count - 1)).squareRoot() : 0
        ntpmMean = mean(results.map { $0.ntpm })
        accuracyMean = mean(results.map { $0.accuracy })
        let withTargets = results.filter { !$0.targetTimes.isEmpty }
        timePerTargetMean = withTargets.isEmpty ? 0 : mean(withTargets.map { $0.meanTimePerTarget })
        correctMean = mean(results.map { Double($0.correct) })
        incorrectMean = mean(results.map { Double($0.incorrect) })
        fixationJitterMean = mean(results.map { $0.fixationJitter })
        self.results = results
    }

    /// Runs `seeds` sessions in deterministic serial order and aggregates them.
    /// The benchmark CLI can choose its own process-level parallelism later; the
    /// core V1 keeps one straightforward execution path.
    public static func run(config: SimConfig, seeds: [UInt64]) -> SimSummary {
        let results = seeds.map { WebgridSimulator(config: config, seed: $0).run() }
        return SimSummary(results: results)
    }
}
