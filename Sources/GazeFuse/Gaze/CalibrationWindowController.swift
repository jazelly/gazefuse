// CalibrationWindowController.swift
//
// A small AppKit front end for CalibrationPlan. Vision and the camera pipeline
// remain outside this class: callers feed GazeSample values from their camera
// queue.

import AppKit
import Foundation
import GazeFuseCore
import QuartzCore

public enum CalibrationWindowError: Error, LocalizedError {
    case cancelled
    case screenMismatch
    case noCalibrationSamples
    case fitFailed(String)
    case noValidationSamples

    public var errorDescription: String? {
        switch self {
        case .cancelled: return "Calibration cancelled"
        case .screenMismatch: return "Calibration plan coordinates do not match the selected screen"
        case .noCalibrationSamples: return "No usable gaze samples were received during calibration"
        case .fitFailed(let message): return "Calibration fit failed: \(message)"
        case .noValidationSamples: return "No usable gaze samples were received during validation"
        }
    }
}

private final class CalibrationWindow: NSWindow {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
}

private final class CalibrationCanvasView: NSView {
    var targetPoint = NSPoint.zero
    var showTarget = true
    var progress: CGFloat = 0
    var titleText = "Look at the target"
    var detailText = "Press Escape to cancel"
    var onEscape: (() -> Void)?

    override var acceptsFirstResponder: Bool { true }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        window?.makeFirstResponder(self)
    }

    override func keyDown(with event: NSEvent) {
        if event.keyCode == 53 { // Escape
            onEscape?()
        } else {
            super.keyDown(with: event)
        }
    }

    override func draw(_ dirtyRect: NSRect) {
        NSColor.black.setFill()
        dirtyRect.fill()

        if showTarget {
            let radius = max(10, 34 - 22 * progress)
            let circle = NSRect(x: targetPoint.x - radius,
                                y: targetPoint.y - radius,
                                width: radius * 2,
                                height: radius * 2)
            NSColor.white.setFill()
            NSBezierPath(ovalIn: circle).fill()
        }

        let titleAttributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 22, weight: .medium),
            .foregroundColor: NSColor.white
        ]
        let detailAttributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 14, weight: .regular),
            .foregroundColor: NSColor.lightGray
        ]
        let title = NSAttributedString(string: titleText, attributes: titleAttributes)
        let detail = NSAttributedString(string: detailText, attributes: detailAttributes)
        let titleSize = title.size()
        let detailSize = detail.size()
        title.draw(at: NSPoint(x: bounds.midX - titleSize.width / 2, y: bounds.maxY - 80))
        detail.draw(at: NSPoint(x: bounds.midX - detailSize.width / 2, y: 35))
    }
}

/// Borderless, full-screen calibration flow for a single NSScreen.
///
/// Samples may be fed from any queue. `feed` posts a small closure to the main
/// queue and returns immediately; all AppKit and mapper calls stay on main.
public final class CalibrationWindowController: NSObject {
    public let screen: NSScreen
    public let mapper: GazeMapper
    public let plan: CalibrationPlan

    /// Called exactly once, on the main thread, after success or cancellation.
    public var onFinish: ((Result<CalibrationReport, Error>) -> Void)?

    private enum Phase { case idle, calibration, validation, summary, finished }
    private var phase: Phase = .idle
    private var targetIndex = 0
    private var currentSamples: [GazeSample] = []
    private var calibrationSampleCount = 0
    private var validationErrors: [Double] = []
    private var validationMissingTargetCount = 0
    private var fitReport: CalibrationReport?
    private var phaseStartedAt: TimeInterval = 0
    private var targetStartedAt: TimeSec = 0
    private var summaryResult: Result<CalibrationReport, Error>?
    private var timer: Timer?
    private var window: CalibrationWindow?
    private var canvas: CalibrationCanvasView?
    private var didFinish = false

    public init(screen: NSScreen, mapper: GazeMapper, plan: CalibrationPlan) {
        self.screen = screen
        self.mapper = mapper
        self.plan = plan
        super.init()
    }

    deinit {
        timer?.invalidate()
    }

    /// Starts a fresh calibration session. A controller can be started once;
    /// create a new instance after cancellation or completion.
    public func start() {
        if !Thread.isMainThread {
            DispatchQueue.main.async { [weak self] in self?.start() }
            return
        }
        guard phase == .idle else { return }
        let primaryTop = NSScreen.screens.first?.frame.maxY ?? screen.frame.maxY
        let expectedCGOriginY = primaryTop - screen.frame.maxY
        guard abs(plan.screen.origin.x - screen.frame.minX) <= 1,
              abs(plan.screen.origin.y - expectedCGOriginY) <= 1,
              abs(plan.screen.width - screen.frame.width) <= 1,
              abs(plan.screen.height - screen.frame.height) <= 1 else {
            finish(.failure(CalibrationWindowError.screenMismatch))
            return
        }
        phase = .calibration
        targetIndex = 0
        currentSamples.removeAll(keepingCapacity: true)
        calibrationSampleCount = 0
        validationErrors.removeAll(keepingCapacity: true)
        validationMissingTargetCount = 0
        fitReport = nil
        summaryResult = nil
        // Starting a named calibration session intentionally discards the old
        // fit. It happens here, rather than in init, so construction has no side
        // effects.
        mapper.reset()
        createWindowIfNeeded()
        beginTarget()
    }

    /// Cancels the current flow and reports a typed error. Safe from any queue;
    /// repeated calls after completion are ignored.
    public func cancel() {
        if !Thread.isMainThread {
            DispatchQueue.main.async { [weak self] in self?.cancel() }
            return
        }
        finish(.failure(CalibrationWindowError.cancelled))
    }

    /// Supplies a camera sample. Only samples received while a target is active
    /// are retained; the list is cleared at every target transition.
    public func feed(_ sample: GazeSample) {
        DispatchQueue.main.async { [weak self] in
            guard let self, self.phase == .calibration || self.phase == .validation else { return }
            let duration = self.phase == .calibration ? self.plan.config.targetDuration : self.plan.config.validationDuration
            // Camera timestamps are monotonic. Strict target bounds prevent
            // frames queued during a main-thread transition from contaminating
            // the next target's aggregate.
            guard sample.t >= self.targetStartedAt,
                  sample.t <= self.targetStartedAt + duration else { return }
            self.currentSamples.append(sample)
        }
    }

    private func createWindowIfNeeded() {
        guard window == nil else { return }
        let contentFrame = NSRect(origin: .zero, size: screen.frame.size)
        let created = CalibrationWindow(contentRect: screen.frame,
                                        styleMask: [.borderless],
                                        backing: .buffered,
                                        defer: false)
        created.isOpaque = true
        created.isReleasedWhenClosed = false
        created.backgroundColor = .black
        created.level = .screenSaver
        created.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]

        let view = CalibrationCanvasView(frame: contentFrame)
        view.autoresizingMask = [.width, .height]
        view.onEscape = { [weak self] in self?.cancel() }
        created.contentView = view
        NSApplication.shared.activate()
        created.makeKeyAndOrderFront(nil)
        created.makeFirstResponder(view)
        window = created
        canvas = view
    }

    private func beginTarget() {
        guard !plan.targets.isEmpty else {
            finish(.failure(CalibrationWindowError.noCalibrationSamples))
            return
        }
        currentSamples.removeAll(keepingCapacity: true)
        targetStartedAt = CACurrentMediaTime()
        phaseStartedAt = targetStartedAt
        let target = plan.targets[targetIndex]
        canvas?.targetPoint = windowPoint(for: target)
        canvas?.showTarget = true
        canvas?.progress = 0
        canvas?.titleText = "Look at the target  \(targetIndex + 1)/\(plan.targets.count)"
        canvas?.detailText = "Keep looking at the target · Press Escape to cancel"
        canvas?.needsDisplay = true
        scheduleTimer()
    }

    private func scheduleTimer() {
        timer?.invalidate()
        let newTimer = Timer(timeInterval: 1.0 / 60.0,
                             repeats: true) { [weak self] _ in self?.tick() }
        timer = newTimer
        RunLoop.main.add(newTimer, forMode: .common)
    }

    private func tick() {
        guard phase == .calibration || phase == .validation || phase == .summary else { return }
        let duration: TimeSec
        switch phase {
        case .calibration: duration = plan.config.targetDuration
        case .validation: duration = plan.config.validationDuration
        case .summary: duration = plan.config.summaryDuration
        case .idle, .finished: return
        }
        let elapsed = CACurrentMediaTime() - phaseStartedAt
        canvas?.progress = CGFloat(min(max(elapsed / duration, 0), 1))
        canvas?.needsDisplay = true
        if elapsed >= duration {
            if phase == .calibration {
                endCalibrationTarget()
            } else if phase == .validation {
                endValidationTarget()
            } else if phase == .summary, let summaryResult {
                finish(summaryResult)
            }
        }
    }

    private func endCalibrationTarget() {
        timer?.invalidate()
        let target = plan.targets[targetIndex]
        if !currentSamples.isEmpty {
            mapper.addCalibrationSamples(currentSamples, target: target)
            calibrationSampleCount += currentSamples.count
        }
        targetIndex += 1
        if targetIndex < plan.targets.count {
            beginTarget()
        } else {
            fitAndBeginValidation()
        }
    }

    private func fitAndBeginValidation() {
        guard calibrationSampleCount > 0 else {
            finish(.failure(CalibrationWindowError.noCalibrationSamples))
            return
        }
        canvas?.showTarget = false
        canvas?.titleText = "Fitting calibration"
        canvas?.detailText = "Please keep your position while the model is fitted"
        canvas?.needsDisplay = true
        canvas?.displayIfNeeded()
        do {
            fitReport = try mapper.fit()
        } catch {
            finish(.failure(CalibrationWindowError.fitFailed(error.localizedDescription)))
            return
        }

        guard !plan.validationTargets.isEmpty else {
            finish(.failure(CalibrationWindowError.noValidationSamples))
            return
        }
        phase = .validation
        targetIndex = 0
        currentSamples.removeAll(keepingCapacity: true)
        beginValidationTarget()
    }

    private func beginValidationTarget() {
        currentSamples.removeAll(keepingCapacity: true)
        targetStartedAt = CACurrentMediaTime()
        phaseStartedAt = targetStartedAt
        let target = plan.validationTargets[targetIndex]
        canvas?.targetPoint = windowPoint(for: target)
        canvas?.showTarget = true
        canvas?.progress = 0
        canvas?.titleText = "Checking accuracy  \(targetIndex + 1)/\(plan.validationTargets.count)"
        canvas?.detailText = "Keep looking at the target"
        canvas?.needsDisplay = true
        scheduleTimer()
    }

    private func endValidationTarget() {
        timer?.invalidate()
        let target = plan.validationTargets[targetIndex]
        let usable = currentSamples.filter { sample in
            sample.t >= targetStartedAt + plan.config.settleDuration
        }
        let perTarget = usable.compactMap { mapper.estimate($0)?.point.distance(to: target) }
        // Median aggregation is deliberately per target: a single blink or
        // transient outlier cannot dominate the final report.
        if let median = median(perTarget) {
            validationErrors.append(median)
        } else {
            validationMissingTargetCount += 1
        }
        targetIndex += 1
        if targetIndex < plan.validationTargets.count {
            beginValidationTarget()
        } else {
            finishValidation()
        }
    }

    private func finishValidation() {
        guard validationMissingTargetCount == 0, validationErrors.count == plan.validationTargets.count else {
            finish(.failure(CalibrationWindowError.noValidationSamples))
            return
        }
        let sorted = validationErrors.sorted()
        let mean = sorted.reduce(0, +) / Double(sorted.count)
        let p95Index = min(sorted.count - 1, max(0, Int(ceil(Double(sorted.count) * 0.95)) - 1))
        let p95 = sorted[p95Index]
        let angular = degrees(forScreenPoints: mean)
        let coverage = "validation coverage \(validationErrors.count)/\(plan.validationTargets.count) targets"
        let cvText: String
        if let cv = fitReport {
            cvText = String(format: "CV mean %.1f pt, p95 %.1f pt; validation median-per-target mean %.1f pt, p95 %.1f pt (≈ %.2f° mean; %@; assumptions: %.0f mm display width, %.0f mm viewing distance)",
                            cv.meanErrorPoints, cv.p95ErrorPoints, mean, p95, angular, coverage,
                            plan.config.assumedDisplayWidthMetres * 1000,
                            plan.config.assumedViewingDistanceMetres * 1000)
        } else {
            cvText = String(format: "Validation median-per-target mean %.1f pt, p95 %.1f pt (≈ %.2f° mean; %@; assumptions: %.0f mm display width, %.0f mm viewing distance)",
                            mean, p95, angular, coverage,
                            plan.config.assumedDisplayWidthMetres * 1000,
                            plan.config.assumedViewingDistanceMetres * 1000)
        }
        let report = CalibrationReport(pointCount: fitReport?.pointCount ?? plan.targets.count,
                                       sampleCount: fitReport?.sampleCount ?? calibrationSampleCount,
                                       meanErrorPoints: mean,
                                       p95ErrorPoints: p95,
                                       notes: cvText)
        // Keep the measured result visible briefly so the calibration has a
        // concrete user-boundary outcome before the window is dismissed.
        summaryResult = .success(report)
        phase = .summary
        phaseStartedAt = CACurrentMediaTime()
        canvas?.showTarget = false
        canvas?.progress = 1
        canvas?.titleText = String(format: "Calibration complete · %.1f pt mean / %.1f pt p95", mean, p95)
        canvas?.detailText = String(format: "≈ %.2f° mean · assuming %.0f cm distance and %.1f cm display width · closing in %.1f s",
                                    angular, plan.config.assumedViewingDistanceMetres * 100,
                                    plan.config.assumedDisplayWidthMetres * 100, plan.config.summaryDuration)
        canvas?.needsDisplay = true
        scheduleTimer()
    }

    private func median(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[mid - 1] + sorted[mid]) / 2
        }
        return sorted[mid]
    }

    private func finish(_ result: Result<CalibrationReport, Error>) {
        guard !didFinish else { return }
        didFinish = true
        phase = .finished
        timer?.invalidate()
        timer = nil
        window?.orderOut(nil)
        window?.close()
        window = nil
        canvas = nil
        onFinish?(result)
    }

    private func degrees(forScreenPoints points: Double) -> Double {
        // Explicit, labelled assumptions keep this estimate from being mistaken
        // for a physical eye-tracker angular measurement.
        let metres = points * plan.config.assumedDisplayWidthMetres / max(plan.screen.width, 1)
        return atan2(metres, plan.config.assumedViewingDistanceMetres) * 180 / Double.pi
    }

    private func windowPoint(for point: Point2) -> NSPoint {
        // AppKit uses global bottom-left coordinates. GazeFuse's global screen
        // coordinates use the main display's top-left. Convert through the main
        // NSScreen frame, then into the selected window's local coordinates.
        let mainFrame = NSScreen.screens.first?.frame ?? screen.frame
        let appKitGlobal = NSPoint(x: point.x, y: mainFrame.maxY - point.y)
        return NSPoint(x: appKitGlobal.x - screen.frame.minX,
                       y: appKitGlobal.y - screen.frame.minY)
    }
}


