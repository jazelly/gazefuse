//
//  HeadphoneMotionSource.swift
//  GazeFuse
//
//  Thin app-side adapter for CMHeadphoneMotionManager.  The manager delivers
//  Core Motion values on one serial OperationQueue; callers receive the same
//  queue's ordering for status and sample callbacks.
//

import CoreMotion
import Foundation
import GazeFuseCore

/// Availability and connection state reported by ``HeadphoneMotionSource``.
public enum HeadphoneMotionStatus: Equatable {
    /// The host does not provide the headphone-motion service.
    case unavailable
    /// Motion access is restricted or denied in macOS privacy settings.
    case notAuthorized
    /// The service is available, but no supported headphones have connected yet.
    case waitingForHeadphones
    /// Supported headphones are connected and updates are being requested.
    case connected
    /// A previously connected headphone disconnected.
    case disconnected
    /// Core Motion reported an operational error.
    case error(String)
}

/// Streams CMHeadphoneMotionManager updates into the hardware-free core model.
///
/// Core Motion's ``CMLogItem.timestamp`` is a monotonic, boot-relative value
/// expressed in seconds.  It therefore already uses the same uptime time base
/// as ``CACurrentMediaTime``/``ProcessInfo.systemUptime`` and is copied directly
/// into ``HeadMotionSample.t``.
public final class HeadphoneMotionSource: NSObject, CMHeadphoneMotionManagerDelegate {
    /// Called on the source's dedicated serial operation queue. Set this before
    /// the first ``start()`` and keep it stable for this source's lifetime;
    /// callback properties are intentionally unsynchronized because the source
    /// itself is serialized, rather than adding a lock to this thin adapter.
    public var onSample: ((HeadMotionSample) -> Void)?

    /// Called on the source's dedicated serial operation queue when state changes.
    /// Set it before the first ``start()`` and keep it stable for this source's
    /// lifetime.
    public var onStatusChange: ((HeadphoneMotionStatus) -> Void)?

    /// Current Core Motion authorization state for headphone motion.
    public static var authorizationStatus: CMAuthorizationStatus {
        CMHeadphoneMotionManager.authorizationStatus()
    }

    /// Whether this Mac currently supports the headphone motion service.
    public var isDeviceMotionAvailable: Bool {
        manager.isDeviceMotionAvailable
    }

    private let manager: CMHeadphoneMotionManager
    private let callbackQueue: OperationQueue
    private var started = false
    private var lastStatus: HeadphoneMotionStatus?

    public override init() {
        manager = CMHeadphoneMotionManager()
        callbackQueue = OperationQueue()
        callbackQueue.name = "com.gazefuse.headphone-motion"
        callbackQueue.maxConcurrentOperationCount = 1
        callbackQueue.qualityOfService = .userInteractive
        super.init()
        manager.delegate = self
    }

    deinit {
        // This is intentionally synchronous and minimal.  The motion manager's
        // handler captures self weakly, so stopping here cannot create a retain
        // cycle while queued callbacks drain.
        manager.stopDeviceMotionUpdates()
        manager.stopConnectionStatusUpdates()
        manager.delegate = nil
        callbackQueue.cancelAllOperations()
    }

    /// Start connection and device-motion updates.  Work is serialized with
    /// incoming Core Motion callbacks on the source queue.
    public func start() {
        callbackQueue.addOperation { [weak self] in
            self?.startOnQueue()
        }
    }

    /// Stop both update streams.  A later ``start()`` can start the same source
    /// again without constructing another manager.
    public func stop() {
        callbackQueue.addOperation { [weak self] in
            self?.stopOnQueue()
        }
    }

    // MARK: CMHeadphoneMotionManagerDelegate

    public func headphoneMotionManagerDidConnect(_ manager: CMHeadphoneMotionManager) {
        // Apple documents these delegate callbacks as running on the operation
        // queue used for device-motion updates.  Keep the explicit queue hop as
        // well, so the guarantee also holds for connection-only notifications.
        callbackQueue.addOperation { [weak self] in
            guard let self, self.started else { return }
            self.publishStatus(.connected)
            self.restartMotionUpdatesOnQueue()
        }
    }

    public func headphoneMotionManagerDidDisconnect(_ manager: CMHeadphoneMotionManager) {
        callbackQueue.addOperation { [weak self] in
            guard let self, self.started else { return }
            self.manager.stopDeviceMotionUpdates()
            self.publishStatus(.disconnected)
        }
    }

    // MARK: Queue-confined implementation

    private func startOnQueue() {
        guard !started else { return }

        switch Self.authorizationStatus {
        case .denied, .restricted:
            publishStatus(.notAuthorized)
            return
        default:
            break
        }

        started = true
        manager.delegate = self
        manager.startConnectionStatusUpdates()

        // `isDeviceMotionAvailable` describes service support, not a guaranteed
        // live headphone connection.  It can be false until a compatible
        // headphone is present, so stay subscribed to connection updates and
        // allow a later delegate callback to restart motion delivery.  Do not
        // infer a connection from `isDeviceMotionActive`: it only says that a
        // stream was requested.  The first valid sample or a connect delegate
        // callback promotes this state to .connected.
        publishStatus(.waitingForHeadphones)
        restartMotionUpdatesOnQueue()
    }

    private func stopOnQueue() {
        guard started else {
            lastStatus = nil
            return
        }
        started = false
        manager.stopDeviceMotionUpdates()
        manager.stopConnectionStatusUpdates()
        manager.delegate = nil
        lastStatus = nil
    }

    private func restartMotionUpdatesOnQueue() {
        guard started else { return }
        manager.stopDeviceMotionUpdates()
        manager.startDeviceMotionUpdates(to: callbackQueue) { [weak self] motion, error in
            guard let self, self.started else { return }

            if let error {
                if self.isAuthorizationError(error) {
                    self.publishStatus(.notAuthorized)
                } else if self.isUnavailableError(error) {
                    self.publishStatus(.unavailable)
                } else {
                    let message = error.localizedDescription.isEmpty
                        ? String(describing: error)
                        : error.localizedDescription
                    self.publishStatus(.error(message))
                }
                return
            }

            guard let motion else { return }
            self.publishStatus(.connected)
            self.deliver(motion)
        }
    }

    private func deliver(_ motion: CMDeviceMotion) {
        let attitude = motion.attitude
        let quaternion = attitude.quaternion
        let sample = HeadMotionSample(
            t: motion.timestamp,
            yaw: attitude.yaw,
            pitch: attitude.pitch,
            roll: attitude.roll,
            quaternion: [quaternion.x, quaternion.y, quaternion.z, quaternion.w],
            rotationRate: Vector3(motion.rotationRate.x, motion.rotationRate.y, motion.rotationRate.z),
            userAcceleration: Vector3(
                motion.userAcceleration.x,
                motion.userAcceleration.y,
                motion.userAcceleration.z
            ),
            gravity: Vector3(motion.gravity.x, motion.gravity.y, motion.gravity.z)
        )
        onSample?(sample)
    }

    private func isAuthorizationError(_ error: Error) -> Bool {
        switch Self.authorizationStatus {
        case .denied, .restricted:
            return true
        default:
            break
        }

        let nsError = error as NSError
        guard nsError.domain == CMErrorDomain else { return false }
        return nsError.code == Int(CMErrorMotionActivityNotAuthorized.rawValue)
            || nsError.code == Int(CMErrorNotAuthorized.rawValue)
    }

    private func isUnavailableError(_ error: Error) -> Bool {
        let nsError = error as NSError
        guard nsError.domain == CMErrorDomain else { return false }
        return nsError.code == Int(CMErrorNotAvailable.rawValue)
            || nsError.code == Int(CMErrorMotionActivityNotAvailable.rawValue)
    }

    private func publishStatus(_ status: HeadphoneMotionStatus) {
        guard lastStatus != status else { return }
        lastStatus = status
        onStatusChange?(status)
    }
}
