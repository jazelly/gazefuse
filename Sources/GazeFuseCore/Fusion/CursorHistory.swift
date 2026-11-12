// CursorHistory.swift — fixed-capacity ring buffer of recent cursor states, used by
// the fusion layer for `position(at:)` (click arbitration at gesture onset) and for
// out-of-sequence gaze measurements (which are compared against the cursor state
// at their capture time). Timestamps must be non-decreasing; older appends are ignored.
import Foundation

public struct FusionCursorHistory {
    public struct Entry {
        public var t: TimeSec
        /// Unclamped filter position.
        public var state: Point2
        /// Clamped position actually shown to the user.
        public var display: Point2
        public init(t: TimeSec, state: Point2, display: Point2) { self.t = t; self.state = state; self.display = display }
    }

    private var buffer: [Entry]
    private var next = 0        // physical index of the next write
    public private(set) var count = 0
    public let capacity: Int

    public init(capacity: Int) {
        self.capacity = max(capacity, 2)
        buffer = Array(repeating: Entry(t: 0, state: .zero, display: .zero), count: self.capacity)
    }

    public var latest: Entry? { count == 0 ? nil : buffer[(next - 1 + capacity) % capacity] }
    public var oldest: Entry? { count == 0 ? nil : buffer[(next - count + capacity) % capacity] }

    private func entry(_ logical: Int) -> Entry { buffer[(next - count + logical + capacity) % capacity] }

    public mutating func removeAll() { next = 0; count = 0 }

    /// Appends if `e.t` is not older than the latest entry.
    public mutating func append(_ e: Entry) {
        if let l = latest, e.t < l.t { return }
        buffer[next] = e
        next = (next + 1) % capacity
        if count < capacity { count += 1 }
    }

    /// Linear interpolation between the two entries bracketing `t`. Returns the latest
    /// entry for `t` past the end and nil for `t` before the oldest entry (or when empty).
    public func lookup(at t: TimeSec) -> Entry? {
        guard count > 0 else { return nil }
        let last = entry(count - 1)
        if t >= last.t { return last }
        let first = entry(0)
        if t < first.t { return nil }
        // Binary search for the first entry with time > t.
        var lo = 0, hi = count - 1
        while lo < hi {
            let mid = (lo + hi) / 2
            if entry(mid).t <= t { lo = mid + 1 } else { hi = mid }
        }
        let b = entry(lo), a = entry(max(lo - 1, 0))
        let span = b.t - a.t
        guard span > 1e-9 else { return a }
        let w = (t - a.t) / span
        return Entry(t: t, state: a.state + (b.state - a.state) * w, display: a.display + (b.display - a.display) * w)
    }
}
