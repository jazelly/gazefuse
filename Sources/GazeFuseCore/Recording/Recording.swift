// Recording.swift — JSON-Lines log of raw sensor streams + derived events,
// so every algorithm can be replayed and regression-tested offline.
//
// File format: one JSON object per line: {"type": "...", "t": <sec>, "payload": {...}}.
// The first line is a `header`.
import Foundation

public struct RecordingHeader: Codable, Equatable {
    public var formatVersion: Int
    public var createdAt: String
    public var screen: ScreenRect
    public var appVersion: String
    public var notes: String
    public init(formatVersion: Int = 1, createdAt: String = ISO8601DateFormatter().string(from: Date()),
                screen: ScreenRect, appVersion: String = GazeFuseVersion.string, notes: String = "") {
        self.formatVersion = formatVersion; self.createdAt = createdAt; self.screen = screen
        self.appVersion = appVersion; self.notes = notes
    }
}

public enum GazeFuseVersion { public static let string = "0.1.0" }

public enum RecordedEvent: Equatable {
    case header(RecordingHeader)
    case gaze(GazeSample)
    case gazeEstimate(GazeEstimate)
    case hand(HandSample)
    case handLost(TimeSec)
    case head(HeadMotionSample)
    case input(InputEvent)
    case cursor(CursorState)
    case action(ClickAction)
    case calibrationTarget(t: TimeSec, target: Point2)
    case marker(t: TimeSec, text: String)

    public var t: TimeSec {
        switch self {
        case .header: return 0
        case .gaze(let s): return s.t
        case .gazeEstimate(let s): return s.t
        case .hand(let s): return s.t
        case .handLost(let t): return t
        case .head(let s): return s.t
        case .input(let s): return s.t
        case .cursor(let s): return s.t
        case .action(let s): return s.t
        case .calibrationTarget(let t, _): return t
        case .marker(let t, _): return t
        }
    }
}

extension RecordedEvent: Codable {
    private enum K: String, CodingKey { case type, t, payload, target, text }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: K.self)
        let type = try c.decode(String.self, forKey: .type)
        switch type {
        case "header": self = .header(try c.decode(RecordingHeader.self, forKey: .payload))
        case "gaze": self = .gaze(try c.decode(GazeSample.self, forKey: .payload))
        case "gazeEstimate": self = .gazeEstimate(try c.decode(GazeEstimate.self, forKey: .payload))
        case "hand": self = .hand(try c.decode(HandSample.self, forKey: .payload))
        case "handLost": self = .handLost(try c.decode(Double.self, forKey: .t))
        case "head": self = .head(try c.decode(HeadMotionSample.self, forKey: .payload))
        case "input": self = .input(try c.decode(InputEvent.self, forKey: .payload))
        case "cursor": self = .cursor(try c.decode(CursorState.self, forKey: .payload))
        case "action": self = .action(try c.decode(ClickAction.self, forKey: .payload))
        case "calibrationTarget":
            self = .calibrationTarget(t: try c.decode(Double.self, forKey: .t), target: try c.decode(Point2.self, forKey: .target))
        case "marker":
            self = .marker(t: try c.decode(Double.self, forKey: .t), text: try c.decode(String.self, forKey: .text))
        default:
            throw DecodingError.dataCorruptedError(forKey: .type, in: c, debugDescription: "unknown event type \(type)")
        }
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: K.self)
        try c.encode(t, forKey: .t)
        switch self {
        case .header(let p): try c.encode("header", forKey: .type); try c.encode(p, forKey: .payload)
        case .gaze(let p): try c.encode("gaze", forKey: .type); try c.encode(p, forKey: .payload)
        case .gazeEstimate(let p): try c.encode("gazeEstimate", forKey: .type); try c.encode(p, forKey: .payload)
        case .hand(let p): try c.encode("hand", forKey: .type); try c.encode(p, forKey: .payload)
        case .handLost: try c.encode("handLost", forKey: .type)
        case .head(let p): try c.encode("head", forKey: .type); try c.encode(p, forKey: .payload)
        case .input(let p): try c.encode("input", forKey: .type); try c.encode(p, forKey: .payload)
        case .cursor(let p): try c.encode("cursor", forKey: .type); try c.encode(p, forKey: .payload)
        case .action(let p): try c.encode("action", forKey: .type); try c.encode(p, forKey: .payload)
        case .calibrationTarget(_, let target): try c.encode("calibrationTarget", forKey: .type); try c.encode(target, forKey: .target)
        case .marker(_, let text): try c.encode("marker", forKey: .type); try c.encode(text, forKey: .text)
        }
    }
}

/// Thread-safe JSONL writer.
public final class SessionRecorder {
    public let url: URL
    private let handle: FileHandle
    private let queue = DispatchQueue(label: "gazefuse.recorder")
    private let encoder: JSONEncoder = { let e = JSONEncoder(); e.outputFormatting = [.sortedKeys]; return e }()
    public private(set) var count = 0

    public init(url: URL, header: RecordingHeader) throws {
        self.url = url
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try FileHandle(forWritingTo: url)
        write(.header(header))
    }
    public func write(_ e: RecordedEvent) {
        queue.async { [self] in
            guard var data = try? encoder.encode(e) else { return }
            data.append(0x0A)
            handle.write(data)
            count += 1
        }
    }
    public func close() { queue.sync { try? handle.synchronize(); try? handle.close() } }
}

public enum RecordingReader {
    /// Reads a JSONL recording. Unknown / corrupt lines are skipped (counted in `skipped`).
    public static func read(url: URL) throws -> (events: [RecordedEvent], skipped: Int) {
        let text = try String(contentsOf: url, encoding: .utf8)
        return parse(text)
    }
    public static func parse(_ text: String) -> (events: [RecordedEvent], skipped: Int) {
        let dec = JSONDecoder()
        var out: [RecordedEvent] = []; var skipped = 0
        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            if let e = try? dec.decode(RecordedEvent.self, from: Data(line.utf8)) { out.append(e) } else { skipped += 1 }
        }
        return (out, skipped)
    }
    public static func encode(_ events: [RecordedEvent]) -> String {
        let enc = JSONEncoder(); enc.outputFormatting = [.sortedKeys]
        return events.compactMap { try? enc.encode($0) }.compactMap { String(data: $0, encoding: .utf8) }.joined(separator: "\n") + "\n"
    }
}
