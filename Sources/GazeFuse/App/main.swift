import AppKit

// GazeFuse is a menu-bar (LSUIElement) app.
let cliArgs = CommandLine.arguments
if let i = cliArgs.firstIndex(of: "--gaze-video"), i + 1 < cliArgs.count {
    let model = cliArgs.firstIndex(of: "--appearance-model").flatMap { $0 + 1 < cliArgs.count ? cliArgs[$0 + 1] : nil }
    exit(VideoSelfTest.run(path: cliArgs[i + 1], modelPath: model))
}
let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
