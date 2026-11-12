// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "GazeFuse",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "GazeFuseCore", targets: ["GazeFuseCore"]),
        .executable(name: "GazeFuse", targets: ["GazeFuse"]),
        .executable(name: "gazefuse-bench", targets: ["gazefuse-bench"]),
    ],
    targets: [
        // Pure, hardware-free algorithms: filters, EKF, calibration, classifiers,
        // fusion, recording format, Webgrid simulator. Fully unit-tested.
        .target(name: "GazeFuseCore"),
        // macOS menu-bar app: camera (Vision), AirPods (CoreMotion), CGEvent output.
        .executableTarget(
            name: "GazeFuse",
            dependencies: ["GazeFuseCore"],
            linkerSettings: [
                .linkedFramework("AVFoundation"),
                .linkedFramework("Vision"),
                .linkedFramework("CoreMotion"),
                .linkedFramework("AppKit"),
            ]
        ),
        // Offline benchmark / tuner / replay tool.
        .executableTarget(name: "gazefuse-bench", dependencies: ["GazeFuseCore"]),
        .testTarget(name: "GazeFuseCoreTests", dependencies: ["GazeFuseCore"]),
    ],
    swiftLanguageVersions: [.v5]
)
