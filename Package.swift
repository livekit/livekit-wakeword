// swift-tools-version:5.9
// The Swift client (client/swift). It sits at the repo root so Xcode can add the package by URL.
import PackageDescription

let package = Package(
    name: "LiveKitWakeWord",
    platforms: [.iOS(.v16), .macOS(.v14)],
    products: [
        .library(name: "LiveKitWakeWord", targets: ["LiveKitWakeWord"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/microsoft/onnxruntime-swift-package-manager",
            from: "1.20.0"
        ),
    ],
    targets: [
        .target(
            name: "LiveKitWakeWord",
            dependencies: [
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
            ],
            path: "client/swift/Sources/LiveKitWakeWord"
        ),
        .testTarget(
            name: "LiveKitWakeWordTests",
            dependencies: ["LiveKitWakeWord"],
            path: "client/swift/Tests/LiveKitWakeWordTests"
        ),
    ]
)
