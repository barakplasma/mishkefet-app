// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "MishkefetCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "MishkefetCore", targets: ["MishkefetCore"]),
    ],
    dependencies: [
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", exact: "1.24.2"),
    ],
    targets: [
        // C wrapper over ORT's C++ API for the backbone (one float input, one float output).
        .target(
            name: "CMishkefetORT",
            dependencies: [
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
            ]
        ),
        .target(
            name: "MishkefetCore",
            dependencies: ["CMishkefetORT"]
        ),
        .testTarget(
            name: "MishkefetCoreTests",
            dependencies: ["MishkefetCore"]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
