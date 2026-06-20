// swift-tools-version: 6.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "RLSwift",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .tvOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "RLSwift",
            targets: ["RLSwift"]
        ),
        .library(
            name: "RLSwiftMLX",
            targets: ["RLSwiftMLX"]
        ),
        .library(
            name: "RLSwiftTensorRT",
            targets: ["RLSwiftTensorRT"]
        ),
        .executable(
            name: "rl-swift",
            targets: ["RLSwiftCLI"]
        ),
    ],
    traits: [
        .default(enabledTraits: ["MLXBackend"]),
        .trait(
            name: "MLXBackend",
            description: "Build the MLX-backed RLSwift integration target for Apple platforms."
        ),
        .trait(
            name: "TensorRTBackend",
            description: "Build the TensorRT-backed RLSwift integration target for NVIDIA Linux."
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
        .package(url: "https://github.com/wendylabsinc/tensorrt-swift", .upToNextMinor(from: "0.0.5")),
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "RLSwift",
            dependencies: [],
            swiftSettings: [
                .enableUpcomingFeature("ExistentialAny"),
                .enableUpcomingFeature("InternalImportsByDefault"),
                .enableUpcomingFeature("MemberImportVisibility"),
            ]
        ),
        .target(
            name: "RLSwiftMLX",
            dependencies: [
                "RLSwift",
                .product(name: "MLX", package: "mlx-swift", condition: .when(traits: ["MLXBackend"])),
                .product(name: "MLXNN", package: "mlx-swift", condition: .when(traits: ["MLXBackend"])),
                .product(name: "MLXOptimizers", package: "mlx-swift", condition: .when(traits: ["MLXBackend"])),
                .product(name: "MLXRandom", package: "mlx-swift", condition: .when(traits: ["MLXBackend"])),
            ],
            swiftSettings: [
                .define("SWIFTRL_ENABLE_MLX", .when(traits: ["MLXBackend"])),
                .enableUpcomingFeature("ExistentialAny"),
                .enableUpcomingFeature("InternalImportsByDefault"),
                .enableUpcomingFeature("MemberImportVisibility"),
            ]
        ),
        .target(
            name: "RLSwiftTensorRT",
            dependencies: [
                "RLSwift",
                .product(
                    name: "TensorRT",
                    package: "tensorrt-swift",
                    condition: .when(platforms: [.linux], traits: ["TensorRTBackend"])
                ),
            ],
            swiftSettings: [
                .define("SWIFTRL_ENABLE_TENSORRT", .when(platforms: [.linux], traits: ["TensorRTBackend"])),
                .enableUpcomingFeature("ExistentialAny"),
                .enableUpcomingFeature("InternalImportsByDefault"),
                .enableUpcomingFeature("MemberImportVisibility"),
            ]
        ),
        .executableTarget(
            name: "RLSwiftCLI",
            dependencies: ["RLSwift"],
            swiftSettings: [
                .enableUpcomingFeature("ExistentialAny"),
                .enableUpcomingFeature("InternalImportsByDefault"),
                .enableUpcomingFeature("MemberImportVisibility"),
            ]
        ),
        .testTarget(
            name: "RLSwiftTests",
            dependencies: ["RLSwift"]
        ),
        .testTarget(
            name: "RLSwiftMLXTests",
            dependencies: [
                "RLSwiftMLX",
                .product(name: "MLX", package: "mlx-swift", condition: .when(traits: ["MLXBackend"])),
            ],
            swiftSettings: [
                .define("SWIFTRL_ENABLE_MLX", .when(traits: ["MLXBackend"])),
                .enableUpcomingFeature("ExistentialAny"),
                .enableUpcomingFeature("InternalImportsByDefault"),
                .enableUpcomingFeature("MemberImportVisibility"),
            ]
        ),
        .testTarget(
            name: "RLSwiftTensorRTTests",
            dependencies: [
                "RLSwiftTensorRT",
                .product(
                    name: "TensorRT",
                    package: "tensorrt-swift",
                    condition: .when(platforms: [.linux], traits: ["TensorRTBackend"])
                ),
            ],
            swiftSettings: [
                .define("SWIFTRL_ENABLE_TENSORRT", .when(platforms: [.linux], traits: ["TensorRTBackend"])),
                .enableUpcomingFeature("ExistentialAny"),
                .enableUpcomingFeature("InternalImportsByDefault"),
                .enableUpcomingFeature("MemberImportVisibility"),
            ]
        ),
    ],
    swiftLanguageModes: [.v6]
)
