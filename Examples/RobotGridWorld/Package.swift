// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "RobotGridWorld",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(
            name: "robot-grid-world",
            targets: ["RobotGridWorldExample"]
        ),
    ],
    dependencies: [
        .package(name: "RLSwiftPackage", path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "RobotGridWorldExample",
            dependencies: [
                .product(name: "RLSwift", package: "RLSwiftPackage"),
            ],
            swiftSettings: [
                .enableUpcomingFeature("ExistentialAny"),
                .enableUpcomingFeature("InternalImportsByDefault"),
                .enableUpcomingFeature("MemberImportVisibility"),
            ]
        ),
    ],
    swiftLanguageModes: [.v6]
)
