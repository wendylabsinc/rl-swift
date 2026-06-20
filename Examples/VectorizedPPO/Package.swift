// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "VectorizedPPO",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(
            name: "vectorized-ppo",
            targets: ["VectorizedPPOExample"]
        ),
    ],
    dependencies: [
        .package(name: "RLSwiftPackage", path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "VectorizedPPOExample",
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
