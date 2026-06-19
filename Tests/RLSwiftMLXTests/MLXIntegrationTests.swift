import Testing
@testable import RLSwiftMLX

@Suite struct MLXIntegrationTests {
    @Test func reportsBackendSupport() {
        let support = MLXBackendSupport.current

#if SWIFTRL_ENABLE_MLX
        #expect(support.isMLXAvailable)
        #expect(support.explanation.contains("available"))
#else
        #expect(!support.isMLXAvailable)
        #expect(support.explanation.contains("MLXBackend"))
#endif
    }

    @Test func storesExplicitSupportInformation() {
        let support = MLXBackendSupport(isMLXAvailable: false, explanation: "disabled")

        #expect(!support.isMLXAvailable)
        #expect(support.explanation == "disabled")
    }

#if SWIFTRL_ENABLE_MLX
    @Test func exposesMLXTensorType() {
        #expect(String(describing: RLTensor.self).contains("MLXArray"))
    }
#endif
}
