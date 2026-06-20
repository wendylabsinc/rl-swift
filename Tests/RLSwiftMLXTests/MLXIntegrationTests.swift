import Testing
@testable import RLSwiftMLX
#if SWIFTRL_ENABLE_MLX
import MLX
import RLSwift
#endif

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

    @Test func buildsObservationTensorBatch() throws {
        let batch = try MLXObservationBatch(rows: [[1, 2, 3], [4, 5, 6]])

        #expect(batch.batchSize == 2)
        #expect(batch.featureCount == 3)
        #expect(batch.shape == [2, 3])
        #expect(batch.rowMajorValues == [1, 2, 3, 4, 5, 6])

        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try MLXObservationBatch(rows: [])
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 0)) {
            _ = try MLXObservationBatch(rows: [[]])
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try MLXObservationBatch(rows: [[1, 2], [3]])
        }
    }
#endif
}
