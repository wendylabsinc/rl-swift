import Testing
@testable import RLSwift

@Suite struct ObservationNormalizerTests {
    @Test func validatesDimension() throws {
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 0)) {
            _ = try ObservationNormalizer(dimension: 0)
        }

        var normalizer = try ObservationNormalizer(dimension: 2)
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 1)) {
            try normalizer.update(with: [1])
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try normalizer.normalize([1])
        }
    }

    @Test func tracksMeanVarianceAndNormalizes() throws {
        var normalizer = try ObservationNormalizer(dimension: 2)

        #expect(normalizer.count == 0)
        #expect(normalizer.mean == [0, 0])
        #expect(normalizer.variance == [0, 0])

        try normalizer.update(with: [1, 3])
        try normalizer.update(with: [3, 7])

        #expect(normalizer.count == 2)
        #expect(normalizer.mean == [2, 5])
        #expect(normalizer.variance == [1, 4])
        #expect(try normalizer.normalize([3, 7], epsilon: 0) == [1, 1])

        normalizer.reset()
        #expect(normalizer.count == 0)
        #expect(normalizer.mean == [0, 0])
    }
}
