import Testing
@testable import SwiftRL

@Suite struct ContinuousBoxSpaceTests {
    @Test func validatesBounds() {
        #expect(throws: SwiftRLError.dimensionMismatch(expected: 1, actual: 2)) {
            _ = try ContinuousBoxSpace(lowerBounds: [0], upperBounds: [1, 2])
        }
        #expect(throws: SwiftRLError.invalidBounds(index: 0, lower: 2, upper: 1)) {
            _ = try ContinuousBoxSpace(lowerBounds: [2], upperBounds: [1])
        }
    }

    @Test func checksContainsAndClamps() throws {
        let space = try ContinuousBoxSpace(lowerBounds: [-1, 0], upperBounds: [1, 10])

        #expect(space.dimension == 2)
        #expect(space.contains([0, 5]))
        #expect(!space.contains([0]))
        #expect(!space.contains([2, 5]))
        #expect(try space.clamp([-2, 12]) == [-1, 10])
        #expect(throws: SwiftRLError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try space.clamp([0])
        }
    }

    @Test func scalesAndNormalizesUnitCoordinates() throws {
        let space = try ContinuousBoxSpace(lowerBounds: [-2, 4], upperBounds: [2, 4])

        #expect(try space.scaleFromUnit([-1, 0]) == [-2, 4])
        #expect(try space.scaleFromUnit([1, 1]) == [2, 4])
        #expect(try space.normalizeToUnit([2, 4]) == [1, 0])
        #expect(throws: SwiftRLError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try space.scaleFromUnit([0])
        }
        #expect(throws: SwiftRLError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try space.normalizeToUnit([0])
        }
    }
}
