import Testing
@testable import RLSwift

@Suite struct ActionSpaceTests {
    @Test func rejectsEmptyActions() throws {
        #expect(throws: RLSwiftError.emptyActionSpace) {
            _ = try DiscreteActionSpace<Int>([])
        }
    }

    @Test func checksMembershipAndDeterministicRandomAction() throws {
        let space = try DiscreteActionSpace(["left", "right"])
        var generator = SeededGenerator(seed: 1)

        #expect(space.contains("left"))
        #expect(!space.contains("up"))
        #expect(["left", "right"].contains(space.randomAction(using: &generator)))
    }

    @Test func seededGeneratorIsReproducible() {
        var first = SeededGenerator(seed: 42)
        var second = SeededGenerator(seed: 42)

        #expect(first.next() == second.next())
        #expect(first.next() == second.next())
    }
}
