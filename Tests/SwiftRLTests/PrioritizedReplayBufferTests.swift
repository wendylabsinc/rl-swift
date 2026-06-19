import Testing
@testable import SwiftRL

@Suite struct PrioritizedReplayBufferTests {
    @Test func validatesPrioritizedBufferInputs() throws {
        #expect(throws: SwiftRLError.invalidCapacity(0)) {
            _ = try PrioritizedReplayBuffer<Int>(capacity: 0)
        }
        #expect(throws: SwiftRLError.invalidWeight(-1)) {
            _ = try PrioritizedReplayBuffer<Int>(capacity: 1, exponent: -1)
        }

        var buffer = try PrioritizedReplayBuffer<Int>(capacity: 2)
        #expect(throws: SwiftRLError.invalidPriority(0)) {
            try buffer.append(1, priority: 0)
        }
        #expect(throws: SwiftRLError.invalidSampleCount(-1)) {
            _ = try buffer.sample(count: -1)
        }
        #expect(throws: SwiftRLError.invalidPriority(-1)) {
            try buffer.updatePriority(at: 0, priority: -1)
        }
        #expect(throws: SwiftRLError.dimensionMismatch(expected: 0, actual: 0)) {
            try buffer.updatePriority(at: 0, priority: 1)
        }
    }

    @Test func appendsWrapsAndUpdatesPriority() throws {
        var buffer = try PrioritizedReplayBuffer<String>(capacity: 2, seed: 0)

        try buffer.append("a", priority: 1)
        try buffer.append("b", priority: 2)
        try buffer.append("c", priority: 3)
        try buffer.updatePriority(at: 0, priority: 4)

        #expect(buffer.count == 2)
        #expect(buffer.isFull)
        #expect(buffer.elements == ["c", "b"])
        #expect(buffer.priorities == [4, 2])
        #expect(try buffer.sample(count: 2) == ["c", "b"])
        #expect(try buffer.sample(count: 3) == ["c", "b"])
    }

    @Test func samplesPriorityWeightedCandidatesDeterministically() throws {
        var finalPathBuffer = try PrioritizedReplayBuffer<String>(capacity: 3, exponent: 0, seed: 0)
        try finalPathBuffer.append("a", priority: 1)
        try finalPathBuffer.append("b", priority: 1)
        try finalPathBuffer.append("c", priority: 1)

        #expect(try finalPathBuffer.sample(count: 0) == [])
        #expect(try finalPathBuffer.sample(count: 1) == ["c"])

        var earlyPathBuffer = try PrioritizedReplayBuffer<String>(capacity: 3, exponent: 0, seed: 3)
        try earlyPathBuffer.append("a", priority: 1)
        try earlyPathBuffer.append("b", priority: 1)
        try earlyPathBuffer.append("c", priority: 1)

        #expect(try earlyPathBuffer.sample(count: 1) == ["a"])
        #expect(earlyPathBuffer.count == 3)
    }
}
