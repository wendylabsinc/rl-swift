import Testing
@testable import SwiftRL

@Suite struct ReplayBufferTests {
    @Test func rejectsInvalidCapacity() {
        #expect(throws: SwiftRLError.invalidCapacity(0)) {
            _ = try ReplayBuffer<Int>(capacity: 0)
        }
    }

    @Test func appendsAndWrapsAtCapacity() throws {
        var buffer = try ReplayBuffer<Int>(capacity: 2, seed: 1)

        buffer.append(1)
        buffer.append(2)
        buffer.append(3)

        #expect(buffer.count == 2)
        #expect(buffer.isFull)
        #expect(buffer.elements == [3, 2])
    }

    @Test func samplesDeterministically() throws {
        var buffer = try ReplayBuffer<Int>(capacity: 3, seed: 10)
        buffer.append(1)
        buffer.append(2)
        buffer.append(3)

        #expect(try buffer.sample(count: 3) == [1, 2, 3])
        #expect(try buffer.sample(count: 4) == [1, 2, 3])
        #expect(try buffer.sample(count: 0) == [])
        #expect(try buffer.sample(count: 2).count == 2)
        #expect(throws: SwiftRLError.invalidSampleCount(-1)) {
            _ = try buffer.sample(count: -1)
        }
    }
}
