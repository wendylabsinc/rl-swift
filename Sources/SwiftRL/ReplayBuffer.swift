/// A fixed-capacity ring buffer for replay memory.
public struct ReplayBuffer<Element: Sendable>: Sendable {
    private let capacity: Int
    private var storage: [Element]
    private var nextIndex: Int
    private var generator: SeededGenerator

    /// Creates a replay buffer with deterministic sampling.
    public init(capacity: Int, seed: UInt64 = 0) throws {
        guard capacity > 0 else {
            throw SwiftRLError.invalidCapacity(capacity)
        }
        self.capacity = capacity
        storage = []
        nextIndex = 0
        generator = SeededGenerator(seed: seed)
    }

    /// The number of elements currently stored.
    public var count: Int {
        storage.count
    }

    /// Whether the buffer has filled its configured capacity.
    public var isFull: Bool {
        storage.count == capacity
    }

    /// Returns elements in storage order, which becomes ring order after wraparound.
    public var elements: [Element] {
        storage
    }

    /// Adds an element, overwriting the oldest ring slot after the buffer is full.
    public mutating func append(_ element: Element) {
        if storage.count < capacity {
            storage.append(element)
        } else {
            storage[nextIndex] = element
        }
        nextIndex = (nextIndex + 1) % capacity
    }

    /// Samples up to `count` elements without replacement using the buffer's deterministic generator.
    public mutating func sample(count requestedCount: Int) throws -> [Element] {
        guard requestedCount >= 0 else {
            throw SwiftRLError.invalidSampleCount(requestedCount)
        }
        guard requestedCount < storage.count else {
            return storage
        }

        var shuffled = storage
        if shuffled.count > 1 {
            for index in stride(from: shuffled.count - 1, through: 1, by: -1) {
                let swapIndex = Int(generator.next() % UInt64(index + 1))
                shuffled.swapAt(index, swapIndex)
            }
        }
        return Array(shuffled.prefix(requestedCount))
    }
}
