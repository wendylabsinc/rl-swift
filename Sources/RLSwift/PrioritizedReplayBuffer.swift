import Foundation

/// A fixed-capacity replay buffer with priority-weighted sampling.
public struct PrioritizedReplayBuffer<Element: Sendable>: Sendable {
    private let capacity: Int
    private let exponent: Double
    private var storage: [(element: Element, priority: Double)]
    private var nextIndex: Int
    private var generator: SeededGenerator

    /// Creates a prioritized replay buffer.
    public init(capacity: Int, exponent: Double = 1, seed: UInt64 = 0) throws {
        guard capacity > 0 else {
            throw RLSwiftError.invalidCapacity(capacity)
        }
        guard exponent >= 0 else {
            throw RLSwiftError.invalidWeight(exponent)
        }
        self.capacity = capacity
        self.exponent = exponent
        storage = []
        nextIndex = 0
        generator = SeededGenerator(seed: seed)
    }

    /// The number of stored elements.
    public var count: Int {
        storage.count
    }

    /// Whether the buffer has filled its configured capacity.
    public var isFull: Bool {
        storage.count == capacity
    }

    /// The elements in storage order.
    public var elements: [Element] {
        storage.map(\.element)
    }

    /// The priorities in storage order.
    public var priorities: [Double] {
        storage.map(\.priority)
    }

    /// Adds an element with a strictly positive priority.
    public mutating func append(_ element: Element, priority: Double) throws {
        try validatePriority(priority)
        if storage.count < capacity {
            storage.append((element, priority))
        } else {
            storage[nextIndex] = (element, priority)
        }
        nextIndex = (nextIndex + 1) % capacity
    }

    /// Updates the priority for a stored element.
    public mutating func updatePriority(at index: Int, priority: Double) throws {
        try validatePriority(priority)
        guard storage.indices.contains(index) else {
            throw RLSwiftError.dimensionMismatch(expected: storage.count, actual: index)
        }
        storage[index].priority = priority
    }

    /// Samples up to `count` elements without replacement according to priority weights.
    public mutating func sample(count requestedCount: Int) throws -> [Element] {
        guard requestedCount >= 0 else {
            throw RLSwiftError.invalidSampleCount(requestedCount)
        }
        guard requestedCount < storage.count else {
            return elements
        }
        var candidates = storage
        var result: [Element] = []
        for _ in 0..<requestedCount {
            let selectedIndex = weightedIndex(in: candidates)
            result.append(candidates.remove(at: selectedIndex).element)
        }
        return result
    }

    private func validatePriority(_ priority: Double) throws {
        guard priority > 0 else {
            throw RLSwiftError.invalidPriority(priority)
        }
    }

    private mutating func weightedIndex(in candidates: [(element: Element, priority: Double)]) -> Int {
        let weights = candidates.map { pow($0.priority, exponent) }
        let total = weights.reduce(0, +)
        var threshold = (Double(generator.next()) / Double(UInt64.max)) * total
        for index in weights.indices.dropLast() {
            threshold -= weights[index]
            if threshold <= 0 {
                return index
            }
        }
        return weights.count - 1
    }
}
