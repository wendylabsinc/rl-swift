/// A deterministic random number generator suitable for reproducible reinforcement learning tests.
public struct SeededGenerator: RandomNumberGenerator, Sendable {
    private var state: UInt64

    /// Creates a generator from a fixed seed.
    public init(seed: UInt64) {
        state = seed
    }

    /// Returns the next 64 bits from a SplitMix64 sequence.
    public mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
        value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
        return value ^ (value >> 31)
    }
}
