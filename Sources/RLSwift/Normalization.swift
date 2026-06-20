import Foundation

/// Tracks running mean and variance for vector observations.
public struct ObservationNormalizer: Sendable, Equatable {
    private let dimension: Int
    private var meanStorage: [Double]
    private var m2Storage: [Double]

    /// The number of samples incorporated into the statistics.
    public private(set) var count: Int

    /// Creates a normalizer for fixed-width feature vectors.
    public init(dimension: Int) throws {
        guard dimension > 0 else {
            throw RLSwiftError.dimensionMismatch(expected: 1, actual: dimension)
        }
        self.dimension = dimension
        meanStorage = Array(repeating: 0, count: dimension)
        m2Storage = Array(repeating: 0, count: dimension)
        count = 0
    }

    /// The running mean for each dimension.
    public var mean: [Double] {
        meanStorage
    }

    /// The population variance for each dimension.
    public var variance: [Double] {
        guard count > 0 else {
            return Array(repeating: 0, count: dimension)
        }
        var scratch = VectorScratch(count: dimension)
        for index in 0..<dimension {
            scratch.set(m2Storage[index] / Double(count), at: index)
        }
        return scratch.finish()
    }

    /// Incorporates one sample using Welford's online algorithm.
    public mutating func update(with sample: [Double]) throws {
        try validate(sample)
        count += 1
        for index in 0..<dimension {
            let delta = sample[index] - meanStorage[index]
            meanStorage[index] += delta / Double(count)
            let delta2 = sample[index] - meanStorage[index]
            m2Storage[index] += delta * delta2
        }
    }

    /// Normalizes a sample with the current statistics.
    public func normalize(_ sample: [Double], epsilon: Double = 1e-8) throws -> [Double] {
        try validate(sample)
        let currentVariance = variance
        var scratch = VectorScratch(count: dimension)
        for index in 0..<dimension {
            let standardDeviation = sqrt(currentVariance[index] + epsilon)
            scratch.set((sample[index] - meanStorage[index]) / standardDeviation, at: index)
        }
        return scratch.finish()
    }

    /// Resets all accumulated statistics.
    public mutating func reset() {
        meanStorage = Array(repeating: 0, count: dimension)
        m2Storage = Array(repeating: 0, count: dimension)
        count = 0
    }

    private func validate(_ sample: [Double]) throws {
        guard sample.count == dimension else {
            throw RLSwiftError.dimensionMismatch(expected: dimension, actual: sample.count)
        }
    }
}
