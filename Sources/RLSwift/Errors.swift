/// Errors thrown by RLSwift when an algorithm or data structure receives invalid input.
public enum RLSwiftError: Error, Equatable, Sendable {
    /// Indicates that an action-dependent component was created without any available actions.
    case emptyActionSpace

    /// Indicates that a probability value was outside the closed `0...1` range.
    case invalidProbability(Double)

    /// Indicates that a softmax temperature was not strictly positive.
    case invalidTemperature(Double)

    /// Indicates that a bounded storage container was created with a non-positive capacity.
    case invalidCapacity(Int)

    /// Indicates that a sample request used a negative count.
    case invalidSampleCount(Int)

    /// Indicates that a lower/upper bound pair is invalid at an index.
    case invalidBounds(index: Int, lower: Double, upper: Double)

    /// Indicates that a vector had a different dimension than required.
    case dimensionMismatch(expected: Int, actual: Int)

    /// Indicates that an n-step or rollout horizon was not positive.
    case invalidHorizon(Int)

    /// Indicates that a duration field was negative.
    case invalidDuration(name: String, value: Double)

    /// Indicates that a weighting value was negative.
    case invalidWeight(Double)

    /// Indicates that a replay priority was not strictly positive.
    case invalidPriority(Double)

    /// Indicates that a stable identifier or name was empty.
    case emptyIdentifier(name: String)

    /// Indicates that a stable identifier appeared more than once.
    case duplicateIdentifier(String)

    /// Indicates that an indexed collection contained duplicate indices.
    case duplicateIndex(Int)

    /// Indicates that a versioned contract or manifest used an empty version string.
    case invalidVersion(String)
}
