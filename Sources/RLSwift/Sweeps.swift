import Foundation

/// One hyperparameter and the values to try in a grid sweep.
public struct SweepParameter: Sendable, Equatable, Codable {
    /// Parameter name.
    public let name: String

    /// Candidate values.
    public let values: [Double]

    /// Creates a sweep parameter.
    public init(name: String, values: [Double]) throws {
        guard !name.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "sweep.parameter")
        }
        guard !values.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        self.name = name
        self.values = values
    }
}

/// One concrete trial in a hyperparameter sweep.
public struct SweepTrial: Sendable, Equatable, Codable {
    /// Stable trial identifier.
    public let id: String

    /// Hyperparameter values for the trial.
    public let parameters: [String: Double]

    /// Deterministic seed for the trial.
    public let seed: UInt64

    /// Creates a sweep trial.
    public init(id: String, parameters: [String: Double], seed: UInt64) throws {
        guard !id.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "sweep.trialID")
        }
        guard !parameters.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        self.id = id
        self.parameters = parameters
        self.seed = seed
    }
}

/// A deterministic grid of hyperparameter trials.
public struct SweepPlan: Sendable, Equatable, Codable {
    /// Trials in execution order.
    public let trials: [SweepTrial]

    /// Creates a sweep plan.
    public init(trials: [SweepTrial]) throws {
        guard !trials.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        var seen: Set<String> = []
        for trial in trials {
            guard seen.insert(trial.id).inserted else {
                throw RLSwiftError.duplicateIdentifier(trial.id)
            }
        }
        self.trials = trials
    }

    /// Builds a deterministic grid sweep from parameters and seeds.
    public static func grid(parameters: [SweepParameter], seeds: [UInt64]) throws -> SweepPlan {
        guard !parameters.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        guard !seeds.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        var names: Set<String> = []
        for parameter in parameters {
            guard names.insert(parameter.name).inserted else {
                throw RLSwiftError.duplicateIdentifier(parameter.name)
            }
        }

        var assignments: [[String: Double]] = [[:]]
        for parameter in parameters {
            var next: [[String: Double]] = []
            for assignment in assignments {
                for value in parameter.values {
                    var updated = assignment
                    updated[parameter.name] = value
                    next.append(updated)
                }
            }
            assignments = next
        }

        var trials: [SweepTrial] = []
        for seed in seeds {
            for index in assignments.indices {
                trials.append(try SweepTrial(
                    id: "seed-\(seed)-trial-\(index)",
                    parameters: assignments[index],
                    seed: seed
                ))
            }
        }
        return try SweepPlan(trials: trials)
    }
}

/// Result from one sweep trial.
public struct SweepResult: Sendable, Equatable, Codable {
    /// Completed trial.
    public let trial: SweepTrial

    /// Higher-is-better score.
    public let score: Double

    /// Lower-is-better cost, commonly wall-clock time or GPU hours.
    public let cost: Double

    /// Creates a sweep result.
    public init(trial: SweepTrial, score: Double, cost: Double) throws {
        guard cost >= 0 else {
            throw RLSwiftError.invalidDuration(name: "sweep.cost", value: cost)
        }
        self.trial = trial
        self.score = score
        self.cost = cost
    }
}

/// Pareto-frontier selection for score/cost tuning.
public enum SweepTuner {
    /// Returns all results that are not dominated by another result with greater-or-equal score and lower-or-equal cost.
    public static func paretoFrontier(_ results: [SweepResult]) throws -> [SweepResult] {
        guard !results.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        return results.filter { candidate in
            !results.contains { other in
                other.trial.id != candidate.trial.id
                    && other.score >= candidate.score
                    && other.cost <= candidate.cost
                    && (other.score > candidate.score || other.cost < candidate.cost)
            }
        }
        .sorted {
            $0.score == $1.score ? $0.cost < $1.cost : $0.score > $1.score
        }
    }
}

/// Scale used to map Protein tuner unit samples into parameter values.
public enum ProteinParameterScale: String, Sendable, Equatable, Codable {
    /// Linear interpolation between lower and upper bounds.
    case linear

    /// Log-space interpolation between positive lower and upper bounds.
    case logarithmic
}

/// Bounded hyperparameter used by the Protein-style tuner.
public struct ProteinParameter: Sendable, Equatable, Codable {
    /// Parameter name.
    public let name: String

    /// Lower inclusive bound.
    public let lowerBound: Double

    /// Upper inclusive bound.
    public let upperBound: Double

    /// Parameter interpolation scale.
    public let scale: ProteinParameterScale

    /// Creates a bounded Protein parameter.
    public init(
        name: String,
        lowerBound: Double,
        upperBound: Double,
        scale: ProteinParameterScale = .linear
    ) throws {
        guard !name.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "protein.parameter")
        }
        guard upperBound > lowerBound else {
            throw RLSwiftError.invalidBounds(index: 0, lower: lowerBound, upper: upperBound)
        }
        if scale == .logarithmic {
            guard lowerBound > 0 else {
                throw RLSwiftError.invalidBounds(index: 0, lower: lowerBound, upper: upperBound)
            }
        }
        self.name = name
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.scale = scale
    }

    /// Maps a unit value in `0...1` into this parameter's bounded range.
    public func value(unit: Double) throws -> Double {
        guard (0...1).contains(unit) else {
            throw RLSwiftError.invalidProbability(unit)
        }
        switch scale {
        case .linear:
            return lowerBound + (upperBound - lowerBound) * unit
        case .logarithmic:
            return exp(log(lowerBound) + (log(upperBound) - log(lowerBound)) * unit)
        }
    }

    /// Maps a parameter value back to unit space.
    public func unit(value: Double) throws -> Double {
        guard (lowerBound...upperBound).contains(value) else {
            throw RLSwiftError.invalidBounds(index: 0, lower: lowerBound, upper: upperBound)
        }
        switch scale {
        case .linear:
            return (value - lowerBound) / (upperBound - lowerBound)
        case .logarithmic:
            return (log(value) - log(lowerBound)) / (log(upperBound) - log(lowerBound))
        }
    }
}

/// One trial suggested by the Protein-style tuner.
public struct ProteinSuggestion: Sendable, Equatable, Codable {
    /// Stable suggestion identifier.
    public let id: String

    /// Suggested parameter values.
    public let parameters: [String: Double]

    /// Suggested trial seed.
    public let seed: UInt64

    /// Creates one Protein suggestion.
    public init(id: String, parameters: [String: Double], seed: UInt64) throws {
        guard !id.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "protein.suggestionID")
        }
        guard !parameters.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        self.id = id
        self.parameters = parameters
        self.seed = seed
    }
}

/// Completed Protein-style tuning observation.
public struct ProteinObservation: Sendable, Equatable, Codable {
    /// Suggested trial that was evaluated.
    public let suggestion: ProteinSuggestion

    /// Higher-is-better objective score.
    public let score: Double

    /// Lower-is-better trial cost.
    public let cost: Double

    /// Creates one completed tuning observation.
    public init(suggestion: ProteinSuggestion, score: Double, cost: Double) throws {
        guard cost >= 0 else {
            throw RLSwiftError.invalidDuration(name: "protein.cost", value: cost)
        }
        self.suggestion = suggestion
        self.score = score
        self.cost = cost
    }
}

/// Pure Swift Protein-style tuner for bounded hyperparameter search.
public struct ProteinTuner: Sendable, Equatable, Codable {
    /// Search parameters.
    public let parameters: [ProteinParameter]

    /// Base seed for deterministic suggestions.
    public let seed: UInt64

    /// Creates a Protein-style tuner.
    public init(parameters: [ProteinParameter], seed: UInt64 = 0) throws {
        guard !parameters.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        var names: Set<String> = []
        for parameter in parameters {
            guard names.insert(parameter.name).inserted else {
                throw RLSwiftError.duplicateIdentifier(parameter.name)
            }
        }
        self.parameters = parameters
        self.seed = seed
    }

    /// Suggests new trials from low-discrepancy exploration or Pareto-front refinement.
    public func suggest(completed: [ProteinObservation] = [], count: Int) throws -> [ProteinSuggestion] {
        guard count > 0 else {
            throw RLSwiftError.invalidSampleCount(count)
        }
        let frontier = completed.isEmpty ? [] : try Self.paretoFrontier(completed)
        var suggestions: [ProteinSuggestion] = []
        for suggestionIndex in 0..<count {
            var values: [String: Double] = [:]
            for parameterIndex in parameters.indices {
                let parameter = parameters[parameterIndex]
                let unit: Double
                if frontier.isEmpty {
                    unit = Self.lowDiscrepancyUnit(seed: seed, sampleIndex: suggestionIndex, parameterIndex: parameterIndex)
                } else {
                    let anchor = frontier[suggestionIndex % frontier.count]
                    let anchorValue: Double
                    if let existingValue = anchor.suggestion.parameters[parameter.name] {
                        anchorValue = existingValue
                    } else {
                        anchorValue = try parameter.value(unit: 0.5)
                    }
                    let anchorUnit = try parameter.unit(value: anchorValue)
                    let jitter = (Self.lowDiscrepancyUnit(
                        seed: seed,
                        sampleIndex: completed.count + suggestionIndex,
                        parameterIndex: parameterIndex
                    ) - 0.5) * 0.25
                    unit = min(max(anchorUnit + jitter, 0), 1)
                }
                values[parameter.name] = try parameter.value(unit: unit)
            }
            suggestions.append(try ProteinSuggestion(
                id: "protein-\(completed.count + suggestionIndex)",
                parameters: values,
                seed: seed &+ UInt64(completed.count + suggestionIndex)
            ))
        }
        return suggestions
    }

    /// Returns non-dominated observations ordered by score and cost.
    public static func paretoFrontier(_ observations: [ProteinObservation]) throws -> [ProteinObservation] {
        guard !observations.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        return observations.filter { candidate in
            !observations.contains { other in
                other.suggestion.id != candidate.suggestion.id
                    && other.score >= candidate.score
                    && other.cost <= candidate.cost
                    && (other.score > candidate.score || other.cost < candidate.cost)
            }
        }
        .sorted {
            $0.score == $1.score ? $0.cost < $1.cost : $0.score > $1.score
        }
    }

    private static func lowDiscrepancyUnit(seed: UInt64, sampleIndex: Int, parameterIndex: Int) -> Double {
        let seedOffset = Double((seed % 997) + 1) / 997.0
        let raw = seedOffset
            + Double(sampleIndex + 1) * 0.618_033_988_749_894_9
            + Double(parameterIndex + 1) * 0.414_213_562_373_095_1
        return raw - floor(raw)
    }
}
