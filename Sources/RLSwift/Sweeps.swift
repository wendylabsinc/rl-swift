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
