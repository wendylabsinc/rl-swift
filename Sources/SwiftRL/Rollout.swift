/// Return-calculation utilities for rollout-based reinforcement learning.
public extension Episode {
    /// Computes a discounted return starting at every transition.
    func discountedReturns(gamma: Double) -> [Double] {
        let recordedTransitions = transitions
        var result = Array(repeating: 0.0, count: recordedTransitions.count)
        var running = 0.0
        for index in recordedTransitions.indices.reversed() {
            running = recordedTransitions[index].reward + gamma * running
            result[index] = running
        }
        return result
    }

    /// Computes finite-horizon n-step returns for every transition.
    func nStepReturns(gamma: Double, horizon: Int, bootstrap: Double = 0) throws -> [Double] {
        guard horizon > 0 else {
            throw SwiftRLError.invalidHorizon(horizon)
        }
        let recordedTransitions = transitions
        return recordedTransitions.indices.map { startIndex in
            var multiplier = 1.0
            var total = 0.0
            var offset = 0
            while offset < horizon, startIndex + offset < recordedTransitions.count {
                let transition = recordedTransitions[startIndex + offset]
                total += multiplier * transition.reward
                if transition.isTerminal {
                    return total
                }
                multiplier *= gamma
                offset += 1
            }
            return total + multiplier * bootstrap
        }
    }
}
