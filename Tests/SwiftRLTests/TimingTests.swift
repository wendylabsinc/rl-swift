import Testing
@testable import SwiftRL

@Suite struct TimingTests {
    @Test func validatesTimingDurations() {
        #expect(throws: SwiftRLError.invalidDuration(name: "deltaTime", value: -0.1)) {
            _ = try ControlTiming(stepIndex: 0, deltaTime: -0.1)
        }
        #expect(throws: SwiftRLError.invalidDuration(name: "sensorAge", value: -0.1)) {
            _ = try ControlTiming(stepIndex: 0, deltaTime: 0.02, sensorAge: -0.1)
        }
        #expect(throws: SwiftRLError.invalidDuration(name: "actionLatency", value: -0.1)) {
            _ = try ControlTiming(stepIndex: 0, deltaTime: 0.02, actionLatency: -0.1)
        }
    }

    @Test func computesClosedLoopLatencyAndDeadlines() throws {
        let timing = try ControlTiming(stepIndex: 4, deltaTime: 0.02, sensorAge: 0.01, actionLatency: 0.015)

        #expect(timing.stepIndex == 4)
        #expect(timing.deltaTime == 0.02)
        #expect(timing.closedLoopLatency == 0.025)
        #expect(try timing.missesDeadline(maximumLatency: 0.02))
        #expect(!(try timing.missesDeadline(maximumLatency: 0.03)))
        #expect(throws: SwiftRLError.invalidDuration(name: "maximumLatency", value: -1)) {
            _ = try timing.missesDeadline(maximumLatency: -1)
        }
    }
}
