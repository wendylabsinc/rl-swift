import Foundation
import Testing
import RLSwift
@testable import RLSwiftMuJoCo

@Suite struct MuJoCoIntegrationTests {
    @Test func reportsBackendSupport() {
        let support = MuJoCoBackendSupport.current

#if SWIFTRL_ENABLE_MUJOCO
        #expect(support.isNativeMuJoCoAvailable)
        #expect(support.version != nil)
        #expect(support.explanation.contains("available"))
#else
        #expect(!support.isNativeMuJoCoAvailable)
        #expect(support.version == nil)
        #expect(support.explanation.contains("MuJoCoBackend"))
#endif
    }

    @Test func storesExplicitSupportInformation() {
        let support = MuJoCoBackendSupport(
            isNativeMuJoCoAvailable: true,
            version: "3.x",
            explanation: "installed"
        )

        #expect(support.isNativeMuJoCoAvailable)
        #expect(support.version == "3.x")
        #expect(support.explanation == "installed")
    }

    @Test func validatesEnvironmentConfigurationAndActionObservationTypes() throws {
        let configuration = try MuJoCoEnvironmentConfiguration(
            modelPath: "humanoid.xml",
            frameSkip: 4,
            maxEpisodeSteps: 128,
            rewardSource: .qvel(index: 0, scale: 2),
            actionMode: .clipped,
            includeContacts: true
        )
        let action = MuJoCoAction(controls: [0.1, -0.2])
        let contact = MuJoCoContact(
            geom1: 0,
            geom2: 1,
            geom1Name: "floor",
            geom2Name: "foot",
            distance: -0.01,
            position: [0, 0, 0],
            normal: [0, 0, 1],
            force: [1, 2, 3, 4, 5, 6]
        )
        let observation = MuJoCoObservation(
            time: 0.25,
            qpos: [1, 2],
            qvel: [3],
            sensorData: [4, 5],
            actuatorActivations: [6],
            controls: [0.1, -0.2],
            contacts: [contact]
        )
        let actuator = MuJoCoActuatorSummary(
            index: 0,
            name: "hip",
            isControlLimited: true,
            minimumControl: -1,
            maximumControl: 1
        )
        let sensor = MuJoCoSensorSummary(index: 0, name: "imu", address: 0, dimension: 3)
        let summary = MuJoCoModelSummary(
            qposCount: 2,
            qvelCount: 1,
            actuatorCount: 2,
            sensorDataCount: 2,
            actuators: [actuator],
            sensors: [sensor]
        )
        let snapshot = MuJoCoStateSnapshot(signature: .integration, values: [0, 1, 2])

        #expect(configuration.modelPath == "humanoid.xml")
        #expect(configuration.frameSkip == 4)
        #expect(configuration.maxEpisodeSteps == 128)
        #expect(configuration.actionMode == .clipped)
        #expect(configuration.includeContacts)
        #expect(action.controls == [0.1, -0.2])
        #expect(observation.flattenedFeatures == [1, 2, 3, 4, 5])
        #expect(observation.actuatorActivations == [6])
        #expect(observation.controls == [0.1, -0.2])
        #expect(observation.contacts == [contact])
        #expect(summary.actuatorCount == 2)
        #expect(summary.actuators == [actuator])
        #expect(summary.sensors == [sensor])
        #expect(snapshot.signature == .integration)
        #expect(snapshot.values == [0, 1, 2])

        #expect(throws: RLSwiftError.emptyIdentifier(name: "mujoco.modelPath")) {
            _ = try MuJoCoEnvironmentConfiguration(modelPath: "")
        }
        #expect(throws: RLSwiftError.invalidHorizon(0)) {
            _ = try MuJoCoEnvironmentConfiguration(modelPath: "model.xml", frameSkip: 0)
        }
        #expect(throws: RLSwiftError.invalidHorizon(0)) {
            _ = try MuJoCoEnvironmentConfiguration(modelPath: "model.xml", maxEpisodeSteps: 0)
        }
    }

    @Test func rewardSourcesReadMuJoCoObservationComponents() throws {
        let observation = MuJoCoObservation(
            time: 1,
            qpos: [0.9, 1.2],
            qvel: [2, -1],
            sensorData: [3, 4]
        )

        #expect(try MuJoCoRewardSource.constant(1.5).reward(for: observation) == 1.5)
        #expect(try MuJoCoRewardSource.sensor(index: 1, scale: 2, offset: -1).reward(for: observation) == 7)
        #expect(try MuJoCoRewardSource.qvel(index: 0, scale: 0.5).reward(for: observation) == 1)
        #expect(try MuJoCoRewardSource.qposTarget(
            index: 0,
            target: 1,
            tolerance: 0.2,
            successReward: 10,
            failureReward: -1
        ).reward(for: observation) == 10)
        #expect(try MuJoCoRewardSource.qposTarget(
            index: 1,
            target: 1,
            tolerance: 0.1,
            successReward: 10,
            failureReward: -1
        ).reward(for: observation) == -1)

        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 3)) {
            _ = try MuJoCoRewardSource.sensor(index: 2, scale: 1, offset: 0).reward(for: observation)
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 0)) {
            _ = try MuJoCoRewardSource.qvel(index: -1, scale: 1).reward(for: observation)
        }
        #expect(throws: RLSwiftError.invalidWeight(-0.1)) {
            _ = try MuJoCoRewardSource.qposTarget(
                index: 0,
                target: 1,
                tolerance: -0.1,
                successReward: 1,
                failureReward: 0
            ).reward(for: observation)
        }
    }

    @Test func environmentReportsUnavailableWithoutMuJoCoTrait() throws {
        let configuration = try MuJoCoEnvironmentConfiguration(modelPath: "model.xml")

#if SWIFTRL_ENABLE_MUJOCO
        #expect(Bool(true))
#else
        #expect(throws: MuJoCoBackendError.backendUnavailable(MuJoCoBackendSupport.current.explanation)) {
            _ = try MuJoCoEnvironment(configuration: configuration)
        }
        var environment = MuJoCoEnvironment(
            configuration: configuration,
            cachedObservation: MuJoCoObservation(time: 0, qpos: [1], qvel: [2], sensorData: [3])
        )
        #expect(environment.modelSummary == nil)
        #expect(environment.reset().flattenedFeatures == [1, 2, 3])
        #expect(try environment.reset(to: .modelDefault).flattenedFeatures == [1, 2, 3])
        #expect(throws: MuJoCoBackendError.backendUnavailable(MuJoCoBackendSupport.current.explanation)) {
            _ = try environment.reset(to: .keyframe(0))
        }
        #expect(throws: MuJoCoBackendError.backendUnavailable(MuJoCoBackendSupport.current.explanation)) {
            _ = try environment.reset(to: .state(MuJoCoStateSnapshot(signature: .integration, values: [])))
        }
        #expect(throws: MuJoCoBackendError.backendUnavailable(MuJoCoBackendSupport.current.explanation)) {
            _ = try environment.step(MuJoCoAction(controls: []))
        }
#endif
    }

#if SWIFTRL_ENABLE_MUJOCO
    @Test func nativeMuJoCoStepsSimpleModelWhenTraitIsEnabled() throws {
        let modelURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("rlswift-mujoco-\(UUID().uuidString).xml")
        let xml = """
        <mujoco>
          <compiler autolimits="true"/>
          <option timestep="0.01"/>
          <worldbody>
            <body name="body" pos="0 0 0">
              <joint name="slide" type="slide" axis="1 0 0"/>
              <geom type="sphere" size="0.05" mass="1"/>
            </body>
          </worldbody>
          <actuator>
            <motor name="slide_motor" joint="slide" gear="1" ctrlrange="-2 2"/>
          </actuator>
          <sensor>
            <jointpos name="slide_position" joint="slide"/>
          </sensor>
          <keyframe>
            <key name="offset" qpos="0.5" qvel="0" ctrl="0"/>
          </keyframe>
        </mujoco>
        """
        try xml.write(to: modelURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: modelURL) }

        let configuration = try MuJoCoEnvironmentConfiguration(
            modelPath: modelURL.path,
            frameSkip: 2,
            maxEpisodeSteps: 1,
            rewardSource: .qvel(index: 0, scale: 1),
            actionMode: .normalizedToControlRange
        )
        var environment = try MuJoCoEnvironment(configuration: configuration)
        let initial = environment.reset()
        let summary = try #require(environment.modelSummary)
        let keyframeObservation = try environment.reset(to: .keyframe(0))
        let result = try environment.step(MuJoCoAction(controls: [2]))

        #expect(summary.qposCount == 1)
        #expect(summary.qvelCount == 1)
        #expect(summary.actuatorCount == 1)
        #expect(summary.sensorDataCount == 1)
        #expect(summary.actuators.first?.name == "slide_motor")
        #expect(summary.actuators.first?.isControlLimited == true)
        #expect(summary.actuators.first?.minimumControl == -2)
        #expect(summary.actuators.first?.maximumControl == 2)
        #expect(summary.sensors.first == MuJoCoSensorSummary(index: 0, name: "slide_position", address: 0, dimension: 1))
        #expect(initial.qpos.count == 1)
        #expect(abs(keyframeObservation.qpos[0] - 0.5) < 1e-9)
        #expect(result.observation.time > 0)
        #expect(result.observation.controls == [2])
        #expect(result.termination.isTruncated)
        #expect(throws: MuJoCoBackendError.invalidControlCount(expected: 1, actual: 0)) {
            _ = try environment.step(MuJoCoAction(controls: []))
        }
        #expect(throws: MuJoCoBackendError.invalidKeyframe(index: 1, count: 1)) {
            _ = try environment.reset(to: .keyframe(1))
        }
    }

    @Test func nativeMuJoCoSnapshotsRestoreAndReportContacts() throws {
        let modelURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("rlswift-mujoco-contact-\(UUID().uuidString).xml")
        let xml = """
        <mujoco>
          <option timestep="0.01" gravity="0 0 -9.81"/>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 0.1"/>
            <body name="ball_body" pos="0 0 0.04">
              <freejoint/>
              <geom name="ball" type="sphere" size="0.05" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        try xml.write(to: modelURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: modelURL) }

        let simulation = try MuJoCoSimulation(modelPath: modelURL.path)
        let resetObservation = simulation.reset(includeContacts: true)
        let snapshot = simulation.stateSnapshot(signature: .integration)
        let steppedObservation = try simulation.step(controls: [], frameSkip: 3, includeContacts: true)
        let restoredObservation = try simulation.restore(snapshot: snapshot, includeContacts: true)

        #expect(!snapshot.values.isEmpty)
        #expect(steppedObservation.time > resetObservation.time)
        #expect(abs(restoredObservation.time - resetObservation.time) < 1e-9)
        #expect(!restoredObservation.contacts.isEmpty)
        #expect(restoredObservation.contacts.contains { $0.geom1Name == "floor" || $0.geom2Name == "floor" })
        #expect(restoredObservation.contacts.contains { $0.geom1Name == "ball" || $0.geom2Name == "ball" })
        #expect(restoredObservation.contacts.allSatisfy { $0.position.count == 3 && $0.normal.count == 3 && $0.force.count == 6 })
        #expect(throws: MuJoCoBackendError.invalidStateSize(expected: snapshot.values.count, actual: 1)) {
            _ = try simulation.restore(snapshot: MuJoCoStateSnapshot(signature: .integration, values: [0]))
        }
    }
#endif
}
