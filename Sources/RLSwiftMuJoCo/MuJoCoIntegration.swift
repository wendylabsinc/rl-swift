import Foundation
public import RLSwift
#if SWIFTRL_ENABLE_MUJOCO
import CMuJoCo
#endif

/// Errors thrown by the RLSwift MuJoCo backend.
public enum MuJoCoBackendError: Error, Equatable, Sendable {
    /// The package was built without the `MuJoCoBackend` trait.
    case backendUnavailable(String)

    /// MuJoCo could not load the requested MJCF/XML model.
    case modelLoadFailed(path: String, message: String)

    /// MuJoCo could not allocate `mjData` for a loaded model.
    case dataAllocationFailed(path: String)

    /// The action vector did not match the model actuator count.
    case invalidControlCount(expected: Int, actual: Int)

    /// The requested keyframe index does not exist in the loaded model.
    case invalidKeyframe(index: Int, count: Int)

    /// A saved state vector does not match the requested MuJoCo state signature.
    case invalidStateSize(expected: Int, actual: Int)
}

/// Runtime support information for the RLSwift MuJoCo backend.
public struct MuJoCoBackendSupport: Equatable, Sendable {
    /// Whether native MuJoCo simulation is compiled into this build.
    public let isNativeMuJoCoAvailable: Bool

    /// MuJoCo runtime version string when available.
    public let version: String?

    /// A short explanation suitable for diagnostics or setup output.
    public let explanation: String

    /// Creates MuJoCo support information.
    public init(isNativeMuJoCoAvailable: Bool, version: String? = nil, explanation: String) {
        self.isNativeMuJoCoAvailable = isNativeMuJoCoAvailable
        self.version = version
        self.explanation = explanation
    }

    /// Support information for the currently compiled platform.
#if SWIFTRL_ENABLE_MUJOCO
    public static var current: MuJoCoBackendSupport {
        MuJoCoBackendSupport(
            isNativeMuJoCoAvailable: true,
            version: MuJoCoRuntime.versionString,
            explanation: "MuJoCo backend is available for this build."
        )
    }
#else
    public static let current = MuJoCoBackendSupport(
        isNativeMuJoCoAvailable: false,
        explanation: "MuJoCo backend is disabled; build with the MuJoCoBackend trait and pkg-config-visible MuJoCo headers/libraries to enable it."
    )
#endif
}

/// Action controls applied to MuJoCo actuators.
public struct MuJoCoAction: Sendable, Equatable, Codable {
    /// Control values in MuJoCo actuator order.
    public let controls: [Double]

    /// Creates one MuJoCo control action.
    public init(controls: [Double]) {
        self.controls = controls
    }
}

/// How action values are interpreted before they are written to `mjData.ctrl`.
public enum MuJoCoActionMode: Sendable, Equatable, Codable {
    /// Write controls exactly as supplied.
    case direct

    /// Clamp controls to the model actuator control range when a range exists.
    case clipped

    /// Treat each control as `[-1, 1]` and map it into the actuator control range.
    case normalizedToControlRange
}

/// Control metadata for one MuJoCo actuator.
public struct MuJoCoActuatorSummary: Sendable, Equatable, Codable {
    /// Actuator index in MuJoCo control order.
    public let index: Int

    /// Actuator name when the MJCF model defines one.
    public let name: String?

    /// Whether the actuator declares a control limit.
    public let isControlLimited: Bool

    /// Lower control bound when `isControlLimited` is true.
    public let minimumControl: Double?

    /// Upper control bound when `isControlLimited` is true.
    public let maximumControl: Double?

    /// Creates actuator metadata.
    public init(
        index: Int,
        name: String? = nil,
        isControlLimited: Bool,
        minimumControl: Double? = nil,
        maximumControl: Double? = nil
    ) {
        self.index = index
        self.name = name
        self.isControlLimited = isControlLimited
        self.minimumControl = minimumControl
        self.maximumControl = maximumControl
    }
}

/// Sensor metadata for one MuJoCo sensor.
public struct MuJoCoSensorSummary: Sendable, Equatable, Codable {
    /// Sensor index in MuJoCo sensor order.
    public let index: Int

    /// Sensor name when the MJCF model defines one.
    public let name: String?

    /// First element in `mjData.sensordata`.
    public let address: Int

    /// Number of scalar values emitted by this sensor.
    public let dimension: Int

    /// Creates sensor metadata.
    public init(index: Int, name: String? = nil, address: Int, dimension: Int) {
        self.index = index
        self.name = name
        self.address = address
        self.dimension = dimension
    }
}

/// Contact information detected by MuJoCo collision processing.
public struct MuJoCoContact: Sendable, Equatable, Codable {
    /// First MuJoCo geometry id.
    public let geom1: Int

    /// Second MuJoCo geometry id.
    public let geom2: Int

    /// First geometry name when available.
    public let geom1Name: String?

    /// Second geometry name when available.
    public let geom2Name: String?

    /// Signed distance between nearest points; negative values indicate penetration.
    public let distance: Double

    /// Contact point in world coordinates.
    public let position: [Double]

    /// Contact normal pointing from geom1 to geom2.
    public let normal: [Double]

    /// Contact force and torque in MuJoCo's six-element contact-frame order.
    public let force: [Double]

    /// Creates contact metadata.
    public init(
        geom1: Int,
        geom2: Int,
        geom1Name: String? = nil,
        geom2Name: String? = nil,
        distance: Double,
        position: [Double],
        normal: [Double],
        force: [Double] = []
    ) {
        self.geom1 = geom1
        self.geom2 = geom2
        self.geom1Name = geom1Name
        self.geom2Name = geom2Name
        self.distance = distance
        self.position = position
        self.normal = normal
        self.force = force
    }
}

/// Observation read from MuJoCo `mjData` after reset or step.
public struct MuJoCoObservation: Sendable, Equatable, Codable {
    /// Simulation time in seconds.
    public let time: Double

    /// Generalized positions.
    public let qpos: [Double]

    /// Generalized velocities.
    public let qvel: [Double]

    /// Sensor values in MuJoCo sensor-data order.
    public let sensorData: [Double]

    /// Actuator activation state.
    public let actuatorActivations: [Double]

    /// Last control values written to `mjData.ctrl`.
    public let controls: [Double]

    /// Detected contacts when requested by the environment configuration.
    public let contacts: [MuJoCoContact]

    /// Creates a MuJoCo observation.
    public init(
        time: Double,
        qpos: [Double],
        qvel: [Double],
        sensorData: [Double],
        actuatorActivations: [Double] = [],
        controls: [Double] = [],
        contacts: [MuJoCoContact] = []
    ) {
        self.time = time
        self.qpos = qpos
        self.qvel = qvel
        self.sensorData = sensorData
        self.actuatorActivations = actuatorActivations
        self.controls = controls
        self.contacts = contacts
    }

    /// Flattened model-input features in `qpos`, `qvel`, `sensorData` order.
    public var flattenedFeatures: [Double] {
        qpos + qvel + sensorData
    }
}

/// Lightweight model metadata exposed by the MuJoCo backend.
public struct MuJoCoModelSummary: Sendable, Equatable, Codable {
    /// Number of generalized position coordinates.
    public let qposCount: Int

    /// Number of generalized velocity coordinates.
    public let qvelCount: Int

    /// Number of actuator controls.
    public let actuatorCount: Int

    /// Number of sensor-data values.
    public let sensorDataCount: Int

    /// Actuator metadata in MuJoCo control order.
    public let actuators: [MuJoCoActuatorSummary]

    /// Sensor metadata in MuJoCo sensor order.
    public let sensors: [MuJoCoSensorSummary]

    /// Creates model summary metadata.
    public init(
        qposCount: Int,
        qvelCount: Int,
        actuatorCount: Int,
        sensorDataCount: Int,
        actuators: [MuJoCoActuatorSummary] = [],
        sensors: [MuJoCoSensorSummary] = []
    ) {
        self.qposCount = qposCount
        self.qvelCount = qvelCount
        self.actuatorCount = actuatorCount
        self.sensorDataCount = sensorDataCount
        self.actuators = actuators
        self.sensors = sensors
    }
}

/// MuJoCo state components captured in a snapshot.
public enum MuJoCoStateSignature: String, Sendable, Equatable, Codable {
    /// Position, velocity, actuator activation, and history buffers.
    case physics

    /// Time plus physics state and plugin state.
    case fullPhysics

    /// User-controlled inputs such as controls, applied forces, mocap, and userdata.
    case user

    /// Full state needed to continue integration including warmstart data.
    case integration
}

/// Saved MuJoCo state vector for deterministic replay or branch-and-restore evaluation.
public struct MuJoCoStateSnapshot: Sendable, Equatable, Codable {
    /// State signature used to capture `values`.
    public let signature: MuJoCoStateSignature

    /// Raw state vector in MuJoCo signature order.
    public let values: [Double]

    /// Creates a state snapshot.
    public init(signature: MuJoCoStateSignature, values: [Double]) {
        self.signature = signature
        self.values = values
    }
}

/// Reset source for a MuJoCo simulation.
public enum MuJoCoResetMode: Sendable, Equatable, Codable {
    /// Reset to the model default state.
    case modelDefault

    /// Reset to a keyframe embedded in the loaded MJCF model.
    case keyframe(Int)

    /// Restore a previously captured state vector.
    case state(MuJoCoStateSnapshot)
}

/// Reward source for generic MuJoCo environment smoke loops.
public enum MuJoCoRewardSource: Sendable, Equatable, Codable {
    /// Fixed reward per environment step.
    case constant(Double)

    /// Reward equal to a scaled sensor value plus an offset.
    case sensor(index: Int, scale: Double, offset: Double)

    /// Reward equal to a scaled generalized velocity.
    case qvel(index: Int, scale: Double)

    /// Sparse reward when one position coordinate is within tolerance of a target.
    case qposTarget(index: Int, target: Double, tolerance: Double, successReward: Double, failureReward: Double)

    /// Computes a reward from an observation.
    public func reward(for observation: MuJoCoObservation) throws -> Double {
        switch self {
        case let .constant(value):
            return value
        case let .sensor(index, scale, offset):
            let value = try Self.value(at: index, in: observation.sensorData)
            return value * scale + offset
        case let .qvel(index, scale):
            let value = try Self.value(at: index, in: observation.qvel)
            return value * scale
        case let .qposTarget(index, target, tolerance, successReward, failureReward):
            guard tolerance >= 0 else {
                throw RLSwiftError.invalidWeight(tolerance)
            }
            let value = try Self.value(at: index, in: observation.qpos)
            return abs(value - target) <= tolerance ? successReward : failureReward
        }
    }

    private static func value(at index: Int, in values: [Double]) throws -> Double {
        guard index >= 0, index < values.count else {
            throw RLSwiftError.dimensionMismatch(expected: values.count, actual: index + 1)
        }
        return values[index]
    }
}

/// Configuration for wrapping one MuJoCo model as an RLSwift environment.
public struct MuJoCoEnvironmentConfiguration: Sendable, Equatable, Codable {
    /// Path to an MJCF/XML model file.
    public let modelPath: String

    /// Number of MuJoCo physics steps to advance per RLSwift environment step.
    public let frameSkip: Int

    /// Maximum RLSwift environment steps before truncation.
    public let maxEpisodeSteps: Int

    /// Generic reward source for smoke loops.
    public let rewardSource: MuJoCoRewardSource

    /// How input actions are transformed before being written to MuJoCo controls.
    public let actionMode: MuJoCoActionMode

    /// Whether observations should include detected contacts.
    public let includeContacts: Bool

    /// Creates a MuJoCo environment configuration.
    public init(
        modelPath: String,
        frameSkip: Int = 1,
        maxEpisodeSteps: Int = 1_000,
        rewardSource: MuJoCoRewardSource = .constant(0),
        actionMode: MuJoCoActionMode = .direct,
        includeContacts: Bool = false
    ) throws {
        guard !modelPath.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "mujoco.modelPath")
        }
        guard frameSkip > 0 else {
            throw RLSwiftError.invalidHorizon(frameSkip)
        }
        guard maxEpisodeSteps > 0 else {
            throw RLSwiftError.invalidHorizon(maxEpisodeSteps)
        }
        self.modelPath = modelPath
        self.frameSkip = frameSkip
        self.maxEpisodeSteps = maxEpisodeSteps
        self.rewardSource = rewardSource
        self.actionMode = actionMode
        self.includeContacts = includeContacts
    }
}

/// An RLSwift environment backed by one MuJoCo `mjModel`/`mjData` pair.
public struct MuJoCoEnvironment: Environment {
    /// Observation type emitted by the backend.
    public typealias Observation = MuJoCoObservation

    /// Action type consumed by the backend.
    public typealias Action = MuJoCoAction

    /// Environment configuration.
    public let configuration: MuJoCoEnvironmentConfiguration

#if SWIFTRL_ENABLE_MUJOCO
    private let simulation: MuJoCoSimulation
#else
    private var cachedObservation: MuJoCoObservation
#endif
    private var stepIndex: Int

    /// Creates a MuJoCo-backed environment.
    public init(configuration: MuJoCoEnvironmentConfiguration) throws {
        self.configuration = configuration
        stepIndex = 0
#if SWIFTRL_ENABLE_MUJOCO
        simulation = try MuJoCoSimulation(modelPath: configuration.modelPath)
#else
        cachedObservation = MuJoCoObservation(time: 0, qpos: [], qvel: [], sensorData: [])
        throw MuJoCoBackendError.backendUnavailable(MuJoCoBackendSupport.current.explanation)
#endif
    }

#if !SWIFTRL_ENABLE_MUJOCO
    init(configuration: MuJoCoEnvironmentConfiguration, cachedObservation: MuJoCoObservation) {
        self.configuration = configuration
        self.cachedObservation = cachedObservation
        stepIndex = 0
    }
#endif

    /// Current MuJoCo model dimensions.
    public var modelSummary: MuJoCoModelSummary? {
#if SWIFTRL_ENABLE_MUJOCO
        simulation.summary
#else
        nil
#endif
    }

    /// Resets the simulation and returns the initial observation.
    public mutating func reset() -> MuJoCoObservation {
        stepIndex = 0
#if SWIFTRL_ENABLE_MUJOCO
        return simulation.reset(includeContacts: configuration.includeContacts)
#else
        return cachedObservation
#endif
    }

    /// Resets the simulation from a keyframe or saved state and returns the initial observation.
    public mutating func reset(to mode: MuJoCoResetMode) throws -> MuJoCoObservation {
        stepIndex = 0
#if SWIFTRL_ENABLE_MUJOCO
        return try simulation.reset(to: mode, includeContacts: configuration.includeContacts)
#else
        switch mode {
        case .modelDefault:
            return cachedObservation
        case .keyframe, .state:
            throw MuJoCoBackendError.backendUnavailable(MuJoCoBackendSupport.current.explanation)
        }
#endif
    }

    /// Applies actuator controls and advances MuJoCo.
    public mutating func step(_ action: MuJoCoAction) throws -> StepResult<MuJoCoObservation> {
#if SWIFTRL_ENABLE_MUJOCO
        let observation = try simulation.step(
            controls: action.controls,
            frameSkip: configuration.frameSkip,
            actionMode: configuration.actionMode,
            includeContacts: configuration.includeContacts
        )
        stepIndex += 1
        let termination: StepTermination = stepIndex >= configuration.maxEpisodeSteps
            ? .truncated(reason: "max_steps")
            : .continuing
        return StepResult(
            observation: observation,
            reward: try configuration.rewardSource.reward(for: observation),
            isTerminal: termination.endsEpisode,
            termination: termination
        )
#else
        throw MuJoCoBackendError.backendUnavailable(MuJoCoBackendSupport.current.explanation)
#endif
    }
}

#if SWIFTRL_ENABLE_MUJOCO
/// Direct owner for one MuJoCo model/data pair.
public final class MuJoCoSimulation: @unchecked Sendable {
    private let model: UnsafeMutablePointer<mjModel>
    private let data: UnsafeMutablePointer<mjData>

    /// Path used to load the model.
    public let modelPath: String

    /// Loads a MuJoCo model from an MJCF/XML file.
    public init(modelPath: String) throws {
        guard !modelPath.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "mujoco.modelPath")
        }
        var loadError = Array<CChar>(repeating: 0, count: 1_024)
        let loadErrorCapacity = Int32(loadError.count)
        let loadedModel = loadError.withUnsafeMutableBufferPointer { errorBuffer in
            modelPath.withCString { pathCString in
                mj_loadXML(pathCString, nil, errorBuffer.baseAddress, loadErrorCapacity)
            }
        }
        guard let loadedModel else {
            let message = loadError.withUnsafeBufferPointer { buffer in
                String(cString: buffer.baseAddress!)
            }
            throw MuJoCoBackendError.modelLoadFailed(path: modelPath, message: message)
        }
        guard let allocatedData = mj_makeData(loadedModel) else {
            mj_deleteModel(loadedModel)
            throw MuJoCoBackendError.dataAllocationFailed(path: modelPath)
        }
        self.modelPath = modelPath
        model = loadedModel
        data = allocatedData
    }

    deinit {
        mj_deleteData(data)
        mj_deleteModel(model)
    }

    /// Current MuJoCo model dimensions.
    public var summary: MuJoCoModelSummary {
        MuJoCoModelSummary(
            qposCount: Int(model.pointee.nq),
            qvelCount: Int(model.pointee.nv),
            actuatorCount: Int(model.pointee.nu),
            sensorDataCount: Int(model.pointee.nsensordata),
            actuators: actuatorSummaries(),
            sensors: sensorSummaries()
        )
    }

    /// Resets `mjData` for the loaded model and returns the initial observation.
    public func reset(includeContacts: Bool = false) -> MuJoCoObservation {
        mj_resetData(model, data)
        mj_forward(model, data)
        return observation(includeContacts: includeContacts)
    }

    /// Resets `mjData` from a keyframe or saved state.
    public func reset(to mode: MuJoCoResetMode, includeContacts: Bool = false) throws -> MuJoCoObservation {
        switch mode {
        case .modelDefault:
            return reset(includeContacts: includeContacts)
        case let .keyframe(index):
            guard index >= 0, index < Int(model.pointee.nkey) else {
                throw MuJoCoBackendError.invalidKeyframe(index: index, count: Int(model.pointee.nkey))
            }
            mj_resetDataKeyframe(model, data, Int32(index))
            mj_forward(model, data)
            return observation(includeContacts: includeContacts)
        case let .state(snapshot):
            return try restore(snapshot: snapshot, includeContacts: includeContacts)
        }
    }

    /// Writes controls, advances the simulation, and returns the new observation.
    public func step(
        controls: [Double],
        frameSkip: Int,
        actionMode: MuJoCoActionMode = .direct,
        includeContacts: Bool = false
    ) throws -> MuJoCoObservation {
        guard controls.count == Int(model.pointee.nu) else {
            throw MuJoCoBackendError.invalidControlCount(expected: Int(model.pointee.nu), actual: controls.count)
        }
        guard frameSkip > 0 else {
            throw RLSwiftError.invalidHorizon(frameSkip)
        }
        let preparedControls = transformedControls(controls, mode: actionMode)
        if let ctrl = data.pointee.ctrl {
            for index in preparedControls.indices {
                ctrl[index] = mjtNum(preparedControls[index])
            }
        }
        for _ in 0..<frameSkip {
            mj_step(model, data)
        }
        return observation(includeContacts: includeContacts)
    }

    /// Reads the current MuJoCo state without advancing physics.
    public func observation(includeContacts: Bool = false) -> MuJoCoObservation {
        MuJoCoObservation(
            time: Double(data.pointee.time),
            qpos: Self.vector(data.pointee.qpos, count: Int(model.pointee.nq)),
            qvel: Self.vector(data.pointee.qvel, count: Int(model.pointee.nv)),
            sensorData: Self.vector(data.pointee.sensordata, count: Int(model.pointee.nsensordata)),
            actuatorActivations: Self.vector(data.pointee.act, count: Int(model.pointee.na)),
            controls: Self.vector(data.pointee.ctrl, count: Int(model.pointee.nu)),
            contacts: includeContacts ? contactSummaries() : []
        )
    }

    /// Captures a MuJoCo state vector for deterministic replay.
    public func stateSnapshot(signature: MuJoCoStateSignature = .integration) -> MuJoCoStateSnapshot {
        let rawSignature = signature.rawMuJoCoSignature
        let count = Int(mj_stateSize(model, rawSignature))
        var values = Array<mjtNum>(repeating: 0, count: count)
        values.withUnsafeMutableBufferPointer { buffer in
            mj_getState(model, data, buffer.baseAddress, rawSignature)
        }
        return MuJoCoStateSnapshot(signature: signature, values: values.map { Double($0) })
    }

    /// Restores a previously captured MuJoCo state vector.
    public func restore(snapshot: MuJoCoStateSnapshot, includeContacts: Bool = false) throws -> MuJoCoObservation {
        let rawSignature = snapshot.signature.rawMuJoCoSignature
        let expectedCount = Int(mj_stateSize(model, rawSignature))
        guard snapshot.values.count == expectedCount else {
            throw MuJoCoBackendError.invalidStateSize(expected: expectedCount, actual: snapshot.values.count)
        }
        let values = snapshot.values.map { mjtNum($0) }
        values.withUnsafeBufferPointer { buffer in
            mj_setState(model, data, buffer.baseAddress, rawSignature)
        }
        mj_forward(model, data)
        return observation(includeContacts: includeContacts)
    }

    /// Actuator metadata in MuJoCo control order.
    public func actuatorSummaries() -> [MuJoCoActuatorSummary] {
        let count = Int(model.pointee.nu)
        guard count > 0 else {
            return []
        }
        return (0..<count).map { index in
            let limited = model.pointee.actuator_ctrllimited?[index] ?? false
            let lower = limited ? Double(model.pointee.actuator_ctrlrange[index * 2]) : nil
            let upper = limited ? Double(model.pointee.actuator_ctrlrange[index * 2 + 1]) : nil
            return MuJoCoActuatorSummary(
                index: index,
                name: name(for: mjOBJ_ACTUATOR, id: index),
                isControlLimited: limited,
                minimumControl: lower,
                maximumControl: upper
            )
        }
    }

    /// Sensor metadata in MuJoCo sensor order.
    public func sensorSummaries() -> [MuJoCoSensorSummary] {
        let count = Int(model.pointee.nsensor)
        guard count > 0 else {
            return []
        }
        return (0..<count).map { index in
            MuJoCoSensorSummary(
                index: index,
                name: name(for: mjOBJ_SENSOR, id: index),
                address: Int(model.pointee.sensor_adr[index]),
                dimension: Int(model.pointee.sensor_dim[index])
            )
        }
    }

    private static func vector(_ pointer: UnsafeMutablePointer<mjtNum>?, count: Int) -> [Double] {
        guard let pointer, count > 0 else {
            return []
        }
        return UnsafeBufferPointer(start: pointer, count: count).map { Double($0) }
    }

    private func transformedControls(_ controls: [Double], mode: MuJoCoActionMode) -> [Double] {
        let actuators = actuatorSummaries()
        return zip(controls, actuators).map { value, actuator in
            guard
                actuator.isControlLimited,
                let minimum = actuator.minimumControl,
                let maximum = actuator.maximumControl
            else {
                return value
            }
            switch mode {
            case .direct:
                return value
            case .clipped:
                return min(max(value, minimum), maximum)
            case .normalizedToControlRange:
                let normalized = min(max(value, -1), 1)
                return minimum + ((normalized + 1) * 0.5 * (maximum - minimum))
            }
        }
    }

    private func contactSummaries() -> [MuJoCoContact] {
        let count = Int(data.pointee.ncon)
        guard count > 0, let contacts = data.pointee.contact else {
            return []
        }
        return (0..<count).map { index in
            let contact = contacts[index]
            var force = Array<mjtNum>(repeating: 0, count: 6)
            force.withUnsafeMutableBufferPointer { buffer in
                mj_contactForce(model, data, Int32(index), buffer.baseAddress)
            }
            return MuJoCoContact(
                geom1: Int(contact.geom1),
                geom2: Int(contact.geom2),
                geom1Name: name(for: mjOBJ_GEOM, id: Int(contact.geom1)),
                geom2Name: name(for: mjOBJ_GEOM, id: Int(contact.geom2)),
                distance: Double(contact.dist),
                position: [
                    Double(contact.pos.0),
                    Double(contact.pos.1),
                    Double(contact.pos.2),
                ],
                normal: [
                    Double(contact.frame.0),
                    Double(contact.frame.1),
                    Double(contact.frame.2),
                ],
                force: force.map { Double($0) }
            )
        }
    }

    private func name(for objectType: mjtObj, id: Int) -> String? {
        guard id >= 0, let pointer = mj_id2name(model, Int32(objectType.rawValue), Int32(id)) else {
            return nil
        }
        return String(cString: pointer)
    }
}

enum MuJoCoRuntime {
    static var versionString: String {
        String(cString: mj_versionString())
    }
}

extension MuJoCoStateSignature {
    var rawMuJoCoSignature: Int32 {
        switch self {
        case .physics:
            return Int32(mjSTATE_PHYSICS.rawValue)
        case .fullPhysics:
            return Int32(mjSTATE_FULLPHYSICS.rawValue)
        case .user:
            return Int32(mjSTATE_USER.rawValue)
        case .integration:
            return Int32(mjSTATE_INTEGRATION.rawValue)
        }
    }
}
#endif
