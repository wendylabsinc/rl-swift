import Foundation

/// One scalar metric at a training step.
public struct MetricSeriesPoint: Sendable, Equatable, Codable {
    /// Step associated with the value.
    public let step: Int

    /// Scalar metric value.
    public let value: Double

    /// Creates a metric point.
    public init(step: Int, value: Double) throws {
        guard step >= 0 else {
            throw RLSwiftError.invalidSampleCount(step)
        }
        self.step = step
        self.value = value
    }
}

/// Named series of scalar training metrics.
public struct TrainingMetricSeries: Sendable, Equatable, Codable {
    /// Series name.
    public let name: String

    /// Ordered metric points.
    public let points: [MetricSeriesPoint]

    /// Creates a metric series.
    public init(name: String, points: [MetricSeriesPoint]) throws {
        guard !name.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "metric.name")
        }
        self.name = name
        self.points = points
    }

    /// Most recent metric value.
    public var latestValue: Double? {
        points.last?.value
    }

    /// ASCII sparkline for terminals and CI logs.
    public func sparkline(width: Int = 24) throws -> String {
        guard width > 0 else {
            throw RLSwiftError.invalidCapacity(width)
        }
        guard !points.isEmpty else {
            return ""
        }
        let values = points.suffix(width).map(\.value)
        let minimum = values.min()!
        let maximum = values.max()!
        let marks = Array(" .:-=+*#%@")
        if minimum == maximum {
            return String(repeating: String(marks[marks.count / 2]), count: values.count)
        }
        return values.map { value in
            let normalized = (value - minimum) / (maximum - minimum)
            let index = Int((normalized * Double(marks.count - 1)).rounded())
            return String(marks[index])
        }
        .joined()
    }
}

/// Lightweight dashboard snapshot suitable for local CLI output.
public struct TrainingDashboardSnapshot: Sendable, Equatable, Codable {
    /// Dashboard title.
    public let title: String

    /// Metric series included in the dashboard.
    public let series: [TrainingMetricSeries]

    /// Creates a dashboard snapshot.
    public init(title: String, series: [TrainingMetricSeries]) throws {
        guard !title.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "dashboard.title")
        }
        guard !series.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        self.title = title
        self.series = series
    }

    /// Renders a deterministic text table for logs or terminals.
    public func renderTextTable() throws -> String {
        var lines = [title, "metric latest sparkline"]
        for item in series {
            let latest = item.latestValue ?? 0
            lines.append("\(item.name) \(format(latest)) \(try item.sparkline())")
        }
        return lines.joined(separator: "\n")
    }

    private func format(_ value: Double) -> String {
        String(format: "%.4f", value)
    }
}
