// Move-only scratch storage for hot vector transforms that need exactly one
// owned output array without intermediate map closures or accidental copies.
@usableFromInline
struct VectorScratch: ~Copyable {
    @usableFromInline
    var values: [Double]

    @usableFromInline
    init(count: Int) {
        values = Array(repeating: 0, count: count)
    }

    @usableFromInline
    mutating func set(_ value: Double, at index: Int) {
        values[index] = value
    }

    @usableFromInline
    consuming func finish() -> [Double] {
        values
    }
}
