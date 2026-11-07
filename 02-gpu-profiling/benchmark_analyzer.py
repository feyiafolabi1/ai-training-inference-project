
# Chapter 2 — GPU Profiling and Roofline Analysis
# Exercise 1: Benchmark Statistics Analyzer
#
# Standard-library Python only.
# All measurements are hypothetical, NOT actual GPU timings.

import math
import statistics


def nearest_rank_percentile(values, percentile):
    """
    Compute a percentile using the nearest-rank method.

    Note: This differs from interpolated percentile methods.
    """
    if not values:
        raise ValueError("At least one measurement is required.")

    if not 0 < percentile <= 100:
        raise ValueError("Percentile must be > 0 and <= 100.")

    ordered = sorted(values)
    rank = math.ceil(percentile / 100 * len(ordered))

    return ordered[rank - 1]


def analyze_benchmark(name, measurements):
    """
    Analyze execution times in milliseconds.

    Assumes:
    - All runs execute the same workload.
    - Warmup has already been completed.
    - Measurements represent completed execution.
    """

    if not measurements:
        raise ValueError("Measurements cannot be empty.")

    if any(t <= 0 for t in measurements):
        raise ValueError("Execution times must be positive.")

    results = {
        "mean": statistics.mean(measurements),
        "median": statistics.median(measurements),
        "min": min(measurements),
        "max": max(measurements),
        "p90": nearest_rank_percentile(measurements, 90),
        "p95": nearest_rank_percentile(measurements, 95),
        "p99": nearest_rank_percentile(measurements, 99),
        "count": len(measurements),
    }

    print("\n" + "=" * 52)
    print(f"BENCHMARK ANALYSIS — {name}")
    print("=" * 52)

    print(f"Samples:          {results['count']}")
    print(f"Mean latency:     {results['mean']:.3f} ms")
    print(f"Median latency:   {results['median']:.3f} ms")
    print(f"Minimum latency:  {results['min']:.3f} ms")
    print(f"Maximum latency:  {results['max']:.3f} ms")
    print(f"P90 latency:      {results['p90']:.3f} ms")
    print(f"P95 latency:      {results['p95']:.3f} ms")
    print(f"P99 latency:      {results['p99']:.3f} ms")

    print("=" * 52)

    return results


def compare_benchmarks(baseline, optimized):
    """
    Compare two implementations using median execution time.

    Assumes both implementations perform equivalent work
    and satisfy the same correctness requirements.
    """

    baseline_time = baseline["median"]
    optimized_time = optimized["median"]

    speedup = baseline_time / optimized_time

    latency_reduction = (
        (baseline_time - optimized_time) / baseline_time
    ) * 100

    print("\n" + "=" * 52)
    print("BENCHMARK COMPARISON")
    print("=" * 52)

    print(f"Baseline median:     {baseline_time:.3f} ms")
    print(f"Optimized median:    {optimized_time:.3f} ms")
    print(f"Speedup:             {speedup:.3f}x")
    print(f"Latency reduction:   {latency_reduction:.2f}%")

    if optimized_time < baseline_time:
        print("Median result:       Optimized is faster")
    elif optimized_time > baseline_time:
        print("Median result:       Baseline is faster")
    else:
        print("Median result:       Equal latency")

    print("=" * 52)


def main():

    print("CHAPTER 2 — BENCHMARK STATISTICS ANALYZER")
    print("Hypothetical measurements in milliseconds")
    print("Percentiles use the nearest-rank convention.")

    # These reproduce our earlier benchmarking example.
    kernel_a_times = [
        3.0, 3.1, 3.0, 3.2, 3.1
    ]

    kernel_b_times = [
        2.8, 2.9, 2.8, 2.9, 7.5
    ]

    kernel_a = analyze_benchmark(
        "Kernel A (Baseline)",
        kernel_a_times
    )

    kernel_b = analyze_benchmark(
        "Kernel B (Optimized)",
        kernel_b_times
    )

    compare_benchmarks(kernel_a, kernel_b)

    # Additional sample: ten request latencies.
    request_latencies = [
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 50
    ]

    analyze_benchmark(
        "Inference Request Latencies",
        request_latencies
    )

    print("\nIMPORTANT NOTES:")
    print("- These measurements are hypothetical.")
    print("- Correctness and fair workload comparison come first.")
    print("- Short samples do not provide reliable P99 estimates.")
    print("- Outliers require investigation, not automatic deletion.")
    print("- Real GPU timings require proper synchronization/events.")


if __name__ == "__main__":
    main()
