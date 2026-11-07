
# Chapter 2 — GPU Profiling and Roofline Analysis
# Exercise 4: Application Optimization Comparison
#
# All kernel measurements and proposed improvements are hypothetical.
# This is an analytical model, not a real AMD GPU profiler.

from dataclasses import dataclass


@dataclass
class Kernel:
    name: str
    baseline_ms: float
    optimized_ms: float
    roofline_efficiency_pct: float


def speedup(baseline_ms, optimized_ms):
    """Return speedup for equivalent useful work."""
    return baseline_ms / optimized_ms


def latency_reduction(baseline_ms, optimized_ms):
    """Return percentage reduction in execution time."""
    return (
        (baseline_ms - optimized_ms) / baseline_ms
    ) * 100


def validate_kernels(kernels):
    """Check input values before analyzing results."""

    if not kernels:
        raise ValueError("At least one kernel is required.")

    names = set()

    for kernel in kernels:
        if kernel.name in names:
            raise ValueError("Kernel names must be unique.")

        names.add(kernel.name)

        if kernel.baseline_ms <= 0:
            raise ValueError("Baseline latency must be positive.")

        if kernel.optimized_ms <= 0:
            raise ValueError("Optimized latency must be positive.")

        if not 0 <= kernel.roofline_efficiency_pct <= 100:
            raise ValueError(
                "Roofline efficiency must be between 0 and 100."
            )


def analyze_application(kernels):
    """Compare original and optimized application execution."""

    validate_kernels(kernels)

    total_baseline = sum(
        kernel.baseline_ms for kernel in kernels
    )

    total_optimized = sum(
        kernel.optimized_ms for kernel in kernels
    )

    print("\n" + "=" * 72)
    print("APPLICATION PERFORMANCE COMPARISON")
    print("=" * 72)

    for kernel in kernels:
        kernel_speedup = speedup(
            kernel.baseline_ms,
            kernel.optimized_ms
        )

        reduction = latency_reduction(
            kernel.baseline_ms,
            kernel.optimized_ms
        )

        baseline_share = (
            kernel.baseline_ms / total_baseline
        ) * 100

        optimized_share = (
            kernel.optimized_ms / total_optimized
        ) * 100

        print(f"\n{kernel.name}")
        print(f"  Baseline latency:       {kernel.baseline_ms:.3f} ms")
        print(f"  Optimized latency:      {kernel.optimized_ms:.3f} ms")
        print(f"  Kernel speedup:         {kernel_speedup:.3f}x")
        print(f"  Latency reduction:      {reduction:.2f}%")
        print(f"  Baseline runtime share: {baseline_share:.2f}%")
        print(f"  New runtime share:      {optimized_share:.2f}%")
        print(
            f"  Baseline roofline eff.: "
            f"{kernel.roofline_efficiency_pct:.1f}%"
        )

    application_speedup = speedup(
        total_baseline,
        total_optimized
    )

    print("\n" + "=" * 72)
    print("APPLICATION SUMMARY")
    print("=" * 72)

    print(f"Original runtime:     {total_baseline:.3f} ms")
    print(f"Optimized runtime:    {total_optimized:.3f} ms")
    print(
        f"Total time saved:     "
        f"{total_baseline - total_optimized:.3f} ms"
    )
    print(f"Application speedup:  {application_speedup:.3f}x")
    print(
        f"Latency reduction:    "
        f"{latency_reduction(total_baseline, total_optimized):.2f}%"
    )

    return total_baseline, total_optimized


def rank_optimization_opportunities(kernels, proposals):
    """
    Evaluate one proposed optimization at a time.

    proposals maps kernel names to EXPECTED speedup factors.

    Assumes:
    - Equivalent useful computation.
    - No changes to other kernel runtimes.
    - No execution overlap or scheduling changes.
    """

    baseline_total = sum(
        kernel.baseline_ms for kernel in kernels
    )

    results = []

    for kernel in kernels:
        if kernel.name not in proposals:
            continue

        expected_speedup = proposals[kernel.name]

        if expected_speedup <= 0:
            raise ValueError("Speedup factors must be positive.")

        new_kernel_time = (
            kernel.baseline_ms / expected_speedup
        )

        time_saved = kernel.baseline_ms - new_kernel_time

        new_application_time = baseline_total - time_saved

        app_speedup = (
            baseline_total / new_application_time
        )

        results.append({
            "name": kernel.name,
            "kernel_speedup": expected_speedup,
            "time_saved": time_saved,
            "new_app_time": new_application_time,
            "app_speedup": app_speedup,
        })

    results.sort(
        key=lambda item: item["time_saved"],
        reverse=True
    )

    print("\n" + "=" * 72)
    print("PROPOSED OPTIMIZATION RANKINGS")
    print("=" * 72)

    for rank, result in enumerate(results, start=1):
        print(f"\n{rank}. {result['name']}")
        print(
            f"   Expected kernel speedup: "
            f"{result['kernel_speedup']:.2f}x"
        )
        print(
            f"   Expected time saved:     "
            f"{result['time_saved']:.3f} ms"
        )
        print(
            f"   New application time:    "
            f"{result['new_app_time']:.3f} ms"
        )
        print(
            f"   Application speedup:     "
            f"{result['app_speedup']:.3f}x"
        )

    if results:
        print(
            "\nLargest predicted application-level benefit: "
            + results[0]["name"]
        )

    return results


def main():
    print("CHAPTER 2 — GPU OPTIMIZATION COMPARISON")
    print("All performance data is hypothetical.")

    # Observed baseline and one hypothetical completed optimization.
    # Attention runtime was reduced by 40%, from 4.0 to 2.4 ms.

    kernels = [
        Kernel(
            name="GEMM",
            baseline_ms=5.0,
            optimized_ms=5.0,
            roofline_efficiency_pct=90.0
        ),
        Kernel(
            name="Attention",
            baseline_ms=4.0,
            optimized_ms=2.4,
            roofline_efficiency_pct=35.0
        ),
        Kernel(
            name="RMSNorm",
            baseline_ms=1.0,
            optimized_ms=1.0,
            roofline_efficiency_pct=55.0
        ),
    ]

    analyze_application(kernels)

    # Separate what-if analysis against the ORIGINAL baseline.
    # These are predictions, not additional applied optimizations.

    proposed_speedups = {
        "GEMM": 1.10,
        "Attention": 1.20,
        "RMSNorm": 2.00,
    }

    rank_optimization_opportunities(
        kernels,
        proposed_speedups
    )

    print("\nIMPORTANT NOTES")
    print("- Optimize complete application performance.")
    print("- Higher roofline efficiency does not guarantee speedup.")
    print("- Predictions require experimental validation.")
    print("- Always verify kernel output correctness.")
    print("- Reprofile after implementing each optimization.")


if __name__ == "__main__":
    main()
