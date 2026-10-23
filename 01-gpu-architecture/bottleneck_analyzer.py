
# Chapter 1 — GPU Architecture
# Exercise 5: Integrated GPU Bottleneck Analyzer
#
# Simplified analytical model using approximate AMD MI300X
# BF16 matrix-compute specifications.
#
# All kernel measurements below are HYPOTHETICAL.
# This is not an actual ROCm profiling tool.

PEAK_COMPUTE_TFLOPS = 1300.0
PEAK_HBM_TBPS = 5.3

KERNELS = [
    {
        "name": "Kernel A",
        "ai": 50.0,
        "achieved_tflops": 200.0,
        "achieved_tbps": 4.0,
        "occupancy_percent": 85.0,
        "registers_per_thread": 32,
    },
    {
        "name": "Kernel B",
        "ai": 300.0,
        "achieved_tflops": 650.0,
        "achieved_tbps": 650.0 / 300.0,
        "occupancy_percent": 35.0,
        "registers_per_thread": 96,
    },
    {
        "name": "Kernel C",
        "ai": 100.0,
        "achieved_tflops": 400.0,
        "achieved_tbps": 4.0,
        "occupancy_percent": 30.0,
        "registers_per_thread": 80,
    },
]


def analyze_kernel(kernel):
    """Calculate roofline and utilization metrics."""

    name = kernel["name"]
    ai = kernel["ai"]
    achieved_compute = kernel["achieved_tflops"]
    achieved_bandwidth = kernel["achieved_tbps"]

    ridge_point = PEAK_COMPUTE_TFLOPS / PEAK_HBM_TBPS

    bandwidth_ceiling = ai * PEAK_HBM_TBPS

    roofline_ceiling = min(
        PEAK_COMPUTE_TFLOPS,
        bandwidth_ceiling,
    )

    if ai < ridge_point:
        roofline_region = "Bandwidth-limited region"
    elif ai > ridge_point:
        roofline_region = "Compute-limited region"
    else:
        roofline_region = "At ridge point"

    compute_utilization = (
        achieved_compute / PEAK_COMPUTE_TFLOPS
    ) * 100

    bandwidth_utilization = (
        achieved_bandwidth / PEAK_HBM_TBPS
    ) * 100

    roofline_efficiency = (
        achieved_compute / roofline_ceiling
    ) * 100

    performance_gap = roofline_ceiling - achieved_compute

    implied_compute = ai * achieved_bandwidth

    print("\n" + "=" * 60)
    print(f"GPU BOTTLENECK ANALYSIS — {name}")
    print("=" * 60)

    print(f"Arithmetic intensity:        {ai:.2f} FLOPs/byte")
    print(f"Ridge point:                 {ridge_point:.2f} FLOPs/byte")
    print(f"Roofline region:             {roofline_region}")

    print("-" * 60)

    print(f"Achieved compute:            {achieved_compute:.2f} TFLOPS")
    print(f"Achieved HBM bandwidth:      {achieved_bandwidth:.2f} TB/s")
    print(f"Occupancy:                   {kernel['occupancy_percent']:.2f}%")
    print(f"Registers per thread:        {kernel['registers_per_thread']}")

    print("-" * 60)

    print(f"Bandwidth compute ceiling:   {bandwidth_ceiling:.2f} TFLOPS")
    print(f"Peak compute ceiling:        {PEAK_COMPUTE_TFLOPS:.2f} TFLOPS")
    print(f"Final roofline ceiling:      {roofline_ceiling:.2f} TFLOPS")

    print("-" * 60)

    print(f"Compute utilization:         {compute_utilization:.2f}%")
    print(f"HBM bandwidth utilization:   {bandwidth_utilization:.2f}%")
    print(f"Roofline efficiency:         {roofline_efficiency:.2f}%")
    print(f"Gap to roofline:             {performance_gap:.2f} TFLOPS")

    print("-" * 60)

    # Check whether measured throughput and traffic agree
    # with the supplied arithmetic intensity.
    difference = abs(achieved_compute - implied_compute)
    tolerance = max(1e-9, achieved_compute * 0.01)

    if difference <= tolerance:
        print("Measurement consistency:     PASS")
    else:
        print("Measurement consistency:     CHECK INPUTS")
        print(f"AI x bandwidth implies:      {implied_compute:.2f} TFLOPS")

    print("=" * 60)


def main():
    print("CHAPTER 1 — INTEGRATED GPU BOTTLENECK ANALYZER")
    print("Approximate MI300X BF16 matrix-compute roofline")
    print("Kernel profiles are hypothetical, not measured.")

    for kernel in KERNELS:
        analyze_kernel(kernel)


if __name__ == "__main__":
    main()
