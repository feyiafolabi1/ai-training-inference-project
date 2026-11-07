
# Chapter 2 — GPU Profiling and Roofline Analysis
# Exercise 3: Profiling Diagnostics
#
# All GPU profiles and performance counters are hypothetical.
# This script generates diagnostic hypotheses, not definitive
# explanations of actual GPU bottlenecks.

from dataclasses import dataclass


# Simplified MI300X BF16 hardware model
PEAK_TFLOPS = 1300.0
PEAK_HBM_TBPS = 5.3


@dataclass
class KernelProfile:
    name: str

    useful_tflops: float
    arithmetic_intensity: float
    hbm_bandwidth_tbps: float

    occupancy_pct: float
    matrix_activity_pct: float
    memory_stalls_pct: float
    cache_hit_pct: float

    outstanding_requests: int
    scratch_traffic_gb: float


def validate_profile(profile):
    """Check basic ranges and roofline consistency."""

    if profile.useful_tflops <= 0:
        raise ValueError("Throughput must be positive.")

    if profile.arithmetic_intensity <= 0:
        raise ValueError("Arithmetic intensity must be positive.")

    if profile.hbm_bandwidth_tbps < 0:
        raise ValueError("Bandwidth cannot be negative.")

    for name in (
        "occupancy_pct",
        "matrix_activity_pct",
        "memory_stalls_pct",
        "cache_hit_pct",
    ):
        value = getattr(profile, name)

        if not 0 <= value <= 100:
            raise ValueError(
                f"{name} must be between 0 and 100."
            )

    if profile.outstanding_requests < 0:
        raise ValueError(
            "Outstanding requests cannot be negative."
        )

    if profile.scratch_traffic_gb < 0:
        raise ValueError("Scratch traffic cannot be negative.")

    if profile.hbm_bandwidth_tbps > PEAK_HBM_TBPS:
        print(
            "WARNING: Reported HBM bandwidth exceeds "
            "the simplified hardware peak."
        )

    expected_tflops = (
        profile.arithmetic_intensity
        * profile.hbm_bandwidth_tbps
    )

    relative_error = abs(
        profile.useful_tflops - expected_tflops
    ) / profile.useful_tflops

    if relative_error > 0.01:
        print(
            "WARNING: Throughput, arithmetic intensity, "
            "and HBM bandwidth are not consistent."
        )
        print(
            "Check whether FLOP counts, memory traffic, "
            "and measurement intervals use matching definitions."
        )

    return relative_error


def calculate_roofline(profile):
    """Calculate theoretical performance limits."""

    ridge = PEAK_TFLOPS / PEAK_HBM_TBPS

    ceiling = min(
        PEAK_TFLOPS,
        profile.arithmetic_intensity * PEAK_HBM_TBPS
    )

    efficiency = (
        profile.useful_tflops / ceiling
    ) * 100

    bandwidth_utilization = (
        profile.hbm_bandwidth_tbps / PEAK_HBM_TBPS
    ) * 100

    if profile.arithmetic_intensity < ridge:
        region = "Bandwidth-side"
    elif profile.arithmetic_intensity > ridge:
        region = "Compute-side"
    else:
        region = "At ridge point"

    return {
        "ridge": ridge,
        "ceiling": ceiling,
        "efficiency": efficiency,
        "bandwidth_utilization": bandwidth_utilization,
        "region": region,
    }


def generate_hypotheses(profile, metrics):
    """Generate possible causes and experiments.

    Thresholds are illustrative teaching heuristics,
    NOT validated AMD hardware profiling rules.
    """

    findings = []

    if (
        metrics["bandwidth_utilization"] >= 85
        and profile.memory_stalls_pct >= 50
    ):
        findings.append((
            "POSSIBLE HBM BANDWIDTH SATURATION",
            "HBM utilization is near peak and memory stalls are high.",
            "Test improved data reuse, tiling, or reduced HBM traffic."
        ))

    if (
        metrics["bandwidth_utilization"] < 50
        and profile.memory_stalls_pct >= 50
        and profile.outstanding_requests < 50
    ):
        findings.append((
            "POSSIBLE MEMORY-LATENCY / MLP LIMITATION",
            "Memory stalls are high, HBM bandwidth is low, "
            "and few requests are outstanding.",
            "Test more independent loads or improved cache locality."
        ))

    if (
        profile.occupancy_pct < 40
        and profile.scratch_traffic_gb >= 1
    ):
        findings.append((
            "POSSIBLE REGISTER PRESSURE / SPILLING",
            "Occupancy is low and significant scratch "
            "traffic is present.",
            "Inspect register allocation and test reduced "
            "live ranges or smaller tiles."
        ))

    if (
        profile.occupancy_pct >= 70
        and profile.matrix_activity_pct < 40
    ):
        findings.append((
            "HIGH OCCUPANCY, LOW MATRIX ACTIVITY",
            "Many wavefronts may be resident without "
            "keeping matrix units active.",
            "Inspect instruction dependencies, memory stalls, "
            "and whether this kernel performs matrix operations."
        ))

    if metrics["efficiency"] < 50:
        findings.append((
            "LARGE ROOFLINE PERFORMANCE GAP",
            "Achieved throughput is less than half "
            "of the simplified roofline ceiling.",
            "Investigate hardware counters and run controlled "
            "experiments before assigning a bottleneck."
        ))

    return findings


def analyze_kernel(profile):
    print("\n" + "=" * 66)
    print(f"GPU PROFILE: {profile.name}")
    print("=" * 66)

    error = validate_profile(profile)
    metrics = calculate_roofline(profile)

    print("\nPERFORMANCE METRICS")
    print("-" * 66)

    print(
        f"Achieved throughput:   "
        f"{profile.useful_tflops:.2f} TFLOPS"
    )
    print(
        f"Arithmetic intensity:  "
        f"{profile.arithmetic_intensity:.2f} FLOPs/byte"
    )
    print(
        f"HBM bandwidth:         "
        f"{profile.hbm_bandwidth_tbps:.3f} TB/s"
    )
    print(
        f"HBM utilization:       "
        f"{metrics['bandwidth_utilization']:.1f}%"
    )
    print(
        f"Roofline ceiling:      "
        f"{metrics['ceiling']:.2f} TFLOPS"
    )
    print(
        f"Roofline efficiency:   "
        f"{metrics['efficiency']:.1f}%"
    )
    print(
        f"Roofline region:       "
        f"{metrics['region']}"
    )
    print(
        f"FLOP/byte consistency error: {error * 100:.3f}%"
    )

    print("\nHARDWARE COUNTERS")
    print("-" * 66)

    print(f"Occupancy:             {profile.occupancy_pct}%")
    print(f"Matrix activity:       {profile.matrix_activity_pct}%")
    print(f"Memory stalls:         {profile.memory_stalls_pct}%")
    print(f"Cache hit rate:        {profile.cache_hit_pct}%")
    print(f"Outstanding requests:  {profile.outstanding_requests}")
    print(f"Scratch traffic:       {profile.scratch_traffic_gb} GB")

    findings = generate_hypotheses(profile, metrics)

    print("\nDIAGNOSTIC HYPOTHESES")
    print("-" * 66)

    if not findings:
        print("No configured heuristic was triggered.")
        print("Additional profiling evidence is needed.")

    for index, (title, evidence, experiment) in enumerate(
        findings, start=1
    ):
        print(f"\n{index}. {title}")
        print(f"   Evidence:   {evidence}")
        print(f"   Experiment: {experiment}")

    print("\nNOTE: These are hypotheses, not proven bottlenecks.")

    return metrics, findings


def main():
    print("CHAPTER 2 — GPU PROFILING DIAGNOSTICS")
    print("All data is hypothetical.")
    print("Hardware counter thresholds are illustrative.")

    # Each profile is internally consistent:
    # useful TFLOPS = AI * achieved HBM TB/s.

    profiles = [
        KernelProfile(
            name="Kernel A — Streaming",
            useful_tflops=190.8,
            arithmetic_intensity=40,
            hbm_bandwidth_tbps=4.77,
            occupancy_pct=85,
            matrix_activity_pct=30,
            memory_stalls_pct=75,
            cache_hit_pct=20,
            outstanding_requests=350,
            scratch_traffic_gb=0.0
        ),

        KernelProfile(
            name="Kernel B — Dependent Loads",
            useful_tflops=130.0,
            arithmetic_intensity=200,
            hbm_bandwidth_tbps=0.65,
            occupancy_pct=75,
            matrix_activity_pct=25,
            memory_stalls_pct=85,
            cache_hit_pct=30,
            outstanding_requests=20,
            scratch_traffic_gb=0.0
        ),

        KernelProfile(
            name="Kernel C — Register Pressure",
            useful_tflops=600.0,
            arithmetic_intensity=400,
            hbm_bandwidth_tbps=1.50,
            occupancy_pct=25,
            matrix_activity_pct=45,
            memory_stalls_pct=35,
            cache_hit_pct=65,
            outstanding_requests=100,
            scratch_traffic_gb=4.0
        ),
    ]

    for profile in profiles:
        analyze_kernel(profile)

    print("\n" + "=" * 66)
    print("ANALYSIS COMPLETE")
    print("=" * 66)


if __name__ == "__main__":
    main()
