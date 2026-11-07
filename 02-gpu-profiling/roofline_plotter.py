
# Chapter 2 — GPU Profiling and Roofline Analysis
# Exercise 2: Roofline Plotter with Optimization Comparison
#
# Hypothetical MI300X BF16 specs.
# Kernel measurements are illustrative, not real profiling data.

import numpy as np
import matplotlib.pyplot as plt


PEAK_TFLOPS = 1300.0
PEAK_BANDWIDTH_TBPS = 5.3

KERNELS = {
    "Kernel A": {"ai": 50, "achieved_tflops": 150},
    "Kernel B": {"ai": 200, "achieved_tflops": 500},
    "Kernel C": {"ai": 400, "achieved_tflops": 750},
    "Kernel B Optimized": {
        "ai": 350,
        "achieved_tflops": 800
    },
}


def roofline_performance(ai):
    return np.minimum(
        PEAK_TFLOPS,
        ai * PEAK_BANDWIDTH_TBPS
    )


def get_metrics(kernel):
    ai = kernel["ai"]
    achieved = kernel["achieved_tflops"]

    ceiling = float(roofline_performance(ai))
    efficiency = achieved / ceiling * 100

    ridge = PEAK_TFLOPS / PEAK_BANDWIDTH_TBPS

    if ai < ridge:
        region = "Bandwidth-side"
    elif ai > ridge:
        region = "Compute-side"
    else:
        region = "At ridge point"

    return ceiling, efficiency, region


def main():
    ridge = PEAK_TFLOPS / PEAK_BANDWIDTH_TBPS

    ai_values = np.logspace(0, 4, 500)
    roofline_values = roofline_performance(ai_values)

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.plot(
        ai_values,
        roofline_values,
        linewidth=2.5,
        label="Theoretical BF16 Roofline"
    )

    ax.axvline(
        ridge,
        linestyle="--",
        linewidth=1.5,
        label=f"Ridge Point ({ridge:.1f} FLOPs/byte)"
    )

    print("=" * 65)
    print("CHAPTER 2 — ROOFLINE OPTIMIZATION ANALYSIS")
    print("=" * 65)

    for name, kernel in KERNELS.items():
        ai = kernel["ai"]
        achieved = kernel["achieved_tflops"]

        ceiling, efficiency, region = get_metrics(kernel)

        ax.scatter(
            ai,
            achieved,
            s=110,
            zorder=5,
            label=f"{name} ({efficiency:.1f}%)"
        )

        ax.annotate(
            name,
            (ai, achieved),
            xytext=(8, -12),
            textcoords="offset points",
            fontsize=9
        )

        print(f"\n{name}")
        print(f"  Arithmetic intensity: {ai:.1f} FLOPs/byte")
        print(f"  Achieved throughput:  {achieved:.1f} TFLOPS")
        print(f"  Roofline ceiling:     {ceiling:.1f} TFLOPS")
        print(f"  Roofline efficiency:  {efficiency:.2f}%")
        print(f"  Roofline region:      {region}")

    baseline = KERNELS["Kernel B"]
    optimized = KERNELS["Kernel B Optimized"]

    ax.annotate(
        "",
        xy=(optimized["ai"], optimized["achieved_tflops"]),
        xytext=(baseline["ai"], baseline["achieved_tflops"]),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 2,
            "linestyle": "--",
            "color": "black"
        }
    )

    speedup = (
        optimized["achieved_tflops"]
        / baseline["achieved_tflops"]
    )

    print("\n" + "=" * 65)
    print("KERNEL B OPTIMIZATION SUMMARY")
    print("=" * 65)

    print(
        f"Arithmetic intensity: "
        f"{baseline['ai']} -> {optimized['ai']} FLOPs/byte"
    )
    print(
        f"Throughput: "
        f"{baseline['achieved_tflops']} -> "
        f"{optimized['achieved_tflops']} TFLOPS"
    )
    print(f"Throughput speedup: {speedup:.3f}x")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(1, 10000)
    ax.set_ylim(1, 2000)

    ax.set_xlabel("Arithmetic Intensity (FLOPs/byte)")
    ax.set_ylabel("Achieved Throughput (TFLOPS)")
    ax.set_title("MI300X BF16 Roofline: Kernel Optimization")

    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("roofline_optimization.png", dpi=200)

    print("\nSaved plot: roofline_optimization.png")

    plt.show()


if __name__ == "__main__":
    main()
