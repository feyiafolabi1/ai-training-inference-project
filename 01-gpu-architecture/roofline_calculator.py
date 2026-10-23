
# Chapter 1 — GPU Architecture
# Exercise 4: Arithmetic Intensity and Roofline Calculator
#
# Analytical model only — not actual GPU profiling.
#
# Assumptions:
# - Square matrix multiplication: C = A @ B
# - Read A once, read B once, write C once from/to HBM
# - No extra traffic for caching, temporary buffers, or spills
# - AMD MI300X approximate advertised BF16 matrix-compute peak
# - Decimal TFLOPS and TB/s units


MI300X_BF16_TFLOPS = 1300.0
MI300X_HBM_TBPS = 5.3

DTYPE_BYTES = {
    "BF16": 2,
    "FP32": 4,
}


def analyze_gemm(n, input_dtype="BF16", output_dtype="BF16"):
    """
    Estimate GEMM FLOPs, minimum HBM traffic,
    arithmetic intensity, and roofline upper bound.

    The compute ceiling assumes BF16 matrix operations.
    The output may be stored in BF16 or FP32.

    This model does not estimate actual execution time.
    """

    if n <= 0:
        raise ValueError("Matrix dimension must be positive.")

    if input_dtype != "BF16":
        raise ValueError(
            "This exercise uses the MI300X BF16 compute ceiling. "
            "Input matrices must be BF16."
        )

    if output_dtype not in DTYPE_BYTES:
        raise ValueError("Unsupported output dtype.")

    input_bytes = DTYPE_BYTES[input_dtype]
    output_bytes = DTYPE_BYTES[output_dtype]

    # Each output requires N multiplications and N-1 additions.
    output_elements = n * n
    flops_per_output = 2 * n - 1
    exact_flops = output_elements * flops_per_output

    # Standard approximate GEMM FLOP convention.
    approximate_flops = 2 * n**3

    # Idealized memory traffic:
    # read input A, read input B, write output C.
    bytes_a = n**2 * input_bytes
    bytes_b = n**2 * input_bytes
    bytes_c = n**2 * output_bytes

    total_bytes = bytes_a + bytes_b + bytes_c

    # Use the standard approximate GEMM convention for AI.
    arithmetic_intensity = approximate_flops / total_bytes

    ridge_point = (
        MI300X_BF16_TFLOPS / MI300X_HBM_TBPS
    )

    bandwidth_ceiling = (
        arithmetic_intensity * MI300X_HBM_TBPS
    )

    roofline_ceiling = min(
        MI300X_BF16_TFLOPS,
        bandwidth_ceiling,
    )

    if arithmetic_intensity < ridge_point:
        roofline_region = "Bandwidth-limited"
    elif arithmetic_intensity > ridge_point:
        roofline_region = "Compute-limited"
    else:
        roofline_region = "At ridge point"

    print("\n" + "=" * 58)
    print(f"GEMM ROOFLINE ANALYSIS — {n} x {n}")
    print("=" * 58)

    print(f"Input dtype:             {input_dtype}")
    print(f"Output dtype:            {output_dtype}")
    print(f"Output elements:         {output_elements:,}")
    print(f"FLOPs per output:        {flops_per_output:,}")
    print(f"Exact arithmetic FLOPs:  {exact_flops:,}")
    print(f"Approximate GEMM FLOPs:  {approximate_flops:,}")

    print("-" * 58)

    print(f"Read A:                  {bytes_a:,} bytes")
    print(f"Read B:                  {bytes_b:,} bytes")
    print(f"Write C:                 {bytes_c:,} bytes")
    print(f"Total ideal HBM traffic: {total_bytes:,} bytes")

    print("-" * 58)

    print(
        f"Arithmetic intensity:    "
        f"{arithmetic_intensity:.2f} FLOPs/byte"
    )

    print(
        f"Ridge point:             "
        f"{ridge_point:.2f} FLOPs/byte"
    )

    print(
        f"Bandwidth ceiling:       "
        f"{bandwidth_ceiling:.2f} TFLOPS"
    )

    print(
        f"Compute ceiling:         "
        f"{MI300X_BF16_TFLOPS:.2f} TFLOPS"
    )

    print(
        f"Roofline ceiling:        "
        f"{roofline_ceiling:.2f} TFLOPS"
    )

    print(f"Roofline region:         {roofline_region}")
    print("=" * 58)

    return {
        "n": n,
        "flops": approximate_flops,
        "bytes": total_bytes,
        "arithmetic_intensity": arithmetic_intensity,
        "roofline_ceiling": roofline_ceiling,
        "roofline_region": roofline_region,
    }


def main():
    print("CHAPTER 1 — GEMM ROOFLINE CALCULATOR")
    print("Approximate AMD MI300X BF16 performance model")
    print("These are analytical bounds, not measured results.")

    # Small BF16 GEMM
    analyze_gemm(
        n=256,
        input_dtype="BF16",
        output_dtype="BF16",
    )

    # Larger BF16 GEMM
    analyze_gemm(
        n=1024,
        input_dtype="BF16",
        output_dtype="BF16",
    )

    # Same inputs, but FP32 output
    analyze_gemm(
        n=256,
        input_dtype="BF16",
        output_dtype="FP32",
    )


if __name__ == "__main__":
    main()
