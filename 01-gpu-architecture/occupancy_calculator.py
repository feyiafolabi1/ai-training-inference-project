# Chapter 1 — GPU Architecture
# Exercise 2: Simplified GPU Occupancy Calculator

# Hypothetical CU specifications
WAVEFRONT_SIZE = 64
REGISTERS_PER_CU = 65_536
LDS_PER_CU_KIB = 64
MAX_RESIDENT_WAVEFRONTS = 32


def analyze_occupancy(
    kernel_name,
    threads_per_workgroup,
    registers_per_thread,
    lds_per_workgroup_kib,
):
    """
    Calculate simplified CU occupancy constrained by:
    - Register capacity
    - LDS capacity
    - Maximum resident wavefronts

    Assumptions:
    - Workgroup size is a multiple of 64.
    - Registers are allocated per thread without rounding.
    - LDS is allocated per workgroup without rounding.
    - No additional architectural limits apply.
    - Sufficient workgroups exist to fill the CU.

    This is an analytical model, not an actual GPU profiler.
    """

    if threads_per_workgroup <= 0:
        raise ValueError("Threads per workgroup must be positive.")

    if threads_per_workgroup % WAVEFRONT_SIZE != 0:
        raise ValueError(
            "Threads per workgroup must be a multiple of 64."
        )

    if registers_per_thread <= 0:
        raise ValueError("Registers per thread must be positive.")

    if lds_per_workgroup_kib < 0:
        raise ValueError("LDS allocation cannot be negative.")

    wavefronts_per_workgroup = (
        threads_per_workgroup // WAVEFRONT_SIZE
    )

    registers_per_workgroup = (
        threads_per_workgroup * registers_per_thread
    )

    max_workgroups_registers = (
        REGISTERS_PER_CU // registers_per_workgroup
    )

    if lds_per_workgroup_kib == 0:
        max_workgroups_lds = float("inf")
    else:
        max_workgroups_lds = (
            LDS_PER_CU_KIB // lds_per_workgroup_kib
        )

    max_workgroups_wavefronts = (
        MAX_RESIDENT_WAVEFRONTS // wavefronts_per_workgroup
    )

    resident_workgroups = int(
        min(
            max_workgroups_registers,
            max_workgroups_lds,
            max_workgroups_wavefronts,
        )
    )

    resident_wavefronts = (
        resident_workgroups * wavefronts_per_workgroup
    )

    occupancy = (
        resident_wavefronts / MAX_RESIDENT_WAVEFRONTS
    ) * 100

    limiting_resources = []

    if resident_workgroups == max_workgroups_registers:
        limiting_resources.append("Registers")

    if resident_workgroups == max_workgroups_lds:
        limiting_resources.append("LDS")

    if resident_workgroups == max_workgroups_wavefronts:
        limiting_resources.append("Wavefront capacity")

    print("\n" + "=" * 55)
    print(f"OCCUPANCY ANALYSIS: {kernel_name}")
    print("=" * 55)

    print(f"Threads per workgroup:      {threads_per_workgroup}")
    print(f"Wavefronts per workgroup:   {wavefronts_per_workgroup}")
    print(f"Registers per thread:      {registers_per_thread}")
    print(f"Registers per workgroup:   {registers_per_workgroup:,}")
    print(f"LDS per workgroup:         {lds_per_workgroup_kib} KiB")

    print("-" * 55)

    print(f"WG limit from registers:    {max_workgroups_registers}")
    print(f"WG limit from LDS:          {max_workgroups_lds}")
    print(f"WG limit from wavefronts:   {max_workgroups_wavefronts}")

    print("-" * 55)

    print(f"Resident workgroups:       {resident_workgroups}")
    print(f"Resident wavefronts:       {resident_wavefronts}")
    print(f"Occupancy:                 {occupancy:.2f}%")
    print(f"Limiting resource(s):      {', '.join(limiting_resources)}")

    print("=" * 55)


def main():

    analyze_occupancy(
        kernel_name="Kernel A",
        threads_per_workgroup=256,
        registers_per_thread=32,
        lds_per_workgroup_kib=8,
    )

    analyze_occupancy(
        kernel_name="Kernel B",
        threads_per_workgroup=256,
        registers_per_thread=64,
        lds_per_workgroup_kib=24,
    )

    analyze_occupancy(
        kernel_name="Kernel C",
        threads_per_workgroup=256,
        registers_per_thread=128,
        lds_per_workgroup_kib=4,
    )


if __name__ == "__main__":
    main()