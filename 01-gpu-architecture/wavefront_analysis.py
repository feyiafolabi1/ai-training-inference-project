import math


# Simplified AMD CDNA execution model
WAVEFRONT_SIZE = 64


def analyze_workload(num_elements, threads_per_workgroup):
    """
    Analyze how a 1D workload maps to GPU workgroups and wavefronts.

    Assumptions:
    - One thread processes one element.
    - Wavefront size is 64.
    - Workgroup size is a multiple of 64.
    - This is an analytical model, not actual GPU profiling.
    """

    if num_elements <= 0:
        raise ValueError("Number of elements must be positive.")

    if threads_per_workgroup <= 0:
        raise ValueError("Threads per workgroup must be positive.")

    if threads_per_workgroup % WAVEFRONT_SIZE != 0:
        raise ValueError("Workgroup size must be a multiple of 64.")

    workgroups = math.ceil(
        num_elements / threads_per_workgroup
    )

    wavefronts_per_workgroup = (
        threads_per_workgroup // WAVEFRONT_SIZE
    )

    total_wavefronts = (
        workgroups * wavefronts_per_workgroup
    )

    launched_threads = (
        workgroups * threads_per_workgroup
    )

    inactive_threads = (
        launched_threads - num_elements
    )

    utilization = (
        num_elements / launched_threads
    ) * 100

    print("\n" + "=" * 50)
    print("GPU WAVEFRONT ANALYSIS")
    print("=" * 50)

    print(f"Elements to process:     {num_elements:,}")
    print(f"Threads per workgroup:   {threads_per_workgroup}")
    print(f"Wavefront size:          {WAVEFRONT_SIZE}")
    print(f"Number of workgroups:    {workgroups:,}")
    print(f"Wavefronts per WG:       {wavefronts_per_workgroup}")
    print(f"Total wavefronts:        {total_wavefronts:,}")
    print(f"Total launched threads:  {launched_threads:,}")
    print(f"Inactive threads:        {inactive_threads:,}")
    print(f"Useful thread fraction:  {utilization:.2f}%")

    print("=" * 50)


def main():
    analyze_workload(
        num_elements=1000,
        threads_per_workgroup=256
    )

    analyze_workload(
        num_elements=1025,
        threads_per_workgroup=256
    )

    analyze_workload(
        num_elements=1_000_000,
        threads_per_workgroup=64
    )

    analyze_workload(
        num_elements=1_000_000,
        threads_per_workgroup=256
    )


if __name__ == "__main__":
    main()