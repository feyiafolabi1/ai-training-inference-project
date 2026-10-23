
# Chapter 1 — GPU Architecture
# Exercise 3: Memory Coalescing Simulator
#
# This is a simplified analytical model.
# It does NOT measure actual AMD GPU memory transactions.

WAVEFRONT_SIZE = 64
BYTES_PER_ELEMENT = 4


def analyze_access_pattern(stride_bytes, sector_size=32):
    """
    Simulate one wavefront of FP32 memory loads.

    Each of the 64 threads reads one 4-byte FP32 value.
    Thread i accesses address i * stride_bytes.

    Assumptions:
    - Base address is zero and sector-aligned.
    - Every distinct touched sector transfers its full size.
    - No cache hits or reuse across separate analyses.
    - Sectors are fixed-size, non-overlapping memory regions.
    - This is NOT a cycle-accurate GPU simulator.
    """

    if stride_bytes <= 0:
        raise ValueError("Stride must be positive.")

    if sector_size <= 0:
        raise ValueError("Sector size must be positive.")

    if sector_size < BYTES_PER_ELEMENT:
        raise ValueError("Sector must be at least 4 bytes.")

    # Starting byte address for each thread's FP32 load
    addresses = [
        thread_id * stride_bytes
        for thread_id in range(WAVEFRONT_SIZE)
    ]

    # Track all sectors touched by each 4-byte load.
    # This also handles loads that cross sector boundaries.
    touched_sectors = set()

    for address in addresses:
        first_sector = address // sector_size
        last_sector = (
            address + BYTES_PER_ELEMENT - 1
        ) // sector_size

        for sector in range(first_sector, last_sector + 1):
            touched_sectors.add(sector)

    useful_bytes = WAVEFRONT_SIZE * BYTES_PER_ELEMENT
    distinct_sectors = len(touched_sectors)
    transferred_bytes = distinct_sectors * sector_size

    efficiency = (
        useful_bytes / transferred_bytes
    ) * 100

    return {
        "stride": stride_bytes,
        "sector_size": sector_size,
        "useful_bytes": useful_bytes,
        "sectors": distinct_sectors,
        "transferred_bytes": transferred_bytes,
        "efficiency": efficiency,
    }


def print_comparison(sector_size):
    """Compare different access strides."""

    strides = [4, 8, 16, 32, 128, 4096]

    print("\n" + "=" * 79)
    print(f"MEMORY COALESCING — {sector_size}-BYTE SECTORS")
    print("=" * 79)

    print(
        f"{'Stride':>10}"
        f"{'Useful B':>12}"
        f"{'Sectors':>12}"
        f"{'Traffic B':>14}"
        f"{'Efficiency':>16}"
    )

    print("-" * 79)

    for stride in strides:
        result = analyze_access_pattern(
            stride_bytes=stride,
            sector_size=sector_size,
        )

        print(
            f"{result['stride']:>10}"
            f"{result['useful_bytes']:>12}"
            f"{result['sectors']:>12}"
            f"{result['transferred_bytes']:>14}"
            f"{result['efficiency']:>15.2f}%"
        )

    print("=" * 79)


def main():
    print("\nCHAPTER 1 — MEMORY COALESCING SIMULATOR")
    print("64 threads, one FP32 load per thread")
    print("Useful bytes per wavefront: 256")
    print("Results are modeled, NOT actual GPU measurements.")

    for sector_size in [32, 64, 128]:
        print_comparison(sector_size)


if __name__ == "__main__":
    main()
