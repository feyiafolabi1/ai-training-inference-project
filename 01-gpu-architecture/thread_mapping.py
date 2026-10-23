# ==================================================
# AI TRAINING AND INFERENCE PROJECT
# CHAPTER 1 — GPU THREAD MAPPING
# ==================================================
#
# Simulates the logical structure of an AMD
# GPU kernel launch.
#
# This is NOT real GPU execution.
#
# Concepts:
#   - Work items / threads
#   - Workgroups / blocks
#   - Wavefronts
#   - Bounds checking
#   - Inactive work items
#
# AMD Instinct wavefront size: 64
# ==================================================

from math import ceil


WAVEFRONT_SIZE = 64

THREADS_PER_BLOCK = 256


def simulate_kernel(element_count):

    blocks = ceil(
        element_count / THREADS_PER_BLOCK
    )

    total_threads = (
        blocks * THREADS_PER_BLOCK
    )

    wavefronts_per_block = ceil(
        THREADS_PER_BLOCK / WAVEFRONT_SIZE
    )

    total_wavefronts = (
        blocks * wavefronts_per_block
    )

    processed_indices = []

    for block_id in range(blocks):

        for thread_id in range(THREADS_PER_BLOCK):

            global_id = (
                block_id * THREADS_PER_BLOCK
                + thread_id
            )

            # Simulates a GPU bounds check.

            if global_id < element_count:

                processed_indices.append(
                    global_id
                )

    assert processed_indices == list(
        range(element_count)
    )

    inactive_threads = (
        total_threads - element_count
    )

    active_percentage = (
        100 * element_count / total_threads
        if total_threads > 0
        else 0
    )

    print()
    print("=" * 60)

    print(
        f"Array size: {element_count}"
    )

    print(
        f"Threads per block: {THREADS_PER_BLOCK}"
    )

    print(
        f"Wavefront size: {WAVEFRONT_SIZE}"
    )

    print("-" * 60)

    print(f"Blocks launched: {blocks}")

    print(
        f"Total threads launched: {total_threads}"
    )

    print(
        f"Wavefronts per block: "
        f"{wavefronts_per_block}"
    )

    print(
        f"Total wavefronts: {total_wavefronts}"
    )

    print(
        f"Inactive threads: {inactive_threads}"
    )

    print(
        f"Useful thread percentage: "
        f"{active_percentage:.2f}%"
    )

    print("=" * 60)


def main():

    test_sizes = [
        64,
        65,
        255,
        256,
        1000,
        1024,
        1025,
    ]

    for size in test_sizes:

        simulate_kernel(size)


if __name__ == "__main__":
    main()