# Chapter 1 — GPU Architecture and Execution Model

## AI Training and Inference Project

This chapter introduces the GPU architecture and performance concepts needed to understand, develop, and optimize AI training and inference workloads, with an emphasis on AMD GPU execution models.

The objective is to build a quantitative understanding of how GPU threads execute, how hardware resources constrain parallelism, how memory access patterns affect efficiency, and how theoretical performance limits can be analyzed.

The chapter includes five Python exercises that model these concepts without requiring access to a discrete GPU.

> **Important:** These exercises are analytical simulations, not actual GPU profiling tools. Hardware resource values used in the occupancy exercises are hypothetical. The roofline exercises use approximate published AMD Instinct MI300X peak specifications and hypothetical workloads.

---

## 1. Learning Objectives

By the end of this chapter, I can:

- Explain the relationship between compute units, workgroups, wavefronts, and threads.
- Calculate workgroup counts, wavefront counts, and inactive thread slots.
- Describe the roles of registers, LDS, caches, and HBM.
- Explain how register pressure and LDS allocation affect occupancy.
- Distinguish occupancy from GPU utilization and achieved throughput.
- Analyze memory coalescing using simplified sector-based transaction models.
- Derive GEMM floating-point operation counts and idealized memory traffic.
- Calculate arithmetic intensity and theoretical roofline bounds.
- Interpret hypothetical GPU performance metrics and form optimization hypotheses.

---

## 2. GPU Execution Model

An AMD GPU organizes computation using a hierarchy of workgroups, wavefronts, and workitems.

In the CDNA execution model used throughout these exercises, a wavefront contains 64 workitems.

For a one-dimensional workload, assuming one workitem processes one element:

```text
GPU
 |
 +-- Compute Units (CUs)
       |
       +-- Resident Workgroups
             |
             +-- Wavefronts (64 workitems each)
                   |
                   +-- Individual Workitems
```

### Core calculations

For workgroups whose size is a multiple of 64:

```text
Workgroups = ceil(Elements / Threads per Workgroup)

Wavefronts per Workgroup = Threads per Workgroup / 64

Total Wavefronts = Workgroups * Wavefronts per Workgroup

Launched Workitems = Workgroups * Threads per Workgroup

Inactive Workitems = Launched Workitems - Elements
```

The inactive-workitem calculation assumes one element per workitem and a boundary check that prevents out-of-range processing.

Launching fewer workgroups does not necessarily improve performance. Workgroup size also affects resource allocation, scheduling, and latency hiding.

---

## 3. GPU Memory Hierarchy

The major memory resources discussed in this chapter include:

| Resource | Role |
|---|---|
| Registers | Fast, limited storage for thread-local values and intermediate results |
| LDS (Local Data Share) | On-chip memory shared among workitems in the same workgroup |
| L1 cache | Hardware-managed caching close to compute resources |
| L2 cache | Larger shared caching infrastructure |
| HBM | High-capacity, high-bandwidth memory for model weights, activations, and other data |

Registers and LDS are not simply sequential stages through which every memory access must pass.

### Data reuse

Data reuse can reduce redundant memory accesses.

Examples:

- Registers can retain frequently used thread-local values.
- LDS can enable workitems within a workgroup to cooperate and reuse data.
- Tiled matrix multiplication can reuse input matrix elements across multiple computations.

However, allocating more registers or LDS consumes limited resources and may reduce the number of resident workgroups or wavefronts.

---

## 4. Occupancy and Resource Constraints

Occupancy describes the proportion of available wavefront slots occupied by resident wavefronts.

```text
Occupancy = Resident Wavefronts / Maximum Resident Wavefronts
```

For the simplified occupancy exercises, the hypothetical CU has:

| Resource | Capacity |
|---|---:|
| Registers | 65,536 32-bit registers |
| LDS | 64 KiB |
| Maximum resident wavefronts | 32 |
| Wavefront size | 64 workitems |

For each kernel, the model calculates:

```text
Wavefronts per Workgroup =
    Threads per Workgroup / 64

Registers per Workgroup =
    Threads per Workgroup * Registers per Thread

Register Workgroup Limit =
    floor(Total Registers / Registers per Workgroup)

LDS Workgroup Limit =
    floor(Total LDS / LDS per Workgroup)

Wavefront Workgroup Limit =
    floor(Maximum Resident Wavefronts / Wavefronts per Workgroup)

Resident Workgroups =
    min(Register Limit, LDS Limit, Wavefront Limit)

Occupancy =
    (Resident Workgroups * Wavefronts per Workgroup)
    / Maximum Resident Wavefronts
```

The simulation ignores several real hardware constraints, including register allocation granularity and additional residency restrictions.

### Example results

| Kernel | Registers/Thread | LDS/WG | Limiting Resource | Occupancy |
|---|---:|---:|---|---:|
| A | 32 | 8 KiB | Registers, LDS, wavefront slots | 100% |
| B | 64 | 24 KiB | LDS | 25% |
| C | 128 | 4 KiB | Registers | 25% |

### Key takeaway

Higher occupancy is not automatically better performance.

A kernel with lower occupancy can still outperform another kernel if additional registers or LDS enable better data reuse, instruction efficiency, or other optimizations.

---

## 5. Memory Coalescing

Memory coalescing describes how nearby memory requests from workitems can be serviced efficiently by the memory system.

A memory stride is the distance between the starting addresses of consecutive accesses.

For FP32 loads:

```text
Element size = 4 bytes

Thread i address = Base Address + i * Stride
```

Examples:

```text
4-byte stride:
Thread 0 -> 0
Thread 1 -> 4
Thread 2 -> 8

8-byte stride:
Thread 0 -> 0
Thread 1 -> 8
Thread 2 -> 16

4096-byte stride:
Thread 0 -> 0
Thread 1 -> 4096
Thread 2 -> 8192
```

A 4-byte stride produces contiguous FP32 accesses with no gaps.

An 8-byte stride leaves four unrequested bytes between neighboring FP32 loads.

### Simplified transaction model

The simulator assumes:

- One wavefront with 64 workitems.
- One 4-byte FP32 load per workitem.
- A sector-aligned starting address.
- A configurable sector size.
- Every distinct touched sector contributes its full size in transferred bytes.
- No cache hits.

For a 32-byte sector size:

| Stride | Sectors Touched | Modeled Traffic | Useful-Data Efficiency |
|---|---:|---:|---:|
| 4 bytes | 8 | 256 bytes | 100% |
| 8 bytes | 16 | 512 bytes | 50% |
| 16 bytes | 32 | 1,024 bytes | 25% |
| 32 bytes | 64 | 2,048 bytes | 12.5% |
| 128 bytes | 64 | 2,048 bytes | 12.5% |
| 4096 bytes | 64 | 2,048 bytes | 12.5% |

Efficiency is calculated as:

```text
Useful-Data Efficiency =
    Useful Bytes / Modeled Transferred Bytes * 100%
```

The distance between memory addresses does not imply that all intervening bytes must be transferred.

These are simplified sector counts and should not be interpreted as measured AMD GPU memory transactions.

---

## 6. GEMM Arithmetic Intensity

General Matrix Multiplication (GEMM) is a foundational operation in AI training and inference.

For square matrix multiplication:

```text
C = A @ B

A shape: N x N
B shape: N x N
C shape: N x N
```

### Floating-point operation count

Each output element requires:

```text
N multiplications
N - 1 additions

Total = 2N - 1 FLOPs per output
```

There are N squared output elements.

Therefore:

```text
Exact FLOPs = N^2 * (2N - 1)

            = 2N^3 - N^2
```

For large matrices, the common approximation is:

```text
Approximate GEMM FLOPs = 2N^3
```

Standard GEMM throughput reporting also conventionally counts each multiply-accumulate as two FLOPs.

### Idealized memory traffic

Assume each input matrix is read once from HBM and the output matrix is written once.

For BF16 inputs and a BF16 output:

```text
Read A  = 2N^2 bytes
Read B  = 2N^2 bytes
Write C = 2N^2 bytes

Total = 6N^2 bytes
```

For BF16 inputs with an FP32 output:

```text
Read A  = 2N^2 bytes
Read B  = 2N^2 bytes
Write C = 4N^2 bytes

Total = 8N^2 bytes
```

These values represent idealized memory traffic, not guaranteed actual traffic.

### Arithmetic intensity

```text
Arithmetic Intensity = FLOPs / Bytes Transferred
```

Using the approximate GEMM FLOP count:

| Input / Output Format | Idealized Traffic | Arithmetic Intensity |
|---|---|---|
| FP32 / FP32 | 12N^2 | N / 6 |
| BF16 / BF16 | 6N^2 | N / 3 |
| BF16 / FP32 | 8N^2 | N / 4 |

Reducing data traffic without reducing computation increases arithmetic intensity.

This motivates optimizations such as tiling and data reuse through registers, LDS, and caches.

---

## 7. Roofline Performance Model

The roofline model estimates an upper bound on performance based on compute throughput, memory bandwidth, and arithmetic intensity.

```text
Roofline Performance =
    min(Peak Compute, Arithmetic Intensity * Peak Bandwidth)
```

### Approximate MI300X reference values

The BF16 matrix-compute exercises use the following rounded reference values:

| Metric | Value |
|---|---:|
| Peak BF16 matrix throughput | 1,300 TFLOPS |
| Peak HBM bandwidth | 5.3 TB/s |

These values describe theoretical peak capabilities, not measured sustained performance.

The simplified ridge point is:

```text
Ridge Point = Peak Compute / Peak Bandwidth

            = 1300 / 5.3

            = approximately 245.28 FLOPs/byte
```

When arithmetic intensity is below the ridge point, the HBM bandwidth line provides the tighter upper bound.

When arithmetic intensity is above the ridge point, peak compute throughput provides the tighter upper bound.

This classification does not, on its own, prove the kernel's actual execution bottleneck.

### GEMM example

For a 256 x 256 BF16 GEMM with BF16 output:

```text
Approximate FLOPs = 2 * 256^3
                  = 33,554,432 FLOPs

Idealized Traffic = 6 * 256^2
                  = 393,216 bytes

Arithmetic Intensity = 85.33 FLOPs/byte

Roofline Ceiling = min(1300, 85.33 * 5.3)
                 = approximately 452.27 TFLOPS
```

The roofline model places this example in the bandwidth-limited region.

Actual performance may be lower because of instruction efficiency, scheduling, occupancy, memory access behavior, synchronization, or other constraints.

---

## 8. Integrated Performance Analysis

The final exercise combines arithmetic intensity, achieved throughput, measured bandwidth, occupancy, and theoretical roofline bounds.

For each hypothetical kernel, it calculates:

```text
Compute Utilization =
    Achieved Compute Throughput / Peak Compute Throughput

Bandwidth Utilization =
    Achieved HBM Bandwidth / Peak HBM Bandwidth

Roofline Efficiency =
    Achieved Throughput / Roofline Ceiling

Gap to Roofline =
    Roofline Ceiling - Achieved Throughput
```

### Example profiles

| Metric | Kernel A | Kernel B | Kernel C |
|---|---:|---:|---:|
| Arithmetic intensity (FLOPs/byte) | 50 | 300 | 100 |
| Achieved throughput (TFLOPS) | 200 | 650 | 400 |
| Roofline ceiling (TFLOPS) | 265 | 1,300 | 530 |
| Roofline efficiency | 75.47% | 50.00% | 75.47% |
| Occupancy | 85% | 35% | 30% |

Kernel B has the largest relative gap to its theoretical roofline bound.

An initial profiling investigation could examine matrix-unit utilization, instruction dependencies, occupancy, register pressure, synchronization, and memory behavior.

The model does not establish the root cause of underperformance.

### Key engineering principle

**A performance metric is evidence, not a diagnosis by itself.**

For example:

- Low occupancy does not necessarily mean occupancy is limiting throughput.
- High HBM bandwidth does not automatically imply that every memory access is efficient.
- A compute-limited roofline region does not prove matrix units are saturated.
- Increasing data reuse can improve performance even when it reduces occupancy.

Optimization should be based on targeted hypotheses, profiling measurements, and controlled benchmarks.

---

## 9. Python Exercises

| File | Description |
|---|---|
| `thread_mapping.py` | Initial workitem and thread-mapping exploration |
| `wavefront_analysis.py` | Workgroup and wavefront count calculations |
| `occupancy_calculator.py` | Register-, LDS-, and wavefront-limited occupancy |
| `memory_coalescing.py` | Memory stride and sector-efficiency simulation |
| `roofline_calculator.py` | GEMM arithmetic intensity and roofline calculation |
| `bottleneck_analyzer.py` | Integrated hypothetical GPU performance analysis |

All exercises use Python's standard library and run locally without ROCm or discrete GPU hardware.

### Running the exercises

From this chapter's directory:

```bash
python3 wavefront_analysis.py
python3 occupancy_calculator.py
python3 memory_coalescing.py
python3 roofline_calculator.py
python3 bottleneck_analyzer.py
```

The initial `thread_mapping.py` can also be explored independently.

---

## 10. Limitations

These exercises deliberately simplify several aspects of real GPU behavior:

1. Occupancy calculations ignore detailed hardware allocation granularity and additional residency restrictions.
2. The memory-coalescing simulator assumes full-sector transfers and does not model actual cache or memory transaction behavior.
3. Idealized GEMM traffic assumes input matrices are fetched once and the output is written once.
4. The roofline calculator uses advertised peak bandwidth and BF16 matrix throughput, not sustained benchmark results.
5. Hypothetical kernel profiles are provided for analytical practice rather than obtained from real hardware.
6. A roofline bound cannot identify every hardware or software performance bottleneck.

These limitations will help guide future profiling and optimization experiments.

---

## 11. Key Takeaways

The most important lessons from Chapter 1 are:

- GPU performance depends on parallel execution, resource residency, memory efficiency, and useful computation.
- Larger workgroups do not automatically make workloads faster.
- Registers and LDS can enable data reuse, but consume finite hardware resources.
- More occupancy does not guarantee more throughput.
- Coalesced memory accesses can reduce unnecessary transferred data.
- Arithmetic intensity measures computational work relative to data movement.
- GEMM offers substantial opportunities for data reuse.
- Roofline models establish performance ceilings, not guaranteed performance.
- Real optimization requires measurements and evidence-driven reasoning.

---

## 12. Next Chapter

**Chapter 2 — Roofline Modeling and GPU Profiling**

The next chapter builds on the analytical foundations introduced here.

Planned topics include:

- GPU benchmarking methodology.
- Latency, throughput, and bandwidth measurements.
- Building and visualizing roofline models.
- Understanding GPU performance counters.
- Introduction to AMD ROCm profiling tools.
- Separating theoretical bottlenecks from measured limitations.
- Preparing a repeatable profiling workflow for future HIP kernels.

The objective is to move from calculating theoretical performance limits toward a practical, measurement-oriented GPU performance workflow.

---

## Project Progress

**Chapter 1 of 17 — Completed**

The larger project will explore AMD GPU programming, kernel optimization, transformer inference, efficient attention, inference engines, and distributed training, culminating in an AMD Instinct inference optimization capstone.