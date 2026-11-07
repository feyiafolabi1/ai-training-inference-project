
# Chapter 2 — GPU Profiling and Roofline Analysis

## Overview

This chapter explores how GPU performance is measured, modeled, and investigated.

Building on Chapter 1's GPU architecture foundations, it introduces benchmarking methodology, roofline analysis, hardware profiling concepts, and application-level performance optimization.

The goal is to develop an evidence-based approach to GPU performance engineering: measure performance, identify plausible bottlenecks, design targeted experiments, and evaluate their impact.

All performance measurements in this chapter are hypothetical. The tools run locally on a MacBook and do not collect real AMD GPU hardware counters.

## Learning Objectives

By completing this chapter, I learned how to:

- Distinguish latency, throughput, and memory bandwidth.
- Calculate arithmetic intensity and theoretical roofline limits.
- Interpret compute-limited and bandwidth-limited roofline regions.
- Understand the relationship between occupancy and latency hiding.
- Differentiate memory latency limitations from HBM bandwidth saturation.
- Investigate register pressure, spilling, and recomputation trade-offs.
- Interpret profiling counters and develop optimization hypotheses.
- Calculate speedups and application-level performance improvements.
- Understand the purpose of AMD ROCm profiling tools.

## Chapter Structure

### Part 1 — Performance Metrics and Benchmarking

Covered:

- Kernel latency and compute throughput
- Achieved versus peak HBM bandwidth
- Arithmetic intensity
- Benchmark warmup and synchronization
- Mean, median, and latency percentiles
- Steady-state versus cold-start performance
- Fair benchmark comparisons

### Part 2 — Roofline Visualization

Covered:

- Roofline performance ceilings
- Arithmetic intensity and ridge points
- Compute-side versus bandwidth-side behavior
- Logarithmic roofline plots
- Roofline efficiency
- Interpreting optimization movements

The simplified roofline equation is:

P_roofline = min(P_peak, AI × BW_peak)

### Part 3 — GPU Profiling Fundamentals

Covered:

- Wavefront occupancy
- Matrix-unit activity
- Memory stalls and latency hiding
- Memory-level parallelism
- Register pressure and spilling
- Scratch memory traffic
- Cache behavior
- Hardware performance counters
- Evidence-based bottleneck diagnosis

### Part 4 — AMD ROCm Profiling Tools

Introduced:

- ROCprofiler-SDK
- rocprofv3
- ROCprofiler Compute
- ROCprofiler Systems
- GPU kernel tracing
- Performance counter collection
- Profiling workflows for AMD GPUs

Real ROCm profiling experiments are deferred until a supported AMD GPU environment is available.

### Part 5 — Performance Bottleneck Investigation

Practiced:

1. Identifying expensive kernels.
2. Comparing performance against roofline ceilings.
3. Interpreting profiling measurements.
4. Developing optimization hypotheses.
5. Selecting controlled experiments.
6. Measuring application-level speedups.
7. Reprofiling after optimization.

### Part 6 — Practical Profiling Toolkit

Built four Python scripts to apply the chapter's concepts.

## Python Tools

### 1. benchmark_analyzer.py

Analyzes benchmark timing measurements.

Features:

- Mean and median execution time
- Minimum and maximum latency
- P90, P95, and P99 percentiles
- Median-based speedup
- Latency reduction percentages
- Baseline versus optimized comparisons

Uses the nearest-rank method for percentile calculations and the conventional median for median latency.

Small sample sizes are intended for demonstration, not reliable tail-latency estimation.

Run:

    python3 benchmark_analyzer.py

### 2. roofline_plotter.py

Visualizes a simplified AMD MI300X BF16 roofline model.

Features:

- Log-log roofline visualization
- Compute and HBM bandwidth ceilings
- Ridge-point calculation
- Hypothetical kernel performance points
- Roofline efficiency calculations
- Baseline versus optimized kernel comparison
- PNG plot generation

The illustrative model uses:

- Peak BF16 throughput: 1,300 TFLOPS
- Peak HBM bandwidth: 5.3 TB/s
- Ridge point: approximately 245.28 FLOPs/byte

These are simplified reference values, not measured device performance.

Run:

    python3 roofline_plotter.py

### 3. profiling_diagnostics.py

Analyzes synthetic GPU profiling reports and generates possible bottleneck hypotheses.

Features:

- Performance consistency checks
- Roofline ceiling and efficiency calculations
- HBM bandwidth utilization
- Potential bandwidth saturation detection
- Potential memory-latency and memory-level parallelism limitations
- Possible register spilling and occupancy issues
- Suggested optimization experiments

Diagnostic thresholds are educational heuristics, not validated AMD profiling rules.

The tool generates hypotheses rather than definitive bottleneck diagnoses.

Run:

    python3 profiling_diagnostics.py

### 4. optimization_comparison.py

Evaluates kernel-level and application-level performance improvements.

Features:

- Kernel execution-time comparisons
- Kernel speedup calculations
- Latency reduction percentages
- Application execution-time breakdowns
- Application-level speedup calculations
- Ranking of proposed optimization opportunities
- Comparison of baseline and optimized workloads

This tool demonstrates why the kernel with the largest theoretical performance gap is not necessarily the best optimization target.

Run:

    python3 optimization_comparison.py

## Key Engineering Lessons

### 1. Higher occupancy does not guarantee higher performance

Occupancy describes resident wavefront capacity. It provides opportunities for latency hiding but does not directly measure useful compute activity.

### 2. Memory bandwidth and memory latency are different constraints

A kernel can suffer from long memory waits without saturating HBM bandwidth.

Insufficient independent outstanding requests can limit achieved bandwidth.

### 3. Higher arithmetic intensity does not guarantee faster execution

Increasing data reuse can improve theoretical performance limits, but execution time depends on other resources and implementation costs.

### 4. Lower HBM bandwidth utilization can accompany better performance

An optimized kernel may transfer fewer bytes and complete faster while using less achieved HBM bandwidth.

### 5. Roofline regions do not prove actual bottlenecks

The roofline identifies theoretical performance ceilings. Hardware counter evidence and experiments are required to diagnose practical limitations.

### 6. Optimization requires trade-offs

Increasing tile sizes may improve reuse but also increase register pressure and reduce occupancy.

Eliminating spills through recomputation may increase instruction count while reducing memory traffic.

### 7. Optimize the complete application

Kernel optimization priorities should reflect application runtime contribution, realistically achievable speedups, correctness, and engineering effort.

### 8. Reprofile after every meaningful optimization

Improving one kernel can change the application's performance distribution and reveal new optimization priorities.

## Validation

The following scripts were executed successfully:

    python3 benchmark_analyzer.py
    python3 roofline_plotter.py
    python3 profiling_diagnostics.py
    python3 optimization_comparison.py

The roofline visualization generated successfully.

All datasets are hypothetical and should not be interpreted as actual measurements of AMD Instinct GPUs.

## Technology

- Python 3
- NumPy
- Matplotlib
- Python standard library
- AMD CDNA / Instinct architecture concepts
- AMD ROCm profiling concepts

## Next Chapter

Chapter 3 — HIP GPU Programming

The next chapter introduces AMD HIP programming, GPU kernel implementation, host/device execution, thread mapping, synchronization, and the first practical GPU programming experiments.

The long-term objective is to apply these skills to training and inference optimization on AMD Instinct GPUs.
