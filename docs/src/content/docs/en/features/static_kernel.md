---
title: "CANN Static Kernel"
sidebar:
  order: 82
---

## Overview

Static Kernel specializes an operator for a given set of parameters. Based on its shape, dtype, format, and attributes, it makes known tiling parameters, such as block sizes, loop bounds, and tail sizes, constants in the compiled kernel. These execution settings are fixed at compile time; input data and weight values are still supplied at runtime.

General-purpose kernels support multiple input configurations. Some tiling parameters are determined at runtime, so the kernel must retain the corresponding parameter reads, scalar calculations, and branches. Static compilation exposes these parameters to the compiler in advance, allowing it to generate code for the specific workload. See [Ascend C AOT compilation optimization](https://asc.gitcode.com/guide/programming_guide/advanced_programming/aot_compilation_optimization.html) for how constants enable loop unrolling, constant propagation, and branch elimination.

xLLM uses CANN's operator parameter collection, compilation, and kernel reselection interfaces directly to build and load static kernels at startup. The Python executor manages collection, compilation, and loading, without a dependency on TorchAir or `torch.compile`.

## Why It Improves Performance

Static compilation can improve execution inside individual operators:

- **Less scalar computation and control overhead.** Once block sizes and loop counts become constants, the compiler can evaluate related expressions in advance, remove unnecessary branches, and unroll loops with known bounds, reducing control instructions on the device.
- **Less conservative synchronization.** With more information about data movement and computation, the compiler may remove synchronization that was needed when those details were unknown, allowing transfers and computation to overlap more effectively.
- **Better buffer and instruction layout.** The compiler can arrange buffers and instructions for the actual data sizes and tiling, reducing overhead retained by a general-purpose implementation to support different inputs.

These are optimization opportunities; the actual benefit depends on the operator implementation. Operators with a larger share of control and synchronization overhead are generally more likely to benefit. [TorchAir's Static Kernel documentation](https://gitcode.com/Ascend/torchair/blob/26.1.0/docs/zh/npugraph_ex/basic/static_kernel_compile.md) also describes these sources of improvement.

### Relationship to ACL Graph

| Feature | Optimization Target | Main Effect |
|---|---|---|
| ACL Graph | Task execution order and dependencies | Reuses captured tasks to reduce per-operator dispatch and scheduling overhead |
| Static Kernel | Device code for an individual operator | Uses known parameters to specialize compilation and reduce kernel execution overhead |

ACL Graph can capture general-purpose kernels, which still execute their runtime parameter handling and general-purpose logic. Loading static kernels before graph capture combines task scheduling improvements with faster operator execution.

## Usage

Add these flags when starting the service:

```shell
--model_impl=python \
--enable_graph=true \
--enable_static_kernel=true \
--static_kernel_cache_dir=/data/xllm/static_kernel_cache
```

Static Kernel is disabled by default and has been validated on CANN 9.1. Install the CANN compilation tools and set up the CANN environment before starting xllm. Cache reuse has the same requirements because the compilation tools are still used to verify the cache's compilation environment. TorchNPU must expose the operator argument collection and custom-path kernel reselection interfaces; startup reports any missing interfaces.

### First Startup and Cache Reuse

During the actual decode graph warmup buckets, workers collect operator parameters, merge descriptions within each shared cache directory, and compile them with CANN `op_compiler`. Each bucket loads its static kernels before graph capture. Later buckets compile only new descriptions, while their installations retain the operators covered earlier. Ordinary prefill warmup does not collect static kernels, and Static Kernel does not add a separate short eager warmup. Compilation and installation therefore add time to the first startup.

Subsequent startups still collect the actual warmup parameters and reuse the cache when the descriptions and compilation environment match. Workers manage collection, compilation, and loading themselves, synchronizing preparation through TCPStore. Ordinary requests and graph replay do not enter this startup flow. Shapes not covered by the cache continue to use CANN's regular kernel selection at runtime.

If no cache directory is specified, xLLM uses `$XDG_CACHE_HOME/xllm/static_kernel`. If `XDG_CACHE_HOME` is unset, it uses `~/.cache/xllm/static_kernel`.

## Excluding Operators

Use this environment variable to exclude operators from static compilation:

```shell
export XLLM_STATIC_KERNEL_BLACKLIST="MatMulV2,Cast"
```

Names are **CANN OpTypes**, rather than PyTorch function names or `aclnn*` interface names. Separate names with commas; matching is case-sensitive. No operators are excluded by default. Restart the service after changing the list, and use the same list for all workers. Excluded operators keep their existing execution paths.

Operator names can be found in `REG_OP(...)` declarations in the CANN registration headers under `opp/built-in/op_proto/inc/*.h`. Existing profiling results also expose them in the `OP Type` column, or the `Type` column in TorchNPU profiling.

## Scope

Static Kernel coverage depends on CANN's capabilities and the custom blacklist. Operators that CANN cannot compile statically, operators internal to ATB, and custom kernels keep their existing execution paths.

See the [CLI Reference](/en/cli_reference/) for the full parameter and environment variable descriptions.
