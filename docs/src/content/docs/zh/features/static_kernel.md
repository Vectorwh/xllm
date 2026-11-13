---
title: "CANN Static Kernel"
sidebar:
  order: 82
---

## 功能介绍

Static Kernel 是针对具体算子参数编译的专用内核。它根据算子的 shape、dtype、format 和属性，将能够确定的 tiling 参数固化到编译结果中，例如数据分块、循环边界和尾块大小。固化的是这些执行配置，输入数据和权重数值仍在运行时传入。

常规内核需要适配多种输入，部分 tiling 参数在运行时才确定，内核中会保留相应的参数读取、标量计算和分支处理。静态编译让编译器提前看到这些参数，从而生成更适合当前输入规模的代码。常量化带来的循环展开、常量传播和分支裁剪可参考 [Ascend C AOT 编译优化](https://asc.gitcode.com/guide/programming_guide/advanced_programming/aot_compilation_optimization.html)。

xLLM 直接使用 CANN 的算子参数采集、编译和重新选核能力，在启动阶段生成并加载静态内核。 当前接入 Python 执行器，由 Python 执行器管理采集、编译和加载，不依赖 TorchAir 或 `torch.compile`。

## 为什么能提升性能

静态编译的收益来自算子内部的执行优化：

- **减少标量计算和控制开销。** 分块大小、循环次数等参数成为常量后，编译器可以提前算出相关表达式，裁剪不需要的分支，并对已知边界的循环进行展开，减少设备运行时的控制指令。
- **减少保守的同步。** 数据搬运和计算涉及的范围更加明确，编译器有机会去掉原本因信息不足而加入的同步，让搬运与计算更好地并行。
- **改善存储和代码安排。** 编译器可以按实际数据规模、分块方式安排缓冲区和指令，减少通用实现为兼容不同输入而保留的开销。

这些是静态编译能够利用的优化机会，具体效果由算子实现决定。控制和同步开销占比较高的算子，通常更容易从中受益。 [TorchAir 的静态 Kernel 说明](https://gitcode.com/Ascend/torchair/blob/26.1.0/docs/zh/npugraph_ex/basic/static_kernel_compile.md)也介绍了这些收益来源。

### 与 ACL Graph 的关系

| 功能 | 优化对象 | 主要作用 |
|---|---|---|
| ACL Graph | 一组任务的执行顺序和依赖 | 复用捕获的任务，减少逐算子下发与调度开销 |
| Static Kernel | 单个算子的设备代码 | 利用已知参数做专用编译，减少内核执行开销 |

ACL Graph 捕获的仍可能是常规内核，其中的动态参数处理和通用逻辑会继续执行。先加载静态内核再捕获图，可以同时获得任务调度和算子执行两方面的优化。

## 使用方式

启动服务时添加：

```shell
--model_impl=python \
--enable_graph=true \
--enable_static_kernel=true \
--static_kernel_cache_dir=/data/xllm/static_kernel_cache
```

默认关闭，当前在 CANN 9.1 上验证。使用 Static Kernel 需要安装 CANN 编译工具并加载 CANN 环境变量；复用已有缓存时也有此要求，用于核对缓存对应的编译环境。TorchNPU 还需提供算子参数采集和按自定义路径重新选核的接口，缺少接口时启动会明确报错。

### 首次启动与缓存

启动时，各个 worker 在实际 decode graph 预热档位中采集算子参数，汇总同一共享缓存目录内的描述，交给 CANN `op_compiler` 编译。每个档位加载静态内核后，再进行图捕获；后续档位只编译新增的描述，安装包保留此前已经覆盖的算子。普通 prefill 预热不参与静态采集，也不为 Static Kernel 额外执行短 eager 预热。因此，首次启动会增加编译和安装时间。

再次启动时仍会采集预热中的实际参数；描述和编译环境不变时，直接复用缓存。采集、编译和加载由 worker 自行完成，通过 TCPStore 同步准备状态，普通请求和 graph replay 不进入这套启动流程。运行中遇到未覆盖的 shape，仍使用 CANN 常规选核。

未指定缓存目录时，使用 `$XDG_CACHE_HOME/xllm/static_kernel`；未设置 `XDG_CACHE_HOME` 时，使用 `~/.cache/xllm/static_kernel`。

## 排除算子

通过环境变量指定不参与静态编译的算子：

```shell
export XLLM_STATIC_KERNEL_BLACKLIST="MatMulV2,Cast"
```

名称使用 **CANN OpType**，不是 PyTorch 函数名或 `aclnn*` 接口名。多个名称以逗号分隔，区分大小写，默认不排除算子。修改后重启服务生效，多 worker 应使用同一份列表。被排除的算子仍按原路径执行。

算子名可以从 CANN 注册头文件 `opp/built-in/op_proto/inc/*.h` 的 `REG_OP(...)` 中查询；已有 profiling 也可以查看 `OP Type` 列，TorchNPU profiling 对应 `Type` 列。

## 支持范围

Static Kernel 实际覆盖的算子由 CANN 能力和自定义黑名单决定。 CANN 不支持的静态编译的算子、ATB 内部算子以及自定义 kernel，保持原执行路径。

完整参数和环境变量说明见 [服务启动参数](/zh/cli_reference/)。
