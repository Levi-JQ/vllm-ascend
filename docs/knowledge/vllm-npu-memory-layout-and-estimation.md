# vLLM（昇腾）显存构成与预估机制分析

## 0. 三个问题的快速答案

| 问题 | 答案 |
|---|---|
| 显存占用包括哪些 | 权重、KV cache、激活值（瞬态峰值）、持久运行缓冲、non-torch（HCCL/ACL/workspace/graph 池）、allocator 滞留余量六大类；**prefix cache 不额外占显存**（见 1.3 纠正） |
| 激活/non-torch 运行期变动吃哪块 | **不挤 KV cache**（启动时一次性分配、地址固定），也不专属于 `(1-util)` 那块；它吃的是 device 物理剩余空闲（profiling 低估差额 + 未用满的预算 + 1-util 余量），吃穿即 OOM（见第 2 节） |
| 显存预估机制 | 启动时 `determine_available_memory`：dummy forward（max_num_batched_tokens 形状）实测激活峰值 + 权重计量 + non-torch 增量，`KV 预算 = util×总显存 − 三项之和`；运行期无任何预算 enforcement（见第 3 节） |

---

## 1. 显存占用分类

### 1.1 排布总览

单卡（以 64GB NPU 为例，util=0.92）的显存排布：

```
┌───────────────────────────────────────────────────────────┐
│                    1 - gpu_memory_utilization              │  ← 留给系统/其他进程/驱动，
│                    （本例 ~5.1 GB）                        │    vLLM 的目标是不碰它
├───────────────────────────────────────────────────────────┤
│ ① 权重 weights（target + draft，load_model 计量）           │
│ ② KV cache（启动时一次性分配，地址/大小运行期固定）           │  ← requested_memory
│ ③ 激活值瞬态峰值（profile_run 实测预算）                     │    = util × total
│ ④ 持久运行缓冲（input buffers、PP 接收缓冲、sampler 状态）    │    内的"非 KV"部分
│ ⑤ non-torch：HCCL 通信缓冲、ACL context、算子 workspace、    │
│    aclgraph/CUDA graph 池                                  │
│ ⑥ allocator 滞留余量：reserved − allocated（只增不缩）       │
└───────────────────────────────────────────────────────────┘
```

### 1.2 分类明细

| # | 类别 | 分配时机 | 管理者 | 预估/计量方式 | 代码位置 |
|---|---|---|---|---|---|
| 1 | 权重（含 draft/speculator） | load_model | torch | `DeviceMemoryProfiler` 包住加载，torch allocated 净增量 `model_memory_usage`（NPU 实现：`reset_peak` 后取 `max_memory_allocated` 的存量差） | `model_runner.py:394-427`、`vllm_ascend/platform.py:206-208` |
| 2 | KV cache | 启动时一次性 | vLLM scheduler（逻辑块管理） | `available_kv = requested_memory − non_kv_cache_memory` 换算 block 数 | `worker.py:623`、`worker.py:1159-1165` |
| 3 | 激活值 | 每次 forward 瞬态 | torch caching allocator | profile_run 的 dummy forward 峰值 `torch_peak_increase`，graph capture 前记录 | `worker.py:596-604` |
| 4 | 持久运行缓冲 | load_model 后、profiling 前 | torch | **无独立计量**（落在 weights 计量与 peak 基线之间，见 3.5 缺口④） | `gpu_model_runner.py:528` 等 |
| 5a | non-torch：HCCL 通信缓冲、ACL context | 分布式初始化、通信建立 | CANN/HCCL（torch 之外） | `non_torch_increase` = device 占用 − torch reserved 的增量 | `mem_utils.py:152-159`、`worker.py:605` |
| 5b | non-torch：算子 workspace（PTA_WORKSPACE） | 运行期首次触发算子时 | NPUWorkspaceAllocator——不经 torch caching allocator、直接落 device，**同属 non-torch** | dummy run **走到的算子已计入** `non_torch_increase`（指标 B）；**未走到的算子（如 attn，`skip_attn=True`）无预估**，运行期首次触发时才动态申请（内部案例的 240MB，见 2.3） | msMemScope 内存拆解（第 4 节） |
| 5c | graph 池（aclgraph） | KV 分配后 capture | torch graph pool | `npugraph_memory_bytes` 单独统计，在 KV 预算之外 | `worker.py:885-903` |
| 6 | allocator 滞留（reserved−allocated） | 运行期池扩张后 | torch caching allocator | **无预估**；reserved 只增不缩（除非 empty_cache） | 内部案例 batch1 滞留分析（见 2.3） |

> - **5a 与 5b 同属 non-torch**（口径 = device 已用 − torch reserved，`mem_utils.py:152-159`），差别只在预估覆盖：dummy run（eager 真实 forward）**走到的算子**，其 workspace 在 profile 期间即申请、随指标 B 计入；**没走到的算子**（如 attn，`skip_attn=True`）运行期首次触发才申请，构成缺口③。MC2 预热补丁（见 3.3）就是"把 non-torch 分配提前到 profile 期被计入"的官方先例
> - non_kv_cache_memory = A + B + C
>   - torch_peak_increase        = 瞬态峰值（激活预算）          ← 指标 A
>   - non_torch_increase         = (device占用 − torch_reserved) 的增量   ← 指标 B
>   - weights_memory             = model_memory_usage                ← 指标 C

vLLM 上游对内存的官方三分法（`vllm/utils/mem_utils.py:242-283` 注释）：

1. **其他进程**占用（category 1）；
2. **本实例 torch** 占用（category 2）：权重 + 激活；
3. **本实例非 torch** 占用（category 3）：HCCL/NCCL、attention 后端缓冲、workspace 等。

### 1.3 纠正：prefix cache 不额外占显存

**"GPU 利用率内的额外空间用来做 prefix cache" 这个说法需要纠正。**

自动前缀缓存（APC，`--enable-prefix-caching`）是 **KV cache 池内部 block 的逻辑复用机制**，不是一块独立显存：

- prefix cache 命中时，调度器通过 block hash 表直接**引用已存在的物理 block**（挂在多个请求的 block table 上），既不重新计算 prefill，也不新分配显存；
- 未命中时，照常从 KV 池的 free blocks 里分配——与不开 APC 完全一致；
- 空间紧张时，cached 状态的 block 由 LRU evictor 逐出、归还 free 池给新请求用。

所以 APC **消耗的就是 KV cache 池本身，没有独立预算，反而节省 KV 占用**（相同前缀的请求共享物理 block）。它影响的是"KV 池能同时服务多少 token"的有效容量，而不是显存分类里多出一项。

### 1.4 "OS 预留"与 `(1 - utilization)` 那块空间的关系

`gpu_memory_utilization` 只在启动时用于计算 `requested_memory = total × util`（`worker.py:453-455`），进而得出 KV 预算。"OS 预留"需要拆三层看，**主体不在 `(1-util)` 那块里**：

| 层 | 内容 | 位置 |
|---|---|---|
| ① 固件/驱动底层保留 | CANN 固件/驱动在 HBM 的常驻保留 | **`total_memory` 之外**——`get_memory_info`（`mem_utils.py:142`）报告前已扣除；**util 的分母都不含它**，上层不可见（64GB 物理 HBM，total 约 62-63GB） |
| ② vLLM 进程的 CANN 运行时 | ACL context、HCCL communicator、驱动映射到进程的部分 | **util 预算内**：`non_torch_increase` 以 `before_create`（init_snapshot）为基线（`mem_utils.py:304-306`），设备/分布式初始化的 non-torch 增长全部计入并从 `available_kv` 扣除；仅少量低估漏出的部分落到物理空闲（150MiB 安全垫的由来，`worker.py:893-895`） |
| ③ `(1-util)×total` 物理空闲 | 扣完 ①② 后的自由空间 | 预算外，留给：同卡其他进程（若有）、**运行期预算外增长的兜底**（见第 2 节，allocator 扩张不看 util 只看物理空闲）、碎片余量 |

观测提示：`npu-smi` 的 HBM Used 与 torch 的 allocated+reserved 的差值，主要来自 ①（total 外保留）和 ② 中 driver layer 的部分，不要把它当作可用空间。层 ② 是预算内成本——util 相同时 CANN 运行时占得越多，KV cache 越小，它不从 `(1-util)` 里出。

---

## 2. 运行期变动占用哪块空间

### 2.1 KV cache 先到先得、地址固定

KV cache 在 `initialize_from_config → initialize_kv_cache`（`worker.py:1159-1165`）**启动时一次性分配**。此后：

- 池的大小和地址**永不改变**；
- 运行期的 preemption/recompute 只是调度器在**池内**释放逻辑块（把请求的 KV block 归还 free 池），不缩池、不移动；
- torch allocator 分配激活时**不可能**落在 KV 已占的地址上。

因此运行期激活/non-torch 的变动**不会挤占 KV cache 空间**——KV 能存多少 token 在启动那一刻就定死了。

### 2.2 三层内存视图

理解"变动吃哪里"需要三层视图（NPU 上 torch_npu 与 CUDA 同构）：

```
allocated  ⊆  reserved  ⊆  device 已用
   ↑            ↑                ↑
当前活着的张量   caching allocator   物理显存（含 non-torch）
                池持有的块          + 其他进程
```

运行期激活增长时的分配顺序：

1. **优先从 reserved 池内空闲块切**（快，不向 device 申请）；
2. 池内不够 → 向 device 申请新块，**reserved 扩张**——此时只看 device 物理空闲，**不检查 gpu_memory_utilization**；
3. device 物理空闲不足，或碎片化后无足够**连续**块 
   1.  torch 路径触发 `empty_cache` 池重整后重试（可能成功）；
   2. **非 torch 路径（PTA_WORKSPACE 等）无重整，直接 OOM**。

所以答案：运行期变动吃的是 **device 物理剩余空闲**，其构成 = profiling 低估的差额 + 预算内没用满的部分 + `(1-util)` 那块。它不属于任何一项专用空间；预算只是"目标值"，运行期没有 enforcement。

### 2.3 案例复盘：DBO 双流 + ATB workspace + 碎片化

下面复盘一起内部线上案例（DSR1 部署，util 0.95/0.94；数据来自内部 profiling 复盘，未公开发布，仅作机制演示）——它完整演示了 2.2 的每一步：

**现象**：服务拉起后与压测后显存差 ~5GB，benchmark 后单 curl 请求 OOM；profiling 对比显示算子分配无异常增长，增长集中在"算子保留/进程保留"（即 reserved 池扩张），排除泄漏。

**机制一：DBO 双流内存池独立**。启用 DBO 后推理分流为主流 + Micro Batch 从流，**两个流的 caching allocator 池独立、无法复用**，实际峰值 = 两流叠加：

| 阶段 | 池变化 |
|---|---|
| profile_run | 主流池扩张 ~3.2GB（dummy 按理论最大跑），结束后 `empty_cache` 清理 |
| batch 1（不触发 DBO） | 全部在主流，池扩张 ~2.3GB（util 0.95 场景）并**永久滞留** |
| batch 2+（触发 DBO） | 从流另立门户 +1.8GB；主流因首轮已撑大不再扩张 |
| batch 9（进入 chunk_prefill） | OOM |

**机制二：首 batch 池膨胀不可逆**。首 batch 长序列不触发 DBO 分流，主流池被撑到单流理论峰值（~3.2GB）；之后即使 DBO 双流各自只需 1.7-1.8GB，**主流已扩张的 reserved 池不会收缩**，且从流还要再要一份——实际多占 ~1.8GB 冗余。

**机制三：ATB workspace 绕过内存池**。chunk_prefill 阶段 `torch._ops.atb.npu_ring_mla` 通过 `NPUWorkspaceAllocator`（PTA_WORKSPACE）申请 **240MB 连续物理块**，独立于 NPUCaching 内存池：
- 总空闲 500MB > 240MB，但碎片化后**找不到连续块**；
- 该路径不触发 `empty_cache` 重整，**直接 OOM**（torch 路径若此时加载大 tensor 反而能触发重整成功）。

**结论数据**：util 每降 0.01，物理空闲约 +600MB；0.94 → ~500MB 空闲偶发 OOM；**0.93 → ~1.1GB 空闲稳定**（覆盖 240MB 连续块 + 1.8GB 双流冗余的分摊 + 碎片余量）。

### 2.4 本节结论

| 变动来源 | 是否影响 KV cache | 实际吃哪块 | 后果 |
|---|---|---|---|
| 激活峰值超过 profiling 估计 | 否（KV 地址固定） | reserved 扩张 → device 空闲 | 吃穿空闲后 OOM（torch 路径有重整兜底） |
| DBO 多流各立内存池 | 否 | 每流独立的 reserved | 冗余滞留，等效抬高真实峰值 |
| 算子 workspace（PTA_WORKSPACE） | 否 | non-torch 直占 device，且需**连续块** | 无重整，碎片化下直接 OOM |
| HCCL 通信缓冲动态增长 | 否 | non-torch 直占 device | 同上类 |
| KV 不足（请求太多/序列太长） | —（这就是 KV 池本身） | 触发 preemption/排队 | 不 OOM，降吞吐 |

---

## 3. 显存预估的具体机制

### 3.1 启动时序全景

```
① init_device / MemorySnapshot               worker.py:453-455
     init_snapshot：初始 free/total/torch_peak
     requested_memory = total × gpu_memory_utilization
     （free < requested 直接报错拒绝启动）

② load_model                                 gpu_model_runner.py:388-532
     with DeviceMemoryProfiler() as m:       :394
         权重加载 + speculator(draft) 加载     （draft 权重计入 weights）
     model_memory_usage = m.consumed_memory   :427  → weights_memory
     之后：init_model_state、PP 接收缓冲等持久分配  :435, :528

③ determine_available_memory                 worker.py:556
     a. 若显式指定 --kv-cache-memory：仍跑 profile（触发编译），
        但直接采用指定值，跳过计算                :571-586
     b. with memory_profiling(init_snapshot, weights_memory):  :590
          见 3.2
     c. available_kv = requested_memory − non_kv_cache_memory :623

④ initialize_kv_cache                        worker.py:1159-1165
     按 available_kv 换算 block 数（向下取整、多 group 缩放 :714-738）

⑤ compile_or_warm_up_model                   worker.py:857-887
     warmup_sizes 逐档 _dummy_run              :879
     graph capture → npugraph_memory_bytes     :887（在 KV 之后、单独统计）
     打印"建议 --kv-cache-memory"日志            :889-917（含 150MiB 安全垫）
```

关键顺序：**峰值统计（③）在 graph capture（⑤）之前**——`worker.py:596-604` 注释明确："Record torch peak INSIDE the context and BEFORE graph capture, so that graph pool allocations don't inflate the activation peak"。graph 池单算并计入建议值。

### 3.2 memory_profiling：三类内存与三个指标

`memory_profiling`（`vllm/utils/mem_utils.py:228-297`）入口做三件事：`gc.collect()` + `empty_cache()` + **`reset_peak_memory_stats()`**（:284-286）。重置后 peak = 当前 allocated，因此：

```python
before_profile.torch_peak  = 权重 + 持久缓冲（load_model 后的全部存量）
profile_torch_peak         = before + dummy forward 的瞬态峰值
torch_peak_increase        = 瞬态峰值（激活预算）          ← 指标 A
non_torch_increase         = (device占用 − torch_reserved) 的增量   ← 指标 B
weights_memory             = model_memory_usage                ← 指标 C

non_kv_cache_memory = A + B + C                              worker.py:605-607
available_kv        = requested_memory − non_kv_cache_memory  worker.py:623
```

**逐量的确切算法**（谁减谁、在哪个口径上减）：

| 量 | 确切算法 | 代码 |
|---|---|---|
| 指标 C `weights_memory` | `DeviceMemoryProfiler` 的 enter/exit 各采一次 torch allocated 存量（NPU 实现：`reset_peak` 后取 `max_memory_allocated`，即当时存量），`consumed = exit − enter` = load_model 窗口内 allocated 净增量，主体就是权重张量（含 draft/LoRA）字节 | `model_runner.py:394-427`、`platform.py:206-208` |
| `before_profile.torch_peak` | 上下文入口先 `gc + empty_cache + reset_peak_memory_stats`，再 `measure()` 读 `allocated_bytes.all.peak`；reset 使 peak := 当时 allocated → = 权重 + 持久缓冲存量 | `mem_utils.py:284-294` |
| `profile_torch_peak` | `profile_run()` 刚结束、**仍在 with 内、graph capture 之前**读 `torch.npu.memory_stats()["allocated_bytes.all.peak"]`，即自 reset 以来 allocated 的最大值 | `worker.py:594-600` |
| 指标 A | `profile_torch_peak − before_profile.torch_peak`（显式 override mem_utils 通用式；graph 池尚未 capture，不会混入激活峰值） | `worker.py:602-604` |
| 指标 B | `after_profile.non_torch − **before_create**.non_torch`，其中每点 `non_torch = (total−free) − torch_reserved`；基线 `before_create` = `init_snapshot`，在 worker `__init__` 中、**分布式初始化之前**采集 | `mem_utils.py:152-159, 303-306`、`worker.py:452-454` |

两个容易踩的口径细节：

- **peak 是水位线**：`allocated_bytes.all.peak` 自上次 reset 起只增不落，dummy 结束后的 `gc + empty_cache` 只降 allocated/reserved、不回写 peak——所以 after_profile 事后测量照样拿得到瞬态峰值；
- **A 与 B 的基线不同**：A 的基线是 before_profile（profile 前一刻），B 的基线是 before_create（实例创建前）——B 比 A 多覆盖一段"init_device / HCCL / 分布式初始化"的 non-torch；其他进程的常量占用在 B 的相减中抵消（运行期间变动才会污染，`worker.py:614-622` 的 assert 就在防这个）。

**数值例子**（64GB 卡，`total`=62 GiB，util=0.90；为聚焦主线，假设 torch reserved ≈ allocated，忽略 segment 对齐差，单位 GiB）：

| 时刻 | 事件 | allocated | peak 水位线 | device 已用 | non_torch=已用−torch池 |
|---|---|---|---|---|---|
| t0 | `init_snapshot`（before_create，分布式初始化前）；卡上其他进程占 0.5 | 0 | 0 | 0.5 | 0.5 |
| t1 | init_device + HCCL/分布式：ACL context、communicator +0.8 | 0 | 0 | 1.3 | 1.3 |
| t2 | `load_model`（指标 C 的窗口 t1→t2）：权重+draft 张量 +20.0 | 20.0 | 20.0 | 21.3 | 1.3 |
| t3 | 持久缓冲（input buffers、PP 接收缓冲、sampler 状态）+0.7 | 20.7 | 20.7 | 22.0 | 1.3 |
| t4 | 进入 `memory_profiling`（`gc+empty_cache+reset_peak` → peak:=20.7），测得 before_profile | 20.7 | 20.7 | 22.0 | 1.3 |
| t5 | dummy forward 峰值：瞬态激活 +4.5；走到的算子 workspace + MC2 预热缓冲等 non-torch +0.6 | 25.2 | 25.2 | 27.1 | 1.9 |
| t6 | 退出上下文（`gc+empty_cache`：激活释放、**peak 不回落**），测得 after_profile | 20.7 | **25.2** | 22.6 | 1.9 |

由表读出五个量：

```text
C   = weights_memory         = allocated(t2) − allocated(t1)             = 20.0
before_profile.torch_peak    = t4 reset 后的 peak（权重+持久缓冲存量）    = 20.7
profile_torch_peak           = t5 的水位线（with 内、graph capture 前读） = 25.2
A   = torch_peak_increase    = 25.2 − 20.7                               = 4.5
B   = non_torch_increase     = non_torch(t6) − non_torch(t0) = 1.9 − 0.5 = 1.4
                              （= init 0.8 + profiling 期 0.6；其他进程 0.5 相减抵消）
non_kv_cache_memory = A + B + C                                           = 25.9
available_kv = util×total − non_kv = 0.90 × 62 − 25.9                      = 29.9 GiB
```

**缺口④的数值体现与上游对照**：t3 的持久缓冲 0.7——在 C 的窗口外（t2 时窗口已关闭）、在 A 的基线内（t4 时已是存量）、属 torch 侧不进 B——A+B+C 三项谁都没扣它：运行期实际 non-KV 存量 = 20.0 + 0.7 + 1.4 = 22.1，预算却只扣 25.9，KV 预算虚高 0.7 GiB。对照上游 GPU worker：它沿用 mem_utils 通用式 `non_kv = total_consumed + transient_peak_headroom`（`mem_utils.py:309-321`），其中 `total_consumed = before_create.free − after.free` 为 **device 全量净增**（本例 22.1，含持久缓冲），算出 26.6，没有此缺口；vllm-ascend 在 `worker.py:605-607` 改写为 A+B+C 重组，正是丢掉"load_model 之后、profiling 之前"这段 torch 分配的直接原因。另一差异：上游还会在 available_kv 中显式扣 graph 池先验估计（`gpu_worker.py:588-592, 621-624`，opt-in），NPU 侧则是 capture 之后单独统计（缺口⑤）。

三项指标对应的正是 1.2 表中的类别 3（激活）、5a（non-torch）、5b 中 dummy run 已触发算子的 workspace（同由指标 B 覆盖）、1（权重）。**类别 4（持久缓冲）、5b 中未被触发的 workspace（如 attn）、5c（graph 池）、6（滞留）不在公式内**——这是预估与实际的全部差距来源。

### 3.3 dummy run：形状与覆盖范围

`profile_run → _dummy_run(self.max_num_tokens, skip_attn=True, is_profile=True)`（`gpu_model_runner.py:955`）：

| 维度 | 行为 | 代码 |
|---|---|---|
| batch 形状 | `max_num_tokens = max_num_batched_tokens`（如 8192），`num_reqs = min(·, max_num_reqs)` 均匀分布（16×512） | :219, :793-809 |
| 执行模式 | **eager**（`need_eager=is_profile`），模型 forward 完整真实执行 | :1693, :1948 |
| attention | `skip_attn=True`：跳过 attention 元数据，attention 相关运行期 workspace **不被触发** | :1772 |
| PP | 所有 PP rank 各自跑本 stage forward（模型边界代码真实执行，含 SP gather）；但**绕过 Worker 层，PP send/recv 通信不发生** | :822-824, :846-848 |
| 投机解码 | 末 stage dummy 跑 speculator.propose，draft 激活被覆盖 | :862-912 |
| 采样 | 末 stage `_dummy_sampler_run`；注释明示可能低估 top_k/top_p/logprobs | :918-930 |
| MoE 通信 | MRV2 override：先用 `mc2_tokens_capacity` 额外跑一次预热，**提前分配 MC2 的 HCCL 缓冲**——这正是"运行期通信缓冲低估"的一个官方补丁先例 | vllm-ascend `model_runner.py:357-374` |
| 收尾 | `gc.collect()`；profile 阶段 reserved 池的扩张随后被 `empty_cache` 清理（内部案例表格可证） | :970 |

MRV1 对照：PP+SP（`enable_sp()`）时 dummy run 显式将 intermediate token 数 **÷ tp_size**，注释直言"否则非首 rank 会多做一次 all-gather，导致内存估算错误甚至 OOM"（`model_runner_v1.py:4287-4304`）——说明"profiling 形状 ≠ 运行期形状 → 估算偏差 → OOM"这类问题在 MRV1 已被踩过并修过。

### 3.4 预算公式汇总

```
KV 预算 = util × 总显存 − (权重 + 激活峰值增量 + non-torch 增量)

运行期真实需求 = 权重 + KV + 激活真实峰值 + 持久缓冲
              + non-torch 真实值（HCCL/ACL + **全部**算子 workspace，
                含运行期才触发申请的部分）
              + graph 池 + allocator 滞留

健康条件：运行期真实需求 ≤ 总显存 − 系统余量
```

两个预算的差 = 预估缺口 + 人为余量。util 拉得越高，留给缺口的缓冲越小。

### 3.5 预估的已知缺口

| # | 缺口 | 机制 | 实证 |
|---|---|---|---|
| ① | DBO/多流叠加 | dummy run 单流跑，双流内存池独立不可复用 | 内部案例 batch2 从流 +1.8GB |
| ② | 首 batch 池膨胀滞留 | reserved 只增不缩，单流峰值永久占据 | 内部案例 batch1 滞留 ~0.5-1.4GB |
| ③ | 运行期算子 workspace（dummy run 未触发部分） | workspace 同属 non-torch，走到的算子已计入指标 B；`skip_attn=True` 使 attn 算子未被触发，运行期首次执行才申请。PTA 独立通道不是漏估原因，只决定申请时无重整、直接 OOM | `npu_ring_mla` 240MB 连续块 |
| ④ | 持久缓冲记账缝隙 | `weights_memory` 在 :427 捕获，其后 :528 分配的 PP 接收缓冲既不在 weights 也不在 peak 增量里；PP+DSpark 的 aux 接收缓冲（每个 `(2+N_aux)×max_tokens×hidden` ≈ 数百 MB/rank）同此——上游通用式（`total_consumed` 含它）无此缺口，vllm-ascend 的 A+B+C 重组（`worker.py:605-607`）是直接成因（数值例子见 3.2） | PP 接收缓冲代码路径推演（`gpu_model_runner.py:528`）；K3+DSpark 特性背景见 vLLM 官方博客（附录 B） |
| ⑤ | graph 池 | KV 分配后才 capture，从 device 剩余里出 | `worker.py:885-903` 单独统计 |
| ⑥ | non-torch 只计增量 | 通信缓冲按建立时增量算，运行期按 batch 形状动态变化的 HCCL buffer 不覆盖 | MC2 预热补丁（3.3）即为佐证 |
| ⑦ | 采样/logprobs 低估 | dummy sampler 跳过 top_k/top_p/logprobs 路径 | `gpu_model_runner.py:926-928` 注释 |
| ⑧ | 碎片化 | 预算字节数 ≠ 连续块可用性；PTA_WORKSPACE 无重整 | 内部案例 500MB 空闲装不下 240MB |

### 3.6 util 阈值的实证：为什么是 0.93

```
util 0.94 → 物理空闲 ~500MB  → 240MB 连续块 + 双流冗余 → 偶发 OOM
util 0.93 → 物理空闲 ~1.1GB  → 稳定
每降 0.01 ≈ 释放 600MB 给 KV 的预算外空间
```

本质：`KV 预算 = util×total − non_kv`，util 降 0.01 时 KV 少 ~640MB（64GB 卡），物理空闲多 ~600MB。当预估缺口（3.5 各项之和）大于剩余空闲时，OOM 只是时间问题。**0.93 不是普适常数，而是"该模型的缺口总和 ≈ 0.07×64GB"的经验值**；换模型/换拓扑（如 PP+DSpark 引入缺口④）应重新校准。

---

## 4. OOM 排查工具速查

| 工具 | 用途 | 关键操作 |
|---|---|---|
| torch profiler（含内存） | 看"算子保留 vs 进程保留"变化，区分泄漏与池扩张 | `export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=1`、`VLLM_TORCH_PROFILER_WITH_STACK=1`、`VLLM_TORCH_PROFILER_DIR=...`（放启动命令**最后**） |
| memory snapshot | 时序化 allocated/reserved，定位池扩张点与峰值叠加 | `torch_npu.npu.memory._record_memory_history(stacks='python')` + `_dump_snapshot()`，拖入 [pytorch memory_viz](https://docs.pytorch.org/memory_viz)；打桩点：load_model 前 start / initialize_kv_cache 后 stop / execute_model 计数 stop / OOM except 里 stop |
| msMemScope（MindStudio） | 按内存池（NPUCaching vs PTA_WORKSPACE 等）拆分申请，找绕过 allocator 的 workspace | 开源工具，提供泄漏检测/内存拆解/内存对比，打点 `.db` 可视化：[gitcode.com/Ascend/msmemscope](https://gitcode.com/Ascend/msmemscope) |
| npu-smi / 启动日志 | 总量水位、KV capacity、peak activation、建议 kv-cache-memory | 日志含 weights/peak activation/non-torch/NPU graph memory 分类打印（`worker.py:907-917`） |

打桩点见上表 memory snapshot 行；snapshot 采集与分析见 [PyTorch torch_cuda_memory 文档](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)，msMemScope 的采集用法见其开源仓库。

---

## 5. 实践建议

1. **util 不拉满**：为 3.5 的缺口留缓冲。经验起点 0.90-0.93，出现"benchmark 后单请求 OOM"即下调 0.01 观察（每档 ≈ 600MB）。
2. **新拓扑先实测水位**：换模型/PP/EP/DSpark/DBO 组合后，用 snapshot 观察前 10 个 batch 的 reserved 增长，确认稳定态峰值再定 util。
3. **优先用 `--kv-cache-memory` 显式控制**：启动日志会给出建议值（`worker.py:889-917`，含 150MiB 安全垫），比单纯调 util 更可复现。
4. **长稳测试必须覆盖 chunk_prefill 与并发爬坡**：内部案例的 OOM 全部发生在 chunk_prefill 阶段的 workspace 申请，且依赖 batch 数积累。
5. **怀疑碎片化时看连续性而非总量**：空闲 500MB ≠ 能分配 240MB 连续块；torch 路径有重整兜底，PTA_WORKSPACE 没有。
6. **多流特性（DBO 等）按"每流一份峰值"预估**，并注意首 batch 单流峰值会把主流池永久撑大。

---

## 附录 A：关键代码索引

代码位置均可在公开仓库查证（行号基于撰写时的工作副本，**随版本漂移**，建议按函数/类名检索）：vllm → [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)；vllm-ascend → [github.com/vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)。

| 机制 | 位置 |
|---|---|
| requested_memory / init_snapshot | `vllm_ascend/worker/worker.py:453-455` |
| determine_available_memory 全流程 | `vllm_ascend/worker/worker.py:556-626` |
| non_kv_cache_memory / available_kv 公式 | `vllm_ascend/worker/worker.py:605-607, 623` |
| memory_profiling（reset peak / 三类内存模型） | `vllm/utils/mem_utils.py:228-297` |
| MemorySnapshot（torch_peak/reserved/non_torch 采集） | `vllm/utils/mem_utils.py:104-160` |
| load_model / weights 计量 / 持久接收缓冲 | `vllm/v1/worker/gpu/model_runner.py:388-532` |
| profile_run → dummy run（max_num_batched_tokens） | `vllm/v1/worker/gpu/model_runner.py:938-970` |
| _dummy_run（形状构造 / PP 切片 / 输出丢弃） | `vllm/v1/worker/gpu/model_runner.py:775-916` |
| dummy execute_model（eager 真实 forward） | `vllm/v1/worker/gpu/model_runner.py:1631-1993` |
| MRV2 profile_run override（MC2 预热） | `vllm_ascend/worker/v2/model_runner.py:357-374` |
| MRV1 PP+SP ÷tp_size 修正 | `vllm_ascend/worker/model_runner_v1.py:4287-4304` |
| KV cache 分配 | `vllm_ascend/worker/worker.py:1159-1165` |
| graph capture / 建议 kv-cache-memory 日志 | `vllm_ascend/worker/worker.py:857-917` |

## 附录 B：参考资料（公开可查）

**源码仓库**

- vLLM：[github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)——`vllm/utils/mem_utils.py`（MemorySnapshot / memory_profiling）、`vllm/v1/worker/gpu_worker.py`（determine_available_memory 通用式）、`vllm/v1/worker/gpu/model_runner.py`（load_model / profile_run / _dummy_run）
- vLLM-Ascend：[github.com/vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)——`vllm_ascend/worker/worker.py`（NPU 侧 A+B+C 重组、150MiB 安全垫）、`vllm_ascend/worker/v2/model_runner.py`（MC2 预热）
- Ascend Extension for PyTorch（torch_npu，NPUWorkspaceAllocator / `torch.npu.memory_stats`）：[gitee.com/ascend](https://gitee.com/ascend)
- 昇腾开源生态索引（含 vLLM-Ascend、msMemScope）：[ascend.github.io/docs](https://ascend.github.io/docs/)

**官方文档**

- vLLM 文档（`gpu_memory_utilization`、`--kv-cache-memory`、prefix caching）：[docs.vllm.ai](https://docs.vllm.ai/)
- PyTorch CUDA 内存管理（caching allocator、allocated/reserved）：[docs.pytorch.org/docs/stable/notes/cuda.html](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- torch.cuda memory snapshot / stats API：[docs.pytorch.org/docs/stable/torch_cuda_memory.html](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)
- PyTorch memory_viz 可视化：[docs.pytorch.org/memory_viz](https://docs.pytorch.org/memory_viz)
- CANN / HCCL 文档：[hiascend.com/document](https://www.hiascend.com/document)
- MindStudio 内存检测 msMemScope：[gitcode.com/Ascend/msmemscope](https://gitcode.com/Ascend/msmemscope)

**背景阅读**

- vLLM 官方博客：Kimi K3 day-0 支持（KDA / DSpark / PD 分离，2026-07）：[vllm-project.github.io/_posts/2026-07-27-k3.md](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-07-27-k3.md)

**内部实证说明**：2.3 的案例复盘、3.6 的 util 阈值及 240MB workspace 等实测数据，来自一起内部生产部署（DSR1）的 profiling 复盘（torch profiler + memory snapshot + msMemScope），未公开发布；文中引用的代码机制均可在上述公开仓库对应位置查证。
