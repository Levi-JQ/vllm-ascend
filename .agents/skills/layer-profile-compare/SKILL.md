---
name: layer-profile-compare
description: 两轮 NPU profile（A/B 实验，如 DCP/SP/EP 开关、优化前后对比）按模块抽取代表层做算子级逐实例对比的完整方法论与脚本：层窗提取、跨轮配对、kernel_details.csv 逐行对查硬校验、Excel 明细+对比表输出。用户提到「取一层对比 / 各模块取一层算子对比 / 从 kernel_details 取某层耗时 / 某层算子 A/B / 对比 DCP 前后某模块」时触发。前置：已有单步 profile 数据（perf-breakdown 的 raw_ops/structure/spec，或等价的组件切分+单步算子表）。
---

# 层级 profile 抽取与 A/B 对比

对两轮 profile（run A vs run B）回答"每个模块一层里到底发生了什么、A 比 B 多/少了哪些算子、各贵多少"。
核心纪律：**逐实例、无平均；每行都能在 kernel_details.csv 里 grep 到**。

## 前置产物（每轮一份，路径相对 workspace 根）

| 文件 | 内容 |
|---|---|
| `raw_ops*.json` | 单步算子表：operators[].{index, start_time_us, duration_us, normalized_name, input_shapes} |
| `raw_ops_*_details.json` | 同序号顺序：{index, name(=csv Name 原值), type, duration_us, wait_time_us, 26 项 aic/aiv 指标} |
| `structure_draft*.json` | op_to_component + components[].{component_id, type, layer_idx, phase, op_indices} |
| `*_spec.json` | component_clusters：每组件类型的算子分桶规则（cluster） |
| 原始 `kernel_details.csv` | 对查 ground truth（含 Name/Type/Duration(us)/Start Time(us)，全窗口几十万行） |

来源：`/model-infer-perf-breakdown` 的 runs/ 输出 + analyze_kernels 单步提取。单步必须是**稳态步**
（5s 窗口中段、热身完成；若同窗口多步，取算子分布众数步），并在表注写明 step 出处。

## 方法论（六步）

### 1. 组件清点
dump 全部 components 的 (type, layer_idx, phase, n_ops)。**不要假设 phase 取值**——见过
`'mtp_head'` 而非 `'draft'`，按猜测写分支会让整块区域层标签全空。层标签规则：draft/MTP 侧
`D<idx>`、target 侧 `L<idx>`、非层内区段 `—`。

### 2. 代表层选择
重复结构（同 type 多层）每模块只取**一个代表层**——通常取该 type 的首个常规层。结构变体层
（块写入层 kda_first/mla_last、末层）排除但在表注记录；特殊区段（如 draft→target 交接、
模块间边界）无层概念，**整段保留**。代表层需写进 config `SCOPES` 并在表注明示。

### 3. 层窗提取
每层时间窗 = [该层组件首算子 start, 末算子 end]。归属规则两条，顺序不可换：

1. **匹配算子恒归属自己组件的 scope**——层间接缝有 ~0.2us 流重叠（下一层首算子可早于本层
   末算子结束启动），纯窗包含会把邻层算子偷进来；
2. 未匹配算子（通信/胶水）按窗包含归属（多个窗命中取先定义的）。

边界区段窗 = [前段模块最后算子 end, 后段模块首个算子 start)。通信行保留 hcom 汇总行
（AivKernel 展开行在 analyze_kernels 阶段已去重，不重复计数）。

### 4. 跨轮配对（逐实例）
配对键 = 模块 + 层 + 算子名 + **层窗内第 k 次出现**。通信算子名含任务号
（`hcom_allReduce__503_1471_1`）**跨轮不同，绝不能做配对键**——必须剥成类型
（`hcom_allReduce`）再按层内序号配对，否则相同算子全部变成连续的"仅基线/新增"假象。
单边行：`新增`（A 特有）/`仅基线`（B 特有），逐条给出解释（真实新增算子，如 DCP 引入的
pack/lse/a2a）。

### 5. 硬校验门（脚本内置，全过才准出数）
`agg_and_verify.py` 按序执行，任一失败 `exit 1`：
1. **csv 对查**：全部行的 (Start Time, Duration) 逐行存在于对应 kernel_details.csv
   （浮点 round 2 位后比对）；
2. 小计一致：每模块 count/Σ 与明细行吻合；
3. 对比表 Σ = 各轮 shown total；
4. **时序断言**：对 ground-truth 链（用户贴过的片段或首次人工核对结果）做顺序校验
   （顺序不要求相邻）；链写进 config `SEQUENCE_ASSERTS`，两侧各写各的实际顺序——相邻
   辅流算子 ±1-2us 次序跨轮翻转是真实现象，不是 bug。

### 6. Excel 输出
`build_xlsx.py` 生成四表：run A / run B 明细（模块→切分策略→层→算子→优化思路→
Start Time→Duration/个数/总时间/占比/Wait+25 项指标）+ 算子对比（**两侧 Start Time 都给**，
可分别 grep 两份 csv）+ Review 活公式校验。表注必备：step 出处与部署形态、代表层口径、
未展开范围、接缝/抖动说明、占比分母。Start Time 一律存**文本列**（16 位数值超出 Excel
15 位有效数字会静默截断）。构建后按 `/xlsx` skill 跑 validate/audit/scan。

## 脚本用法

```
cp .zcode/skills/layer-profile-compare/configs/k3_dcp8_vs_baseline.py /tmp/<task>_cfg.py
# 改 RUNS（两轮五件套路径）/ MODULE_OF / DRAFT_MODS / MODULE_ORDER / SCOPES / SHARD，
# 可选 IDEA_RULES（优化思路列）、SEQUENCE_ASSERTS、BUILD（xlsx 路径/标题/任务专属表注）
python3 .zcode/skills/layer-profile-compare/scripts/agg_and_verify.py /tmp/<task>_cfg.py
python3 .zcode/skills/layer-profile-compare/scripts/build_xlsx.py /tmp/<task>_cfg.py
```

相对路径按 workspace 根解析。agg 产物 `agg_result.json`（含全步逐实例明细）随 xlsx 一起归档到
任务 profiling 目录，progress.md 记录位置。

## 已知坑（全部实测踩过）

| 坑 | 后果 | 规则 |
|---|---|---|
| 平均（跨层/跨实例/组开始时间） | 同名算子多次出现被合并、均值落中间打乱时序 | 一律逐实例，无平均 |
| splits/kernels.csv 行序 | 文件行序不是时序序 | 展示前必按 Start Time(us) 自行排序 |
| 通信原始任务号做配对键 | 跨轮不同→相同算子变连续单边行 | 类型+层内序号配对 |
| 假设组件 phase 取值 | 实际 'mtp_head' → draft 区层标签全空 | 先 dump 实际值 |
| 纯窗包含归属 | 接缝 0.2us 流重叠偷邻层算子 | 匹配算子恒归本组件 |
| 辅流算子 ±1-2us 次序翻转 | 看似时序矛盾 | 真实数据如实呈现，断言链按各自顺序写 |
| Start Time 存数值 | Excel 15 位有效数字截断 | 存文本列 |
| (start,dur) 直接比 | 浮点/字符串形态不一致 | round(,2) 后比对 |

## 参照案例

`task/kimi-k3-mainline-reduce-layers/`（K3 13 层裁剪模型 DCP=8 vs baseline，Round4 cc6 两轮）：
config 即 `configs/k3_dcp8_vs_baseline.py`；产物
`profiling/perf-breakdown/{agg_result.json, k3_op_summary_dcp8_vs_baseline.xlsx}`；
演进史与全部教训见该任务 `progress.md` Round5.1。

已知限制：仅支持两轮对比；多于两轮需两两运行或扩展脚本。
