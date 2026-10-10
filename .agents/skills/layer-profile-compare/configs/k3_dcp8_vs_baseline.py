# Worked example config: K3 13层裁剪模型 DCP=8 vs baseline（Round4 cc6 两轮 profile）。
# 这是 layer-profile-compare 的参照实现配置，复制本文件并按新任务改 RUNS/SCOPES 等段。
# 路径相对 workspace 根（~/workspace），也可写绝对路径。

RUNS = [
    dict(name='dcp8',
         raw_ops='task/kimi-k3-mainline-reduce-layers/profiling/perf-breakdown/raw_ops.json',
         details='task/kimi-k3-mainline-reduce-layers/profiling/perf-breakdown/raw_ops_details.json',
         structure='task/kimi-k3-mainline-reduce-layers/profiling/perf-breakdown/structure_draft.json',
         spec='task/kimi-k3-mainline-reduce-layers/profiling/perf-breakdown/kimi_k3_reduced_spec.json',
         kernel_details_csv='task/kimi-k3-mainline-reduce-layers/profiling/dcp8-cc6/'
                            'dcp8_cc6_rank7_profile/ASCEND_PROFILER_OUTPUT/kernel_details.csv'),
    dict(name='baseline',
         raw_ops='task/kimi-k3-mainline-reduce-layers/profiling/perf-breakdown/raw_ops_baseline.json',
         details='task/kimi-k3-mainline-reduce-layers/profiling/perf-breakdown/raw_ops_baseline_details.json',
         structure='task/kimi-k3-mainline-reduce-layers/profiling/perf-breakdown/structure_draft_baseline.json',
         spec='task/kimi-k3-mainline-reduce-layers/profiling/perf-breakdown/kimi_k3_reduced_spec_baseline.json',
         kernel_details_csv='task/kimi-k3-mainline-reduce-layers/profiling/baseline-cc6/'
                            'base1_cc6_rank7_profile/ASCEND_PROFILER_OUTPUT/kernel_details.csv'),
]

# component type -> 展示模块名
MODULE_OF = {
    'kda': 'KDA', 'kda_first': 'KDA',
    'mla': 'MLA', 'mla_last': 'MLA',
    'moe': 'MOE', 'dense_mlp': 'MLP',
    'draft_attn': 'DRAFT_MLA', 'draft_ffn': 'DRAFT_FFN',
}
# draft/MTP 侧模块集合（层标签前缀 D，其余 L）
DRAFT_MODS = {'DRAFT_MLA', 'DRAFT_FFN'}
# sheet 内模块块顺序
MODULE_ORDER = ['DRAFT_MLA', 'DRAFT_FFN', 'draft→target 交接', 'KDA', 'MLA', 'MOE', 'MLP']

# 每模块代表层：(组件类型, layer_idx)；特殊区段整段：('gap', from_modules, to_modules|None)
# K3 结构：kda L1,2,4,5,6,8,9,10 + kda_first L0；mla L3,7,11 + mla_last L12；
# moe L1-L12；dense_mlp L0；draft_attn/draft_ffn L13-17（MTP，每层 FFN 尾一条小 AR）
SCOPES = [
    ('DRAFT_MLA', ('draft_attn', 13)),
    ('DRAFT_FFN', ('draft_ffn', 13)),
    ('draft→target 交接', ('gap', ('DRAFT_MLA', 'DRAFT_FFN'), None)),
    ('KDA', ('kda', 1)),
    ('MLA', ('mla', 3)),
    ('MOE', ('moe', 1)),
    ('MLP', ('dense_mlp', 0)),
]

SHARD = {
    ('dcp8', 'DRAFT_MLA'): 'TP8+DCP8', ('baseline', 'DRAFT_MLA'): 'TP8',
    ('dcp8', 'DRAFT_FFN'): 'TP8', ('baseline', 'DRAFT_FFN'): 'TP8',
    ('dcp8', 'draft→target 交接'): 'TP8', ('baseline', 'draft→target 交接'): 'TP8',
    ('dcp8', 'KDA'): 'TP8+SP', ('baseline', 'KDA'): 'TP8+SP',
    ('dcp8', 'MLA'): 'TP8+SP+DCP8', ('baseline', 'MLA'): 'TP8+SP',
    ('dcp8', 'MOE'): 'EP8+SP', ('baseline', 'MOE'): 'EP8+SP',
    ('dcp8', 'MLP'): 'TP8', ('baseline', 'MLP'): 'TP8',
}

# 优化思路列（sparse 标注，仅每组首个实例）：((模块, cluster|None, 名称子串), run 过滤, 文案)
IDEA_RULES = [
    (('draft→target 交接', None, 'hcom_allReduce'), None,
     '同步屏障（embedding TP 求和）：时长绝大部分是等 draft 区最慢 rank（a2a 串行化偏斜）；'
     '削偏斜来源可收回 ~4ms/步'),
    (('DRAFT_MLA', None, 'hcom_alltoall'), None,
     'draft 侧 DCP a2a 交换税（~200us/层×5）；draft 退出 DCP 或 a2a 融合可省 ~1.4ms/步'),
    (('DRAFT_MLA', 'tree_merge', None), 'dcp8',
     'DCP 树形布局 merge（AttentionUpdate+Slice），可与 prep 胶水融合'),
    (('MLA', 'dcp_pack', None), 'dcp8',
     'DCP 打包 kernel，可与 lse_combine 融合（两者合计 ~127us/层×4）'),
    (('MLA', 'lse_combine', None), 'dcp8',
     'LSE 合并，可与 dcp_pack 融合（两者合计 ~127us/层×4）'),
    (('MLA', 'head_pad', None), 'baseline',
     '无 DCP 时 12→16 头填充（MemSet/PadV3 47.5us/层）；不 pad 路径可消除'),
    (('MOE', None, 'DequantSituQuant'), None,
     'AIV 反量化是 MoE 内最大单项；已做 round-robin 负载均衡（−17%），可探索进一步融合'),
    (('MOE', 'experts_gemm', 'GroupedMatmul'), None,
     'MC2 两次 F.pad 已改静态零块 cat（−53%）；残留 4 对 pad/step 可继续消除'),
]

# 时序断言（ground truth 链，顺序不要求相邻；首次人工核对 kernel_details 后固化）。
# 注意两侧 RmsNorm 后 DQ/rS 相邻次序跨轮翻转（±2us 辅流抖动），链按各自实际顺序写：
SEQUENCE_ASSERTS = [
    ('baseline', 'MOE', ['MatMulV2', 'MoeGatingTopK', 'MoeDistributeDispatch',
                         'GroupedMatmul', 'DequantSituQuant', 'MoeDistributeCombine',
                         'RmsNorm', 'hcom_reduceScatter', 'DynamicQuant']),
    ('dcp8', 'MOE', ['MatMulV2', 'MoeGatingTopK', 'MoeDistributeDispatch',
                     'GroupedMatmul', 'DequantSituQuant', 'MoeDistributeCombine',
                     'RmsNorm', 'DynamicQuant', 'hcom_reduceScatter']),
]

OUTPUT = 'task/kimi-k3-mainline-reduce-layers/profiling/perf-breakdown/agg_result.json'

# ---- 构建段（build_xlsx.py 读取；表注自动部分由脚本生成，这里只放任务专属内容）----
BUILD = dict(
    xlsx_out='task/kimi-k3-mainline-reduce-layers/profiling/perf-breakdown/'
             'k3_op_summary_dcp8_vs_baseline.xlsx',
    titles={
        'dcp8': 'K3 13层裁剪模型 Decode 算子明细（每模块代表层）— DCP=8（step 70，逐实例）',
        'baseline': 'K3 13层裁剪模型 Decode 算子明细（每模块代表层）— Baseline 无 DCP（step 96，逐实例）',
    },
    extra_notes={
        'dcp8': [
            '数据出处：DCP=8 = step 70，取自 Round4 wave-1 纯 decode 5s profile 窗口（rank7/dp0，'
            'random 40k/700 cc=6，TP8/DP2/EP8）中段，服务与图捕获热身均已完成。',
            'draft 共 5 层（MTP layer_idx 13-17），每层 FFN 末尾一条小 allReduce（fc2→AR→AddRmsNormBias）；'
            '后续层（D16/D17）该 AR 膨胀至 136-605us（DCP 偏斜税），本表未展开。',
            '『draft→target 交接』= draft 头投影/采样/embedding/同步 AR 区，巨型 allReduce 为 embedding TP '
            '求和同步屏障（5979us，时长以等待为主）。',
            '全步模块级小计参考（us，含归属通信）：DRAFT_MLA 5586/DRAFT_FFN 1068/交接 7776/KDA 3192/'
            'MLA 3065/MOE 9577/MLP 209 vs baseline 1978/572/3248/3074/2470/9470/214。',
        ],
        'baseline': [
            '数据出处：Baseline = step 96（算子分布众数步），取自 Round4 wave-1 纯 decode 5s profile 窗口'
            '（rank7/dp0，random 40k/700 cc=6，TP8/DP2/EP8）中段，热身已完成。无 DCP：MLA 12 头 pad 到 16。',
            'draft 共 5 层（MTP layer_idx 13-17），每层 FFN 末尾一条小 allReduce（fc2→AR→AddRmsNormBias）。',
            '『draft→target 交接』= draft 头投影/采样/embedding/同步 AR 区，巨型 allReduce 为 embedding TP '
            '求和同步屏障（1826us，等待为主）。',
            '全步模块级小计参考（us）：见 dcp8 表注末条。',
        ],
        'comparison': [
            '模块速读（选定层，us）：DRAFT_MLA D13 668 vs 380（draft DCP 税）、MLA L3 807 vs 625'
            '（DCP pack/lse/a2a）、KDA L1 283 vs 281 / MOE L1 866 vs 885 / MLP L0 209 vs 214（持平）、'
            'DRAFT_FFN D13 114 vs 119、交接 7776 vs 3248（巨型同步 AR 5979 vs 1826）。',
        ],
    },
)
