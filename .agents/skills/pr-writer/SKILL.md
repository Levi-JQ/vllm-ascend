# ---
# name: pr-writer
# description: 根据 commit 撰写 vLLM-Ascend / vLLM 的 PR 描述文档并保存为 PR 描述文件。当用户要提交 PR、推送 PR 到远端仓库（push 分支 / gh pr create / 准备 PR 材料）时自动触发，先根据 commit 生成 PR 描述再执行提交/推送。触发场景：提交 PR、推送 PR 到远端仓库、撰写 PR 描述、PR 提交前整理 commit 内容、用户提到"写 PR"、"PR 描述"、"提 PR"、"推 PR"、"push 后提 PR"。
# ---

# PR Writer — 根据 commit 生成 PR 描述

根据指定 commit（一个或多个）撰写 PR 描述文档，落盘为 PR 描述文件（默认当前任务目录或仓库根目录的 `pr-description.md`），供用户提交 PR 时直接粘贴。

## PR 标题规范

标题必须带类型前缀，格式 `<前缀> <描述>`，多类别用多个前缀拼接。前缀**只能**选自 CI 工作流（PR title check）的白名单：

| 前缀 | 含义 |
|---|---|
| `[Feature]` | 新功能 |
| `[BugFix]` | 错误修复 |
| `[Performance]` | 性能优化 |
| `[Refactor]` | 重构（不改行为） |
| `[Test]` | 测试（如单元测试） |
| `[CI]` | 构建或 CI 改进 |
| `[Doc]` | 文档修复与改进 |
| `[Community]` | 社区事务（如 contributor 说明） |
| `[Misc]` | 不属于上述类别（谨慎使用） |

**注意**：白名单以 CI 校验为准，文档页可能滞后。`[Feat]`（须写全 `[Feature]`）以及 `[Kernel]`/`[Core]`/`[Attention]`/`[Communicator]`/`[Platform]`/`[Worker]` 等细分类前缀**不在白名单内**，用了会挂 `check-pr-title`（::error::PR title must contain one of the following prefixes: ...）。前缀也可以出现在标题末尾（如 `Add new feature [Feature]`）。

示例：`[Feature] dspark: replicate markov head via ReplicatedGroup`、`[Performance][BugFix] fix dequant load imbalance`。

规范来源：vllm-ascend CI `check-pr-title` 工作流实测白名单（2026-09-10）：
`[BugFix], [Performance], [Test], [CI], [Feature], [Doc], [Misc], [Community], [Refactor]`

## PR 描述模板（严格遵守，不增删 section）

```markdown
<!--  Thanks for sending a pull request!

BEFORE SUBMITTING, PLEASE READ https://docs.vllm.ai/en/latest/contributing/overview.html

-->
### What this PR does / why we need it?
<!--
- Please clarify what changes you are proposing. The purpose of this section is to outline the changes and how this PR fixes the issue.
If possible, please consider writing useful notes for better and faster reviews in your PR.

- Please clarify why the changes are needed. For instance, the use case and bug description.

- Fixes #
-->

### Does this PR introduce _any_ user-facing change?
<!--
Note that it means *any* user-facing change including all aspects such as API, interface or other behavior changes.
Documentation-only updates are not considered user-facing changes.
-->

### How was this patch tested?
<!--
CI passed with new added/existing test.
If it was tested in a way different from regular unit tests, please clarify how you tested step by step, ideally copy and paste-able, so that other reviewers can test and check, and descendants can verify in the future.
If tests were not added, please describe why they were not added and/or why it was difficult to add.
-->
```

## 自动触发时机

用户要求**提交 PR** 或**推送 PR 到远端仓库**时（如 "push 分支并提 PR"、"推到远端准备提 PR"、执行 `git push` 后要走 PR 流程），本 skill 自动触发：先用下述流程根据 commit 生成 PR 描述文档，再配合用户完成推送/提交（如 `git push`、`gh pr create` 或产出 pr-description.md 供用户手动提交）。

## 工作流程

1. **收集 commit 信息**：让用户指定 commit（哈希/分支/范围），或从上下文推断。用 `git show --stat <commit>`、`git log`、`git show <commit>` 阅读每个 commit 的改动与提交信息。多 commit PR 需综合提炼，逐个看 diff 而不是只看 commit message。
2. **写标题**：按上文前缀规范拟定（**只用 CI 白名单前缀**）；从 commit 的改动位置推断类别（新功能 → `[Feature]`，性能优化 → `[Performance]`，修 bug → `[BugFix]`，重构 → `[Refactor]`）。
3. **填充模板三段**，内容根据 commit 提炼，**尽可能简洁**：
   - **What this PR does / why we need it**：改了什么（机制/方案一两句话说清）+ 为什么需要（优化动机如性能数据、bug 现象）。内部环境/机器名等敏感信息不要写入。若关联 issue，替换 `Fixes #` 为实际编号；无则删除该行。
   - **user-facing change**：如实回答。纯内部优化/重构填 `No.`；改了行为、接口、参数则简述变化。
   - **How was this patch tested**：写明测试方式（新增/已有单测及运行命令、A/B 验证方法、性能对比数据等），给出可复现命令；若未加测试，说明原因。
4. **落盘**：写入 PR 描述文件。优先放任务目录（如 `task/<task-name>/pr-description.md`）；非任务场景放仓库根目录 `pr-description.md`。文件首行放标题（`# 标题` 或 `Title: 标题`），正文为完整模板填充结果。
5. **追加中文翻译版本**：文件末尾加一节分隔线 `---`，标题 `## 中文翻译`，给出标题及三个 section 内容的完整中文对照（模板注释无需重复翻译）。该附录仅供内部参考，提交到 GitHub 时不要包含。
6. **输出摘要**：向用户展示标题与三段要点，报告文件路径。

## 约束

- 模板三个 section 及注释保持原样，只在对应位置填充，不要增删 section、不要改模板注释。
- 基于事实撰写：所有结论（性能数字、测试结果）须来自 commit message、diff 或对话中已有的实测数据，不得编造。
- 语言跟随 commit 的语言习惯（社区 PR 通常英文），文件末尾必须附中文翻译版本（见工作流程第 5 步）。
- 已知限制：若 commit 信息不足且对话中无实测数据，测试段如实写待补充或说明，不要虚构数据。
