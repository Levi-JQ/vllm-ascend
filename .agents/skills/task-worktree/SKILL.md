---
name: task-worktree
description: |
  为"需要变更分支执行的任务"创建/管理 git worktree，实现多任务并行代码修改互不干扰。
  触发场景：用户发来需要在 vllm/vllm-ascend/vllm-ascend-tools 上改代码并切分支的任务；
  需要并行多个任务互不干扰；创建任务 worktree、查看任务状态、清理任务 worktree。
  用户提到"创建任务 worktree"、"新任务分支"、"task worktree"、"并行任务"、"切任务分支"时使用。
compatibility: 依赖 scripts/task_worktree.sh、codebases/ 下三仓库（vllm/vllm-ascend/vllm-ascend-tools）
---

# Task Worktree Skill

## 核心机制

所有任务源代码只有 `codebases/` 下一份（主工作树，三仓库各自独立 git）。每个"需要变更分支执行的任务"在 `task/<task-name>/code/` 下为三个仓库各创建一个 git worktree，分支名统一 `<task-name>`，基于 codebases 当前 HEAD 切出。多任务各自在独立 worktree/分支上改代码，互不干扰——主工作树始终保持原分支不动。

## 目录结构

```
codebases/                        # 唯一源代码（主工作树，三仓库各自独立 git）
├── vllm/                         #   vllm 主工作树（保持在基础分支，如 v0.26.0）
├── vllm-ascend/                  #   vllm-ascend 主工作树
└── vllm-ascend-tools/            #   vllm-ascend-tools 主工作树
task/
└── <task-name>/                  # 任务英文名（kebab-case，如 pd-timing-opt）
    ├── code/                     # worktree 根（.gitignore，不纳入 workspace git）
    │   ├── vllm/                 #   codebases/vllm 的 worktree @ branch <task-name>
    │   ├── vllm-ascend/          #   codebases/vllm-ascend 的 worktree @ branch <task-name>
    │   └── vllm-ascend-tools/    #   codebases/vllm-ascend-tools 的 worktree @ branch <task-name>
    ├── progress.md               # 任务进度（workspace git 跟踪）
    └── ...                       # 文档/bench/笔记（workspace git 跟踪）
```

## 命令（scripts/task_worktree.sh）

| 命令 | 作用 |
| :--- | :--- |
| `new <task-name> [--base <branch>]` | 三仓库各创建 worktree 到 `task/<task>/code/`，分支名统一 `<task-name>`，基于 codebases 当前 HEAD；`--base` 可指定基础分支（三仓库都需存在该分支） |
| `list [<task-name>]` | 列出所有任务 worktree（指定 `<task-name>` 显示该任务详情） |
| `info <task-name>` | 查看某任务三仓库的分支/commit/未提交文件数 |
| `remove <task-name>` | 删除任务 worktree 目录 + 三仓库 `<task-name>` 分支 |

## 何时创建 worktree

- ✅ 需要**修改 vllm/vllm-ascend/vllm-ascend-tools 代码**的任务 → 创建 worktree
- ✅ 需要切到**独立分支**验证某特性/修复的任务 → 创建 worktree
- ❌ 纯部署测试、不改代码 → 不创建，直接用默认模式 `sync.sh`（同步 codebases 主分支）启动

## 完整流程

1. 收到改代码任务，确定任务英文名 `<task-name>`（kebab-case，如 `pd-timing-opt`、`fix-oom-handshake`）
2. （可选）在 `codebases/` 各仓库主工作树切到想要的基础分支——默认基于当前 HEAD，所以先切好基础分支再创建 worktree
3. 创建 worktree：
   ```bash
   scripts/task_worktree.sh new <task-name>
   ```
4. 在 `task/<task-name>/code/{vllm,vllm-ascend,vllm-ascend-tools}` 下编辑代码（已在分支 `<task-name>` 上）
5. 同步到远程（任务模式）：
   ```bash
   scripts/sync.sh <itask-name> --task <task-name> --port <port>
   ```
   sync 只同步共享文件 + 该任务目录，代码按内容比对并 `--link-dest` 硬链接复用远端已有任务代码（新任务仅传分支差量；引用任务自动选远端最新，`--link-dest-task` 可指定）
6. 启动服务：`/start-service`（CODE_BASE 指向 `$WS/task/<task-name>/code`，start-service skill 自动适配）
7. 任务完成：在 worktree 里 commit 到分支 `<task-name>`；如需合并到主干，在 codebases 主工作树 merge/cherry-pick
8. **worktree 与分支保留不删除**（便于追溯）：任务完成后**不要**调用 `remove`，保留 `task/<task-name>/code/` worktree 和三仓库 `<task-name>` 分支。仅在明确不需要追溯时才用 `scripts/task_worktree.sh remove <task-name>` 清理。

## 注意事项

- 三仓库分支名**统一 `<task-name>`**，与 task 目录名一一对应
- worktree 的 `.git` 是指向 codebases 主仓库的文件（`gitdir: .../codebases/<repo>/.git/worktrees/...`），**不能同步到远程**——`sync.sh --task` 自动排除所有 `.git`
- vllm-ascend 的 catlass submodule 在 worktree 里需单独 `git submodule update --init`（`task_worktree.sh new` 自动处理）
- `task/<task>/code/` 被 `.gitignore` 忽略（worktree 是 codebases 仓库的嵌套 git，不纳入 workspace git）；`task/<task>/` 下的文档（progress.md/notes/bench 等）仍正常被 workspace git 跟踪
- **分支保留策略**：任务完成后 worktree 和分支默认保留（不 remove），方便后续追溯任务改了什么代码。`remove` 仅在明确不需要时手动调用
- 多任务并行：每个任务一个 `<task-name>`，各自的 worktree/分支完全独立，主工作树始终不动
- 只有 **vllm / vllm-ascend / vllm-ascend-tools** 三个仓库纳入 worktree 机制，其他 codebases 仓库（torch-npu、MindStudio-ModelSlim 等）不管理
