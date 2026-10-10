# 社区版 vllm-ascend 内网 patch

社区版 vllm-ascend（来自 github.com/vllm-project/vllm-ascend）在内网购建可能失败，原因：
- 外部下载 URL（gitcode.com、gitee.com）被拦截或很慢
- CMake 第三方依赖下载失败

## 何时应用

检查 vllm-ascend 是否来自社区上游：
```bash
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "cd <vllm_ascend_source> && git remote -v"
# origin 指向 github.com → 社区上游 → 可能需要 patch
# origin 指向内部 git（code.alipay.com）→ 已 patch → 跳过
```

## 已知 patch

### CMake 第三方下载 URL（gitcode.com → 内网 OSS）

vllm-ascend 的 `csrc/cmake/third_party/` 下有 `.cmake` 文件从 `gitcode.com`（外部）下载依赖。替换为蚂蚁内网 OSS 镜像：

| 文件 | 包 | gitcode.com URL → OSS URL |
| :--- | :--- | :--- |
| `abseil-cpp.cmake` | abseil-cpp | `gitcode.com/.../abseil-cpp-20230802.1.tar.gz` → `antsys-language-compiler...oss-alipay.../abseil-cpp-20230802.1.tar.gz?OSS...` |
| `ascend_protobuf.cmake` / `protobuf.cmake` | protobuf | `gitcode.com/.../protobuf-25.1.tar.gz` → `antsys-language-compiler...oss-alipay.../protobuf-25.1.tar.gz?OSS...` |
| `gtest.cmake` | googletest | `gitcode.com/.../googletest-1.14.0.tar.gz` → `antsys-language-compiler...oss-alipay.../googletest-1.14.0.tar.gz?OSS...` |
| `json.cmake` | nlohmann json | `gitcode.com/.../include.zip` → `log.obs.cn-wulan.../include.zip?AccessKeyId...` |
| `makeself-fetch.cmake` | makeself | `gitcode.com/.../makeself-release-2.5.0-patch1.tar.gz` → `antsys-language-compiler...oss-alipay.../makeself-release-2.5.0-patch1.tar.gz?OSS...` |

**Note**：v0.23.0+ 可能还有 `secure_c.cmake`，里面的 libboundscheck 用 `gitee.com` URL——需要单独的内网镜像。

### Patch 文件

用户维护的 patch 文件位于：`<repo_base>/github-compile.patch`

### 应用 patch

**`git apply` 可能失败**（patch 针对不同版本时行号不匹配）。用 `--3way`：
```bash
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "cd <vllm_ascend_source> && git apply --3way <patch_file>"
```

**`--3way` 可能在文件里留下合并冲突标记**（`<<<<<<<`、`=======`、`>>>>>>>`）。CMake 无法解析 → **必须清理**：
```bash
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "
  cd <vllm_ascend_source>
  grep -rl '<<<<<<<' csrc/
  # 解决: 保留 ours（patched），删除 theirs + 标记
  for f in \$(grep -rl '<<<<<<<' csrc/); do
    sed -i '/^=======$/,/^>>>>>>>/d' \"\$f\"
    sed -i '/^<<<<<<< /d' \"\$f\"
  done
"
```

**替代方案**：完全跳过 patch 文件，对每个 gitcode.com URL 做定向 `sed` 替换。跨版本更稳健。

## 检查 patch 是否已应用

```bash
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "cd <vllm_ascend_source> && git diff --stat HEAD"
# 空 → 未应用 patch
# 非空 → 已有 patch，复审
```

## Note

本文件应随着构建尝试中发现的 patch 细节持续更新。每个 patch 记录：
- 修复什么（哪个 URL/index/path）
- 如何应用（sed 命令 / patch 文件）
- 是否在特定 vllm-ascend 版本上测过