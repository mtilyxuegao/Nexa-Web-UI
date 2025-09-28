# Nexa Web-UI 测试指南

## 概述

本项目集成了 Nexa SDK 与 Web-UI，实现本地多模态 LLM 驱动的浏览器自动化。目前使用 `NexaAI/Qwen3-VL-4B-MLX-8bit` 模型进行测试。

## 环境配置

### 1. 安装 Nexa SDK

从官方 GitHub 仓库下载安装包：

前往 https://github.com/NexaAI/nexa-sdk/releases/tag/v0.2.35 根据您的平台下载对应的安装包并安装。

### 2. 安装 Web-UI 依赖

基于 [browser-use/web-ui](https://github.com/browser-use/web-ui) 项目，安装所需依赖：

```bash
# 进入 web-ui 目录
cd web-ui

# 使用 uv 创建虚拟环境（推荐）
uv venv --python 3.11
source .venv/bin/activate  # macOS/Linux
# 或 .\.venv\Scripts\Activate.ps1  # Windows PowerShell

# 安装 Python 依赖
uv pip install -r requirements.txt

# 安装 Playwright 浏览器（推荐仅安装 Chromium）
playwright install chromium --with-deps
```

### 3. 配置环境变量

项目已包含预配置的 `web-ui/.env` 文件，主要配置如下：

```bash
# LLM 提供商设置
DEFAULT_LLM=nexa
NEXA_ENDPOINT=http://127.0.0.1:8080/v1

# 其他 API 密钥（如需使用其他 LLM）
# OPENAI_API_KEY=your_openai_key
# ANTHROPIC_API_KEY=your_anthropic_key
```

### 4. 下载模型

设置 Hugging Face 令牌并下载模型（该模型为 Private 模型，需要 HF Token）：

```bash
# 设置 Hugging Face 令牌（需要两个环境变量）
export HUGGINGFACE_HUB_TOKEN="your_huggingface_token"
export NEXA_HFTOKEN="your_huggingface_token"

# 下载多模态 VLM 模型
nexa pull NexaAI/Qwen3-VL-4B-MLX-8bit
```

**注意**：
- 该模型为 Private 模型，需要有效的 Hugging Face Token
- 模型约 4GB，确保有足够的存储空间和网络带宽
- 确保您的 HF Token 有访问该模型的权限

## 测试准备

### 0. 清理端口

在开始测试前，确保端口干净：

```bash
# 杀掉所有相关进程
lsof -ti:8080,7788 | xargs kill -9 2>/dev/null
pkill -f "nexa serve"
pkill -f "webui.py"
```

### 1. 启动 Nexa 服务器

```bash
cd /Users/jason/Desktop/Nexa-Web-UI
nexa serve --host 127.0.0.1:8080 --keepalive 600
```

等待看到 `Localhosting on http://127.0.0.1:8080/docs/ui` 提示。

### 2. 启动 Web-UI

在新终端窗口中：

```bash
cd /Users/jason/Desktop/Nexa-Web-UI
source .venv/bin/activate
python web-ui/webui.py --ip 127.0.0.1 --port 7788
```

等待看到 `Running on local URL: http://127.0.0.1:7788` 提示。

## 测试步骤

访问 http://127.0.0.1:7788 进行测试：

### 完整测试任务（三步骤）
输入任务：`Go to google.com, search for 'nexa ai', and click the first result`

**步骤 1: 导航到 Google（应该成功）**
- 动作：导航到 google.com
- 预期结果：✅ 成功打开 Google 首页

**步骤 2: 执行搜索（应该成功）**  
- 动作：在搜索框输入 'nexa ai' 并搜索
- 预期结果：✅ 成功显示搜索结果页面

**步骤 3: 点击第一个结果（目前失败）**
- 动作：点击第一个搜索结果
- 预期结果：❌ 失败，无法执行点击操作
- 现象：Agent 会不断重试，但 `Action 1/1: {}` 显示为空，基于之前的记忆继续尝试

## 当前问题分析

### 核心错误

**Action 传递失败**：尽管 LLM 正确输出了 JSON 格式的 action（例如 `{"click_element": {"index": 8}}`），但最终传递给 browser-use 的 action 为空 `{}`。

### 日志分析示例

```
INFO [src.utils.nexa_adapter] 📝 模型原始输出: {"current_state": {...}}, "action": [{"click_element": {"index": 8}}]}
INFO [src.utils.nexa_adapter] 🔧 修复 current_state 和 action 之间的结构错误
INFO [src.utils.nexa_adapter] ✅ JSON 修复成功
INFO [agent] 🛠️ Action 1/1: {}  ← 问题：action 为空
```

### 观察到的性能问题

相比 Ollama 等其他 LLM 提供商，使用 Nexa 时存在以下问题：

1. **Steps 数量更多**：相同的任务需要更多步骤完成
2. **重试频率高**：LLM 输出格式不完全符合要求时会触发重试
3. **Context 长度敏感**：随着对话历史增长，问题变得更加明显

## 错误原因分析

### 1. LLM 能力限制
- **模型规模**：量化后的模型在复杂 JSON 结构生成上存在局限
- **指令跟随能力**：对严格的 JSON 格式要求的遵循不够稳定
- **上下文处理**：长对话历史下的性能下降

### 2. JSON 格式问题
- **结构性错误**：模型经常输出 `{"current_state": {...}}, "action": [...]` 而非正确的单一 JSON 对象
- **修复不完善**：当前的正则表达式修复方案仍有 edge cases

### 3. 多模态处理挑战
- **图像理解**：对空白页面或复杂页面的理解可能不准确
- **视觉-语言对齐**：图像内容与 JSON 输出之间的对齐存在问题

## 可能的解决方案

### 方案 1：模型升级
**优点**：根本性解决能力问题
```
- 使用更大参数的模型（如 7B/13B）
- 选择专门优化过的 instruction-following 模型
- 考虑支持更长 context 的模型
```

**缺点**：资源消耗增加，推理速度下降

### 方案 2：二阶段处理
**优点**：保持当前模型，增加格式校正步骤
```
阶段1: VLM 模型理解图像并生成初步 JSON
阶段2: 轻量级 LLM（不带视觉）纯文本修正 JSON 格式
```

**缺点**：增加延迟和复杂度

### 方案 3：增强正则修复
**优点**：低延迟，当前正在使用的方案
```
- 完善正则表达式覆盖更多 edge cases
- 添加多层次的格式检查和修复
- 实现更智能的 JSON 结构重建
```

**缺点**：治标不治本，难以覆盖所有情况

### 方案 4：Prompt 工程优化
**优点**：无需更改架构
```
- 提供更多 JSON 格式示例
- 使用 few-shot prompting
- 添加格式验证的自我检查机制
```

**缺点**：效果有限，依赖于模型的指令跟随能力

## 当前实现状态

### 已实现功能
- ✅ Nexa SDK 与 LangChain 集成
- ✅ 多模态图像处理（base64 转文件路径）
- ✅ Markdown 代码块解析
- ✅ 基础 JSON 格式修复
- ✅ 调试日志和图像保存

### 已知限制
- ❌ Action 传递不稳定
- ❌ 复杂任务成功率低
- ❌ 相比其他 LLM 提供商性能差距明显

### 下一步计划
1. 深入调试 browser-use 的 JSON 解析逻辑
2. 实现方案2的二阶段处理架构
3. 评估升级到更大规模模型的可行性

## 调试技巧

### 查看详细日志
```bash
# 查看 Nexa 服务器日志
tail -f nexa_server.log

# 查看 Web-UI 日志（控制台输出）
# 关注 [src.utils.nexa_adapter] 的输出
```

### 检查保存的图片
```bash
ls -la /Users/jason/Desktop/Nexa-Web-UI/debug_images/
# 查看 LLM 实际接收到的图像内容
```

### 验证 JSON 修复
在 `nexa_adapter.py` 中查找修复日志：
- `🔧 修复 current_state 和 action 之间的结构错误`
- `✅ JSON 修复成功`
- `❌ JSON 修复失败`

---

*最后更新：2025年9月28日*