code
Markdown
download
content_copy
expand_less
# ✨ GPT-OSS Galaxy Agent (Bioinformatics Copilot)

![Ollama](https://img.shields.io/badge/Ollama-Local_AI-black?style=flat&logo=ollama)
![Galaxy](https://img.shields.io/badge/BioBlend-Galaxy_Project-blue)
![Platform](https://img.shields.io/badge/Platform-Web%20|%20Mobile-blue)
![Status](https://img.shields.io/badge/Status-Active_v1.3-success)

## 📖 项目概述

**GPT-OSS Galaxy Agent** 是一个基于本地大模型（LLM）的生物信息学智能中间件。它旨在打破自然语言与复杂生信分析平台（Galaxy）之间的壁垒。

通过集成 **RAG（检索增强生成）**、**多模态视觉识别（OCR）** 以及 **BioBlend** 自动化接口，用户可以通过自然语言对话，自动完成从“查找工具”到“上传数据”，再到“编写并执行脚本”的全流程操作。

*   **核心理念**：让 AI 像生信专家一样思考——既能处理模糊的分析需求，也能精准执行系统管理指令。
*   **隐私安全**：核心推理（GPT-OSS）、视觉识别（Llama-Vision）与向量检索全部在本地运行，数据不离境。

---

## 🔥 最新更新 (v1.3) - 交互体验与智能引导升级

本次更新聚焦于前端交互的现代化与对话的引导性，大幅降低了用户的操作成本。

*   **📂 全能文件交互 (Seamless I/O)**：
    *   **拖拽上传**：支持直接将文件拖入聊天窗口，即刻触发上传与分析。
    *   **剪贴板支持**：支持 `Ctrl+V` 直接粘贴截图或文件，OCR 识别一触即发。
    *   **视觉升级**：全新的文件图标与上传状态指示。
*   **💡 智能追问引导 (Cognitive Guidance)**：
    *   **动态建议**：后端根据当前的执行结果（如代码输出、OCR 文本），自动生成 3 个“猜你想问”的胶囊按钮（Chips）。
    *   **场景示例**：上传数据后提示 `[做质控]`；代码跑完后提示 `[解释结果]` 或 `[可视化]`。
*   **🧠 上下文记忆增强**：
    *   完善了短期记忆机制，AI 现在能精准理解“这个结果说明了什么”等基于上文的追问。

---

## ✨ 核心技术创新

### 1. 🧠 智能双源决策系统 (Parallel Context Logic)
解决了“海量工具检索干扰基础指令”的难题。系统将知识分为两类，并行提交给 LLM 进行加权决策：
*   **Source A (系统常识)**：内置高频管理指令（如查询用户、查看历史、上传文件），保证基础操作零误差。
*   **Source B (工具检索)**：基于 ChromaDB 从 2000+ Galaxy 工具中检索 Top-N 候选，解决长尾分析需求。
*   **智能路由**：LLM 自动判断是直接生成代码（精准匹配），还是列出选项让用户选择（模糊匹配）。

### 2. 👁️ 多模态视觉-语义对齐 (Visual-Semantic Alignment)
集成 `llama3.2-vision` 模型。
*   **场景**：用户上传包含 DNA 序列、报错信息或参数配置的截图。
*   **能力**：自动提取图片中的文本信息，并建立图像特征与文本意图的映射，填补了传统脚本无法处理非结构化输入的空白。

### 3. 🔄 状态感知的沙箱闭环 (State-Aware Execution)
*   **状态机**：构建了包含“文件状态”的本地沙箱。系统在生成代码前会自动校验前置条件（如文件是否上传）。
*   **自治闭环**：实现了“感知 $\rightarrow$ 决策 $\rightarrow$ 执行 $\rightarrow$ 反馈”的完整自治流程，且所有 Python 脚本均在受控环境中运行。

---

## 🏗️ 系统架构

![System Architecture](system_architecture.jpeg)

> **流程说明**：
> 1. **用户前端**：支持文本/拖拽/粘贴交互。
> 2. **智能后端**：通过 FastAPI 编排 Vision 模型（看）、GPT-OSS（想）和 ChromaDB（查）。
> 3. **执行环境**：生成的 BioBlend 脚本在本地沙箱运行，安全调用 Galaxy Server API。

---

## ⚙️ 部署与使用指南
1. 环境准备

确保服务器已安装 Python 3.10+ 和 Ollama，并拉取以下模型：


# 1. 主力推理模型 (负责逻辑判断与代码生成)
```
ollama pull gpt-oss:latest
```
# 2. 向量化模型 (负责 RAG 检索)
```
ollama pull nomic-embed-text
```
# 3. 视觉模型 (负责 OCR)
```
ollama pull llama3.2-vision:11b
```

2. 启动 Ollama 服务 (关键配置)

为了支持局域网访问并允许 Web 端跨域请求，必须配置 Host 为 0.0.0.0 并允许跨域。请在服务器执行：

code
Bash
download
content_copy
expand_less
# 停止旧服务
```
pkill ollama
```
# 启动新服务 (允许所有来源跨域，监听所有网卡)
```
export OLLAMA_HOST=0.0.0.0
export OLLAMA_ORIGINS="*"
nohup ollama serve > ollama.log 2>&1 &
```
3. 配置项目

创建 .env 文件，填入你的 Galaxy 服务器信息：

code
Ini
download
content_copy
expand_less

4. 初始化知识库 (首次运行或更新工具时)

从 Galaxy 服务器抓取最新工具列表，并构建向量索引：

code
Bash
download
content_copy
expand_less
# 1. 爬取工具规则
```
python extract_rules.py
```
# 2. 构建向量数据库
```
python rebuild_db.py
```
5. 启动服务
code
Bash
download
content_copy
expand_less
```
python app.py
```
后台运行
```
nohup python app.py > app.log 2>&1 &
```
查找进程
```
ps -ef | grep app.py
```
停止进程
```
pkill -f "python app.py"
```
服务默认运行在 http://0.0.0.0:8082。

🖥️ 交互场景示例
场景 A：系统管理 (精准匹配 Source A)

用户：“查询我的用户信息”
AI：识别为系统指令，直接调用 gi.users.get_current_user()。
结果：直接显示用户名、ID、邮箱。

场景 B：生信分析 (模糊匹配 Source B)

用户：“我想做质控”
AI：检索到 FastQC, MultiQC 等工具，无法确定具体意图。
界面：显示工具选择列表。
用户：点击 [FastQC] 卡片 -> 系统自动关联已上传文件。
AI：生成 gi.tools.run_tool 代码并提交任务。

场景 C：多模态交互 (v1.3 新特性)

用户：(直接 Ctrl+V 粘贴一张报错截图)
AI：调用 Vision 模型提取报错文本 -> 自动分析原因。
界面：弹出引导按钮 [尝试修复] [搜索解决方案]。

⚠️ 注意事项

Galaxy 连接：请确保服务器能访问 .env 中配置的 GALAXY_URL。

Ollama 资源：同时运行 Vision 和 LLM 模型需要一定的显存（建议 16GB+ VRAM），否则速度可能较慢。

安全性：生成的代码在本地沙箱执行，但仍建议不要连接生产环境的管理员账号。

🛠️ Powered By
<p align="center">
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
<img src="https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama" />
</p>

<p align="center">
<img src="https://img.shields.io/badge/LangChain-RAG_Framework-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain" />
<img src="https://img.shields.io/badge/ChromaDB-Vector_Store-fc521f?style=for-the-badge&logo=chroma&logoColor=white" alt="ChromaDB" />
<img src="https://img.shields.io/badge/Galaxy_BioBlend-Bioinformatics-2C3143?style=for-the-badge&logo=galaxy&logoColor=white" alt="Galaxy" />
</p>


Maintainer: Simon (Zyzk)
Last Updated: 2025-02 (v1.3)

code
Code
download
content_copy
expand_less

# GIBH Qwen Galaxy - 更新日志 (2025-12-12)

## 🚀 重大更新 (Major Updates)

### 1. 核心引擎升级 (Core Engine Upgrade)
- **模型切换**：底层模型由 `gpt-oss` (20B) 全面升级为 **`qwen3-vl:32b`**。
- **架构统一**：移除了独立的视觉模块，将“大脑”与“眼睛”整合为单体多模态智能体，充分利用 Qwen3-VL 卓越的代码生成与视觉理解能力。
- **效果提升**：显著增强了意图识别准确率、BioBlend 代码生成的稳定性，并新增了对生信图表和报错截图的原生分析能力。

### 2. 智能工作流执行 (Intelligent Workflow Execution)
- **自然语言触发**：用户现在可以通过自然语言（如“执行 Seurat 流程”）直接触发复杂工作流，无需强制点击按钮。
- **自动参数注入**：在 `WORKFLOW_TEMPLATES` 中内置了 Seurat 单细胞分析的“最佳实践参数”，小白用户无需手动配置即可运行。
- **动态工具解析**：废弃了硬编码的 Tool ID，改为基于关键词的动态搜索机制 (`_find_tool_id_by_keyword`)，自动适配不同版本的 Galaxy 环境。
- **参数映射层**：新增 `_map_params_for_galaxy` 中间层，自动将用户友好的参数转换为 Galaxy 内部格式（修复了 `nFeature_RNA` 命名规范等兼容性问题）。

### 3. 稳健的后端逻辑 (Robust Backend Logic)
- **上下文感知路由**：路由模块现在具备“短期记忆”，能结合上一轮对话（如 AI 询问“是否使用历史数据”）来准确判断用户简短回复（如“是”）的真实意图。
- **智能文件处理**：
    - 自动检测输入文件缺失，并引导用户选择历史数据。
    - 支持“中途上车”模式（例如跳过第一步，直接基于现有的 RDS 文件从第二步开始分析）。
    - **NLP 文件解析**：支持从用户指令中直接提取文件需求（如“使用 5, 6, 7 号文件”）。
- **指数退避轮询**：优化了后台任务监控机制，轮询间隔随时间动态增加（2s -> 30s），大幅降低对 Galaxy 服务器的请求压力。

### 4. UI/UX 交互重构 (UI/UX Overhaul)
- **数据选择器升级**：
    - 新增滚动条支持，完美适配大量历史数据（70+ 条目）。
    - 优化点击交互，支持整行点击选中。
    - 修复了“空消息幻觉” Bug，现在点击确认后会发送明确的指令文本。
- **工作流监控**：新增垂直进度条组件，实时可视化后台任务的执行状态。
- **灵活配置**：工作流卡片新增复选框，允许用户自由勾选或跳过特定分析步骤。
