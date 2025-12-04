# 🧬 GPT-OSS Galaxy Agent (Bioinformatics Copilot)

![Ollama](https://img.shields.io/badge/Ollama-Local_AI-black?style=flat&logo=ollama)
![Galaxy](https://img.shields.io/badge/BioBlend-Galaxy_Project-blue)
![Python](https://img.shields.io/badge/Backend-FastAPI-green)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📖 项目概述

**GPT-OSS Galaxy Agent** 是一个基于本地大模型（LLM）的生物信息学智能中间件。它旨在打破自然语言与复杂生信分析平台（Galaxy）之间的壁垒。

通过集成 **RAG（检索增强生成）**、**多模态视觉识别（OCR）** 以及 **BioBlend** 自动化接口，用户可以通过自然语言对话，自动完成从“查找工具”到“上传数据”，再到“编写并执行脚本”的全流程操作。

*   **核心理念**：让 AI 像生信专家一样思考——既能处理模糊的分析需求，也能精准执行系统管理指令。
*   **隐私安全**：核心推理（GPT-OSS）、视觉识别（Llama-Vision）与向量检索全部在本地运行，数据不离境。

---

## 🔥 最新更新 (v1.2) - 体验与智能升级

本次更新着重优化了交互体验与上下文理解能力，实现了真正的“对话式”分析。

*   **🧠 短期记忆 (Context Awareness)**：
    *   新增对话历史追踪功能。用户可以基于上一轮的执行结果进行追问（例如：“这个报错是什么意思？”或“结果说明了什么？”），AI 将结合上下文进行回答。
*   **🌐 语言自适应 (Language Consistency)**：
    *   优化 Prompt 约束。AI 现在会严格跟随用户的提问语言（中文问 -> 中文回），包括 OCR 识别结果和错误提示，彻底告别“中英夹杂”。
*   **🎨 界面去技术化 (White-labeling)**：
    *   前端文案全面优化，移除底层模型技术术语，打造更专业、更友好的“生信智能助手”形象。

---

## ✨ 核心功能特性

### 1. 🧠 智能双源决策系统 (Parallel Context Logic)
解决了“海量工具检索干扰基础指令”的难题。系统将知识分为两类，并行提交给 LLM 进行加权决策：
*   **Source A (系统常识)**：内置高频管理指令（如查询用户、查看历史、上传文件），保证基础操作零误差。
*   **Source B (工具检索)**：基于 ChromaDB 从 2000+ Galaxy 工具中检索 Top-N 候选，解决长尾分析需求。
*   **智能路由**：LLM 自动判断是直接生成代码（精准匹配），还是列出选项让用户选择（模糊匹配）。

### 2. 👁️ 多模态视觉支持 (OCR)
集成 `llama3.2-vision` 模型。
*   **场景**：用户上传包含 DNA 序列、报错信息或参数配置的截图。
*   **能力**：自动提取图片中的文本信息，并将其作为上下文无缝融入对话分析中。

### 3. 🔄 全链路自动化闭环
*   **搜索 (Search)**：自然语言模糊搜索工具（如“做质控”、“比对序列”）。
*   **选择 (Select)**：提供交互式卡片，用户确认工具意图。
*   **上传 (Upload)**：前端直接上传文件，自动对接 Galaxy History API。
*   **执行 (Execute)**：后端沙箱环境自动生成并运行 Python (BioBlend) 脚本，实时返回结果。

---

## 🏗️ 系统架构

![Computer Science AI + Bio-Integration System Workflow](./system_architecture.png)

> **流程说明**：
> 1. **用户前端 (User Frontend)**：支持文本查询与文件上传，提供可视化交互界面。
> 2. **智能后端 (Intelligent Backend)**：
>    - **意图识别 (Intent Recognition)**：区分系统指令与分析需求。
>    - **决策规划 (Decision & Planning)**：结合 BioBlend 知识库 (RAG) 生成执行策略。
>    - **结果整合 (Result Integration)**：处理执行反馈并生成自然语言报告。
> 3. **外部计算 (External Bio-Compute)**：Galaxy 平台负责实际的工具调用与大规模数据处理。

---

## 📂 项目目录结构

```text
.
├── app.py                  # [核心] Web 后端主程序 (FastAPI + LangChain)
├── extract_rules.py        # [ETL] Galaxy 工具爬虫与规则提取工具
├── rebuild_db.py           # [构建] 向量数据库构建脚本 (JSON -> ChromaDB)
├── verify_galaxy.py        # [测试] Galaxy 连接性验证脚本
├── templates
│   └── index.html          # [前端] 交互界面 (Bootstrap + Markdown渲染)
├── data
│   ├── bioblend_knowledge.json  # 提取出的工具元数据
│   └── chroma_db_bioblend       # 持久化向量数据库文件
├── assets
│   └── system_architecture.png  # 系统架构图
└── .env                    # 环境变量配置
```

⚙️ 部署与使用指南
1. 环境准备

确保服务器已安装 Python 3.10+ 和 Ollama，并拉取以下模型：

code
Bash
download
content_copy
expand_less
# 1. 主力推理模型 (负责逻辑判断与代码生成)
ollama pull gpt-oss:latest

# 2. 向量化模型 (负责 RAG 检索)
ollama pull nomic-embed-text

# 3. 视觉模型 (负责 OCR)
ollama pull llama3.2-vision:11b
2. 配置项目

创建 .env 文件，填入你的 Galaxy 服务器信息：

code
Ini
download
content_copy
expand_less

3. 初始化知识库 (首次运行或更新工具时)

从 Galaxy 服务器抓取最新工具列表，并构建向量索引：

code
Bash
download
content_copy
expand_less
# 1. 提取工具规则
python extract_rules.py

# 2. 构建向量数据库
python rebuild_db.py
4. 启动服务
code
Bash
download
content_copy
expand_less
python app.py

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
用户：点击 [FastQC] 卡片 -> 上传 data.fastq。
AI：生成 gi.tools.run_tool 代码并提交任务。

场景 C：图片识别 (多模态)

用户：点击上传按钮 -> 选择一张包含 DNA 序列的截图。
AI：调用 Vision 模型提取序列文本 -> 用户确认后进行 BLAST 比对。

⚠️ 注意事项

Galaxy 连接：请确保服务器能访问 .env 中配置的 GALAXY_URL。

Ollama 资源：同时运行 Vision 和 LLM 模型需要一定的显存（建议 16GB+ VRAM），否则速度可能较慢。

安全性：生成的代码在本地沙箱执行，但仍建议不要连接生产环境的管理员账号。

Maintainer: Simon (Zyzk)
Last Updated: 2025-02

code
Code
download
content_copy
expand_less
