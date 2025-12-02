# ✨ GPT-OSS 本地 Web 对话平台

![Ollama](https://img.shields.io/badge/Ollama-Local_AI-black?style=flat&logo=ollama)
![Status](https://img.shields.io/badge/Status-Active-success)
![Platform](https://img.shields.io/badge/Platform-Web%20|%20Mobile-blue)

## 📖 项目概述

本项目基于 Ollama 工具链本地化部署 `gpt-oss:latest` 大语言模型，通过轻量 Web 界面实现自然语言对话能力。

*   **核心优势**：数据全程运行于本地，不经过第三方服务器，兼顾隐私安全与使用便捷性。
*   **适用场景**：无需依赖 OpenAI API，部署后即可通过浏览器随时随地交互，支持团队内网共享访问。

## 🖼️ 功能预览

简洁直观的对话界面，适配电脑/手机等设备，提供流畅的交互体验：

*   ✅ **清晰交互**：区分用户/模型消息气泡，支持 Markdown 渲染。
*   ✅ **动态反馈**：实时打字机加载动画，错误状态提示。
*   ✅ **便捷操作**：支持 Shift+Enter 换行，移动端完美适配。

## 🚀 核心优势

| 功能 | 说明 |
| :--- | :--- |
| **🔒 本地化部署** | 模型运行于本地服务器，数据闭环存储，杜绝隐私泄露。 |
| **🌐 全网络支持** | 兼容内网（同局域网）访问，配置端口转发后亦可支持公网。 |
| **👥 多人共享** | 无 IP 限制，支持多设备并发访问，适合小型办公/团队协作。 |
| **⚡ 体验优化** | 预设最佳参数（Temp 0.5），平衡回复质量与速度。 |
| **📱 多端适配** | 响应式布局设计，PC、平板、手机访问体验一致。 |

## ⚙️ 快速启动 (关键配置)

> **前提**：服务器已安装 Ollama，且已拉取 `gpt-oss:latest` 模型。

### 1. 启动 Ollama 服务 (后端)

为了支持局域网访问，**必须**配置 Host 为 `0.0.0.0` 并允许跨域请求。请在服务器执行：

```bash
# 停止旧服务
pkill ollama

# 启动新服务 (允许所有来源跨域，监听所有网卡)
export OLLAMA_HOST=0.0.0.0
export OLLAMA_ORIGINS="*"
nohup ollama serve > ollama.log 2>&1 &
```
---

###  🧬 Galaxy BioBlend RAG Module (新增子系统)

> **功能**: 基于本地大模型 (GPT-OSS) 与 BioBlend 库的 Galaxy 平台自动化操作代理。

本模块通过 RAG 技术解决了 Galaxy API 文档复杂难懂的问题，实现了从“自然语言指令”到“安全可执行 Python 脚本”的转化。

### 🛠️ 工程架构

1.  **数据层 (ETL)**: `extract_rules.py` - 扫描 `bioblend` 库，清洗并提取 160+ 个 API 方法签名，生成 `bioblend_knowledge.json`。
2.  **存储层**: `nomic-embed-text` 向量化 -> 本地 ChromaDB 存储。
3.  **逻辑层**: `local_rag.py` - 检索 API 文档 -> GPT-OSS 生成代码 -> 强制植入配置读取逻辑。
4.  **执行层**: `verify_galaxy.py` - 验证脚本在真实环境的执行效果。

### 📂 文件说明

*   `extract_rules.py`: **[构建]** 知识库提取工具。
*   `local_rag.py`: **[核心]** 智能问答与代码生成主程序。
*   `verify_galaxy.py`: **[测试]** 连接与鉴权验证脚本。
*   `secrets/galaxy_config.json`: **[配置]** 敏感信息配置文件（需手动创建，勿上传 Git）。

### 🚀 快速开始

**1. 配置密钥**
创建 `secrets/galaxy_config.json`:
```json
{
  "galaxy_url": "https://usegalaxy.org",
  "api_key": "YOUR_REAL_API_KEY"
}
```
**2. 初始化知识库**
```
python extract_rules.py
```
**3. 启动助手**
```
python local_rag.py
```
