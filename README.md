# 🏥 医疗知识问答助手 Agent (基于 LangGraph)

## 📌 项目简介
针对医疗领域内容参差不齐、大模型易产生幻觉的问题，从0到1搭建了一套**支持多轮对话、RAG检索、自主工具调用与引用溯源**的垂直领域智能体系统。
本项目经历了从 V1 基础对话、V2 多轮记忆，最终演进为 V3/V4 基于 **LangGraph 状态机**的循环智能体（Cyclic Agent），实现了大模型自主规划、动态选择工具及闭环纠错。

## 🛠 技术栈
- **核心框架**：Python, LangChain, LangGraph
- **大模型**：阿里云百炼 DashScope (Qwen-Plus / Qwen3.7-Flash)
- **向量与检索**：Chroma DB, DashScope Embeddings (自定义封装)
- **工具生态**：Coze, MCP (Model Context Protocol)
- **前端交互**：Gradio (多会话管理、流式交互)
- **工程化**：FastAPI, Python-dotenv, Git

## ✨ 核心功能与工程亮点

### 1. 架构演进：从线性 Chain 到 LangGraph 循环智能体
- **痛点**：传统线性 Chain 流程死板，无论用户问什么都会强制检索一遍知识库。
- **方案**：引入 **LangGraph 状态机**，设计 `agent -> tools -> agent` 的循环路由（Conditional Edges）。
- **成果**：Agent 能够根据用户输入**自主决策**“直接回答”还是“调用外部工具”，支持在检索后反思并在必要时更换工具重试，实现了真正的 ReAct 模式。

### 2. 攻克 RAG 检索与 Embedding 兼容性 Bug
- **痛点**：官方库（DashScope SDK / OpenAI 兼容接口）存在 API 格式冲突，导致向量化或检索时抛出 `input.texts should be array` 等底层报错。
- **方案**：**继承 LangChain `Embeddings` 基类，手写自定义嵌入类（`MyTcmEmbedding`）**。实现 `document` 与 `query` 分离，内置 Batch Size 分批处理（每次10条），并加入空值兜底逻辑。
- **成果**：彻底解决三方库兼容性问题，保障了本地 Chroma 向量库的检索精度与稳定性。

### 3. 防幻觉机制与引用溯源 (Citation Tracing)
- **痛点**：医疗场景容错率极低，大模型易使用自身预训练知识胡编乱造。
- **方案**：通过严格的 System Prompt 约束，并在工具返回时打上 `[1] 来源: 文档名` 的标记，要求模型生成时必须带上引用编号。一旦资料为空，强制回复“根据现有知识库，我无法回答”。
- **成果**：实现了可溯源的严谨医学回答，彻底杜绝模型编造医学结论的法律合规风险。

### 4. 多工具协同与查询改写优化
- **查询改写（Query Rewriting）**：在 RAG 检索前，利用 LLM 将用户口语化、带指代的问题（如“那风热感冒呢？”）重写为规范的中医术语（“风热感冒的症状和调理方法”），大幅提升检索命中率。
- **工具集集成**：
  - `search_tcm_knowledge`：本地知识库检索（带查询改写和引用溯源）
  - `search_web`：DashScope 联网搜索（本地没有时自动触发）
  - `calculate_bmi`：BMI 计算器（结合中医调理建议合并回答）

### 5. 多会话管理与会话持久化
- 前端基于 Gradio 构建 ChatGPT 风格的多会话界面。
- 利用 LangGraph 内置的 `MemorySaver` 和 `thread_id` 实现多用户上下文隔离，支持会话的新建、切换与删除。

## 🚀 快速开始

### 1. 环境配置
```bash
# 创建并激活虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# 安装依赖
pip install -r requirements.txt

# 配置环境变量 (.env 文件)
echo "DASHSCOPE_API_KEY=你的阿里云百炼API_KEY" > .env
