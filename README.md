# AI Customer Service

基于 LangChain、LangGraph、RAG 与 Streamlit 的扫地机器人智能客服系统。

项目面向扫地机器人和扫拖一体机器人场景，结合产品知识库、对话记忆、工具调用和多模态输入，为用户提供可追溯的售前与售后问答体验。

## 界面截图

### 天气环境判断与清洁建议

![天气环境判断与清洁建议](docs/images/chat-weather.png)

### 使用报告生成

![扫地机器人使用情况报告与保养建议](docs/images/monthly-report.png)

### 对话历史、快捷输入与附件/语音

![对话历史、快捷输入与附件语音功能](docs/images/conversation-history.png)

## 功能

- RAG 知识库问答：从产品手册、常见问题和维护资料中检索相关内容。
- 混合检索：支持向量检索、BM25 关键词检索、RRF 融合和可选 rerank。
- ReAct Agent：根据问题规划步骤，并调用知识库、天气、定位和外部数据工具。
- 多轮对话：保留当前会话上下文，并支持长期记忆摘要。
- 多模态输入：支持文本、附件和语音输入；语音通过 DashScope ASR 转写。
- Streamlit 界面：提供对话历史、流式输出、快捷问题和报告生成。
- 国内/国际 API 区域：通过配置切换 DashScope 国内接口或兼容 OpenAI 的国际接口。

## 技术栈

- Python 3.10+
- Streamlit
- LangChain / LangGraph
- Chroma 或 PostgreSQL + pgvector
- BM25、RRF、可选 DashScope rerank
- DashScope Qwen 模型与 Embedding
- 高德地图 Web 服务 API（天气与 IP 定位）

## 项目结构

```text
app.py                  Streamlit 应用入口
agent/                  Agent、工具和中间件
config/                 模型、RAG、记忆和外部工具配置
data/                   知识库资料与外部示例数据
memory/                 长期记忆实现
model/                  LLM、Embedding 和 ASR 工厂
prompts/                系统提示词和报告提示词
rag/                    向量库、混合检索和 RAG 服务
utils/                  配置、日志、文件和网络环境工具
eval/                   检索评测脚本与测试用例
```

## 安装

建议使用虚拟环境：

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## API 配置

不要把真实 API Key 写入代码、YAML 或提交记录。使用环境变量配置：

```powershell
$env:DASHSCOPE_API_KEY = "your_dashscope_api_key"
$env:GAODE_API_KEY = "your_gaode_api_key"
```

Linux/macOS：

```bash
export DASHSCOPE_API_KEY="your_dashscope_api_key"
export GAODE_API_KEY="your_gaode_api_key"
```

模型区域和名称可在 `config/rag.yml` 中调整。高德服务地址和超时配置在 `config/agent.yml` 中；Key 只从 `GAODE_API_KEY` 环境变量读取。

## 启动

```bash
streamlit run app.py
```

Windows 也可以运行：

```powershell
.\start_robot_service.ps1
```

启动后访问 `http://localhost:8501`。

首次运行时，系统会根据 `config/chroma.yml` 中的配置初始化或读取向量库。请确保 `data/` 下的知识库资料存在，并按需准备 API 配额。

## 使用 pgvector

默认使用本地 Chroma。若切换到 PostgreSQL + pgvector，请在配置中选择对应后端，并通过环境变量提供连接字符串：

```powershell
$env:PGVECTOR_CONNECTION_STRING = "postgresql+psycopg://user:password@localhost:5432/dbname"
```

不要把真实数据库密码提交到仓库。

## 安全说明

- `.env`、数据库、日志、缓存、虚拟环境和运行时产物已加入忽略规则。
- API Key、数据库密码和外部服务 Token 必须通过环境变量注入。
- 如果凭据曾经出现在代码、日志或 Git 历史中，应立即在服务商后台撤销并重新生成。
- 外部 HTTP 工具默认关闭；启用前请审查 URL、请求头、权限和返回数据。

## 免责声明

本项目仅供学习、研究和原型验证使用。实际部署时请根据业务场景完善权限控制、隐私保护、限流、审计和错误处理。
