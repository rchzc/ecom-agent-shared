# ecom-agent-shared · MCP & Agent 共享集群

跨境电商 AI 生态的**最底层共享集群**。所有业务 Agent（售前咨询、内容运营、数据中台、
销售考核）都通过这一个包接入模型、检索知识、读写记忆、调用工具 —— 而不是每个 Agent
各写一套。

本仓库对应生态架构图中的 **项目六 · MCP & Agent 共享集群（LLM 网关 / RAG / 记忆 / Prompt）**。

```bash
python demo.py     # 不需要 API Key，不联网，直接跑
```

---

## 一、在生态里的位置

```
        ┌──────────────────────────────────────────────────────┐
        │  业务层（各自独立仓库，只依赖本包）                      │
        │  售前咨询 Agent │ 内容运营 Agent │ 数据中台 │ 销售考核系统  │
        └───────────────────────┬──────────────────────────────┘
                                │  pip / 本地路径安装
        ┌───────────────────────▼──────────────────────────────┐
        │  ecom-agent-shared（本仓库）                          │
        │                                                      │
        │  gateway/  LLM 网关    多厂商统一接入 · 复杂度路由       │
        │                        三级 JSON 容错 · 用量成本记账    │
        │  rag/      RAG         语义切分 · 三种检索后端           │
        │                        向量召回 + 词面混合重排          │
        │  memory/   记忆        会话记忆（多轮）· 长期事实记忆     │
        │  prompts/  Prompt 中心 模板注册 · 版本 · 可覆写         │
        │  mcp/      工具协议     工具注册 · JSON-RPC · stdio/HTTP│
        └──────────────────────────────────────────────────────┘
                                ▲
        ┌───────────────────────┴──────────────────────────────┐
        │  Agent 运行时底座（独立仓库 ecom-agent-runtime）          │
        │  ReAct Loop · LangGraph 状态图 · 意图路由              │
        └──────────────────────────────────────────────────────┘
```

**依赖方向是单向的：业务依赖共享，共享不依赖业务。**
这是本包能独立成仓库的技术前提 —— 业务数据（商品、店铺指标、飞书凭证）通过
**依赖注入**进来，共享层只定义契约，不认识任何业务数据结构。

---

## 二、五块能力

### 1. LLM 网关：多厂商统一接入 + 复杂度路由

四家厂商（阿里云百炼 / DeepSeek / OpenAI / Ollama）统一走 OpenAI 兼容协议，
**换厂商只改一个环境变量，业务代码零改动**。

```python
from ecom_shared import SharedCluster

cluster = SharedCluster.build()
model, tier, score = cluster.gateway.resolve_route("帮我分析竞品定价策略")
# → ('qwen-max', 'heavy', 6)   复杂度得分 > 0，走重模型

answer, meta = await cluster.gateway.complete(system_prompt, user_prompt)
```

路由是**规则路由**（关键词加权 + 文本长度），不是训练出来的分类器 —— 这点在代码注释
和「已知边界」里都写明了，不假装是模型能力。它带来的成本下降来自
**选对模型**，不是靠压 token 数。

三级 JSON 容错解析解决的是"模型输出不可信"：

| 级别 | 处理 | 典型场景 |
|---|---|---|
| 1 | 直接解析 | 模型规规矩矩返回 JSON |
| 2 | 剥掉 Markdown 围栏 | 模型包了一层 ` ```json ` |
| 3 | 括号配对截取首个对象 | 模型在 JSON 前后加了"好的，分析如下：" |

三级都失败抛 `ModelOutputError` → 502。**宁可报错，也不把脏数据透传给前端。**

### 2. RAG：语义切分 + 三种可切换的检索后端

切分策略是 RAG 面试的第一问：

- **段落优先** —— 固定长度切分会把完整段落拦腰截断
- **长段按句切** —— 超阈值时按句滚动累积
- **重叠只补在切断处** —— 整段独立成块时**不**拼接上一段

第三条最容易写错：无差别给每个块都加前缀，会导致知识库里几乎每个切片都被上一段污染，
向量被无关文本稀释。切分数量完全正常、检索也不报错，**不看切片原文根本发现不了**。

重排用轻量融合分，解决"字面命中却被排到后面"：

```python
hits = await cluster.search("ACOS 过高怎么优化", domain="ads")
# 融合分 = 0.7 × 语义相似度 + 0.3 × 词面重叠率
```

`α=0.7` 不是拍脑袋的 —— `alpha` 是参数，离线评测脚本可以扫参看命中率怎么变
（见 `scripts/eval_retrieval.py`）。

三种检索后端，同一套接口：

| 后端 | 用途 | 依赖 |
|---|---|---|
| `chroma` | 生产默认，HNSW 索引 + 持久化 | `pip install chromadb` |
| `numpy` | 零外部依赖的暴力检索，几万条内比 HNSW 还快 | 需厂商提供 embedding |
| `lexical` | BM25 纯词法，**不需要任何向量** | 无 —— CI / 离线演示靠它 |

`lexical` 不只是"能用就行"的替身：它也是真实的降级方案 ——
query 里有明确型号、SKU、专业名词时，BM25 往往比向量检索更稳。

### 3. 记忆：短期会话 + 长期事实

```python
cluster.session_memory.append("s1", "user", "帮我看看这款降噪耳机")
cluster.session_memory.append("s1", "user", "它有什么优惠")
# 「它」要靠历史才能解析出指的是哪款耳机
```

会话记忆**按轮数截断而不是按字符数** —— 按字符截会把最后一轮提问拦腰砍掉，
模型收到半句话直接跑偏。

长期记忆**不用向量检索**，这是有意的取舍：事实条目是短句、量级几百到几千，
远达不到需要 ANN 索引的规模。这个规模下一次 embedding 调用的收益，
还不如直接做「词面重叠 × 时间衰减」—— 而且 embedding 调用本身是链路里最容易挂的一环。

```python
lt.remember("主营站点", "北美站", scope="shop_001")
lt.recall("站点", scope="shop_001")     # 半衰期 14 天，过期事实排不过新事实
```

### 4. Prompt 注册中心

Prompt 是 AI 应用里改得最频繁的东西，把它当资产管：

```python
cluster.prompts.render("rag_answer", persona="售前顾问", question="…", context="…")
```

两个刻意的设计：

- **缺失变量不抛异常**，保留 `{variable}` 字面量。前者一眼能发现，后者要翻日志。
- **同名注册默认报错**，显式 `override=True` 才覆盖。静默覆盖会导致
  "改 A 的结果 B 变了"，这是最难查的一类 bug。

### 5. MCP 工具协议

**Function Calling 和 MCP 差在哪：**

- **Function Calling 是厂商私有的** —— OpenAI / Anthropic / 百炼的函数格式互不兼容，
  换厂商工具要重写。
- **MCP 是协议标准** —— 工具的 `tools/list` 与 `tools/call` 走同一套 JSON-RPC，
  工具实现一次，任何 MCP 客户端都能接。
- 一句话：**Function Calling 是"某个模型的工具"，MCP 是"所有模型的工具"。**

```python
resp = await cluster.tools.handle({
    "jsonrpc": "2.0", "id": 1, "method": "tools/call",
    "params": {"name": "rag_search", "arguments": {"query": "ACOS 过高"}},
})
```

内置 8 个工具：`rag_search` / `memory_recall` / `memory_remember` / `session_context` /
`product_query` / `metrics_query` / `notify` / `alert`。

其中 `product_query` 与 `metrics_query` 是**注入式**的 —— 共享层只定义契约，
业务实现由业务仓传进来。没注入时明确返回"未注入"而不是空 dict，
**不能让 Agent 把"没接"误判成"查了但没有"**。

一条关键约定：**工具失败不抛异常**，返回 `isError: true` 的成功 RPC 响应。
因为 Agent 循环需要把失败原因当成观察结果喂回模型，让它换个策略 ——
抛异常会让整轮对话中断。

---

## 三、快速开始

```bash
git clone <repo> && cd ecom-agent-shared
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -e ".[dev,chroma]"

python demo.py                   # 离线演示，不需要任何 Key
pytest                           # 137 个用例，全离线
```

接真实模型：复制 `.env.example` 为 `.env`，填 `LLM_API_KEY` 即可。**代码零改动。**

作为库使用（`name @ git+url` 这种写法，`pip install git+url` 的旧写法已废弃、会告警）：

```bash
pip install "ecom-agent-shared @ git+https://github.com/rchzc/ecom-agent-shared.git@main"
```

---

## 四、目录结构

```
ecom_shared/
├── config.py           集中配置 + 五家厂商 preset + 复杂度关键词表
├── errors.py           类型化错误体系（业务仓共享同一套错误语义）
├── logging_setup.py    结构化 JSON 日志 + 请求 ID（contextvar 贯穿异步链路）
├── cluster.py          SharedCluster 门面：一次组装，业务侧全套可用
├── gateway/            LLM 网关
│   ├── router.py         复杂度路由 + 三级容错解析（纯函数，好测）
│   ├── llm.py            真实厂商接入 + 用量成本记账
│   └── mock.py           离线替身（也是显式 provider，不是隐式兜底）
├── rag/                RAG
│   ├── chunking.py       语义切分
│   ├── embedder.py       向量化（含本地降级路径）
│   ├── backends.py       chroma / numpy / lexical 三后端 + 工厂
│   ├── rerank.py         混合重排（纯本地、可复现）
│   └── service.py        链路编排
├── memory/             会话记忆 + 长期事实记忆（存储可插拔）
├── prompts/            Prompt 注册中心 + 3 个内置模板
└── mcp/                工具协议
    ├── registry.py       工具注册 + JSON-RPC 分帧
    ├── builtin.py        8 个内置工具（含依赖注入点）
    └── server.py         stdio / HTTP 传输
```

---

## 五、几个刻意的设计决定

| 决定 | 原因 |
|---|---|
| `mock` 是一个**显式 provider**，不是"没配 key 就降级" | 隐式降级会让人以为 key 配好了其实没生效 |
| 切分重叠只补在切断处 | 无差别补前缀会污染切片，且不报错、极难发现 |
| 长期记忆不用向量检索 | 规模不够，embedding 调用的成本 > 收益，且多一个易挂的环节 |
| 工具失败返回 `isError` 而不是抛异常 | Agent 循环要把失败当观察结果，抛异常会中断整轮 |
| 同名 Prompt / 工具默认不允许覆盖 | 静默覆盖会表现为"改 A 结果 B 变了" |
| 三种检索后端而不是硬绑定 chromadb | 让 CI 和离线演示有一条不依赖模型下载的确定路径 |
| 业务数据用依赖注入而不是直接 import | 保证依赖方向单向，本包才能独立存活 |

---

## 六、相关仓库

| 仓库 | 生态位置 |
|---|---|
| **[ecom-agent-runtime](https://github.com/rchzc/ecom-agent-runtime)** | Agent 运行时底座：ReAct Loop / LangGraph / 意图路由 / 轨迹自进化 |
| **[ecommerce-ai-workbench](https://github.com/rchzc/ecommerce-ai-workbench)** | 业务应用层：数据中台 / 售前咨询 / 内容运营 / 销售考核 |

三者同属「跨境电商 AI 电商工作台」生态，按层拆分，依赖方向：
**业务应用 → 运行时底座 → 共享集群（本仓库）**。

---

## 七、已知边界

写清楚没做什么，比假装都做了更可靠：

- **会话记忆存在进程内**，多实例部署或需要重启保留时必须换 Redis。
  接口已收敛成 `MemoryStore` 协议（`load` / `save` 两个方法），换实现不影响调用方。
- **用量统计不落库**，进程重启即清零。成本看板展示的是估算值，
  真实账单以厂商后台为准 —— `PRICE_TABLE` 里的单价只用于展示，不参与任何计费逻辑。
- **MCP HTTP 端点没有鉴权**，`serve_http()` 默认只绑 `127.0.0.1`。
  对外提供服务前必须先加鉴权。
- **规则路由依赖关键词表**，换业务领域需要重调词表。要更准可以先用小模型做一次分类，
  代价是多一次调用。
- **`numpy` 后端需要外部提供向量**，不做"没有向量就造一个假的"这种事 ——
  假向量会让检索退化成随机排序，比明确报错更糟。

---

## License

MIT
