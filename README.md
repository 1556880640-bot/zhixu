# AI-CN 原生中文编程语言

> 用中文写 AI 代码。和 AI 对话用中文，写代码也用中文，省掉翻译损耗。

---

## 核心理念

现在的 AI 编程流程是：
```
你的想法（中文）→ Prompt（中文）→ AI 生成代码（英文）→ 执行
```

AI-CN 的目标是：
```
你的想法（中文）→ Prompt（中文）→ AI 生成代码（中文）→ 执行
```

省掉「中文想法 → 英文代码」这道不必要的翻译损耗。

## 快速开始

```bash
# 执行中文代码文件
python ai_cn.py run demo.ai_cn

# 导出为标准 Python
python ai_cn.py export demo.ai_cn demo.py

# 交互模式
python ai_cn.py repl
```

## 示例

```python
# demo.ai_cn
引入 torch
定义 测试AI功能():
    模型 = 加载模型("qwen2.5:7b")
    结果 = 模型.推理("用一句话解释深度学习")
    输出(结果)
    
    搭建知识库("./公司文档")
    部署服务(端口=8080)

如果 入口程序:
    测试AI功能()
```

## 功能

### 基础语法（全中文关键字）

| 中文 | Python |
|------|--------|
| `如果` | `if` |
| `否则` | `else` |
| `否则如果` | `elif` |
| `当` | `while` |
| `循环` | `for` |
| `在` | `in` |
| `定义` | `def` |
| `返回` | `return` |
| `类` | `class` |
| `引入` | `import` |
| `从` | `from` |
| `输出` | `print` |
| `真/假/空` | `True/False/None` |
| `入口程序` | `__name__ == '__main__'` |
| `与/或/非` | `and/or/not` |
| `尝试/捕获/最终` | `try/except/finally` |

### AI 原生操作

| 中文 | 功能 |
|------|------|
| `加载模型()` | 加载 AI 模型（自动检测 Ollama / OpenAI / HuggingFace） |
| `模型.推理()` | 模型推理 |
| `流式回答()` | 流式对话（逐字输出） |
| `搭建知识库()` | 构建 RAG 知识库 |
| `微调模型()` | LoRA 微调 |
| `量化模型()` | 模型量化压缩 |
| `部署服务()` | 部署为 API 服务 |
| `清洗数据()` | 数据集清洗 |
| `嵌入向量()` | 文本嵌入 |
| `显存优化()` | 显存优化 |

## 后端支持

AI-CN 自动检测可用的 AI 后端：

| 后端 | 条件 | 命令 |
|------|------|------|
| **Ollama** | 本地安装 Ollama | `ollama pull qwen2.5:7b` |
| **OpenAI** | 设置 `OPENAI_API_KEY` | - |
| **HuggingFace** | 安装 PyTorch | `pip install torch transformers` |
| **模拟模式** | 以上均无（自动降级） | - |

## 应用场景

- **AI 应用原型开发**：快速搭建模型加载、RAG、部署的全流程
- **教学入门**：零基础学编程，无需先学英文关键字
- **个人自动化脚本**：自己用的小工具，怎么写顺手就怎么写
- **数据处理**：数据清洗、分析脚本

## 不适合的场景

- 大型生产系统（生态工具链待完善）
- 团队协作项目（需统一语言规范）
- 已有成熟 Python 项目的场景

## 项目结构

```
ai-cn/
├── ai_cn.py         # 编译器/解释器（词法分析 + 代码生成 + CLI）
├── ai_cn_utils.py   # AI 工具库（Ollama/OpenAI/HuggingFace 后端）
├── demo.ai_cn       # 示例代码
├── pyproject.toml   # 包配置
└── README.md
```

## License

MIT
