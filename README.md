# 知序 (Zhixu)

> 中文编程语言。用中文写代码，与 AI 自然对话。

"知"是认知，"序"是程序——知序，让编程回归自然语言。

---

## 快速开始

```bash
# 运行 demo
python zhixu.py run demo.zx

# 也可以用启动脚本
zx run demo.zx

# 导出为标准 Python
zx export demo.zx demo.py

# 交互模式
zx repl
```

## 示例

```python
# demo.zx
输出("你好，知序！")

定义 计算平均值(数字列表):
    总 = 0
    循环 数 在 数字列表:
        总 = 总 + 数
    返回 总 / 长度(数字列表)

分数 = [92, 78, 85, 63]
结果 = 计算平均值(分数)
输出(f"平均分: {结果}")

如果 结果 >= 80:
    输出("优秀")
否则:
    输出("继续加油")
```

## 功能一览

### 基础语法

| 中文 | 对应 Python | 说明 |
|------|-------------|------|
| `如果` / `否则` / `否则如果` | if / else / elif | 条件分支 |
| `当` / `循环` / `在` | while / for / in | 循环 |
| `定义` / `返回` | def / return | 函数 |
| `类` | class | 类定义 |
| `引入` / `从` / `传入` | import / from / as | 模块导入 |
| `输出` / `输入` | print / input | 输入输出 |
| `尝试` / `捕获` / `最终` | try / except / finally | 异常处理 |
| `真` / `假` / `空` | True / False / None | 常量 |
| `与` / `或` / `非` | and / or / not | 逻辑运算 |
| `长度` / `范围` / `类型` | len / range / type | 内置函数 |
| `入口程序` | `__name__ == '__main__'` | 入口检测 |

### AI 原生操作

| 中文 | 功能 |
|------|------|
| `加载模型()` | 加载 AI 模型（自动检测 Ollama / OpenAI / HuggingFace） |
| `模型.推理()` | 模型推理 |
| `流式回答()` | 流式对话 |
| `搭建知识库()` | 构建 RAG 知识库 |
| `微调模型()` | LoRA 微调 |
| `量化模型()` | 模型量化 |
| `部署服务()` | 部署 API 服务 |
| `清洗数据()` | 数据集清洗 |
| `嵌入向量()` | 文本嵌入 |

## 项目结构

```
zhixu/
├── zhixu.py          # 编译器/运行时（词法分析 + 代码生成 + CLI）
├── zhixu_utils.py    # AI 工具库（Ollama/OpenAI/HuggingFace 后端）
├── demo.zx           # 基础示例
├── demo_实用.zx      # 实用能力示例
├── zx.bat            # Windows 启动脚本
├── pyproject.toml    # 包配置
└── README.md
```

## 文件扩展名

知序代码使用 `.zx` 扩展名。

## License

MIT
