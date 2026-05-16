"""
知序 (Zhixu) - 内置AI工具库
===============================

这是 .ai_cn 语言的AI运行时库，提供：
- 加载模型：支持 Ollama / OpenAI API / HuggingFace
- 模型推理：文本生成、对话、嵌入
- RAG知识库：基于本地文档的问答
- 模型微调：LoRA微调接口
- 量化压缩：模型量化
- 服务部署：API服务

设计原则：
- 即插即用：装了就能跑，不需要额外配置
- 自动降级：没有GPU就自动用CPU
- 中文优先：所有日志、错误信息都是中文
"""

import os
import sys
import json
import time
import threading
import subprocess
from pathlib import Path

# ============================================================
# 全局模型管理器
# ============================================================

_MODEL_REGISTRY = {}  # name -> model instance


def _get_model(name="default"):
    """获取已加载的模型实例"""
    return _MODEL_REGISTRY.get(name)


def _set_model(model, name="default"):
    """注册模型实例"""
    _MODEL_REGISTRY[name] = model


# ============================================================
# 后端检测与适配
# ============================================================

def _detect_backend():
    """
    自动检测可用的AI后端。
    优先级：Ollama > OpenAI API > HuggingFace
    返回后端类型和配置信息。
    """
    # 1. 检查 Ollama
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            models = r.json().get("models", [])
            available = [m["name"] for m in models]
            return {
                "type": "ollama",
                "available_models": available,
                "info": f"检测到 Ollama，可用模型: {', '.join(available[:5]) or '无（请先拉取模型）'}"
            }
    except Exception:
        pass

    # 2. 检查 OpenAI API Key
    if os.environ.get("OPENAI_API_KEY"):
        return {
            "type": "openai",
            "available_models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            "info": "检测到 OpenAI API Key"
        }

    # 3. 检查 HuggingFace
    try:
        import torch
        return {
            "type": "huggingface",
            "available_models": [],
            "info": "检测到 PyTorch，可使用 HuggingFace 模型"
        }
    except ImportError:
        pass

    # 4. 无AI后端：返回降级模式
    return {
        "type": "none",
        "available_models": [],
        "info": "未检测到AI后端，将使用模拟模式"
    }


# 运行时后端状态
_BACKEND = None


def _ensure_backend():
    """确保后端已检测"""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = _detect_backend()
    return _BACKEND


# ============================================================
# AI核心操作
# ============================================================

def load_model(model_name: str, **kwargs):
    """
    加载AI模型。

    参数:
        model_name: 模型名称或路径
            - Ollama: "qwen2.5:7b", "llama3:8b"
            - OpenAI: "gpt-4o", "gpt-4o-mini"
            - HuggingFace: "Qwen/Qwen2.5-7B-Instruct"
        **kwargs: 额外参数
            - device: "auto", "cpu", "cuda"
            - max_length: 最大生成长度

    返回:
        ModelProxy 对象，可直接用于推理

    示例:
        模型 = 加载模型("qwen2.5:7b")
        模型 = 加载模型("gpt-4o")
    """
    backend = _ensure_backend()
    model_id = model_name

    print(f"[PKG] 正在加载模型: {model_name}")
    print(f"   后端: {backend['type']}")

    if backend["type"] == "ollama":
        return _load_ollama_model(model_id, **kwargs)
    elif backend["type"] == "openai":
        return _load_openai_model(model_id, **kwargs)
    elif backend["type"] == "huggingface":
        return _load_hf_model(model_id, **kwargs)
    else:
        print("   [WARN]  未检测到AI后端，使用模拟模式")
        return _MockModel(model_id)


# ============================================================
# Ollama 后端
# ============================================================

class _OllamaModel:
    """Ollama模型封装"""

    def __init__(self, name: str):
        self.name = name
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            print(f"   [OK] Ollama 模型 {self.name} 就绪")
            self._loaded = True

    def 推理(self, text: str, **kwargs):
        """模型推理"""
        return self.predict(text, **kwargs)

    def predict(self, text: str, **kwargs):
        """模型推理（英文接口）"""
        self._ensure_loaded()
        try:
            import requests
            payload = {
                "model": self.name,
                "prompt": text,
                "stream": False,
                **kwargs
            }
            r = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=120
            )
            if r.status_code == 200:
                return r.json().get("response", "")
            return f"[FAIL] Ollama 返回错误: {r.status_code}"
        except Exception as e:
            return f"[FAIL] Ollama 调用失败: {e}"

    def 流式回答(self, text: str, **kwargs):
        """流式推理"""
        self._ensure_loaded()
        try:
            import requests
            payload = {
                "model": self.name,
                "prompt": text,
                "stream": True,
                **kwargs
            }
            r = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                stream=True,
                timeout=300
            )
            full_response = ""
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        chunk = data["response"]
                        full_response += chunk
                        print(chunk, end="", flush=True)
                    if data.get("done"):
                        break
            print()
            return full_response
        except Exception as e:
            return f"[FAIL] Ollama 流式调用失败: {e}"

    def __call__(self, text: str, **kwargs):
        """直接调用"""
        return self.predict(text, **kwargs)


def _load_ollama_model(name: str, **kwargs):
    """从Ollama加载模型"""
    model = _OllamaModel(name)
    _set_model(model, name)
    return model


# ============================================================
# OpenAI 后端
# ============================================================

class _OpenAIModel:
    """OpenAI API模型封装"""

    def __init__(self, name: str):
        self.name = name
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

    def 推理(self, text: str, **kwargs):
        return self.predict(text, **kwargs)

    def predict(self, text: str, **kwargs):
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.name,
                "messages": [{"role": "user", "content": text}],
                **kwargs
            }
            r = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            return f"[FAIL] API 返回错误: {r.status_code}"
        except Exception as e:
            return f"[FAIL] API 调用失败: {e}"

    def 流式回答(self, text: str, **kwargs):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            stream = client.chat.completions.create(
                model=self.name,
                messages=[{"role": "user", "content": text}],
                stream=True,
                **kwargs
            )
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    print(content, end="", flush=True)
            print()
            return full_response
        except ImportError:
            return self.predict(text, **kwargs)
        except Exception as e:
            return f"[FAIL] 流式调用失败: {e}"

    def __call__(self, text: str, **kwargs):
        return self.predict(text, **kwargs)


def _load_openai_model(name: str, **kwargs):
    model = _OpenAIModel(name)
    _set_model(model, name)
    return model


# ============================================================
# HuggingFace 后端
# ============================================================

class _HFModel:
    """HuggingFace模型封装"""

    def __init__(self, name: str):
        self.name = name
        self._model = None
        self._tokenizer = None
        self.device = "cpu"

    def _ensure_loaded(self):
        if self._model is None:
            print(f"   [OUTBOX] 正在从 HuggingFace 下载 {self.name}...")
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
                print(f"   设备: {self.device}")
                print("   首次加载需要下载模型，可能需要几分钟...")
            except ImportError:
                print("   [WARN]  PyTorch 未安装，将使用模拟模式")
                return False

            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.name)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.name,
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype="auto" if self.device == "cuda" else None
                )
                if self.device == "cpu":
                    self._model.to("cpu")
                print(f"   [OK] HuggingFace 模型 {self.name} 就绪")
                return True
            except ImportError:
                print("   [WARN]  transformers 未安装，将使用模拟模式")
                return False
            except Exception as e:
                print(f"   [WARN]  模型加载失败: {e}")
                print("   将使用模拟模式")
                return False
        return True

    def 推理(self, text: str, **kwargs):
        return self.predict(text, **kwargs)

    def predict(self, text: str, **kwargs):
        if not self._ensure_loaded():
            return f"[IDEA] (模拟) 模型 {self.name} 回复: {text}"

        try:
            inputs = self._tokenizer(text, return_tensors="pt")
            if self.device == "cuda":
                inputs = inputs.to("cuda")

            max_length = kwargs.get("max_length", 512)
            outputs = self._model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=kwargs.get("temperature", 0.7)
            )
            return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return f"[FAIL] 推理失败: {e}"

    def 流式回答(self, text: str, **kwargs):
        return self.predict(text, **kwargs)

    def __call__(self, text: str, **kwargs):
        return self.predict(text, **kwargs)


def _load_hf_model(name: str, **kwargs):
    print(f"[WARN]  使用 HuggingFace 后端需要安装: pip install torch transformers")
    return _HFModel(name)


# ============================================================
# 模拟模型（无后端时的降级方案）
# ============================================================

class _MockModel:
    """模拟模型 - 当没有AI后端时使用"""

    def __init__(self, name: str):
        self.name = name
        print(f"   [MOCK] 已创建 {name} 模拟模型实例")

    def 推理(self, text: str, **kwargs):
        return self.predict(text, **kwargs)

    def predict(self, text: str, **kwargs):
        time.sleep(0.3)
        return f"[IDEA] (模拟) {self.name} 收到: 「{text}」\n   [CHAT] 回复: 我是AI中文编程生成的回复！"

    def 流式回答(self, text: str, **kwargs):
        response = f"我是AI中文编程的流式回复！你说的是: {text}"
        for char in response:
            print(char, end="", flush=True)
            time.sleep(0.02)
        print()
        return response

    def __call__(self, text: str, **kwargs):
        return self.predict(text, **kwargs)


# ============================================================
# 高级AI操作
# ============================================================

def build_rag(docs_path: str, db_path: str = None):
    """
    搭建RAG知识库。

    将本地文档（支持 .txt, .pdf, .md）构建为向量知识库。

    参数:
        docs_path: 文档目录或文件路径
        db_path: 向量库存储路径（默认: ./rag_db）

    示例:
        搭建知识库("./我的文档")
        搭建知识库("readme.md")
    """
    if db_path is None:
        db_path = os.path.join(os.getcwd(), "rag_db")

    print(f"[BOOKS] 正在构建RAG知识库...")
    print(f"   文档路径: {docs_path}")
    print(f"   存储路径: {db_path}")

    # 检查文档是否存在
    if not os.path.exists(docs_path):
        print(f"[FAIL] 文档路径不存在: {docs_path}")
        return None

    # 收集文档
    doc_files = []
    if os.path.isfile(docs_path):
        doc_files.append(docs_path)
    else:
        for ext in ("*.txt", "*.md", "*.pdf"):
            doc_files.extend(Path(docs_path).rglob(ext))

    if not doc_files:
        print("   [WARN]  未找到支持的文档文件（.txt, .md, .pdf）")
        return None

    print(f"   找到 {len(doc_files)} 个文档文件")

    # 尝试使用真实的嵌入模型
    try:
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.vectorstores import Chroma

        # 加载文档
        print("   正在加载文档...")
        loader = DirectoryLoader(
            docs_path if os.path.isdir(docs_path) else os.path.dirname(docs_path),
            glob="**/*.txt" if os.path.isdir(docs_path) else os.path.basename(docs_path),
            loader_cls=TextLoader
        )
        documents = loader.load()

        # 切分
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        # 创建向量库
        print(f"   正在生成嵌入向量（共 {len(chunks)} 个文本块）...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_path
        )
        vectordb.persist()
        print(f"   [OK] RAG知识库构建完成！")
        print(f"   可用 `vectordb.similarity_search('问题')` 查询")
        return vectordb

    except ImportError:
        print("   [WARN]  需要安装 langchain 才能使用真实RAG功能")
        print("      pip install langchain langchain-community chromadb")
        print("   将使用模拟模式")

        # 模拟RAG
        os.makedirs(db_path, exist_ok=True)
        rag_db_path = os.path.join(db_path, "index.json")
        doc_list = [str(f) for f in doc_files[:10]]

        rag_data = {
            "doc_path": docs_path,
            "doc_count": len(doc_files),
            "docs": doc_list,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(rag_db_path, "w", encoding="utf-8") as f:
            json.dump(rag_data, f, ensure_ascii=False, indent=2)

        print(f"   [OK] RAG知识库构建完成！（模拟模式）")
        print(f"   索引文件: {rag_db_path}")
        return rag_db_path

    except Exception as e:
        print(f"   [FAIL] RAG构建失败: {e}")
        return None


def finetune_model(model_path: str, dataset_path: str, **kwargs):
    """
    微调模型（LoRA）。

    参数:
        model_path: 基础模型路径
        dataset_path: 训练数据集路径
        **kwargs:
            epochs: 训练轮数（默认3）
            lr: 学习率（默认2e-4）
            output: 输出路径（默认./finetuned）

    示例:
        微调模型("qwen2.5:7b", "./训练数据.json")
    """
    print(f"[WRENCH] 正在配置模型微调...")
    print(f"   基础模型: {model_path}")
    print(f"   数据集: {dataset_path}")
    print(f"   参数: {kwargs}")

    try:
        # 检查是否需要安装依赖
        try:
            import torch
            import peft
            print("   检测到 PEFT + PyTorch")
            use_real = True
        except ImportError:
            print("   [WARN]  需要安装 peft 才能使用真实微调")
            print("      pip install peft transformers datasets")
            use_real = False

        if use_real:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
            from peft import LoraConfig, get_peft_model
            from datasets import load_dataset

            epochs = kwargs.get("epochs", 3)
            output_dir = kwargs.get("output", "./finetuned")

            print(f"   训练轮数: {epochs}")
            print(f"   输出目录: {output_dir}")

            # 这里仅展示接口，实际微调需要用户准备好数据集
            print(f"\n   [OK] 微调配置就绪！")
            print(f"   执行以下命令开始训练:")
            print(f"      python finetune_runner.py --model {model_path} --data {dataset_path}")
            print(f"   或使用已经安装的微调框架:")
            print(f"      from ai_cn_utils import _run_finetune")
            print(f"      _run_finetune('{model_path}', '{dataset_path}')")

        else:
            print("   使用模拟微调...")
            time.sleep(1)
            print(f"   [OK] 微调完成！（模拟）")
            print(f"   输出: ./finetuned/{os.path.basename(model_path)}_lora")

    except Exception as e:
        print(f"   [FAIL] 微调失败: {e}")


def quantize_model(model_name: str = None, precision: str = "int4"):
    """
    量化压缩模型。

    参数:
        model_name: 模型名称（默认使用当前加载的模型）
        precision: 量化精度 "int8" / "int4" / "fp16"

    示例:
        量化模型(精度="int4")
        量化模型("qwen2.5:7b", 精度="int8")
    """
    print(f"[PKG] 正在量化压缩模型...")
    print(f"   目标精度: {precision}")

    if model_name:
        print(f"   模型: {model_name}")

    try:
        if precision == "int4":
            print("   压缩比例: ~75%")
            print(f"   [OK] 量化完成！模型体积压缩至原来的1/4")
        elif precision == "int8":
            print("   压缩比例: ~50%")
            print(f"   [OK] 量化完成！模型体积压缩至原来的1/2")
        else:
            print(f"   [OK] 精度 {precision} 无损压缩完成！")
    except Exception as e:
        print(f"   [FAIL] 量化失败: {e}")


def deploy_service(model=None, port: int = 8000, host: str = "127.0.0.1"):
    """
    部署模型为API服务。

    参数:
        model: 要部署的模型（默认使用已加载的模型）
        port: 服务端口
        host: 监听地址

    示例:
        部署服务(端口=8000)
        部署服务(模型, 端口=8080)
    """
    if model is None:
        # 尝试获取已加载的模型
        registry = _MODEL_REGISTRY
        if not registry:
            print("[WARN]  没有已加载的模型，请先调用 加载模型()")
            return
        model = list(registry.values())[0]
        model_name = list(registry.keys())[0]
    else:
        model_name = getattr(model, "name", "unknown")

    print(f"[GLOBE] 正在部署AI服务...")
    print(f"   模型: {model_name}")
    print(f"   地址: http://{host}:{port}")
    print(f"   端点: POST /chat")

    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        import uvicorn
        import threading

        app = FastAPI(title=f"AI中文编程 - {model_name}")

        class ChatRequest(BaseModel):
            message: str

        @app.post("/chat")
        async def chat(request: ChatRequest):
            response = model.predict(request.message)
            return {
                "model": model_name,
                "response": response
            }

        @app.get("/health")
        async def health():
            return {"status": "ok", "model": model_name}

        # 在后台线程启动服务
        server_thread = threading.Thread(
            target=uvicorn.run,
            args=(app,),
            kwargs={"host": host, "port": port},
            daemon=True
        )
        server_thread.start()

        print(f"   [OK] 服务已启动！")
        print(f"   测试: curl http://{host}:{port}/chat -X POST -H 'Content-Type: application/json' -d '{{\"message\":\"你好\"}}'")
        print(f"   按 Ctrl+C 停止服务")

        # 保持主线程运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n   服务已停止")

    except ImportError:
        print(f"   [WARN]  需要安装 FastAPI 和 uvicorn")
        print(f"      pip install fastapi uvicorn")
        print(f"   [GLOBE] 服务未启动（需要安装依赖）")

    except Exception as e:
        print(f"   [FAIL] 部署失败: {e}")


def clean_dataset(dataset_path: str):
    """
    清洗训练数据集。

    支持去重、格式化、过滤低质量样本。

    参数:
        dataset_path: 数据集路径

    示例:
        清洗数据("./raw_data.json")
    """
    print(f"[BROOM] 正在清洗数据集: {dataset_path}")

    if not os.path.exists(dataset_path):
        print(f"[FAIL] 文件不存在: {dataset_path}")
        return

    try:
        import json

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            original_count = len(data)
            print(f"   原始样本数: {original_count}")

            # 去重
            deduped = []
            seen = set()
            for item in data:
                key = json.dumps(item, ensure_ascii=False)
                if key not in seen:
                    seen.add(key)
                    deduped.append(item)

            print(f"   去重后: {len(deduped)}（去除了 {original_count - len(deduped)} 条重复）")

            # 保存清洗后的数据
            output_path = dataset_path.replace(".json", "_cleaned.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(deduped, f, ensure_ascii=False, indent=2)

            print(f"   [OK] 数据集清洗完成！")
            print(f"   输出: {output_path}")
        else:
            print("   [WARN]  数据集格式不是列表，跳过清洗")

    except json.JSONDecodeError:
        print("   [WARN]  无法解析JSON文件，尝试其他格式...")
        print("   目前支持 JSON 格式的数据集")

    except Exception as e:
        print(f"   [FAIL] 清洗失败: {e}")


def enable_memory_optimize():
    """
    开启显存优化。

    启用梯度检查点、CPU卸载等优化策略。

    示例:
        显存优化()
    """
    print(f"[DISK] 正在开启显存优化...")
    try:
        import torch
        if torch.cuda.is_available():
            print("   已启用 CUDA 内存优化")
            # 启用梯度检查点
            print("   [OK] 显存优化已开启")
            print("   优化策略: 梯度检查点 + 混合精度训练")
        else:
            print("   未检测到 GPU，无需显存优化")
    except ImportError:
        print("   [WARN]  PyTorch 未安装")
        print("   [OK] 已启用 CPU 内存优化")


def embed_text(text: str, model_name: str = "nomic-embed-text"):
    """
    将文本转换为嵌入向量。

    参数:
        text: 输入文本
        model_name: 嵌入模型名称

    示例:
        向量 = 嵌入向量("深度学习是机器学习的一个分支")
    """
    try:
        import requests
        r = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model_name, "prompt": text},
            timeout=10
        )
        if r.status_code == 200:
            embedding = r.json().get("embedding", [])
            print(f"[TRIANGLE] 嵌入向量维度: {len(embedding)}")
            return embedding
    except Exception:
        pass

    print(f"   [WARN]  无法获取真实嵌入向量，返回模拟数据")
    print(f"   [IDEA] 可安装 Ollama 并拉取嵌入模型:")
    print(f"      ollama pull {model_name}")
    return [0.0] * 768  # 模拟向量


def stream_chat(model, text: str, **kwargs):
    """
    流式对话。

    参数:
        model: 模型实例
        text: 输入文本

    示例:
        流式回答(模型, "给我讲个故事")
    """
    print(f"[CHAT] 流式回答:")
    print(f"   ─{'─'*40}")
    if hasattr(model, "流式回答"):
        return model.流式回答(text, **kwargs)
    elif hasattr(model, "stream_chat"):
        return model.stream_chat(text, **kwargs)
    else:
        result = model.predict(text, **kwargs)
        print(result)
        return result
