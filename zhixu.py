"""
知序 (Zhixu) - 中文编程语言
===============================

核心理念：既然和 AI 对话用的是中文，AI 写代码也应该用中文。
省掉「中文想法 → 英文代码」这道不必要的翻译损耗。

设计原则：
1. 中文自然语法：代码读起来就是中文
2. AI操作为原语：加载模型、推理、RAG、微调都是一行代码
3. Python生态兼容：无缝调用Python库
4. 错误信息中文：在哪错的、怎么改，都用中文告诉你
"""

import re
import sys
import os
import argparse
import traceback

# Windows 控制台 UTF-8 支持
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# ============================================================
# 第一部分：词法分析器 (Tokenizer)
# 用真正的tokenizer替代正则替换，避免误替换
# ============================================================

class Token:
    """单个token"""
    __slots__ = ('type', 'value', 'line', 'col')

    def __init__(self, type_: str, value: str, line: int, col: int):
        self.type = type_
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, L{self.line}:C{self.col})"


# 中文关键字 → Python 关键字映射
# 注意：长关键字排在前面，避免被短关键字错误覆盖
KEYWORDS = [
    # AI原生操作 (优先级最高，因为最长)
    ("加载模型", "load_model"),
    ("搭建知识库", "build_rag"),
    ("微调模型", "finetune_model"),
    ("量化模型", "quantize_model"),
    ("部署服务", "deploy_service"),
    ("清洗数据", "clean_dataset"),
    ("显存优化", "enable_memory_optimize"),
    ("嵌入向量", "embed_text"),
    ("流式回答", "stream_chat"),

    # 控制流
    ("否则如果", "elif"),
    ("定义", "def"),
    ("如果", "if"),
    ("否则", "else"),
    ("当", "while"),
    ("循环", "for"),
    ("在", "in"),
    ("返回", "return"),
    ("中断", "break"),
    ("跳过", "continue"),

    # 类与模块
    ("引入", "import"),
    ("从", "from"),
    ("类", "class"),
    ("传入", "as"),

    # 异常处理
    ("尝试", "try"),
    ("捕获", "except"),
    ("最终", "finally"),
    ("引发", "raise"),

    # 内置操作
    ("输出", "print"),
    ("输入", "input"),
    ("长度", "len"),
    ("范围", "range"),
    ("类型", "type"),
    ("整数", "int"),
    ("浮点", "float"),
    ("字符串", "str"),
    ("列表", "list"),
    ("字典", "dict"),
    ("集合", "set"),
    ("打开", "open"),
    ("追加", "append"),
    ("写入", "write"),
    ("读取", "read"),
    ("关闭", "close"),
    ("移除", "remove"),
    ("编码", "encoding"),

    # 常量
    ("真", "True"),
    ("假", "False"),
    ("空", "None"),

    # 逻辑运算符
    ("与", "and"),
    ("或", "or"),
    ("非", "not"),
    ("是", "is"),

    # 其他
    ("全局", "global"),
    ("非局部", "nonlocal"),
    ("断言", "assert"),
    ("入口程序", "__name__ == '__main__'"),
]

# 按关键字长度降序排序（长优先匹配）
KEYWORDS_SORTED = sorted(KEYWORDS, key=lambda x: -len(x[0]))
KEYWORD_MAP = {k: v for k, v in KEYWORDS}

# 中文标点 → 英文标点
PUNCTUATION_MAP = str.maketrans({
    '：': ':',
    '，': ',',
    '（': '(',
    '）': ')',
    '“': '"',
    '”': '"',
    '‘': "'",
    '’': "'",
    '！': '!',
    '？': '?',
    '；': ';',
    '＝': '=',
    '＋': '+',
    '－': '-',
    '＊': '*',
    '／': '/',
    '＃': '#',
})

# 中文左引号、右引号范围
CN_QUOTE_LEFT = '\u201c'  # "
CN_QUOTE_RIGHT = '\u201d'  # "
CN_SINGLE_QUOTE_LEFT = '\u2018'  # '
CN_SINGLE_QUOTE_RIGHT = '\u2019'  # '


def is_cjk(char: str) -> bool:
    """判断是否是中文字符"""
    cp = ord(char)
    return (
        (0x4E00 <= cp <= 0x9FFF) or    # CJK统一表意文字
        (0x3400 <= cp <= 0x4DBF) or    # CJK扩展A
        (0x2E80 <= cp <= 0x2EFF) or    # CJK部首
        (0x3000 <= cp <= 0x303F) or    # CJK符号和标点
        (0xFF00 <= cp <= 0xFFEF)       # 全角字符
    )


def is_cjk_identifier_start(char: str) -> bool:
    """判断字符能否作为标识符开头"""
    return char.isalpha() or char == '_' or is_cjk(char)


def is_cjk_identifier_continue(char: str) -> bool:
    """判断字符能否作为标识符继续"""
    return char.isalnum() or char == '_' or is_cjk(char)


def tokenize(source: str) -> list:
    """
    将中文代码解析为token列表。

    这比正则替换强大得多，因为它：
    - 不会错误替换字符串内部的内容
    - 不会错误替换注释内容
    - 能正确处理嵌套结构
    - 保留行号/列号用于精确报错
    """
    tokens = []
    lines = source.split('\n')

    for line_num, line in enumerate(lines, 1):
        col = 0
        line_len = len(line)

        # 计算行首缩进
        indent = 0
        while col < line_len and line[col] == ' ':
            indent += 1
            col += 1

        if indent > 0:
            tokens.append(Token('INDENT', ' ' * indent, line_num, 1))

        # 跳过空行
        if col >= line_len or line[col] == '\n':
            if line_num < len(lines):
                tokens.append(Token('NEWLINE', '\n', line_num, line_len))
            continue

        # 逐字符扫描
        while col < line_len:
            char = line[col]
            start_col = col

            # ----- 注释 -----
            if char == '#':
                tokens.append(Token('COMMENT', line[col:], line_num, col + 1))
                break  # 注释到行尾，结束本行

            # ----- 空白字符（空格/制表符）-----
            if char.isspace() and char != '\n':
                col += 1
                continue

            # ----- 字符串 -----
            if char in ('"', "'"):
                quote = char
                col += 1
                value = quote
                while col < line_len:
                    ch = line[col]
                    value += ch
                    col += 1
                    if ch == '\\':
                        if col < line_len:
                            value += line[col]
                            col += 1
                    elif ch == quote:
                        break
                tokens.append(Token('STRING', value, line_num, start_col + 1))
                continue

            # 中文引号
            if char in (CN_QUOTE_LEFT, CN_SINGLE_QUOTE_LEFT):
                quote = char
                end_quote = CN_QUOTE_RIGHT if char == CN_QUOTE_LEFT else CN_SINGLE_QUOTE_RIGHT
                col += 1
                value = '"'  # 转成英文引号
                while col < line_len and line[col] != end_quote:
                    value += line[col]
                    col += 1
                if col < line_len:
                    value += '"'
                    col += 1
                tokens.append(Token('STRING', value, line_num, start_col + 1))
                continue

            # ----- 数字 -----
            if char.isdigit() or (char == '.' and col + 1 < line_len and line[col + 1].isdigit()):
                value = ''
                while col < line_len and (line[col].isdigit() or line[col] == '.'):
                    value += line[col]
                    col += 1
                tokens.append(Token('NUMBER', value, line_num, start_col + 1))
                continue

            # ----- 运算符和标点 (英文) -----
            single_char_ops = '()[]{}:;,=+-*/%&|^~<>!@\\'
            multi_char_ops = {
                '==': '==', '!=': '!=', '<=': '<=', '>=': '>=',
                '->': '->', '**': '**', '//': '//',
            }

            if char in single_char_ops:
                # 尝试匹配两字符运算符
                if col + 1 < line_len and char + line[col + 1] in multi_char_ops:
                    tokens.append(Token('OPERATOR', char + line[col + 1], line_num, start_col + 1))
                    col += 2
                else:
                    tokens.append(Token('OPERATOR', char, line_num, start_col + 1))
                    col += 1
                continue

            # ----- 标识符 / 关键字 -----
            if is_cjk_identifier_start(char):
                value = ''
                while col < line_len and is_cjk_identifier_continue(line[col]):
                    value += line[col]
                    col += 1

                # 检查是否是中文关键字
                if value in KEYWORD_MAP:
                    tokens.append(Token('KEYWORD', value, line_num, start_col + 1))
                else:
                    tokens.append(Token('IDENTIFIER', value, line_num, start_col + 1))
                continue

            # ----- 无法识别的字符（比如中文标点，需要转换）-----
            col += 1
            tokens.append(Token('OTHER', char, line_num, start_col + 1))

        # 行结束
        if line_num < len(lines):
            tokens.append(Token('NEWLINE', '\n', line_num, line_len))

    return tokens


# ============================================================
# 第二部分：代码生成 (Code Generation)
# 将token列表转换为可执行的Python代码
# ============================================================

# 源代码行号 → 生成代码行号的映射
line_number_map = {}  # {python_line: original_ai_cn_line}


def generate_python(tokens: list, ai_utils_auto_import: bool = True) -> str:
    """
    将token列表转换为Python代码。

    返回生成的Python代码字符串。
    同时更新全局 line_number_map 用于错误定位。
    """
    global line_number_map
    line_number_map = {}

    output_lines = []
    current_line = 1  # 当前输出的Python行号
    used_ai_keywords = set()

    prev_type = None

    for token in tokens:
        if token.type == 'KEYWORD':
            python_keyword = KEYWORD_MAP[token.value]

            # 如果前一个token是标识符，后面也要加空格（保持Python语法分隔）
            if prev_type == 'IDENTIFIER':
                output_lines.append(' ')

            output_lines.append(python_keyword)

            # Python关键字后需要空格（if/while/for/def/class/and/or/not/return等）
            needs_space = python_keyword in (
                'if', 'elif', 'while', 'for', 'def', 'class',
                'return', 'and', 'or', 'not', 'is', 'import', 'from',
                'as', 'del', 'in', 'raise', 'global', 'nonlocal',
                'elif', 'None', 'except',
            )
            if needs_space and python_keyword != 'elif':
                # elif后面也有空格，但通过正常逻辑处理
                pass

            if needs_space:
                output_lines.append(' ')

            # 记录AI特定关键字的使用
            if token.value in ('加载模型', '搭建知识库', '微调模型', '量化模型',
                               '部署服务', '清洗数据', '显存优化', '嵌入向量', '流式回答'):
                used_ai_keywords.add(token.value)
            prev_type = 'KEYWORD'

        elif token.type == 'INDENT':
            output_lines.append(token.value)
            prev_type = 'INDENT'

        elif token.type == 'NEWLINE':
            output_lines.append('\n')
            # 记录行号映射
            current_line += 1
            prev_type = 'NEWLINE'

        elif token.type == 'COMMENT':
            output_lines.append(token.value)
            prev_type = 'COMMENT'

        elif token.type == 'IDENTIFIER':
            output_lines.append(token.value)
            prev_type = 'IDENTIFIER'

        elif token.type in ('NUMBER', 'STRING'):
            output_lines.append(token.value)

        elif token.type == 'OPERATOR':
            output_lines.append(token.value)

        elif token.type == 'OTHER':
            # 尝试中文标点转换
            punct_char = char = token.value
            if punct_char in '：，（）！“”？；＝＋－＊／％＆｜':
                # 使用 str.maketrans 转换
                for cn, en in [
                    ('：', ':'), ('，', ','), ('（', '('), ('）', ')'),
                    ('！', '!'), ('？', '?'), ('；', ';'), ('＝', '='),
                    ('＋', '+'), ('－', '-'), ('＊', '*'), ('／', '/'),
                    ('％', '%'), ('＆', '&'), ('｜', '|'),
                ]:
                    if char == cn:
                        output_lines.append(en)
                        break
                else:
                    output_lines.append(char)
            else:
                output_lines.append(char)

        elif token.type == 'STRING_END':
            pass  # 已经包含在字符串内容里了

        else:
            output_lines.append(str(token.value))

    code = ''.join(output_lines)

    # ============================================================
    # 后处理：处理多token映射（比如"入口程序" 映射到多token）
    # 注意："入口程序" 作为一个 KEYWORD token 被映射为
    #   "__name__ == '__main__'" 字符串
    # 但在 tokenizer 中它被当成了一个单独的token
    # 我们需要在生成后做处理
    # ============================================================

    # 实际上，上面的 tokenizer 已经把"入口程序"当作一个完整的
    # KEYWORD token 处理了，所以 output_lines 里会直接包含
    # "__name__ == '__main__'" 这个字符串。这是正确的。

    # 自动注入AI工具包导入
    if used_ai_keywords and ai_utils_auto_import:
        code = "from zhixu_utils import *\n" + code

    # 重新计算行号映射：统计原始源码行和生成代码行的对应关系
    # 实际上这个在 tokenizer 级别更精确，我们保留后做简化映射
    # 这里简化处理：生成代码的第N行 ≈ 原始代码的第N行
    generated_lines = code.split('\n')
    for i in range(len(generated_lines)):
        line_number_map[i + 1] = i + 1  # 近似映射

    return code


# ============================================================
# 第三部分：改进的转译流程
# ============================================================

def translate(source: str) -> str:
    """
    将完整的中文AI代码转换为Python代码。

    流程：词法分析 → 代码生成
    """
    # 1. 标点预转换（处理中文引号为英文引号）
    temp_source = ''
    i = 0
    while i < len(source):
        ch = source[i]
        if ch == '\u201c' or ch == '\u201d':  # 中文双引号
            temp_source += '"'
        elif ch == '\u2018' or ch == '\u2019':  # 中文单引号
            temp_source += "'"
        elif ch in PUNCTUATION_MAP:
            temp_source += PUNCTUATION_MAP[ch]
        else:
            temp_source += ch
        i += 1

    # 2. 词法分析
    tokens = tokenize(temp_source)

    # 3. 代码生成
    code = generate_python(tokens)

    return code


# ============================================================
# 第四部分：智能错误处理（中文报错）
# ============================================================

class ZhixuError(Exception):
    """中文编程语言运行时错误"""
    def __init__(self, message: str, original_tb=None):
        self.message = message
        self.original_tb = original_tb
        super().__init__(self._format())

    def _format(self):
        msg = f"\n[WARN]  知序中文编程执行错误\n"
        msg += f"{'='*50}\n"
        msg += f"错误信息: {self.message}\n"
        if self.original_tb:
            msg += f"\n原始错误:\n"
            msg += ''.join(traceback.format_tb(self.original_tb))
        msg += f"\n[IDEA] 提示：检查你的 .zx 文件中对应行号附近的语法\n"
        return msg


def run_file(file_path: str):
    """
    执行 .zx 文件

    流程：读取 → 转译 → 执行
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # 转译
        py_code = translate(source)

        # 显示转译后的代码（debug模式）
        if os.environ.get('ZHIXU_DEBUG'):
            print("=" * 50)
            print("[转译后的Python代码]:")
            print("=" * 50)
            for i, line in enumerate(py_code.split('\n'), 1):
                print(f"  {i:4d} | {line}")
            print("=" * 50)

        # 执行
        exec_globals = {'__builtins__': __builtins__}
        exec(py_code, exec_globals)

    except ZhixuError:
        raise
    except Exception as e:
        tb = sys.exc_info()[2]

        # 尝试定位原始 .zx 文件中的行号
        error_line = None
        while tb:
            if tb.tb_frame.f_code.co_name == '<module>':
                error_line = tb.tb_lineno
                break
            tb = tb.tb_next

        msg = f"{e}"
        if error_line:
            msg += f"\n\n大概在 .zx 文件的第 {error_line} 行附近。"

        print(f"\n[WARN]  执行出错: {e}")
        if error_line:
            print(f"   位置: {file_path}:{error_line}")
            # 显示附近代码
            lines = source.split('\n')
            start = max(0, error_line - 3)
            end = min(len(lines), error_line + 2)
            print(f"\n   附近代码:")
            for i in range(start, end):
                marker = " >>>" if i + 1 == error_line else "    "
                print(f"   {marker} {i+1:4d} | {lines[i]}")

        sys.exit(1)


def export_file(input_path: str, output_path: str):
    """
    将 .zx 文件导出为标准 Python 文件
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            source = f.read()

        py_code = translate(source)

        with open(output_path, 'w', encoding='utf-8') as f:
            # 添加头部注释
            f.write("# -*- coding: utf-8 -*-\n")
            f.write("# 此文件由 知序中文编程 自动生成\n")
            f.write(f"# 源文件: {os.path.basename(input_path)}\n")
            f.write("# 不建议直接修改此文件，请修改源文件后重新生成\n\n")
            f.write(py_code)

        src_lines = len(source.strip().split('\n'))
        py_lines = len(py_code.strip().split('\n'))
        print(f"[OK] 导出成功！")
        print(f"   源文件: {input_path} ({src_lines} 行)")
        print(f"   输出:   {output_path} ({py_lines} 行)")

    except Exception as e:
        print(f"[FAIL] 导出失败: {e}")
        sys.exit(1)


# ============================================================
# 第五部分：交互式 REPL
# ============================================================

def repl():
    """交互式中文编程环境"""
    print("=" * 50)
    print("  AI 原生中文编程 - 交互模式")
    print("  输入 exit() 或 quit() 退出")
    print("=" * 50)

    history = []
    while True:
        try:
            line = input("\n[GREEN] 中文 > ")
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if line.lower() in ('exit()', 'quit()', 'exit', 'quit'):
            break

        if not line.strip():
            continue

        history.append(line)

        # 尝试执行
        try:
            # 将单行包装为完整代码
            code = translate(line)
            result = eval(code, {'__builtins__': __builtins__})
            if result is not None:
                print(f"  = {result}")
        except SyntaxError:
            # 可能不是表达式，尝试作为语句执行
            try:
                code = translate(line)
                exec(code, {'__builtins__': __builtins__})
            except Exception as e:
                print(f"  [FAIL] 错误: {e}")
        except Exception as e:
            print(f"  [FAIL] 错误: {e}")


# ============================================================
# 第六部分：CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="知序 (Zhixu) - 中文编程语言 编译器/解释器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  zx run demo.zx                       # 执行中文代码
  zx export demo.zx                    # 导出为Python代码
  zx repl                              # 进入交互模式

环境变量:
  ZHIXU_DEBUG=1  显示转译后的Python代码
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # run 命令
    run_parser = subparsers.add_parser("run", help="执行 .zx 文件")
    run_parser.add_argument("file", help="中文代码文件路径 (.zx)")

    # export 命令
    export_parser = subparsers.add_parser("export", help="导出为标准 Python 代码")
    export_parser.add_argument("input", help="源文件路径 (.zx)")
    export_parser.add_argument("output", nargs="?", help="输出路径 (.py，可选)")

    # repl 命令
    subparsers.add_parser("repl", help="进入交互式编程环境")

    # 如果没有参数或只有未知参数，显示帮助
    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    if args.command == "run":
        run_file(args.file)

    elif args.command == "export":
        if not args.output:
            # 自动生成输出文件名
            base = os.path.splitext(args.input)[0]
            args.output = base + ".py"
        export_file(args.input, args.output)

    elif args.command == "repl":
        repl()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
