@echo off
REM 知序 (Zhixu) - 中文编程语言启动脚本
REM 用法: zx run demo.zx
REM       zx export demo.zx demo.py
REM       zx repl

python "%~dp0zhixu.py" %*
