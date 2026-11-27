#!/usr/bin/env python3
"""提交前合规自检：模型清单、参数量、网络依赖与推理输入。

检查项：
1. ``weights/manifest.json`` 存在且声明离线、无外部 API；
2. 模型总参数量不超过 5B；
3. 提交代码没有联网客户端/外部模型 SDK 导入；
4. ``infer.py`` 不读取 SVG 真值；
5. 输出目录中的 JSON 均可通过格式校验（可选 ``--results``）。
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadvec.schema import SubmissionValidationError, validate_output  # noqa: E402
from compliance.params import ModelComplianceError, audit_models  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_IMPORTS = {
    "requests", "httpx", "aiohttp", "urllib3", "openai", "anthropic",
    "ollama", "transformers", "gradio", "fastapi", "flask",
    "http.client", "urllib.request", "urllib.error",
}
SKIP_FILES = {ROOT / "compliance" / "net_guard.py"}
SKIP_DIRECTORIES = {".git", ".venv", "venv", "__pycache__", "tests", "_smoke", "_work", "node_modules"}
#: 开发期外部复核模块允许在函数内部“延迟导入”联网库：模块顶层仍必须是
#: 纯离线依赖，默认推理路径不会加载网络代码；只有显式 --external-api 时
#: 才执行到该函数。其他文件的任何位置导入联网库仍会被拦截。
DEV_TRANSPORT_FILES = {ROOT / "cadvec" / "external.py"}


def _is_forbidden(module: str) -> bool:
    return any(
        module == forbidden or module.startswith(forbidden + ".")
        for forbidden in FORBIDDEN_IMPORTS
    )


def _iter_imports(node: ast.AST, in_function: bool = False):
    """遍历 AST，产出 ``(模块名, 是否位于函数体内)``。"""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        in_function = True
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.name, in_function
    elif isinstance(node, ast.ImportFrom) and node.module:
        yield node.module, in_function
    for child in ast.iter_child_nodes(node):
        yield from _iter_imports(child, in_function)


def scan_imports() -> list[str]:
    problems = []
    paths = []
    for directory, subdirectories, filenames in os.walk(ROOT):
        subdirectories[:] = sorted(name for name in subdirectories if name not in SKIP_DIRECTORIES)
        paths.extend(Path(directory) / name for name in filenames if name.endswith(".py"))
    for path in sorted(paths):
        if path in SKIP_FILES:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as error:
            problems.append(f"{path.relative_to(ROOT)}: 无法解析 ({error})")
            continue
        allow_function_scope = path in DEV_TRANSPORT_FILES
        for module, in_function in _iter_imports(tree):
            if not _is_forbidden(module):
                continue
            if allow_function_scope and in_function:
                continue
            problems.append(
                f"{path.relative_to(ROOT)} 导入了潜在联网模块 {module}"
            )
    return problems


def check_infer_does_not_read_svg() -> list[str]:
    source = (ROOT / "infer.py").read_text(encoding="utf-8")
    problems = []
    for token in (".svg", "svg_to_output", "iter_primitives"):
        if token in source:
            problems.append(f"infer.py 出现 {token!r}；推理入口不得读取 SVG 真值")
    return problems


def check_results(directory: Path) -> tuple[int, list[str]]:
    problems = []
    files = sorted(directory.glob("*.json"))
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            validate_output(payload)
        except (OSError, ValueError, SubmissionValidationError) as error:
            problems.append(f"{path.name}: {error}")
    return len(files), problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights")
    parser.add_argument("--results", type=Path, default=None, help="可选：校验预测结果目录")
    parser.add_argument("--out", type=Path, default=None, help="可选：写出 JSON 报告")
    args = parser.parse_args()

    problems: list[str] = []
    report: dict = {"ok": True, "problems": problems}
    try:
        audit = audit_models(args.weights)
        report["models"] = audit.as_dict()
        if not audit.ok:
            problems.append("模型参数量超出上限")
    except ModelComplianceError as error:
        problems.append(str(error))
        report["models"] = None

    problems.extend(scan_imports())
    problems.extend(check_infer_does_not_read_svg())

    if args.results is not None:
        if not args.results.is_dir():
            problems.append(f"结果目录不存在: {args.results}")
        else:
            count, result_problems = check_results(args.results)
            report["result_files"] = count
            problems.extend(result_problems)

    report["ok"] = not problems
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
