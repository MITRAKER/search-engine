"""Check Python code against our adaptation of NASA/JPL's "Power of Ten" rules.

Gerard Holzmann, "The Power of Ten: Rules for Developing Safety-Critical
Code", IEEE Computer, 2006. Most rules are enforced by ruff and mypy (see
docs/CODING_STANDARD.md). This script covers the ones they don't:

    JPL1  no recursion, direct or through other functions in the same file
    JPL2  no loops that are unbounded by construction (``while True``)
    JPL4  no function longer than 60 lines
    JPL6  no ``global`` or ``nonlocal`` state
    JPL8  no dynamic code: eval, exec, compile, __import__
    JPL9  no dynamic attribute access by computed name (getattr/setattr/delattr)

Usage:  python scripts/check_jpl.py searchengine scripts
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

MAX_FUNCTION_LINES: Final = 60
DYNAMIC_CODE: Final = frozenset({"eval", "exec", "compile", "__import__"})
DYNAMIC_ATTRIBUTES: Final = frozenset({"getattr", "setattr", "delattr"})

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True, slots=True)
class Violation:
    path: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule} {self.message}"


def _functions(tree: ast.AST) -> Iterator[tuple[str, FunctionNode]]:
    """Yield (qualified name, node) for every function, including methods."""
    stack: list[tuple[str, ast.AST]] = [("", tree)]
    while stack:  # bounded: each node is pushed exactly once
        prefix, node = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                stack.append((f"{prefix}{child.name}.", child))
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield f"{prefix}{child.name}", child
                stack.append((f"{prefix}{child.name}.<locals>.", child))


def _callee(call: ast.Call, owner: str) -> str | None:
    """The qualified name a call refers to, when it can be known statically."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if (
        owner
        and isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id in ("self", "cls")
    ):
        return f"{owner}{func.attr}"
    return None


def _call_graph(tree: ast.AST) -> tuple[dict[str, set[str]], dict[str, int]]:
    graph: dict[str, set[str]] = {}
    lines: dict[str, int] = {}
    for name, node in _functions(tree):
        owner = name.rsplit(".", 1)[0] + "." if "." in name else ""
        calls = {_callee(n, owner) for n in ast.walk(node) if isinstance(n, ast.Call)}
        graph[name] = {callee for callee in calls if callee}
        lines[name] = node.lineno
    for name, callees in graph.items():
        graph[name] = {callee for callee in callees if callee in graph}
    return graph, lines


def _on_cycle(graph: dict[str, set[str]], start: str) -> bool:
    """True if ``start`` can reach itself through the call graph."""
    seen: set[str] = set()
    frontier = list(graph[start])
    while frontier:  # bounded: each function is visited at most once
        name = frontier.pop()
        if name == start:
            return True
        if name not in seen:
            seen.add(name)
            frontier.extend(graph[name])
    return False


def check_recursion(tree: ast.AST, path: str) -> list[Violation]:
    graph, lines = _call_graph(tree)
    return [
        Violation(path, lines[name], "JPL1", f"{name} is recursive")
        for name in sorted(graph)
        if _on_cycle(graph, name)
    ]


def check_function_length(tree: ast.AST, path: str) -> list[Violation]:
    found = []
    for name, node in _functions(tree):
        length = (node.end_lineno or node.lineno) - node.lineno + 1
        if length > MAX_FUNCTION_LINES:
            message = f"{name} is {length} lines long (max {MAX_FUNCTION_LINES})"
            found.append(Violation(path, node.lineno, "JPL4", message))
    return found


def _check_node(node: ast.AST, path: str) -> Violation | None:
    if isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value:
        return Violation(path, node.lineno, "JPL2", "loop has no upper bound (while True)")
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return Violation(path, node.lineno, "JPL6", "global/nonlocal state is not allowed")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name in DYNAMIC_CODE:
            return Violation(path, node.lineno, "JPL8", f"dynamic code via {name}()")
        dynamic_name = len(node.args) > 1 and not isinstance(node.args[1], ast.Constant)
        if name in DYNAMIC_ATTRIBUTES and dynamic_name:
            return Violation(path, node.lineno, "JPL9", f"{name}() with a computed name")
    return None


def check_statements(tree: ast.AST, path: str) -> list[Violation]:
    found = (_check_node(node, path) for node in ast.walk(tree))
    return [violation for violation in found if violation is not None]


def check_source(source: str, path: str = "<string>") -> list[Violation]:
    """Run every check on one file's source code."""
    tree = ast.parse(source, filename=path)
    violations = check_recursion(tree, path)
    violations += check_function_length(tree, path)
    violations += check_statements(tree, path)
    return sorted(violations, key=lambda v: (v.path, v.line, v.rule))


def python_files(targets: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        path = Path(target)
        files.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    return files


def main(argv: Sequence[str] | None = None) -> int:
    targets = list(sys.argv[1:] if argv is None else argv) or ["searchengine"]
    violations: list[Violation] = []
    for path in python_files(targets):
        violations += check_source(path.read_text(encoding="utf-8"), str(path))
    for violation in violations:
        print(violation)
    print(f"JPL check: {len(violations)} violation(s)")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
