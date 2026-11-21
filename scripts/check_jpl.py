"""Check Python code against our adaptation of NASA/JPL's "Power of Ten" rules.

Gerard Holzmann, "The Power of Ten: Rules for Developing Safety-Critical
Code", IEEE Computer, 2006. Most rules are enforced by ruff and mypy (see
docs/CODING_STANDARD.md). This script covers the ones they don't, one file
at a time:

    JPL1  no recursion: no function or method that can reach itself through
          calls made in the same file. Calls are followed by name, the way
          Python looks names up (so nested functions count), through self,
          cls and type(self), through ClassName.method, and from
          ClassName(...) or cls(...) to __init__. A call inside a lambda
          counts as a call made by the function that holds the lambda.
    JPL2  no loop that is unbounded by construction: no ``while True`` (or
          any other true constant, or ``not`` a false one), and no loop or
          comprehension over itertools.count, itertools.cycle,
          itertools.repeat without a count, or iter(function, sentinel).
    JPL4  no function longer than 60 lines.
    JPL6  no shared state that can change: no ``global`` or ``nonlocal``, no
          globals(), no changes to sys.modules or sys.path. Every name set
          at module level is marked Final (or TypeAlias; dunder names such
          as __all__ need no mark), and no list, dict, set or other mutable
          container is stored at module or class level unless it is frozen
          on the spot by tuple, frozenset or MappingProxyType. Test files
          are exempt from this module and class level part, so that they
          can keep their tables of cases.
    JPL8  no code made or loaded at runtime: no eval, exec, compile or
          __import__ (also under another name or through builtins), no
          importlib.import_module, runpy or exec_module; and no
          monkey-patching, that is, setting or deleting an attribute of
          something imported.
    JPL9  no attribute found by a name computed at runtime: getattr,
          setattr, delattr, hasattr, inspect.getattr_static,
          operator.attrgetter and methodcaller, the __getattribute__ family,
          and lookups in vars(), globals(), locals() or __dict__. A string
          literal, or a module-level Final string, is a fixed name.

What one file cannot show is left to review: calls through objects passed in
as arguments, through properties, dunder methods and inheritance, and calls
between files; whether a value built by a function call can change; and
code written on purpose to slip past these checks.

Usage:  python scripts/check_jpl.py searchengine scripts tests
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, TypeAlias

MAX_FUNCTION_LINES: Final = 60
BINDING_NODES: Final = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.ExceptHandler,
    ast.MatchAs,
    ast.MatchStar,
)
BLOCK_NODES: Final = (ast.stmt, ast.excepthandler, ast.match_case)

# JPL2: iterators that never stop.
ENDLESS_ITERATORS: Final = frozenset({"itertools.count", "itertools.cycle"})

# JPL6: what makes state shared and changeable.
MUTABLE_DISPLAYS: Final = (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)
MUTABLE_FACTORIES: Final = frozenset(
    {
        "list",
        "dict",
        "set",
        "bytearray",
        "collections.defaultdict",
        "collections.OrderedDict",
        "collections.Counter",
        "collections.deque",
    }
)
FREEZERS: Final = frozenset({"tuple", "frozenset", "types.MappingProxyType"})
CONSTANT_MARKS: Final = frozenset({"Final", "TypeAlias"})
SHARED_MODULE_STATE: Final = frozenset({"sys.modules", "sys.path"})
MUTATING_METHODS: Final = frozenset(
    {
        "append",
        "extend",
        "insert",
        "remove",
        "pop",
        "popitem",
        "clear",
        "update",
        "setdefault",
        "sort",
        "reverse",
        "__setitem__",
        "__delitem__",
    }
)
MUTABLE_MESSAGE: Final = (
    "a mutable container at {level} level is shared state"
    " (use a tuple, frozenset or MappingProxyType)"
)

# JPL8: ways to run code that is not written in the file.
DYNAMIC_CODE: Final = frozenset({"eval", "exec", "compile", "__import__"})
CODE_LOADERS: Final = frozenset(
    {"importlib.import_module", "importlib.__import__", "runpy.run_path", "runpy.run_module"}
)
LOADER_METHODS: Final = frozenset({"exec_module", "load_module"})

# JPL9: functions that look an attribute up by name, and which argument
# holds the name.
NAME_ARGUMENT: Final = MappingProxyType(
    {
        "getattr": 1,
        "setattr": 1,
        "delattr": 1,
        "hasattr": 1,
        "inspect.getattr_static": 1,
        "operator.methodcaller": 0,
    }
)
# The same for the __getattribute__ family, counted from the end, because a
# call can be bound, obj.__getattribute__(name), or not,
# object.__getattribute__(obj, name).
NAME_ARGUMENT_FROM_END: Final = MappingProxyType(
    {"__getattribute__": 1, "__getattr__": 1, "__delattr__": 1, "__setattr__": 2}
)
NAMESPACES: Final = frozenset({"vars", "globals", "locals"})

FunctionNode: TypeAlias = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True, slots=True)
class Violation:
    path: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule} {self.message}"


@dataclass(frozen=True, slots=True)
class Function:
    """A function or method, and where the names it uses are looked up."""

    name: str  # qualified, like "Tree.walk" or "outer.<locals>.inner"
    node: FunctionNode
    scopes: tuple[str, ...]  # enclosing function scopes, innermost first; "" is the module
    owner: str  # "Tree." when self and cls mean an instance of Tree, else ""


@dataclass(frozen=True, slots=True)
class Module:
    """What the checks need to know about a whole file."""

    path: str
    functions: tuple[Function, ...]
    classes: frozenset[str]  # qualified class names
    names: frozenset[str]  # qualified names of every function and class
    imports: Mapping[str, str]  # local name -> what it was imported as
    constants: frozenset[str]  # module-level Final strings
    is_test: bool


# ---------- reading a file ----------


def _definitions(tree: ast.AST) -> tuple[list[Function], set[str]]:
    """Every function and class, wherever it is defined (also inside if,
    try or with blocks), with qualified names like Python's __qualname__."""
    functions: list[Function] = []
    classes: set[str] = set()
    stack: list[tuple[ast.AST, str, tuple[str, ...], str]] = [(tree, "", ("",), "")]
    while stack:  # bounded: each node is pushed exactly once
        node, prefix, scopes, owner = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                name = f"{prefix}{child.name}"
                classes.add(name)
                stack.append((child, f"{name}.", scopes, f"{name}."))
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}"
                functions.append(Function(name, child, scopes, owner))
                inner = f"{name}.<locals>."
                stack.append((child, inner, (inner, *scopes), owner))
            else:
                stack.append((child, prefix, scopes, owner))
    return functions, classes


def _imports(tree: ast.AST) -> dict[str, str]:
    """Each imported name, and the module or module attribute it stands for.
    Relative imports keep their dots, like ".scoring.bm25"."""
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                names[alias.asname or top] = alias.name if alias.asname else top
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            separator = "." if node.module else ""
            for alias in node.names:
                names[alias.asname or alias.name] = f"{base}{separator}{alias.name}"
    return names


def _annotation_name(annotation: ast.expr) -> str:
    """The name of a type annotation, without its arguments or module:
    "Final" for ``Final``, ``Final[int]`` and ``typing.Final``."""
    if isinstance(annotation, ast.Subscript):
        annotation = annotation.value
    if isinstance(annotation, ast.Attribute):
        return annotation.attr
    return annotation.id if isinstance(annotation, ast.Name) else ""


def _final_strings(tree: ast.Module) -> frozenset[str]:
    """Module-level names declared ``NAME: Final = "text"``."""
    return frozenset(
        node.target.id
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and _annotation_name(node.annotation) == "Final"
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def read_module(tree: ast.Module, path: str) -> Module:
    functions, classes = _definitions(tree)
    names = frozenset(f.name for f in functions) | classes
    is_test = "tests" in Path(path).parts or Path(path).name.startswith("test_")
    return Module(
        path,
        tuple(functions),
        frozenset(classes),
        names,
        MappingProxyType(_imports(tree)),
        _final_strings(tree),
        is_test,
    )


def dotted_name(node: ast.AST, imports: Mapping[str, str]) -> str:
    """The full name an expression refers to, like "itertools.count" for
    ``it.count`` after ``import itertools as it``, or "" if it is not a
    name. Builtins lose their prefix, so ``builtins.eval`` is just "eval"."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):  # bounded by the depth of the expression
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    return _without_builtins(".".join([imports.get(node.id, node.id), *reversed(parts)]))


def _without_builtins(name: str) -> str:
    return name.removeprefix("builtins.").removeprefix("__builtins__.")


def _runs_here(node: ast.AST) -> list[ast.AST]:
    """The parts of ``node`` that run when the code around it runs. A nested
    function's body runs later, when it is called, but its decorators and
    default values run where it is defined. A class body runs at once."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        defaults = [*node.args.defaults, *node.args.kw_defaults]
        return [*node.decorator_list, *(value for value in defaults if value is not None)]
    return list(ast.iter_child_nodes(node))


def own_nodes(function: FunctionNode) -> Iterator[ast.AST]:
    """The nodes that run when ``function`` runs: its body, without the
    bodies of the functions defined inside it."""
    stack: list[ast.AST] = list(function.body)
    while stack:  # bounded: each node is visited once
        node = stack.pop()
        yield node
        stack.extend(_runs_here(node))


def _line(node: ast.AST) -> int:
    """Where ``node`` starts. A comprehension's ``for`` clause has no
    position of its own, so it takes the position of what it loops over."""
    if isinstance(node, ast.comprehension):
        node = node.iter
    return getattr(node, "lineno", 1)


# ---------- JPL1: recursion ----------


def _local_names(node: FunctionNode) -> set[str]:
    """The names a function binds itself: its parameters, and the names it
    assigns, imports, catches or defines."""
    args = node.args
    every = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
    names = {arg.arg for arg in every if arg is not None}
    for child in own_nodes(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in child.names)
        elif isinstance(child, BINDING_NODES) and child.name:
            names.add(child.name)
    return names


def _resolve(name: str, function: Function, module: Module, local: Mapping[str, set[str]]) -> str:
    """The function or class a bare name means inside ``function``, looked
    up the way Python does: in the function itself, then in the functions
    around it, then in the module. "" if a variable of that name is found
    first, or nothing is."""
    for scope in (f"{function.name}.<locals>.", *function.scopes):
        if f"{scope}{name}" in module.names:
            return f"{scope}{name}"
        if name in local.get(scope, ()):
            return ""
    return ""


def _callee(
    func: ast.expr, function: Function, module: Module, local: Mapping[str, set[str]]
) -> str:
    """The qualified name of the function or method a call runs, or "" if
    this file cannot tell."""
    owner = function.owner
    if owner and ast.unparse(func) in ("cls", "type(self)"):
        return f"{owner}__init__"
    if isinstance(func, ast.Name):
        target = _resolve(func.id, function, module, local)
        return f"{target}.__init__" if target in module.classes else target
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        base = func.value.id
        if owner and base in ("self", "cls"):
            return f"{owner}{func.attr}"
        target = _resolve(base, function, module, local)
        return f"{target}.{func.attr}" if target in module.classes else ""
    return ""


def call_graph(module: Module) -> dict[str, set[str]]:
    """The functions of the file that each function calls."""
    local: dict[str, set[str]] = {}
    for f in module.functions:
        local.setdefault(f"{f.name}.<locals>.", set()).update(_local_names(f.node))
    graph: dict[str, set[str]] = {}
    for f in module.functions:
        calls = (node for node in own_nodes(f.node) if isinstance(node, ast.Call))
        callees = {_callee(call.func, f, module, local) for call in calls}
        graph.setdefault(f.name, set()).update(callees)
    return {name: callees & graph.keys() for name, callees in graph.items()}


def _on_cycle(graph: Mapping[str, set[str]], start: str) -> bool:
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


def check_recursion(module: Module) -> list[Violation]:
    graph = call_graph(module)
    return [
        Violation(module.path, f.node.lineno, "JPL1", f"{f.name} is recursive")
        for f in module.functions
        if _on_cycle(graph, f.name)
    ]


# ---------- JPL4: function length ----------


def check_function_length(module: Module) -> list[Violation]:
    found = []
    for f in module.functions:
        length = (f.node.end_lineno or f.node.lineno) - f.node.lineno + 1
        if length > MAX_FUNCTION_LINES:
            message = f"{f.name} is {length} lines long (max {MAX_FUNCTION_LINES})"
            found.append(Violation(module.path, f.node.lineno, "JPL4", message))
    return found


# ---------- JPL2: loops ----------


def _always_true(test: ast.expr) -> bool:
    """True for a constant that is true, or ``not`` a constant that is false."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return isinstance(test.operand, ast.Constant) and not test.operand.value
    return isinstance(test, ast.Constant) and bool(test.value)


def _endless(iterable: ast.expr, module: Module) -> str:
    """The name of an iterator that never stops, if ``iterable`` is one."""
    if not isinstance(iterable, ast.Call):
        return ""
    name = dotted_name(iterable.func, module.imports)
    counted = len(iterable.args) > 1 or any(kw.arg == "times" for kw in iterable.keywords)
    if name in ENDLESS_ITERATORS or (name == "itertools.repeat" and not counted):
        return name
    return "iter(function, sentinel)" if name == "iter" and len(iterable.args) == 2 else ""


def check_loop(node: ast.AST, module: Module) -> str | None:
    """JPL2: loops that never end by construction."""
    if isinstance(node, ast.While) and _always_true(node.test):
        return f"loop has no upper bound (while {ast.unparse(node.test)})"
    if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
        endless = _endless(node.iter, module)
        if endless:
            return f"loop has no upper bound (for over {endless})"
    return None


# ---------- JPL6: shared state ----------


def _changed_module_state(node: ast.AST, module: Module) -> str:
    """ "sys.modules" or "sys.path" if ``node`` changes one of them."""
    if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(
        node.ctx, (ast.Store, ast.Del)
    ):
        target = node.value
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr not in MUTATING_METHODS:
            return ""
        target = node.func.value
    else:
        return ""
    name = dotted_name(target, module.imports)
    while name not in SHARED_MODULE_STATE and isinstance(target, (ast.Attribute, ast.Subscript)):
        target = target.value  # bounded by the depth of the expression
        name = dotted_name(target, module.imports)
    return name if name in SHARED_MODULE_STATE else ""


def _unmarked_names(node: ast.Assign | ast.AnnAssign | ast.AugAssign) -> str:
    """The names a module-level assignment sets without marking them Final
    (or TypeAlias). Dunder names such as __all__ need no mark."""
    if isinstance(node, ast.AnnAssign) and _annotation_name(node.annotation) in CONSTANT_MARKS:
        return ""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names = [
        child.id
        for target in targets
        for child in ast.walk(target)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
    ]
    return ", ".join(name for name in names if not name.startswith("__"))


def _elements(container: ast.expr) -> list[ast.expr]:
    """What a container display or comprehension stores."""
    if isinstance(container, (ast.List, ast.Set, ast.Tuple)):
        return container.elts
    if isinstance(container, ast.Dict):
        return container.values
    if isinstance(container, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        return [container.elt]
    return [container.value] if isinstance(container, ast.DictComp) else []


def _mutable_value(value: ast.expr, module: Module) -> bool:
    """True if ``value`` stores a list, dict, set or other mutable container
    that is not frozen on the spot by tuple, frozenset or MappingProxyType."""
    stack = [value]
    while stack:  # bounded: each part of the expression is looked at once
        part = stack.pop()
        name = dotted_name(part.func, module.imports) if isinstance(part, ast.Call) else ""
        if isinstance(part, MUTABLE_DISPLAYS) or name in MUTABLE_FACTORIES:
            return True
        if isinstance(part, ast.Tuple):
            stack.extend(part.elts)
        elif isinstance(part, ast.IfExp):
            stack.extend((part.body, part.orelse))
        elif isinstance(part, ast.Call) and name in FREEZERS:
            stack.extend(element for arg in part.args for element in _elements(arg))
    return False


def _stored_state(node: ast.AST, module: Module, level: str) -> str | None:
    """JPL6 for a statement that runs at module level or in a class body."""
    if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) or node.value is None:
        return None
    unmarked = _unmarked_names(node) if level == "module" else ""
    if unmarked:
        return f"module-level name {unmarked} is not marked Final"
    if _mutable_value(node.value, module):
        return MUTABLE_MESSAGE.format(level=level)
    return None


def check_state(node: ast.AST, module: Module, level: str | None) -> str | None:
    """JPL6: state that is shared and can change. ``level`` is "module" or
    "class" for a statement that runs at module level or in a class body."""
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return "global/nonlocal state is not allowed"
    if isinstance(node, ast.Call) and dotted_name(node.func, module.imports) == "globals":
        return "module state through globals() is not allowed"
    changed = _changed_module_state(node, module)
    if changed:
        return f"changing {changed} is not allowed"
    if level is None or module.is_test:
        return None
    return _stored_state(node, module, level)


# ---------- JPL8: dynamic code and monkey-patching ----------


def _code_loader(call: ast.Call, module: Module) -> str:
    """How ``call`` loads code at runtime, if it does."""
    name = dotted_name(call.func, module.imports)
    if name in CODE_LOADERS:
        return f"{name}()"
    if isinstance(call.func, ast.Attribute) and call.func.attr in LOADER_METHODS:
        return f"{call.func.attr}()"
    looked_up = call.args[1] if name == "getattr" and len(call.args) > 1 else None
    if not isinstance(looked_up, ast.Constant) or not isinstance(looked_up.value, str):
        return ""
    attribute = looked_up.value
    full = _without_builtins(f"{dotted_name(call.args[0], module.imports)}.{attribute}")
    if full in DYNAMIC_CODE or full in CODE_LOADERS or attribute in LOADER_METHODS:
        return f"getattr(..., {attribute!r})"
    return ""


def _patched(node: ast.AST, module: Module) -> str:
    """What ``node`` sets or deletes on something imported, if anything."""
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
        target: ast.expr = node
    elif (
        isinstance(node, ast.Call)
        and node.args
        and dotted_name(node.func, module.imports) in ("setattr", "delattr")
    ):
        target = node.args[0]
    else:
        return ""
    root = target
    while isinstance(root, (ast.Attribute, ast.Subscript)):  # bounded by expression depth
        root = root.value
    imported = isinstance(root, ast.Name) and root.id in module.imports
    return ast.unparse(node) if imported else ""


def check_dynamic_code(node: ast.AST, module: Module) -> str | None:
    """JPL8: code made or loaded at runtime, and monkey-patching."""
    if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
        name = dotted_name(node, module.imports)
        if name in DYNAMIC_CODE:
            return f"dynamic code via {name}"
    loader = _code_loader(node, module) if isinstance(node, ast.Call) else ""
    if loader:
        return f"code loaded at runtime via {loader}"
    patched = _patched(node, module)
    return f"monkey-patching: {patched}" if patched else None


# ---------- JPL9: attributes by computed names ----------


def _fixed_name(node: ast.expr, module: Module) -> bool:
    """True for a literal, or a module-level Final string."""
    if isinstance(node, ast.Constant):
        return True
    return isinstance(node, ast.Name) and node.id in module.constants


def _attribute_names(call: ast.Call, module: Module) -> list[ast.expr]:
    """The arguments of ``call`` that name an attribute to look up, [] if it
    looks nothing up. An argument unpacked with * could hold the name."""
    name = dotted_name(call.func, module.imports)
    if name == "operator.attrgetter":
        return call.args
    if name in NAME_ARGUMENT:
        position = NAME_ARGUMENT[name]
    elif isinstance(call.func, ast.Attribute) and call.func.attr in NAME_ARGUMENT_FROM_END:
        position = len(call.args) - NAME_ARGUMENT_FROM_END[call.func.attr]
    else:
        return []
    starred: list[ast.expr] = [arg for arg in call.args if isinstance(arg, ast.Starred)]
    return starred or call.args[max(position, 0) : position + 1]


def check_attribute_access(node: ast.AST, module: Module) -> str | None:
    """JPL9: attributes found by a name computed at runtime."""
    if isinstance(node, ast.Call):
        names = _attribute_names(node, module)
        if not all(_fixed_name(name, module) for name in names):
            return f"{ast.unparse(node.func)}() with a computed name"
    if isinstance(node, ast.Subscript) and not _fixed_name(node.slice, module):
        value = node.value
        called = dotted_name(value.func, module.imports) if isinstance(value, ast.Call) else ""
        if called in NAMESPACES or (isinstance(value, ast.Attribute) and value.attr == "__dict__"):
            return f"{ast.unparse(value)}[...] with a computed name"
    return None


# ---------- running the checks ----------


def _statement_levels(tree: ast.Module) -> dict[int, str]:
    """The statements that run at module level or in a class body (also
    inside if, try, with or for blocks), by id: "module" or "class"."""
    levels: dict[int, str] = {}
    stack: list[tuple[ast.AST, str]] = [(statement, "module") for statement in tree.body]
    while stack:  # bounded: each statement is visited once
        node, level = stack.pop()
        levels[id(node)] = level
        if isinstance(node, ast.ClassDef):
            stack.extend((child, "class") for child in node.body)
        elif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            blocks = [
                child for child in ast.iter_child_nodes(node) if isinstance(child, BLOCK_NODES)
            ]
            stack.extend((child, level) for child in blocks)
    return levels


def check_statements(tree: ast.Module, module: Module) -> list[Violation]:
    levels = _statement_levels(tree)
    found = []
    for node in ast.walk(tree):
        checks = (
            ("JPL2", check_loop(node, module)),
            ("JPL6", check_state(node, module, levels.get(id(node)))),
            ("JPL8", check_dynamic_code(node, module)),
            ("JPL9", check_attribute_access(node, module)),
        )
        line = _line(node)
        found += [
            Violation(module.path, line, rule, message) for rule, message in checks if message
        ]
    return found


def check_source(source: str | bytes, path: str = "<string>") -> list[Violation]:
    """Run every check on one file's source code. Bytes are decoded the way
    Python decodes a file, so a BOM or an encoding comment works."""
    try:
        tree = ast.parse(source, filename=path)
    except (SyntaxError, ValueError) as error:
        line = (error.lineno if isinstance(error, SyntaxError) else None) or 1
        return [Violation(path, line, "JPL0", f"cannot be parsed: {error}")]
    module = read_module(tree, path)
    violations = check_recursion(module)
    violations += check_function_length(module)
    violations += check_statements(tree, module)
    return sorted(violations, key=lambda v: (v.path, v.line, v.rule, v.message))


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
        try:
            source = path.read_bytes()
        except OSError as error:
            violations.append(Violation(str(path), 1, "JPL0", f"cannot be read: {error}"))
            continue
        violations += check_source(source, str(path))
    for violation in violations:
        print(violation)
    print(f"JPL check: {len(violations)} violation(s)")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
