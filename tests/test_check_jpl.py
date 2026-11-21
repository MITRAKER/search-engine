"""Tests for the NASA/JPL rule checker in scripts/check_jpl.py.

Each rule gets cases it must catch and cases it must leave alone, because a
checker that cries wolf gets switched off just like one that misses things.
"""

import sys
import textwrap
from pathlib import Path

import pytest

from scripts import check_jpl

ROOT = Path(__file__).resolve().parent.parent


def found(source: str, path: str = "module.py") -> list[str]:
    """Each violation in ``source`` as "line rule", in order."""
    violations = check_jpl.check_source(textwrap.dedent(source), path)
    return [f"{violation.line} {violation.rule}" for violation in violations]


def rules(source: str, path: str = "module.py") -> list[str]:
    violations = check_jpl.check_source(textwrap.dedent(source), path)
    return [violation.rule for violation in violations]


def messages(source: str, path: str = "module.py") -> list[str]:
    violations = check_jpl.check_source(textwrap.dedent(source), path)
    return [f"{violation.rule} {violation.message}" for violation in violations]


def test_clean_code_passes():
    assert rules("def add(a, b):\n    return a + b\n") == []


def test_a_violation_prints_as_path_line_rule_message():
    violation = check_jpl.Violation("a.py", 3, "JPL1", "f is recursive")
    assert str(violation) == "a.py:3: JPL1 f is recursive"


# ---------- JPL1: recursion ----------


def test_direct_recursion():
    assert messages("def f(n):\n    return f(n - 1)\n") == ["JPL1 f is recursive"]


def test_mutual_recursion():
    source = """
        def ping():
            return pong()

        def pong():
            return ping()
    """
    assert found(source) == ["2 JPL1", "5 JPL1"]


def test_method_recursion_through_self():
    source = """
        class A:
            def walk(self):
                return self.walk()
    """
    assert messages(source) == ["JPL1 A.walk is recursive"]


def test_recursion_through_the_class_name():
    source = """
        class Tree:
            def walk(self, n):
                return 0 if n == 0 else Tree.walk(self, n - 1)

            @staticmethod
            def depth(n):
                return 0 if n == 0 else Tree.depth(n - 1)
    """
    assert messages(source) == ["JPL1 Tree.walk is recursive", "JPL1 Tree.depth is recursive"]


@pytest.mark.parametrize(
    "construct", ["Node(depth - 1)", "type(self)(depth - 1)", "self.make(depth - 1)"]
)
def test_recursion_through_the_constructor(construct):
    source = f"""
        class Node:
            def __init__(self, depth):
                self.child = {construct} if depth else None

            @classmethod
            def make(cls, depth):
                return cls(depth)
    """
    assert "JPL1 Node.__init__ is recursive" in messages(source)


def test_recursion_of_a_nested_function():
    source = """
        def outer(depth):
            def inner(n):
                return 0 if n == 0 else inner(n - 1)

            return inner(depth)
    """
    assert messages(source) == ["JPL1 outer.<locals>.inner is recursive"]


def test_recursion_inside_a_comprehension():
    source = """
        def outer(tree):
            def walk(node):
                return [walk(child) for child in node]

            return walk(tree)
    """
    assert messages(source) == ["JPL1 outer.<locals>.walk is recursive"]


@pytest.mark.parametrize(
    "block",
    [
        "if sys.platform == 'win32':",
        "if __name__ == '__main__':",
        "for _ in range(1):",
        "with open(__file__):",
    ],
)
def test_functions_inside_blocks_are_checked(block):
    source = f"""
        import sys

        {block}
            def countdown(n):
                return countdown(n - 1)
    """
    assert found(source) == ["5 JPL1"]


def test_functions_inside_try_and_match_are_checked():
    source = """
        try:
            import tomllib
        except ImportError:
            def fallback(n):
                return fallback(n + 1)

        match tomllib:
            case None:
                def other(n):
                    return other(n + 1)
    """
    assert found(source) == ["5 JPL1", "10 JPL1"]


def test_async_recursion():
    assert rules("async def f():\n    await f()\n") == ["JPL1"]


def test_calling_a_same_named_method_elsewhere_is_not_recursion():
    source = """
        class A:
            def add(self, x):
                return self.items.add(x)
    """
    assert rules(source) == []


def test_a_factory_whose_closure_calls_it_is_not_recursion():
    # The closure runs later, after make_counter has returned.
    source = """
        def make_counter(start):
            def advance():
                return make_counter(start + 1)

            return advance
    """
    assert rules(source) == []


def test_a_local_function_shadows_a_module_level_one():
    source = """
        def build():
            def step():
                return 1

            return step()  # the local step, not the module-level one

        def step():
            return build()
    """
    assert rules(source) == []


@pytest.mark.parametrize(
    "binding",
    [
        "def g(f): return f()",  # a parameter
        "def g(): f = len; return f([])",  # a variable
        "def g():\n    import f\n    return f()",  # an import
        "def g():\n    try: pass\n    except E as f: return f()",  # a caught exception
    ],
)
def test_a_local_name_shadows_a_function(binding):
    source = f"def f():\n    return g(len)\n{binding}\n"
    assert rules(source) == []


def test_a_default_value_runs_when_the_function_is_defined():
    source = """
        def outer(n):
            def inner(x=outer(n - 1)):
                return x

            return inner
    """
    assert messages(source) == ["JPL1 outer is recursive"]


def test_a_class_body_runs_when_the_class_is_defined():
    source = """
        def build(n):
            class Local:
                value = build(n - 1)

            return Local
    """
    assert messages(source) == ["JPL1 build is recursive"]


def test_a_call_in_a_lambda_counts_for_the_function_holding_it():
    source = "def f(xs):\n    return sorted(xs, key=lambda x: f(x))\n"
    assert rules(source) == ["JPL1"]


def test_a_function_defined_twice_keeps_the_calls_of_both():
    source = """
        if FLAG:
            def f():
                return g()
        else:
            def f():
                return 1

        def g():
            return f()
    """
    assert found(source) == ["3 JPL1", "6 JPL1", "9 JPL1"]


def test_calls_through_objects_are_left_to_review():
    source = """
        def visit(node, n):
            return node.accept(n)

        class Node:
            def accept(self, n):
                return visit(self, n - 1)
    """
    assert rules(source) == []


# ---------- JPL2: unbounded loops ----------


@pytest.mark.parametrize("test", ["True", "1", "'yes'", "not False", "not 0", "not None"])
def test_while_a_true_constant(test):
    assert messages(f"def f():\n    while {test}:\n        pass\n") == [
        f"JPL2 loop has no upper bound (while {test})"
    ]


@pytest.mark.parametrize("test", ["xs", "not done", "False", "0", "not True"])
def test_while_with_a_real_condition_is_fine(test):
    assert rules(f"def f(xs, done):\n    while {test}:\n        xs.pop()\n") == []


@pytest.mark.parametrize(
    ("imports", "iterable", "name"),
    [
        ("import itertools", "itertools.count()", "itertools.count"),
        ("import itertools as it", "it.count(5)", "itertools.count"),
        ("from itertools import cycle", "cycle(xs)", "itertools.cycle"),
        ("import itertools", "itertools.repeat(0)", "itertools.repeat"),
        ("import itertools", "itertools.repeat(0, *n)", None),
        ("", "iter(int, 1)", "iter(function, sentinel)"),
    ],
)
def test_loops_over_endless_iterators(imports, iterable, name):
    source = f"{imports}\ndef f(xs, n):\n    for _ in {iterable}:\n        pass\n"
    expected = [f"JPL2 loop has no upper bound (for over {name})"] if name else []
    assert messages(source) == expected


@pytest.mark.parametrize(
    "iterable", ["itertools.repeat(0, 3)", "itertools.repeat(0, times=3)", "iter(xs)", "xs"]
)
def test_loops_over_bounded_iterators_are_fine(iterable):
    assert rules(f"import itertools\ndef f(xs):\n    for _ in {iterable}:\n        pass\n") == []


def test_comprehensions_and_async_loops_are_checked():
    source = """
        import itertools

        async def first_big(stream):
            async for _ in itertools.count():
                pass
            return next(n for n in itertools.count() if n > 5)
    """
    assert found(source) == ["5 JPL2", "7 JPL2"]


# ---------- JPL4: function length ----------


def test_long_function():
    body = "".join(f"    x{i} = {i}\n" for i in range(60))
    assert messages(f"def f():\n{body}") == ["JPL4 f is 61 lines long (max 60)"]


def test_a_function_of_exactly_60_lines_is_fine():
    body = "".join(f"    x{i} = {i}\n" for i in range(59))
    assert rules(f"def f():\n{body}") == []


def test_long_functions_inside_blocks_and_classes_are_checked():
    body = "".join(f"            x{i} = {i}\n" for i in range(60))
    source = f"try:\n    pass\nexcept ImportError:\n    class A:\n        def f(self):\n{body}"
    assert messages(source) == ["JPL4 A.f is 61 lines long (max 60)"]


# ---------- JPL6: shared state ----------


def test_global_and_nonlocal():
    source = """
        X: Final = 1

        def f():
            global X
            def g():
                nonlocal y
    """
    assert messages(source) == [
        "JPL6 global/nonlocal state is not allowed",
        "JPL6 global/nonlocal state is not allowed",
    ]


def test_globals_is_shared_state():
    assert messages("def f():\n    return globals()\n") == [
        "JPL6 module state through globals() is not allowed"
    ]


@pytest.mark.parametrize(
    "change",
    [
        "sys.modules['x'] = None",
        "sys.modules[__name__].COUNT = 2",
        "del sys.modules['x']",
        "sys.modules.pop('x')",
        "sys.modules.update({})",
        "modules['x'] = None",
        "sys.path.insert(0, 'src')",
        "sys.path.append('src')",
        "sys.path[0] = 'src'",
    ],
)
def test_changing_sys_modules_or_sys_path(change):
    source = f"import sys\nfrom sys import modules\ndef f():\n    {change}\n"
    assert any(message.startswith("JPL6 changing sys.") for message in messages(source))


@pytest.mark.parametrize(
    "read", ["'x' in sys.modules", "sys.modules.get('x')", "sys.path[0]", "list(sys.path)"]
)
def test_reading_sys_modules_or_sys_path_is_fine(read):
    assert rules(f"import sys\ndef f():\n    return {read}\n") == []


@pytest.mark.parametrize(
    ("statement", "names"),
    [
        ("counter = 0", "counter"),
        ("X: int = 1", "X"),
        ("x, (y, z) = 1, (2, 3)", "x, y, z"),
        ("X: Final = 1\nX += 1", "X"),
        ("if A:\n    LIMIT = 1", "LIMIT"),
        ("try:\n    import tomllib\nexcept ImportError:\n    tomllib = None", "tomllib"),
        ("if __name__ == '__main__':\n    args = parse()", "args"),
    ],
)
def test_module_level_names_must_be_final(statement, names):
    assert messages(statement) == [f"JPL6 module-level name {names} is not marked Final"]


@pytest.mark.parametrize(
    "statement",
    [
        "LIMIT: Final = 10",
        "LIMIT: Final[int] = 10",
        "LIMIT: typing.Final = 10",
        "Ranker: TypeAlias = Literal['bm25', 'tfidf']",
        "__all__ = ('a', 'b')",
        "__version__ = '1.0'",
        "LIMIT: Final[int]",  # a declaration, not an assignment
        "def f():\n    x = []\n    return x",  # not module level
        "obj.attribute = 1",  # sets no name
    ],
)
def test_marked_names_and_other_statements_are_fine(statement):
    assert rules(statement) == []


@pytest.mark.parametrize(
    "value",
    [
        "{}",
        "[]",
        "{1}",
        "[x for x in range(3)]",
        "{x: x for x in range(3)}",
        "{x for x in range(3)}",
        "set()",
        "dict(a=1)",
        "bytearray(3)",
        "collections.defaultdict(list)",
        "deque()",
        "(1, [2])",
        "((1, 2), {3: 4})",
        "tuple([1, [2]])",
        "tuple([x] for x in range(3))",
        "MappingProxyType({'a': [1]})",
        "MappingProxyType({k: [k] for k in 'ab'})",
        "frozenset() if A else []",
    ],
)
def test_mutable_containers_at_module_level(value):
    source = f"""
        import collections
        from collections import deque
        from types import MappingProxyType

        X: Final = {value}
    """
    assert messages(source) == [
        "JPL6 a mutable container at module level is shared state"
        " (use a tuple, frozenset or MappingProxyType)"
    ]


@pytest.mark.parametrize(
    "value",
    [
        "(1, 2)",
        "((1, 2), ('a',))",
        "frozenset({'a', 'b'})",
        "frozenset(['a', 'b'])",
        "tuple([1, 2])",
        "tuple(x for x in range(3))",
        "MappingProxyType({'a': 1})",
        "MappingProxyType({k: len(k) for k in ['a', 'b']})",
        "types.MappingProxyType({'a': (1, 2)})",
        "re.compile('a|b')",
        "'|'.join(['a', 'b'])",  # the list is not kept
        "A | B",
        "lambda: []",  # a new list on each call
    ],
)
def test_frozen_values_at_module_level_are_fine(value):
    source = f"import re, types\nfrom types import MappingProxyType\nX: Final = {value}\n"
    assert rules(source) == []


def test_mutable_containers_at_class_level():
    source = """
        class A:
            items = []
            names = ("a", "b")
            limit = 3
            if FLAG:
                table = {}

            def method(self):
                cache = {}
                return cache
    """
    assert found(source) == ["3 JPL6", "7 JPL6"]


def test_test_files_may_keep_tables_of_cases():
    source = "CASES = [1, 2]\nclass TestA:\n    seen = []\n"
    assert rules(source, "tests/test_a.py") == []
    assert rules(source, "test_a.py") == []
    assert rules(source, "src/a.py") == ["JPL6", "JPL6"]


def test_test_files_are_still_checked_for_other_shared_state():
    source = "import sys\nsys.modules['x'] = None\ndef f():\n    global X\n"
    assert rules(source, "tests/test_a.py") == ["JPL6", "JPL6"]


# ---------- JPL8: dynamic code and monkey-patching ----------


@pytest.mark.parametrize(
    "use",
    [
        "eval('1 + 1')",
        "exec('x = 1')",
        "compile('1', 'f', 'eval')",
        "__import__('os')",
        "run = eval",
        "builtins.eval('1')",
        "builtins_eval('1')",
        "__builtins__.exec('1')",
    ],
)
def test_dynamic_code(use):
    source = f"import builtins\nfrom builtins import eval as builtins_eval\ndef f():\n    {use}\n"
    assert rules(source) == ["JPL8"]


@pytest.mark.parametrize(
    ("call", "how"),
    [
        ("importlib.import_module(name)", "importlib.import_module()"),
        ("import_module(name)", "importlib.import_module()"),
        ("importlib.__import__(name)", "importlib.__import__()"),
        ("runpy.run_path(name)", "runpy.run_path()"),
        ("runpy.run_module(name)", "runpy.run_module()"),
        ("spec.loader.exec_module(module)", "exec_module()"),
        ("loader.load_module(name)", "load_module()"),
        ("getattr(builtins, 'eval', None)", "getattr(..., 'eval')"),
        ("getattr(importlib, 'import_module')", "getattr(..., 'import_module')"),
        ("getattr(__builtins__, 'exec')", "getattr(..., 'exec')"),
        ("getattr(spec.loader, 'exec_module')", "getattr(..., 'exec_module')"),
    ],
)
def test_code_loaded_at_runtime(call, how):
    source = f"""
        import builtins
        import importlib
        import runpy
        from importlib import import_module

        def f(name, spec, module, loader):
            return {call}
    """
    assert messages(source) == [f"JPL8 code loaded at runtime via {how}"]


@pytest.mark.parametrize(
    "use", ["re.compile('a')", "ast.literal_eval('1')", "getattr(obj, 'compile', None)"]
)
def test_lookalikes_are_not_dynamic_code(use):
    assert rules(f"import ast, re\ndef f(obj):\n    return {use}\n") == []


@pytest.mark.parametrize(
    ("patch", "shown"),
    [
        ("json.loads = fake", "json.loads"),
        ("json.decoder.scanstring = fake", "json.decoder.scanstring"),
        ("del json.loads", "json.loads"),
        ("setattr(json, 'loads', fake)", "setattr(json, 'loads', fake)"),
        ("delattr(json.decoder, 'x')", "delattr(json.decoder, 'x')"),
        ("Path.home = fake", "Path.home"),
        ("scoring.K1 = 2.0", "scoring.K1"),
        ("bm25.__defaults__ = (1,)", "bm25.__defaults__"),
    ],
)
def test_monkey_patching(patch, shown):
    source = f"""
        import json
        from pathlib import Path
        from . import scoring
        from .scoring import bm25

        def f(fake):
            {patch}
    """
    assert messages(source) == [f"JPL8 monkey-patching: {shown}"]


@pytest.mark.parametrize(
    "change",
    [
        "self.x = 1",
        "obj.x = 1",
        "setattr(obj, 'x', 1)",
        "monkeypatch.setattr(json, 'loads', len)",  # pytest undoes it
    ],
)
def test_setting_attributes_of_local_objects_is_fine(change):
    assert rules(f"import json\ndef f(self, obj, monkeypatch):\n    {change}\n") == []


# ---------- JPL9: attributes by computed names ----------


@pytest.mark.parametrize(
    "lookup",
    [
        "getattr(obj, name)",
        "getattr(obj, name, None)",
        "setattr(obj, name, 1)",
        "delattr(obj, name)",
        "hasattr(obj, name)",
        "getattr(obj, f'get_{name}')",
        "getattr(obj, NAME)",  # NAME is not Final
        "builtins.getattr(obj, name)",
        "getattr(*pair)",
        "getattr(obj, *names)",
        "inspect.getattr_static(obj, name)",
        "operator.attrgetter('a', name)",
        "operator.methodcaller(name)",
        "object.__getattribute__(obj, name)",
        "obj.__getattribute__(name)",
        "super().__getattr__(name)",
        "object.__setattr__(obj, name, 1)",
        "obj.__setattr__(name, 'value')",
        "obj.__delattr__(name)",
        "vars(obj)[name]",
        "obj.__dict__[name]",
        "locals()[name]",
    ],
)
def test_attributes_by_computed_name(lookup):
    source = f"""
        import builtins, inspect, operator

        NAME = "x"

        def f(obj, name, names, pair):
            return {lookup}
    """
    assert "JPL9" in rules(source)


@pytest.mark.parametrize(
    "lookup",
    [
        "getattr(obj, 'x', None)",
        "getattr(obj, NAME, None)",  # NAME is a Final string
        "setattr(obj, NAME, 1)",
        "operator.attrgetter('a', 'b.c')",
        "operator.methodcaller('split', sep)",  # sep is an argument, not a name
        "object.__setattr__(obj, 'x', value)",
        "obj.__getattribute__('x')",
        "vars(obj)['x']",
        "obj.__dict__['x']",
        "table[name]",
        "obj.items[name]",
    ],
)
def test_attributes_by_fixed_name_are_fine(lookup):
    source = f"""
        import operator

        NAME: Final = "x"

        def f(obj, name, table, sep, value):
            return {lookup}
    """
    assert rules(source) == []


def test_globals_lookup_is_shared_state_and_a_computed_name():
    assert rules("def f(name):\n    return globals()[name]()\n") == ["JPL6", "JPL9"]


# ---------- reading files ----------


def test_a_syntax_error_is_reported_not_raised():
    assert messages("def f(:\n")[0].startswith("JPL0 cannot be parsed")
    assert found("x = 1\ny = (\n") == ["2 JPL0"]


def test_a_null_byte_is_reported_not_raised():
    assert rules("x = 1\0\n") == ["JPL0"]


def test_files_are_decoded_like_python_does(tmp_path):
    with_bom = tmp_path / "bom.py"
    with_bom.write_bytes(b"\xef\xbb\xbfdef f() -> int:\n    return 1\n")
    latin1 = tmp_path / "latin1.py"
    latin1.write_bytes(b"# -*- coding: latin-1 -*-\nNAME: Final = 'caf\xe9'\n")
    assert check_jpl.main([str(tmp_path)]) == 0


def test_main_reports_and_fails(tmp_path, capsys):
    (tmp_path / "bad.py").write_text("def f():\n    return f()\n")
    (tmp_path / "good.py").write_text("def g():\n    return 1\n")
    assert check_jpl.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert f"{tmp_path / 'bad.py'}:1: JPL1 f is recursive" in out
    assert "JPL check: 1 violation(s)" in out


def test_main_reports_a_file_it_cannot_read(tmp_path, capsys):
    assert check_jpl.main([str(tmp_path / "missing.py")]) == 1
    assert "JPL0 cannot be read" in capsys.readouterr().out


def test_main_reads_its_targets_from_the_command_line(monkeypatch, tmp_path, capsys):
    (tmp_path / "good.py").write_text("def g():\n    return 1\n")
    monkeypatch.setattr(sys, "argv", ["check_jpl.py", str(tmp_path)])
    assert check_jpl.main() == 0
    assert "JPL check: 0 violation(s)" in capsys.readouterr().out


def test_main_checks_the_package_by_default(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert check_jpl.main([]) == 0


def test_project_code_follows_the_rules():
    targets = [str(ROOT / "searchengine"), str(ROOT / "scripts"), str(ROOT / "tests")]
    assert check_jpl.main(targets) == 0
