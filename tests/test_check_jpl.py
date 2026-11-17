"""Tests for the NASA/JPL rule checker in scripts/check_jpl.py."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_jpl.py"
spec = importlib.util.spec_from_file_location("check_jpl", SCRIPT)
assert spec is not None
assert spec.loader is not None
check_jpl = importlib.util.module_from_spec(spec)
sys.modules["check_jpl"] = check_jpl
spec.loader.exec_module(check_jpl)


def rules(source: str) -> list[str]:
    return [violation.rule for violation in check_jpl.check_source(source)]


def test_clean_code_passes():
    assert rules("def add(a, b):\n    return a + b\n") == []


def test_direct_recursion():
    assert rules("def f(n):\n    return f(n - 1)\n") == ["JPL1"]


def test_mutual_recursion():
    source = "def ping():\n    return pong()\n\ndef pong():\n    return ping()\n"
    assert rules(source) == ["JPL1", "JPL1"]


def test_method_recursion():
    source = "class A:\n    def walk(self):\n        return self.walk()\n"
    assert rules(source) == ["JPL1"]


def test_calling_a_same_named_method_elsewhere_is_not_recursion():
    source = "class A:\n    def add(self, x):\n        return self.items.add(x)\n"
    assert rules(source) == []


def test_unbounded_loop():
    assert rules("def f():\n    while True:\n        pass\n") == ["JPL2"]


def test_bounded_while_is_fine():
    assert rules("def f(xs):\n    while xs:\n        xs.pop()\n") == []


def test_long_function():
    body = "".join(f"    x{i} = {i}\n" for i in range(60))
    assert rules(f"def f():\n{body}") == ["JPL4"]


def test_global_state():
    assert rules("X = 1\ndef f():\n    global X\n    X = 2\n") == ["JPL6"]


def test_dynamic_code():
    assert rules("eval('1 + 1')\n") == ["JPL8"]


@pytest.mark.parametrize(
    ("source", "expected"), [("getattr(o, name)\n", ["JPL9"]), ("getattr(o, 'x')\n", [])]
)
def test_dynamic_attributes(source, expected):
    assert rules(source) == expected


def test_main_reports_and_fails(tmp_path, capsys):
    bad = tmp_path / "bad.py"
    bad.write_text("def f():\n    return f()\n")
    assert check_jpl.main([str(tmp_path)]) == 1
    assert "JPL1 f is recursive" in capsys.readouterr().out


def test_project_code_follows_the_rules():
    root = SCRIPT.parent.parent
    assert check_jpl.main([str(root / "searchengine"), str(root / "scripts")]) == 0
