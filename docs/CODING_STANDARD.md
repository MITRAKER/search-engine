# Coding standard: NASA/JPL's "Power of Ten", adapted to Python

In 2006 Gerard Holzmann of NASA's Jet Propulsion Laboratory published ten
rules for safety-critical code ("The Power of Ten: Rules for Developing
Safety-Critical Code", *IEEE Computer* 39(6)). They were written for C on
spacecraft, where a crash cannot be fixed by restarting the program. Their
purpose carries over to any language: code simple enough that a tool can
check it and a reviewer can trust it.

Python is not C, so some rules need translating. The table shows each rule,
what it means here, and what checks it. CI fails when one of the tools listed
finds a rule broken. "Review" means a person checks it in the pull request,
because no tool can see it. Our own checker, `scripts/check_jpl.py`, reads one
file at a time, so what a single file cannot show (calls through objects passed
as arguments, or between files) is also left to review. Its docstring lists
exactly what it covers.

| # | Original rule (C) | Our Python adaptation | Enforced by |
|---|---|---|---|
| 1 | Simple control flow: no `goto`, `setjmp`, or recursion | No recursion, direct or mutual. No exceptions used as `goto` (no `return`/`break` in `finally`). | `scripts/check_jpl.py` (JPL1) within a file, ruff `B012`; review for calls through objects and between files |
| 2 | Every loop has a fixed upper bound | No `while True`, nor any other loop that cannot end by construction (`while 1`, a loop over `itertools.count()`). Every loop runs over a finite collection or has a provable bound, and every input that drives a loop is size-limited (`MAX_TEXT_LENGTH`, `MAX_TOKEN_LENGTH`, `MAX_QUERY_LENGTH`, `MAX_QUERY_TERMS`, `MAX_RESULTS`, `MAX_FIELDS`, `MAX_LINE_BYTES`). | `check_jpl.py` (JPL2), ruff `PLR0133`, input checks, tests |
| 3 | No dynamic memory allocation after initialization | Python allocates constantly, so we bound memory instead: inputs, query sizes and result sizes are capped, data files are read one bounded line at a time, a saved index is read whole but may be at most `MAX_INDEX_BYTES` (500 MB), and nothing grows without limit (no unbounded caches). | input checks, tests |
| 4 | No function longer than one printed page (~60 lines) | At most 60 lines per function, McCabe complexity at most 10, at most 12 branches and 40 statements. | `check_jpl.py` (JPL4), ruff `C901`, `PLR0912`, `PLR0915` |
| 5 | Assertion density of at least two per function on average, used for anomalous conditions | Every public function validates its inputs and raises a clear `ValueError` or `TypeError`. `assert` is used only for internal invariants (Python removes asserts under `-O`, so they never guard user input). | review, tests for every error path |
| 6 | Declare data at the smallest possible scope | No `global` or `nonlocal`, no `globals()`, no changes to `sys.modules` or `sys.path`. Every module-level name is marked `Final` (`TypeAlias` for type aliases), and values kept at module or class level are immutable (`tuple`, `frozenset`, `MappingProxyType`, frozen dataclasses). Test files may keep module-level tables of cases. | `check_jpl.py` (JPL6), mypy `Final`; review for values built by function calls |
| 7 | Check every return value and every parameter | `mypy --strict` forces `None` and error cases to be handled. Functions check their arguments. Loaded files are validated before use. | mypy, ruff, tests |
| 8 | Limit the preprocessor to includes and simple macros | No code generated or loaded at runtime: no `eval`, `exec`, `compile`, `__import__`, `importlib.import_module` or `runpy`. No monkey-patching: nothing sets or deletes an attribute of something imported. Tests may use pytest's `monkeypatch` fixture, which undoes every change when the test ends. | `check_jpl.py` (JPL8), ruff `S307` |
| 9 | Restrict pointers: one level of dereference, no function pointers | Keep the call graph static so tools can follow it: no attribute found by a computed name (`getattr`, `setattr`, `hasattr`, `vars(obj)[name]` and the like), and choose behavior with explicit `if`/`else` rather than tables of functions. | `check_jpl.py` (JPL9); review for tables of functions |
| 10 | Compile with all warnings on, fix every warning, run static analyzers daily | ruff with 20 rule sets, `mypy --strict`, CodeQL, zero warnings (pytest turns any warning into a failure), and CI runs every day even without new commits. | CI (`ci.yml` runs daily), `filterwarnings = error` |

## Why each rule matters, in one line

1. Recursion can overflow the stack on an input you did not expect, and makes
   it hard to prove a function terminates.
2. A loop with no bound can hang the program. Bounding the input bounds the loop.
3. Unbounded growth eventually exhausts memory. Caps make the worst case known.
4. A function that fits on a screen can be understood, tested and reviewed as a unit.
5. Checks catch a bad state where it starts instead of three calls later.
6. The less code can change a value, the fewer places a bug can hide.
7. Ignored errors turn into wrong answers far away from their cause.
8. Code built at runtime cannot be checked before it runs.
9. Indirection hides control flow from both readers and tools.
10. Every warning ignored today is a bug report tomorrow.

## Other conventions

- Format with `ruff format` (Black-compatible), line length 100.
- Type-annotate everything. `mypy --strict` must pass.
- Docstrings explain *why*, not just *what*.
- Tests: one behavior per test, named for the behavior (`test_phrase_respects_word_order`).
  Property-based tests (Hypothesis) for code that must work on any input.
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/).
