"""An agent harness built from scratch: an explicit state machine with schema-
validated tool calls, context-window accounting, and error recovery (Phase 25).

The model is a scripted fake -- a deterministic queue of intended actions, tool
calls or a final answer -- so the harness is the code under test: parsing,
schema validation, execution, result formatting, accounting, and the loop itself.

The state machine
  IDLE -> PLAN -> ACT -> PLAN -> ... -> ANSWER -> DONE
  any  -> ERROR on an uncaught violation (the run asserts ERROR is never entered)

The scripted episode (a CVE triage task) deliberately contains two failures:
  - a schema-invalid call (wrong argument type), rejected and reported back
  - a runtime error in the tool (division by zero), reported back
and the scripted model recovers from both, which the checks assert.

Checks (self-test mode, the default)
  task-completed     the episode reaches DONE with the expected final answer
  schema-rejection   the malformed call is rejected with a typed-schema error,
                     never executed, and a valid retry succeeds
  runtime-recovery   the dividing call errors, the error text is returned to the
                     model, and the retry succeeds
  context-accounting the recorded peak context tokens equal the peak recomputed
                     independently from the transcript
  state-machine      every observed transition is in the allowed set and ERROR
                     is never entered

Run:  ./.venv/bin/python code/phase25_harness/agent_loop.py
"""
from __future__ import annotations

import ast
import math
import re


def tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


class SchemaError(ValueError):
    pass


class Tool:
    def __init__(self, name: str, arg_types: dict[str, str], fn):
        self.name, self.arg_types, self.fn = name, arg_types, fn

    def validate(self, args) -> None:
        if not isinstance(args, dict):
            raise SchemaError(f"{self.name}: arguments must be a mapping")
        for k, v in args.items():
            if k not in self.arg_types:
                raise SchemaError(f"{self.name}: unknown argument {k!r}")
        for k, want in self.arg_types.items():
            if k not in args:
                raise SchemaError(f"{self.name}: missing required argument {k!r}")
            got = type(args[k])
            if got is not want:
                raise SchemaError(
                    f"{self.name}: argument {k!r} must be {want.__name__}, got {got.__name__}")


def safe_calc(expression: str) -> float:
    """Evaluate a numeric expression over + - * / ** and parentheses only."""
    def ok(node):
        if isinstance(node, ast.Expression):
            return ok(node.body)
        if isinstance(node, ast.BinOp) and isinstance(node.op,
                (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            return ok(node.left) and ok(node.right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return ok(node.operand)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return True
        raise ValueError(f"disallowed expression element: {ast.dump(node)}")
    ok(ast.parse(expression, mode="eval"))
    return float(eval(expression))  # only arithmetic constants/operators survive ok()


CVE_TABLE = [
    {"id": "CVE-2024-3010", "cvss": 9.8, "summary": "Kernel heap overflow in module X"},
    {"id": "CVE-2024-1122", "cvss": 7.5, "summary": "Buffer overread in network stack"},
    {"id": "CVE-2023-9001", "cvss": 9.1, "summary": "Integer overflow in image parser"},
]
FILES = {
    "/etc/cve_notes.txt":
        "triage notes\nCVE-2024-3010 confirmed in kernel 6.1\n"
        "CVE-2023-9001 patched upstream\nfollow up: verify fix release",
}


def build_tools() -> list[Tool]:
    def search_cve(query: str, min_severity: float) -> str:
        hits = [c for c in CVE_TABLE
                if query.lower() in c["summary"].lower() and c["cvss"] >= min_severity]
        return " | ".join(f"{c['id']} ({c['cvss']})" for c in hits) or "no match"

    def read_file(path: str) -> str:
        if path not in FILES:
            raise KeyError(f"no such file: {path}")
        return FILES[path]

    def grep(pattern: str, path: str) -> str:
        if path not in FILES:
            raise KeyError(f"no such file: {path}")
        return "\n".join(l for l in FILES[path].splitlines() if re.search(pattern, l))

    def python(expression: str) -> str:
        return repr(safe_calc(expression))

    def calculator(expression: str) -> str:
        return repr(safe_calc(expression))

    return [
        Tool("search_cve", {"query": str, "min_severity": float}, search_cve),
        Tool("read_file", {"path": str}, read_file),
        Tool("grep", {"pattern": str, "path": str}, grep),
        Tool("python", {"expression": str}, python),
        Tool("calculator", {"expression": str}, calculator),
    ]


# the scripted model: a queue of intended actions for each PLAN state
SCRIPT = [
    ("tool", "search_cve", {"query": "kernel", "min_severity": 9.0}),
    ("tool", "read_file", {"path": 42}),                          # schema-invalid
    ("tool", "read_file", {"path": "/etc/cve_notes.txt"}),        # the retry
    ("tool", "calculator", {"expression": "1/0"}),                # runtime error
    ("tool", "calculator", {"expression": "10 / 4"}),             # the retry
    ("tool", "grep", {"pattern": r"CVE-\d+-\d+", "path": "/etc/cve_notes.txt"}),
    ("final", "CVE-2024-3010 (CVSS 9.8) is the top kernel CVE; "
              "notes confirm it; retry quotient is 2.5"),
]

ALLOWED = {
    ("IDLE", "PLAN"), ("PLAN", "ACT"), ("PLAN", "ANSWER"),
    ("ACT", "PLAN"), ("ANSWER", "DONE"),
}
SYSTEM_PROMPT = ("You are a security triage agent. Use the tools to answer. "
                 "Call one tool per step, then continue.")


class Harness:
    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}
        self.state = "IDLE"
        self.transitions: list[tuple[str, str]] = []
        self.transcript: list[str] = [SYSTEM_PROMPT]
        self.context_tokens = len(tokenize(SYSTEM_PROMPT))
        self.context_peak = self.context_tokens
        self.steps = 0
        self.errors = 0
        self.invalid_calls = 0
        self.answer = None

    def go(self, target: str) -> None:
        pair = (self.state, target)
        if pair not in ALLOWED:
            self.state = "ERROR"
            return
        self.transitions.append(pair)
        self.state = target

    def observe(self, message: str) -> None:
        """Append a message to the transcript and update the context accounting."""
        self.transcript.append(message)
        self.context_tokens += len(tokenize(message))
        self.context_peak = max(self.context_peak, self.context_tokens)

    def run_tool(self, name: str, args: dict) -> str:
        tool = self.tools.get(name)
        if tool is None:
            self.invalid_calls += 1
            return f"error: unknown tool {name!r}"
        try:
            tool.validate(args)
        except SchemaError as exc:
            self.invalid_calls += 1
            return f"error: schema: {exc}"
        try:
            return str(tool.fn(**args))
        except Exception as exc:  # reported back to the model, not raised
            self.errors += 1
            return f"error: {type(exc).__name__}: {exc}"

    def run(self, script: list[tuple]) -> None:
        self.go("PLAN")
        for action in script:
            self.steps += 1
            if action[0] == "tool":
                _, name, args = action
                self.go("ACT")
                self.observe(f"call {name} {args}")
                result = self.run_tool(name, args)
                self.observe(f"result {result}")
                self.go("PLAN")
            else:
                _, text = action
                self.go("ANSWER")
                self.observe(f"answer {text}")
                self.answer = text
                self.go("DONE")
                break


def main() -> int:
    harness = Harness(build_tools())
    harness.run(SCRIPT)

    # ------------------------------------------------------------- task-completed
    assert harness.state == "DONE", f"episode ended in {harness.state}"
    assert harness.answer is not None and harness.answer.startswith("CVE-2024-3010"), \
        f"final answer wrong: {harness.answer!r}"
    print(f"PASS task-completed (episode DONE in {harness.steps} steps; answer "
          f"names CVE-2024-3010)")

    # ----------------------------------------------------------- schema-rejection
    assert harness.invalid_calls == 1, f"invalid calls {harness.invalid_calls} != 1"
    rejected = [m for m in harness.transcript if "error: schema:" in m]
    assert len(rejected) == 1 and "must be str, got int" in rejected[0], \
        f"schema rejection missing or wrong: {rejected}"
    assert any("triage notes" in m for m in harness.transcript), \
        "the valid read_file retry never returned the file"
    print(f"PASS schema-rejection (call with int path rejected with a typed error, "
          f"never executed; the str retry returned the file)")

    # --------------------------------------------------------- runtime-recovery
    assert harness.errors == 1, f"runtime errors {harness.errors} != 1"
    zero = [m for m in harness.transcript if "ZeroDivisionError" in m]
    assert len(zero) == 1, f"division-by-zero not reported once: {zero}"
    assert any("2.5" in m for m in harness.transcript), "the retry result is missing"
    print(f"PASS runtime-recovery (1/0 raised, the error text went back to the model, "
          f"the retry returned 2.5)")

    # ------------------------------------------------------- context-accounting
    recomputed = 0
    peak = 0
    for message in harness.transcript:
        recomputed += len(tokenize(message))
        peak = max(peak, recomputed)
    assert recomputed == harness.context_tokens, \
        f"accounting {harness.context_tokens} != transcript {recomputed}"
    assert peak == harness.context_peak, f"peak {harness.context_peak} != {peak}"
    print(f"PASS context-accounting (final {harness.context_tokens} tokens and peak "
          f"{harness.context_peak} both match the independent transcript recount)")

    # -------------------------------------------------------------- state-machine
    for a, b in harness.transitions:
        assert (a, b) in ALLOWED, f"illegal transition {a} -> {b}"
    assert "ERROR" not in [s for _, s in harness.transitions] \
        and harness.state != "ERROR", "the machine entered ERROR"
    n_act = sum(1 for _, b in harness.transitions if b == "ACT")
    assert n_act == 6, f"ACT entries {n_act} != 6 tool steps"
    print(f"PASS state-machine (all {len(harness.transitions)} transitions legal, "
          f"ERROR never entered, {n_act} ACT entries for {n_act} tool steps)")

    print(f"METRIC steps={harness.steps}")
    print(f"METRIC context_peak_tokens={harness.context_peak}")
    print(f"METRIC recovered_errors={harness.errors + harness.invalid_calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
