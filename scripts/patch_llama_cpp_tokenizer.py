#!/usr/bin/env python3
"""Apply Model Lab's authoritative ByteLevel overlay to llama.cpp."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

MARKER = "# Marker: Start get_vocab_base_pre"
OVERLAY_MARKER = "# Model Lab ByteLevel pre-tokenizer override (authoritative)"
OVERLAY_PHRASE = "Model Lab uses tokenizers.ByteLevel(add_prefix_space=False)."
OVERLAY_SIGNATURES = (
    'backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)',
    'pre_tokenizer = getattr(backend_tokenizer, "pre_tokenizer", None)',
    'pre_tokenizer.__class__.__name__ == "ByteLevel"',
)


def _target_function(lines: list[str]) -> ast.FunctionDef:
    try:
        tree = ast.parse("\n".join(lines) + "\n")
    except SyntaxError as exc:
        raise RuntimeError(f"llama.cpp converter source is not valid Python: {exc}") from exc
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "get_vocab_base_pre"
    ]
    markers = [i for i, line in enumerate(lines) if line.strip() == MARKER]
    if len(markers) != 1:
        raise RuntimeError("Unsupported llama.cpp converter source: generated get_vocab_base_pre marker is missing or ambiguous")
    if len(functions) != 1:
        raise RuntimeError("Unsupported llama.cpp converter source: expected exactly one get_vocab_base_pre function")
    function = functions[0]
    marker_line = markers[0]
    if marker_line >= function.lineno - 1 or function.lineno - 1 - marker_line > 5:
        raise RuntimeError("Unsupported llama.cpp converter source: generated marker is not adjacent to target function")
    return function


def _is_failure_check(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "res"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Is)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value is None
    )


def _overlay(indent: str) -> list[str]:
    inner = indent + "    "
    return [
        f"{indent}{OVERLAY_MARKER}",
        f'{indent}backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)',
        f'{indent}pre_tokenizer = getattr(backend_tokenizer, "pre_tokenizer", None)',
        f'{indent}if pre_tokenizer is not None and pre_tokenizer.__class__.__name__ == "ByteLevel":',
        f"{inner}# Model Lab uses tokenizers.ByteLevel(add_prefix_space=False).",
        f"{inner}# Map the standard ByteLevel behavior to llama.cpp's GPT-2",
        f"{inner}# pre-tokenizer implementation without relying on a vocab hash.",
        f'{inner}res = "gpt-2"',
        "",
    ]


def _remove_existing_overlay(lines: list[str], function: ast.FunctionDef) -> list[str]:
    start = function.lineno - 1
    end = function.end_lineno
    body = lines[start:end]
    markers = [i for i, line in enumerate(body) if OVERLAY_MARKER in line]
    phrases = [i for i, line in enumerate(body) if OVERLAY_PHRASE in line]
    if len(markers) > 1 or len(phrases) > 1:
        raise RuntimeError("Multiple tokenizer overlays found; refusing to modify source")

    if not markers and not phrases:
        if any(signature in line for line in body for signature in OVERLAY_SIGNATURES):
            raise RuntimeError("Unmarked tokenizer overlay-like code found; refusing to modify source")
        return lines

    if markers:
        block_start = markers[0]
        if len(phrases) != 1 or phrases[0] < block_start:
            raise RuntimeError("Malformed marked tokenizer overlay; refusing to modify source")
        phrase = phrases[0]
    else:
        phrase = phrases[0]
        candidates = [
            i for i, line in enumerate(body[:phrase])
            if line.strip() == OVERLAY_SIGNATURES[0]
        ]
        if len(candidates) != 1:
            raise RuntimeError("Unsupported legacy tokenizer overlay; refusing to modify source")
        block_start = candidates[0]

    results = [
        i for i, line in enumerate(body[phrase:], start=phrase)
        if line.strip() == 'res = "gpt-2"'
    ]
    if len(results) != 1:
        raise RuntimeError("Tokenizer overlay has no unique result assignment")
    block_end = results[0]
    indent = body[block_start][:len(body[block_start]) - len(body[block_start].lstrip())]
    expected = _overlay(indent)[:-1]
    if not markers:
        expected = expected[1:]
    if [line.rstrip() for line in body[block_start:block_end + 1]] != [line.rstrip() for line in expected]:
        raise RuntimeError("Malformed tokenizer overlay; refusing to modify source")

    # Fail closed if overlay-like statements also occur outside the recognized block.
    for i, line in enumerate(body):
        if block_start <= i <= block_end:
            continue
        if any(signature in line for signature in OVERLAY_SIGNATURES):
            raise RuntimeError("Unmarked tokenizer overlay-like code found outside the recognized overlay; refusing to modify source")

    absolute_start, absolute_end = start + block_start, start + block_end + 1
    if absolute_end < len(lines) and not lines[absolute_end].strip():
        absolute_end += 1
    return lines[:absolute_start] + lines[absolute_end:]


def patch_tokenizer_registry(llamacpp_dir: str | Path) -> bool:
    """Apply or migrate the overlay; return whether the converter file changed."""
    path = Path(llamacpp_dir).resolve() / "conversion" / "base.py"
    if not path.is_file():
        raise FileNotFoundError(f"llama.cpp converter source not found: {path}")
    raw = path.read_bytes()
    original = raw.decode("utf-8")
    newline = "\r\n" if b"\r\n" in raw else "\n"
    lines = original.splitlines()
    function = _target_function(lines)
    function_start = function.lineno - 1
    function_end = function.end_lineno
    for index, line in enumerate(lines):
        if function_start <= index < function_end:
            continue
        if any(signature in line for signature in OVERLAY_SIGNATURES):
            raise RuntimeError(
                "Unmarked tokenizer overlay-like code found outside target function; refusing to modify source"
            )
    lines = _remove_existing_overlay(lines, function)
    function = _target_function(lines)
    checks = [node for node in function.body if _is_failure_check(node)]
    if len(checks) != 1:
        raise RuntimeError("Unsupported llama.cpp converter source: expected exactly one top-level 'if res is None' failure check")
    failure_line = checks[0].lineno - 1
    indent = lines[failure_line][:len(lines[failure_line]) - len(lines[failure_line].lstrip())]
    if not indent or indent.strip():
        raise RuntimeError("Unexpected indentation at converter failure check")
    lines[failure_line:failure_line] = _overlay(indent)
    updated = newline.join(lines) + (newline if original.endswith(("\n", "\r")) else "")

    final_lines = updated.splitlines()
    final_function = _target_function(final_lines)
    final_checks = [node for node in final_function.body if _is_failure_check(node)]
    positions = [i for i, line in enumerate(final_lines) if line.strip() == OVERLAY_MARKER]
    if len(final_checks) != 1 or len(positions) != 1:
        raise RuntimeError("Internal validation failed: tokenizer overlay placement is ambiguous")
    if not (final_function.lineno - 1 < positions[0] < final_checks[0].lineno - 1):
        raise RuntimeError("Internal validation failed: tokenizer overlay placement is ambiguous")
    ast.parse(updated)
    if updated == original:
        return False
    path.write_bytes(updated.encode("utf-8"))
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply Model Lab's llama.cpp tokenizer compatibility overlay")
    parser.add_argument("--llamacpp-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    changed = patch_tokenizer_registry(args.llamacpp_dir)
    print("Model Lab llama.cpp tokenizer overlay applied" if changed else "Model Lab llama.cpp tokenizer overlay already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
