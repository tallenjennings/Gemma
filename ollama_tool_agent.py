#!/usr/bin/env python3
"""
Local Ollama tool-calling agent (host-driven execution loop).

This script talks to Ollama's HTTP API directly and enforces a strict JSON
protocol for tool use:

1) Tool call:
   {"type":"tool_call","tool":"tool_name","arguments":{...}}
2) Final answer:
   {"type":"final","content":"..."}

The host process is the only component that executes tools.
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SYSTEM_PROMPT = """You are a careful assistant operating in a host-driven tool loop.

Available tools:
1) get_current_time
   arguments: {"timezone": string (optional, default "UTC")}
   returns: current time in ISO format for the timezone, or structured error string.

2) calculator
   arguments: {"expression": string (required)}
   returns: arithmetic evaluation result, or structured error string.
   supports numbers, parentheses, spaces, decimals, and + - * / // % **.

3) echo
   arguments: {"text": string (required)}
   returns: same text.

Critical rules:
- Respond with EXACTLY one JSON object and no extra text.
- Your response must be exactly one of:
  {"type":"tool_call","tool":"tool_name","arguments":{...}}
  {"type":"final","content":"..."}
- Never invent tool results.
- If you need information/computation from a tool, emit tool_call JSON.
- After receiving a tool result, either emit another tool_call or emit final.
"""


@dataclass
class ParsedResponse:
    """Normalized representation of the model response."""

    kind: str
    raw_text: str
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    content: str | None = None


def debug_log(debug: bool, message: str) -> None:
    """Print debug output when enabled."""
    if debug:
        print(f"[debug] {message}")


def call_ollama(
    messages: list[dict[str, str]],
    model: str,
    host: str,
    timeout_s: int = 60,
) -> str:
    """
    Call Ollama /api/chat in non-streaming mode and return assistant content.
    """
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    try:
        response = requests.post(url, json=payload, timeout=timeout_s)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Invalid JSON response from Ollama: {response.text}") from exc

    message = data.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"Unexpected Ollama response format: {data}")

    content = message.get("content", "")
    if not isinstance(content, str):
        raise RuntimeError(f"Unexpected message content type: {type(content)}")
    return content.strip()


def extract_first_json_object(text: str) -> str | None:
    """
    Extract the first top-level JSON object substring from text.
    Handles quoted strings and escaped characters while scanning.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def parse_model_response(raw_text: str) -> ParsedResponse:
    """
    Parse model output into protocol shape.

    Parsing strategy:
    1) strict full-text JSON parse
    2) extract first JSON object and parse
    3) fallback to natural language final answer
    """
    parsed_obj: Any | None = None
    text = raw_text.strip()

    try:
        parsed_obj = json.loads(text)
    except json.JSONDecodeError:
        candidate = extract_first_json_object(text)
        if candidate is not None:
            try:
                parsed_obj = json.loads(candidate)
            except json.JSONDecodeError:
                parsed_obj = None

    if isinstance(parsed_obj, dict):
        msg_type = parsed_obj.get("type")
        if msg_type == "tool_call":
            tool = parsed_obj.get("tool")
            arguments = parsed_obj.get("arguments", {})
            if isinstance(tool, str) and isinstance(arguments, dict):
                return ParsedResponse(
                    kind="tool_call",
                    raw_text=raw_text,
                    tool=tool,
                    arguments=arguments,
                )
        elif msg_type == "final":
            content = parsed_obj.get("content")
            if isinstance(content, str):
                return ParsedResponse(kind="final", raw_text=raw_text, content=content)

    # Fallback: natural-language final answer.
    return ParsedResponse(kind="final", raw_text=raw_text, content=text or raw_text)


def get_current_time(timezone: str = "UTC") -> str:
    """Return current ISO timestamp in the specified timezone."""
    tz = timezone or "UTC"
    try:
        now = datetime.now(ZoneInfo(tz))
    except ZoneInfoNotFoundError:
        return f'ERROR: invalid timezone "{tz}"'
    return now.isoformat()


ALLOWED_BIN_OPS: dict[type[ast.AST], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

ALLOWED_UNARY_OPS: dict[type[ast.AST], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

SAFE_EXPR_PATTERN = re.compile(r"^[0-9+\-*/%().\s]+$")


def safe_calculate(expression: str) -> str:
    """
    Safely evaluate arithmetic expressions via AST with restricted nodes.
    """
    expr = expression.strip()
    if not expr:
        return "ERROR: expression is required"

    if not SAFE_EXPR_PATTERN.fullmatch(expr):
        return "ERROR: expression contains invalid characters"

    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        return "ERROR: invalid arithmetic expression"

    def eval_node(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return eval_node(n.body)

        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)

        if isinstance(n, ast.BinOp):
            op_type = type(n.op)
            fn = ALLOWED_BIN_OPS.get(op_type)
            if fn is None:
                raise ValueError("unsupported binary operator")
            left = eval_node(n.left)
            right = eval_node(n.right)
            return fn(left, right)

        if isinstance(n, ast.UnaryOp):
            op_type = type(n.op)
            fn = ALLOWED_UNARY_OPS.get(op_type)
            if fn is None:
                raise ValueError("unsupported unary operator")
            operand = eval_node(n.operand)
            return fn(operand)

        raise ValueError(f"unsupported syntax: {type(n).__name__}")

    try:
        value = eval_node(node)
    except ZeroDivisionError:
        return "ERROR: division by zero"
    except ValueError as exc:
        return f"ERROR: {exc}"
    except OverflowError:
        return "ERROR: numeric overflow"
    except Exception as exc:  # defensive guardrail
        return f"ERROR: calculation failed ({exc})"

    # Keep integer-looking results tidy.
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def execute_tool(tool: str, arguments: dict[str, Any] | None) -> str:
    """Execute the named tool with validated arguments."""
    args = arguments or {}

    if tool == "get_current_time":
        timezone = args.get("timezone", "UTC")
        if not isinstance(timezone, str):
            return 'ERROR: "timezone" must be a string'
        return get_current_time(timezone)

    if tool == "calculator":
        expression = args.get("expression")
        if not isinstance(expression, str):
            return 'ERROR: "expression" is required and must be a string'
        return safe_calculate(expression)

    if tool == "echo":
        text = args.get("text")
        if not isinstance(text, str):
            return 'ERROR: "text" is required and must be a string'
        return text

    return f'ERROR: unknown tool "{tool}"'


def run_agent(user_input: str, model: str, host: str, debug: bool = False) -> str:
    """
    Run host-driven tool loop and return final answer.

    Guardrails:
    - max 8 turns
    - fallback to last model text if no valid final JSON is produced
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    max_turns = 8
    last_raw = ""

    for turn in range(1, max_turns + 1):
        debug_log(debug, f"turn={turn}")
        try:
            raw = call_ollama(messages=messages, model=model, host=host)
        except Exception as exc:
            return f"Agent error: {exc}"

        last_raw = raw
        debug_log(debug, f"raw_model_output={raw}")
        parsed = parse_model_response(raw)

        if parsed.kind == "final":
            final_text = parsed.content if parsed.content is not None else raw
            debug_log(debug, f"final_answer={final_text}")
            return final_text

        # Tool call path.
        tool_name = parsed.tool or ""
        arguments = parsed.arguments or {}
        debug_log(debug, f"parsed_tool_call tool={tool_name} arguments={arguments}")
        tool_result = execute_tool(tool_name, arguments)
        debug_log(debug, f"tool_result={tool_result}")

        # Append assistant tool request and host tool result back into chat.
        messages.append({"role": "assistant", "content": raw})
        tool_payload = {
            "tool": tool_name,
            "arguments": arguments,
            "result": tool_result,
        }
        messages.append({"role": "user", "content": f"TOOL_RESULT: {json.dumps(tool_payload)}"})

    # Fallback if loop exits without final JSON result.
    fallback = last_raw.strip()
    return fallback if fallback else "Agent stopped without producing a final answer."


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Host-side tool-calling loop for local Ollama models."
    )
    parser.add_argument("prompt", nargs="?", help="Optional one-shot user prompt")
    parser.add_argument("--model", default="gemma4:e2b", help="Ollama model name")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")
    args = parser.parse_args()

    if args.prompt:
        answer = run_agent(args.prompt, model=args.model, host=args.host, debug=args.debug)
        print(answer)
        return

    # Interactive REPL mode.
    print(
        "Interactive mode. Type a prompt and press Enter.\n"
        "Type 'exit' or 'quit' to leave."
    )
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        answer = run_agent(user_input, model=args.model, host=args.host, debug=args.debug)
        print(answer)


if __name__ == "__main__":
    main()
