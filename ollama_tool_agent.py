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

2) get_weather
   arguments: {"city": string (required)}
   returns: current weather summary for that city, or structured error string.

3) calculator
   arguments: {"expression": string (required)}
   returns: arithmetic evaluation result, or structured error string.
   supports numbers, parentheses, spaces, decimals, and + - * / // % **.

4) echo
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


WEATHER_CODE_DESCRIPTIONS: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def get_weather(city: str) -> str:
    """Return current weather summary for a city via Open-Meteo APIs."""
    city_name = city.strip()
    if not city_name:
        return 'ERROR: "city" is required and must be a non-empty string'

    geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
    forecast_url = "https://api.open-meteo.com/v1/forecast"

    try:
        geocode_resp = requests.get(
            geocode_url,
            params={"name": city_name, "count": 1, "language": "en", "format": "json"},
            timeout=20,
        )
        geocode_resp.raise_for_status()
        geocode_data = geocode_resp.json()
    except requests.RequestException as exc:
        return f"ERROR: weather geocoding request failed ({exc})"
    except ValueError:
        return "ERROR: weather geocoding returned invalid JSON"

    results = geocode_data.get("results")
    if not isinstance(results, list) or not results:
        return f'ERROR: city "{city_name}" not found'

    location = results[0]
    name = location.get("name")
    latitude = location.get("latitude")
    longitude = location.get("longitude")
    country = location.get("country")

    if not isinstance(name, str) or not isinstance(latitude, (int, float)) or not isinstance(
        longitude, (int, float)
    ):
        return "ERROR: geocoding response missing required location fields"

    try:
        weather_resp = requests.get(
            forecast_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,wind_speed_10m,weather_code",
            },
            timeout=20,
        )
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()
    except requests.RequestException as exc:
        return f"ERROR: weather forecast request failed ({exc})"
    except ValueError:
        return "ERROR: weather forecast returned invalid JSON"

    current = weather_data.get("current")
    if not isinstance(current, dict):
        return "ERROR: weather forecast missing current conditions"

    temperature = current.get("temperature_2m")
    wind_speed = current.get("wind_speed_10m")
    weather_code = current.get("weather_code")

    if not isinstance(temperature, (int, float)) or not isinstance(wind_speed, (int, float)):
        return "ERROR: weather forecast missing temperature or wind speed"

    if isinstance(weather_code, int):
        weather_desc = WEATHER_CODE_DESCRIPTIONS.get(weather_code, f"code {weather_code}")
    else:
        weather_desc = "unknown conditions"

    location_label = f"{name}, {country}" if isinstance(country, str) and country else name
    return (
        f"Current weather in {location_label}: "
        f"{float(temperature):.1f} C, "
        f"wind {float(wind_speed):.1f} km/h, "
        f"{weather_desc}."
    )


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

    if tool == "get_weather":
        city = args.get("city")
        if not isinstance(city, str):
            return 'ERROR: "city" is required and must be a string'
        return get_weather(city)

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
