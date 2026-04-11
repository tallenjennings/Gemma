# Ollama Tool Agent

`ollama_tool_agent.py` is a host-driven tool-calling agent for local Ollama models.
It enforces a strict JSON protocol so the model can request tools, while only the
Python host actually executes them.

## Requirements

- Python 3.11+
- `requests` package
- Ollama server running at `http://localhost:11434` (or another configured host)

Install dependency:

```bash
pip install requests
```

## Run

One-shot prompt:

```bash
python ollama_tool_agent.py "What is the weather in Calgary and what is 2*8?"
```

Interactive REPL:

```bash
python ollama_tool_agent.py
```

Exit the REPL with `exit` or `quit`.

## CLI flags

- `--model` (default: `gemma4:e2b`)
- `--host` (default: `http://localhost:11434`)
- `--debug` (prints turn count, raw model outputs, parsed tool calls, and tool results)

## JSON protocol

The model must respond with exactly one JSON object in one of these shapes:

Tool call:

```json
{"type":"tool_call","tool":"tool_name","arguments":{...}}
```

Final answer:

```json
{"type":"final","content":"..."}
```

If the model output is not valid JSON, the host attempts to extract the first JSON
object. If parsing still fails, output is treated as a final natural-language answer.

## Available tools (host-executed)

- `get_current_time(timezone="UTC")`
- `get_weather(city)`
- `internet_search(query)`
- `calculator(expression)`
- `powershell_access(command)`
- `echo(text)`

Unknown tools return a structured `ERROR:` result and the loop continues.

## Safety and guardrails

- Host-only tool execution (model never executes code directly)
- Max 8 turns per request
- AST-restricted arithmetic evaluator for `calculator`
- Clear `ERROR:` strings for invalid arguments and execution failures
