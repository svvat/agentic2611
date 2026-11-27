
import json
import os
import subprocess
import sys
from pathlib import Path

import requests


MODEL = "qwen2.5-coder:1.5b"
OLLAMA_URL = "http://localhost:11434/api/chat"
MAX_STEPS = 15

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "src" / "calculator.py"
TESTS = ROOT / "tests" / "test_calculator.py"


def run_tests():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=ROOT
    )

    return {
        "passed": result.returncode == 0,
        "output": result.stdout + result.stderr
    }


def execute_tool(action):
    tool = action.get("tool")

    if tool == "read_code":
        return SOURCE.read_text()

    if tool == "read_tests":
        return TESTS.read_text()

    if tool == "write_code":
        SOURCE.write_text(action["code"])
        return "Source code updated."

    if tool == "run_tests":
        return run_tests()

    return "Unknown tool."


def ask_model(messages):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0
            }
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


def main():
    task = os.getenv(
        "AGENT_TASK",
        "Investigate and repair the calculator."
    )

    system_prompt = """
You are an autonomous Python software engineer.

Your objective is to complete the user's task.

Available tools:

{"tool": "read_code"}
{"tool": "read_tests"}
{"tool": "run_tests"}
{"tool": "write_code", "code": "complete source code"}
{"tool": "finish"}

Return exactly one JSON object per response.

Choose your next action based on previous results.

Investigate the problem before changing code.
Preserve existing functionality.
Never modify the tests.
Run tests after changing code.
Only finish when all tests pass.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": task
        }
    ]

    for step in range(MAX_STEPS):

        print(f"\n--- Agent step {step + 1} ---")

        raw = ask_model(messages)
        print(raw)

        messages.append({
            "role": "assistant",
            "content": raw
        })

        try:
            action = json.loads(raw)

            if action.get("tool") == "finish":
                result = run_tests()

                if result["passed"]:
                    print("SUCCESS: All tests passed.")
                    print(result["output"])
                    return

            else:
                result = execute_tool(action)

        except (ValueError, KeyError, TypeError,
                AttributeError) as exc:
            result = {"error": str(exc)}

        print(result)

        messages.append({
            "role": "user",
            "content": json.dumps({
                "tool_result": result
            })
        })

    raise RuntimeError("Agent step limit reached.")


if __name__ == "__main__":
    main()
