"""Safe Python code execution tool for data analysis and computation."""

import io
import math
import re
import sys
import threading
from typing import Any

from agent.mcp.tools.base import (
    BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult,
)

# Modules allowed in the sandbox
_ALLOWED_MODULES = {
    "math",
    "statistics",
    "datetime",
    "json",
    "re",
    "collections",
    "itertools",
}

# Patterns that indicate dangerous code
_BLOCKED_PATTERNS = [
    r"\b__\w+__\b",        # dunder attributes (__import__, __builtins__, etc.)
    r"\bimport\s+",        # import statements
    r"\bopen\s*\(",        # file operations
    r"\bexec\s*\(",        # exec calls
    r"\beval\s*\(",        # eval calls
    r"\bcompile\s*\(",     # compile calls
    r"\bglobals\s*\(",     # globals access
    r"\blocals\s*\(",      # locals access
    r"\bgetattr\s*\(",     # attribute access
    r"\bsetattr\s*\(",     # attribute setting
    r"\bdelattr\s*\(",     # attribute deletion
    r"\bbreakpoint\s*\(",  # debugger
    r"\bos\.\w+",          # os module usage
    r"\bsys\.\w+",         # sys module usage
    r"\bsubprocess\b",     # subprocess module
    r"\bshutil\b",         # shutil module
    r"\bsocket\b",         # socket module
    r"\bctypes\b",         # ctypes module
    r"\bpickle\b",         # pickle module
    r"\bshelve\b",         # shelve module
]

_BLOCKED_RE = re.compile("|".join(_BLOCKED_PATTERNS), re.IGNORECASE)

MAX_OUTPUT_LENGTH = 10_000


def _build_safe_namespace() -> dict[str, Any]:
    """Build a restricted namespace with only safe builtins and pre-imported modules."""
    import collections
    import datetime
    import itertools
    import json as json_mod
    import re as re_mod
    import statistics

    # Safe subset of builtins
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bin": bin,
        "bool": bool,
        "chr": chr,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "frozenset": frozenset,
        "hash": hash,
        "hex": hex,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "iter": iter,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "print": None,  # Replaced below with captured print
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "type": type,
        "zip": zip,
    }

    namespace = {
        "__builtins__": safe_builtins,
        # Pre-imported safe modules
        "math": math,
        "statistics": statistics,
        "datetime": datetime,
        "json": json_mod,
        "re": re_mod,
        "collections": collections,
        "itertools": itertools,
    }

    return namespace


class _TimeoutError(Exception):
    """Raised when code execution exceeds timeout."""
    pass


class CodeExecutorTool(BaseTool):
    """안전한 Python 코드 실행 도구 -- 데이터 분석 및 계산"""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="code_executor",
            description="Python 코드를 안전한 샌드박스에서 실행합니다.",
            category="computation",
            help_text=(
                "Python 코드를 안전한 샌드박스에서 실행합니다. "
                "math, statistics, datetime, json 등 기본 모듈만 사용 가능합니다.\n"
                "사용 가능 모듈: math, statistics, datetime, json, re, collections, itertools\n"
                "제한: import, open, exec, eval, os, sys 등 위험한 기능은 차단됩니다.\n"
                "최대 실행 시간: 10초 (기본값), 최대 출력: 10,000자"
            ),
            parameters=[
                ToolParameter(
                    name="code",
                    type=ToolParamType.STRING,
                    description="실행할 Python 코드",
                ),
                ToolParameter(
                    name="timeout",
                    type=ToolParamType.INTEGER,
                    description="최대 실행 시간 (초, 기본값: 10, 최대: 30)",
                    required=False,
                    default=10,
                ),
            ],
        )

    def _validate_code(self, code: str) -> str | None:
        """Validate code for dangerous patterns. Returns error message or None."""
        match = _BLOCKED_RE.search(code)
        if match:
            return f"Blocked pattern detected: '{match.group()}'. Dangerous operations are not allowed."
        return None

    async def execute(self, **kwargs) -> ToolResult:
        code = kwargs.get("code", "")
        timeout = kwargs.get("timeout", 10)

        if not code:
            return ToolResult(success=False, error="code parameter is required")

        # Clamp timeout
        timeout = max(1, min(timeout, 30))

        # Static validation
        error = self._validate_code(code)
        if error:
            return ToolResult(success=False, error=error)

        try:
            output, result_value = self._run_sandboxed(code, timeout)

            data: dict[str, Any] = {}
            if output:
                if len(output) > MAX_OUTPUT_LENGTH:
                    data["stdout"] = output[:MAX_OUTPUT_LENGTH]
                    data["truncated"] = True
                else:
                    data["stdout"] = output
            if result_value is not None:
                data["result"] = result_value

            if not data:
                data["stdout"] = "(no output)"

            return ToolResult(
                success=True,
                data=data,
                metadata={"timeout": timeout},
            )

        except _TimeoutError:
            return ToolResult(
                success=False,
                error=f"Code execution timed out after {timeout}s",
            )
        except SyntaxError as e:
            return ToolResult(
                success=False,
                error=f"Syntax error: {e}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Execution error: {type(e).__name__}: {e}",
            )

    def _run_sandboxed(self, code: str, timeout: int) -> tuple[str, Any]:
        """
        Execute code in a sandboxed namespace with timeout enforcement.
        Returns (stdout_output, last_expression_value).
        """
        namespace = _build_safe_namespace()

        # Capture stdout
        stdout_capture = io.StringIO()

        # Inject captured print
        def safe_print(*args, **print_kwargs):
            sep = print_kwargs.get("sep", " ")
            end = print_kwargs.get("end", "\n")
            text = sep.join(str(a) for a in args) + end
            stdout_capture.write(text)

        namespace["__builtins__"]["print"] = safe_print

        # Execute with timeout using threading
        exec_result: dict[str, Any] = {"value": None, "error": None}

        def target():
            try:
                # Try to extract last expression for result
                compiled = compile(code, "<sandbox>", "exec")
                exec(compiled, namespace)

                # Check if last line is a pure expression and capture its value.
                # Skip statements, assignments, and function calls that produce
                # side effects (like print) to avoid double-execution.
                lines = code.strip().split("\n")
                last_line = lines[-1].strip() if lines else ""
                _skip_prefixes = (
                    "if ", "for ", "while ", "def ", "class ", "import ",
                    "from ", "try:", "except", "finally:", "with ", "raise ",
                    "return ", "pass", "break", "continue", "#", "elif ",
                    "else:", "print(", "print (",
                )
                if (
                    last_line
                    and not any(last_line.startswith(kw) for kw in _skip_prefixes)
                    and "=" not in last_line.split("#")[0]
                    and "(" not in last_line  # skip any function calls to avoid side effects
                ):
                    try:
                        val = eval(last_line, namespace)
                        if val is not None:
                            exec_result["value"] = repr(val)
                    except Exception:
                        pass
            except Exception as e:
                exec_result["error"] = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            raise _TimeoutError("Execution timed out")

        if exec_result["error"] is not None:
            raise exec_result["error"]

        output = stdout_capture.getvalue()
        return output, exec_result["value"]
