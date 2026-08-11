from __future__ import annotations

import asyncio
import ast
import hashlib
import inspect
from typing import Any, Mapping

from ..runtime import RunContext
from ..types import ProtocolError, ScaffoldLabError


_LAST_VALUE = "__scaffoldlab_repl_last_value__"


class _RecoverableREPLError(Exception):
    def __init__(self, error_type: str, message: str, printed: str = "") -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.printed = printed

    def observation(self) -> str:
        prefix = self.printed
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        return f"{prefix}{self.error_type}: {self.message}"


class _RestrictedSyntaxValidator(ast.NodeVisitor):
    """Reject Python capabilities that can escape the in-process namespace."""

    _DISALLOWED = (
        ast.AsyncFor,
        ast.AsyncFunctionDef,
        ast.AsyncWith,
        ast.ClassDef,
        ast.Delete,
        ast.FunctionDef,
        ast.Global,
        ast.Import,
        ast.ImportFrom,
        ast.Lambda,
        ast.Match,
        ast.Nonlocal,
        ast.Raise,
        ast.Try,
        ast.While,
        ast.With,
        ast.Yield,
        ast.YieldFrom,
    )

    def __init__(
        self,
        *,
        reserved_names: set[str],
        max_nodes: int,
    ) -> None:
        self.reserved_names = reserved_names
        self.max_nodes = max_nodes
        self.nodes = 0

    def generic_visit(self, node: ast.AST) -> None:
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise ProtocolError(
                f"restricted Python action exceeds {self.max_nodes} AST nodes"
            )
        if isinstance(node, self._DISALLOWED):
            raise ProtocolError(
                f"restricted Python does not allow {type(node).__name__}"
            )
        try_star = getattr(ast, "TryStar", None)
        if try_star is not None and isinstance(node, try_star):
            raise ProtocolError("restricted Python does not allow TryStar")
        super().generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            raise ProtocolError(
                "restricted Python does not allow attribute assignment or deletion"
            )
        introspection_prefixes = ("cr_", "gi_", "ag_", "f_", "tb_", "co_")
        coroutine_controls = {
            "cancel",
            "close",
            "exception",
            "format",
            "format_map",
            "get_coro",
            "get_stack",
            "print_stack",
            "result",
            "send",
            "throw",
            "uncancel",
        }
        if (
            node.attr.startswith("_")
            or node.attr.startswith(introspection_prefixes)
            or node.attr in coroutine_controls
        ):
            raise ProtocolError(
                "restricted Python does not allow private or runtime-introspection "
                "attributes"
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            raise ProtocolError("restricted Python does not allow dunder names")
        if (
            isinstance(node.ctx, (ast.Store, ast.Del))
            and node.id in self.reserved_names
        ):
            raise ProtocolError(f"cannot replace reserved REPL binding {node.id!r}")
        self.generic_visit(node)


class RestrictedPersistentPythonREPL:
    """Small persistent Python namespace with no import, I/O, or introspection.

    This is a capability restriction, not an operating-system security boundary. It
    deliberately exposes only immutable task data, safe builtins, and harness-provided
    async functions. Paid or adversarial runs still require an outer sandbox.
    """

    def __init__(
        self,
        bindings: Mapping[str, Any],
        *,
        max_source_chars: int = 12_000,
        max_ast_nodes: int = 500,
        max_output_chars: int = 20_000,
        max_range_items: int = 100_000,
        max_variables: int = 128,
    ) -> None:
        if max_source_chars < 1 or max_ast_nodes < 1 or max_output_chars < 1:
            raise ValueError("REPL limits must be positive")
        if max_range_items < 1 or max_variables < 1:
            raise ValueError("REPL range and namespace limits must be positive")
        self.max_source_chars = max_source_chars
        self.max_ast_nodes = max_ast_nodes
        self.max_output_chars = max_output_chars
        self.max_range_items = max_range_items
        self.max_variables = max_variables
        self._printed: list[str] = []

        def bounded_range(*args: int) -> range:
            if not 1 <= len(args) <= 3:
                raise ValueError("range expects one to three integer arguments")
            if any(
                not isinstance(value, int) or isinstance(value, bool) for value in args
            ):
                raise TypeError("range arguments must be integers")
            value = range(*args)
            if len(value) > self.max_range_items:
                raise ValueError(
                    f"range exceeds the {self.max_range_items}-item REPL limit"
                )
            return value

        def repl_print(*values: Any, sep: str = " ", end: str = "\n") -> None:
            if not isinstance(sep, str) or not isinstance(end, str):
                raise TypeError("print sep and end must be strings")
            self._printed.append(sep.join(str(value) for value in values) + end)

        safe_builtins: dict[str, Any] = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "range": bounded_range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }
        self._reserved_names = {
            "__builtins__",
            "print",
            *safe_builtins,
            *bindings,
        }
        self._namespace: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "print": repl_print,
            **dict(bindings),
        }

    @property
    def user_variables(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                name
                for name in self._namespace
                if name not in self._reserved_names and not name.startswith("__")
            )
        )

    async def execute(self, source: str) -> str:
        if not isinstance(source, str) or not source.strip():
            raise ProtocolError("restricted Python action must be non-empty")
        if len(source) > self.max_source_chars:
            raise ProtocolError(
                f"restricted Python action exceeds {self.max_source_chars} characters"
            )
        try:
            tree = ast.parse(source, mode="exec")
        except SyntaxError as exc:
            raise _RecoverableREPLError("SyntaxError", exc.msg) from exc
        _RestrictedSyntaxValidator(
            reserved_names=self._reserved_names,
            max_nodes=self.max_ast_nodes,
        ).visit(tree)

        if tree.body and isinstance(tree.body[-1], ast.Expr):
            final_expression = tree.body[-1]
            tree.body[-1] = ast.Assign(
                targets=[ast.Name(id=_LAST_VALUE, ctx=ast.Store())],
                value=final_expression.value,
                lineno=final_expression.lineno,
                col_offset=final_expression.col_offset,
                end_lineno=getattr(final_expression, "end_lineno", None),
                end_col_offset=getattr(final_expression, "end_col_offset", None),
            )
            ast.fix_missing_locations(tree)

        self._printed = []
        self._namespace.pop(_LAST_VALUE, None)
        try:
            compiled = compile(
                tree,
                "<restricted-repl>",
                "exec",
                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
                dont_inherit=True,
            )
            pending = eval(compiled, self._namespace, self._namespace)
            if inspect.isawaitable(pending):
                await pending
        except (asyncio.CancelledError, ScaffoldLabError):
            raise
        except SyntaxError as exc:
            raise _RecoverableREPLError(
                "SyntaxError",
                exc.msg,
                "".join(self._printed),
            ) from exc
        except (
            ArithmeticError,
            AssertionError,
            AttributeError,
            LookupError,
            NameError,
            TypeError,
            ValueError,
        ) as exc:
            raise _RecoverableREPLError(
                type(exc).__name__,
                str(exc),
                "".join(self._printed),
            ) from exc

        variables = self.user_variables
        if len(variables) > self.max_variables:
            raise ProtocolError(
                f"restricted Python namespace exceeds {self.max_variables} variables"
            )
        last_value = self._namespace.pop(_LAST_VALUE, None)
        output = "".join(self._printed)
        if last_value is not None:
            output += repr(last_value)
        if len(output) > self.max_output_chars:
            omitted = len(output) - self.max_output_chars
            output = output[: self.max_output_chars] + f"\n...[{omitted} chars omitted]"
        return output


async def execute_repl_tool(
    repl: RestrictedPersistentPythonREPL,
    source: str,
    context: RunContext,
    *,
    agent_id: str,
    role: str,
    tool_name: str,
) -> str:
    """Execute one REPL action with shared tool-budget and trace accounting."""

    await context.ledger.reserve_tool_call()
    source_bytes = source.encode("utf-8")
    started_data: dict[str, Any] = {
        "tool": tool_name,
        "kind": "restricted_python",
        "source_chars": len(source),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }
    if context.capture_content:
        started_data["source"] = source
    await context.trace.emit(
        "tool_call_started",
        agent_id=agent_id,
        role=role,
        data=started_data,
    )
    is_error = False
    try:
        try:
            output = await repl.execute(source)
        except _RecoverableREPLError as exc:
            output = exc.observation()
            is_error = True
        if len(output) > repl.max_output_chars:
            omitted = len(output) - repl.max_output_chars
            output = output[: repl.max_output_chars] + f"\n...[{omitted} chars omitted]"
        output_bytes = len(output.encode("utf-8"))
        await context.ledger.record_tool_output(output_bytes)
    except BaseException as exc:
        event = (
            "tool_call_cancelled"
            if isinstance(exc, asyncio.CancelledError)
            else "tool_call_failed"
        )
        await context.trace.emit(
            event,
            agent_id=agent_id,
            role=role,
            data={
                "tool": tool_name,
                "kind": "restricted_python",
                "error": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise
    completed_data: dict[str, Any] = {
        "tool": tool_name,
        "kind": "restricted_python",
        "output_bytes": output_bytes,
        "output_chars": len(output),
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "variables": list(repl.user_variables),
        "is_error": is_error,
    }
    if context.capture_content:
        completed_data["output"] = output
    await context.trace.emit(
        "tool_call_completed",
        agent_id=agent_id,
        role=role,
        data=completed_data,
    )
    return output
