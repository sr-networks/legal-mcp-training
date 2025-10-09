from typing import Callable

from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages
from verifiers.utils.tool_utils import convert_func_to_oai_tool


class ToolRubric(Rubric):
    """Simple rubric that counts tool calls in completion messages."""

    def __init__(self, tools: list[Callable] | None = None):
        self.tools = tools or []
        self.oai_tools = [convert_func_to_oai_tool(tool) for tool in self.tools]
        self.tool_names = [tool.__name__ for tool in self.tools]

        # Build initial reward functions and weights
        reward_funcs = []
        reward_funcs.append(self.total_tool_calls)
        reward_weights = [0.0]

        for tool_name in self.tool_names:
            reward_funcs.append(self.get_tool_call_count_func(tool_name))
            reward_weights.append(0.0)

        # Pass them to parent class
        super().__init__(funcs=reward_funcs, weights=reward_weights)

    async def total_tool_calls(self, completion: Messages) -> float:
        """Count only executed tool calls by matching tool_results to tool_call_id.

        Strategy:
        - Find all non-blocked tool result messages and collect their `tool_call_id`.
        - Count unique IDs as executed tool calls.
        This avoids counting assistant-declared calls that were blocked.
        """
        assert isinstance(completion, list)
        executed_ids: set[str] = set()
        for msg in completion:
            if msg.get("role") != "tool":
                continue
            if msg.get("is_blocked_tool_call"):
                continue
            content = str(msg.get("content", ""))
            if content.startswith("Too many parallel tool calls:"):
                continue
            tcid = msg.get("tool_call_id")
            if isinstance(tcid, str) and tcid:
                executed_ids.add(tcid)
        return float(len(executed_ids))

    def get_tool_call_count_func(self, tool_name: str) -> Callable:
        """Create a reward function that counts calls to a specific tool."""

        async def tool_call_count_func(completion: Messages) -> float:
            """Count executed calls to a specific tool (matches by tool_call_id)."""
            count = 0
            assert isinstance(completion, list)

            # Build mapping from tool_call_id -> tool_name from assistant messages
            id_to_name: dict[str, str] = {}
            for msg in completion:
                if msg.get("role") == "assistant" and "tool_calls" in msg:
                    tool_calls = msg.get("tool_calls", [])
                    if not isinstance(tool_calls, list):
                        continue
                    for tc in tool_calls:
                        try:
                            tc_id = getattr(tc, "id", None) or (hasattr(tc, "model_dump") and tc.model_dump().get("id"))
                            func = getattr(tc, "function", None)
                            func_name = getattr(func, "name", None)
                            if isinstance(tc_id, str) and isinstance(func_name, str):
                                id_to_name[tc_id] = func_name
                        except Exception:
                            continue

            # Count only tool results that correspond to this tool and are not blocked
            executed_ids: set[str] = set()
            for msg in completion:
                if msg.get("role") != "tool":
                    continue
                if msg.get("is_blocked_tool_call"):
                    continue
                content = str(msg.get("content", ""))
                if content.startswith("Too many parallel tool calls:"):
                    continue
                tcid = msg.get("tool_call_id")
                if isinstance(tcid, str) and tcid:
                    executed_ids.add(tcid)

            for tcid in executed_ids:
                if id_to_name.get(tcid) == tool_name:
                    count += 1

            return float(count)

        tool_call_count_func.__name__ = f"{tool_name}_calls"
        return tool_call_count_func
