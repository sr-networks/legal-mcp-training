from __future__ import annotations

import atexit
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

import verifiers as vf
from datasets import Dataset


_MCP = None
_DISPATCH = None


def _load_legalgenius_module(legalgenius_path: str) -> Any:
    """Dynamically import legalgenius/client/agent_cli.py and return the module.

    This avoids adding a hard dependency on the external project while letting
    us reuse its MCPClient and dispatch functions.
    """
    agent_cli = Path(legalgenius_path).expanduser().resolve() / "client" / "agent_cli.py"
    if not agent_cli.exists():
        raise FileNotFoundError(f"agent_cli.py not found at: {agent_cli}")
    # Ensure 'client/...' intra-project imports resolve
    import sys
    sys.path.insert(0, str(Path(legalgenius_path).resolve()))
    spec = importlib.util.spec_from_file_location("legalgenius_agent_cli", str(agent_cli))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for: {agent_cli}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _ensure_mcp(legalgenius_path: str, cfg: dict | None = None, server_cmd: Optional[list[str]] = None) -> None:
    """Ensure a singleton MCP client and DISPATCH dict are available."""
    global _MCP, _DISPATCH
    if _MCP is not None and _DISPATCH is not None:
        return
    mod = _load_legalgenius_module(legalgenius_path)
    MCPClient = getattr(mod, "MCPClient")
    build_dispatch_functions = getattr(mod, "build_dispatch_functions")
    env = os.environ.copy()
    # Ensure PYTHONPATH includes the legalgenius project for any relative imports
    env.setdefault("PYTHONPATH", str(Path(legalgenius_path).resolve()))
    # propagate LEGAL_DOC_ROOT if present in cfg
    if cfg and "legal_doc_root" in cfg:
        env.setdefault("LEGAL_DOC_ROOT", str(cfg["legal_doc_root"]))
    _MCP = MCPClient(server_cmd=server_cmd, cwd=Path(legalgenius_path), env=env, logger=None)
    _DISPATCH = build_dispatch_functions(_MCP, cfg or {})

    def _cleanup() -> None:
        try:
            if _MCP is not None:
                _MCP.close()
        except Exception:
            pass

    atexit.register(_cleanup)


def _require_dispatch() -> dict[str, Callable[..., Any]]:
    if _DISPATCH is None:
        raise RuntimeError("MCP dispatch not initialized. Call load_environment with legalgenius_path.")
    return _DISPATCH


# --------- Tool wrappers (type hints + docstrings) ---------

def elasticsearch_search(query: str, document_type: str = "all", max_results: int = 10, context_lines: int = 2) -> str:
    """Full-text search with relevance ranking over German legal corpus.

    Args:
        query: Search terms or phrases (e.g., "Kündigungsfrist", "BGB § 573").
        document_type: One of {"all","gesetze","urteile"}.
        max_results: Maximum number of results to return (3..50).
        context_lines: Number of lines of context to include around matches (0..10).
    Returns:
        JSON-serialized string with search results.
    """
    d = _require_dispatch()
    res = d["elasticsearch_search"](query=query, document_type=document_type, max_results=max_results, context_lines=context_lines)
    # dispatch returns a python object; ensure string
    return res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)


def read_file_range(path: str, line_number: Optional[int] = None, context_lines: int = 2, start: Optional[int] = None, end: Optional[int] = None) -> str:
    """Read a UTF-8 snippet by line-number (recommended) or byte-range.

    Provide (line_number[, context_lines]) for line-based mode, or (start,end) for byte-based mode.
    """
    d = _require_dispatch()
    res = d["read_file_range"](path=path, line_number=line_number, context_lines=context_lines, start=start, end=end)
    return res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)


def file_search(query: str, glob: Optional[str] = None, max_results: int = 20) -> str:
    """Return files whose contents match a boolean query (AND/OR, parentheses)."""
    d = _require_dispatch()
    res = d["file_search"](query=query, glob=glob, max_results=max_results)
    return res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)


# --------- Rewards: judge + cost penalties ---------

async def token_penalty(completion: list[dict], state: dict | None = None, **_kwargs) -> float:
    """Return total token usage accumulated in state.responses.

    The reward runner passes `completion` and `state` by keyword. Keep the
    parameter name `completion` for compatibility, and pull `state` from either
    the explicit param or kwargs to be robust to different callers.
    """
    st = state or _kwargs.get("state") or {}
    total = 0
    for r in st.get("responses", []):
        try:
            # Support both object-style and dict-style usage payloads
            u = getattr(r, "usage", None)
            if u is None and isinstance(r, dict):
                u = r.get("usage")
            if u:
                prompt_toks = getattr(u, "prompt_tokens", None)
                if prompt_toks is None and isinstance(u, dict):
                    prompt_toks = u.get("prompt_tokens", 0)
                completion_toks = getattr(u, "completion_tokens", None)
                if completion_toks is None and isinstance(u, dict):
                    completion_toks = u.get("completion_tokens", 0)
                total += (prompt_toks or 0) + (completion_toks or 0)
        except Exception:
            # Be conservative and continue on malformed entries
            pass
    return float(total)


async def tool_call_penalty(completion: list[dict], **_kwargs) -> float:
    count = 0
    for m in completion:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            try:
                count += len(m["tool_calls"])  # type: ignore[arg-type]
            except Exception:
                pass
    return float(count)


def _make_rubric(judge_model: str, token_penalty_weight: float, toolcall_penalty_weight: float, judge_api_key: Optional[str] = None, judge_base_url: Optional[str] = None) -> vf.Rubric:
    from openai import AsyncOpenAI
    judge_client = AsyncOpenAI(api_key=judge_api_key or os.getenv("OPENAI_API_KEY", "dummy"), base_url=judge_base_url or os.getenv("JUDGE_BASE_URL", None))
    judge = vf.JudgeRubric(parser=vf.Parser(), judge_client=judge_client, judge_model=judge_model, judge_sampling_args={"max_tokens": 64})
    costs = vf.Rubric(funcs=[token_penalty, tool_call_penalty], weights=[token_penalty_weight, toolcall_penalty_weight])
    return vf.RubricGroup([judge, costs])


def _to_dataset(data: list[dict[str, Any]] | None) -> Dataset | None:
    if not data:
        return None
    # Expect entries to have prompt (chat messages) and answer (str)
    return Dataset.from_list(data)  # type: ignore[arg-type]


def load_environment(
    dataset: Optional[Dataset] = None,
    data: Optional[list[dict[str, Any]]] = None,
    judge_model: str = "gpt-4.1-nano",
    token_penalty_weight: float = -0.0005,
    toolcall_penalty_weight: float = -0.1,
    max_turns: int = 10,
    judge_base_url: Optional[str] = None,
    judge_api_key: Optional[str] = None,
    legalgenius_path: Optional[str] = None,
    mcp_server_cmd: Optional[list[str]] = None,
    cfg: Optional[dict] = None,
    enable_tools: bool = True,
) -> vf.ToolEnv:
    """Load the Legal MCP ToolEnv environment.

    Args:
        dataset: Optional HF Dataset with 'prompt' (chat messages) and 'answer' per row.
        data: Optional in-memory list of dicts (same fields) to build a Dataset.
        judge_model: OpenAI-compatible model for judge scoring.
        token_penalty_weight: Negative weight applied to total tokens over rollout.
        toolcall_penalty_weight: Negative weight applied to number of tool calls.
        max_turns: Max assistant-tool turns for ToolEnv.
        legalgenius_path: Path to the legalgenius project root (defaults env LEGALGENIUS_PATH or '/home/sten/legalgenius').
        mcp_server_cmd: Optional explicit command to launch the MCP server.
        cfg: Optional config dict propagated to the MCP environment (e.g., legal_doc_root).
        enable_tools: If False, disables tool-use (no tools passed to the model).

    Returns:
        A configured vf.ToolEnv ready for evaluation/training.
    """
    lg_path = legalgenius_path or os.environ.get("LEGALGENIUS_PATH", "/home/sten/legalgenius")
    _ensure_mcp(lg_path, cfg=cfg or {}, server_cmd=mcp_server_cmd)

    ds = dataset or _to_dataset(data)
    if ds is None:
        # Minimal placeholder to satisfy Environment constructor; user should pass real data
        ds = Dataset.from_list([
            {"prompt": [{"role": "user", "content": "Frage: Beispiel"}], "answer": ""}
        ])

    rubric = _make_rubric(judge_model, token_penalty_weight, toolcall_penalty_weight, judge_api_key=judge_api_key, judge_base_url=judge_base_url)
    tools = [elasticsearch_search, read_file_range, file_search] if enable_tools else []
    env = vf.ToolEnv(
        dataset=ds,
        rubric=rubric,
        tools=tools,
        max_turns=max_turns,
    )
    return env
