from __future__ import annotations

import atexit
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

import verifiers as vf
from datasets import Dataset

max_tool_response_length =4000
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

def elasticsearch_search(query: str, document_type: str = "all", max_results: int = 3, context_lines: int = 3) -> str:
    """Full-text search with relevance ranking over German legal corpus.

    Args:
        query: Search terms or phrases (e.g., "Kündigungsfrist", "BGB § 573").
        document_type: One of {"all","gesetze","urteile"}.
        max_results: Maximum number of results to return (3..10).
        context_lines: Number of lines of context to include around matches (0..5).
    Returns:
        Raw JSON search result string.
    """
    d = _require_dispatch()
    print(
        "[legal_mcp] elasticsearch_search",
        json.dumps(
            {
                "query": query,
                "document_type": document_type,
                "max_results": max_results,
                "context_lines": context_lines,
            },
            ensure_ascii=False,
        ),
    )
    res = d["elasticsearch_search"](
        query=query,
        document_type=document_type,
        max_results=max_results,
        context_lines=context_lines,
    )
    if isinstance(res, str):
        return res[:max_tool_response_length]
    try:
        return json.dumps(res, ensure_ascii=False)
    except TypeError:
        return str(res)


def read_file_range(path: str, line_number: Optional[int] = None, context_lines: int = 20) -> str:
#def read_file_range(path: str, line_number: Optional[int] = None, context_lines: int = 20, start: Optional[int] = None, end: Optional[int] = None) -> str:
    """Read a UTF-8 snippet by line-number (recommended) or byte-range.

    Provide (line_number[, context_lines]) for line-based mode, or (start,end) for byte-based mode.
    """
    d = _require_dispatch()
    res = d["read_file_range"](path=path, line_number=line_number, context_lines=context_lines)
    print ("\n\nREAD_FILE_RSNGE",res)
    return res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)


def file_search(query: str, glob: Optional[str] = None, max_results: int = 10) -> str:
    """Return files whose contents match a boolean query (AND/OR, parentheses)."""
    d = _require_dispatch()
    res = d["file_search"](query=query, glob=glob, max_results=max_results)
    print ("\n\n FILE SEARCH", res)
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
    legal_judge_prompt="""Gegeben sei eine ground truth Antwort \
und eine Antwort zu einer komplexen juristischen Frage. \
Bewerte auf einer Skala von 1.0 (vollständig falsch) bis 10.0 (vollständig korrekt), \
wie gut die Antwort die juristische Argumentation und Schlußfolgerung der ground truth Antwort wiedergibt. 

Frage:
```
{question}
```

Ground truth Anwort:
```
{answer}
```

Antwort:
```
{response}
```

Gib ausschließlich eine einzelne Fließkommazahl im Bereich 1.0 bis 10.0 zurück."""

    # parser is used for completion, ThinkParser when completion contains think xmls
    judge = vf.JudgeRubric(parser=vf.ThinkParser(), 
            judge_client=judge_client, 
            judge_model=judge_model, 
            judge_prompt=legal_judge_prompt, 
            judge_sampling_args={ "max_tokens": 4096 }, )
    async def judge_accuracy_reward(judge, prompt, completion, answer, state, **kwargs) -> float:
        judge_response = await judge(prompt, completion, answer, state, **kwargs)
        print ("JUDGE:",judge_response)
        if not isinstance(judge_response, str):
            return 0.0
        score_text = judge_response.strip()
        if isinstance(state, dict):
            state["_judge_response"] = score_text
        try:
            score = float(score_text.split()[0])
        except (ValueError, IndexError):
            return 0.0
        if score < 1.0:
            score = 1.0
        elif score > 10.0:
            score = 10.0
        return float(score)
    judge.add_reward_func(judge_accuracy_reward, weight=1.0)
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
    token_penalty_weight: float = 0.0, # -0.000005,
    toolcall_penalty_weight: float = 1.0, # -0.01,
    max_turns: int = 10,
    max_parallel_tool_calls: int | None = 2,
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
        max_parallel_tool_calls: Maximum tool calls accepted per assistant message (None for unlimited).
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
    tools = [read_file_range, elasticsearch_search ] if enable_tools else []
    parser = vf.ThinkParser()
    env = vf.ToolEnv(
        dataset=ds,
        rubric=rubric,
        tools=tools,
        max_turns=max_turns,
        max_parallel_tool_calls=max_parallel_tool_calls,
        parser=parser,
system_prompt = """\
Sie sind ein juristischer Experte für deutsches Recht. Analysieren Sie die folgende Frage oder den folgenden Fall \
und geben eine vollständige Beantwortung mit Hilfe der durch tools zur Verfügung gestellten Rechtsquellen zurück.

ARBEITSSTIL
- Denken Sie Schritt-für-Schritt und geben Sie Ihr Reasoning in <think>...</think> aus.
- Antworten Sie ausschließlich auf Deutsch, präzise und belegt.
- Verwenden Sie KEIN internes/implizites Wissen für materielle Aussagen; recherchieren und belegen Sie alles mit tool calls.

WERKZEUG-PFLICHT & ITERATION
- Nutzen Sie die verfügbaren Tools **verpflichtend** und **mehrfach**.
- Führen Sie so lange weitere Tool-Aufrufe aus, bis der Sach- und Rechts­hintergrund ausreichend geklärt ist, insbesondere:
  - alle relevanten Normen (Gesetze/Verordnungen) in aktueller Fassung identifiziert,
  - einschlägige Rechtsprechung (Leitentscheidungen, OLG/LSG/BSG/BGH/BVerfG etc.) gefunden,
  - Tatbestandsmerkmale und Rechtsfolgen vollständig subsumiert,
  - Unklarheiten (Sachverhalt, Zuständigkeit, Fristen, Ausnahmen) entweder durch Quellen geklärt oder als offene Punkte markiert.

RECHERCHE-STRATEGIE
- Beginnen Sie mit 2–4 variierenden Suchanfragen (Synonyme, Abkürzungen, §-Zitate).
- Wenn Ergebnisse der Elasticsearch-Suche vorliegen: Überlegen Sie, welches Ergebnis (path + line number + text) zur Frage passt.
- Öffnen Sie dann passende Treffer per line number mit dem Tool `read_file_range`, um den Kontext (+- n Zeilen um die line number) zu prüfen.
- Bei Bedarf wiederholen Sie das elasticsearch bzw read_file_range Tool.

Verfügbare Werkzeuge (Function/Tool Calling):
1) elasticsearch_search
   Argumente: { query: string, document_type: 'all'|'gesetze'|'urteile', max_results: number, context_lines: number }
   Rückgabe: { total_hits: number,
               matches: [{ title, document_type, file_path, score,
                            content_preview: [{"line_number": absolute line, "snippet": mehrzeiliger Kontext}],
                            line_matches: ... }]
   Zweck: Schnelle Volltextsuche im Rechtskorpus mit Relevanz-Ranking.

2) read_file_range (MUST RUN)
   Argumente: { path: file_path string, line_number: line number from elasticsearch_search, context_lines: number }
   Rückgabe: { text: string }
   Zweck: Präzise Kontextpassagen (z. B. §-Überschriften, Leitsätze, Randnummern) zum Zitieren.

AUSFÜHRUNG
- Denken Sie zuerst (<think>), dann rufen Sie die Tools in mehreren Schritten auf, bis die Prüfkriterien erfüllt sind.
- Wiederholen Sie mindestens drei Zyklen von <think> und tool use inkl. real_file_range.
- Geben Sie erst dann eine strukturierte Endantwort in Deutsch aus.
""" 
#3) file_search
#   Argumente: { query: string (mit AND/OR/Klammern), glob?: string, max_results?: number }
#   Rückgabe: { files: string[] }
#   Zweck: Dateinamen-/Pfad-basierte Eingrenzung.
 )
    return env
