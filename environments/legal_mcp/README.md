# legal-mcp

- **Environment ID**: `legal-mcp`
- **Short description**: Multi-turn tool-use environment that wraps an external MCP tool server from the `legalgenius` project, adds judge-based accuracy reward vs. gold answers, and applies negative rewards for token usage and tool-call count.
- **Tags**: legal, tools, multi-turn, MCP, judge, cost-penalty, train

## Overview

This environment exposes your existing legal MCP tools to Verifiers' `ToolEnv`. It:
- Converts your MCP tool endpoints into Python functions (with type hints/docstrings) used as OpenAI tools during inference via vLLM
- Adds a `JudgeRubric` to compare the final model output against a gold answer
- Adds token and tool-call penalties via custom reward functions
- Aggregates rewards using `RubricGroup`

Use it with local vLLM or the OpenAI API for evaluation. For RL training via `vf.GRPOTrainer`, use a local HF model (closed API models like OpenAI cannot be optimized directly).

## Requirements

- The external repo with the MCP server (defaults to `/home/sten/legalgenius`). You can override this path at load time.
- A working vLLM server with function calling on your GPUs.
- A dataset containing chat `prompt` messages and an `answer` (gold) per example.

## Usage

1) Install locally from this repo root:

```bash
# from the verifiers repo root
uv run vf-install legal-mcp -p ./environments
```

2a) Option A — Start vLLM on your GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 vf-vllm --model <your-model> \
  --data-parallel-size 2 --enforce-eager --disable-log-requests
```

2b) Option B — Use OpenAI API directly:

```bash
export OPENAI_API_KEY=sk-...
```

3) Load and evaluate/train in Python:

```python
import verifiers as vf
from datasets import Dataset
from legal_mcp import load_environment

data = [
  {"prompt": [{"role":"user","content":"Frage: Kündigungsfrist bei Mietvertrag?"}], "answer": "…gold…"},
]
ds = Dataset.from_list(data)

vf_env = load_environment(
  dataset=ds,
  judge_model="gpt-4.1-nano",           # or any OpenAI-compatible judge
  token_penalty_weight=-0.0005,
  toolcall_penalty_weight=-0.1,
  legalgenius_path="/home/sten/legalgenius",  # override as needed
)

# Quick eval
from openai import OpenAI

# Option A — local vLLM endpoint
# results = vf_env.evaluate(
#   client=OpenAI(base_url="http://localhost:8000/v1", api_key="dummy"),
#   model="http://localhost:8000/v1",
#   num_examples=1,
#   rollouts_per_example=1,
# )

# Option B — OpenAI API (remote)
results = vf_env.evaluate(
  client=OpenAI(),
  model="gpt-5-nano",  # change to your target OpenAI model
  num_examples=1,
  rollouts_per_example=1,
)
print(results.reward, results.metrics)
```

4) Train with GRPO (local models only):

```python
from verifiers import GRPOTrainer, grpo_defaults, get_model_and_tokenizer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model, tok = get_model_and_tokenizer(model_name)
args = grpo_defaults(run_name="legal-mcp-grpo")
args.per_device_train_batch_size = 4
args.num_generations = 8
args.gradient_accumulation_steps = 8
args.max_tokens = 1024
args.max_seq_len = 4096
args.eval_strategy = "steps"
args.eval_steps = 20

trainer = GRPOTrainer(model=model, processing_class=tok, env=vf_env, args=args)
trainer.train()
```

## Notes

- The tools here are wrappers over your MCP server. At load time, the environment will import the `MCPClient` and `build_dispatch_functions` from your `legalgenius/client/agent_cli.py` using a path you specify (`legalgenius_path`).
- Token usage is aggregated from `state['responses']` (OpenAI-compatible responses with `.usage`).
- Tool-call counts are taken from the final `completion` messages; negative weights penalize overuse.

## FAQ

- Can I do RL training using OpenAI models like `gpt-5-nano`?
  - No. GRPO/optimizer updates require access to model weights. Use a local HF model for training, and evaluate with OpenAI if desired. You can still use an OpenAI judge (`judge_model`) during training.

- How do I point to an OpenAI-compatible endpoint that is not api.openai.com?
  - Pass `client=OpenAI(base_url="https://your-endpoint/v1", api_key="...")` and set `model` accordingly.
