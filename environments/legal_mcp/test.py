"""Offline smoke test for legal_mcp environment.

This test avoids network calls and heavyweight training. It simply verifies
that the environment can be constructed with a minimal dataset and exposes
the expected tool wrappers.
"""

import logging
import os
import csv
import glob
import random

import torch
from datasets import Dataset
from transformers import logging as hf_logging
from transformers.utils import is_flash_attn_2_available
from transformers import AutoTokenizer, AutoModelForImageTextToText
from legal_mcp import load_environment
from verifiers import GRPOTrainer, get_model_and_tokenizer, lora_defaults
from verifiers.trainers import GRPOConfig


def _discover_csvs() -> list[str]:
  # Preferred: comma-separated absolute/relative file paths
  csvs = os.getenv("LEGAL_MCP_CSVS")
  if csvs:
    paths = [p.strip() for p in csvs.split(",") if p.strip()]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
      raise FileNotFoundError(f"CSV(s) not found: {missing}")
    return paths

  # Or: directory containing CSVs
  csv_dir = os.getenv("LEGAL_MCP_CSV_DIR")
  if csv_dir:
    if not os.path.isdir(csv_dir):
      raise NotADirectoryError(f"LEGAL_MCP_CSV_DIR is not a directory: {csv_dir}")
    paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not paths:
      raise FileNotFoundError(f"No CSV files found in directory: {csv_dir}")
    return paths

  # Or: a glob pattern
  csv_glob = os.getenv("LEGAL_MCP_CSV_GLOB")
  if csv_glob:
    paths = sorted(glob.glob(csv_glob))
    if not paths:
      raise FileNotFoundError(f"LEGAL_MCP_CSV_GLOB matched no files: {csv_glob}")
    return paths

  raise RuntimeError(
    "Provide training CSVs via one of: LEGAL_MCP_CSVS (comma-separated paths), "
    "LEGAL_MCP_CSV_DIR (directory of CSVs), or LEGAL_MCP_CSV_GLOB (glob pattern)."
  )

os.environ.setdefault("WANDB_DISABLED", "false")
logging.basicConfig(level=logging.INFO)
hf_logging.set_verbosity_info()
hf_logging.enable_default_handler()
hf_logging.enable_explicit_format()

if is_flash_attn_2_available():
  attn_impl = "flash_attention_2"
else:
  attn_impl = "eager"
  logging.warning(
    "FlashAttention2 not available; falling back to eager attention. Install `flash-attn` to enable it."
  )

# Build data entries: each row -> {prompt: [{role, content}], answer}
data: list[dict] = []
for path in _discover_csvs():
  with open(path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    # Expect columns: url, question_text, answer_text
    for row in reader:
      q = (row.get("question_text") or "").strip()
      a = (row.get("answer_text") or "").strip()
      if not q or not a:
        continue
      data.append({
        "prompt": [{"role": "user", "content": q}],
        "answer": a,
      })

random.shuffle(data)
ds = Dataset.from_list(data)

logging.getLogger("AsyncBatchGenerator").setLevel(logging.DEBUG)
logging.getLogger("AsyncBatchGenerator").addHandler(logging.StreamHandler())

vf_env = load_environment(
  dataset=ds,
  judge_model=os.getenv("JUDGE_MODEL", "gpt-5-nano-2025-08-07" ), #"gpt-5-nano-2025-08-07"  gpt-4.1-nano-2025-04-14
  token_penalty_weight=0.0, #-0.000005,    # penalty when negative
  toolcall_penalty_weight=0.5, # -0.01,     # penalty when negative
  legalgenius_path=os.getenv("LEGALGENIUS_PATH", "/disk/legalgenius"),
  judge_base_url=os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1"),
  judge_api_key=os.getenv("JUDGE_API_KEY", os.getenv("OPENAI_API_KEY")),
  enable_tools=True # (os.getenv("DISABLE_TOOLS", "0").lower() not in {"1","true","yes"}),
)

# Quick eval
from openai import AsyncOpenAI
import asyncio

# Policy model served by vLLM (OpenAI-compatible server)
vllm_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
vllm_api_key = os.getenv("VLLM_API_KEY", "EMPTY")  # vLLM often ignores auth; keep placeholder
#policy_model = os.getenv("POLICY_MODEL", "willcb/Qwen3-4B")

# Preflight: ensure policy endpoint is reachable and model is served
policy_client = AsyncOpenAI(base_url=vllm_base_url, api_key=vllm_api_key)

#model_name = "ServiceNow-AI/Apriel-1.5-15b-Thinker"
model_name = "willcb/Qwen3-8B"
#model_name = "Qwen/Qwen3-8B"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_kwargs = {
  "torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
  "attn_implementation": attn_impl,
  "use_cache": False,
}
if device.type == "cpu":
  logging.warning("CUDA not available; using CPU for model execution.")
# Use Liger kernels only when running on CUDA. On CPU, Liger's Triton kernels
# require a GPU driver and will fail. This keeps CPU execution working.
use_liger = (device.type == "cuda")
model, tok = get_model_and_tokenizer(model_name, use_liger=use_liger, model_kwargs=model_kwargs)
#model = AutoModelForImageTextToText.from_pretrained(
#    model_name, 
#    torch_dtype=torch.bfloat16, 
#    device_map="auto"
#)
#tok = AutoTokenizer.from_pretrained(model_name)

model.to(device)

# Training config
args = GRPOConfig(
  output_dir="outputs/legal-mcp-grpo",
  run_name="legal-mcp-grpo",
  learning_rate=1e-6,
  lr_scheduler_type="constant_with_warmup",
  warmup_steps=10,
  max_steps=500,
  bf16=(device.type == "cuda"),
  fp16=False,
  no_cuda=(device.type != "cuda"),
  max_grad_norm=0.01,
  num_iterations=1,
  # Use a smaller context window by default to reduce VRAM
  max_prompt_length=1024,  # test because default=512
  max_seq_len=16348,    # prompt + multiple completions incl thoughts, tool results
  max_tokens=16348,  # multiple thoughts and final results
  per_device_train_batch_size=1,
  num_generations=8,
  gradient_accumulation_steps=8,
  gradient_checkpointing=True,
  save_strategy="steps",
  save_steps=5,
  save_only_model=True,
  logging_steps=1,
  log_on_each_node=False,
  log_completions=True,
  report_to=[],
)
args.max_tokens = 4096
args.logging_strategy = "steps"
args.disable_tqdm = False
args.vllm_server_host = "127.0.0.1"
args.logging_first_step = True
args.report_to = "wandb"

# Optional environment overrides for quick tuning
_max_steps = os.getenv("MAX_STEPS")
if _max_steps:
  try:
    args.max_steps = int(_max_steps)
  except Exception:
    pass
_pbs = os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE")
if _pbs:
  try:
    args.per_device_train_batch_size = int(_pbs)
  except Exception:
    pass
_ng = os.getenv("NUM_GENERATIONS")
if _ng:
  try:
    args.num_generations = int(_ng)
  except Exception:
    pass

# Optional: Override max sequence length via env for quick VRAM tuning
_msl = os.getenv("MAX_SEQ_LEN")
if _msl:
  try:
    args.max_seq_len = int(_msl)
  except Exception:
    pass

# Enable LoRA/PEFT to reduce trainable parameters and VRAM usage
_lora_r = int(os.getenv("LORA_R", "16"))
_lora_alpha = int(os.getenv("LORA_ALPHA", "64"))

peft_cfg = lora_defaults(r=_lora_r, alpha=_lora_alpha)


print ("RUNNING")
trainer = GRPOTrainer(
  model=model,
  processing_class=tok,
  env=vf_env,
  args=args,
  peft_config=peft_cfg,
)
trainer.train()
print ("DONE")

async def _preflight():
  models = await policy_client.models.list()
  ids = [m.id for m in models.data]
  print("Policy models:", ids)
  if policy_model not in ids:
    print(f"Warning: policy model '{policy_model}' not listed by server.")

#try:
#  asyncio.run(_preflight())
#except Exception as e:
#  print(f"Failed to reach policy server at {vllm_base_url}: {e}")
#  print("Hint: start vLLM with:\n  vllm serve Qwen/Qwen3-4B-Instruct-2507 --host 0.0.0.0 --port 8000 --api-key EMPTY --served-model-name 'Qwen/Qwen3-4B-Instruct-2507'")
#  raise

#results = vf_env.evaluate(
#  client=policy_client,
#  model=policy_model,
#  num_examples=10,
#  rollouts_per_example=8,
#)

#print(results.reward, results.metrics)
