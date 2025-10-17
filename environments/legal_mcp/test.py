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
from peft import LoraConfig

_original_flatten_conversation = GRPOTrainer._flatten_conversation


def _flatten_conversation_without_system(messages):
  if isinstance(messages, (list, tuple)):
    filtered = []
    for msg in messages:
      if isinstance(msg, dict) and msg.get("role") == "system":
        continue
      filtered.append(msg)
    return _original_flatten_conversation(filtered)
  return _original_flatten_conversation(messages)


# Strip system-only scaffolding from WandB prompt logs so the field reflects user input.
GRPOTrainer._flatten_conversation = staticmethod(_flatten_conversation_without_system)

system_prompt = """
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
- Beginnen Sie mit variierenden Suchanfragen (Synonyme, Abkürzungen, §-Zitate).
- Rufen Sie pro Schritt nur ein Tool auf.
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

2) read_file_range
   Argumente: { path: file_path string, line_number: line number from elasticsearch_search, context_lines: number }
   Rückgabe: { text: string }
   Zweck: Präzise Kontextpassagen (z. B. §-Überschriften, Leitsätze, Randnummern) zum Zitieren.

AUSFÜHRUNG
- Denken Sie zuerst (<think>), dann rufen Sie die Tools in mehreren Schritten auf, bis die Prüfkriterien erfüllt sind.
- Geben Sie erst dann eine strukturierte Endantwort in Deutsch aus.
"""

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
QUESTION_KEYS = ("question_text", "sachverhalt")
ANSWER_KEYS = ("answer_text", "urteil")


def _clean_cell(value: str | None) -> str:
  text = (value or "").strip()
  if len(text) >= 2 and text[0] == text[-1] == '"':
    text = text[1:-1].strip()
  return text


def _pick(row: dict[str, str | None], keys: tuple[str, ...]) -> str:
  for key in keys:
    if key in row:
      return _clean_cell(row.get(key))
  return ""


data: list[dict] = []
for path in _discover_csvs():
  with open(path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f, skipinitialspace=True)
    for row in reader:
      q = _pick(row, QUESTION_KEYS)
      a = _pick(row, ANSWER_KEYS)
      if not q or not a:
        continue
      data.append({
        "prompt": [{"role": "system", "content": system_prompt},
          {"role": "user", "content": q}],
        "answer": a,
      })

random.shuffle(data)
ds = Dataset.from_list(data)

logging.getLogger("AsyncBatchGenerator").setLevel(logging.DEBUG)
logging.getLogger("AsyncBatchGenerator").addHandler(logging.StreamHandler())

vf_env = load_environment(
  dataset=ds,
  judge_model=os.getenv("JUDGE_MODEL", "gpt-5-nano-2025-08-07" ), #"gpt-5-nano-2025-08-07"  gpt-4.1-nano-2025-04-14
  token_penalty_weight=-0.000005,    # penalty when negative
  toolcall_penalty_weight=0.0, # -0.01,     # penalty when negative
  legalgenius_path=os.getenv("LEGALGENIUS_PATH", "/disk/legalgenius"),
  judge_base_url=os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1"),
  judge_api_key=os.getenv("JUDGE_API_KEY", os.getenv("OPENAI_API_KEY")),
  enable_tools=True, # (os.getenv("DISABLE_TOOLS", "0").lower() not in {"1","true","yes"}),
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
model_name = "willcb/Qwen3-14B"
#model_name = "Qwen/Qwen3-8B"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_kwargs = {
  "torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
  "attn_implementation": attn_impl,
  "use_cache": False,
    # >>> Add YaRN to match vLLM <<<
#  "rope_scaling": {
#    "type": "yarn",                           # same as vLLM's rope_type
#    "factor": 4.0,                            # same factor
#    "original_max_position_embeddings": 32768 # same as your vLLM flag
#  }
}
if device.type == "cpu":
  logging.warning("CUDA not available; using CPU for model execution.")
# Use Liger kernels only when running on CUDA. On CPU, Liger's Triton kernels
# require a GPU driver and will fail. This keeps CPU execution working.
use_liger = (device.type == "cuda")
model, tok = get_model_and_tokenizer(model_name, use_liger=use_liger, model_kwargs=model_kwargs)

if tok.pad_token is None:
    tok.pad_token = tok.eos_token
if tok.bos_token is None and hasattr(tok, "eos_token"):
    tok.bos_token = tok.eos_token  # Qwen often uses same
tok.model_max_length = max(tok.model_max_length, 131072)  # match YaRN context


model.to(device)

#sanity checks
#assert getattr(model.config, "rope_scaling", None), "YaRN not applied locally"
#rs = model.config.rope_scaling
#assert rs.get("type") == "yarn" and float(rs.get("factor")) == 4.0 \
#       and int(rs.get("original_max_position_embeddings")) == 32768


# Training config
args = GRPOConfig(
  output_dir="outputs/legal-mcp-grpo",
  run_name="legal-mcp-grpo",
  learning_rate=1e-5,
  lr_scheduler_type="constant_with_warmup",
  warmup_steps=10,
  max_steps=200,
  bf16=(device.type == "cuda"),
  fp16=False,
  no_cuda=(device.type != "cuda"),
  num_iterations=1,
  # Use a smaller context window by default to reduce VRAM
  max_prompt_length=3000,  # test because default=512
  max_seq_len=20000,    # (verifiers) Model's context window 131072, prompt + multiple completions incl thoughts, tool results
  max_tokens=1500,  # (verifiers) Max tokens per (turn-level) response 
  per_device_train_batch_size=1,
  num_generations=4,
  gradient_accumulation_steps=32,
  gradient_checkpointing=True,
  save_strategy="steps",
  save_steps=30,
  save_only_model=True,
  logging_steps=1,
  log_on_each_node=False,
  log_completions=True,
  report_to=[],
  beta=0.0,
  max_grad_norm=0.5,
)

args.logging_strategy = "steps"
args.disable_tqdm = False
args.vllm_server_host = "127.0.0.1"
args.logging_first_step = True
args.report_to = "wandb"
args.mask_truncated_completions = True  # quick truncation flag

model.train()
model.config.use_cache = False
if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


# Enable LoRA/PEFT to reduce trainable parameters and VRAM usage
_lora_r = int(os.getenv("LORA_R", "1"))
_lora_alpha = int(os.getenv("LORA_ALPHA", "16"))

#peft_cfg = lora_defaults(r=_lora_r, alpha=_lora_alpha,)

mlp_only_targets = ["gate_proj", "up_proj", "down_proj"]  # adjust to your model’s naming

peft_cfg = LoraConfig(
    r=_lora_r,
    lora_alpha=_lora_alpha,
    target_modules=mlp_only_targets,
    task_type="CAUSAL_LM",
    lora_dropout=0.0,        # keep whatever settings you need
    bias="none",
)


print ("RUNNING")
trainer = GRPOTrainer(
  model=model,
  processing_class=tok,
  env=vf_env,
  args=args,
  peft_config=peft_cfg,
)

n_all  = sum(p.numel() for p in model.parameters())
n_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {n_grad}/{n_all}")
assert n_grad > 0, "No trainable parameters – check LoRA/PEFT config"



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
