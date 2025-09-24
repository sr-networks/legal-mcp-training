import argparse
import os

import verifiers as vf
from legal_mcp import load_environment
from datasets import Dataset


def build_dataset_inline() -> Dataset:
    # Replace this with your real dataset
    data = [
        {"prompt": [{"role": "user", "content": "Frage: Kündigungsfrist bei Mietvertrag?"}], "answer": "...gold..."},
    ]
    return Dataset.from_list(data)


def main():
    ap = argparse.ArgumentParser(description="Train GRPO on legal_mcp ToolEnv")
    ap.add_argument("--model", default=os.getenv("TRAIN_MODEL", "Qwen/Qwen3-1.7B"))
    ap.add_argument("--judge-model", default="gpt-4.1-nano")
    ap.add_argument("--legalgenius-path", default=os.getenv("LEGALGENIUS_PATH", "/home/sten/legalgenius"))
    ap.add_argument("--judge-api-key", default=os.getenv("OPENAI_API_KEY", None))
    ap.add_argument("--max-steps", type=int, default=200)
    args = ap.parse_args()

    ds = build_dataset_inline()
    env = load_environment(
        dataset=ds,
        judge_model=args.judge_model,
        token_penalty_weight=-0.0005,
        toolcall_penalty_weight=-0.1,
        legalgenius_path=args.legalgenius_path,
        judge_api_key=args.judge_api_key,
    )

    model, tokenizer = vf.get_model_and_tokenizer(args.model)
    run_name = f"legal-mcp-grpo-{args.model}".replace("/", "-")
    training_args = vf.grpo_defaults(run_name=run_name)
    training_args.per_device_train_batch_size = 4
    training_args.num_generations = 8
    training_args.gradient_accumulation_steps = 8
    training_args.max_tokens = 1024
    training_args.max_seq_len = 4096
    training_args.eval_strategy = "steps"
    training_args.eval_steps = 20
    training_args.max_steps = args.max_steps

    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=env,
        args=training_args,
    )
    trainer.train()


if __name__ == "__main__":
    main()
