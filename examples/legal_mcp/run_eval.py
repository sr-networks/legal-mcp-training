import argparse
import os

from datasets import Dataset
from openai import OpenAI

from legal_mcp import load_environment


def build_dataset_inline() -> Dataset:
    data = [
        {"prompt": [{"role": "user", "content": "Frage: Kündigungsfrist bei Mietvertrag?"}], "answer": "...gold..."},
    ]
    return Dataset.from_list(data)


def main():
    ap = argparse.ArgumentParser(description="Run eval for legal_mcp against a local vLLM server")
    ap.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "dummy"))
    ap.add_argument("--judge-model", default="gpt-4.1-nano")
    ap.add_argument("--legalgenius-path", default=os.getenv("LEGALGENIUS_PATH", "/home/sten/legalgenius"))
    ap.add_argument("--num-examples", type=int, default=5)
    ap.add_argument("--rollouts", type=int, default=2)
    args = ap.parse_args()

    ds = build_dataset_inline()  # Replace by loading your dataset
    env = load_environment(
        dataset=ds,
        judge_model=args.judge_model,
        token_penalty_weight=-0.0005,
        toolcall_penalty_weight=-0.1,
        legalgenius_path=args.legalgenius_path,
        judge_api_key=args.api_key,
    )

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    results = env.evaluate(
        client=client,
        model=args.base_url,
        num_examples=args.num_examples,
        rollouts_per_example=args.rollouts,
    )
    print("avg_reward:", sum(results.reward)/max(1,len(results.reward)))
    print("metrics keys:", list(results.metrics.keys()))


if __name__ == "__main__":
    main()

