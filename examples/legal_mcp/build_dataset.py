import argparse
import json
from typing import List, Dict

from datasets import Dataset


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def to_hf_dataset(rows: List[Dict]) -> Dataset:
    formatted = []
    for r in rows:
        q = r.get("question") or r.get("query") or r.get("prompt") or ""
        a = r.get("answer") or ""
        formatted.append({
            "prompt": [{"role": "user", "content": f"Frage: {q}"}],
            "answer": a,
        })
    return Dataset.from_list(formatted)


def main():
    ap = argparse.ArgumentParser(description="Build HF dataset for legal_mcp from JSONL (question, answer)")
    ap.add_argument("--input", required=True, help="Path to JSONL with fields: question, answer")
    ap.add_argument("--save", default="", help="Optional path to save as HF dataset (arrow) using .to_json")
    args = ap.parse_args()

    rows = load_jsonl(args.input)
    ds = to_hf_dataset(rows)
    print(ds)
    if args.save:
        ds.to_json(args.save)
        print(f"Saved dataset to {args.save}")


if __name__ == "__main__":
    main()

