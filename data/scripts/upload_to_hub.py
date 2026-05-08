"""Upload Horizon training dataset to HuggingFace Hub (SafeCircle org, private)."""

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence, Value
from huggingface_hub import HfApi
from tqdm import tqdm


HF_REPO_ID = "SafeCircle/horizon-training-data"

RAW_FILES = {
    "grooming": "data/raw/grooming.jsonl",
    "bullying": "data/raw/bullying.jsonl",
    "sexual_content": "data/raw/sexual_content.jsonl",
    "isolation": "data/raw/isolation.jsonl",
    "personal_info": "data/raw/personal_info.jsonl",
    "platform_migration": "data/raw/platform_migration.jsonl",
    "threats": "data/raw/threats.jsonl",
    "benign": "data/raw/benign.jsonl",
}

PROCESSED_FILES = {
    "train": "data/processed/train.jsonl",
    "eval": "data/processed/eval.jsonl",
}


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in tqdm(f, desc=f"Loading {Path(path).name}", leave=False):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_raw_dataset(root: Path) -> DatasetDict:
    splits = {}
    for category, rel_path in RAW_FILES.items():
        full_path = root / rel_path
        if not full_path.exists():
            print(f"  Skipping {category} — {full_path} not found")
            continue
        records = load_jsonl(str(full_path))
        splits[category] = Dataset.from_list(records)
        print(f"  {category}: {len(records):,} examples")
    return DatasetDict(splits)


def build_processed_dataset(root: Path) -> DatasetDict:
    splits = {}
    for split, rel_path in PROCESSED_FILES.items():
        full_path = root / rel_path
        if not full_path.exists():
            print(f"  Skipping {split} — {full_path} not found")
            continue
        records = load_jsonl(str(full_path))
        splits[split] = Dataset.from_list(records)
        print(f"  {split}: {len(records):,} examples")
    return DatasetDict(splits)


def ensure_repo_exists(api: HfApi, repo_id: str) -> None:
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"Repository {repo_id} already exists.")
    except Exception:
        print(f"Creating private dataset repo: {repo_id}")
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Horizon dataset to HuggingFace Hub")
    parser.add_argument(
        "--split",
        choices=["raw", "processed", "all"],
        default="all",
        help="Which data to upload (default: all)",
    )
    parser.add_argument(
        "--token",
        help="HuggingFace token (defaults to HF_TOKEN env var / cached login)",
    )
    parser.add_argument(
        "--repo-id",
        default=HF_REPO_ID,
        help=f"HuggingFace repo ID (default: {HF_REPO_ID})",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent

    api = HfApi(token=args.token)
    ensure_repo_exists(api, args.repo_id)

    if args.split in ("raw", "all"):
        print("\nBuilding raw dataset (per-category splits)...")
        raw_ds = build_raw_dataset(root)
        if raw_ds:
            print(f"Pushing raw dataset to {args.repo_id} (config: raw)...")
            raw_ds.push_to_hub(args.repo_id, config_name="raw", private=True, token=args.token)
            print("Raw dataset uploaded.")

    if args.split in ("processed", "all"):
        print("\nBuilding processed dataset (train/eval splits)...")
        processed_ds = build_processed_dataset(root)
        if processed_ds:
            print(f"Pushing processed dataset to {args.repo_id} (config: processed)...")
            processed_ds.push_to_hub(args.repo_id, config_name="processed", private=True, token=args.token)
            print("Processed dataset uploaded.")

    print(f"\nDone. Dataset available at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
