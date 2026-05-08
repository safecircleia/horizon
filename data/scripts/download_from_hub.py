"""Download Horizon training dataset from HuggingFace Hub."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


HF_REPO_ID = "safecircleai/horizon-training-data"

RAW_CATEGORIES = [
    "grooming", "bullying", "sexual_content", "isolation",
    "personal_info", "platform_migration", "threats", "benign",
]


def save_jsonl(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in tqdm(records, desc=f"Writing {path.name}", leave=False):
            f.write(json.dumps(dict(record)) + "\n")
    print(f"  Saved {len(records):,} records → {path}")


def download_raw(repo_id: str, token: str | None, output_dir: Path) -> None:
    print(f"\nDownloading raw dataset from {repo_id}...")
    ds = load_dataset(repo_id, name="raw", token=token)
    for category in RAW_CATEGORIES:
        if category not in ds:
            print(f"  Skipping {category} — not in dataset")
            continue
        save_jsonl(ds[category], output_dir / "raw" / f"{category}.jsonl")


def download_processed(repo_id: str, token: str | None, output_dir: Path) -> None:
    print(f"\nDownloading processed dataset from {repo_id}...")
    ds = load_dataset(repo_id, name="processed", token=token)
    for split in ("train", "eval"):
        if split not in ds:
            print(f"  Skipping {split} — not in dataset")
            continue
        save_jsonl(ds[split], output_dir / "processed" / f"{split}.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Horizon dataset from HuggingFace Hub")
    parser.add_argument(
        "--split",
        choices=["raw", "processed", "all"],
        default="all",
        help="Which data to download (default: all)",
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
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory (default: data/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    token = args.token

    if args.split in ("raw", "all"):
        download_raw(args.repo_id, token, output_dir)

    if args.split in ("processed", "all"):
        download_processed(args.repo_id, token, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
