#!/usr/bin/env python3
"""Upload Horizon models to HuggingFace Hub and Cloudflare R2.

Uploads to both destinations in sequence: HuggingFace first (model cards + files),
then R2 (versioned folder structure with manifest). Use --skip-hf or --skip-r2
to upload to only one.

Usage:
    python scripts/upload.py --what edge-2b --version 2.1.0
    python scripts/upload.py --what all --version 2.1.0
    python scripts/upload.py --what edge-2b --version 2.1.0 --skip-hf
    python scripts/upload.py --what full gguf --version 2.0.0 --dry-run

Environment:
    HF_TOKEN               HuggingFace token (or run: huggingface-cli login)
    S3_BUCKET              R2 bucket name
    S3_ENDPOINT_URL        https://<account_id>.r2.cloudflarestorage.com
    S3_ACCESS_KEY_ID       R2 access key
    S3_SECRET_ACCESS_KEY   R2 secret key
    S3_PREFIX              Folder prefix in bucket (default: "models")

Prerequisites:
    uv pip install huggingface_hub boto3
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, create_repo

# ── Model registry ───────────────────────────────────────────────────────────

ORG = "safecircleai"

MODELS = {
    "full": {
        "hf_repo": f"{ORG}/horizon-full",
        "local_dir": "models/horizon-full-merged",
        "patterns": ["*.safetensors", "*.json", "*.jinja", "*.model"],
        "hf_upload_folder": True,
        "hf_ignore": ["*.py", "*.sh"],
    },
    "gguf": {
        "hf_repo": f"{ORG}/horizon-full-gguf",
        "local_dir": "models/horizon-full-gguf",
        "patterns": ["*.gguf"],
    },
    "mobile": {
        "hf_repo": f"{ORG}/horizon-mobile",
        "local_dir": "models/mobile-standard",
        "patterns": ["*.litertlm"],
        "extra_dirs": ["models/mobile-lite"],
    },
    "edge-2b": {
        "hf_repo": f"{ORG}/horizon-edge-2b",
        "local_dir": "models/horizon-edge-2b-litert",
        "patterns": ["*.litertlm"],
    },
    "edge-4b": {
        "hf_repo": f"{ORG}/horizon-edge-4b",
        "local_dir": "models/horizon-edge-4b-litert",
        "patterns": ["*.litertlm"],
    },
}
}


def collect_files(model_key: str) -> list[Path]:
    config = MODELS[model_key]
    dirs = [config["local_dir"]] + config.get("extra_dirs", [])
    files = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        for pattern in config["patterns"]:
            files.extend(sorted(p.glob(pattern)))
    return files


# ── HuggingFace ──────────────────────────────────────────────────────────────

def _ensure_repo(api: HfApi, repo_id: str):
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True, private=True)
    except Exception as e:
        print(f"    Warning: {e}")


def _upload_hf_model(api: HfApi, model_key: str, card: str, dry_run: bool):
    config = MODELS[model_key]
    repo = config["hf_repo"]
    files = collect_files(model_key)

    if not files:
        print(f"  [{model_key}] No files found - skipping HF")
        return

    print(f"  [{model_key}] -> {repo} ({len(files)} files)")
    if dry_run:
        for f in files:
            print(f"    [DRY RUN] {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
        return

    _ensure_repo(api, repo)

    if card:
        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=repo,
            commit_message="Update model card",
        )

    license_path = Path(__file__).parent.parent / "LICENSE-SAFECIRCLE.md"
    if license_path.exists():
        api.upload_file(
            path_or_fileobj=str(license_path),
            path_in_repo="LICENSE-SAFECIRCLE.md",
            repo_id=repo,
            commit_message="Add license",
        )

    if config.get("hf_upload_folder"):
        api.upload_folder(
            folder_path=config["local_dir"],
            repo_id=repo,
            commit_message=f"Upload {model_key}",
            ignore_patterns=config.get("hf_ignore", []),
        )
    else:
        for f in files:
            size_mb = f.stat().st_size / 1e6
            print(f"    {f.name} ({size_mb:.0f} MB)")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f.name,
                repo_id=repo,
                commit_message=f"Upload {f.name}",
            )

    print(f"    Done: https://huggingface.co/{repo}")


def upload_to_hf(models_to_upload: list[str], cards: dict, dry_run: bool):
    print("\n" + "=" * 60)
    print("HUGGINGFACE")
    print("=" * 60)

    api = HfApi()
    try:
        user = api.whoami()
        print(f"  Logged in as: {user['name']}")
    except Exception:
        print("  ERROR: Not logged in. Run: huggingface-cli login")
        return False

    for key in models_to_upload:
        _upload_hf_model(api, key, cards.get(key, ""), dry_run)

    return True


# ── Cloudflare R2 (S3-compatible) ────────────────────────────────────────────

def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _upload_r2_file(s3, bucket: str, local_path: Path, s3_key: str, dry_run: bool) -> bool:
    size_mb = local_path.stat().st_size / 1e6
    print(f"    {'[DRY RUN] ' if dry_run else ''}{local_path.name} ({size_mb:.1f} MB) -> {s3_key}")
    if dry_run:
        return True
    try:
        s3.upload_file(
            str(local_path), bucket, s3_key,
            ExtraArgs={
                "ContentType": "application/octet-stream",
                "Metadata": {
                    "md5": _md5_file(local_path),
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                },
            },
        )
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def _write_manifest(s3, bucket: str, prefix: str, version: str, models_uploaded: dict, dry_run: bool):
    manifest = {
        "version": version,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "models": models_uploaded,
    }
    manifest_json = json.dumps(manifest, indent=2)

    for key in [f"{prefix}/manifests/v{version}.json", f"{prefix}/manifests/latest.json"]:
        print(f"    {'[DRY RUN] ' if dry_run else ''}manifest -> s3://{bucket}/{key}")
        if not dry_run:
            s3.put_object(Bucket=bucket, Key=key, Body=manifest_json.encode(), ContentType="application/json")


def upload_to_r2(models_to_upload: list[str], version: str, dry_run: bool,
                 bucket: str = None, prefix: str = None, endpoint_url: str = None):
    print("\n" + "=" * 60)
    print("CLOUDFLARE R2")
    print("=" * 60)

    bucket = bucket or os.environ.get("S3_BUCKET")
    prefix = prefix or os.environ.get("S3_PREFIX", "models")
    endpoint_url = endpoint_url or os.environ.get("S3_ENDPOINT_URL")
    access_key = os.environ.get("S3_ACCESS_KEY_ID")
    secret_key = os.environ.get("S3_SECRET_ACCESS_KEY")

    if not bucket:
        print("  SKIPPED: S3_BUCKET not set")
        return False
    if not endpoint_url:
        print("  SKIPPED: S3_ENDPOINT_URL not set")
        return False

    print(f"  Bucket: s3://{bucket}/{prefix}/")
    print(f"  Version: v{version}")

    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        print("  ERROR: boto3 not installed. Run: uv pip install boto3")
        return False

    client_kwargs = {"endpoint_url": endpoint_url}
    if access_key and secret_key:
        client_kwargs["aws_access_key_id"] = access_key
        client_kwargs["aws_secret_access_key"] = secret_key

    s3 = boto3.client("s3", region_name="auto", **client_kwargs)

    if not dry_run:
        try:
            s3.head_bucket(Bucket=bucket)
        except ClientError as e:
            print(f"  ERROR: Cannot access bucket: {e}")
            return False

    models_uploaded = {}
    for key in models_to_upload:
        files = collect_files(key)
        if not files:
            print(f"  [{key}] No files - skipping")
            continue

        folder = MODELS[key]["hf_repo"].split("/")[1]
        s3_base = f"{prefix}/{folder}/v{version}"
        print(f"  [{key}] -> s3://{bucket}/{s3_base}/ ({len(files)} files)")

        for f in files:
            _upload_r2_file(s3, bucket, f, f"{s3_base}/{f.name}", dry_run)

        models_uploaded[key] = {
            "s3_path": f"{s3_base}/",
            "files": [f.name for f in files],
            "total_size_mb": round(sum(f.stat().st_size for f in files) / 1e6, 1),
        }

    if models_uploaded:
        _write_manifest(s3, bucket, prefix, version, models_uploaded, dry_run)

    return True


# ── Model cards ──────────────────────────────────────────────────────────────

def _load_cards() -> dict:
    """Load model cards from upload_to_hf.py (keeps cards in one place)."""
    hf_path = Path(__file__).parent / "upload_to_hf.py"
    if not hf_path.exists():
        return {}
    spec = importlib.util.spec_from_file_location("hf_cards", str(hf_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {
        "full": getattr(mod, "FULL_MODEL_CARD", ""),
        "gguf": getattr(mod, "GGUF_MODEL_CARD", ""),
        "mobile": getattr(mod, "MOBILE_MODEL_CARD", ""),
        "edge-2b": getattr(mod, "EDGE_2B_MODEL_CARD", ""),
        "edge-4b": getattr(mod, "EDGE_4B_MODEL_CARD", ""),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Upload Horizon models to HuggingFace and Cloudflare R2"
    )
    parser.add_argument(
        "--what", nargs="+", required=True,
        choices=["full", "gguf", "mobile", "edge-2b", "edge-4b", "all"],
    )
    parser.add_argument("--version", required=True, help="Semantic version (e.g. 2.1.0)")
    parser.add_argument("--skip-hf", action="store_true", help="Skip HuggingFace upload")
    parser.add_argument("--skip-r2", action="store_true", help="Skip Cloudflare R2 upload")
    parser.add_argument("--dry-run", action="store_true", help="Preview without uploading")
    parser.add_argument("--bucket", default=None, help="Override S3_BUCKET env")
    parser.add_argument("--endpoint-url", default=None, help="Override S3_ENDPOINT_URL env")
    parser.add_argument("--prefix", default=None, help="Override S3_PREFIX env")
    args = parser.parse_args()

    models = list(MODELS.keys()) if "all" in args.what else args.what

    print(f"Models:  {', '.join(models)}")
    print(f"Version: v{args.version}")
    if args.dry_run:
        print("Mode:    DRY RUN")

    cards = _load_cards()

    if not args.skip_hf:
        upload_to_hf(models, cards, args.dry_run)

    if not args.skip_r2:
        upload_to_r2(
            models, args.version, args.dry_run,
            bucket=args.bucket, prefix=args.prefix, endpoint_url=args.endpoint_url,
        )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
