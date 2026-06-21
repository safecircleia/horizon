#!/usr/bin/env python3
"""Upload all Horizon model artifacts to S3 with versioned folder structure.

Folder layout in bucket:
    s3://{BUCKET}/
    ├── horizon-full/
    │   └── v{VERSION}/
    │       ├── model.safetensors
    │       └── ...
    ├── horizon-full-gguf/
    │   └── v{VERSION}/
    │       ├── horizon-full-Q4_K_M.gguf
    │       └── ...
    ├── horizon-mobile/
    │   └── v{VERSION}/
    │       ├── horizon-mobile-int8.litertlm
    │       └── ...
    ├── horizon-edge-2b/
    │   └── v{VERSION}/
    │       ├── horizon-edge-e2b.litertlm
    │       ├── horizon-edge-e2b_Google_Tensor_G5.litertlm
    │       └── ...
    └── horizon-edge-4b/
        └── v{VERSION}/
            ├── horizon-edge-e4b.litertlm
            └── ...

Usage:
    python scripts/upload_to_s3.py --what all --version 2.1.0
    python scripts/upload_to_s3.py --what edge-2b --version 2.1.0
    python scripts/upload_to_s3.py --what full gguf --version 2.0.0
    python scripts/upload_to_s3.py --what all --version 2.1.0 --dry-run

Environment:
    S3_BUCKET              Bucket name (required)
    S3_PREFIX              Optional prefix under bucket (default: "models")
    S3_ENDPOINT_URL        S3-compatible endpoint (e.g. https://<id>.r2.cloudflarestorage.com)
    S3_ACCESS_KEY_ID       Access key ID (Cloudflare R2 or AWS)
    S3_SECRET_ACCESS_KEY   Secret access key

Prerequisites:
    uv pip install boto3
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("ERROR: boto3 not installed. Run: uv pip install boto3")
    sys.exit(1)


MODEL_CONFIGS = {
    "full": {
        "folder": "horizon-full",
        "local_dir": "models/horizon-full-merged",
        "patterns": ["*.safetensors", "*.json", "*.jinja", "*.model"],
    },
    "gguf": {
        "folder": "horizon-full-gguf",
        "local_dir": "models/horizon-full-gguf",
        "patterns": ["*.gguf"],
    },
    "mobile": {
        "folder": "horizon-mobile",
        "local_dir": "models/mobile-standard",
        "patterns": ["*.litertlm"],
        "extra_dirs": ["models/mobile-lite"],
    },
    "edge-2b": {
        "folder": "horizon-edge-2b",
        "local_dir": "models/horizon-edge-2b-litert",
        "patterns": ["*.litertlm"],
    },
    "edge-4b": {
        "folder": "horizon-edge-4b",
        "local_dir": "models/horizon-edge-4b-litert",
        "patterns": ["*.litertlm"],
    },
}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_files(local_dir: str, patterns: list[str]) -> list[Path]:
    """Collect files matching glob patterns from a directory."""
    p = Path(local_dir)
    if not p.exists():
        return []
    files = []
    for pattern in patterns:
        files.extend(sorted(p.glob(pattern)))
    return files


def upload_file(s3, bucket: str, local_path: Path, s3_key: str, dry_run: bool) -> bool:
    size_mb = local_path.stat().st_size / 1e6
    print(f"  {'[DRY RUN] ' if dry_run else ''}Uploading {local_path.name} ({size_mb:.1f} MB) → s3://{bucket}/{s3_key}")

    if dry_run:
        return True

    try:
        s3.upload_file(
            str(local_path),
            bucket,
            s3_key,
            ExtraArgs={
                "ContentType": _content_type(local_path),
                "Metadata": {
                    "md5": md5_file(local_path),
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                },
            },
        )
        return True
    except ClientError as e:
        print(f"  ERROR: {e}")
        return False


def _content_type(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".safetensors": "application/octet-stream",
        ".gguf": "application/octet-stream",
        ".litertlm": "application/octet-stream",
        ".json": "application/json",
        ".jinja": "text/plain",
        ".model": "application/octet-stream",
    }.get(ext, "application/octet-stream")


def upload_model(s3, bucket: str, prefix: str, model_key: str, version: str, dry_run: bool) -> dict:
    config = MODEL_CONFIGS[model_key]
    folder = config["folder"]
    s3_base = f"{prefix}/{folder}/v{version}"

    dirs = [config["local_dir"]] + config.get("extra_dirs", [])
    files = []
    for d in dirs:
        files.extend(collect_files(d, config["patterns"]))

    if not files:
        print(f"\n[{model_key}] No files found in {dirs}")
        return {"model": model_key, "uploaded": 0, "failed": 0, "skipped": True}

    print(f"\n{'='*60}")
    print(f"[{model_key}] Uploading {len(files)} files → s3://{bucket}/{s3_base}/")
    print(f"{'='*60}")

    uploaded, failed = 0, 0
    for f in files:
        s3_key = f"{s3_base}/{f.name}"
        if upload_file(s3, bucket, f, s3_key, dry_run):
            uploaded += 1
        else:
            failed += 1

    return {"model": model_key, "uploaded": uploaded, "failed": failed, "skipped": False}


def write_manifest(s3, bucket: str, prefix: str, version: str, results: list[dict], dry_run: bool):
    """Write a version manifest JSON to S3 for tracking deployed versions."""
    manifest = {
        "version": version,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "models": {},
    }

    for r in results:
        if r["skipped"]:
            continue
        model_key = r["model"]
        config = MODEL_CONFIGS[model_key]
        folder = config["folder"]
        dirs = [config["local_dir"]] + config.get("extra_dirs", [])
        files = []
        for d in dirs:
            files.extend(collect_files(d, config["patterns"]))

        manifest["models"][model_key] = {
            "s3_path": f"{prefix}/{folder}/v{version}/",
            "files": [f.name for f in files],
            "total_size_mb": round(sum(f.stat().st_size for f in files) / 1e6, 1),
        }

    manifest_key = f"{prefix}/manifests/v{version}.json"
    manifest_json = json.dumps(manifest, indent=2)
    print(f"\n  {'[DRY RUN] ' if dry_run else ''}Writing manifest → s3://{bucket}/{manifest_key}")

    if not dry_run:
        s3.put_object(
            Bucket=bucket,
            Key=manifest_key,
            Body=manifest_json.encode(),
            ContentType="application/json",
        )

    latest_key = f"{prefix}/manifests/latest.json"
    print(f"  {'[DRY RUN] ' if dry_run else ''}Updating latest → s3://{bucket}/{latest_key}")
    if not dry_run:
        s3.put_object(
            Bucket=bucket,
            Key=latest_key,
            Body=manifest_json.encode(),
            ContentType="application/json",
        )


def main():
    parser = argparse.ArgumentParser(description="Upload Horizon models to S3")
    parser.add_argument(
        "--what", nargs="+", required=True,
        choices=["full", "gguf", "mobile", "edge-2b", "edge-4b", "all"],
        help="Which models to upload",
    )
    parser.add_argument("--version", required=True, help="Semantic version (e.g. 2.1.0)")
    parser.add_argument("--bucket", default=None, help="S3 bucket (overrides S3_BUCKET env)")
    parser.add_argument("--prefix", default=None, help="S3 prefix (overrides S3_PREFIX env, default: 'models')")
    parser.add_argument("--endpoint-url", default=None, help="S3-compatible endpoint URL (overrides S3_ENDPOINT_URL env)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded without uploading")

    # Override local dirs
    parser.add_argument("--full-dir", default=None)
    parser.add_argument("--gguf-dir", default=None)
    parser.add_argument("--mobile-dir", default=None)
    parser.add_argument("--edge-2b-dir", default=None)
    parser.add_argument("--edge-4b-dir", default=None)
    args = parser.parse_args()

    bucket = args.bucket or os.environ.get("S3_BUCKET")
    if not bucket:
        print("ERROR: S3 bucket required. Set S3_BUCKET env or pass --bucket")
        sys.exit(1)

    prefix = args.prefix or os.environ.get("S3_PREFIX", "models")
    endpoint_url = args.endpoint_url or os.environ.get("S3_ENDPOINT_URL")
    access_key = os.environ.get("S3_ACCESS_KEY_ID")
    secret_key = os.environ.get("S3_SECRET_ACCESS_KEY")

    # Apply dir overrides
    dir_overrides = {
        "full": args.full_dir,
        "gguf": args.gguf_dir,
        "mobile": args.mobile_dir,
        "edge-2b": args.edge_2b_dir,
        "edge-4b": args.edge_4b_dir,
    }
    for key, override in dir_overrides.items():
        if override:
            MODEL_CONFIGS[key]["local_dir"] = override

    models_to_upload = list(MODEL_CONFIGS.keys()) if "all" in args.what else args.what

    print(f"Bucket:   s3://{bucket}/{prefix}/")
    print(f"Endpoint: {endpoint_url or 'default (AWS)'}")
    print(f"Version:  v{args.version}")
    print(f"Models:   {', '.join(models_to_upload)}")
    if args.dry_run:
        print("Mode:     DRY RUN")

    client_kwargs = {}
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
    if access_key and secret_key:
        client_kwargs["aws_access_key_id"] = access_key
        client_kwargs["aws_secret_access_key"] = secret_key

    s3 = boto3.client("s3", region_name="auto", **client_kwargs)

    # Verify bucket access
    if not args.dry_run:
        try:
            s3.head_bucket(Bucket=bucket)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "404":
                print(f"ERROR: Bucket '{bucket}' does not exist")
            elif code == "403":
                print(f"ERROR: Access denied to bucket '{bucket}'")
            else:
                print(f"ERROR: {e}")
            sys.exit(1)

    results = []
    for model_key in models_to_upload:
        result = upload_model(s3, bucket, prefix, model_key, args.version, args.dry_run)
        results.append(result)

    write_manifest(s3, bucket, prefix, args.version, results, args.dry_run)

    # Summary
    print(f"\n{'='*60}")
    print("Upload Summary")
    print(f"{'='*60}")
    total_uploaded, total_failed = 0, 0
    for r in results:
        if r["skipped"]:
            print(f"  [{r['model']}] SKIPPED (no files found)")
        else:
            status = "OK" if r["failed"] == 0 else "PARTIAL"
            print(f"  [{r['model']}] {status} — {r['uploaded']} uploaded, {r['failed']} failed")
            total_uploaded += r["uploaded"]
            total_failed += r["failed"]

    print(f"\nTotal: {total_uploaded} uploaded, {total_failed} failed")
    if total_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
