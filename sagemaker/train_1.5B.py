#!/usr/bin/env python3
"""
Complete Training Pipeline for vAGI 1.5B Model on AWS SageMaker

Usage:
    python train_1.5B.py --s3-bucket YOUR_BUCKET --role YOUR_ROLE_ARN

This script will:
1. Download training data from Hugging Face
2. Prepare and tokenize data
3. Upload to S3
4. Launch SageMaker training job with spot instances

Estimated cost: $500-1500 (with spot instances, 48-72h)
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import boto3


def check_dependencies():
    """Check if required packages are installed."""
    try:
        import datasets
        import transformers
        import sagemaker
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Installing required packages...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "datasets", "transformers", "sagemaker", "boto3"
        ], check=True)


def download_training_data(output_dir: str, num_samples: int = 500000):
    """Download training data from Hugging Face."""
    from datasets import load_dataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {num_samples} samples from public datasets...")

    all_texts = []

    # 1. OpenWebText (high-quality web text)
    print("Loading OpenWebText...")
    try:
        owt = load_dataset("openwebtext", split=f"train[:{num_samples//2}]")
        all_texts.extend([{"text": item["text"]} for item in owt])
        print(f"  Added {len(owt)} OpenWebText samples")
    except Exception as e:
        print(f"  Warning: Could not load OpenWebText: {e}")

    # 2. Wikipedia
    print("Loading Wikipedia...")
    try:
        wiki = load_dataset("wikipedia", "20220301.en", split=f"train[:{num_samples//4}]")
        all_texts.extend([{"text": item["text"]} for item in wiki])
        print(f"  Added {len(wiki)} Wikipedia samples")
    except Exception as e:
        print(f"  Warning: Could not load Wikipedia: {e}")

    # 3. Alpaca instructions
    print("Loading Alpaca instructions...")
    try:
        alpaca = load_dataset("tatsu-lab/alpaca", split="train")
        for item in alpaca:
            text = f"### Instruction:\n{item['instruction']}\n"
            if item.get("input"):
                text += f"\n### Input:\n{item['input']}\n"
            text += f"\n### Response:\n{item['output']}"
            all_texts.append({"text": text})
        print(f"  Added {len(alpaca)} Alpaca samples")
    except Exception as e:
        print(f"  Warning: Could not load Alpaca: {e}")

    # 4. CodeAlpaca
    print("Loading CodeAlpaca...")
    try:
        code = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        for item in code:
            text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
            all_texts.append({"text": text})
        print(f"  Added {len(code)} CodeAlpaca samples")
    except Exception as e:
        print(f"  Warning: Could not load CodeAlpaca: {e}")

    # 5. Math instructions
    print("Loading Math data...")
    try:
        math = load_dataset("gsm8k", "main", split="train")
        for item in math:
            text = f"### Question:\n{item['question']}\n\n### Answer:\n{item['answer']}"
            all_texts.append({"text": text})
        print(f"  Added {len(math)} Math samples")
    except Exception as e:
        print(f"  Warning: Could not load Math data: {e}")

    # Save to JSONL
    output_file = output_path / "raw_data.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in all_texts:
            f.write(json.dumps(item) + "\n")

    print(f"\nTotal samples: {len(all_texts)}")
    print(f"Saved to: {output_file}")

    return str(output_file)


def prepare_training_data(
    raw_file: str,
    output_dir: str,
    max_seq_len: int = 2048,
    obs_dim: int = 1024,
):
    """Tokenize and prepare data for training."""
    from transformers import AutoTokenizer
    import random

    print(f"\nPreparing training data...")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "train.jsonl"

    count = 0
    with open(raw_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            try:
                data = json.loads(line.strip())
                text = data.get("text", "")
                if not text:
                    continue

                # Tokenize
                tokens = tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_seq_len
                )

                if len(tokens) < 10:
                    continue

                sample = {
                    "input_ids": tokens[:-1],
                    "labels": tokens[1:],
                    "obs": [random.gauss(0, 1) for _ in range(obs_dim)],
                    "action": 0,
                    "reward": 0.0,
                }

                fout.write(json.dumps(sample) + "\n")
                count += 1

                if count % 10000 == 0:
                    print(f"  Processed {count} samples...")

            except Exception:
                continue

    print(f"Prepared {count} training samples")
    print(f"Saved to: {output_file}")

    return str(output_file)


def upload_to_s3(local_dir: str, s3_bucket: str, s3_prefix: str):
    """Upload data to S3."""
    print(f"\nUploading to s3://{s3_bucket}/{s3_prefix}/...")

    s3 = boto3.client("s3")
    local_path = Path(local_dir)

    count = 0
    for file_path in local_path.glob("**/*"):
        if file_path.is_file():
            s3_key = f"{s3_prefix}/{file_path.relative_to(local_path)}"
            print(f"  Uploading {file_path.name}...")
            s3.upload_file(str(file_path), s3_bucket, s3_key)
            count += 1

    print(f"Uploaded {count} files")
    return f"s3://{s3_bucket}/{s3_prefix}"


def create_training_job(
    s3_bucket: str,
    s3_data_uri: str,
    role_arn: str,
    region: str = "us-east-1",
):
    """Create SageMaker training job for 1.5B model."""
    import sagemaker
    from sagemaker.pytorch import PyTorch

    print(f"\nCreating SageMaker training job...")

    session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    job_name = f"vagi-1-5B-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # 1.5B model configuration
    # - 2x ml.p4d.24xlarge (8x A100 each = 640GB total VRAM)
    # - Or 4x ml.g5.48xlarge (8x A10G each = 768GB total VRAM)

    estimator = PyTorch(
        entry_point="train_sagemaker.py",
        source_dir=str(Path(__file__).parent),
        role=role_arn,
        instance_type="ml.p4d.24xlarge",
        instance_count=2,
        framework_version="2.1.0",
        py_version="py310",
        hyperparameters={
            "model-size": "xlarge",  # Will use 1.5B config
            "epochs": 50,
            "batch-size": 2,
            "lr": 5e-5,
            "fp16": "",
            "warmup-steps": 2000,
            "max-grad-norm": 1.0,
        },
        output_path=f"s3://{s3_bucket}/vagi/output",
        code_location=f"s3://{s3_bucket}/vagi/code",
        job_name=job_name,

        # Use spot instances for 60-90% cost savings
        use_spot_instances=True,
        max_wait=72 * 3600,  # 72 hours max wait
        max_run=72 * 3600,   # 72 hours max run

        # Checkpointing for spot interruption recovery
        checkpoint_s3_uri=f"s3://{s3_bucket}/vagi/checkpoints/{job_name}",
        checkpoint_local_path="/opt/ml/checkpoints",

        # Performance settings
        debugger_hook_config=False,
        environment={
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
            "NCCL_DEBUG": "INFO",
        },

        # Distributed training
        distribution={
            "torch_distributed": {"enabled": True}
        },

        tags=[
            {"Key": "Project", "Value": "vAGI"},
            {"Key": "ModelSize", "Value": "1.5B"},
        ],
    )

    print(f"\nJob Configuration:")
    print(f"  Name: {job_name}")
    print(f"  Instances: 2 x ml.p4d.24xlarge (16x A100, 640GB VRAM)")
    print(f"  Spot: Enabled (60-90% savings)")
    print(f"  Max Runtime: 72 hours")
    print(f"  Estimated Cost: $500-1500")
    print()

    confirm = input("Proceed with training? [y/N]: ")
    if confirm.lower() != "y":
        print("Aborted.")
        return None

    # Launch training
    estimator.fit(
        inputs={"train": s3_data_uri},
        wait=False,
        logs=False,
    )

    print(f"\n{'='*60}")
    print(f"Training job submitted successfully!")
    print(f"{'='*60}")
    print(f"Job Name: {job_name}")
    print()
    print(f"Monitor progress:")
    print(f"  Console: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}")
    print()
    print(f"Check status:")
    print(f"  aws sagemaker describe-training-job --training-job-name {job_name}")
    print()
    print(f"Download model when complete:")
    print(f"  aws s3 cp s3://{s3_bucket}/vagi/output/{job_name}/output/model.tar.gz ./")
    print(f"{'='*60}")

    return job_name


def main():
    parser = argparse.ArgumentParser(description="Train vAGI 1.5B on SageMaker")

    parser.add_argument("--s3-bucket", type=str, required=True,
                        help="S3 bucket name")
    parser.add_argument("--role", type=str, required=True,
                        help="SageMaker IAM role ARN")
    parser.add_argument("--region", type=str, default="us-east-1",
                        help="AWS region")
    parser.add_argument("--num-samples", type=int, default=500000,
                        help="Number of training samples to download")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download (use existing data)")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip data preparation")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip S3 upload")

    args = parser.parse_args()

    print("="*60)
    print("vAGI 1.5B Training Pipeline")
    print("="*60)
    print(f"S3 Bucket: {args.s3_bucket}")
    print(f"IAM Role: {args.role}")
    print(f"Region: {args.region}")
    print("="*60)

    # Check dependencies
    check_dependencies()

    # Setup paths
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    prepared_dir = data_dir / "sagemaker"

    # Step 1: Download data
    if not args.skip_download:
        raw_file = download_training_data(str(raw_dir), args.num_samples)
    else:
        raw_file = str(raw_dir / "raw_data.jsonl")
        print(f"\nSkipping download, using: {raw_file}")

    # Step 2: Prepare data
    if not args.skip_prepare:
        prepare_training_data(raw_file, str(prepared_dir))
    else:
        print(f"\nSkipping preparation, using: {prepared_dir}")

    # Step 3: Upload to S3
    if not args.skip_upload:
        s3_data_uri = upload_to_s3(
            str(prepared_dir),
            args.s3_bucket,
            "vagi/data"
        )
    else:
        s3_data_uri = f"s3://{args.s3_bucket}/vagi/data"
        print(f"\nSkipping upload, using: {s3_data_uri}")

    # Step 4: Launch training
    create_training_job(
        s3_bucket=args.s3_bucket,
        s3_data_uri=s3_data_uri,
        role_arn=args.role,
        region=args.region,
    )


if __name__ == "__main__":
    main()
