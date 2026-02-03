#!/usr/bin/env python3
"""Launch vAGI training job on AWS SageMaker.

Usage:
    python launch_training.py --model-size small --instance-type ml.g5.xlarge
    python launch_training.py --model-size medium --instance-type ml.g5.2xlarge
    python launch_training.py --model-size large --instance-type ml.p4d.24xlarge --instance-count 2
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch


# =============================================================================
# Instance Type Recommendations
# =============================================================================

INSTANCE_RECOMMENDATIONS = {
    "tiny": {
        "instance_type": "ml.g5.xlarge",  # 1x A10G 24GB
        "instance_count": 1,
        "batch_size": 32,
        "estimated_cost_per_hour": 1.006,
    },
    "small": {
        "instance_type": "ml.g5.2xlarge",  # 1x A10G 24GB + more CPU/RAM
        "instance_count": 1,
        "batch_size": 16,
        "estimated_cost_per_hour": 1.515,
    },
    "medium": {
        "instance_type": "ml.g5.4xlarge",  # 1x A10G 24GB + even more resources
        "instance_count": 1,
        "batch_size": 8,
        "estimated_cost_per_hour": 2.534,
    },
    "large": {
        "instance_type": "ml.g5.12xlarge",  # 4x A10G 96GB total
        "instance_count": 1,
        "batch_size": 4,
        "estimated_cost_per_hour": 7.090,
    },
    "xlarge": {
        "instance_type": "ml.p4d.24xlarge",  # 8x A100 320GB total
        "instance_count": 2,
        "batch_size": 2,
        "estimated_cost_per_hour": 32.77,
    },
}

# Alternative cheaper instances
BUDGET_INSTANCES = {
    "tiny": "ml.g4dn.xlarge",      # T4 16GB - $0.526/hr
    "small": "ml.g4dn.2xlarge",    # T4 16GB - $0.752/hr
    "medium": "ml.g4dn.8xlarge",   # T4 16GB - $2.176/hr
    "large": "ml.g5.8xlarge",      # A10G 24GB - $4.568/hr
    "xlarge": "ml.g5.48xlarge",    # 8x A10G 192GB - $20.27/hr
}


def upload_data_to_s3(
    local_data_dir: str,
    s3_bucket: str,
    s3_prefix: str = "vagi/data",
) -> str:
    """Upload training data to S3."""
    s3 = boto3.client('s3')

    local_path = Path(local_data_dir)
    uploaded_files = 0

    for file_path in local_path.glob("**/*"):
        if file_path.is_file():
            s3_key = f"{s3_prefix}/{file_path.relative_to(local_path)}"
            print(f"Uploading {file_path} to s3://{s3_bucket}/{s3_key}")
            s3.upload_file(str(file_path), s3_bucket, s3_key)
            uploaded_files += 1

    print(f"Uploaded {uploaded_files} files to S3")
    return f"s3://{s3_bucket}/{s3_prefix}"


def upload_code_to_s3(
    s3_bucket: str,
    s3_prefix: str = "vagi/code",
) -> str:
    """Upload source code to S3."""
    import tarfile
    import tempfile

    # Create tarball of source code
    code_dir = Path(__file__).parent.parent

    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
        with tarfile.open(tmp.name, 'w:gz') as tar:
            # Add core module
            tar.add(code_dir / 'core', arcname='core')
            # Add sagemaker training script
            tar.add(code_dir / 'sagemaker', arcname='sagemaker')
            # Add pyproject.toml
            tar.add(code_dir / 'pyproject.toml', arcname='pyproject.toml')

        # Upload to S3
        s3 = boto3.client('s3')
        s3_key = f"{s3_prefix}/source.tar.gz"
        s3.upload_file(tmp.name, s3_bucket, s3_key)

        print(f"Uploaded code to s3://{s3_bucket}/{s3_key}")

    os.unlink(tmp.name)
    return f"s3://{s3_bucket}/{s3_prefix}"


def create_training_job(args):
    """Create and launch SageMaker training job."""

    # Get SageMaker session
    session = sagemaker.Session()
    role = args.role or sagemaker.get_execution_role()

    # Get recommended settings
    rec = INSTANCE_RECOMMENDATIONS.get(args.model_size, INSTANCE_RECOMMENDATIONS["small"])

    instance_type = args.instance_type or rec["instance_type"]
    instance_count = args.instance_count or rec["instance_count"]
    batch_size = args.batch_size or rec["batch_size"]

    # Use budget instance if requested
    if args.budget:
        instance_type = BUDGET_INSTANCES.get(args.model_size, instance_type)
        print(f"Using budget instance: {instance_type}")

    # Estimate cost
    cost_per_hour = rec["estimated_cost_per_hour"] * instance_count
    estimated_hours = args.epochs * 2  # rough estimate
    estimated_cost = cost_per_hour * estimated_hours

    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Model Size: {args.model_size}")
    print(f"Instance Type: {instance_type}")
    print(f"Instance Count: {instance_count}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Estimated Cost: ${estimated_cost:.2f} ({estimated_hours}h @ ${cost_per_hour:.2f}/hr)")
    print(f"{'='*60}\n")

    if not args.yes:
        confirm = input("Proceed with training? [y/N]: ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return

    # Upload data if local path provided
    if args.data_dir and not args.data_dir.startswith("s3://"):
        s3_data_uri = upload_data_to_s3(
            args.data_dir,
            args.s3_bucket,
            f"vagi/data/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
    else:
        s3_data_uri = args.data_dir or f"s3://{args.s3_bucket}/vagi/data"

    # Job name
    job_name = f"vagi-{args.model_size}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Create estimator
    estimator = PyTorch(
        entry_point="train_sagemaker.py",
        source_dir=str(Path(__file__).parent),
        role=role,
        instance_type=instance_type,
        instance_count=instance_count,
        framework_version="2.1.0",
        py_version="py310",
        hyperparameters={
            "model-size": args.model_size,
            "epochs": args.epochs,
            "batch-size": batch_size,
            "lr": args.lr,
            "fp16": "",  # Flag to enable
        },
        output_path=f"s3://{args.s3_bucket}/vagi/output",
        code_location=f"s3://{args.s3_bucket}/vagi/code",
        job_name=job_name,

        # Enable spot instances for cost savings (up to 90% off)
        use_spot_instances=args.spot,
        max_wait=args.max_wait * 3600 if args.spot else None,  # Max wait time for spot
        max_run=args.max_run * 3600,  # Max training time

        # Checkpointing for spot interruption recovery
        checkpoint_s3_uri=f"s3://{args.s3_bucket}/vagi/checkpoints/{job_name}" if args.spot else None,
        checkpoint_local_path="/opt/ml/checkpoints" if args.spot else None,

        # Debugger and profiler
        debugger_hook_config=False,  # Disable for performance

        # Tags for cost tracking
        tags=[
            {"Key": "Project", "Value": "vAGI"},
            {"Key": "ModelSize", "Value": args.model_size},
        ],

        # Environment variables
        environment={
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        },

        # Distribution for multi-GPU/multi-node
        distribution={
            "torch_distributed": {
                "enabled": True
            }
        } if instance_count > 1 or "p4d" in instance_type or "g5.12" in instance_type else None,
    )

    # Start training
    print(f"Starting training job: {job_name}")
    estimator.fit(
        inputs={"train": s3_data_uri},
        wait=args.wait,
        logs="All" if args.wait else None,
    )

    if not args.wait:
        print(f"\nTraining job submitted: {job_name}")
        print(f"Monitor at: https://console.aws.amazon.com/sagemaker/home#/jobs/{job_name}")

    return job_name


def main():
    parser = argparse.ArgumentParser(description="Launch vAGI training on SageMaker")

    # Required
    parser.add_argument("--s3-bucket", type=str, required=True,
                        help="S3 bucket for data and outputs")

    # Model configuration
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["tiny", "small", "medium", "large", "xlarge"])

    # Instance configuration
    parser.add_argument("--instance-type", type=str, default=None,
                        help="Override instance type")
    parser.add_argument("--instance-count", type=int, default=None,
                        help="Number of instances")
    parser.add_argument("--budget", action="store_true",
                        help="Use budget instances (slower but cheaper)")
    parser.add_argument("--spot", action="store_true",
                        help="Use spot instances (up to 90% cheaper)")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Data
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Local data directory to upload, or S3 URI")

    # SageMaker settings
    parser.add_argument("--role", type=str, default=None,
                        help="SageMaker execution role ARN")
    parser.add_argument("--max-run", type=int, default=24,
                        help="Maximum training hours")
    parser.add_argument("--max-wait", type=int, default=48,
                        help="Maximum wait time for spot (hours)")

    # Control
    parser.add_argument("--wait", action="store_true",
                        help="Wait for training to complete")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation")

    args = parser.parse_args()

    create_training_job(args)


if __name__ == "__main__":
    main()
