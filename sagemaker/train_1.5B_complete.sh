#!/bin/bash
# =============================================================================
# Complete Training Script for vAGI 1.5B Model on AWS SageMaker
# =============================================================================
#
# Prerequisites:
# 1. AWS CLI configured: aws configure
# 2. S3 bucket created
# 3. SageMaker IAM role created
#
# Usage:
#   chmod +x train_1.5B_complete.sh
#   ./train_1.5B_complete.sh YOUR_S3_BUCKET YOUR_IAM_ROLE_ARN
#
# Estimated cost: ~$500-1500 with spot instances (48-72h training)
# =============================================================================

set -e

# Configuration
S3_BUCKET="${1:-vagi-training}"
IAM_ROLE="${2:-arn:aws:iam::YOUR_ACCOUNT:role/SageMakerVAGIRole}"
REGION="${3:-us-east-1}"
MODEL_SIZE="1.5B"

echo "=============================================="
echo "vAGI 1.5B Training Pipeline"
echo "=============================================="
echo "S3 Bucket: $S3_BUCKET"
echo "IAM Role: $IAM_ROLE"
echo "Region: $REGION"
echo "=============================================="

# Step 1: Create S3 bucket if not exists
echo ""
echo "[Step 1/6] Creating S3 bucket..."
aws s3 mb s3://$S3_BUCKET --region $REGION 2>/dev/null || echo "Bucket already exists"

# Step 2: Download public training data
echo ""
echo "[Step 2/6] Downloading training data..."

mkdir -p data/raw
cd data/raw

# Option A: Use Hugging Face datasets (recommended)
echo "Downloading from Hugging Face..."

# Download OpenWebText (cleaned web text)
python3 << 'PYTHON'
from datasets import load_dataset
import json

print("Loading OpenWebText subset...")
dataset = load_dataset("openwebtext", split="train[:100000]")  # 100k samples

print("Saving to JSONL...")
with open("openwebtext_100k.jsonl", "w") as f:
    for item in dataset:
        f.write(json.dumps({"text": item["text"]}) + "\n")

print(f"Saved {len(dataset)} samples")
PYTHON

# Download additional instruction data
python3 << 'PYTHON'
from datasets import load_dataset
import json

print("Loading instruction data...")

# Alpaca instructions
try:
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    with open("alpaca.jsonl", "w") as f:
        for item in alpaca:
            text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
            f.write(json.dumps({"text": text}) + "\n")
    print(f"Saved {len(alpaca)} Alpaca samples")
except:
    print("Could not load Alpaca dataset")

# Code instructions
try:
    code = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    with open("code_alpaca.jsonl", "w") as f:
        for item in code:
            text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
            f.write(json.dumps({"text": text}) + "\n")
    print(f"Saved {len(code)} Code samples")
except:
    print("Could not load CodeAlpaca dataset")
PYTHON

cd ../..

# Step 3: Prepare training data
echo ""
echo "[Step 3/6] Preparing training data..."

mkdir -p data/sagemaker

# Merge and prepare all data
python3 sagemaker/prepare_data.py \
    --format text \
    --input data/raw/openwebtext_100k.jsonl \
    --output data/sagemaker/train_web.jsonl \
    --max-seq-len 2048 \
    --obs-dim 1024

# Add instruction data if available
if [ -f data/raw/alpaca.jsonl ]; then
    python3 sagemaker/prepare_data.py \
        --format text \
        --input data/raw/alpaca.jsonl \
        --output data/sagemaker/train_alpaca.jsonl \
        --max-seq-len 2048 \
        --obs-dim 1024
fi

if [ -f data/raw/code_alpaca.jsonl ]; then
    python3 sagemaker/prepare_data.py \
        --format text \
        --input data/raw/code_alpaca.jsonl \
        --output data/sagemaker/train_code.jsonl \
        --max-seq-len 2048 \
        --obs-dim 1024
fi

# Combine all training data
cat data/sagemaker/train_*.jsonl > data/sagemaker/train_combined.jsonl
echo "Combined training samples: $(wc -l < data/sagemaker/train_combined.jsonl)"

# Step 4: Upload data to S3
echo ""
echo "[Step 4/6] Uploading data to S3..."
aws s3 sync data/sagemaker/ s3://$S3_BUCKET/vagi/data/ --exclude "*.tmp"
echo "Data uploaded to s3://$S3_BUCKET/vagi/data/"

# Step 5: Upload source code to S3
echo ""
echo "[Step 5/6] Uploading source code..."
tar -czf /tmp/vagi_source.tar.gz core/ sagemaker/ pyproject.toml
aws s3 cp /tmp/vagi_source.tar.gz s3://$S3_BUCKET/vagi/code/source.tar.gz
rm /tmp/vagi_source.tar.gz

# Step 6: Launch SageMaker Training Job
echo ""
echo "[Step 6/6] Launching SageMaker training job..."

JOB_NAME="vagi-1-5B-$(date +%Y%m%d-%H%M%S)"
INSTANCE_TYPE="ml.p4d.24xlarge"  # 8x A100 40GB = 320GB VRAM
INSTANCE_COUNT=2  # 2 nodes = 640GB VRAM total

# Create training job using AWS CLI
aws sagemaker create-training-job \
    --training-job-name "$JOB_NAME" \
    --role-arn "$IAM_ROLE" \
    --algorithm-specification '{
        "TrainingImage": "763104351884.dkr.ecr.'$REGION'.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker",
        "TrainingInputMode": "File",
        "EnableSageMakerMetricsTimeSeries": true
    }' \
    --hyper-parameters '{
        "model-size": "xlarge",
        "epochs": "50",
        "batch-size": "2",
        "lr": "5e-5",
        "fp16": "true",
        "max-grad-norm": "1.0"
    }' \
    --input-data-config '[{
        "ChannelName": "train",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": "s3://'$S3_BUCKET'/vagi/data/",
                "S3DataDistributionType": "FullyReplicated"
            }
        }
    }]' \
    --output-data-config '{
        "S3OutputPath": "s3://'$S3_BUCKET'/vagi/output/"
    }' \
    --resource-config '{
        "InstanceType": "'$INSTANCE_TYPE'",
        "InstanceCount": '$INSTANCE_COUNT',
        "VolumeSizeInGB": 500
    }' \
    --stopping-condition '{
        "MaxRuntimeInSeconds": 259200
    }' \
    --enable-managed-spot-training \
    --checkpoint-config '{
        "S3Uri": "s3://'$S3_BUCKET'/vagi/checkpoints/'$JOB_NAME'/"
    }' \
    --tags '[
        {"Key": "Project", "Value": "vAGI"},
        {"Key": "ModelSize", "Value": "1.5B"}
    ]' \
    --region "$REGION"

echo ""
echo "=============================================="
echo "Training job submitted successfully!"
echo "=============================================="
echo "Job Name: $JOB_NAME"
echo "Instance: $INSTANCE_COUNT x $INSTANCE_TYPE"
echo ""
echo "Monitor progress:"
echo "  Console: https://$REGION.console.aws.amazon.com/sagemaker/home?region=$REGION#/jobs/$JOB_NAME"
echo ""
echo "Check status:"
echo "  aws sagemaker describe-training-job --training-job-name $JOB_NAME --region $REGION"
echo ""
echo "View logs:"
echo "  aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix $JOB_NAME --follow"
echo ""
echo "Download model when complete:"
echo "  aws s3 cp s3://$S3_BUCKET/vagi/output/$JOB_NAME/output/model.tar.gz ./"
echo "=============================================="
