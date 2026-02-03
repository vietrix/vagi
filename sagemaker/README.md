# Hướng Dẫn Train vAGI trên AWS SageMaker

## Mục Lục

1. [Tổng Quan](#1-tổng-quan)
2. [Chuẩn Bị AWS](#2-chuẩn-bị-aws)
3. [Chuẩn Bị Data](#3-chuẩn-bị-data)
4. [Launch Training](#4-launch-training)
5. [Giám Sát Training](#5-giám-sát-training)
6. [Download Model](#6-download-model)
7. [Chi Phí Ước Tính](#7-chi-phí-ước-tính)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Tổng Quan

### Model Sizes

| Size | Parameters | Instance Khuyến Nghị | VRAM | Thời Gian Train |
|------|-----------|---------------------|------|-----------------|
| tiny | 3.5M | ml.g5.xlarge | 24GB | ~1h |
| small | 30M | ml.g5.2xlarge | 24GB | ~4h |
| medium | 100M | ml.g5.4xlarge | 24GB | ~12h |
| large | 350M | ml.g5.12xlarge | 96GB | ~24h |
| xlarge | 1B | ml.p4d.24xlarge x2 | 640GB | ~48h |

---

## 2. Chuẩn Bị AWS

### 2.1 Tạo S3 Bucket

```bash
# Tạo bucket (thay YOUR_BUCKET_NAME)
aws s3 mb s3://vagi-training-YOUR_BUCKET_NAME --region us-east-1
```

### 2.2 Tạo IAM Role

1. Vào AWS Console → IAM → Roles → Create role
2. Chọn "SageMaker" làm trusted entity
3. Thêm policies:
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess`
4. Đặt tên: `SageMakerVAGIRole`
5. Copy ARN: `arn:aws:iam::ACCOUNT_ID:role/SageMakerVAGIRole`

### 2.3 Cài đặt AWS CLI

```bash
# Install
pip install awscli boto3 sagemaker

# Configure
aws configure
# AWS Access Key ID: [your-key]
# AWS Secret Access Key: [your-secret]
# Default region name: us-east-1
# Default output format: json

# Verify
aws sts get-caller-identity
```

---

## 3. Chuẩn Bị Data

### 3.1 Generate Synthetic Data (để test)

```bash
# Generate 10,000 samples
python sagemaker/prepare_data.py \
    --format synthetic \
    --output data/sagemaker/train.jsonl \
    --num-samples 10000 \
    --max-seq-len 512 \
    --obs-dim 256
```

### 3.2 Chuẩn Bị Data Thật

```bash
# Từ text file
python sagemaker/prepare_data.py \
    --format text \
    --input data/raw_text.txt \
    --output data/sagemaker/train.jsonl

# Từ experience data (JSON)
python sagemaker/prepare_data.py \
    --format experience \
    --input data/experiences.jsonl \
    --output data/sagemaker/train.jsonl
```

### 3.3 Upload Data lên S3

```bash
aws s3 sync data/sagemaker/ s3://YOUR_BUCKET/vagi/data/
```

---

## 4. Launch Training

### 4.1 Training Cơ Bản

```bash
# Small model - phù hợp demo
python sagemaker/launch_training.py \
    --s3-bucket YOUR_BUCKET \
    --model-size small \
    --epochs 10 \
    --data-dir data/sagemaker
```

### 4.2 Training với Spot Instances (tiết kiệm 70-90%)

```bash
python sagemaker/launch_training.py \
    --s3-bucket YOUR_BUCKET \
    --model-size medium \
    --spot \
    --epochs 20
```

### 4.3 Training Production (Large Model)

```bash
python sagemaker/launch_training.py \
    --s3-bucket YOUR_BUCKET \
    --model-size large \
    --instance-type ml.p4d.24xlarge \
    --instance-count 1 \
    --epochs 50 \
    --spot
```

### 4.4 Training Budget Mode

```bash
# Dùng instance rẻ hơn (chậm hơn ~2x)
python sagemaker/launch_training.py \
    --s3-bucket YOUR_BUCKET \
    --model-size small \
    --budget \
    --epochs 10
```

### 4.5 Các Options

| Option | Mô tả |
|--------|-------|
| `--model-size` | tiny/small/medium/large/xlarge |
| `--instance-type` | Override instance type |
| `--instance-count` | Số instance (multi-node) |
| `--spot` | Dùng spot instances |
| `--budget` | Dùng instance rẻ hơn |
| `--epochs` | Số epochs train |
| `--batch-size` | Batch size |
| `--lr` | Learning rate |
| `--wait` | Đợi training xong |
| `-y` | Bỏ qua confirmation |

---

## 5. Giám Sát Training

### 5.1 SageMaker Console

1. Vào AWS Console → SageMaker → Training jobs
2. Click vào job để xem:
   - Logs
   - Metrics (loss, throughput)
   - Resource usage

### 5.2 CloudWatch Logs

```bash
# Xem logs
aws logs get-log-events \
    --log-group-name /aws/sagemaker/TrainingJobs \
    --log-stream-name vagi-small-TIMESTAMP/algo-1-TIMESTAMP
```

### 5.3 TensorBoard (Local)

```bash
# Download logs từ S3
aws s3 sync s3://YOUR_BUCKET/vagi/output/JOB_NAME/output/logs ./logs

# Chạy TensorBoard
tensorboard --logdir ./logs
```

---

## 6. Download Model

### 6.1 Download từ S3

```bash
# List outputs
aws s3 ls s3://YOUR_BUCKET/vagi/output/JOB_NAME/output/

# Download model
aws s3 cp s3://YOUR_BUCKET/vagi/output/JOB_NAME/output/model.tar.gz ./

# Extract
tar -xzf model.tar.gz
```

### 6.2 Load Model

```python
import torch
from core.agi.model import AGIModel
from core.agi.config import AGIConfig
import json

# Load config
with open("config.json") as f:
    config_dict = json.load(f)
config = AGIConfig(**config_dict)

# Load model
model = AGIModel(config)
model.load_state_dict(torch.load("model_final.pt"))
model.eval()

# Inference
output = model(input_ids=..., obs=...)
```

---

## 7. Chi Phí Ước Tính

### 7.1 On-Demand Pricing (us-east-1)

| Instance | GPU | VRAM | $/hour |
|----------|-----|------|--------|
| ml.g4dn.xlarge | T4 | 16GB | $0.526 |
| ml.g4dn.2xlarge | T4 | 16GB | $0.752 |
| ml.g5.xlarge | A10G | 24GB | $1.006 |
| ml.g5.2xlarge | A10G | 24GB | $1.515 |
| ml.g5.4xlarge | A10G | 24GB | $2.534 |
| ml.g5.12xlarge | 4×A10G | 96GB | $7.090 |
| ml.p4d.24xlarge | 8×A100 | 320GB | $32.77 |

### 7.2 Spot Pricing (tiết kiệm 60-90%)

| Instance | On-Demand | Spot (ước tính) |
|----------|-----------|-----------------|
| ml.g5.xlarge | $1.006 | ~$0.30 |
| ml.g5.2xlarge | $1.515 | ~$0.45 |
| ml.g5.4xlarge | $2.534 | ~$0.76 |
| ml.p4d.24xlarge | $32.77 | ~$9.83 |

### 7.3 Ước Tính Chi Phí Training

| Model | Instance | Thời gian | On-Demand | Spot |
|-------|----------|-----------|-----------|------|
| small | ml.g5.2xlarge | 4h | ~$6 | ~$2 |
| medium | ml.g5.4xlarge | 12h | ~$30 | ~$9 |
| large | ml.g5.12xlarge | 24h | ~$170 | ~$50 |
| xlarge | ml.p4d.24xlarge×2 | 48h | ~$3,145 | ~$943 |

---

## 8. Troubleshooting

### 8.1 Out of Memory (OOM)

```bash
# Giảm batch size
python sagemaker/launch_training.py \
    --s3-bucket YOUR_BUCKET \
    --model-size medium \
    --batch-size 4  # Giảm từ 8
```

### 8.2 Spot Interruption

- Dùng `--spot` tự động enable checkpointing
- Job sẽ resume từ checkpoint khi có capacity
- Tăng `--max-wait` nếu cần

### 8.3 Slow Training

```bash
# Dùng instance mạnh hơn
python sagemaker/launch_training.py \
    --s3-bucket YOUR_BUCKET \
    --model-size small \
    --instance-type ml.g5.4xlarge  # Thay vì g5.2xlarge
```

### 8.4 Permission Denied

```bash
# Check IAM role có đủ permissions
aws iam list-attached-role-policies --role-name SageMakerVAGIRole

# Thêm S3 policy nếu thiếu
aws iam attach-role-policy \
    --role-name SageMakerVAGIRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### 8.5 Job Stuck in "Starting"

- Check instance capacity trong region
- Thử region khác: `--region us-west-2`
- Hoặc dùng different instance type

---

## Quick Start Commands

```bash
# 1. Prepare data
python sagemaker/prepare_data.py --format synthetic --output data/sagemaker/train.jsonl

# 2. Upload data
aws s3 sync data/sagemaker/ s3://YOUR_BUCKET/vagi/data/

# 3. Launch training (small model, spot instances)
python sagemaker/launch_training.py \
    --s3-bucket YOUR_BUCKET \
    --model-size small \
    --spot \
    --epochs 10 \
    -y

# 4. Download model
aws s3 cp s3://YOUR_BUCKET/vagi/output/JOB_NAME/output/model.tar.gz ./
tar -xzf model.tar.gz
```

---

## Liên Hệ

- Repository: https://github.com/vietrix/vagi
- Issues: https://github.com/vietrix/vagi/issues
- Email: zyntherdev@gmail.com
