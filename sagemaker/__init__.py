"""SageMaker training utilities for vAGI."""

from .train_sagemaker import get_model_config, VAGIDataset

__all__ = ["get_model_config", "VAGIDataset"]
