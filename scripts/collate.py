"""Batch collation utilities."""

from __future__ import annotations

from typing import Callable, Dict, List

import torch


def _pad_sequences(seqs: List[torch.Tensor], pad_id: int, max_length: int) -> torch.Tensor:
    batch_size = len(seqs)
    out = torch.full((batch_size, max_length), pad_id, dtype=torch.long)
    for idx, seq in enumerate(seqs):
        length = min(seq.shape[0], max_length)
        out[idx, :length] = seq[:length]
    return out


def _shift_labels(input_ids: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    labels = input_ids.clone()
    if labels.shape[1] > 1:
        labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = ignore_index
    return labels


def make_collate_fn(
    pad_id: int,
    max_length: int,
    obs_dim: int = 0,
    add_obs: bool = False,
) -> Callable[[List[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]:
    def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        seqs = [item["input_ids"] for item in batch]
        input_ids = _pad_sequences(seqs, pad_id=pad_id, max_length=max_length)
        labels = _shift_labels(input_ids)
        output: Dict[str, torch.Tensor] = {"input_ids": input_ids, "labels": labels}
        if add_obs:
            if "obs" in batch[0]:
                output["obs"] = torch.stack([item["obs"] for item in batch], dim=0)
            else:
                output["obs"] = torch.zeros((len(batch), obs_dim), dtype=torch.float32)
        return output

    return collate
