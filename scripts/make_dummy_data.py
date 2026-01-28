"""Create a dummy torch dataset for vAGI scripts."""

from __future__ import annotations

import argparse

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a dummy dataset for vAGI.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--obs-dim", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    input_ids = torch.randint(0, args.vocab_size, (args.num_samples, args.seq_len), dtype=torch.long)
    labels = input_ids.clone()
    obs = torch.randn(args.num_samples, args.obs_dim)
    actions = torch.randint(0, args.action_dim, (args.num_samples,), dtype=torch.long)
    values = torch.randn(args.num_samples, 1)
    obs_next = torch.randn(args.num_samples, args.obs_dim)

    payload = {
        "input_ids": input_ids,
        "labels": labels,
        "obs": obs,
        "actions": actions,
        "values": values,
        "obs_next": obs_next,
    }
    torch.save(payload, args.output)
    print(f"Saved dummy dataset to {args.output}")


if __name__ == "__main__":
    main()
