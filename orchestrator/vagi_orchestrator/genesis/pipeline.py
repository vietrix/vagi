from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from .data_builder import build_dialogue_intent_samples
from .model import TinyGruLm
from .tokenizer import CharTokenizer
from ..semantic_map import SemanticEpisodeMap


@dataclass(slots=True)
class GenesisConfig:
    repeats: int = 28
    embed_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    seq_len: int = 64
    batch_size: int = 32
    epochs: int = 12
    lr: float = 3e-3
    seed: int = 42
    max_seq_len: int = 128
    titanium_mode: bool = False
    logic_filter_min_score: float = 0.3


def train_and_export(output_dir: Path, config: GenesisConfig | None = None) -> dict[str, Any]:
    config = config or GenesisConfig()
    _set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = _build_training_texts(config=config)
    tokenizer = CharTokenizer.build_from_texts(texts)
    windows = _build_windows(texts=texts, tokenizer=tokenizer, seq_len=config.seq_len)
    train_loader, val_loader = _build_loaders(windows=windows, batch_size=config.batch_size)

    model = TinyGruLm(
        vocab_size=len(tokenizer.tokens),
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    best_state: dict[str, Tensor] | None = None
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            pad_id=tokenizer.pad_id,
            train=True,
        )
        val_loss = _run_epoch(
            model=model,
            dataloader=val_loader,
            optimizer=optimizer,
            pad_id=tokenizer.pad_id,
            train=False,
        )
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().contiguous().to(torch.float32)
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a model state")

    model_path = output_dir / "model.safetensors"
    vocab_path = output_dir / "vocab.json"
    manifest_path = output_dir / "manifest.json"

    save_file(best_state, str(model_path))
    tokenizer.save(vocab_path)

    arch = "vagi-titanium-gru" if config.titanium_mode else "tiny-gru-lm"
    manifest = {
        "model_id": "genesis-v0",
        "arch": arch,
        "version": "0.1.0",
        "vocab_size": len(tokenizer.tokens),
        "embed_dim": config.embed_dim,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
        "pad_id": tokenizer.pad_id,
        "unk_id": tokenizer.unk_id,
        "max_seq_len": config.max_seq_len,
        "model_file": "model.safetensors",
        "vocab_file": "vocab.json",
        "model_sha256": _sha256_file(model_path),
        "vocab_sha256": _sha256_file(vocab_path),
        "metrics": {
            "best_val_loss": best_val_loss,
            "epochs": config.epochs,
            "titanium_mode": config.titanium_mode,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    smoke_text = greedy_generate(
        model_state=best_state,
        tokenizer=tokenizer,
        prompt="User: Xin chao\nAssistant:",
        max_new_tokens=48,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    )

    return {
        "model_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "model_path": str(model_path),
        "vocab_path": str(vocab_path),
        "smoke_text": smoke_text,
        "best_val_loss": best_val_loss,
        "history": history,
        "titanium_mode": config.titanium_mode,
    }


def _build_training_texts(config: GenesisConfig) -> list[str]:
    texts = build_dialogue_intent_samples(repeats=config.repeats)
    if not config.titanium_mode:
        return texts

    # Titanium mode: bias toward logic-dense samples using a semantic map proxy.
    semantic = SemanticEpisodeMap(dim=512)
    for text in texts:
        semantic.add_episode(user_input=text, draft=text)

    scored: list[tuple[float, str]] = []
    probes = [
        "prove correctness with invariants",
        "implement secure login with verifier checks",
        "derive equation and validate each step",
    ]
    for text in texts:
        signal = 0.0
        for probe in probes:
            hits = semantic.query(f"{probe} {text[:120]}", top_k=1, min_score=-1.0)
            if hits:
                signal += float(hits[0].score)
        lower = text.lower()
        if any(token in lower for token in ("assert", "because", "therefore", "verify", "proof")):
            signal += 0.35
        if any(token in lower for token in ("fn ", "def ", "class ", "{")):
            signal += 0.45
        scored.append((signal, text))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return texts
    keep = max(16, int(len(scored) * max(0.15, min(1.0, config.logic_filter_min_score))))
    return [text for _, text in scored[:keep]]


def greedy_generate(
    *,
    model_state: dict[str, Tensor],
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    embed_dim: int,
    hidden_dim: int,
    num_layers: int,
) -> str:
    model = TinyGruLm(
        vocab_size=len(tokenizer.tokens),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    model.load_state_dict(model_state)
    model.eval()

    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    hidden = None
    with torch.no_grad():
        seq = torch.tensor([input_ids], dtype=torch.long)
        logits, hidden = model(seq, hidden)
        generated: list[int] = []
        next_id = int(torch.argmax(logits[0, -1]).item())
        for _ in range(max_new_tokens):
            if next_id == tokenizer.eos_id:
                break
            generated.append(next_id)
            step_in = torch.tensor([[next_id]], dtype=torch.long)
            step_logits, hidden = model(step_in, hidden)
            next_id = int(torch.argmax(step_logits[0, -1]).item())
    return tokenizer.decode(generated)


def _run_epoch(
    *,
    model: TinyGruLm,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    pad_id: int,
    train: bool,
) -> float:
    model.train(mode=train)
    total_loss = 0.0
    batches = 0
    for input_ids, targets in dataloader:
        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=pad_id,
        )
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    if batches == 0:
        return 0.0
    return total_loss / batches


def _build_windows(*, texts: list[str], tokenizer: CharTokenizer, seq_len: int) -> Tensor:
    token_stream: list[int] = []
    for text in texts:
        token_stream.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
    if len(token_stream) < seq_len + 1:
        token_stream = token_stream * 4

    windows: list[list[int]] = []
    stride = max(1, seq_len // 2)
    for offset in range(0, len(token_stream) - 1, stride):
        chunk = token_stream[offset : offset + seq_len + 1]
        if len(chunk) < seq_len + 1:
            chunk = chunk + [tokenizer.pad_id] * (seq_len + 1 - len(chunk))
        windows.append(chunk)
        if offset + seq_len + 1 >= len(token_stream):
            break

    if not windows:
        windows.append([tokenizer.pad_id] * (seq_len + 1))
    return torch.tensor(windows, dtype=torch.long)


def _build_loaders(*, windows: Tensor, batch_size: int) -> tuple[DataLoader, DataLoader]:
    inputs = windows[:, :-1]
    targets = windows[:, 1:]
    split_at = max(1, int(inputs.size(0) * 0.9))
    train_ds = TensorDataset(inputs[:split_at], targets[:split_at])
    val_ds = TensorDataset(inputs[split_at:], targets[split_at:])
    if len(val_ds) == 0:
        val_ds = train_ds
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
