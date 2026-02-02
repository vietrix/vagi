"""Language understanding components for AGI."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class BytePairTokenizer:
    """Simple BPE tokenizer for text processing."""

    def __init__(self, vocab_size: int = 50000) -> None:
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self._build_base_vocab()

    def _build_base_vocab(self) -> None:
        """Initialize with ASCII + Vietnamese characters."""
        idx = 0

        # ASCII printable characters
        for i in range(256):
            char = chr(i)
            self.vocab[char] = idx
            self.inverse_vocab[idx] = char
            idx += 1

        # Vietnamese characters (diacritics)
        vietnamese_chars = (
            # Lowercase vowels with diacritics
            "àáảãạăằắẳẵặâầấẩẫậ"  # a variants
            "èéẻẽẹêềếểễệ"        # e variants
            "ìíỉĩị"              # i variants
            "òóỏõọôồốổỗộơờớởỡợ"  # o variants
            "ùúủũụưừứửữự"        # u variants
            "ỳýỷỹỵ"              # y variants
            "đ"                   # d with stroke
            # Uppercase vowels with diacritics
            "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ"
            "ÈÉẺẼẸÊỀẾỂỄỆ"
            "ÌÍỈĨỊ"
            "ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
            "ÙÚỦŨỤƯỪỨỬỮỰ"
            "ỲÝỶỸỴ"
            "Đ"
        )
        for char in vietnamese_chars:
            if char not in self.vocab:
                self.vocab[char] = idx
                self.inverse_vocab[idx] = char
                idx += 1

        # Common punctuation and symbols
        extra_chars = "–—''""…•°±×÷²³√∞≈≠≤≥πΣ"
        for char in extra_chars:
            if char not in self.vocab:
                self.vocab[char] = idx
                self.inverse_vocab[idx] = char
                idx += 1

        # Special tokens
        special_tokens = [
            "<PAD>", "<UNK>", "<BOS>", "<EOS>",
            "<SEP>", "<CLS>", "<MASK>", "<ACT>"
        ]
        for token in special_tokens:
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token
            idx += 1

    def train(self, texts: List[str], num_merges: int = 10000) -> None:
        """Train BPE on corpus."""
        if num_merges <= 0:
            return
        
        word_freqs: Dict[str, int] = {}
        for text in texts:
            words = text.split()
            for word in words:
                word_freqs[word] = word_freqs.get(word, 0) + 1
        
        vocab = {word: list(word) for word in word_freqs}
        
        for _ in range(num_merges):
            pairs: Dict[Tuple[str, str], int] = {}
            for word, freq in word_freqs.items():
                symbols = vocab[word]
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    pairs[pair] = pairs.get(pair, 0) + freq
            
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            
            if new_token not in self.vocab:
                token_id = len(self.vocab)
                if token_id >= self.vocab_size:
                    break
                self.vocab[new_token] = token_id
                self.inverse_vocab[token_id] = new_token
            
            for word in vocab:
                symbols = vocab[word]
                i = 0
                new_symbols = []
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                        new_symbols.append(new_token)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                vocab[word] = new_symbols

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Encode text to token IDs."""
        tokens = []
        for char in text:
            token_id = self.vocab.get(char, self.vocab.get("<UNK>", 1))
            tokens.append(token_id)
        
        for merge_pair in self.merges:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1:
                    current = self.inverse_vocab.get(tokens[i], "")
                    next_tok = self.inverse_vocab.get(tokens[i + 1], "")
                    if (current, next_tok) == merge_pair:
                        merged = current + next_tok
                        merged_id = self.vocab.get(merged, tokens[i])
                        new_tokens.append(merged_id)
                        i += 2
                        continue
                new_tokens.append(tokens[i])
                i += 1
            tokens = new_tokens
        
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = [self.inverse_vocab.get(tid, "<UNK>") for tid in token_ids]
        return "".join(tokens)

    def save(self, path: str | Path) -> None:
        """Save tokenizer state."""
        path = Path(path)
        import json
        state = {
            "vocab": self.vocab,
            "merges": self.merges,
            "vocab_size": self.vocab_size,
        }
        path.write_text(json.dumps(state, indent=2))

    def load(self, path: str | Path) -> None:
        """Load tokenizer state."""
        path = Path(path)
        import json
        state = json.loads(path.read_text())
        self.vocab = state["vocab"]
        self.vocab_size = state["vocab_size"]
        self.merges = [tuple(pair) for pair in state["merges"]]
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, hidden_size: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-torch.log(torch.tensor(10000.0)) / hidden_size))
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to embeddings."""
        x = x + self.pe[: x.size(1), :]
        return self.dropout(x)


class TextEmbedding(nn.Module):
    """Text embedding with positional encoding."""

    def __init__(self, vocab_size: int, hidden_size: int, max_seq_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_encoding = PositionalEncoding(hidden_size, max_seq_len, dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs."""
        x = self.token_embedding(input_ids)
        x = self.position_encoding(x)
        return self.layer_norm(x)


class LanguageHead(nn.Module):
    """Language modeling head with vocabulary projection."""

    def __init__(self, hidden_size: int, vocab_size: int, tie_weights: bool = True) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.projection = nn.Linear(hidden_size, vocab_size, bias=False)
        self.tie_weights = tie_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits."""
        return self.projection(hidden_states)


class NextTokenPredictor(nn.Module):
    """Next token prediction for language modeling."""

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.head = LanguageHead(hidden_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Predict next tokens and optionally compute loss."""
        logits = self.head(hidden_states)
        
        outputs = {"logits": logits}
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            outputs["loss"] = loss
        
        return outputs


class MaskedLanguageModel(nn.Module):
    """Masked language modeling for pre-training."""

    def __init__(self, hidden_size: int, vocab_size: int, mask_prob: float = 0.15) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob
        self.mask_token_id = vocab_size - 1
        self.predictor = NextTokenPredictor(hidden_size, vocab_size)

    def create_masked_input(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create masked input and labels for MLM."""
        labels = input_ids.clone()
        masked_input = input_ids.clone()
        
        probability_matrix = torch.full(labels.shape, self.mask_prob)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        labels[~masked_indices] = -100
        
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        masked_input[indices_replaced] = self.mask_token_id
        
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        masked_input[indices_random] = random_words[indices_random]
        
        return masked_input, labels

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute MLM loss."""
        _, labels = self.create_masked_input(input_ids)
        return self.predictor(hidden_states, labels)


class SentenceEncoder(nn.Module):
    """Encode sentences to fixed-size vectors."""

    def __init__(self, hidden_size: int, pooling: str = "mean") -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.pooling = pooling

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pool hidden states to sentence embeddings."""
        if self.pooling == "mean":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            return hidden_states.mean(dim=1)
        elif self.pooling == "max":
            return hidden_states.max(dim=1)[0]
        elif self.pooling == "cls":
            return hidden_states[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
