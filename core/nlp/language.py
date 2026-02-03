"""Language understanding components for AGI.

Enhanced with:
1. LRU cache for efficient encoding
2. Proper space handling in decode
3. Complete Vietnamese Unicode coverage
4. Better error handling for OOV tokens
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from collections import OrderedDict
import logging

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class LRUCache:
    """LRU cache for tokenization results."""

    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self.cache: OrderedDict = OrderedDict()

    def get(self, key: str) -> Optional[List[int]]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: List[int]):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
            self.cache[key] = value

    def clear(self):
        self.cache.clear()


class BytePairTokenizer:
    """Enhanced BPE tokenizer with caching and proper space handling.

    Features:
    - LRU cache for repeated encodings
    - Proper space reconstruction in decode
    - Complete Vietnamese Unicode coverage
    - Explicit UNK handling with warnings
    - Word boundary markers for better subword handling
    """

    # Special marker for word boundaries (Ġ is commonly used in GPT-style tokenizers)
    WORD_BOUNDARY = "Ġ"

    def __init__(
        self,
        vocab_size: int = 50000,
        cache_size: int = 10000,
        use_word_boundaries: bool = True,
        unk_token: str = "<UNK>",
    ) -> None:
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.use_word_boundaries = use_word_boundaries
        self.unk_token = unk_token

        # Cache for encoding
        self._cache = LRUCache(cache_size)

        # OOV tracking
        self._oov_count = 0
        self._total_tokens = 0

        self._build_base_vocab()

    def _build_base_vocab(self) -> None:
        """Initialize with comprehensive character coverage."""
        idx = 0

        # ASCII printable characters (0-255)
        for i in range(256):
            char = chr(i)
            self.vocab[char] = idx
            self.inverse_vocab[idx] = char
            idx += 1

        # Word boundary marker
        if self.use_word_boundaries:
            self.vocab[self.WORD_BOUNDARY] = idx
            self.inverse_vocab[idx] = self.WORD_BOUNDARY
            idx += 1

        # Vietnamese characters - COMPLETE coverage
        vietnamese_chars = (
            # Lowercase vowels with all diacritics
            "àáảãạăằắẳẵặâầấẩẫậ"  # a variants (17)
            "èéẻẽẹêềếểễệ"        # e variants (11)
            "ìíỉĩị"              # i variants (5)
            "òóỏõọôồốổỗộơờớởỡợ"  # o variants (17)
            "ùúủũụưừứửữự"        # u variants (11)
            "ỳýỷỹỵ"              # y variants (5)
            "đ"                   # d with stroke (1)
            # Uppercase vowels with all diacritics
            "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ"  # A variants
            "ÈÉẺẼẸÊỀẾỂỄỆ"        # E variants
            "ÌÍỈĨỊ"              # I variants
            "ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"  # O variants
            "ÙÚỦŨỤƯỪỨỬỮỰ"        # U variants
            "ỲÝỶỸỴ"              # Y variants
            "Đ"                   # D with stroke
        )
        for char in vietnamese_chars:
            if char not in self.vocab:
                self.vocab[char] = idx
                self.inverse_vocab[idx] = char
                idx += 1

        # Extended Unicode punctuation and symbols
        extra_chars = (
            # Typography
            "–—''""…•·"
            # Math symbols
            "°±×÷²³¹⁰√∞≈≠≤≥πΣ∑∏∫∂∇±∓∘∙×÷"
            # Currency
            "₫€£¥₹₿₽₩"
            # Arrows
            "←→↑↓↔↕⇒⇔"
            # Misc
            "©®™§¶†‡№℃℉"
            # Combining diacritics (for edge cases)
            "\u0300\u0301\u0302\u0303\u0304\u0306\u0309\u0323"  # grave, acute, circumflex, tilde, macron, breve, hook above, dot below
        )
        for char in extra_chars:
            if char not in self.vocab:
                self.vocab[char] = idx
                self.inverse_vocab[idx] = char
                idx += 1

        # Special tokens
        special_tokens = [
            "<PAD>", "<UNK>", "<BOS>", "<EOS>",
            "<SEP>", "<CLS>", "<MASK>", "<ACT>",
            "<SPACE>", "<NEWLINE>", "<TAB>"
        ]
        for token in special_tokens:
            if token not in self.vocab:
                self.vocab[token] = idx
                self.inverse_vocab[idx] = token
                idx += 1

        # Store special token IDs
        self.pad_token_id = self.vocab.get("<PAD>", 0)
        self.unk_token_id = self.vocab.get("<UNK>", 1)
        self.bos_token_id = self.vocab.get("<BOS>", 2)
        self.eos_token_id = self.vocab.get("<EOS>", 3)

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

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        add_special_tokens: bool = False,
        use_cache: bool = True
    ) -> List[int]:
        """Encode text to token IDs with caching.

        Args:
            text: Text to encode
            max_length: Maximum output length (truncate if exceeded)
            add_special_tokens: Add BOS/EOS tokens
            use_cache: Use LRU cache for repeated texts

        Returns:
            List of token IDs
        """
        # Check cache first
        cache_key = f"{text}:{max_length}:{add_special_tokens}"
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Preprocess: add word boundaries if enabled
        if self.use_word_boundaries:
            # Add boundary marker before each word (after spaces)
            processed_text = ""
            prev_was_space = True
            for char in text:
                if char == " ":
                    prev_was_space = True
                    processed_text += char
                else:
                    if prev_was_space:
                        processed_text += self.WORD_BOUNDARY + char
                    else:
                        processed_text += char
                    prev_was_space = False
            text = processed_text

        # Character-level tokenization
        tokens = []
        for char in text:
            token_id = self.vocab.get(char)
            if token_id is None:
                # OOV handling with warning
                token_id = self.unk_token_id
                self._oov_count += 1
                if self._oov_count <= 10:  # Only log first 10 OOVs
                    logger.debug(f"OOV character: {repr(char)}")
            tokens.append(token_id)
            self._total_tokens += 1

        # Apply BPE merges
        for merge_pair in self.merges:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1:
                    current = self.inverse_vocab.get(tokens[i], "")
                    next_tok = self.inverse_vocab.get(tokens[i + 1], "")
                    if (current, next_tok) == merge_pair:
                        merged = current + next_tok
                        merged_id = self.vocab.get(merged)
                        if merged_id is not None:
                            new_tokens.append(merged_id)
                            i += 2
                            continue
                new_tokens.append(tokens[i])
                i += 1
            tokens = new_tokens

        # Add special tokens
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]

        # Truncate if needed
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]

        # Cache result
        if use_cache:
            self._cache.put(cache_key, tokens)

        return tokens

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_spaces: bool = True
    ) -> str:
        """Decode token IDs to text with proper space handling.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Skip special tokens like PAD, BOS, EOS
            clean_up_spaces: Clean up extra spaces and word boundary markers

        Returns:
            Decoded text string
        """
        special_token_ids = {
            self.pad_token_id, self.bos_token_id, self.eos_token_id,
            self.vocab.get("<SEP>"), self.vocab.get("<CLS>"),
            self.vocab.get("<MASK>"), self.vocab.get("<ACT>")
        }

        tokens = []
        for tid in token_ids:
            if skip_special_tokens and tid in special_token_ids:
                continue
            token = self.inverse_vocab.get(tid, self.unk_token)
            tokens.append(token)

        text = "".join(tokens)

        if clean_up_spaces:
            # Replace word boundary marker with space
            if self.use_word_boundaries:
                text = text.replace(self.WORD_BOUNDARY, " ")

            # Clean up multiple spaces
            while "  " in text:
                text = text.replace("  ", " ")

            # Strip leading/trailing spaces
            text = text.strip()

        return text

    def get_oov_rate(self) -> float:
        """Get OOV (out-of-vocabulary) rate."""
        if self._total_tokens == 0:
            return 0.0
        return self._oov_count / self._total_tokens

    def clear_cache(self):
        """Clear the encoding cache."""
        self._cache.clear()

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

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
    """Language modeling head with vocabulary projection and output normalization.

    Features:
    - Optional layer norm before projection for numerical stability
    - Weight tying support with embedding layer
    - Temperature scaling for generation
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        tie_weights: bool = True,
        use_output_norm: bool = True
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tie_weights = tie_weights

        # Output normalization for numerical stability
        if use_output_norm:
            self.output_norm = nn.LayerNorm(hidden_size)
        else:
            self.output_norm = None

        # Projection layer
        self.projection = nn.Linear(hidden_size, vocab_size, bias=False)

        # For weight tying (set by parent module)
        self._tied_embedding = None

    def tie_to_embedding(self, embedding: nn.Embedding):
        """Tie output weights to embedding weights."""
        if self.tie_weights:
            self._tied_embedding = embedding
            # Share weights
            self.projection.weight = embedding.weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Project hidden states to vocabulary logits.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            temperature: Temperature for logit scaling (for generation)

        Returns:
            Logits of shape [batch, seq_len, vocab_size]
        """
        # Apply output normalization
        if self.output_norm is not None:
            hidden_states = self.output_norm(hidden_states)

        # Project to vocabulary
        logits = self.projection(hidden_states)

        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        return logits


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
