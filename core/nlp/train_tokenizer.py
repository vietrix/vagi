#!/usr/bin/env python3
"""
Custom BPE Tokenizer Training for vAGI.

Trains a Byte Pair Encoding tokenizer optimized for Vietnamese text and
programming code using the Hugging Face tokenizers library.

Usage:
    # Train with default settings
    python core/nlp/train_tokenizer.py --output tokenizers/vagi_tokenizer

    # Train on specific data directories
    python core/nlp/train_tokenizer.py \
        --wiki-dir data/vietnamese_wiki \
        --code-dir data/github_code \
        --output tokenizers/vagi_tokenizer

    # Compare compression ratio with GPT-4 tokenizer
    python core/nlp/train_tokenizer.py --output tokenizers/vagi_tokenizer --compare-gpt4

    # Use pre-downloaded datasets
    python core/nlp/train_tokenizer.py \
        --wiki-files data/wiki/*.txt \
        --code-files data/code/**/*.py \
        --output tokenizers/vagi_tokenizer

Data Sources:
    - Vietnamese Wikipedia: Downloads from Hugging Face datasets
    - GitHub Code: Downloads from codeparrot/github-code or local files

Output:
    - tokenizers/vagi_tokenizer/tokenizer.json: Main tokenizer file
    - tokenizers/vagi_tokenizer/vocab.json: Vocabulary mapping
    - tokenizers/vagi_tokenizer/merges.txt: BPE merge rules
    - tokenizers/vagi_tokenizer/config.json: Tokenizer config
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterator, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Special Tokens Configuration
# ============================================================================

SPECIAL_TOKENS = [
    "<|endoftext|>",   # End of text marker (GPT-style)
    "<pad>",           # Padding token
    "<think>",         # Start reasoning block (for vAGI)
    "</think>",        # End reasoning block
    "<code_start>",    # Start code block
    "<code_end>",      # End code block
    "<|user|>",        # User turn marker
    "<|assistant|>",   # Assistant turn marker
    "<|system|>",      # System prompt marker
    "<unk>",           # Unknown token
    "<mask>",          # Masking token for MLM
]

# Default vocabulary size
DEFAULT_VOCAB_SIZE = 32000

# Languages for code training
CODE_LANGUAGES = ["python", "javascript", "java", "go", "rust", "c", "cpp"]


# ============================================================================
# Data Iterators
# ============================================================================

def iter_vietnamese_wiki(
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Iterator[str]:
    """
    Iterate over Vietnamese Wikipedia articles from Hugging Face datasets.

    Args:
        cache_dir: Cache directory for downloaded data
        max_samples: Maximum number of samples to yield

    Yields:
        Text content of Wikipedia articles
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        return

    logger.info("Loading Vietnamese Wikipedia from Hugging Face...")

    try:
        # Try loading Vietnamese Wikipedia
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.vi",
            split="train",
            cache_dir=cache_dir,
            trust_remote_code=True
        )

        count = 0
        for item in dataset:
            text = item.get("text", "")
            if text and len(text) > 100:  # Filter short articles
                yield text
                count += 1
                if max_samples and count >= max_samples:
                    break

        logger.info(f"Loaded {count} Vietnamese Wikipedia articles")

    except Exception as e:
        logger.warning(f"Could not load Vietnamese Wikipedia: {e}")
        logger.info("Trying alternative dataset...")

        try:
            # Fallback to CC-100 Vietnamese
            dataset = load_dataset(
                "cc100",
                lang="vi",
                split="train",
                cache_dir=cache_dir,
                streaming=True
            )

            count = 0
            for item in dataset:
                text = item.get("text", "")
                if text and len(text) > 50:
                    yield text
                    count += 1
                    if max_samples and count >= max_samples:
                        break

            logger.info(f"Loaded {count} texts from CC-100 Vietnamese")

        except Exception as e2:
            logger.error(f"Could not load alternative dataset: {e2}")


def iter_github_code(
    languages: list[str] = CODE_LANGUAGES,
    cache_dir: Optional[str] = None,
    max_samples_per_lang: int = 10000
) -> Iterator[str]:
    """
    Iterate over GitHub code from Hugging Face datasets.

    Args:
        languages: List of programming languages to include
        cache_dir: Cache directory for downloaded data
        max_samples_per_lang: Maximum samples per language

    Yields:
        Code content from GitHub repositories
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        return

    logger.info(f"Loading GitHub code for languages: {languages}")

    for lang in languages:
        try:
            logger.info(f"Loading {lang} code...")

            # Try codeparrot/github-code dataset
            dataset = load_dataset(
                "codeparrot/github-code",
                languages=[lang],
                split="train",
                cache_dir=cache_dir,
                streaming=True,
                trust_remote_code=True
            )

            count = 0
            for item in dataset:
                code = item.get("code", "")
                if code and len(code) > 50 and len(code) < 100000:  # Filter by length
                    yield code
                    count += 1
                    if count >= max_samples_per_lang:
                        break

            logger.info(f"Loaded {count} {lang} code samples")

        except Exception as e:
            logger.warning(f"Could not load {lang} code: {e}")


def iter_local_files(pattern: str) -> Iterator[str]:
    """
    Iterate over local text files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., "data/*.txt", "code/**/*.py")

    Yields:
        Content of matching files
    """
    files = glob.glob(pattern, recursive=True)
    logger.info(f"Found {len(files)} files matching {pattern}")

    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if content and len(content) > 50:
                    yield content
        except Exception as e:
            logger.warning(f"Could not read {filepath}: {e}")


def batch_iterator(iterator: Iterator[str], batch_size: int = 1000) -> Iterator[list[str]]:
    """Batch an iterator into chunks."""
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ============================================================================
# Tokenizer Training
# ============================================================================

class VAGITokenizerTrainer:
    """Train a BPE tokenizer for vAGI."""

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        min_frequency: int = 2,
        special_tokens: list[str] = SPECIAL_TOKENS
    ):
        """
        Initialize the tokenizer trainer.

        Args:
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for tokens
            special_tokens: List of special tokens to include
        """
        try:
            from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
            self.tokenizers_lib = {
                "Tokenizer": Tokenizer,
                "models": models,
                "trainers": trainers,
                "pre_tokenizers": pre_tokenizers,
                "decoders": decoders,
                "processors": processors
            }
        except ImportError:
            raise ImportError("Please install tokenizers: pip install tokenizers")

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens
        self.tokenizer = None

    def create_tokenizer(self) -> None:
        """Create the base BPE tokenizer."""
        Tokenizer = self.tokenizers_lib["Tokenizer"]
        models = self.tokenizers_lib["models"]
        pre_tokenizers = self.tokenizers_lib["pre_tokenizers"]
        decoders = self.tokenizers_lib["decoders"]
        processors = self.tokenizers_lib["processors"]

        # Create BPE model
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

        # Pre-tokenizer: Split on whitespace and punctuation
        # ByteLevel handles UTF-8 encoding properly
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ])

        # Decoder for converting back to text
        self.tokenizer.decoder = decoders.ByteLevel()

        # Post-processor for adding special tokens
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        logger.info("Created base BPE tokenizer")

    def train(
        self,
        data_iterator: Iterator[str],
        show_progress: bool = True
    ) -> None:
        """
        Train the tokenizer on data.

        Args:
            data_iterator: Iterator yielding text samples
            show_progress: Whether to show training progress
        """
        if self.tokenizer is None:
            self.create_tokenizer()

        trainers = self.tokenizers_lib["trainers"]

        # Create BPE trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=show_progress,
            initial_alphabet=trainers.BpeTrainer.get_initial_alphabet()
        )

        logger.info(f"Training tokenizer (vocab_size={self.vocab_size})...")

        # Train from iterator (batched for memory efficiency)
        self.tokenizer.train_from_iterator(
            batch_iterator(data_iterator),
            trainer=trainer,
            length=None  # Unknown length for iterator
        )

        logger.info(f"Training complete. Vocab size: {self.tokenizer.get_vocab_size()}")

    def save(self, output_dir: str) -> None:
        """
        Save the tokenizer to disk.

        Args:
            output_dir: Directory to save tokenizer files
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not trained yet")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save main tokenizer file
        self.tokenizer.save(str(output_path / "tokenizer.json"))

        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "special_tokens": self.special_tokens,
            "model_type": "BPE",
            "tokenizer_class": "PreTrainedTokenizerFast"
        }
        with open(output_path / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        # Create tokenizer_config.json for HuggingFace compatibility
        tokenizer_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "eos_token": "<|endoftext|>",
            "bos_token": "<|endoftext|>",
            "model_max_length": 8192,
            "special_tokens_map_file": None,
        }
        with open(output_path / "tokenizer_config.json", 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, indent=2)

        logger.info(f"Tokenizer saved to {output_path}")

    def load(self, path: str) -> None:
        """Load a trained tokenizer."""
        Tokenizer = self.tokenizers_lib["Tokenizer"]
        self.tokenizer = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
        logger.info(f"Loaded tokenizer from {path}")

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.tokenizer.decode(ids)


# ============================================================================
# Compression Ratio Calculation
# ============================================================================

def calculate_compression_ratio(
    tokenizer,
    test_texts: list[str],
    tokenizer_name: str = "custom"
) -> dict:
    """
    Calculate compression statistics for a tokenizer.

    Args:
        tokenizer: Tokenizer object with encode method
        test_texts: List of test texts
        tokenizer_name: Name for logging

    Returns:
        Dict with compression statistics
    """
    total_chars = 0
    total_tokens = 0
    total_bytes = 0

    for text in test_texts:
        if hasattr(tokenizer, 'encode'):
            if hasattr(tokenizer.encode(text), 'ids'):
                tokens = tokenizer.encode(text).ids
            else:
                tokens = tokenizer.encode(text)
        else:
            tokens = tokenizer(text)

        total_chars += len(text)
        total_tokens += len(tokens)
        total_bytes += len(text.encode('utf-8'))

    stats = {
        "tokenizer": tokenizer_name,
        "total_chars": total_chars,
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "chars_per_token": total_chars / total_tokens if total_tokens else 0,
        "bytes_per_token": total_bytes / total_tokens if total_tokens else 0,
        "compression_ratio": total_bytes / total_tokens if total_tokens else 0
    }

    return stats


def compare_with_gpt4(custom_tokenizer, test_texts: list[str]) -> None:
    """
    Compare compression ratio with GPT-4 tokenizer.

    Args:
        custom_tokenizer: Our trained tokenizer
        test_texts: Test texts for comparison
    """
    try:
        import tiktoken
    except ImportError:
        logger.warning("tiktoken not installed. Install with: pip install tiktoken")
        logger.warning("Skipping GPT-4 comparison")
        return

    # Load GPT-4 tokenizer (cl100k_base encoding)
    gpt4_tokenizer = tiktoken.get_encoding("cl100k_base")

    # Calculate stats for both
    custom_stats = calculate_compression_ratio(
        custom_tokenizer,
        test_texts,
        "vAGI (custom)"
    )

    gpt4_stats = calculate_compression_ratio(
        lambda text: gpt4_tokenizer.encode(text),
        test_texts,
        "GPT-4 (cl100k_base)"
    )

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPRESSION RATIO COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<25} {'vAGI (Custom)':<20} {'GPT-4':<20}")
    print("-" * 70)
    print(f"{'Total Characters':<25} {custom_stats['total_chars']:<20,} {gpt4_stats['total_chars']:<20,}")
    print(f"{'Total Bytes':<25} {custom_stats['total_bytes']:<20,} {gpt4_stats['total_bytes']:<20,}")
    print(f"{'Total Tokens':<25} {custom_stats['total_tokens']:<20,} {gpt4_stats['total_tokens']:<20,}")
    print(f"{'Chars/Token':<25} {custom_stats['chars_per_token']:<20.2f} {gpt4_stats['chars_per_token']:<20.2f}")
    print(f"{'Bytes/Token':<25} {custom_stats['bytes_per_token']:<20.2f} {gpt4_stats['bytes_per_token']:<20.2f}")
    print("-" * 70)

    # Calculate improvement
    if gpt4_stats['total_tokens'] > 0 and custom_stats['total_tokens'] > 0:
        token_reduction = (gpt4_stats['total_tokens'] - custom_stats['total_tokens']) / gpt4_stats['total_tokens'] * 100
        compression_improvement = (custom_stats['bytes_per_token'] - gpt4_stats['bytes_per_token']) / gpt4_stats['bytes_per_token'] * 100

        print(f"\n{'Token Reduction vs GPT-4:':<25} {token_reduction:+.1f}%")
        print(f"{'Compression Improvement:':<25} {compression_improvement:+.1f}%")

        if token_reduction > 0:
            print(f"\n>>> vAGI tokenizer uses {token_reduction:.1f}% FEWER tokens than GPT-4")
        else:
            print(f"\n>>> vAGI tokenizer uses {-token_reduction:.1f}% MORE tokens than GPT-4")

    print("=" * 70 + "\n")


# ============================================================================
# Vietnamese Character Sets (Complete Coverage)
# ============================================================================

# All Vietnamese characters with diacritics
VIETNAMESE_LOWERCASE = (
    # Base vowels with all 6 tones (ngang, huyền, sắc, hỏi, ngã, nặng)
    "a à á ả ã ạ "    # a
    "ă ằ ắ ẳ ẵ ặ "    # ă (a breve)
    "â ầ ấ ẩ ẫ ậ "    # â (a circumflex)
    "e è é ẻ ẽ ẹ "    # e
    "ê ề ế ể ễ ệ "    # ê (e circumflex)
    "i ì í ỉ ĩ ị "    # i
    "o ò ó ỏ õ ọ "    # o
    "ô ồ ố ổ ỗ ộ "    # ô (o circumflex)
    "ơ ờ ớ ở ỡ ợ "    # ơ (o horn)
    "u ù ú ủ ũ ụ "    # u
    "ư ừ ứ ử ữ ự "    # ư (u horn)
    "y ỳ ý ỷ ỹ ỵ "    # y
    "đ"                # đ (d with stroke)
).replace(" ", "")

VIETNAMESE_UPPERCASE = (
    "A À Á Ả Ã Ạ "
    "Ă Ằ Ắ Ẳ Ẵ Ặ "
    "Â Ầ Ấ Ẩ Ẫ Ậ "
    "E È É Ẻ Ẽ Ẹ "
    "Ê Ề Ế Ể Ễ Ệ "
    "I Ì Í Ỉ Ĩ Ị "
    "O Ò Ó Ỏ Õ Ọ "
    "Ô Ồ Ố Ổ Ỗ Ộ "
    "Ơ Ờ Ớ Ở Ỡ Ợ "
    "U Ù Ú Ủ Ũ Ụ "
    "Ư Ừ Ứ Ử Ữ Ự "
    "Y Ỳ Ý Ỷ Ỹ Ỵ "
    "Đ"
).replace(" ", "")

VIETNAMESE_CHARS = VIETNAMESE_LOWERCASE + VIETNAMESE_UPPERCASE


# ============================================================================
# Test Texts
# ============================================================================

def get_test_texts() -> list[str]:
    """Get sample test texts for compression comparison with full Vietnamese support."""
    return [
        # Vietnamese text - Geography
        """Việt Nam là một quốc gia nằm ở cực đông của bán đảo Đông Dương thuộc khu vực
        Đông Nam Á, phía bắc giáp Trung Quốc, phía tây giáp Lào và Campuchia, phía tây
        nam giáp vịnh Thái Lan, phía đông và phía nam giáp biển Đông.""",

        # Vietnamese - Capital city with all diacritics
        """Hà Nội là thủ đô, đồng thời là thành phố đứng đầu Việt Nam về diện tích
        và thứ hai về dân số với 8.053.663 người (2019). Nằm giữa đồng bằng sông Hồng
        trù phú, Hà Nội đã sớm trở thành trung tâm chính trị và tôn giáo.""",

        # Vietnamese - Complex diacritics test (all 6 tones)
        """Tiếng Việt có sáu thanh điệu: ngang (a), huyền (à), sắc (á), hỏi (ả),
        ngã (ã), và nặng (ạ). Ví dụ: "ma" (ghost), "mà" (but), "má" (mother),
        "mả" (grave), "mã" (horse), "mạ" (rice seedling).""",

        # Vietnamese - Literature with special vowels
        """Truyện Kiều của Nguyễn Du là kiệt tác văn học Việt Nam. Câu thơ nổi tiếng:
        "Trăm năm trong cõi người ta, Chữ tài chữ mệnh khéo là ghét nhau."
        Đây là tác phẩm thể hiện tinh hoa của tiếng Việt với đầy đủ thanh điệu.""",

        # Vietnamese - Technical terms
        """Trí tuệ nhân tạo (AI) đang phát triển mạnh mẽ. Các thuật toán học máy
        (machine learning) và học sâu (deep learning) được ứng dụng rộng rãi trong
        xử lý ngôn ngữ tự nhiên, thị giác máy tính, và nhiều lĩnh vực khác.""",

        # Vietnamese - Daily conversation
        """Xin chào! Bạn khỏe không? Hôm nay thời tiết đẹp quá! Chúng ta đi uống
        cà phê nhé? Quán cà phê ở đường Nguyễn Huệ rất ngon. Giá cả cũng phải chăng,
        khoảng 35.000đ một ly cà phê sữa đá.""",

        # Python code
        """def fibonacci(n: int) -> int:
    '''Calculate the nth Fibonacci number using memoization.'''
    if n <= 1:
        return n
    memo = {0: 0, 1: 1}
    for i in range(2, n + 1):
        memo[i] = memo[i-1] + memo[i-2]
    return memo[n]

# Test the function
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")""",

        # JavaScript code
        """async function fetchUserData(userId) {
    try {
        const response = await fetch(`/api/users/${userId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const userData = await response.json();
        return userData;
    } catch (error) {
        console.error('Failed to fetch user data:', error);
        throw error;
    }
}""",

        # Mixed Vietnamese and code - Complete diacritics
        """# Hàm tính giai thừa với đệ quy
def tinh_giai_thua(n):
    '''Tính n! = 1 × 2 × 3 × ... × n

    Ví dụ: 5! = 5 × 4 × 3 × 2 × 1 = 120
    '''
    if n <= 1:
        return 1
    return n * tinh_giai_thua(n - 1)

# Kiểm tra hàm
for i in range(1, 11):
    print(f"{i}! = {tinh_giai_thua(i)}")""",

        # Vietnamese reasoning trace format
        """<think>
[Bước 1] Đầu tiên, tôi cần hiểu đề bài: Tìm tổng các số nguyên tố nhỏ hơn 100.
[Bước 2] Số nguyên tố là số chỉ chia hết cho 1 và chính nó.
[Bước 3] Tôi sẽ duyệt qua các số từ 2 đến 99 và kiểm tra tính nguyên tố.
[Bước 4] Để tối ưu, tôi chỉ cần kiểm tra các ước số đến căn bậc hai của n.
</think>

Tổng các số nguyên tố nhỏ hơn 100 là 1060.

<code_start>
def la_so_nguyen_to(n):
    '''Kiểm tra số nguyên tố'''
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

tong = sum(n for n in range(2, 100) if la_so_nguyen_to(n))
print(f"Tổng = {tong}")  # Kết quả: 1060
<code_end>""",

        # Vietnamese with currency and numbers
        """Giá vàng hôm nay:
        - Vàng SJC: 79.500.000đ/lượng (mua vào), 81.000.000đ/lượng (bán ra)
        - Vàng nhẫn 9999: 78.200.000đ/lượng
        Tỷ giá USD: 24.850đ (mua), 25.220đ (bán)
        Bitcoin: $67.523,45 (≈ 1.680.000.000đ)""",
    ]


# ============================================================================
# Main CLI
# ============================================================================

def create_data_iterator(args) -> Iterator[str]:
    """Create a data iterator based on command line arguments."""
    # Local files have priority
    if args.wiki_files:
        logger.info(f"Loading Vietnamese text from: {args.wiki_files}")
        yield from iter_local_files(args.wiki_files)

    if args.code_files:
        logger.info(f"Loading code from: {args.code_files}")
        yield from iter_local_files(args.code_files)

    # If no local files, use HuggingFace datasets
    if not args.wiki_files and not args.code_files:
        if not args.no_wiki:
            yield from iter_vietnamese_wiki(
                cache_dir=args.cache_dir,
                max_samples=args.max_wiki_samples
            )

        if not args.no_code:
            yield from iter_github_code(
                languages=args.languages.split(",") if args.languages else CODE_LANGUAGES,
                cache_dir=args.cache_dir,
                max_samples_per_lang=args.max_code_per_lang
            )


def main():
    parser = argparse.ArgumentParser(
        description="Train a custom BPE tokenizer for vAGI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with HuggingFace datasets
    python core/nlp/train_tokenizer.py --output tokenizers/vagi

    # Train on local files
    python core/nlp/train_tokenizer.py \\
        --wiki-files "data/wiki/*.txt" \\
        --code-files "data/code/**/*.py" \\
        --output tokenizers/vagi

    # Compare with GPT-4 tokenizer
    python core/nlp/train_tokenizer.py --output tokenizers/vagi --compare-gpt4

    # Custom vocab size
    python core/nlp/train_tokenizer.py --vocab-size 50000 --output tokenizers/vagi_50k
        """
    )

    # Output settings
    parser.add_argument("--output", "-o", type=str, default="tokenizers/vagi_tokenizer",
                        help="Output directory for tokenizer files")

    # Vocabulary settings
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE,
                        help=f"Target vocabulary size (default: {DEFAULT_VOCAB_SIZE})")
    parser.add_argument("--min-frequency", type=int, default=2,
                        help="Minimum token frequency (default: 2)")

    # Data source settings
    parser.add_argument("--wiki-files", type=str, default=None,
                        help="Glob pattern for Vietnamese text files")
    parser.add_argument("--code-files", type=str, default=None,
                        help="Glob pattern for code files")
    parser.add_argument("--no-wiki", action="store_true",
                        help="Skip Vietnamese Wikipedia data")
    parser.add_argument("--no-code", action="store_true",
                        help="Skip GitHub code data")
    parser.add_argument("--languages", type=str, default=None,
                        help=f"Comma-separated list of programming languages (default: {','.join(CODE_LANGUAGES)})")

    # Dataset limits
    parser.add_argument("--max-wiki-samples", type=int, default=100000,
                        help="Maximum Vietnamese Wikipedia samples (default: 100000)")
    parser.add_argument("--max-code-per-lang", type=int, default=10000,
                        help="Maximum code samples per language (default: 10000)")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Cache directory for downloaded datasets")

    # Comparison
    parser.add_argument("--compare-gpt4", action="store_true",
                        help="Compare compression ratio with GPT-4 tokenizer")

    # Load existing tokenizer
    parser.add_argument("--load", type=str, default=None,
                        help="Load existing tokenizer (skip training)")

    args = parser.parse_args()

    # Create trainer
    trainer = VAGITokenizerTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency
    )

    if args.load:
        # Load existing tokenizer
        trainer.load(args.load)
    else:
        # Train new tokenizer
        logger.info("Preparing data iterator...")
        data_iter = create_data_iterator(args)

        logger.info("Starting tokenizer training...")
        trainer.train(data_iter, show_progress=True)

        # Save tokenizer
        trainer.save(args.output)

    # Run compression comparison
    if args.compare_gpt4 or args.load:
        test_texts = get_test_texts()
        compare_with_gpt4(trainer.tokenizer, test_texts)

    # Show tokenization examples
    print("\n" + "=" * 70)
    print("VÍ DỤ TOKENIZATION / TOKENIZATION EXAMPLES")
    print("=" * 70)

    examples = [
        # Basic Vietnamese with all tones
        "Xin chào thế giới! Hello world!",
        # Vietnamese with special vowels (ă, â, ê, ô, ơ, ư)
        "Việt Nam có dân số khoảng 100 triệu người.",
        # All 6 tones on 'a': a à á ả ã ạ
        "ma mà má mả mã mạ (6 thanh điệu)",
        # Complex Vietnamese sentence
        "Trường Đại học Bách khoa Hà Nội được thành lập năm 1956.",
        # Vietnamese with currency
        "Giá: 1.500.000đ (một triệu năm trăm nghìn đồng)",
        # Code with Vietnamese comments
        "def tính_tổng(a, b): # Hàm cộng hai số\n    return a + b",
        # Reasoning trace format
        "<think>[Bước 1] Phân tích bài toán...</think>",
        # Mixed content
        "AI (Trí tuệ nhân tạo) sử dụng deep learning để xử lý ngôn ngữ tự nhiên.",
    ]

    for text in examples:
        tokens = trainer.tokenizer.encode(text)
        decoded = trainer.tokenizer.decode(tokens.ids)
        print(f"\n📝 Original: {text}")
        print(f"🔢 Tokens ({len(tokens.ids)}): {tokens.ids[:20]}{'...' if len(tokens.ids) > 20 else ''}")
        print(f"✅ Decoded:  {decoded}")

    # Show Vietnamese character coverage
    print("\n" + "=" * 70)
    print("KIỂM TRA HỖ TRỢ TIẾNG VIỆT / VIETNAMESE SUPPORT CHECK")
    print("=" * 70)

    viet_test = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    viet_tokens = trainer.tokenizer.encode(viet_test)
    viet_decoded = trainer.tokenizer.decode(viet_tokens.ids)

    print(f"\nVietnamese chars: {viet_test}")
    print(f"Token count: {len(viet_tokens.ids)}")
    print(f"Decoded: {viet_decoded}")
    print(f"Match: {'✅ PASS' if viet_decoded.replace(' ', '') == viet_test else '❌ FAIL'}")

    print("\n" + "=" * 70)
    logger.info("Hoàn thành! / Done!")


if __name__ == "__main__":
    main()
