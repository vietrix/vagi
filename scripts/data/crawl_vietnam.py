#!/usr/bin/env python3
"""
Vietnamese Tech Article Scraper for vAGI Training Data.

Scrapes high-quality tech articles from Vietnamese tech blogs
and formats them for instruction-tuning.

Features:
- Supports multiple Vietnamese tech sites (Viblo-style structure)
- Extracts title, content, and code blocks
- Removes ads, navigation, and boilerplate
- Outputs in instruction-tuning JSONL format
- Rate limiting to be respectful to servers
- Resume support for interrupted crawls

Requirements:
    pip install beautifulsoup4 requests lxml playwright aiohttp

Usage:
    # Scrape with default settings
    python scripts/data/crawl_vietnam.py --output data/vietnamese_tech.jsonl

    # Scrape specific pages
    python scripts/data/crawl_vietnam.py --max-pages 100 --output data/vn_tech.jsonl

    # Use Playwright for JavaScript-rendered sites
    python scripts/data/crawl_vietnam.py --use-playwright --output data/vn_tech.jsonl

Output Format:
    {"instruction": "Title", "input": "", "output": "Content with code blocks"}
"""

import argparse
import json
import os
import re
import sys
import time
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Generator, Set
from urllib.parse import urljoin, urlparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try importing required packages
try:
    import requests
    from bs4 import BeautifulSoup, NavigableString, Tag
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.error("Missing required packages. Install with: pip install beautifulsoup4 requests lxml")

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Article:
    """Scraped article data."""
    url: str
    title: str
    content: str
    code_blocks: List[Dict[str, str]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    author: Optional[str] = None
    published_date: Optional[str] = None
    views: Optional[int] = None

    def to_instruction_format(self) -> Dict[str, str]:
        """Convert to instruction-tuning format."""
        # Combine content with code blocks
        full_content = self.content

        # Insert code blocks back into content at appropriate places
        for block in self.code_blocks:
            lang = block.get('language', 'python')
            code = block.get('code', '')
            # Add code block in markdown format
            if code not in full_content:
                full_content += f"\n\n```{lang}\n{code}\n```"

        return {
            "instruction": self.title,
            "input": "",
            "output": full_content.strip(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to full dictionary."""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "code_blocks": self.code_blocks,
            "tags": self.tags,
            "author": self.author,
            "published_date": self.published_date,
            "views": self.views,
        }


@dataclass
class SiteConfig:
    """Configuration for a specific site."""
    name: str
    base_url: str
    list_url_pattern: str  # Pattern for article list pages
    article_selectors: Dict[str, str]  # CSS selectors for article parts
    list_selectors: Dict[str, str]  # CSS selectors for list pages
    rate_limit: float = 1.0  # Seconds between requests
    needs_javascript: bool = False


# =============================================================================
# Site Configurations
# =============================================================================

# Generic Viblo-style configuration
VIBLO_CONFIG = SiteConfig(
    name="viblo",
    base_url="https://viblo.asia",
    list_url_pattern="/newest?page={page}",
    article_selectors={
        "title": "h1.article-title, h1.post-title, .article-header h1",
        "content": ".article-content, .post-content, .md-contents, article .content",
        "code": "pre code, .highlight code, pre.highlight",
        "author": ".author-name, .user-name, .article-author a",
        "date": ".post-date, .article-date, time",
        "tags": ".tags a, .tag-list a, .article-tags a",
    },
    list_selectors={
        "articles": ".post-item a, .article-item a, .feed-item a.link",
        "title": ".post-title, .article-title, h3",
    },
    rate_limit=2.0,
)

# Generic tech blog configuration (simpler structure)
GENERIC_BLOG_CONFIG = SiteConfig(
    name="generic",
    base_url="",
    list_url_pattern="/page/{page}",
    article_selectors={
        "title": "h1, .entry-title, .post-title",
        "content": "article, .entry-content, .post-content, .content",
        "code": "pre code, pre, .highlight",
        "author": ".author, .byline, .post-author",
        "date": ".date, time, .post-date",
        "tags": ".tags a, .categories a",
    },
    list_selectors={
        "articles": "article a, .post a, .entry a",
        "title": "h2, h3, .entry-title",
    },
    rate_limit=1.5,
)


# =============================================================================
# Content Cleaner
# =============================================================================

class ContentCleaner:
    """Cleans and normalizes article content."""

    # Elements to remove completely
    REMOVE_TAGS = [
        'script', 'style', 'nav', 'header', 'footer', 'aside',
        'form', 'button', 'input', 'select', 'iframe', 'noscript',
        'svg', 'canvas', '.ad', '.advertisement', '.social-share',
        '.comments', '.related-posts', '.sidebar', '.navigation',
        '.breadcrumb', '.pagination', '.author-bio', '.share-buttons',
    ]

    # Patterns for ad/navigation text
    AD_PATTERNS = [
        r'Quảng cáo',
        r'Advertisement',
        r'Sponsored',
        r'Xem thêm:',
        r'Bài viết liên quan',
        r'Related posts',
        r'Follow us',
        r'Subscribe',
        r'Share this',
        r'Chia sẻ',
        r'Đăng ký',
        r'Newsletter',
    ]

    def __init__(self):
        self.ad_pattern = re.compile('|'.join(self.AD_PATTERNS), re.IGNORECASE)

    def clean_html(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove unwanted elements from soup."""
        # Remove unwanted tags
        for selector in self.REMOVE_TAGS:
            for element in soup.select(selector):
                element.decompose()

        # Remove hidden elements
        for element in soup.find_all(attrs={'style': re.compile(r'display\s*:\s*none', re.I)}):
            element.decompose()

        for element in soup.find_all(attrs={'hidden': True}):
            element.decompose()

        return soup

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""

        # Remove ad patterns
        text = self.ad_pattern.sub('', text)

        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\t+', ' ', text)

        # Remove empty lines with only whitespace
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        text = '\n'.join(lines)

        # Remove common boilerplate phrases
        boilerplate = [
            r'^\s*Bình luận\s*$',
            r'^\s*Comments?\s*$',
            r'^\s*\d+\s*views?\s*$',
            r'^\s*Loading\.*\s*$',
            r'^\s*Đang tải\.*\s*$',
        ]
        for pattern in boilerplate:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)

        return text.strip()

    def extract_code_blocks(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract code blocks from HTML."""
        code_blocks = []
        seen_codes = set()

        # Find code elements
        for code_elem in soup.select('pre code, pre.highlight, .highlight pre, pre'):
            code_text = code_elem.get_text().strip()

            if not code_text or len(code_text) < 10:
                continue

            # Deduplicate
            code_hash = hashlib.md5(code_text.encode()).hexdigest()
            if code_hash in seen_codes:
                continue
            seen_codes.add(code_hash)

            # Detect language
            language = self._detect_language(code_elem, code_text)

            code_blocks.append({
                'language': language,
                'code': code_text,
            })

        return code_blocks

    def _detect_language(self, element: Tag, code: str) -> str:
        """Detect programming language from element classes or content."""
        # Check class names
        classes = element.get('class', [])
        if isinstance(classes, str):
            classes = classes.split()

        for cls in classes:
            cls_lower = cls.lower()
            if 'python' in cls_lower or cls_lower == 'py':
                return 'python'
            elif 'javascript' in cls_lower or cls_lower == 'js':
                return 'javascript'
            elif 'typescript' in cls_lower or cls_lower == 'ts':
                return 'typescript'
            elif 'java' in cls_lower and 'javascript' not in cls_lower:
                return 'java'
            elif 'ruby' in cls_lower or cls_lower == 'rb':
                return 'ruby'
            elif 'php' in cls_lower:
                return 'php'
            elif 'go' in cls_lower or cls_lower == 'golang':
                return 'go'
            elif 'rust' in cls_lower or cls_lower == 'rs':
                return 'rust'
            elif 'sql' in cls_lower:
                return 'sql'
            elif 'bash' in cls_lower or 'shell' in cls_lower or cls_lower == 'sh':
                return 'bash'
            elif 'html' in cls_lower:
                return 'html'
            elif 'css' in cls_lower:
                return 'css'
            elif 'json' in cls_lower:
                return 'json'
            elif 'yaml' in cls_lower or 'yml' in cls_lower:
                return 'yaml'

        # Heuristic detection from content
        if re.search(r'^\s*def\s+\w+\s*\(', code, re.MULTILINE):
            return 'python'
        elif re.search(r'^\s*function\s+\w+\s*\(|=>\s*\{|const\s+\w+\s*=', code, re.MULTILINE):
            return 'javascript'
        elif re.search(r'^\s*public\s+class|^\s*import\s+java\.', code, re.MULTILINE):
            return 'java'
        elif re.search(r'^\s*package\s+main|^\s*func\s+\w+\(', code, re.MULTILINE):
            return 'go'
        elif re.search(r'^\s*<\?php', code):
            return 'php'
        elif re.search(r'^\s*fn\s+\w+\(|^\s*use\s+std::', code, re.MULTILINE):
            return 'rust'
        elif re.search(r'^\s*SELECT|^\s*INSERT|^\s*UPDATE|^\s*CREATE', code, re.IGNORECASE | re.MULTILINE):
            return 'sql'
        elif re.search(r'^\s*<html|^\s*<!DOCTYPE', code, re.IGNORECASE):
            return 'html'
        elif re.search(r'^\s*\{|\s*"[\w]+"\s*:', code):
            return 'json'

        return 'text'


# =============================================================================
# Web Scraper
# =============================================================================

class TechArticleScraper:
    """Scrapes tech articles from Vietnamese websites."""

    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    ]

    def __init__(
        self,
        config: SiteConfig,
        use_playwright: bool = False,
        output_file: Optional[str] = None,
    ):
        self.config = config
        self.use_playwright = use_playwright and PLAYWRIGHT_AVAILABLE
        self.output_file = output_file
        self.cleaner = ContentCleaner()
        self.session = requests.Session() if REQUESTS_AVAILABLE else None
        self.scraped_urls: Set[str] = set()
        self.articles_count = 0

        # Load previously scraped URLs if output file exists
        if output_file and os.path.exists(output_file):
            self._load_scraped_urls()

    def _load_scraped_urls(self):
        """Load URLs from existing output file to avoid duplicates."""
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if 'url' in data:
                        self.scraped_urls.add(data['url'])
            logger.info(f"Loaded {len(self.scraped_urls)} previously scraped URLs")
        except Exception as e:
            logger.warning(f"Could not load previous URLs: {e}")

    def _get_random_user_agent(self) -> str:
        return random.choice(self.USER_AGENTS)

    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch page content."""
        if self.use_playwright:
            return self._fetch_with_playwright(url)
        return self._fetch_with_requests(url)

    def _fetch_with_requests(self, url: str) -> Optional[str]:
        """Fetch page with requests library."""
        if not self.session:
            return None

        try:
            headers = {
                'User-Agent': self._get_random_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
            }

            response = self.session.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or 'utf-8'
            return response.text

        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def _fetch_with_playwright(self, url: str) -> Optional[str]:
        """Fetch page with Playwright for JavaScript-rendered content."""
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent=self._get_random_user_agent(),
                    locale='vi-VN',
                )
                page = context.new_page()
                page.goto(url, wait_until='networkidle', timeout=30000)

                # Wait for content to load
                page.wait_for_timeout(2000)

                content = page.content()
                browser.close()
                return content

        except Exception as e:
            logger.error(f"Playwright fetch failed for {url}: {e}")
            return None

    def _parse_article(self, url: str, html: str) -> Optional[Article]:
        """Parse article from HTML."""
        soup = BeautifulSoup(html, 'lxml')

        # Clean HTML
        soup = self.cleaner.clean_html(soup)

        # Extract title
        title_elem = soup.select_one(self.config.article_selectors['title'])
        if not title_elem:
            logger.warning(f"No title found for {url}")
            return None

        title = self.cleaner.clean_text(title_elem.get_text())
        if not title or len(title) < 5:
            return None

        # Extract content
        content_elem = soup.select_one(self.config.article_selectors['content'])
        if not content_elem:
            logger.warning(f"No content found for {url}")
            return None

        # Extract code blocks before cleaning content
        code_blocks = self.cleaner.extract_code_blocks(content_elem)

        # Get text content
        content = self.cleaner.clean_text(content_elem.get_text())
        if not content or len(content) < 100:
            logger.warning(f"Content too short for {url}")
            return None

        # Extract optional fields
        author = None
        author_elem = soup.select_one(self.config.article_selectors.get('author', ''))
        if author_elem:
            author = self.cleaner.clean_text(author_elem.get_text())

        date = None
        date_elem = soup.select_one(self.config.article_selectors.get('date', ''))
        if date_elem:
            date = self.cleaner.clean_text(date_elem.get_text())

        tags = []
        tag_selector = self.config.article_selectors.get('tags', '')
        if tag_selector:
            for tag_elem in soup.select(tag_selector):
                tag_text = self.cleaner.clean_text(tag_elem.get_text())
                if tag_text:
                    tags.append(tag_text)

        return Article(
            url=url,
            title=title,
            content=content,
            code_blocks=code_blocks,
            tags=tags,
            author=author,
            published_date=date,
        )

    def _get_article_urls(self, list_page_html: str) -> List[str]:
        """Extract article URLs from list page."""
        soup = BeautifulSoup(list_page_html, 'lxml')
        urls = []

        selector = self.config.list_selectors['articles']
        for link in soup.select(selector):
            href = link.get('href')
            if href:
                full_url = urljoin(self.config.base_url, href)
                if full_url not in self.scraped_urls:
                    urls.append(full_url)

        return urls

    def scrape_articles(
        self,
        max_pages: int = 10,
        max_articles: Optional[int] = None,
    ) -> Generator[Article, None, None]:
        """
        Scrape articles from the configured site.

        Args:
            max_pages: Maximum list pages to crawl
            max_articles: Maximum articles to scrape (None for unlimited)

        Yields:
            Article objects
        """
        articles_scraped = 0

        for page_num in range(1, max_pages + 1):
            if max_articles and articles_scraped >= max_articles:
                break

            # Fetch list page
            list_url = self.config.base_url + self.config.list_url_pattern.format(page=page_num)
            logger.info(f"Fetching list page {page_num}: {list_url}")

            list_html = self._fetch_page(list_url)
            if not list_html:
                logger.warning(f"Could not fetch list page {page_num}")
                continue

            # Get article URLs
            article_urls = self._get_article_urls(list_html)
            logger.info(f"Found {len(article_urls)} articles on page {page_num}")

            if not article_urls:
                logger.info("No more articles found, stopping")
                break

            # Scrape each article
            for article_url in article_urls:
                if max_articles and articles_scraped >= max_articles:
                    break

                if article_url in self.scraped_urls:
                    continue

                logger.info(f"Scraping article: {article_url}")

                # Rate limiting
                time.sleep(self.config.rate_limit + random.uniform(0, 1))

                article_html = self._fetch_page(article_url)
                if not article_html:
                    continue

                article = self._parse_article(article_url, article_html)
                if article:
                    self.scraped_urls.add(article_url)
                    articles_scraped += 1
                    self.articles_count += 1
                    yield article

            # Rate limit between pages
            time.sleep(self.config.rate_limit * 2)

        logger.info(f"Scraped {articles_scraped} articles total")


# =============================================================================
# Output Writer
# =============================================================================

class DatasetWriter:
    """Writes scraped articles to JSONL file."""

    def __init__(self, output_path: str, append: bool = True):
        self.output_path = output_path
        self.append = append
        self.count = 0

        # Create directory if needed
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    def write_article(self, article: Article):
        """Write single article to file."""
        mode = 'a' if self.append else 'w'

        with open(self.output_path, mode, encoding='utf-8') as f:
            # Write instruction format
            data = article.to_instruction_format()
            data['url'] = article.url  # Keep URL for reference
            data['tags'] = article.tags

            f.write(json.dumps(data, ensure_ascii=False) + '\n')
            self.count += 1

    def write_articles(self, articles: Generator[Article, None, None]) -> int:
        """Write all articles from generator."""
        for article in articles:
            self.write_article(article)
            logger.info(f"Saved article: {article.title[:50]}...")

        return self.count


# =============================================================================
# Mock Scraper for Testing
# =============================================================================

class MockScraper:
    """Mock scraper that generates sample data for testing."""

    SAMPLE_ARTICLES = [
        {
            "title": "Hướng dẫn Python cho người mới bắt đầu",
            "content": """Python là ngôn ngữ lập trình dễ học và mạnh mẽ.

Để bắt đầu, hãy cài đặt Python từ trang chính thức.

Sau đó, bạn có thể viết chương trình đầu tiên:

```python
print("Xin chào, thế giới!")
```

Python có cú pháp đơn giản, dễ đọc. Các đặc điểm nổi bật:

1. Dễ học cho người mới
2. Hỗ trợ nhiều paradigm lập trình
3. Cộng đồng lớn và nhiều thư viện

Ví dụ về vòng lặp:

```python
for i in range(5):
    print(f"Số: {i}")
```

Hãy thử và khám phá thêm!""",
            "tags": ["python", "tutorial", "beginner"],
        },
        {
            "title": "Xây dựng REST API với FastAPI",
            "content": """FastAPI là framework Python hiện đại để xây dựng API.

Cài đặt FastAPI:

```bash
pip install fastapi uvicorn
```

Tạo API đơn giản:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Xin chào từ FastAPI"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

Chạy server:

```bash
uvicorn main:app --reload
```

FastAPI tự động tạo documentation OpenAPI tại /docs.""",
            "tags": ["python", "fastapi", "api"],
        },
        {
            "title": "Machine Learning cơ bản với PyTorch",
            "content": """PyTorch là thư viện deep learning phổ biến.

Cài đặt:

```bash
pip install torch torchvision
```

Tạo neural network đơn giản:

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNet()
print(model)
```

PyTorch có nhiều ưu điểm cho nghiên cứu AI.""",
            "tags": ["python", "pytorch", "machine-learning"],
        },
    ]

    def scrape_articles(self, max_articles: int = 10) -> Generator[Article, None, None]:
        """Generate mock articles."""
        for i, data in enumerate(self.SAMPLE_ARTICLES[:max_articles]):
            yield Article(
                url=f"https://example.com/article/{i}",
                title=data["title"],
                content=data["content"],
                code_blocks=[],
                tags=data["tags"],
                author="Test Author",
            )
            time.sleep(0.1)  # Simulate network delay


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Vietnamese Tech Article Scraper")

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/vietnamese_tech_instruction.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Maximum list pages to crawl",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Maximum articles to scrape",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://viblo.asia",
        help="Base URL of the site to scrape",
    )
    parser.add_argument(
        "--use-playwright",
        action="store_true",
        help="Use Playwright for JavaScript-rendered sites",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock data for testing",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=2.0,
        help="Seconds between requests",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Vietnamese Tech Article Scraper")
    print("=" * 60)
    print(f"Output: {args.output}")
    print(f"Max pages: {args.max_pages}")
    print(f"Max articles: {args.max_articles or 'unlimited'}")
    print(f"Base URL: {args.base_url}")
    print(f"Use Playwright: {args.use_playwright}")
    print(f"Mock mode: {args.mock}")
    print()

    # Create writer
    writer = DatasetWriter(args.output, append=True)

    if args.mock:
        # Use mock scraper for testing
        scraper = MockScraper()
        count = writer.write_articles(scraper.scrape_articles(args.max_articles or 10))
    else:
        # Check dependencies
        if not REQUESTS_AVAILABLE:
            print("Error: requests and beautifulsoup4 required")
            print("Install with: pip install beautifulsoup4 requests lxml")
            sys.exit(1)

        if args.use_playwright and not PLAYWRIGHT_AVAILABLE:
            print("Warning: Playwright not available, falling back to requests")
            args.use_playwright = False

        # Create config based on URL
        config = VIBLO_CONFIG
        config.base_url = args.base_url
        config.rate_limit = args.rate_limit

        # Create scraper
        scraper = TechArticleScraper(
            config=config,
            use_playwright=args.use_playwright,
            output_file=args.output,
        )

        # Scrape and write
        count = writer.write_articles(
            scraper.scrape_articles(
                max_pages=args.max_pages,
                max_articles=args.max_articles,
            )
        )

    print()
    print("=" * 60)
    print(f"Scraping complete!")
    print(f"Total articles saved: {count}")
    print(f"Output file: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
