"""
Malayalam Corpus Loader
Downloads and manages Malayalam word corpus from multiple sources.
"""

import json
import re
import time
import random
from pathlib import Path
from typing import List, Set, Dict
import requests

CORPUS_DIR = Path(__file__).parent / "corpus"
CORPUS_FILE = CORPUS_DIR / "malayalam_words.json"


def is_malayalam_word(text: str) -> bool:
    """Check if text contains only Malayalam characters and common punctuation."""
    if not text or len(text) < 1:
        return False

    # Malayalam Unicode range: U+0D00 to U+0D7F
    malayalam_pattern = re.compile(r'^[\u0D00-\u0D7F\u200C\u200D]+$')
    return bool(malayalam_pattern.match(text))


def fetch_wikipedia_titles(num_pages: int = 500) -> Set[str]:
    """Fetch random Malayalam Wikipedia article titles."""
    words = set()
    url = "https://ml.wikipedia.org/w/api.php"
    headers = {"User-Agent": "MalayalamHandwriting/1.0"}

    batch_size = 50  # API limit
    batches = (num_pages // batch_size) + 1

    print(f"Fetching {num_pages} Wikipedia titles...")

    for i in range(batches):
        params = {
            "action": "query",
            "list": "random",
            "rnnamespace": 0,
            "rnlimit": batch_size,
            "format": "json"
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.ok:
                data = response.json()
                titles = [p['title'] for p in data.get('query', {}).get('random', [])]

                for title in titles:
                    # Extract individual words from titles
                    for word in title.split():
                        word = word.strip('()[]{}.,;:!?"\'-')
                        if is_malayalam_word(word) and 1 <= len(word) <= 20:
                            words.add(word)

            time.sleep(0.5)  # Be nice to API

        except Exception as e:
            print(f"  Error in batch {i}: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"  Fetched {i + 1}/{batches} batches, {len(words)} unique words so far")

    return words


def fetch_wikipedia_content_words(num_articles: int = 100) -> Set[str]:
    """Fetch words from Wikipedia article content."""
    words = set()
    url = "https://ml.wikipedia.org/w/api.php"
    headers = {"User-Agent": "MalayalamHandwriting/1.0"}

    print(f"Fetching content from {num_articles} Wikipedia articles...")

    # First get random page IDs
    params = {
        "action": "query",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": min(num_articles, 50),
        "format": "json"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.ok:
            data = response.json()
            page_ids = [str(p['id']) for p in data.get('query', {}).get('random', [])]

            # Fetch content for each page
            for i, page_id in enumerate(page_ids[:num_articles]):
                content_params = {
                    "action": "query",
                    "pageids": page_id,
                    "prop": "extracts",
                    "explaintext": True,
                    "exintro": True,
                    "format": "json"
                }

                try:
                    resp = requests.get(url, params=content_params, headers=headers, timeout=10)
                    if resp.ok:
                        content_data = resp.json()
                        pages = content_data.get('query', {}).get('pages', {})
                        for page in pages.values():
                            text = page.get('extract', '')
                            # Extract words
                            for word in re.split(r'\s+', text):
                                word = word.strip('()[]{}.,;:!?"\'-«»')
                                if is_malayalam_word(word) and 2 <= len(word) <= 15:
                                    words.add(word)

                    time.sleep(0.3)

                except Exception as e:
                    continue

                if (i + 1) % 20 == 0:
                    print(f"  Processed {i + 1}/{len(page_ids)} articles, {len(words)} words")

    except Exception as e:
        print(f"Error fetching articles: {e}")

    return words


def categorize_words(words: Set[str]) -> Dict[str, List[str]]:
    """Categorize words by complexity/length."""
    categorized = {
        "short": [],      # 1-2 chars
        "simple": [],     # 3-4 chars
        "medium": [],     # 5-7 chars
        "long": [],       # 8-10 chars
        "complex": [],    # 11+ chars
    }

    for word in words:
        length = len(word)
        if length <= 2:
            categorized["short"].append(word)
        elif length <= 4:
            categorized["simple"].append(word)
        elif length <= 7:
            categorized["medium"].append(word)
        elif length <= 10:
            categorized["long"].append(word)
        else:
            categorized["complex"].append(word)

    # Shuffle each category
    for cat in categorized:
        random.shuffle(categorized[cat])

    return categorized


def download_corpus(num_titles: int = 1000, num_articles: int = 50) -> Dict:
    """Download and build Malayalam corpus."""
    CORPUS_DIR.mkdir(exist_ok=True)

    all_words = set()

    # Fetch from Wikipedia titles
    title_words = fetch_wikipedia_titles(num_titles)
    all_words.update(title_words)
    print(f"Got {len(title_words)} words from titles")

    # Fetch from article content
    content_words = fetch_wikipedia_content_words(num_articles)
    all_words.update(content_words)
    print(f"Got {len(content_words)} words from content")

    print(f"\nTotal unique words: {len(all_words)}")

    # Categorize
    categorized = categorize_words(all_words)

    # Add metadata
    corpus = {
        "metadata": {
            "source": "Malayalam Wikipedia",
            "total_words": len(all_words),
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "categories": categorized,
        "all_words": sorted(list(all_words))
    }

    # Save
    with open(CORPUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"\nSaved corpus to {CORPUS_FILE}")
    print(f"Categories: {', '.join(f'{k}: {len(v)}' for k, v in categorized.items())}")

    return corpus


def load_corpus() -> Dict:
    """Load corpus from file, download if not exists."""
    if not CORPUS_FILE.exists():
        print("Corpus not found, downloading...")
        return download_corpus()

    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_random_words(category: str = "all", count: int = 10) -> List[str]:
    """Get random words from corpus."""
    corpus = load_corpus()

    if category == "all":
        words = corpus.get("all_words", [])
    else:
        words = corpus.get("categories", {}).get(category, [])

    if not words:
        return []

    return random.sample(words, min(count, len(words)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Malayalam Corpus Loader")
    parser.add_argument("--download", action="store_true", help="Download fresh corpus")
    parser.add_argument("--titles", type=int, default=1000, help="Number of Wikipedia titles to fetch")
    parser.add_argument("--articles", type=int, default=50, help="Number of articles to fetch content from")
    parser.add_argument("--stats", action="store_true", help="Show corpus statistics")

    args = parser.parse_args()

    if args.download:
        download_corpus(args.titles, args.articles)
    elif args.stats:
        corpus = load_corpus()
        print(f"Corpus Statistics:")
        print(f"  Total words: {corpus['metadata']['total_words']}")
        print(f"  Downloaded: {corpus['metadata']['downloaded_at']}")
        print(f"  Categories:")
        for cat, words in corpus['categories'].items():
            print(f"    {cat}: {len(words)} words")
    else:
        # Show sample words
        print("Sample words from corpus:")
        for cat in ["short", "simple", "medium", "long", "complex"]:
            words = get_random_words(cat, 5)
            print(f"  {cat}: {', '.join(words)}")
