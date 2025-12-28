import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class PreprocessorConfig:
    min_length: int = 15
    max_length: int = 8000
    include_category: bool = True
    include_title: bool = True

class ReviewPreprocessor:
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')
    WHITESPACE_PATTERN = re.compile(r'\s+')
    HTML_TAG_PATTERN = re.compile(r'<[^>]+>')

    HTML_ENTITIES = {
        '&amp;': '&', '&lt;': '<', '&gt;': '>',
        '&quot;': '"', '&#39;': "'", '&nbsp;': ' ',
        'â€™': "'", 'â€œ': '"', 'â€': '"',
        'â€"': '—', 'â€"': '–', '&ndash;': '–',
        '&mdash;': '—', '&hellip;': '...'
    }

    TEXT_COLUMNS = ['text', 'reviewText', 'review_text', 'body']
    TITLE_COLUMNS = ['title', 'summary', 'review_title', 'headline']
    RATING_COLUMNS = ['rating', 'overall', 'score', 'stars']
    CATEGORY_COLUMNS = ['category', 'main_category', 'product_category']

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config or PreprocessorConfig()

    def _get_field(self, row: Dict[str, Any], cols, default=None):
        for c in cols:
            if c in row and row[c] is not None:
                return row[c]
        return default

    def clean_text(self, text: str) -> Optional[str]:
        if not text or not isinstance(text, str):
            return None
        text = text.strip()
        if not text:
            return None

        text = unicodedata.normalize("NFKC", text)
        text = self.HTML_TAG_PATTERN.sub(" ", text)
        for k, v in self.HTML_ENTITIES.items():
            text = text.replace(k, v)

        text = self.URL_PATTERN.sub("[URL]", text)
        text = self.EMAIL_PATTERN.sub("[EMAIL]", text)
        text = self.WHITESPACE_PATTERN.sub(" ", text).strip()

        if len(text) < self.config.min_length:
            return None
        if len(text) > self.config.max_length:
            text = text[: self.config.max_length]
        return text

    def format_input(self, text: str, title: Optional[str] = None, category: Optional[str] = None) -> str:
        parts = []
        if self.config.include_category and category:
            cat = str(category).replace("_", " ").strip()
            if cat:
                parts.append(f"[Category: {cat}]")
        if self.config.include_title and title and isinstance(title, str):
            t = title.strip()
            if len(t) > 2:
                parts.append(t)
        parts.append(text)
        return " ".join(parts)

    def process_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw_text = self._get_field(row, self.TEXT_COLUMNS, "")
        text = self.clean_text(raw_text)
        if text is None:
            return None

        rating = self._get_field(row, self.RATING_COLUMNS)
        if rating is None:
            return None

        try:
            rating = int(float(rating))
            if not (1 <= rating <= 5):
                return None
        except Exception:
            return None

        title = self._get_field(row, self.TITLE_COLUMNS, "")
        category = self._get_field(row, self.CATEGORY_COLUMNS, "")
        formatted = self.format_input(text, title, category)

        return {"text": formatted, "label": rating - 1}
