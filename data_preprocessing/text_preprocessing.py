"""Text preprocessing utilities for the project.

Provides:
- normalize_text(text): lowercasing, remove/normalize newlines, collapse whitespace
- remove_punctuation(text): remove common punctuation using regex
- tokenize(text, method='regex'|'whitespace'): return list of tokens
- remove_stopwords(tokens, stopwords=None): remove tokens based on built-in stopword lists
- preprocess(text, ...): convenience wrapper to do all steps in sequence

This module intentionally has no heavy external dependencies so it is easy to run in the
project environment.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set

# Simple built-in stopword lists to avoid heavy dependencies; extend as needed
_EN_STOPWORDS: Set[str] = {
    'a', 'an', 'the', 'and', 'or', 'not', 'but', 'if', 'then', 'else', 'for', 'in', 'on', 'at', 'by',
    'of', 'to', 'with', 'without', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'it', 'this', 'that',
    'these', 'those', 'there', 'here', 'from', 'which', 'we', 'you', 'he', 'she', 'they', 'them', 'his',
    'her', 'their', 'what', 'who', 'whom', 'how', 'why', 'when', 'where', 'do', 'did', 'does', 'have', 'has',
    'had', 'i', 'me', 'my', 'mine', 'our', 'ours', 'your', 'yours'
}

# Very small Vietnamese stopword list for demonstration. Extend as needed.
_VI_STOPWORDS: Set[str] = {
    'và', 'là', 'của', 'với', 'trong', 'một', 'những', 'các', 'cho', 'để', 'có', 'không', 'không', 'nhưng', 'vẫn',
    'đã', 'đang', 'sẽ', 'này', 'đó', 'tôi', 'mình', 'của', 'anh', 'chị', 'ông', 'bà'
}

# Regex for punctuation (we keep Unicode word/digit/underscore characters and whitespace)
# Python's `\w` with re.UNICODE covers unicode letters, digits and underscore; keep punctuation removal conservative.
_PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase text, replace newlines/tabs with spaces and collapse multiple spaces.

    Args:
        text: a raw text string
    Returns:
        a normalized string (lowercased and whitespace-normalized)
    """
    if text is None:
        return ''
    # Lowercase
    text = text.lower()
    # Normalize line breaks and tabs to spaces
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Collapse multiple whitespace
    text = _WS_RE.sub(' ', text).strip()
    return text


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters using a conservative Unicode-aware regex.

    Keeps letters (including many Latin letters with diacritics), digits, underscores and whitespace.
    """
    if not text:
        return ''
    return _PUNCTUATION_RE.sub(' ', text)


def tokenize(text: str, method: str = 'regex') -> List[str]:
    """Tokenize text into tokens.

    Supported methods:
        - 'regex' (default): split by any non-word character (after punctuation removal we can use whitespace)
        - 'whitespace': split by whitespace

    Returns:
        list of token strings (lowercased as an expectation by `normalize_text`)
    """
    if text is None:
        return []
    if method == 'whitespace':
        return [t for t in _WS_RE.split(text) if t]
    # default regex: split on whitespace — after cleaning punctuation regex we mostly have whitespace separation
    return [t for t in _WS_RE.split(text) if t]


def remove_stopwords(tokens: Iterable[str], language: str = 'en', custom_stopwords: Optional[Iterable[str]] = None) -> List[str]:
    """Remove stopwords from a list of tokens.

    Args:
        tokens: iterable of token strings
        language: 'en' or 'vi' (en default). If unknown, falls back to English.
        custom_stopwords: optional iterable of stopwords to add (union)
    Returns:
        list of tokens with stopwords removed
    """
    if tokens is None:
        return []
    base = set(custom_stopwords) if custom_stopwords else set()
    if language.lower().startswith('vi'):
        stopset = base.union(_VI_STOPWORDS)
    else:
        stopset = base.union(_EN_STOPWORDS)

    return [t for t in tokens if t and t not in stopset]


def preprocess(text: str,
               lowercase: bool = True,
               remove_punct: bool = True,
               tokenization: str = 'regex',
               remove_stopwords_flag: bool = True,
               stopwords_language: str = 'en',
               custom_stopwords: Optional[Iterable[str]] = None,
               return_tokens: bool = True) -> Optional[List[str] | str]:
    """Full convenience preprocessing pipeline.

    Steps applied in order:
        - normalize (lowercase + collapse whitespace)
        - remove punctuation (optional)
        - tokenize
        - remove stopwords (optional)

    Args:
        text: raw input text
        lowercase: apply lowercasing in normalize_text
        remove_punct: remove punctuation characters
        tokenization: 'regex' or 'whitespace'
        remove_stopwords_flag: if True, apply stopword removal
        stopwords_language: which stopword set to use (default 'en')
        custom_stopwords: additional stopwords to exclude
        return_tokens: if True return tokens; otherwise return the cleaned rejoined string
    Returns:
        tokens or cleaned text string
    """
    if text is None:
        return [] if return_tokens else ''

    # Step 1: normalize
    out = text if not lowercase else normalize_text(text)

    # Step 2: punctuation removal
    out = out if not remove_punct else remove_punctuation(out)
    out = _WS_RE.sub(' ', out).strip()

    # Step 3: tokenize
    tokens = tokenize(out, method=tokenization)

    # Step 4: remove stopwords
    if remove_stopwords_flag:
        tokens = remove_stopwords(tokens, language=stopwords_language, custom_stopwords=custom_stopwords)

    if return_tokens:
        return tokens
    else:
        return ' '.join(tokens)


if __name__ == '__main__':
    # Quick CLI demonstration
    s = """Patient is a 32-year-old male.
    He presented with abdominal pain and nausea, vital signs stable.
    Discharge diagnosis: Appendicitis."""
    print('Original: ', s)
    print('Tokens: ', preprocess(s))
