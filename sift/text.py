"""Text analysis for lexical retrieval: normalization, tokenization, stopwords
and a from-scratch implementation of the Porter (1980) stemming algorithm.

The analyzer mirrors Lucene's ``EnglishAnalyzer`` (possessive stripping,
lowercasing, the 33-word Lucene stop set, Porter stemming) so BM25 numbers are
comparable with the Anserini/Elasticsearch baselines reported in BEIR.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable

# Lucene's default English stop set (EnglishAnalyzer.ENGLISH_STOP_WORDS_SET).
LUCENE_STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the their "
    "then there these they this to was will with".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_POSSESSIVE_RE = re.compile(r"(\w)['’]s\b")


class PorterStemmer:
    """The original Porter stemming algorithm (M. F. Porter, 1980).

    A word is viewed as ``[C](VC){m}[V]`` where C/V are maximal consonant and
    vowel runs; ``m`` (the *measure*) gates most suffix rules. Each step is a
    list of ``(suffix, replacement, condition)`` rules where — as in the paper —
    only the rule with the longest matching suffix is considered: if its
    condition fails, the step stops.

    Validated against NLTK's ``PorterStemmer(mode=ORIGINAL_ALGORITHM)`` on the
    full SciFact + NFCorpus vocabulary (see ``tests/test_text.py``).
    """

    _VOWELS = frozenset("aeiou")

    def __init__(self, cache_size: int = 200_000) -> None:
        self._cache: dict[str, str] = {}
        self._cache_size = cache_size

    # ------------------------------------------------------------------ helpers
    def _is_consonant(self, word: str, i: int) -> bool:
        ch = word[i]
        if ch in self._VOWELS:
            return False
        if ch == "y":
            return i == 0 or not self._is_consonant(word, i - 1)
        return True

    def measure(self, stem: str) -> int:
        """Number of VC sequences in ``stem``."""
        m, prev_vowel = 0, False
        for i in range(len(stem)):
            vowel = not self._is_consonant(stem, i)
            if prev_vowel and not vowel:
                m += 1
            prev_vowel = vowel
        return m

    def _contains_vowel(self, stem: str) -> bool:
        return any(not self._is_consonant(stem, i) for i in range(len(stem)))

    def _ends_double_consonant(self, word: str) -> bool:
        return len(word) >= 2 and word[-1] == word[-2] and self._is_consonant(word, len(word) - 1)

    def _ends_cvc(self, word: str) -> bool:
        n = len(word)
        return (
            n >= 3
            and self._is_consonant(word, n - 3)
            and not self._is_consonant(word, n - 2)
            and self._is_consonant(word, n - 1)
            and word[-1] not in "wxy"
        )

    def _m_gt0(self, stem: str) -> bool:
        return self.measure(stem) > 0

    def _m_gt1(self, stem: str) -> bool:
        return self.measure(stem) > 1

    def _apply(self, word: str, rules: Iterable[tuple[str, str, Callable[[str], bool] | None]]) -> str:
        for suffix, replacement, condition in rules:
            if suffix == "*d":
                if not self._ends_double_consonant(word):
                    continue
                stem = word[:-2]
            elif word.endswith(suffix):
                stem = word[: len(word) - len(suffix)]
            else:
                continue
            return stem + replacement if condition is None or condition(stem) else word
        return word

    # -------------------------------------------------------------------- steps
    def _step1a(self, w: str) -> str:
        return self._apply(w, [("sses", "ss", None), ("ies", "i", None), ("ss", "ss", None), ("s", "", None)])

    def _step1b(self, w: str) -> str:
        if w.endswith("eed"):
            stem = w[:-3]
            return stem + "ee" if self.measure(stem) > 0 else w
        for suffix in ("ed", "ing"):
            if w.endswith(suffix) and self._contains_vowel(w[: -len(suffix)]):
                stem = w[: -len(suffix)]
                break
        else:
            return w
        last = stem[-1]
        return self._apply(
            stem,
            [
                ("at", "ate", None),
                ("bl", "ble", None),
                ("iz", "ize", None),
                ("*d", last, lambda _s: last not in "lsz"),
                ("", "e", lambda s: self.measure(s) == 1 and self._ends_cvc(s)),
            ],
        )

    def _step1c(self, w: str) -> str:
        return self._apply(w, [("y", "i", self._contains_vowel)])

    def _step2(self, w: str) -> str:
        c = self._m_gt0
        return self._apply(
            w,
            [
                ("ational", "ate", c), ("tional", "tion", c), ("enci", "ence", c), ("anci", "ance", c),
                ("izer", "ize", c), ("abli", "able", c), ("alli", "al", c), ("entli", "ent", c),
                ("eli", "e", c), ("ousli", "ous", c), ("ization", "ize", c), ("ation", "ate", c),
                ("ator", "ate", c), ("alism", "al", c), ("iveness", "ive", c), ("fulness", "ful", c),
                ("ousness", "ous", c), ("aliti", "al", c), ("iviti", "ive", c), ("biliti", "ble", c),
            ],
        )  # fmt: skip

    def _step3(self, w: str) -> str:
        c = self._m_gt0
        return self._apply(
            w,
            [
                ("icate", "ic", c), ("ative", "", c), ("alize", "al", c), ("iciti", "ic", c),
                ("ical", "ic", c), ("ful", "", c), ("ness", "", c),
            ],
        )  # fmt: skip

    def _step4(self, w: str) -> str:
        c = self._m_gt1
        return self._apply(
            w,
            [
                ("al", "", c), ("ance", "", c), ("ence", "", c), ("er", "", c), ("ic", "", c),
                ("able", "", c), ("ible", "", c), ("ant", "", c), ("ement", "", c), ("ment", "", c),
                ("ent", "", c), ("ion", "", lambda s: self.measure(s) > 1 and s[-1:] in ("s", "t")),
                ("ou", "", c), ("ism", "", c), ("ate", "", c), ("iti", "", c), ("ous", "", c),
                ("ive", "", c), ("ize", "", c),
            ],
        )  # fmt: skip

    def _step5a(self, w: str) -> str:
        if w.endswith("e"):
            stem = w[:-1]
            m = self.measure(stem)
            if m > 1 or (m == 1 and not self._ends_cvc(stem)):
                return stem
        return w

    def _step5b(self, w: str) -> str:
        if w.endswith("ll") and self.measure(w[:-1]) > 1:
            return w[:-1]
        return w

    # ---------------------------------------------------------------------- api
    def stem(self, word: str) -> str:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        w = word.lower()
        for step in (
            self._step1a, self._step1b, self._step1c, self._step2,
            self._step3, self._step4, self._step5a, self._step5b,
        ):  # fmt: skip
            w = step(w)
        if len(self._cache) < self._cache_size:
            self._cache[word] = w
        return w


def normalize(text: str) -> str:
    """NFKD-normalize, strip combining marks (accents) and lowercase."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


class Analyzer:
    """Text -> list of index terms.

    ``Analyzer()(text)`` is used both at index time and query time so the two
    always agree.
    """

    def __init__(
        self,
        stopwords: frozenset[str] = LUCENE_STOPWORDS,
        stem: bool = True,
        min_stem_len: int = 3,
    ) -> None:
        self.stopwords = stopwords
        self.stemmer = PorterStemmer() if stem else None
        # Stemming 1-2 letter tokens ("as" -> "a") only creates collisions.
        self.min_stem_len = min_stem_len

    def tokens(self, text: str) -> list[str]:
        text = _POSSESSIVE_RE.sub(r"\1", normalize(text))
        return _TOKEN_RE.findall(text)

    def __call__(self, text: str) -> list[str]:
        out: list[str] = []
        stemmer = self.stemmer
        for tok in self.tokens(text):
            if tok in self.stopwords:
                continue
            if stemmer is not None and len(tok) >= self.min_stem_len and not tok.isdigit():
                tok = stemmer.stem(tok)
            out.append(tok)
        return out


# --------------------------------------------------------------------- sentences
_ABBREVIATIONS = frozenset(
    "e.g i.e et al fig figs eq eqs vs approx dr mr mrs ms prof inc ltd co no vol pp ca cf resp".split()
)
_SENT_END_RE = re.compile(r"[.!?]+[\"')\]]*(?=\s+)")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Split ``text`` into sentences, returning ``(start, end)`` char spans.

    Spans cover the text exactly (modulo leading whitespace), so
    ``text[start:end]`` is always a faithful substring — a requirement for
    exact-span citations. Handles common abbreviations, decimals (``0.05``)
    and initials.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    n = len(text)
    while start < n and text[start].isspace():
        start += 1
    for match in _SENT_END_RE.finditer(text):
        end = match.end()
        prev = text[start:match.start()].rsplit(None, 1)
        last_word = prev[-1].lower().strip("([{\"'").rstrip(".") if prev else ""
        nxt = end
        while nxt < n and text[nxt].isspace():
            nxt += 1
        if nxt >= n:
            break
        if last_word in _ABBREVIATIONS or (len(last_word) == 1 and last_word.isalpha()):
            continue
        # Next sentence should begin with an uppercase letter, digit or bracket.
        if not (text[nxt].isupper() or text[nxt].isdigit() or text[nxt] in "([\"'"):
            continue
        if end > start:
            spans.append((start, end))
        start = nxt
    if start < n:
        end = n
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((start, end))
    return spans
