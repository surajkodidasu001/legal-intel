"""NER / information extraction.

Combines pretrained spaCy (en_core_web_sm) for general entities with two
deterministic regex rules for patterns Phase 3 profiling
(scripts/explore_entities_scale.py, validated on a 1000-doc LEDGAR sample)
showed are worth extracting reliably and that spaCy does NOT reliably tag as
a distinct category:

  GOVERNING_LAW  -- "State of X" / "State of New York" (multi-word states
                    handled, case-insensitive so ALL-CAPS clauses match too --
                    see below). Validated externally: Delaware and New York
                    dominate the extracted counts, matching known real-world
                    concentration of US corporate governing-law clauses.
  DEFINED_TERM   -- "(the "X")" / "(this "X")" contract-defined-term
                    declarations, tolerant of whitespace padding inside the
                    quotes (see below). Rare at the individual-provision
                    level in LEDGAR specifically (~2.7% of docs) because
                    LEDGAR samples individual clauses, not whole contracts,
                    and definitions mostly live in the recitals section that
                    downstream clauses don't repeat -- not because the
                    pattern itself is rare in contracts generally.

The original versions of both regexes were validated for PRECISION against
1000 real docs (every match they produced was correct) but not RECALL
(whether they were missing real cases) -- those are different questions,
and only checking the first one is a known trap. A manual review pass over
a second, separate 70-doc batch (generating candidates with these rules,
then having a human check both the candidates AND a sample of the "no
candidates" docs for misses) found two real recall gaps, now fixed:
  1. GOVERNING_LAW missed ALL-CAPS clauses (e.g. "STATE OF NEW YORK") --
     contracts often put governing-law/arbitration clauses in all-caps for
     legal conspicuousness requirements. Fixed with re.IGNORECASE.
  2. DEFINED_TERM missed declarations with whitespace padding directly
     inside the quotes (e.g. '(the " Moving Party ")', likely a document-
     conversion artifact) -- the pattern required the captured text to
     start immediately after the quote mark. Fixed by allowing \\s* inside
     the quotes and trimming it out of the capture.
See tests/test_ner_extract.py for regression tests covering both.

Why rules instead of fine-tuning spaCy for these: both patterns are
near-deterministic in contract boilerplate, and the 15-doc spot check
(Phase 3 exploration) showed spaCy's own GPE/ORG/PERSON/PRODUCT tags
misclassify exactly these spans (e.g. "Guarantor" tagged PRODUCT, "Tax"
tagged PERSON) rather than recognizing them as a coherent category. A
rule-based extractor gets these right with no labeled training data --
fine-tuning is worth revisiting only if error analysis on a hand-labeled
test set shows the rules missing real cases the rules can't be widened to
catch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import spacy
from spacy.language import Language

# Two ways a term gets declared in this corpus, evidenced from the 100-doc
# gold-set evaluation (7/7 GOVERNING_LAW, but DEFINED_TERM recall was only
# 0.60 before this fix -- 3 of 4 misses were bare "(X")" with no "the"/
# "this" immediately before, e.g. 'annual base salary ("Base Salary")'):
#   1. "the"/"this" directly before the quote: (the "X") / (this "X")
#   2. bare "(" directly before the quote, no "the"/"this" needed: ("X")
# NOT matched (still a real, unfixed limitation): "The term "X" as used
# herein..." -- "the" is separated from the quote by another word ("term"),
# a genuinely different construct that would need its own pattern. See
# tests/test_ner_extract.py::test_defined_term_rule_still_misses_the_term_x_construct.
DEFINED_TERM_RE = re.compile(
    r'(?:\(\s*(?:the|this)?\s*|(?:the|this)\s+)'
    r'["\u201c]\s*([A-Z][A-Za-z0-9 /\-]{1,40}?)\s*["\u201d]'
)
# Case-insensitivity is scoped to ONLY the literal "state of" text via the
# inline (?i:...) flag, not applied to the whole pattern -- an earlier
# version used re.IGNORECASE on the full compiled regex, which also
# weakened the capture group's [A-Z] requirement (so lowercase words like
# "organization" or "and" started matching as if they were state names) and
# had no word boundary, so it matched "state of" as a substring inside
# "eSTATE OF" ("estate of"). Found via manual review of a second 70-doc
# batch after the first IGNORECASE attempt; see
# tests/test_ner_extract.py::test_governing_law_rule_rejects_estate_of_false_positive
# and ::test_governing_law_rule_rejects_lowercase_non_state_words.
GOVERNING_LAW_RE = re.compile(
    r"\b(?i:state of) ([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)"
)


@dataclass
class ExtractedEntity:
    text: str
    label: str
    start: int
    end: int
    source: str  # "spacy" or "rule"


@lru_cache(maxsize=1)
def load_pipeline(model: str = "en_core_web_sm") -> Language:
    """Cached so the model loads once per process, not once per call --
    loading en_core_web_sm takes real wall-clock time and there's no reason
    to pay it on every request in the API."""
    return spacy.load(model)


def _extract_rule_matches(
    text: str, pattern: re.Pattern, label: str
) -> list[ExtractedEntity]:
    return [
        ExtractedEntity(
            text=m.group(1),
            label=label,
            start=m.start(1),
            end=m.end(1),
            source="rule",
        )
        for m in pattern.finditer(text)
    ]


def _spans_overlap(a: ExtractedEntity, b: ExtractedEntity) -> bool:
    return a.start < b.end and b.start < a.end


def extract_entities(text: str, nlp: Language | None = None) -> list[ExtractedEntity]:
    """Combine spaCy's pretrained entities with rule-based GOVERNING_LAW and
    DEFINED_TERM extraction. On overlap, the rule match wins and the
    overlapping spaCy tag is dropped -- rules are precision-tuned for this
    domain, spaCy's general-English tags are known (from Phase 3 profiling)
    to mislabel exactly these spans."""
    nlp = nlp or load_pipeline()
    doc = nlp(text)

    spacy_ents = [
        ExtractedEntity(
            text=ent.text,
            label=ent.label_,
            start=ent.start_char,
            end=ent.end_char,
            source="spacy",
        )
        for ent in doc.ents
    ]

    rule_ents = _extract_rule_matches(
        text, GOVERNING_LAW_RE, "GOVERNING_LAW"
    ) + _extract_rule_matches(text, DEFINED_TERM_RE, "DEFINED_TERM")

    kept_spacy = [
        se
        for se in spacy_ents
        if not any(_spans_overlap(se, re_) for re_ in rule_ents)
    ]

    combined = kept_spacy + rule_ents
    combined.sort(key=lambda e: e.start)
    return combined


def extract_structured(text: str, nlp: Language | None = None) -> dict[str, list[str]]:
    """Convenience wrapper: entities grouped by label, text only (no
    offsets) -- the shape a JSON API response would return."""
    entities = extract_entities(text, nlp=nlp)
    grouped: dict[str, list[str]] = {}
    for e in entities:
        grouped.setdefault(e.label, []).append(e.text)
    return grouped
