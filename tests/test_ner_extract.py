"""Tests for legalintel.ner.extract.

Covers: the two regex rules (governing-law multi-word states, defined-term
'the'/'this' variants) and the overlap-suppression logic that lets a rule
match override a conflicting spaCy tag on the same span.
"""
from __future__ import annotations

import pytest

from legalintel.ner.extract import extract_entities, extract_structured, load_pipeline


@pytest.fixture(scope="module")
def nlp():
    return load_pipeline()


def test_governing_law_rule_catches_multiword_state(nlp):
    text = (
        "This Agreement shall be governed by the laws of the State of New York "
        "without regard to conflicts of law."
    )
    ents = extract_entities(text, nlp=nlp)
    gov = [e for e in ents if e.label == "GOVERNING_LAW"]
    assert len(gov) == 1
    assert gov[0].text == "New York"
    assert gov[0].source == "rule"


def test_governing_law_rule_catches_single_word_state(nlp):
    text = "governed under the laws of the State of Georgia."
    ents = extract_entities(text, nlp=nlp)
    gov = [e for e in ents if e.label == "GOVERNING_LAW"]
    assert len(gov) == 1
    assert gov[0].text == "Georgia"


def test_defined_term_rule_catches_this_and_the(nlp):
    text = (
        'This Agreement (this "Amendment") amends the agreement between '
        'the parties (the "Company").'
    )
    ents = extract_entities(text, nlp=nlp)
    defined = {e.text for e in ents if e.label == "DEFINED_TERM"}
    assert "Amendment" in defined
    assert "Company" in defined


def test_rule_match_suppresses_overlapping_spacy_tag(nlp):
    """A span claimed by a rule (e.g. 'Georgia' as GOVERNING_LAW) should not
    also appear as a separate, conflicting spaCy entity (e.g. GPE)."""
    text = "governed under the laws of the State of Georgia."
    ents = extract_entities(text, nlp=nlp)
    gov = [e for e in ents if e.label == "GOVERNING_LAW"][0]
    overlapping = [
        e for e in ents
        if e is not gov and e.start < gov.end and gov.start < e.end
    ]
    assert overlapping == []


def test_extract_structured_groups_by_label(nlp):
    text = 'Governed by the State of Delaware. Defined as (the "Grantor").'
    result = extract_structured(text, nlp=nlp)
    assert "GOVERNING_LAW" in result
    assert "Delaware" in result["GOVERNING_LAW"]
    assert "DEFINED_TERM" in result
    assert "Grantor" in result["DEFINED_TERM"]


def test_spacy_entities_still_pass_through_when_no_rule_applies(nlp):
    """Confirms the pipeline isn't accidentally dropping all spaCy output --
    only overlapping ones should be suppressed."""
    text = "The Plan shall be construed and administered in accordance with ERISA."
    ents = extract_entities(text, nlp=nlp)
    assert any(e.source == "spacy" for e in ents)


def test_governing_law_rule_catches_all_caps_clause(nlp):
    """Regression test: found via manual review of a 70-doc batch (doc 033
    in that batch) -- contracts frequently put governing-law clauses in
    ALL CAPS for legal conspicuousness requirements, and the original regex
    only matched the literal string 'State of' (mixed case), silently
    missing every all-caps instance."""
    text = (
        "SHALL BE CONSTRUED IN ACCORDANCE WITH AND GOVERNED BY THE LAWS OF "
        "THE STATE OF NEW YORK WITHOUT REGARD TO CONFLICT OF LAWS."
    )
    ents = extract_entities(text, nlp=nlp)
    gov = [e for e in ents if e.label == "GOVERNING_LAW"]
    assert len(gov) == 1
    assert gov[0].text.upper() == "NEW YORK"


def test_defined_term_rule_handles_padded_quotes(nlp):
    """Regression test: found via manual review (docs 036, 055 in a 70-doc
    batch) -- real contract text sometimes has a space directly inside the
    curly quotes (e.g. '(the " Moving Party ")'), likely a document-
    conversion artifact. The original regex required the captured text to
    start immediately after the quote mark, so a leading space broke the
    match entirely."""
    text = 'related costs (collectively, the \u201c Transaction Expenses \u201d). Such expenses'
    ents = extract_entities(text, nlp=nlp)
    defined = {e.text for e in ents if e.label == "DEFINED_TERM"}
    assert "Transaction Expenses" in defined


def test_governing_law_rule_rejects_estate_of_false_positive(nlp):
    """Regression test: an earlier fix for ALL-CAPS matching used a blanket
    re.IGNORECASE on the whole pattern, which had no word boundary and
    matched 'state of' as a literal substring inside 'estate of'. Found in
    doc 048 of a 70-doc manual review batch."""
    text = (
        "all right, title, interest and estate of Seller in, to and under "
        "the oil and gas leases"
    )
    ents = extract_entities(text, nlp=nlp)
    gov = [e for e in ents if e.label == "GOVERNING_LAW"]
    assert gov == [], f"expected no GOVERNING_LAW match, got {gov}"


def test_governing_law_rule_rejects_lowercase_non_state_words(nlp):
    """Regression test: the same blanket-IGNORECASE bug also weakened the
    capture group's uppercase requirement, so lowercase words like
    'organization' or 'and' started matching as if they were state names.
    Found in docs 042 and 096 of a 70-doc manual review batch."""
    text1 = "duly organized under the laws of its State of organization and is validly existing"
    text2 = "shall be governed by the laws of the State of Delaware and construed in accordance"

    ents1 = extract_entities(text1, nlp=nlp)
    gov1 = [e for e in ents1 if e.label == "GOVERNING_LAW"]
    assert gov1 == [], f"expected no match on 'State of organization and', got {gov1}"

    ents2 = extract_entities(text2, nlp=nlp)
    gov2 = [e for e in ents2 if e.label == "GOVERNING_LAW"]
    assert len(gov2) == 1 and gov2[0].text == "Delaware", (
        f"expected exactly 'Delaware' (not 'Delaware and'), got {gov2}"
    )


def test_defined_term_rule_catches_bare_parenthetical(nlp):
    """Regression test: found via the 100-doc gold-set evaluation -- 3 of 4
    DEFINED_TERM recall misses were bare '(\"X\")' declarations with no
    'the'/'this' immediately before the quote (e.g. 'annual base salary
    (\"Base Salary\")', 'liabilities (\"Damages\")'). The original rule only
    matched 'the'/'this' + quote."""
    text1 = 'annual base salary ("Base Salary"), evenly paid twice a month'
    text2 = 'liabilities ("Damages"), whether direct, contingent or otherwise'

    ents1 = extract_entities(text1, nlp=nlp)
    defined1 = {e.text for e in ents1 if e.label == "DEFINED_TERM"}
    assert "Base Salary" in defined1

    ents2 = extract_entities(text2, nlp=nlp)
    defined2 = {e.text for e in ents2 if e.label == "DEFINED_TERM"}
    assert "Damages" in defined2


def test_defined_term_rule_bare_paren_does_not_reintroduce_false_positives(nlp):
    """Regression test: confirms widening to bare '(\"X\")' didn't
    reintroduce the two known false-positive risks from earlier manual
    review -- quoted phrases with no '(' immediately before them, referenced
    inline rather than declared."""
    text1 = 'including but not limited to that undated "Consultant/President Agreement Term Sheet"'
    text2 = 'GS Lending Partners only, to any additional "Commitment Parties"'

    ents1 = extract_entities(text1, nlp=nlp)
    assert [e for e in ents1 if e.label == "DEFINED_TERM"] == []

    ents2 = extract_entities(text2, nlp=nlp)
    assert [e for e in ents2 if e.label == "DEFINED_TERM"] == []


def test_defined_term_rule_still_misses_the_term_x_construct(nlp):
    """Documents a KNOWN, still-unfixed limitation (not a regression to
    guard against -- a gap to remember): 'The term "X" as used herein...'
    is not caught, since 'the' is separated from the quote by another word
    ('term'). Found in doc 006 of the 100-doc gold-set evaluation. If this
    test starts failing (i.e. it starts matching), that's an IMPROVEMENT --
    update the docstring above and this test rather than treating it as a
    break."""
    text = 'The term "Closing" as used herein shall refer to the actual consummation'
    ents = extract_entities(text, nlp=nlp)
    defined = {e.text for e in ents if e.label == "DEFINED_TERM"}
    assert "Closing" not in defined
