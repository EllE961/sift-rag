import pytest

from sift.text import LUCENE_STOPWORDS, Analyzer, PorterStemmer, normalize, split_sentences

# Examples from Porter (1980), "An algorithm for suffix stripping".
PAPER_EXAMPLES = {
    "caresses": "caress", "ponies": "poni", "ties": "ti", "caress": "caress", "cats": "cat",
    "feed": "feed", "agreed": "agre", "plastered": "plaster", "bled": "bled", "motoring": "motor",
    "sing": "sing", "conflated": "conflat", "troubled": "troubl", "sized": "size", "hopping": "hop",
    "tanned": "tan", "falling": "fall", "hissing": "hiss", "fizzed": "fizz", "failing": "fail",
    "filing": "file", "happy": "happi", "sky": "sky", "relational": "relat", "conditional": "condit",
    "rational": "ration", "valenci": "valenc", "hesitanci": "hesit", "digitizer": "digit",
    "conformabli": "conform", "radicalli": "radic", "differentli": "differ", "vileli": "vile",
    "analogousli": "analog", "vietnamization": "vietnam", "predication": "predic", "operator": "oper",
    "feudalism": "feudal", "decisiveness": "decis", "hopefulness": "hope", "callousness": "callous",
    "formaliti": "formal", "sensitiviti": "sensit", "sensibiliti": "sensibl", "triplicate": "triplic",
    "formative": "form", "formalize": "formal", "electriciti": "electr", "electrical": "electr",
    "hopeful": "hope", "goodness": "good", "revival": "reviv", "allowance": "allow", "inference": "infer",
    "airliner": "airlin", "gyroscopic": "gyroscop", "adjustable": "adjust", "defensible": "defens",
    "irritant": "irrit", "replacement": "replac", "adjustment": "adjust", "dependent": "depend",
    "adoption": "adopt", "homologou": "homolog", "communism": "commun", "activate": "activ",
    "angulariti": "angular", "homologous": "homolog", "effective": "effect", "bowdlerize": "bowdler",
    "probate": "probat", "rate": "rate", "cease": "ceas", "controll": "control", "roll": "roll",
}  # fmt: skip


@pytest.mark.parametrize("word,expected", sorted(PAPER_EXAMPLES.items()))
def test_porter_paper_examples(word, expected):
    assert PorterStemmer().stem(word) == expected


def _generated_vocab() -> list[str]:
    bases = ["connect", "relat", "condition", "generaliz", "hope", "nation", "operat", "sensitiv", "agree",
             "fall", "hop", "plaster", "electr", "analog", "digit", "triplic", "cell", "protein", "regulat",
             "inhibit", "express", "differ", "activ", "mutat", "transcript", "observ", "measur", "random",
             "sky", "happy", "cry", "fly", "die", "lie", "tie", "spy", "feed", "need", "bleed", "process",
             "access", "pass", "kiss", "bus", "gas", "focus", "virus", "tumor", "cancer", "doctor"]  # fmt: skip
    suffixes = ["", "s", "es", "ed", "ing", "ly", "er", "ers", "ness", "ful", "fulness", "ive", "ively", "iveness",
                "ation", "ational", "ations", "ize", "ized", "izing", "ization", "izer", "able", "ability", "ible",
                "ism", "ist", "ity", "iti", "ous", "ously", "ousness", "ance", "ence", "ent", "ently", "ement",
                "ment", "al", "ally", "alism", "aliti", "icate", "ical", "ative", "y", "ies", "ied", "eed", "ion"]  # fmt: skip
    return sorted({b + s for b in bases for s in suffixes})


def test_porter_matches_nltk_original_algorithm():
    nltk_porter = pytest.importorskip("nltk.stem.porter")
    oracle = nltk_porter.PorterStemmer(mode=nltk_porter.PorterStemmer.ORIGINAL_ALGORITHM)
    ours = PorterStemmer()
    words = _generated_vocab()
    mismatches = [(w, ours.stem(w), oracle.stem(w)) for w in words if ours.stem(w) != oracle.stem(w)]
    assert len(words) > 2000
    assert mismatches == []


def test_measure():
    s = PorterStemmer()
    assert [s.measure(w) for w in ("tr", "ee", "tree", "y", "by")] == [0, 0, 0, 0, 0]
    assert [s.measure(w) for w in ("trouble", "oats", "trees", "ivy")] == [1, 1, 1, 1]
    assert [s.measure(w) for w in ("troubles", "private", "oaten", "orrery")] == [2, 2, 2, 2]


def test_analyzer_pipeline():
    a = Analyzer()
    assert a("The patient's tumours were growing") == ["patient", "tumour", "were", "grow"]
    assert "the" in LUCENE_STOPWORDS and a("the of and") == []
    assert normalize("Café Ñandú") == "cafe nandu"
    assert a("COVID-19 vaccines") == ["covid", "19", "vaccin"]


def test_split_sentences_spans_are_exact():
    text = "Dr. Smith measured 0.05 mg/kg (e.g. in mice). Results were significant! Next, et al. agreed. Done"
    spans = split_sentences(text)
    sents = [text[a:b] for a, b in spans]
    assert sents == [
        "Dr. Smith measured 0.05 mg/kg (e.g. in mice).",
        "Results were significant!",
        "Next, et al. agreed.",
        "Done",
    ]


def test_split_sentences_edge_cases():
    assert split_sentences("") == []
    assert split_sentences("   ") == []
    assert split_sentences("One sentence without a period") == [(0, 29)]
    t = "  Leading space. Trailing  "
    assert [t[a:b] for a, b in split_sentences(t)] == ["Leading space.", "Trailing"]
