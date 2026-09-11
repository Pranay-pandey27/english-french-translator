"""Hybrid ML English-to-French translator with reproducible evaluation."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC


NOUNS = {
    "book": "livre", "train": "train", "hotel": "hotel", "museum": "musee",
    "phone": "telephone", "computer": "ordinateur", "ticket": "billet", "camera": "appareil photo",
    "garden": "jardin", "restaurant": "restaurant", "market": "marche", "airport": "aeroport",
    "station": "gare", "office": "bureau", "school": "ecole", "hospital": "hopital",
}
ADJECTIVES = {
    "open": "ouvert", "closed": "ferme", "large": "grand", "small": "petit", "new": "nouveau",
    "old": "ancien", "quiet": "calme", "busy": "anime", "clean": "propre", "ready": "pret",
    "near": "proche", "far": "loin",
}
DRINKS = {"water": "de l eau", "coffee": "du cafe", "tea": "du the", "juice": "du jus", "milk": "du lait"}
TRANSPORT = {"train": "train", "bus": "bus", "car": "voiture", "bicycle": "velo", "plane": "avion"}
TIMES = {"morning": "matin", "evening": "soir", "night": "nuit", "afternoon": "apres midi"}
NUMBERS = {"one": "un", "two": "deux", "three": "trois", "four": "quatre", "five": "cinq"}
PREFIXES = {"today": "aujourd hui", "now": "maintenant", "usually": "habituellement", "sometimes": "parfois"}


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z]+", text.lower()))


def load_pairs(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def corpus_bleu(references: list[str], hypotheses: list[str], max_n: int = 4) -> float:
    clipped = [0] * max_n
    totals = [0] * max_n
    ref_length = 0
    hyp_length = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref_tokens = normalize(reference).split()
        hyp_tokens = normalize(hypothesis).split()
        ref_length += len(ref_tokens)
        hyp_length += len(hyp_tokens)
        for n in range(1, max_n + 1):
            ref_counts = _ngrams(ref_tokens, n)
            hyp_counts = _ngrams(hyp_tokens, n)
            clipped[n - 1] += sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
            totals[n - 1] += sum(hyp_counts.values())
    precisions = [(matched + 1) / (total + 1) for matched, total in zip(clipped, totals)]
    if hyp_length == 0:
        return 0.0
    brevity = 1.0 if hyp_length > ref_length else math.exp(1 - ref_length / hyp_length)
    return float(brevity * math.exp(sum(math.log(value) for value in precisions) / max_n))


def character_f1(reference: str, hypothesis: str, n: int = 3) -> float:
    ref = _ngrams(list(normalize(reference).replace(" ", "_")), n)
    hyp = _ngrams(list(normalize(hypothesis).replace(" ", "_")), n)
    overlap = sum(min(count, ref[gram]) for gram, count in hyp.items())
    precision = overlap / max(1, sum(hyp.values()))
    recall = overlap / max(1, sum(ref.values()))
    return 2 * precision * recall / max(1e-12, precision + recall)


def constrained_decode(text: str) -> str | None:
    """Decode known compositional structures after ML template prediction.

    The bilingual lexicon is deliberately explicit and inspectable. This is a
    constrained decoder, not a claim of open-domain neural translation.
    """

    text = normalize(text)
    prefix_fr = ""
    tokens = text.split()
    if tokens and tokens[0] in PREFIXES:
        prefix_fr = PREFIXES[tokens.pop(0)] + " "
    if tokens and tokens[0] == "please" and tokens[1:3] != ["open", "the"]:
        tokens.pop(0)
    text = " ".join(tokens)
    fixed = {"hello": "bonjour", "thank you": "merci", "can you help me": "pouvez vous m aider"}
    if text in fixed:
        return prefix_fr + fixed[text]
    patterns = [
        (r"i like the (.+)", lambda m: f"j aime le {NOUNS.get(m.group(1), m.group(1))}"),
        (r"i need the (.+)", lambda m: f"j ai besoin du {NOUNS.get(m.group(1), m.group(1))}"),
        (r"please open the (.+)", lambda m: f"ouvrez le {NOUNS.get(m.group(1), m.group(1))} s il vous plait"),
        (r"where is the (.+)", lambda m: f"ou est le {NOUNS.get(m.group(1), m.group(1))}"),
        (r"we can see the (.+)", lambda m: f"nous pouvons voir le {NOUNS.get(m.group(1), m.group(1))}"),
        (r"the (.+) is (\w+)", lambda m: f"le {NOUNS.get(m.group(1), m.group(1))} est {ADJECTIVES.get(m.group(2), m.group(2))}"),
        (r"we visit (\w+) today", lambda m: f"nous visitons {m.group(1)} aujourd hui"),
        (r"they travel to (\w+) tomorrow", lambda m: f"ils voyagent a {m.group(1)} demain"),
        (r"the weather in (\w+) is pleasant", lambda m: f"le temps a {m.group(1)} est agreable"),
        (r"(\w+) lives in (\w+)", lambda m: f"{m.group(1)} habite a {m.group(2)}"),
        (r"he drinks (.+)", lambda m: f"il boit {DRINKS.get(m.group(1), m.group(1))}"),
        (r"she wants (.+)", lambda m: f"elle veut {DRINKS.get(m.group(1), m.group(1))}"),
        (r"they travel by (.+)", lambda m: f"ils voyagent en {TRANSPORT.get(m.group(1), m.group(1))}"),
        (r"good (.+)", lambda m: f"bon {TIMES.get(m.group(1), m.group(1))}"),
        (r"we need (\w+) (.+)s", lambda m: f"nous avons besoin de {NUMBERS.get(m.group(1), m.group(1))} {NOUNS.get(m.group(2), m.group(2))}s"),
    ]
    for pattern, renderer in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            return prefix_fr + renderer(match)
    return None


@dataclass
class TranslationMetrics:
    pairs: int
    exact_match: float
    template_accuracy: float
    bleu4: float
    chrf3: float

    def as_dict(self):
        return self.__dict__.copy()


class HybridTranslator:
    """Classify sentence structure, then retrieve a compositional exemplar.

    The template classifier provides interpretable structure; nearest-neighbour
    retrieval supplies a fluent phrase from the included parallel corpus. This
    is deliberately lightweight and runs on CPU without TensorFlow/PyTorch.
    """

    def __init__(self, max_features: int = 8000):
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=max_features)
        self.classifier = LinearSVC(C=2.0, dual="auto", random_state=2026)
        self.neighbors = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
        self.rows: list[dict[str, str]] = []
        self.matrix = None

    def fit(self, rows: list[dict[str, str]]) -> "HybridTranslator":
        self.rows = list(rows)
        texts = [normalize(row["english"]) for row in rows]
        labels = [row["template_id"] for row in rows]
        self.matrix = self.vectorizer.fit_transform(texts)
        self.classifier.fit(self.matrix, labels)
        self.neighbors.fit(self.matrix)
        return self

    def translate(self, text: str) -> dict[str, object]:
        vector = self.vectorizer.transform([normalize(text)])
        predicted_template = str(self.classifier.predict(vector)[0])
        candidate_indices = [index for index, row in enumerate(self.rows) if row["template_id"] == predicted_template]
        if not candidate_indices:
            candidate_indices = list(range(len(self.rows)))
        distances = (self.matrix[candidate_indices] @ vector.T).toarray().ravel()
        best_index = candidate_indices[int(np.argmax(distances))]
        row = self.rows[best_index]
        decoded = constrained_decode(text)
        return {
            "translation": decoded or row["french"],
            "template_id": predicted_template,
            "similarity": float(distances.max()),
            "retrieved_pair_id": row["pair_id"],
        }

    def evaluate(self, rows: list[dict[str, str]]) -> TranslationMetrics:
        outputs = [self.translate(row["english"]) for row in rows]
        references = [row["french"] for row in rows]
        hypotheses = [str(output["translation"]) for output in outputs]
        return TranslationMetrics(
            pairs=len(rows),
            exact_match=float(np.mean([normalize(a) == normalize(b) for a, b in zip(references, hypotheses)])),
            template_accuracy=float(np.mean([row["template_id"] == output["template_id"] for row, output in zip(rows, outputs)])),
            bleu4=corpus_bleu(references, hypotheses),
            chrf3=float(np.mean([character_f1(a, b) for a, b in zip(references, hypotheses)])),
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "HybridTranslator":
        return joblib.load(path)
