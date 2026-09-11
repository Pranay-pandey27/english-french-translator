"""Generate a deterministic 10,000-pair English-French compositional corpus."""

from __future__ import annotations

import csv
import random
from pathlib import Path


NOUNS = {
    "book": "livre", "train": "train", "hotel": "hotel", "museum": "musee",
    "phone": "telephone", "computer": "ordinateur", "ticket": "billet", "camera": "appareil photo",
    "garden": "jardin", "restaurant": "restaurant", "market": "marche", "airport": "aeroport",
    "station": "gare", "office": "bureau", "school": "ecole", "hospital": "hopital",
}
ADJECTIVES = {
    "open": "ouvert", "closed": "ferme", "large": "grand", "small": "petit",
    "new": "nouveau", "old": "ancien", "quiet": "calme", "busy": "anime",
    "clean": "propre", "ready": "pret", "near": "proche", "far": "loin",
}
CITIES = ["paris", "lyon", "london", "delhi", "singapore", "boston", "berlin", "rome", "madrid", "tokyo"]
DRINKS = {"water": "de l eau", "coffee": "du cafe", "tea": "du the", "juice": "du jus", "milk": "du lait"}
TRANSPORT = {"train": "train", "bus": "bus", "car": "voiture", "bicycle": "velo", "plane": "avion"}
PEOPLE = ["anna", "david", "marie", "lucas", "pranay", "sarah", "noah", "emma"]
TIMES = {"morning": "matin", "evening": "soir", "night": "nuit", "afternoon": "apres midi"}
NUMBERS = {"one": "un", "two": "deux", "three": "trois", "four": "quatre", "five": "cinq"}


def candidates() -> list[tuple[str, str, str]]:
    rows: set[tuple[str, str, str]] = set()
    for noun, french in NOUNS.items():
        rows.update({
            ("like_noun", f"i like the {noun}", f"j aime le {french}"),
            ("need_noun", f"i need the {noun}", f"j ai besoin du {french}"),
            ("open_noun", f"please open the {noun}", f"ouvrez le {french} s il vous plait"),
            ("find_noun", f"where is the {noun}", f"ou est le {french}"),
            ("see_noun", f"we can see the {noun}", f"nous pouvons voir le {french}"),
        })
        for adjective, adjective_fr in ADJECTIVES.items():
            rows.add(("noun_state", f"the {noun} is {adjective}", f"le {french} est {adjective_fr}"))
    for city in CITIES:
        rows.update({
            ("visit_city", f"we visit {city} today", f"nous visitons {city} aujourd hui"),
            ("travel_city", f"they travel to {city} tomorrow", f"ils voyagent a {city} demain"),
            ("weather_city", f"the weather in {city} is pleasant", f"le temps a {city} est agreable"),
        })
        for person in PEOPLE:
            rows.add(("person_city", f"{person} lives in {city}", f"{person} habite a {city}"))
    for drink, french in DRINKS.items():
        rows.update({
            ("he_drinks", f"he drinks {drink}", f"il boit {french}"),
            ("she_wants_drink", f"she wants {drink}", f"elle veut {french}"),
        })
    for mode, french in TRANSPORT.items():
        rows.add(("transport", f"they travel by {mode}", f"ils voyagent en {french}"))
    for time, french in TIMES.items():
        rows.add(("greeting", f"good {time}", f"bon {french}"))
    for number, number_fr in NUMBERS.items():
        for noun, french in NOUNS.items():
            rows.add(("number_noun", f"we need {number} {noun}s", f"nous avons besoin de {number_fr} {french}s"))

    # Generate controlled surface variation while preserving exact translations.
    expanded = set(rows)
    polite_pairs = [("hello", "bonjour"), ("thank you", "merci"), ("can you help me", "pouvez vous m aider")]
    for english, french in polite_pairs:
        expanded.add(("fixed_phrase", english, french))
    base = list(expanded)
    prefixes = [("today", "aujourd hui"), ("now", "maintenant"), ("usually", "habituellement"), ("sometimes", "parfois")]
    for template_id, english, french in base:
        if template_id not in {"greeting", "fixed_phrase"}:
            for prefix_en, prefix_fr in prefixes:
                expanded.add((f"prefixed_{template_id}", f"{prefix_en} {english}", f"{prefix_fr} {french}"))
    return sorted(expanded)


def build_dataset(output: Path, n_pairs: int = 10_000, seed: int = 2026) -> None:
    rng = random.Random(seed)
    base = candidates()
    rows = []
    # The compositional grammar has fewer than 10k unique semantic pairs, so we
    # add deterministic register/punctuation variants and retain a group id.
    variants = [
        ("plain", lambda text: text),
        ("capitalized", lambda text: text.capitalize()),
        ("question", lambda text: text + " ?"),
        ("statement", lambda text: text + " ."),
        ("please", lambda text: "please " + text),
    ]
    index = 0
    while len(rows) < n_pairs:
        template_id, english, french = base[index % len(base)]
        variant_name, transform = variants[(index // len(base)) % len(variants)]
        english_variant = transform(english)
        french_variant = french + (" ?" if variant_name == "question" else " ." if variant_name == "statement" else "")
        rows.append({
            "pair_id": f"pair_{index:05d}",
            "group_id": f"meaning_{index % len(base):05d}",
            "template_id": template_id,
            "variant": variant_name,
            "english": english_variant,
            "french": french_variant,
        })
        index += 1
    rng.shuffle(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    build_dataset(Path(__file__).with_name("english_french_pairs.tsv"))
