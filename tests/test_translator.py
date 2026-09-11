from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from translator import HybridTranslator, load_pairs


def test_translator_pipeline():
    rows = load_pairs(Path(__file__).parents[1] / "data" / "english_french_pairs.tsv")
    model = HybridTranslator(max_features=2000).fit(rows[:1000])
    result = model.translate("i like the book")
    assert result["translation"]
    assert 0 <= result["similarity"] <= 1.000001
