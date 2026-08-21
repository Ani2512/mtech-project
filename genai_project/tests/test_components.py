"""Retrieval, output normalisation, rendering, and data-room integrity.

The data-room test is the guard against a drift that actually happened: the
committed CSVs stopped matching what `make_sample_data.py` produces, so the
generator documented in the README would have silently replaced the data the
sample report describes.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

from analyst.agents import (
    AgentResult,
    _normalise_eval,
    _normalise_risk,
    _normalise_specialist,
    _normalise_strategy,
)
from analyst.datasets import DataRoom
from analyst.reporting import confidence_bar, render_evaluation, stars
from analyst.retrieval import (
    BM25,
    HybridRetriever,
    Passage,
    build_corpus,
    chunk_markdown,
    tokenize,
)

DOC = """\
# Market outlook

Demand in Southeast Asia is accelerating, with Vietnam and Indonesia leading
regional growth as replacement cycles shorten across the category.

# Cost and regulation

Small-format lithium cell prices are expected to rise 8-12% in 2026, and new
battery labelling rules take effect in the EU.
"""


class TokenizerTests(unittest.TestCase):
    def test_stopwords_and_short_tokens_are_dropped(self):
        self.assertEqual(tokenize("The cost of a cell is"), ["cost", "cell"])

    def test_case_and_punctuation_are_normalised(self):
        self.assertIn("vietnam", tokenize("Vietnam, Indonesia!"))


class ChunkingTests(unittest.TestCase):
    def test_headings_become_sections(self):
        passages = chunk_markdown(DOC, "market/test.md")
        self.assertEqual(len(passages), 2)
        self.assertEqual(passages[0].section, "Market outlook")
        self.assertEqual(passages[1].section, "Cost and regulation")

    def test_citation_carries_file_and_section(self):
        p = chunk_markdown(DOC, "market/test.md")[0]
        self.assertEqual(p.citation, "market/test.md § Market outlook")

    def test_trivially_short_blocks_are_skipped(self):
        self.assertEqual(chunk_markdown("# H\n\nshort\n", "x.md"), [])


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.corpus = chunk_markdown(DOC, "market/test.md")

    def test_exact_term_ranks_its_passage_first(self):
        scores = BM25(self.corpus).scores("Vietnam growth")
        self.assertGreater(scores[0], scores[1])

    def test_unmatched_query_scores_zero(self):
        self.assertEqual(sum(BM25(self.corpus).scores("zebra tuba")), 0.0)


class HybridRetrieverTests(unittest.TestCase):
    def test_empty_query_returns_nothing(self):
        r = HybridRetriever(chunk_markdown(DOC, "x.md"))
        self.assertEqual(r.search("   "), [])

    def test_results_are_ordered_and_capped(self):
        room = DataRoom()
        retriever = HybridRetriever(build_corpus(room), alpha=0.5)
        hits = retriever.search("Southeast Asia expansion", top_k=3)
        self.assertLessEqual(len(hits), 3)
        self.assertEqual([h.score for h in hits], sorted((h.score for h in hits), reverse=True))
        self.assertTrue(all(h.passage.citation for h in hits))

    def test_lexical_only_mode_still_retrieves(self):
        """alpha=0 must degrade to BM25 rather than returning nothing —
        this is the fallback when no dense index can be built."""
        retriever = HybridRetriever(build_corpus(DataRoom()), alpha=0.0)
        self.assertTrue(retriever.search("tariff", top_k=3))

    def test_hit_explains_which_retriever_found_it(self):
        retriever = HybridRetriever(build_corpus(DataRoom()), alpha=0.5)
        hit = retriever.search("competitor pricing", top_k=1)[0]
        self.assertRegex(hit.why(), r"bm25|dense")


class NormalisationTests(unittest.TestCase):
    """Strict JSON schemas cannot express numeric bounds, so ranges are
    clamped in Python. Downstream code trusts these invariants."""

    def test_confidence_is_clamped_to_percent(self):
        self.assertEqual(_normalise_specialist({"confidence": 250})["confidence"], 100)
        self.assertEqual(_normalise_specialist({"confidence": -5})["confidence"], 0)
        self.assertEqual(_normalise_strategy({"confidence": "not a number"})["confidence"], 50)

    def test_risk_scores_are_clamped_to_one_to_five(self):
        data = _normalise_risk(
            {
                "overall_risk": 99,
                "risks": [{"likelihood": 0, "impact": 12}],
                "category_scores": [{"score": -3}],
            }
        )
        self.assertEqual(data["overall_risk"], 5)
        self.assertEqual(data["risks"][0]["likelihood"], 1)
        self.assertEqual(data["risks"][0]["impact"], 5)
        self.assertEqual(data["category_scores"][0]["score"], 1)

    def test_missing_finding_fields_get_defaults(self):
        data = _normalise_specialist({"findings": [{"title": "x"}]})
        self.assertEqual(data["findings"][0]["direction"], "flat")
        self.assertEqual(data["findings"][0]["impact"], "medium")

    def test_evaluation_scores_are_clamped(self):
        data = _normalise_eval({"overall_score": 900, "scores": [{"score": -1}]})
        self.assertEqual(data["overall_score"], 100)
        self.assertEqual(data["scores"][0]["score"], 0)

    def test_malformed_lists_do_not_crash(self):
        _normalise_risk({"risks": "not a list", "category_scores": None})
        _normalise_specialist({"findings": None})


class RenderingTests(unittest.TestCase):
    def test_stars_clamp_and_render(self):
        self.assertEqual(stars(3), "★★★☆☆")
        self.assertEqual(stars(99), "★★★★★")
        self.assertEqual(stars("bad"), "☆☆☆☆☆")

    def test_confidence_bar_is_fixed_width(self):
        self.assertIn("50%", confidence_bar(50))
        self.assertIn("100%", confidence_bar(500))

    def test_evaluation_renders_absent_and_failed_states(self):
        self.assertIn("not run", render_evaluation(None))
        self.assertIn("failed", render_evaluation(AgentResult("evaluator", False, error="boom")))


class DataRoomTests(unittest.TestCase):
    def test_every_declared_source_exists(self):
        self.assertEqual(DataRoom().missing(), [])

    def test_csv_values_are_coerced_to_numbers(self):
        rows = DataRoom().rows("finance/income_statement.csv")
        self.assertTrue(rows)
        self.assertTrue(any(isinstance(v, (int, float)) for v in rows[0].values()))

    def test_committed_data_matches_its_generator(self):
        """Guards the drift that actually occurred.

        If this fails, `python make_sample_data.py --force` would replace the
        data room with something the sample report no longer describes.
        """
        import make_sample_data as gen

        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp) / "data"
            with mock.patch.object(gen, "DATA_DIR", target), \
                 mock.patch.object(sys, "argv", ["make_sample_data.py", "--force"]), \
                 mock.patch("builtins.print"):
                gen.main()

            committed = pathlib.Path(__file__).resolve().parent.parent / "data"
            generated = sorted(p.relative_to(target) for p in target.rglob("*") if p.is_file())
            self.assertTrue(generated, "generator produced no files")
            for rel in generated:
                self.assertEqual(
                    (target / rel).read_bytes(),
                    (committed / rel).read_bytes(),
                    f"{rel} differs from what make_sample_data.py produces",
                )


if __name__ == "__main__":
    unittest.main()