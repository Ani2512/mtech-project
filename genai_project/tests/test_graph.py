"""Orchestration: the full pipeline, and how it degrades.

These run entirely on the mock backend, so the suite is free and offline. What
they check is the wiring — that failure is contained, that a tripped budget
aborts the run without discarding the work already paid for, and that the
report renders whatever survived.
"""

from __future__ import annotations

import unittest
from unittest import mock

from analyst import Settings, render_report, run_analysis
from analyst.agents import AgentResult
from analyst.config import PROVIDER_MOCK
from analyst.graph import AnalysisResult
from analyst.llm import BudgetExceeded

QUESTION = "Why did profit decline?"


def _settings(**kw) -> Settings:
    base = dict(backend=PROVIDER_MOCK, use_cache=False, run_evaluation=False)
    base.update(kw)
    return Settings(**base)


class PipelineTests(unittest.TestCase):
    def test_full_run_produces_a_report(self):
        result = run_analysis(QUESTION, _settings(planner_selects_agents=False))
        self.assertTrue(result.ok)
        self.assertTrue(result.plan.ok)
        self.assertEqual(len(result.specialists), 5)
        self.assertIsNotNone(result.risk)
        self.assertIsNotNone(result.strategy)
        self.assertIn("Executive Analysis", result.report_markdown)
        self.assertGreater(result.usage["calls"], 0)

    def test_evaluation_runs_when_enabled(self):
        result = run_analysis(QUESTION, _settings(run_evaluation=True, enabled_specialists=["finance"]))
        self.assertIsNotNone(result.evaluation)
        self.assertTrue(result.evaluation.ok)
        self.assertIn("overall_score", result.evaluation.data)

    def test_restricting_specialists_is_respected(self):
        result = run_analysis(
            QUESTION, _settings(enabled_specialists=["finance"], planner_selects_agents=False)
        )
        self.assertEqual([r.role for r in result.specialists], ["finance"])

    def test_run_is_serialisable(self):
        result = run_analysis(QUESTION, _settings(enabled_specialists=["finance"]))
        payload = result.as_dict()
        self.assertEqual(payload["question"], QUESTION)
        self.assertIn("report_markdown", payload)
        import json

        json.dumps(payload)  # must not raise


class PartialFailureTests(unittest.TestCase):
    def test_one_dead_specialist_does_not_sink_the_run(self):
        real = __import__("analyst.agents", fromlist=["run_specialist"]).run_specialist

        def flaky(backend, settings, role, *args, **kw):
            if role == "finance":
                return AgentResult(role=role, ok=False, error="AgentCallError: boom")
            return real(backend, settings, role, *args, **kw)

        with mock.patch("analyst.graph.agents.run_specialist", side_effect=flaky):
            result = run_analysis(
                QUESTION,
                _settings(enabled_specialists=["finance", "sales"], planner_selects_agents=False),
            )

        by_role = {r.role: r for r in result.specialists}
        self.assertFalse(by_role["finance"].ok)
        self.assertTrue(by_role["sales"].ok)
        # The run still reaches a recommendation, and the report says what broke.
        self.assertTrue(result.ok)
        self.assertIn("Agent failed", result.report_markdown)


class BudgetAbortTests(unittest.TestCase):
    """A tripped ceiling must abort the run *and* keep the finished work.

    Before the fix, BudgetExceeded was swallowed by the agent-level catch-all,
    so the graph kept dispatching and never set `aborted`.
    """

    def test_budget_abort_is_recorded_and_partial_work_survives(self):
        # 1 planner + 2 specialists, then the ceiling bites at the risk agent.
        result = run_analysis(
            QUESTION,
            _settings(
                max_llm_calls=3,
                enabled_specialists=["finance", "sales"],
                planner_selects_agents=False,
            ),
        )
        self.assertTrue(result.aborted, "the run should record why it stopped")
        self.assertIn("max_llm_calls", result.aborted)
        self.assertFalse(result.ok)
        # The two specialists that completed are still in the report.
        self.assertEqual(len(result.specialists), 2)
        self.assertIn("Run stopped early", result.report_markdown)
        self.assertIn("Specialist findings", result.report_markdown)

    def test_ceiling_of_one_stops_after_the_planner(self):
        result = run_analysis(QUESTION, _settings(max_llm_calls=1))
        self.assertTrue(result.aborted)
        self.assertEqual(result.usage["calls"], 1)

    def test_budget_exceeded_propagates_out_of_an_agent(self):
        """The specific regression: it must not be caught as an agent failure."""
        from analyst import agents

        backend = mock.Mock()
        backend.complete.side_effect = BudgetExceeded("ceiling")
        with self.assertRaises(BudgetExceeded):
            agents.run_specialist(
                backend, _settings(), "finance", QUESTION, "task", "why", "evidence", []
            )


class MissingDataTests(unittest.TestCase):
    def test_incomplete_data_room_fails_loudly(self):
        room = mock.Mock()
        room.missing.return_value = ["finance/income_statement.csv"]
        with self.assertRaises(FileNotFoundError) as ctx:
            run_analysis(QUESTION, _settings(), room=room)
        self.assertIn("make_sample_data.py", str(ctx.exception))


class DegradedRenderingTests(unittest.TestCase):
    def test_report_renders_with_nothing_but_a_failed_plan(self):
        result = AnalysisResult(
            question=QUESTION,
            settings=_settings(),
            plan=AgentResult("planner", False, error="boom"),
        )
        markdown = render_report(result)
        self.assertIn("Executive Analysis", markdown)
        self.assertIn("Not available", markdown)


if __name__ == "__main__":
    unittest.main()