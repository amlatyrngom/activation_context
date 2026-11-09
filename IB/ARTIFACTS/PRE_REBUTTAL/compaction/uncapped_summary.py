"""Compaction summary baseline without the word cap (pre-rebuttal rerun, 2026-11-09).

The paper's summary arm asked for a handoff "under 500 words" (activation/common/ac_parts.py COMPACTION_SUMMARY_PROMPT,
shown to the model as the compact tool's description). This variant removes only that limit: "concise" and the five
sections stay. Selected in run_resume.py with --uncapped-summary.
"""
from activation.agent.agent_tools import CompactionTool
from resume_program import CaptureSummaryTool

CAPPED = "Keep the entire handoff under 500 words, prioritizing details needed to resume."
UNCAPPED = "Prioritize details needed to resume."
assert CompactionTool.description.count(CAPPED) == 1


class UncappedCaptureSummaryTool(CaptureSummaryTool):
    """CaptureSummaryTool whose description (the summary prompt) has no word limit."""
    description = CompactionTool.description.replace(CAPPED, UNCAPPED)
