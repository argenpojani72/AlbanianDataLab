"""
tests/test_run_pipeline.py — Unit tests for scripts/run_pipeline.py.

Tests pure logic: STEPS definitions, step-range validation, model injection,
and step filtering — no subprocesses are actually launched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_pipeline import STEPS


# ---------------------------------------------------------------------------
# STEPS definitions
# ---------------------------------------------------------------------------

class TestStepsDefinitions:
    def test_five_steps_defined(self):
        assert len(STEPS) == 5

    def test_step_numbers_sequential(self):
        numbers = [s["number"] for s in STEPS]
        assert numbers == [1, 2, 3, 4, 5]

    def test_each_step_has_required_keys(self):
        required = {"number", "name", "script", "description", "extra_args"}
        for step in STEPS:
            missing = required - set(step.keys())
            assert not missing, f"Step {step.get('number')} missing keys: {missing}"

    def test_script_names_exist_in_scripts_dir(self):
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        for step in STEPS:
            p = scripts_dir / step["script"]
            assert p.exists(), f"Script not found: {p}"

    def test_step4_is_transcribe(self):
        step4 = next(s for s in STEPS if s["number"] == 4)
        assert "transcribe" in step4["script"]

    def test_step5_is_build_text(self):
        step5 = next(s for s in STEPS if s["number"] == 5)
        assert "text" in step5["script"]

    def test_extra_args_are_lists(self):
        for step in STEPS:
            assert isinstance(step["extra_args"], list)


# ---------------------------------------------------------------------------
# Step selection logic (mirrors main() logic from run_pipeline.py)
# ---------------------------------------------------------------------------

def _selected_steps(from_step: int, to_step: int) -> list[int]:
    """Return numbers of steps that would run given from/to bounds."""
    return [s["number"] for s in STEPS if from_step <= s["number"] <= to_step]


class TestStepSelection:
    def test_default_selects_all_five(self):
        assert _selected_steps(1, 5) == [1, 2, 3, 4, 5]

    def test_audio_only_selects_1_to_4(self):
        assert _selected_steps(1, 4) == [1, 2, 3, 4]

    def test_text_only_selects_step_5(self):
        assert _selected_steps(5, 5) == [5]

    def test_from_step_3(self):
        assert _selected_steps(3, 5) == [3, 4, 5]

    def test_to_step_3(self):
        assert _selected_steps(1, 3) == [1, 2, 3]

    def test_single_step(self):
        assert _selected_steps(2, 2) == [2]

    def test_out_of_range_returns_empty(self):
        assert _selected_steps(6, 6) == []

    def test_reversed_range_returns_empty(self):
        assert _selected_steps(4, 2) == []


# ---------------------------------------------------------------------------
# Model injection into step 4
# ---------------------------------------------------------------------------

class TestModelInjection:
    def test_model_injected_into_step4(self):
        """
        Simulate what main() does: set extra_args on step 4 when --model is given.
        """
        import copy
        steps = copy.deepcopy(STEPS)
        model = "ard-ali/whisper-small-albanian"
        for s in steps:
            if s["number"] == 4:
                s["extra_args"] = ["--model", model]

        step4 = next(s for s in steps if s["number"] == 4)
        assert step4["extra_args"] == ["--model", model]

    def test_no_model_leaves_extra_args_empty(self):
        """Without --model, step 4 extra_args should stay empty."""
        step4 = next(s for s in STEPS if s["number"] == 4)
        # The module-level STEPS have no model injected by default
        assert step4["extra_args"] == []
