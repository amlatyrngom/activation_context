"""
The code bundle the math recipe's cluster jobs run (2026-11-17): sec35/jobs/make_bundle.py's files (the `activation` package of this
working tree, pyproject.toml, uv.lock, the sec35 kit, rq_kit, bench, compaction cmp_*.py) plus math/: this folder's *.py and the
teacher-test helpers they import. One tar.gz + MANIFEST.json (sha256 of every file, git HEAD); checked file by file on arrival.

    .venv/bin/python make_math_bundle.py --out IB/TMP/PRE_REBUTTAL/math_recipe/code/math_code_v1.tar.gz
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "sec35/jobs"))
import make_bundle  # noqa: E402

make_bundle.FILES["math"] = sorted(HERE.glob("*.py")) + [HERE.parent / "teacher_test/build_prompts.py"]
# The vLLM plugin entry point of activation.vllm_residuals (pyproject.toml) for venvs built with --no-install-project: a dist-info on
# PYTHONPATH (math_job.sh eval adds W/code/BUNDLE/ep only when the venv does not already provide the entry point).
make_bundle.FILES["ep/ac_residuals_ep-0.1.dist-info"] = sorted((HERE / "ep/ac_residuals_ep-0.1.dist-info").iterdir())

if __name__ == "__main__":
    make_bundle.main()
