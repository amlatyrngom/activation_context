"""Fig. 3, variant B (user's pick, Oct 27) with the measured ideal-recall points (eval500, mean of 3 passes; golds injected =
perfect recall@k; Text = base 9B thinking off, AC = Arm G 1/10). Points at k = 200 are added when those runs finish.
Ideal tokens: text = rq_text_eval prompt tokens (20i 2,402, 100i 11,996); AC = the frontier's tok_counts values for the same k.
Run: uv run --no-project --python 3.13 --with matplotlib python make_fig3_b_real.py
"""
import make_fig3_mocks as M
from make_fig3_mocks import R, RQ

M.IDEAL = dict(text_tok=[2.402, 11.996], text_acc=[66.33, 64.07], text_k=[20, 100],
               ac_tok=[0.331, 1.306], ac_acc=[67.13, 63.67], ac_k=[20, 100])
M.stamp = lambda fig: None
M.mock_b()
import shutil
for ext in ("png", "pdf"):
    shutil.move(M.OUT / f"mock_B_log_single.{ext}", M.OUT / f"fig3_B_real_partial.{ext}")
