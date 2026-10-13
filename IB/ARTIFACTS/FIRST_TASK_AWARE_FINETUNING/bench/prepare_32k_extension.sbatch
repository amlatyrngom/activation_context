#!/bin/bash
#SBATCH --job-name=freeze_27b_32k
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/prepare_model_comparison.py" \
    --bank-9b "$QA_ROOT/reports/synthesis_bank_v2" \
    --bank-27b "$QA_ROOT/reports/synthesis_bank_27b" \
    --output "$QA_ROOT/model_comparison_32k_material" --items 32000 \
    --extend-from "$QA_ROOT/model_comparison_material/material_27b_16000.json" \
    --prior-material "$QA_ROOT/model_comparison_material/material_9b_16000.json" \
    --reserved-material "$QA_ROOT/hillclimb_material/confirmation.json" \
    --reserved-material "$QA_ROOT/data_budget_material/material_2000.json" \
    --reserved-material "$QA_ROOT/data_budget_material/confirmation.json" \
    --reserved-material "$QA_ROOT/reports/preparation/material.json" \
    --reserved-material "$QA_ROOT/model_comparison_material/confirmation.json" \
    --reserved-material "$QA_ROOT/qa27b_reserved_questions.json"
"$HOME/activation_venv/bin/python" - "$QA_ROOT" <<'PYCODE'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);target=root/'qa27b_reserved_questions.json'
old=json.loads(target.read_text()) if target.exists() else {'questions': []}
new=json.loads((root/'model_comparison_32k_material/reserved_questions.json').read_text())
questions=sorted(set(old['questions']) | set(new['questions']))
temporary=target.with_suffix('.next.json')
temporary.write_text(json.dumps({'questions':questions},ensure_ascii=False))
temporary.replace(target)
print('Reserved questions published:',len(questions))
PYCODE
