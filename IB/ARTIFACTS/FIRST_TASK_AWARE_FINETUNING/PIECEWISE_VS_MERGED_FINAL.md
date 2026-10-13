# Passagewise versus merged encoding: completed matched runs

**Independent passage encoding finishes at 0.916 answer NLL, versus 1.303 for merging all passages.** Both used the same reconstruction checkpoint, 4,000 unique questions × 1 epoch, about 8k evidence including distractors, equal total compressed rows at 1/16, and matched training settings and question order.

| Measurement | Passagewise | Merged |
|---|---:|---:|
| Initial held400 answer NLL | 3.173 | 2.108 |
| Final held400 answer NLL | **0.916** | **1.303** |
| Initial fixed32 training-answer NLL | 2.834 | 1.856 |
| Final fixed32 training-answer NLL | **0.553** | **0.980** |
| Final paired-reader full-context NLL | 0.590 | 0.591 |
| Final paired-reader no-context NLL | 1.806 | 1.804 |
| Questions dropped | 0 | 0 |

Passagewise encoding starts worse but learns substantially better in this experiment. The training-panel advantage also points toward layout or optimization, rather than establishing an insufficient-capacity explanation. The small fixed32 panel is a diagnostic, not the entire training set. Neither run establishes semantic parity with full context.

The original P2 full-context reference is 0.774 NLL for both. The final full/no-context rows above use each run’s trained reader; they are not frozen-reference scores.

The passagewise final has already produced 100 source-reviewed answers. The merged final’s independent 400-question replay, same100 answers and32k readout are queued. The source-reviewed semantic comparison remains pending.

Merged training suffered several preemptions and retained paired weights, Adam state and the original schedule on resume. Its current duration/token counters cover only the last attempt; they must not be compared with the uninterrupted passagewise total. One run per arm and repeated development measurement limit generalization.
