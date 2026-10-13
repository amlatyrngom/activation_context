# How the continuation metrics disagree

The constant-LR 42k checkpoint remains the next-run parent. It answers 71/87 source-supported questions correctly, versus 62/87 after rewarming. Both trained once on the same 10,000 additional questions from the same 32k checkpoint, with inherited Adam state.

| Native 2k evaluation | Constant LR | Rewarm |
|---|---:|---:|
| All 400 development questions, gold NLL | 0.5327 | **0.5248** |
| Selected 100 questions, gold NLL | **0.5544** | 0.5609 |
| Remaining 300 questions, gold NLL | 0.5254 | **0.5128** |
| Source-supported answers fully correct, selected cohort | **71/87** | 62/87 |
| Learning panel, 32 fixed training questions, gold NLL | 0.5186 | **0.4407** |

Part of the apparent disagreement is between samples: NLL itself ranks the checkpoints differently on the selected100 and the other300. We have not reviewed answers for those other300, so the 87-question result does not establish a population-wide winner.

Within the reviewed cohort, rewarming gains two fully correct answers and loses eleven. Eight losses have worse gold NLL too; three have better NLL despite an incomplete or incorrect generated answer. Nine losses become partial answers, and two become incorrect answers. All100 generated replies terminate normally, with no token caps.

Gold NLL scores the supplied gold continuation under its supplied prefix. Greedy generation follows the model's own chosen prefix and can omit a required detail or choose another branch. The two measurements therefore need not agree item by item. These data support that distinction; they do not isolate a specific internal cause.

The low training NLL alone does not prove harmful overfitting: the rewarm checkpoint also improves the full400 held-out NLL. At matched8k evidence, however, constant LR wins both held NLL (0.6219 versus0.6278) and reviewed correctness (61/87 versus54/87). That makes it the better-supported practical parent for the two current follow-ups.

All measurements are repeated development measurements from one run per schedule. The reserved confirmation set was not accessed for this choice. Raw paired decomposition: [CONTINUATION_NLL_SEMANTICS_V1.json](CONTINUATION_NLL_SEMANTICS_V1.json). Actual answers: [native2k](CONTINUATION_COHORT100.tressoir.html), [matched8k](STRONG_LONG_COHORT100.tressoir.html).
