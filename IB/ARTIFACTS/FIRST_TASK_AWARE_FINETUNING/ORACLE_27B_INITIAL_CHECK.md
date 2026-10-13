# Initial 27B oracle check

Manual review of **20 accepted pairs found one definite algebra error and one unsupported transfer of a condition between groups**. Two other pairs have weaker qualification concerns. No temporary Source-number labels appeared in these 20 pairs. The four code examples had the required branch/test/schema evidence; no missing-definition guess was needed.

This supports a small checker A/B, not stopping or rewriting the bank. It does not estimate the population error rate or establish that 27B beats 9B. Source-grounded answers can still inherit weaknesses in their source.

Selection was frozen before reading quality: first ten committed microchunks, containing 246 accepted questions across all nine families; four code and two per other family, ordered by SHA-256 of a fixed seed plus question ID, with distinct source requests. The snapshot manifest was updated at 2026-09-21 21:18:53 UTC. Full IDs, file hashes, line numbers, questions, answers and findings are in `ORACLE_27B_INITIAL_CHECK.json`; frozen source text and selection receipt are under `IB/TMP/FIRST_TASK_AWARE_FINETUNING/oracle_27b_initial_check/`. No model/tool judge, GPU call, production edit or external source audit was used.

## Concrete grounding failures

**`fb62521293af82fce4825c439c2dc7af5ff8a323cbabc759dc47b1cf54bb5d74` — incorrect algebra and question premise.** The arXiv 2006.10264 excerpt defines the derivative confidence-interval denominator as

\[
\sqrt{n(\widehat v-\widehat u)^3}=n^{1/2}(\widehat v-\widehat u)^{3/2}.
\]

The gold says this “corresponds to `(n(v̂ − û))^(3/2)`,” which instead has power **3/2 on n**, making it too large by a factor of n. The question reinforces the error by asking for a single power of the product `n(v̂ − û)`. A sound question would ask for the powers of n and the interval width separately; the answer is **1/2 and 3/2**. This is a definite mathematical error, not notation preference.

**`9eee3d85aad34ecf4ed9d6c7f304baabda08175d1578168c8e2a3caa8f273bb0` — unsupported relationship.** The office source recommends private one-on-one discussion **“In some cultures”**, then says **“In other cultures”** employees are uncomfortable discussing supervisors because of hierarchy. The question asks which setting is recommended specifically for that latter group; the gold answers private one-on-one. The source never makes that connection. The advice may be plausible in practice, but the claimed source support is absent.

## Weaker concerns

- `1cf2899cfafd…`: Read the Docs variable/condition answer matches the template exactly, but its source explicitly describes **not-yet-implemented design features**. The question presents the injected context without that qualifier.
- `a9c82a4175dd…`: PEP 416 rejects a sandbox use case as outside the standard-library proposal's scope; the question calls frozendict technically “unsuitable.” The answer captures the stated policy concern, so this is imprecise framing rather than a definite false answer.
- `9f44c605aa6d…`: Julia's statistics/linear-algebra answer is explicit in the article. The unnecessary Python/external-library contrast is also inherited from that article; treat its breadth as a source-quality caveat, not a new generator hallucination. Omitting that preamble would avoid it.

## All 20 decisions

ID prefixes resolve to full records in the JSON companion.

| # | Family / ID prefix | Decision and evidence |
|---:|---|---|
| 1 | code `420410000464` | Supported: matching query parameter is skipped; null values return before re-addition. |
| 2 | code `12927f4df001` | Supported: `StackSize.DOUBLE → Opcodes.POP2`; nonzero branch verifies `visitInsn`. |
| 3 | code `f3435e38d95f` | Supported: importing `DUMMY_MODULE_2` makes the mocked watcher registration run again. |
| 4 | code `c577c5fa9594` | Supported: Azure secret is `connection_string`; required fields are `container` and `connection_string`. |
| 5 | documents `1cf2899cfafd` | Qualification concern: template condition/repository path correct; proposal status omitted. |
| 6 | documents `a9c82a4175dd` | Qualification concern: sandbox policy rejection becomes “unsuitable” in question. |
| 7 | financial `4ce05abe1b31` | Supported: Greenwich net $299,116 / 27,117 shares ≈ $11.03; correctly uses net proceeds. |
| 8 | financial `890b026dc1e6` | Supported: $2,500,000 + $1,520,000 = $4,020,000, also explicitly stated in DynaResource's filing. |
| 9 | legal `96d22fb3ec15` | Supported: paragraph 20 rejects reliance on similar claims without individual forum connections. |
| 10 | legal `fddf0f7d4ecd` | Supported: discretionary duties and good-faith judgment standard both appear in Haney opinion. |
| 11 | arXiv `ee5bf06b11ac` | Supported: cell compatibility explicitly requires equality of the two cell-map values. |
| 12 | arXiv `fb62521293af` | **Incorrect:** square-root expansion changes exponent of n. |
| 13 | textbook `83bfa59ab9c7` | Supported: Windows spreadsheet paste into R editor produces tab-delimited text. |
| 14 | textbook `90ae4933a673` | Supported: different intensions, same `{Mom,Dad}` extension, hence equal sets in stated domain. |
| 15 | office `9eee3d85aad3` | **Unsupported:** private-dialogue recommendation transferred between differently scoped groups. |
| 16 | office `78c604263ec9` | Supported: $5.40 − $3.60 = $1.80; 800 pizzas reduce fixed cost per unit. |
| 17 | technical web `0e84baa71217` | Supported: SciPy 1.12 C++ implementation and `threadpoolctl` override are explicit. |
| 18 | technical web `9f44c605aa6d` | Source-supported with caveat: broad Python contrast is inherited and unnecessary. |
| 19 | web `3a34a1143a56` | Supported: epidermis persists with primary growth; apical meristems produce those tissues. |
| 20 | web `7e325519091a` | Supported identification: the supplied SSRI passage names serotonin. |

## Minimal next checker comparison

Suggested addition to the current instruction: **“Independently expand each power/product/root in the proposed answer and compare the exponent of every factor with the source; do not accept the question's suggested simplification. Preserve which population or condition each source statement applies to; adjacent statements about different groups do not establish a relationship.”**

For the coordinator's proposed same-model, thinking-off A/B, compare the current boolean-only output against one brief evidence/calculation field **before** the verdict, without another checker pass. For example:

`{"check":"Source: sqrt(n Δ³)=n^(1/2) Δ^(3/2); proposed (nΔ)^(3/2) has n^(3/2), so differs by factor n.","supported":false}`

Keep that field short and factual. Test whether the change catches these failures while retaining supported pairs, and record added time. Change future recipe only if the small comparison helps; these 20 judgments alone do not justify a broader validation framework or a retrospective bank purge.
