# Evidence position in the frozen 32k comparison

There is an early-position advantage in merged answer NLL and a middle-position dip in supported answers. The association is exploratory: different questions occupy each position, and separate passage encoding does **not** remove the semantic dip. These results do not establish a capacity limit or justify reranking evidence.

This uses the existing control900 checkpoint, identical evidence and reader weights, equal total 1/16 rows, and no additional training. Exact source text and reconstructed full inputs match the original material. All 100 questions have one intended source unit; passage counts range from 13 to 64. Position is normalized as the zero-based passage index divided by passage count minus one.

Item 024 contains its exact source twice, at passages 41 and 49 of 63. The seeded shuffle identifies 41 as the inserted source, but a model sees both occurrences. The primary analysis excludes this item (99 questions); the JSON also records an all-100 sensitivity. No other exact source-position match is ambiguous. This locates the designated source, not every possible supporting passage.

| Intended source position | Questions | Merged NLL | Separate NLL | Full-context NLL |
|---|---:|---:|---:|---:|
| Early third | 34 | 0.645 | 0.747 | 0.343 |
| Middle third | 26 | 1.041 | 0.867 | 0.387 |
| Late third | 39 | 0.977 | 0.814 | 0.393 |

Literal gold NLL includes reference concerns. The semantic table instead reuses the existing adjusted reference set: 85 originally eligible, 84 after excluding the duplicated source. “Supported” retains the earlier strict correctness judgment; partial answers are not counted as correct.

| Intended source position | Merged supported | Separate supported | Full-context supported |
|---|---:|---:|---:|
| Early third | 21/30 | 20/30 | 30/30 |
| Middle third | 8/21 | 6/21 | 21/21 |
| Late third | 22/33 | 20/33 | 31/33 |

The middle bin is difficult for both layouts, while full context answers all 21 eligible middle questions correctly. Within code, merged support is 7/9, 2/7, 5/8 across early/middle/late; numeric is 5/7, 1/5, 6/9. Thus the dip is not solely a different domain mix. These cells are small: the document middle bin has only 3 questions and wiki has 5 total.

The continuous position–NLL correlation is weak: Pearson r=0.175 merged, 0.031 separate, 0.074 full. Against each question’s full-context NLL, the corresponding gap correlations are 0.143 and −0.003. A check weighted by recorded per-part encoder lengths gives the same broad pattern: merged NLL 0.679/0.962/1.001 versus separate 0.774/0.809/0.829. That token-position proxy includes repeated question/template overhead; it is not an exact merged-token offset. Including item 024 gives middle merged NLL 1.020 and supported responses 8/22.

A useful next diagnostic is a small frozen-weight order test: choose 12 questions by fixed domain quotas without using their losses or answer locations, then apply three predetermined cyclic rotations to **every** passage list. Compare merged, separate and full context within each question, keeping weights, evidence and row budgets fixed; retain greedy answers as well as gold NLL. This tests order sensitivity without gold-aware reranking. It is a proposal only; no job or training was started.

All existing metrics, source annotations and benchmark questions remain unchanged. No broad distractor or corpus audit was performed. Full per-item positions, original/adjusted denominators, domain bins and sensitivity checks are in [PASSAGE_POSITION_32K.json](PASSAGE_POSITION_32K.json); original responses remain in [the 100-question report](PASSAGE_LAYOUT32K_COHORT100.tressoir.html).
