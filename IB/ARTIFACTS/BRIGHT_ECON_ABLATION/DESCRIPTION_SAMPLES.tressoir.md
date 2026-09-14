# Description probe — what gets a description and what does not

Fourteen chunks from the 512-chunk description probe (4B FP8 generator, thinking off), eight that the generator marked as needing no abstraction bridge and six that it described, drawn at random with the chunk text. The finding: the opt-out is mostly hitting BRIGHT economics' fragment chunks, and it is not a reliable junk detector.

## Numbers

| quantity | value |
| --- | --- |
| corpus documents under 100 chars | 46 % (under 50: 31 %; under 200: 63 %) |
| corpus documents with fewer than 25 words | 61 % |
| probe: skipped chunks, median length | 107 chars (357 chunks) |
| probe: kept chunks, median length | 319 chars (155 chunks) |

Kept rate by chunk length in the probe:

| chunk length | chunks | described |
| --- | --- | --- |
| under 100 chars | 235 | 65 (28 %) |
| 100–200 | 88 | 7 (8 %) |
| 200–400 | 75 | 13 (17 %) |
| 400–1,000 | 67 | 33 (49 %) |
| 1,000–4,096 | 47 | 37 (79 %) |

So the "no bridge" rate rises as the text gets shorter, which is the right direction, but 65 of the 155 descriptions are on chunks under 100 characters, and those descriptions are vacuous (below). Real prose of 400 characters or more gets described about 60 % of the time.

## Samples

### SKIPPED — russia_sanction_oil/indexhtml_131.txt:0 (11 chars)

> Now playing

### SKIPPED — public_debt_default/publicdebtfaqs_91.txt:0 (77 chars)

> Has Treasury ever suspended reinvestment of all or part of the G Fund before?

### SKIPPED — omo_money_supply/0205bennhtml_37.txt:0 (561 chars)

> * [ Services For Financial Institutions ](/banking/services "Services For Financial Institutions") * [ Payment Services ](/banking/payment_services.html "Payment Services") * [ Payment System Oversight ](/banking/payment_oversight.html "Payment System Oversight") * [ International Services, Seminars & Training ](/banking/international.html "International Services, Seminars & Training") * [ Financial Market Infrastructure & Reform ](/financial-services-and-infrastructure/financial-market-infrastructure-and-reform "Financial Market Infrastructure & Reform")

### SKIPPED — elastic_substitution/ugfinalpdf_20.txt:0 (245 chars)

> 2 62 0 94 0 94 2 62 1062 0 0 0 0 105 0 0 0 0 0 93 0 0 0 0 0 218 0 048 0 048 24 26 0 0 0 2301 2 50 136 0 056 110 2 62 0 94 1 1 . . . . . . . . . . . . . ~ ~ ~ ~ . . . . . . . . . . . . . . L 0 94 2 62 1 1 1 1 . . ~ ~ ~ ~ Ê Ë Á Á Á Á ˆ ¯ ˜ ˜ ˜ ˜ Ê

### SKIPPED — russia_rich/Resourcecurse_139.txt:0 (4096 chars)

> 41. ** ^  ** Palley, Thomas I. (December 2003). [ "Lifting the Natural Resource Curse" ](https://www.globalpolicy.org/component/content/article/198/40112.html) . _Foreign Service Journal_ . 42. ** ^  ** o'Brochta, William (2019). [ "A meta-analysis of natural resources and conflict" ](https://doi.org/10.1177%2F2053168018818232) . _Research & Politics _ . **6** : 205316801881823. [ doi ](/wiki/Doi_\(identifier\) "Doi \(identifier\)") :  [ 10.1177/2053168018818232 ](https://doi.org/10.1177%2F2053168018818232) . 43. ** ^  ** Koubi, Vally; Spilker, Gabriele (2017-06-28). "Natural Resources, Climate Change, and Conflict". _Oxford Research Encyclopedia of Politics_ . Vol. 1. [ doi ](/wiki/Doi_\(id…

### SKIPPED — freeze_gemany_japan/AAC00_59.txt:0 (36 chars)

> 5.1.2 Dissemination media and format

### SKIPPED — adjustinflation/inflationadjustment2023_12.txt:0 (379 chars)

> Salary increases are crucial for recognizing employee performance and achievements, but inflation can often make these increases insufficient. For many employees, stagnant salaries have become a reality and salary increase requests are often denied. Inflation adjustment provides a solution to this issue and can help ensure that annual salary increases are granted by employers.

### SKIPPED — valuepriceprofit/S0304393221000040_1.txt:0 (709 chars)

> Search ScienceDirect Search ScienceDirect Outline Highlights Abstract Keywords JEL classification 1. Introduction 2. A tale of two TANK models 3. Household heterogeneity and fiscal policy 4. Conclusion Appendix A. Supplementary materials Research Data References Show full outline Cited by (29) Figures (7) Fig. 1. MPCs in the Data Fig. 2. Theoretical iMPCs for an unanticipated income windfall Fig. 3. Theoretical iMPCs for an anticipated income windfall Fig. 4. Empirical effects of an unanticipated shock to government spending (U Fig. 5. Government spending shocks in alternative TANK models Fig. 6. Fiscal stimulus effects in simple models Show 1 more figure Tables (3) Table 1 Table 2 Table 3 E…

### KEPT — interest_rate_parity/2798_110.txt:0 (11 chars)

> Background:

**Description:** Contextual background information establishing the premise or setting for subsequent discussion.

### KEPT — ces_production/meaningofproductionf_119.txt:0 (5 chars)

> * * *

**Description:** Empty or placeholder content with no substantive information.

### KEPT — decoy_effect/Decoyeffect_8.txt:0 (920 chars)

>   Debate  [  [ edit  ](/w/index.php?title=Decoy_effect&action=edit&section=3 "Edit section: Debate") ]  Some research suggests that the attraction effect does not appear in realistic purchasing scenarios, for example when options are presented graphically, or when the target and the competitor are not exactly of the same value.  [6] [7]  [5]  The original authors had to underline again that the attraction effect occurs only if the consumer is close to indifference between the target and the competitor, if both dimensions of the products (in our example, price and storage capacity) are about as important as each other to the consumer, if the decoy is not too undesirable, and if the dominance …

**Description:** Conditions under which the attraction effect persists in realistic purchasing scenarios despite graphical presentations or value differences between options.

### KEPT — cpi_shift/articleA001enxml_395.txt:0 (7 chars)

> Figures

**Description:** Visual data representations (such as charts, graphs, diagrams, or illustrations) used to convey information or concepts.

### KEPT — exchange_cbdc/S2214845020300351_1.txt:0 (1586 chars)

> Search ScienceDirect Search ScienceDirect Outline Abstract Keywords JEL classification 1. Introduction 2. Succinct empirical review 3. Methodology and data 4. Empirical results 5. Summary and conclusion Appendix. Data Description and Source References Show full outline Cited by (6) Figures (1) Fig.1. Trend of factors Tables (10) Table 1 Table 2 Table 3 Table 4 Table 5 Table 6 Show all tables Borsa Istanbul Review Borsa Istanbul Review Volume 20, Supplement 1, December 2020, Pages S81-S92 Borsa Istanbul Review Review Global financial cycles and exchange rate forecast: A factor analysis Author links open overlay panelIbrahim D. Raheem Show more Add to Mendeley Share Cite https://doi.org/10.101…

**Description:** Exchange rate forecasting using portfolio balance theory and Global Financial Cycle (GFCy) factors; compares against random walk benchmark; highlights superior performance at short-term horizons (1 and 4 quarters); utilizes a dataset of 20 advanced and emerging countries from 1990Q1 to 2017Q2.

### KEPT — diminish_return/Giventhelawofdiminishingreturnsandopportunitycostsdoesiteconomicallymakesensetoeradicateadisease_22.txt:0 (395 chars)

> However this is harder than you think. There are hundreds of medium to large pharmaceutical companies and thousands of start ups all over the world, involving hundreds of thousands of people. You can't keep a secret for long with so many people involved nor can you force every company to keep it secret. All it takes is just one hint that that there is a cure and the public will clamor for it.

**Description:** Challenges of maintaining trade secrets in large pharmaceutical ecosystems involving hundreds of thousands of employees across numerous companies and startups.
