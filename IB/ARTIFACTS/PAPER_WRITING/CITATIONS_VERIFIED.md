# Verified citations for the Activation Context paper (checked 2026-10-26)

How I checked: every arXiv field (title, authors, first-submission date, primary class, author comments) comes from the arXiv metadata API (export.arxiv.org/api/query?id_list=...) or an arxiv.org/abs page. A venue is given only when the authors' own arXiv comment or a proceedings or venue page states it. Otherwise the entry stays `@misc` (arXiv), even when I believe a venue exists. For `@inproceedings` entries the year is the venue year, and the arXiv eprint is kept.

Status legend: VERIFIED = every ID and fact you supplied matched. CORRECTED = something you supplied or implied was wrong or incomplete. NOT FOUND = could not be confirmed, so no entry is given.

---

## Part 1(a) — Works cited by WHALE (arXiv 2609.00196)

WHALE itself: VERIFIED. "WHALE: A Simple Recipe for Joint Harness-Weight Optimization", Haechan Kim, Yoonho Lee, Gisang Lee, Chelsea Finn, Kangwook Lee. Submitted 31 Aug 2026 (v1), cs.LG. Note: the full title starts with "WHALE:".

```bibtex
@misc{kim2026whalesimplerecipejoint,
      title={WHALE: A Simple Recipe for Joint Harness-Weight Optimization},
      author={Haechan Kim and Yoonho Lee and Gisang Lee and Chelsea Finn and Kangwook Lee},
      year={2026},
      eprint={2609.00196},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.00196},
}
```

What WHALE cites in each role. I read the reference list from arxiv.org/html/2609.00196v1 and then checked every ID against the arXiv metadata API.

| Role in WHALE | Work | arXiv ID |
|---|---|---|
| Meta-Harness (the harness-search phase) | Lee, Nair, Zhang, Lee, Khattab, Finn, "Meta-Harness: End-to-End Optimization of Model Harnesses" (2026) | 2603.28052 |
| "Fast–Slow Training" baseline | Tiwari et al., "Learning, Fast and Slow: Towards LLMs That Adapt Continually" (2026) | 2605.12484 |
| Harness-only optimization / agent-design search | Hu, Lu, Clune, ADAS (WHALE lists it twice, as "2024" and "2025": a duplicate bib entry in WHALE) | 2408.08435 |
| | Lin et al., "Agentic Harness Engineering..." (2026) | 2604.25850 |
| | Sengupta & Wang, "HARBOR: Automated Harness Optimization" (2026) | 2604.20938 |
| | Pan et al., "Evolving Agents in the Dark..." (2026) | 2606.05922 |
| | Park et al., "AutoSaddler..." (2026) | 2608.23041 |
| | Zhang et al., AFlow (2024) | 2410.10762 |
| | Shang et al., AgentSquare (2024) | 2410.06153 |
| | Zhuge et al., "Language Agents as Optimizable Graphs" (2024) | 2402.16823 |
| | (prompt-level) Khattab et al., DSPy; Yang et al., "LLMs as Optimizers"; Agrawal et al., GEPA | 2310.03714; (OPRO not checked); 2507.19457 |
| Weight-only agent post-training | Chen et al., FireAct (2023) | 2310.05915 |
| | Feng et al., ReTool (2025) | 2504.11536 |
| | Jin et al., Search-R1 (2025) | 2503.09516 |
| | Dong et al., RAFT (2023, TMLR) | 2304.06767 |
| | Gulcehre et al., ReST (2023) | 2308.08998 |
| | Zelikman et al., STaR (2022) | 2203.14465 |
| | (also cited: WebGPT, Singh et al. "Beyond human data", DeepSeekMath; not checked) | — |
| Joint weight + prompt adaptation | Soylu, Potts, Khattab, "Fine-Tuning and Prompt Optimization..." (2024) | 2407.10930 |
| | Bo et al., "Prompt and Parameter Co-Optimization for LLMs" (ICLR 2026) | 2509.24245 |
| | Lu et al., "P²O: Joint Policy and Prompt Optimization" (2026) | 2603.21877 |
| | Zhang, Chen, Stadie, "Evolutionary System Prompt Learning for RL in LLMs" (2026) | 2602.14697 |
| | Ziems et al., "Composing Policy Gradients and Prompt Optimization for LM Programs" (ACM CAIS 2026) | 2508.04660 |
| | Agrawal et al., GEPA (ICLR 2026) | 2507.19457 |
| Model used | Qwen Team, "Qwen3.5: Towards Native Multimodal Agents" (blog) | none (blog) |

The BibTeX for the Part 1 works not repeated in Part 2 is below. Meta-Harness, ADAS, AFlow, DSPy, Soylu et al. and GEPA are in Part 2.

```bibtex
@misc{tiwari2026learningfastslowtowards,
      title={Learning, Fast and Slow: Towards LLMs That Adapt Continually},
      author={Rishabh Tiwari and Kusha Sareen and Lakshya A Agrawal and Joseph E. Gonzalez and Matei Zaharia and Kurt Keutzer and Inderjit S Dhillon and Rishabh Agarwal and Devvrit Khatri},
      year={2026},
      eprint={2605.12484},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.12484},
}

@misc{lin2026agenticharnessengineeringobservabilitydriven,
      title={Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses},
      author={Jiahang Lin and Shichun Liu and Chengjun Pan and Lizhi Lin and Shihan Dou and Zhiheng Xi and Xuanjing Huang and Hang Yan and Zhenhua Han and Tao Gui and Yu-Gang Jiang},
      year={2026},
      eprint={2604.25850},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.25850},
}

@misc{sengupta2026harborautomatedharnessoptimization,
      title={HARBOR: Automated Harness Optimization},
      author={Biswa Sengupta and Jinhua Wang},
      year={2026},
      eprint={2604.20938},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.20938},
}

@misc{pan2026evolvingagentsdarkretrospective,
      title={Evolving Agents in the Dark: Retrospective Harness Optimization via Self-Preference},
      author={Wenbo Pan and Shujie Liu and Chin-Yew Lin and Jingying Zeng and Xianfeng Tang and Xiangyang Zhou and Yan Lu and Xiaohua Jia},
      year={2026},
      eprint={2606.05922},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.05922},
      note={Accepted to Findings of EMNLP 2026 (per arXiv comment)},
}

@misc{park2026autosaddlerautomaticharnessoptimization,
      title={AutoSaddler: Automatic Harness Optimization with Durable Updates from Agent Execution Traces},
      author={Sungho Park and Wonjoong Kim and Rongyuan Tan and Jue Zhang and Wook-Shin Han and Pengfei Gao and Chanyoung Park and Yongqiang Yao and Rao Fu and Elsie Nallipogu and Qingwei Lin and Saravan Rajmohan and Dongmei Zhang},
      year={2026},
      eprint={2608.23041},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2608.23041},
}

@misc{shang2024agentsquareautomaticllmagent,
      title={AgentSquare: Automatic LLM Agent Search in Modular Design Space},
      author={Yu Shang and Yu Li and Keyu Zhao and Likai Ma and Jiahe Liu and Fengli Xu and Yong Li},
      year={2024},
      eprint={2410.06153},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.06153},
}

@inproceedings{zhuge2024languageagentsoptimizablegraphs,
      title={Language Agents as Optimizable Graphs},
      author={Mingchen Zhuge and Wenyi Wang and Louis Kirsch and Francesco Faccio and Dmitrii Khizbullin and J{\"u}rgen Schmidhuber},
      booktitle={Proceedings of the 41st International Conference on Machine Learning (ICML)},
      year={2024},
      eprint={2402.16823},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2402.16823},
}

@misc{chen2023fireacttowardlanguageagent,
      title={FireAct: Toward Language Agent Fine-tuning},
      author={Baian Chen and Chang Shu and Ehsan Shareghi and Nigel Collier and Karthik Narasimhan and Shunyu Yao},
      year={2023},
      eprint={2310.05915},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2310.05915},
}

@misc{feng2025retoolreinforcementlearningstrategic,
      title={ReTool: Reinforcement Learning for Strategic Tool Use in LLMs},
      author={Jiazhan Feng and Shijue Huang and Xingwei Qu and Ge Zhang and Yujia Qin and Baoquan Zhong and Chengquan Jiang and Jinxin Chi and Wanjun Zhong},
      year={2025},
      eprint={2504.11536},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.11536},
}

@misc{jin2025searchr1trainingllmsreason,
      title={Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning},
      author={Bowen Jin and Hansi Zeng and Zhenrui Yue and Jinsung Yoon and Sercan Arik and Dong Wang and Hamed Zamani and Jiawei Han},
      year={2025},
      eprint={2503.09516},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.09516},
}

@article{dong2023raftrewardrankedfinetuning,
      title={RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment},
      author={Hanze Dong and Wei Xiong and Deepanshu Goyal and Yihan Zhang and Winnie Chow and Rui Pan and Shizhe Diao and Jipeng Zhang and Kashun Shum and Tong Zhang},
      journal={Transactions on Machine Learning Research},
      year={2023},
      eprint={2304.06767},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2304.06767},
}

@misc{gulcehre2023reinforcedselftrainingrestlanguage,
      title={Reinforced Self-Training (ReST) for Language Modeling},
      author={Caglar Gulcehre and Tom Le Paine and Srivatsan Srinivasan and Ksenia Konyushkova and Lotte Weerts and Abhishek Sharma and Aditya Siddhant and Alex Ahern and Miaosen Wang and Chenjie Gu and Wolfgang Macherey and Arnaud Doucet and Orhan Firat and Nando de Freitas},
      year={2023},
      eprint={2308.08998},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2308.08998},
}

@misc{zelikman2022starbootstrappingreasoningreasoning,
      title={STaR: Bootstrapping Reasoning With Reasoning},
      author={Eric Zelikman and Yuhuai Wu and Jesse Mu and Noah D. Goodman},
      year={2022},
      eprint={2203.14465},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2203.14465},
}

@inproceedings{bo2026promptparametercooptimizationlarge,
      title={Prompt and Parameter Co-Optimization for Large Language Models},
      author={Xiaohe Bo and Rui Li and Zexu Sun and Quanyu Dai and Zeyu Zhang and Zihang Tian and Xu Chen and Zhenhua Dong},
      booktitle={International Conference on Learning Representations (ICLR)},
      year={2026},
      eprint={2509.24245},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2509.24245},
}

@misc{lu2026p2ojointpolicyprompt,
      title={P$^2$O: Joint Policy and Prompt Optimization},
      author={Xinyu Lu and Kaiqi Zhang and Jinglin Yang and Boxi Cao and Yaojie Lu and Hongyu Lin and Min He and Xianpei Han and Le Sun},
      year={2026},
      eprint={2603.21877},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.21877},
}

@misc{zhang2026evolutionarysystempromptlearning,
      title={Evolutionary System Prompt Learning for Reinforcement Learning in LLMs},
      author={Lunjun Zhang and Ryan Chen and Bradly C. Stadie},
      year={2026},
      eprint={2602.14697},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.14697},
}

@inproceedings{ziems2026composingpolicygradientsprompt,
      title={Composing Policy Gradients and Prompt Optimization for Language Model Programs},
      author={Noah Ziems and Dilara Soylu and Lakshya A Agrawal and Isaac Miller and Liheng Lai and Chen Qian and Kaiqiang Song and Meng Jiang and Dan Klein and Matei Zaharia and Karel D'Oosterlinck and Christopher Potts and Omar Khattab},
      booktitle={ACM Conference on AI and Agentic Systems (CAIS)},
      year={2026},
      eprint={2508.04660},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2508.04660},
}
```

## Part 1(b) — Tressoir

- **Real arXiv ID: NOT FOUND, and very likely none exists.** An arXiv API search for `all:Tressoir` returns 0 results. An author search for Ngom AND Kraska returns 4 papers (JetStream ×2, Parachute, BRAD), none of them Tressoir. The paper seems to be hosted only on alphaXiv: "2605.tressoir" is an alphaXiv-only ID, and alphaxiv.org/abs/tressoir also works. alphaXiv shows it as submitted 19 May 2026, with the 7 authors as given, all MIT.
- **Venue (confirmed on the venue page):** ACM CAIS 2026, Paper Session 1: Agent Design (caisconf.org/program/2026/papers/tressoir-unifying-online-offline-and-hil-design-and-evolution-of-multi-agent-sys/). That page links an ACM DL DOI, 10.1145/3786335.3813143. dl.acm.org returned HTTP 403, so the DOI is taken from the CAIS page only. Check that it resolves before you submit.
- **Its citations on automated agent/harness design: NOT FOUND.** The alphaXiv abstract, PDF viewer and overview pages did not expose the reference list. The PDF loads in JavaScript, and the only "similar papers" shown were alphaXiv recommendations, not citations. The ACM page was blocked. If you have the Tressoir source or PDF locally, its bib can be read directly. Nothing in /workspace/paper/references.bib covers it.

```bibtex
@inproceedings{ngom2026tressoirunifyingonlineoffline,
      title={Tressoir: Unifying Online, Offline, and HIL Design and Evolution of Multi-Agent Systems via Interpretable Blueprints},
      author={Amadou Latyr Ngom and Ziniu Wu and Jason Mohoney and James Moore and Alex Zhang and Samuel Madden and Tim Kraska},
      booktitle={ACM Conference on AI and Agentic Systems (CAIS)},
      year={2026},
      doi={10.1145/3786335.3813143},
      url={https://www.alphaxiv.org/abs/tressoir},
}
```

---

## Part 2 — Candidate verification

### Co-design / harness design

1. **ADAS (2408.08435)**: VERIFIED. Hu, Lu, Clune; first submitted 15 Aug 2024. (Venue not confirmed here, so kept as @misc.)
2. **AFlow (2410.10762)**: VERIFIED. 14 authors.
3. **Darwin Gödel Machine (2505.22954)**: VERIFIED. arXiv spells it "Godel" with no umlaut. Authors: Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, Jeff Clune.
4. **Meta-Harness**: found at 2603.28052 and VERIFIED. Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, Chelsea Finn; 30 Mar 2026, cs.AI.
5. **DSPy (2310.03714)**: VERIFIED. Caution: WHALE's printed author list for DSPy (which ends "...Josifoski, Guestrin, Zaharia, Potts") does not match arXiv. The arXiv list (13 authors, below) is the authoritative one.
6. **Soylu et al. (2407.10930)**: VERIFIED. The arXiv comment says "EMNLP 2024", so the entry is @inproceedings.
7. **GEPA (2507.19457)**: VERIFIED. 17 authors. Comment: "Accepted to ICLR 2026 (Oral)".
8. **SWE-Gym (2412.21139)**: VERIFIED. Comment: "Accepted at ICML 2025". The exact title is "Training Software Engineering Agents and Verifiers with SWE-Gym".
9. **SWE-smith (2504.21798)**: VERIFIED. 10 authors.
10. Extra (the harness that SWE-smith and SWE-Gym agents run in, and that WHALE cites): **SWE-agent (2405.15793)**: VERIFIED.

```bibtex
@misc{hu2024automateddesignagenticsystems,
      title={Automated Design of Agentic Systems},
      author={Shengran Hu and Cong Lu and Jeff Clune},
      year={2024},
      eprint={2408.08435},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2408.08435},
}

@misc{zhang2024aflowautomatingagenticworkflow,
      title={AFlow: Automating Agentic Workflow Generation},
      author={Jiayi Zhang and Jinyu Xiang and Zhaoyang Yu and Fengwei Teng and Xionghui Chen and Jiaqi Chen and Mingchen Zhuge and Xin Cheng and Sirui Hong and Jinlin Wang and Bingnan Zheng and Bang Liu and Yuyu Luo and Chenglin Wu},
      year={2024},
      eprint={2410.10762},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.10762},
}

@misc{zhang2025darwingodelmachineopenended,
      title={Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents},
      author={Jenny Zhang and Shengran Hu and Cong Lu and Robert Lange and Jeff Clune},
      year={2025},
      eprint={2505.22954},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.22954},
}

@misc{lee2026metaharnessendtoendoptimizationmodel,
      title={Meta-Harness: End-to-End Optimization of Model Harnesses},
      author={Yoonho Lee and Roshen Nair and Qizheng Zhang and Kangwook Lee and Omar Khattab and Chelsea Finn},
      year={2026},
      eprint={2603.28052},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.28052},
}

@misc{khattab2023dspycompilingdeclarativelanguage,
      title={DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines},
      author={Omar Khattab and Arnav Singhvi and Paridhi Maheshwari and Zhiyuan Zhang and Keshav Santhanam and Sri Vardhamanan and Saiful Haq and Ashutosh Sharma and Thomas T. Joshi and Hanna Moazam and Heather Miller and Matei Zaharia and Christopher Potts},
      year={2023},
      eprint={2310.03714},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2310.03714},
}

@inproceedings{soylu2024finetuningpromptoptimizationtwo,
      title={Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together},
      author={Dilara Soylu and Christopher Potts and Omar Khattab},
      booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
      year={2024},
      eprint={2407.10930},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2407.10930},
}

@inproceedings{agrawal2026gepareflectivepromptevolution,
      title={GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning},
      author={Lakshya A Agrawal and Shangyin Tan and Dilara Soylu and Noah Ziems and Rishi Khare and Krista Opsahl-Ong and Arnav Singhvi and Herumb Shandilya and Michael J Ryan and Meng Jiang and Christopher Potts and Koushik Sen and Alexandros G. Dimakis and Ion Stoica and Dan Klein and Matei Zaharia and Omar Khattab},
      booktitle={International Conference on Learning Representations (ICLR)},
      year={2026},
      eprint={2507.19457},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2507.19457},
}

@inproceedings{pan2025trainingsoftwareengineeringagents,
      title={Training Software Engineering Agents and Verifiers with SWE-Gym},
      author={Jiayi Pan and Xingyao Wang and Graham Neubig and Navdeep Jaitly and Heng Ji and Alane Suhr and Yizhe Zhang},
      booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
      year={2025},
      eprint={2412.21139},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2412.21139},
}

@misc{yang2025swesmithscalingdatasoftware,
      title={SWE-smith: Scaling Data for Software Engineering Agents},
      author={John Yang and Kilian Lieret and Carlos E. Jimenez and Alexander Wettig and Kabir Khandpur and Yanzhe Zhang and Binyuan Hui and Ofir Press and Ludwig Schmidt and Diyi Yang},
      year={2025},
      eprint={2504.21798},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2504.21798},
}

@misc{yang2024sweagentagentcomputerinterfacesenable,
      title={SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering},
      author={John Yang and Carlos E. Jimenez and Alexander Wettig and Kilian Lieret and Shunyu Yao and Karthik Narasimhan and Ofir Press},
      year={2024},
      eprint={2405.15793},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2405.15793},
}
```

### Multimodal

1. **BLIP-2 (2301.12597)**: VERIFIED. Li, Li, Savarese, Hoi.
2. **LLaVA (2304.08485)**: VERIFIED. Comment: "NeurIPS 2023 Oral".
3. **ViT (2010.11929)**: VERIFIED. 12 authors.
4. **Whisper (2212.04356)**: VERIFIED. The title is "Robust Speech Recognition via Large-Scale Weak Supervision".
5. **Qwen2-Audio (2407.10759)**: VERIFIED. 12 authors; primary class eess.AS.
6. **Qwen2.5-VL (2502.13923)**: VERIFIED. 27 authors per the abs page.
7. **Qwen2.5-Omni (2503.20215)**: VERIFIED. 14 authors.
8. **Qwen3-VL Technical Report**: it exists at **2511.21631**. First submitted 26 Nov 2025, cs.CV, first author Shuai Bai. It has a very long author list (the API reported 72; one secondary source says 64), so the entry uses "and others".
9. **Qwen3.5**: no Qwen3.5 (text/VL) technical report exists on arXiv. The official model cards (huggingface.co/Qwen/Qwen3.5-2B and huggingface.co/Qwen/Qwen3.5-9B, both confirmed) give the citation below. It is the blog "Qwen3.5: Towards Native Multimodal Agents" (Qwen Team, Feb 2026, https://qwen.ai/blog?id=qwen3.5), which is also what WHALE cites. A separate *Qwen3.5-Omni Technical Report* exists (2604.15804, "Qwen Team", 17 Apr 2026). Cite it only if you use the Omni models.

```bibtex
@misc{li2023blip2bootstrappinglanguageimagepretraining,
      title={BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models},
      author={Junnan Li and Dongxu Li and Silvio Savarese and Steven Hoi},
      year={2023},
      eprint={2301.12597},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2301.12597},
}

@inproceedings{liu2023visualinstructiontuning,
      title={Visual Instruction Tuning},
      author={Haotian Liu and Chunyuan Li and Qingyang Wu and Yong Jae Lee},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2023},
      eprint={2304.08485},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2304.08485},
}

@misc{dosovitskiy2020imageworth16x16words,
      title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
      author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
      year={2020},
      eprint={2010.11929},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2010.11929},
}

@misc{radford2022robustspeechrecognitionlargescale,
      title={Robust Speech Recognition via Large-Scale Weak Supervision},
      author={Alec Radford and Jong Wook Kim and Tao Xu and Greg Brockman and Christine McLeavey and Ilya Sutskever},
      year={2022},
      eprint={2212.04356},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2212.04356},
}

@misc{chu2024qwen2audiotechnicalreport,
      title={Qwen2-Audio Technical Report},
      author={Yunfei Chu and Jin Xu and Qian Yang and Haojie Wei and Xipin Wei and Zhifang Guo and Yichong Leng and Yuanjun Lv and Jinzheng He and Junyang Lin and Chang Zhou and Jingren Zhou},
      year={2024},
      eprint={2407.10759},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2407.10759},
}

@misc{bai2025qwen25vltechnicalreport,
      title={Qwen2.5-VL Technical Report},
      author={Shuai Bai and Keqin Chen and Xuejing Liu and Jialin Wang and Wenbin Ge and Sibo Song and Kai Dang and Peng Wang and Shijie Wang and Jun Tang and Humen Zhong and Yuanzhi Zhu and Mingkun Yang and Zhaohai Li and Jianqiang Wan and Pengfei Wang and Wei Ding and Zheren Fu and Yiheng Xu and Jiabo Ye and Xi Zhang and Tianbao Xie and Zesen Cheng and Hang Zhang and Zhibo Yang and Haiyang Xu and Junyang Lin},
      year={2025},
      eprint={2502.13923},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.13923},
}

@misc{xu2025qwen25omnitechnicalreport,
      title={Qwen2.5-Omni Technical Report},
      author={Jin Xu and Zhifang Guo and Jinzheng He and Hangrui Hu and Ting He and Shuai Bai and Keqin Chen and Jialin Wang and Yang Fan and Kai Dang and Bin Zhang and Xiong Wang and Yunfei Chu and Junyang Lin},
      year={2025},
      eprint={2503.20215},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.20215},
}

@misc{bai2025qwen3vltechnicalreport,
      title={Qwen3-VL Technical Report},
      author={Shuai Bai and Yuxuan Cai and Ruizhe Chen and Keqin Chen and Xionghui Chen and others},
      year={2025},
      eprint={2511.21631},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.21631},
}

@misc{qwen35blog,
      title={{Qwen3.5}: Towards Native Multimodal Agents},
      author={{Qwen Team}},
      month={February},
      year={2026},
      url={https://qwen.ai/blog?id=qwen3.5},
}

@misc{qwen35omnitechnicalreport,
      title={Qwen3.5-Omni Technical Report},
      author={{Qwen Team}},
      year={2026},
      eprint={2604.15804},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.15804},
}
```

### Latent context / compression

All 12 IDs you gave are correct. Notes:
- AutoCompressors: the arXiv title is "Adapting Language Models to Compress Contexts" (EMNLP 2023).
- CEPE: the arXiv title is "Long-Context Language Modeling with Parallel Context Encoding" (ACL 2024).
- Activation Beacon: the current arXiv title is "Long Context Compression with Activation Beacon". An older version had a different title, so use this one.
- KVzip: primary class on arXiv is cs.DB. Comment: "NeurIPS 2025 Oral".
- 500xCompressor and Lexico: no venue in the arXiv metadata, so kept as @misc.
- Prefix-Tuning: no venue in the arXiv metadata, so kept as @misc. I believe it appeared at ACL 2021 but did not confirm that.

```bibtex
@inproceedings{mu2023learningcompresspromptsgist,
      title={Learning to Compress Prompts with Gist Tokens},
      author={Jesse Mu and Xiang Lisa Li and Noah Goodman},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2023},
      eprint={2304.08467},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2304.08467},
}

@inproceedings{chevalier2023adaptinglanguagemodelscompress,
      title={Adapting Language Models to Compress Contexts},
      author={Alexis Chevalier and Alexander Wettig and Anirudh Ajith and Danqi Chen},
      booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
      year={2023},
      eprint={2305.14788},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2305.14788},
}

@inproceedings{ge2024incontextautoencodercontextcompression,
      title={In-context Autoencoder for Context Compression in a Large Language Model},
      author={Tao Ge and Jing Hu and Lei Wang and Xun Wang and Si-Qing Chen and Furu Wei},
      booktitle={International Conference on Learning Representations (ICLR)},
      year={2024},
      eprint={2307.06945},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2307.06945},
}

@inproceedings{cheng2024xragextremecontextcompression,
      title={xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token},
      author={Xin Cheng and Xun Wang and Xingxing Zhang and Tao Ge and Si-Qing Chen and Furu Wei and Huishuai Zhang and Dongyan Zhao},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2024},
      eprint={2405.13792},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2405.13792},
}

@misc{li2024500xcompressorgeneralizedpromptcompression,
      title={500xCompressor: Generalized Prompt Compression for Large Language Models},
      author={Zongqian Li and Yixuan Su and Nigel Collier},
      year={2024},
      eprint={2408.03094},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.03094},
}

@inproceedings{bulatov2022recurrentmemorytransformer,
      title={Recurrent Memory Transformer},
      author={Aydar Bulatov and Yuri Kuratov and Mikhail S. Burtsev},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2022},
      eprint={2207.06881},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2207.06881},
}

@misc{li2021prefixtuningoptimizingcontinuousprompts,
      title={Prefix-Tuning: Optimizing Continuous Prompts for Generation},
      author={Xiang Lisa Li and Percy Liang},
      year={2021},
      eprint={2101.00190},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2101.00190},
}

@inproceedings{yen2024longcontextlanguagemodelingparallel,
      title={Long-Context Language Modeling with Parallel Context Encoding},
      author={Howard Yen and Tianyu Gao and Danqi Chen},
      booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)},
      year={2024},
      eprint={2402.16617},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2402.16617},
}

@inproceedings{tan2024lloco,
      title={LLoCO: Learning Long Contexts Offline},
      author={Sijun Tan and Xiuyu Li and Shishir Patil and Ziyang Wu and Tianjun Zhang and Kurt Keutzer and Joseph E. Gonzalez and Raluca Ada Popa},
      booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
      year={2024},
      eprint={2404.07979},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2404.07979},
}

@misc{zhang2024longcontextcompressionactivation,
      title={Long Context Compression with Activation Beacon},
      author={Peitian Zhang and Zheng Liu and Shitao Xiao and Ninglu Shao and Qiwei Ye and Zhicheng Dou},
      year={2024},
      eprint={2401.03462},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2401.03462},
}

@inproceedings{kim2025kvzipqueryagnostickvcache,
      title={KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction},
      author={Jang-Hyun Kim and Jinuk Kim and Sangwoo Kwon and Jae W. Lee and Sangdoo Yun and Hyun Oh Song},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2025},
      eprint={2505.23416},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2505.23416},
}

@misc{kim2024lexicoextremekvcache,
      title={Lexico: Extreme KV Cache Compression via Sparse Coding over Universal Dictionaries},
      author={Junhyuck Kim and Jongho Park and Jaewoong Cho and Dimitris Papailiopoulos},
      year={2024},
      eprint={2412.08890},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.08890},
}
```

### Latent agent communication

1. **"Let Models Speak Ciphers" (2310.06272)**: VERIFIED. Full title: "Let Models Speak Ciphers: Multiagent Debate through Embeddings". Comment: "Accepted to ICLR 2024".
2. **Cache-to-Cache / C2C**: found at 2510.03215 and VERIFIED. Fu, Min, Zhang, Yan, Dai, Ouyang, Wang. ICLR 2026, per the arXiv comment and the ICLR proceedings page.
3. **KVComm**: **two different 2025 papers share this name.** Pick the one you mean.
   - 2510.03346, "KVComm: Enabling Efficient LLM Communication through Selective KV Sharing" (Shi, Chiesa, Maguire, Kostic; ICLR 2026). This is sender-to-receiver KV sharing, the closer match to the Activation Context idea.
   - 2510.12872, "KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems" (Ye et al.; NeurIPS 2025). This one reuses KV caches during prefill in multi-agent systems.
4. **LatentMAS**: found at 2511.20639 and VERIFIED. The arXiv title is "Latent Collaboration in Multi-Agent Systems" (Zou et al., 13 authors). Comment: "ICML2026 Spotlight".
5. Additional work found (all VERIFIED):
   - Interlat, 2511.09149, "Enabling Agents to Communicate Entirely in Latent Space" (ACL 2026)
   - ThoughtComm, 2510.20733, "Thought Communication in Multiagent Collaboration" (NeurIPS 2025 Spotlight)
   - StateBridge, 2608.13317 (COLM 2026)

```bibtex
@inproceedings{pham2024letmodelsspeakciphers,
      title={Let Models Speak Ciphers: Multiagent Debate through Embeddings},
      author={Chau Pham and Boyi Liu and Yingxiang Yang and Zhengyu Chen and Tianyi Liu and Jianbo Yuan and Bryan A. Plummer and Zhaoran Wang and Hongxia Yang},
      booktitle={International Conference on Learning Representations (ICLR)},
      year={2024},
      eprint={2310.06272},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2310.06272},
}

@inproceedings{fu2026cachetocachedirectsemanticcommunication,
      title={Cache-to-Cache: Direct Semantic Communication Between Large Language Models},
      author={Tianyu Fu and Zihan Min and Hanling Zhang and Jichao Yan and Guohao Dai and Wanli Ouyang and Yu Wang},
      booktitle={International Conference on Learning Representations (ICLR)},
      year={2026},
      eprint={2510.03215},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2510.03215},
}

@inproceedings{shi2026kvcommenablingefficientllm,
      title={KVComm: Enabling Efficient LLM Communication through Selective KV Sharing},
      author={Xiangyu Shi and Marco Chiesa and Gerald Q. Maguire and Dejan Kostic},
      booktitle={International Conference on Learning Representations (ICLR)},
      year={2026},
      eprint={2510.03346},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2510.03346},
}

@inproceedings{ye2025kvcommonlinecrosscontextkvcache,
      title={KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems},
      author={Hancheng Ye and Zhengqi Gao and Mingyuan Ma and Qinsi Wang and Yuzhe Fu and Ming-Yu Chung and Yueqian Lin and Zhijian Liu and Jianyi Zhang and Danyang Zhuo and Yiran Chen},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2025},
      eprint={2510.12872},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2510.12872},
}

@inproceedings{zou2026latentcollaborationmultiagentsystems,
      title={Latent Collaboration in Multi-Agent Systems},
      author={Jiaru Zou and Ruizhong Qiu and Gaotang Li and Xiyuan Yang and Katherine Tieu and Pan Lu and Ke Shen and Hanghang Tong and Yejin Choi and Jingrui He and James Zou and Mengdi Wang and Ling Yang},
      booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
      year={2026},
      eprint={2511.20639},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2511.20639},
}

@inproceedings{du2026enablingagentscommunicateentirely,
      title={Enabling Agents to Communicate Entirely in Latent Space},
      author={Zhuoyun Du and Runze Wang and Huiyu Bai and Zouying Cao and Xiaoyong Zhu and Yu Cheng and Bo Zheng and Wei Chen and Haochao Ying},
      booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
      year={2026},
      eprint={2511.09149},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2511.09149},
}

@inproceedings{zheng2025thoughtcommunicationmultiagentcollaboration,
      title={Thought Communication in Multiagent Collaboration},
      author={Yujia Zheng and Zhuokai Zhao and Zijian Li and Yaqi Xie and Mingze Gao and Lizhu Zhang and Kun Zhang},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2025},
      eprint={2510.20733},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2510.20733},
}

@inproceedings{peng2026statebridgetrainingfreehiddenstate,
      title={StateBridge: Training-free Hidden-state Alignment for Latent Communication in LLM Multi-Agent Systems},
      author={Yanwen Peng and Delvin Ce Zhang and Xi Wang and Nikolaos Aletras},
      booktitle={Conference on Language Modeling (COLM)},
      year={2026},
      eprint={2608.13317},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2608.13317},
}
```

The ACL 2026 edition number ("64th") is my inference, not something I checked. If unsure, use `booktitle={Annual Meeting of the Association for Computational Linguistics (ACL)}`.

### Context management in agents

1. **ACON (2510.00615)**: VERIFIED. "ACON: Optimizing Context Compression for Long-horizon LLM Agents", Kang et al. (Microsoft). Comment: "ICML 2026".
2. **Context-Folding (2510.11967)**: VERIFIED. "Scaling Long-Horizon LLM Agent via Context-Folding", Sun et al. A search result shows a PMLR v306 (ICML 2026) page under a slightly different title, "Scaling Long-Horizon Agent via Context Folding", but the arXiv comment gives no venue. Kept as @misc with the arXiv title.
3. **AgentFold (2510.24699)**: VERIFIED. "AgentFold: Long-Horizon Web Agents with Proactive Context Management", Ye et al. (15 authors). An ICLR 2026 listing appears in search results under the title "...Proactive Context Folding", which is not on arXiv. Kept as @misc.
4. **MEM1 (2506.15841)**: VERIFIED. This one keeps a compact internal state through RL.

```bibtex
@inproceedings{kang2026aconoptimizingcontextcompression,
      title={ACON: Optimizing Context Compression for Long-horizon LLM Agents},
      author={Minki Kang and Wei-Ning Chen and Dongge Han and Huseyin A. Inan and Lukas Wutschitz and Yanzhi Chen and Robert Sim and Saravan Rajmohan},
      booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
      year={2026},
      eprint={2510.00615},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2510.00615},
}

@misc{sun2025scalinglonghorizonllmagent,
      title={Scaling Long-Horizon LLM Agent via Context-Folding},
      author={Weiwei Sun and Miao Lu and Zhan Ling and Kang Liu and Xuesong Yao and Yiming Yang and Jiecao Chen},
      year={2025},
      eprint={2510.11967},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.11967},
}

@misc{ye2025agentfoldlonghorizonwebagents,
      title={AgentFold: Long-Horizon Web Agents with Proactive Context Management},
      author={Rui Ye and Zhongwang Zhang and Kuan Li and Huifeng Yin and Zhengwei Tao and Yida Zhao and Liangcai Su and Liwen Zhang and Zile Qiao and Xinyu Wang and Pengjun Xie and Fei Huang and Siheng Chen and Jingren Zhou and Yong Jiang},
      year={2025},
      eprint={2510.24699},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.24699},
}

@misc{zhou2025mem1learningsynergizememory,
      title={MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents},
      author={Zijian Zhou and Ao Qu and Zhaoxuan Wu and Sunghwan Kim and Alok Prakash and Daniela Rus and Jinhua Zhao and Bryan Kian Hsiang Low and Paul Pu Liang},
      year={2025},
      eprint={2506.15841},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.15841},
}
```

---

## Summary of corrections and caveats

- None of the arXiv IDs you supplied was wrong. All 27 resolve to the intended paper.
- Names that differ on arXiv: AutoCompressors = "Adapting Language Models to Compress Contexts"; CEPE = "Long-Context Language Modeling with Parallel Context Encoding"; LatentMAS = "Latent Collaboration in Multi-Agent Systems"; DGM uses "Godel" with no umlaut; Activation Beacon's current title is "Long Context Compression with Activation Beacon".
- "KVComm" is ambiguous: 2510.03346 (Shi et al., ICLR 2026) and 2510.12872 (Ye et al., NeurIPS 2025) are different papers.
- No Qwen3.5 technical report exists. Cite the official Qwen3.5 blog/model-card entry, which is what the HF cards for Qwen3.5-2B and -9B recommend.
- Tressoir: no arXiv ID exists; the paper is alphaXiv-only plus ACM CAIS 2026. The DOI comes from the CAIS program page because the ACM DL was blocked. Its reference list could not be read.
- WHALE's own bib has two mistakes that should not be copied: a duplicated ADAS entry, and a DSPy author list that does not match arXiv.
- Venues not confirmed, so these are left as @misc: ADAS, AFlow, DGM, DSPy, ViT, Whisper, BLIP-2, Prefix-Tuning, Activation Beacon, 500xCompressor, Lexico, AgentFold, Context-Folding. Upgrade any of them only after checking the proceedings.
