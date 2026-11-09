# Structured-data-into-LLM citations (verified 2026-10-26)

Sources: arXiv API/abs pages for all IDs; venue pages: PMLR v235 (LLaGA), neurips.cc/virtual/2023/poster/71298 (3D-LLM), aclanthology 2023.emnlp-main.966 (MolCA). GraphGPT (SIGIR'2024) and PointLLM (ECCV 2024) venues come from their arXiv comments. Edition numbers and volumes not shown on a checked page were left out.

| Key | Status | Encoder -> LLM? |
|---|---|---|
| perozzi2024letyourgraphdo | VERIFIED (arXiv only, no venue found) | Yes. A GNN graph encoder outputs a fixed number of token embeddings that are prepended to a frozen LLM's prompt embeddings. Only the encoder is trained. |
| chen2024llagalargelanguage | VERIFIED, ICML 2024 | Partly. There is no GNN. Node neighbourhoods are serialised into structure-aware sequences, and a trained projector maps their node embeddings into the LLM token-embedding space. |
| tang2024graphgptgraphinstruction | VERIFIED, SIGIR 2024 | Yes. A graph encoder's node embeddings go through a linear projector to become graph tokens in the LLM input. |
| su2024tablegpt2largemultimodal | VERIFIED (arXiv only) | Yes in the paper. The table encoder's column embeddings pass through a Q-Former-style adapter into the decoder's embedding layer. Caveat: the paper says the encoder is not yet released, only the decoder is. |
| hong20233dllminjecting | VERIFIED, NeurIPS 2023 | Yes. 3D features (taken from rendered multi-view images and kept in the 2D VLM feature space) are fed through the VLM's perceiver/Q-Former into the LLM. |
| xu2024pointllmempoweringlarge | VERIFIED, ECCV 2024 | Yes. Point encoder, then an MLP projector, then point tokens interleaved with text tokens. |
| liu2023molcamoleculargraph | VERIFIED, EMNLP 2023 | Yes. Features from a 2D molecular graph encoder go through a Q-Former cross-modal projector, then are input to the LM (Galactica). |

```bibtex
@misc{perozzi2024letyourgraphdo,
      title={Let Your Graph Do the Talking: Encoding Structured Data for LLMs},
      author={Bryan Perozzi and Bahare Fatemi and Dustin Zelle and Anton Tsitsulin and Mehran Kazemi and Rami Al-Rfou and Jonathan Halcrow},
      year={2024},
      eprint={2402.05862},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.05862},
}

@inproceedings{chen2024llagalargelanguage,
      title={LLaGA: Large Language and Graph Assistant},
      author={Runjin Chen and Tong Zhao and Ajay Kumar Jaiswal and Neil Shah and Zhangyang Wang},
      booktitle={Proceedings of the 41st International Conference on Machine Learning},
      series={Proceedings of Machine Learning Research},
      volume={235},
      pages={7809--7823},
      year={2024},
      eprint={2402.08170},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.08170},
}

@inproceedings{tang2024graphgptgraphinstruction,
      title={GraphGPT: Graph Instruction Tuning for Large Language Models},
      author={Jiabin Tang and Yuhao Yang and Wei Wei and Lei Shi and Lixin Su and Suqi Cheng and Dawei Yin and Chao Huang},
      booktitle={International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
      year={2024},
      eprint={2310.13023},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2310.13023},
}

@misc{su2024tablegpt2largemultimodal,
      title={TableGPT2: A Large Multimodal Model with Tabular Data Integration},
      author={Aofeng Su and Aowen Wang and Chao Ye and Chen Zhou and Ga Zhang and Gang Chen and Guangcheng Zhu and Haobo Wang and Haokai Xu and Hao Chen and Haoze Li and Haoxuan Lan and Jiaming Tian and Jing Yuan and Junbo Zhao and Junlin Zhou and Kaizhe Shou and Liangyu Zha and Lin Long and Liyao Li and Pengzuo Wu and Qi Zhang and Qingyi Huang and Saisai Yang and Tao Zhang and Wentao Ye and Wufang Zhu and Xiaomeng Hu and Xijun Gu and Xinjie Sun and Xiang Li and Yuhang Yang and Zhiqing Xiao},
      year={2024},
      eprint={2411.02059},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.02059},
}

@inproceedings{hong20233dllminjecting,
      title={3D-LLM: Injecting the 3D World into Large Language Models},
      author={Yining Hong and Haoyu Zhen and Peihao Chen and Shuhong Zheng and Yilun Du and Zhenfang Chen and Chuang Gan},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2023},
      eprint={2307.12981},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2307.12981},
}

@inproceedings{xu2024pointllmempoweringlarge,
      title={PointLLM: Empowering Large Language Models to Understand Point Clouds},
      author={Runsen Xu and Xiaolong Wang and Tai Wang and Yilun Chen and Jiangmiao Pang and Dahua Lin},
      booktitle={European Conference on Computer Vision (ECCV)},
      year={2024},
      eprint={2308.16911},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2308.16911},
}

@inproceedings{liu2023molcamoleculargraph,
      title={MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter},
      author={Zhiyuan Liu and Sihang Li and Yanchen Luo and Hao Fei and Yixin Cao and Kenji Kawaguchi and Xiang Wang and Tat-Seng Chua},
      booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
      pages={15623--15638},
      year={2023},
      eprint={2310.12798},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2310.12798},
}
```

Notes:
- The LLaGA author name is "Ajay Kumar Jaiswal" on PMLR and "Ajay Jaiswal" on arXiv. The PMLR form is used.
- 3D-LLM's arXiv comment gives only a project page. NeurIPS 2023 (spotlight) is confirmed by the neurips.cc virtual page.
- For tables, cite TableGPT2 as describing the interface. Its encoder is not public yet.
- Suggested minimal set, one per modality: GraphToken (graphs), TableGPT2 (tables), PointLLM (3D), MolCA (molecules).
