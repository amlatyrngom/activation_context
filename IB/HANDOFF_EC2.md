# Handoff: VS Code agent container → Claude Code GUI on the EC2 host (2026-11-09)

Give this file to the new session first, then `IB/TRESSOIR.md`, `IB/STATE.md` and `IB/CANON/ROOT_CANON.md`, as
`CLAUDE.md` says. `IB/` is the only directory shared between the old container and the host, so everything below
is either already in `IB/` or listed as something to carry over.

## 1. What did NOT live on the host (container-only)

The container's `/workspace` was a container-local working copy. Only `/workspace/IB` was bind-mounted from the
host, and `/source` was a read-only view of the host repo. Outside `IB/`, nothing written in the container reaches
the host.

| Item | Where it was in the container | Status / what to do |
|---|---|---|
| Main code repo | `/workspace` on branch `paper-mvp-20261013`, HEAD `258fce6`, **pushed** to `origin` (github.com/amlatyrngom/activation_context) | To work from that commit: `git fetch origin && git checkout paper-mvp-20261013`. The host repo (`/source` in the container) is at a different commit (`af52a4c` "Apply current activation-context source snapshot") with 54 local changes, so compare before choosing a base. |
| **Uncommitted code changes** (9 files under `activation/`: key/value transfer reader path, trainer, tests; about 650 lines) | `/workspace/activation/...` | Saved as `IB/ARTIFACTS/HANDOFF_EC2/container_uncommitted_vs_258fce6.patch` (stat in the `.stat.txt` next to it). Apply on top of `258fce6`: `git apply IB/ARTIFACTS/HANDOFF_EC2/container_uncommitted_vs_258fce6.patch`. These 9 files differ from the host's `/source` copies. |
| Paper repo | `/workspace/paper` = github.com/amlatyrngom/activation_context_paper (git-synced with Overleaf, branch `main`) | Nothing lost: it matches Overleaf `16a0260` exactly (the submitted version). Clone fresh on the host. The container copy only adds merge commits with identical content. |
| LaTeX toolchain | `~/.TinyTeX` (pdflatex + bibtex via latexmk) | `IB/ARTIFACTS/PAPER_WRITING/build_pdf.sh` builds `main.pdf` and `appendix.pdf` into `IB/ARTIFACTS/PAPER_WRITING/build/`. The host needs TeX (TinyTeX or TeX Live). The script hard-codes `/workspace/paper`; repoint it. |
| Credentials and tool state | `/workspace/.env`, `~/.aws/credentials`, `~/.ssh/{config,id_ed25519_orcd,id_ed25519_orcd.pub,known_hosts}`, `~/.sky/` (SkyPilot client state; no clusters tracked), `~/.gitconfig`, `~/.claude/settings.json` | **Copied** to `IB/ARTIFACTS/HANDOFF_EC2/carry/` (git-ignored, owner-only permissions). Apply with section 1b, then delete the folder. Not copied: `~/.claude.json` and `~/.codex/` (login state; the GUI logs in on its own). The SSH wrappers `IB/TMP/AGENT_AC_PRETRAINING/ac_recon/{csail,orcd}_ssh.sh` need the SSH config. |
| Claude memory | none existed (`~/.claude/projects/-workspace/memory` was never created) | Nothing to carry over. Durable rules live in `IB/CANON/ROOT_CANON.md`, which is shared. |
| Chat transcripts | `~/.claude/projects/-workspace/*.jsonl` (8 files, 954 MB) | The paper session's transcript is copied as `carry/claude/paper_session_6f1c1373.jsonl.gz` (130 MB; search with `zcat … \| grep`). Its durable content is also in `IB/STATE.md` and sections 2–4 below. |

## 1b. Applying `IB/ARTIFACTS/HANDOFF_EC2/carry/` on the host

Run from the host repo root. `cp -n` never overwrites an existing file; anything that already exists on the host
is a merge decision for you.

```bash
C=IB/ARTIFACTS/HANDOFF_EC2/carry
# SSH (CSAIL / ORCD). The config uses ~/.ssh/... paths, so it works under any home directory.
mkdir -p ~/.ssh/cm && chmod 700 ~/.ssh ~/.ssh/cm
cp -n $C/ssh/id_ed25519_orcd $C/ssh/id_ed25519_orcd.pub ~/.ssh/ && chmod 600 ~/.ssh/id_ed25519_orcd
[ -f ~/.ssh/config ] && echo "merge $C/ssh/config into ~/.ssh/config by hand" || cp $C/ssh/config ~/.ssh/config
cat $C/ssh/known_hosts >> ~/.ssh/known_hosts
# AWS + SkyPilot
mkdir -p ~/.aws && cp -n $C/aws/credentials ~/.aws/credentials && chmod 600 ~/.aws/credentials
mkdir -p ~/.sky && cp -rn $C/sky/. ~/.sky/
# Git identity and Claude Code preferences: compare, then merge what you want
diff $C/gitconfig ~/.gitconfig; diff $C/claude/settings.json ~/.claude/settings.json
# .env: the host repo's .env has the same keys plus HF_TOKEN; values may differ. Compare before replacing anything.
# Run this yourself, not through an agent, because it prints token values.
diff <(sort $C/repo/workspace.env) <(sort .env)
# Code changes made only in the container (see the table above)
git fetch origin && git checkout paper-mvp-20261013 && git apply IB/ARTIFACTS/HANDOFF_EC2/container_uncommitted_vs_258fce6.patch
# When done:
rm -rf $C
```

On the host, CSAIL and ORCD still need their interactive auth (Duo) the first time; the user runs that themselves.
| Container-specific commands | git push needed `REMOTE_CONTAINERS_IPC=$(ls -t /tmp/vscode-remote-containers-ipc-*.sock \| head -1)` (VS Code credential relay); `uv` needed `UV_PYTHON_INSTALL_DIR=$HOME/.local/share/uv/python` | Not needed on the host. Use the host's normal git and `gh` credentials. |

## 2. Project status

- **Paper submitted:** "Activation Context: Harness/Model Co-Design at the Architectural Level", MLSys 2027
  research track, on Oct 30, 2026 (deadline 20:00 UTC). Final = Overleaf / paper repo `16a0260`. The body is 10
  pages; references and the mandatory "AI Usage Disclosure" (after the references, user-written) follow. The appendix
  (3 pages, `appendix.tex`) was uploaded separately. The review is double-blind; Tressoir is self-cited in the third
  person.
- **MLSys 2027 dates:** reviews Jan 18, 2027; author responses due Jan 21; notifications Feb 28. The CFP
  is at mlsys.org/Conferences/2027/CallForResearchPapers. It lists citations not "correctly represented",
  fabricated references and undisclosed AI use as grounds for desk rejection. MLSys gives one AI-generated review per
  paper to reviewers.
- **Verified on Oct 30:** all 61 cited references (title, full ordered author list, venue) against
  arXiv/Semantic Scholar/Crossref, and every citing sentence against the cited abstract. Every number in the
  abstract, intro, Section 3 and captions was checked against the figure data.
- **Paper artifacts in `IB/ARTIFACTS/PAPER_WRITING/`:**
  - figure scripts: `figs_compact_1028/`, `fig3_v3_final/`, `sec45_v4/`;
  - `appendix_sources/`: 318 files plus a README mapping each appendix table row to the run file it came from;
  - `diff_html/make_diff_v2.py`: readable HTML diff of the paper against any revision. The user likes reviewing
    changes this way.

## 3. Open items for the rebuttal and the camera-ready / arXiv version

The user's stated priorities after submission are:
1. A **text + reader-LoRA baseline outside the Section 3.5 ablations**, i.e. for the main workloads.
2. Make the training/hyperparameter process **less arbitrary**. Rerun experiments whose hyperparameters were
   missing or undocumented.

Known weaknesses a reviewer or the AI review may raise:
- **Fig 5 (compaction) caption:** it defines difficulty by "the full context reference", but difficulty is actually
  the original run's tokens after the cut (appendix and `IB/ARTIFACTS/PAPER_COMPACTION/RESULTS_SO_FAR.md`). The
  user chose to leave it. A reviewer could read it as bucketing by a plotted arm's own outcome.
- **Compaction timing (7.42 s vs 1.34 s, 5.5×):** measured on the earlier frozen-100 set (27B-written prefixes)
  with arm A's encoder, which has the same architecture as the final one. The paper doesn't say so.
- **Fig 5 "No Context" arm:** it is not introduced in the Section 3.3 text.
- **Retrieval decoding:** text/select arms use presence penalty 1.5 (vLLM); the AC arm (HF generate) has none.
  This is stated in appendix Table 1 only. Controlling it is a possible rebuttal experiment.
- **Section 3.5:** the AC generation cap is 128 new tokens vs 2,048 for text. AC replies average under 10 tokens,
  and at most 3/200 replies of any final model hit the cap (`IB/ARTIFACTS/PAPER_WRITING/sec45_v4/data/*/eval/final.summary.json`).
- **Corpus Input / Input + Residuals variants:** how they were initialized isn't documented in the appendix, and
  `STATE.md` (Oct 19) is ambiguous: encoder-output initialization (R15b/D15b) vs a "closed-form start".
- **An unreported Section 3.5 run, `S_kv_v4fr`** (All-Layer KVs, frozen reader, no reconstruction), reaches
  71.5%, above every reported number (one seed, 200 questions). Worth a rerun.
- **Small multi-agent gains:** the 85.7 vs 82.4–83.5 gain rests on 74 problems.
- **Corpus result:** one sampled pass (81.5 vs 79.5 = 8 of 400 questions).

## 4. How the user likes to work (learned in the paper sessions)

- **Edits to the user's writing:** keep their text and framing. Propose each edit back with exact before/after
  text and apply only after they OK it. Prefer simplicity over detail. Don't add details that raise questions
  without answering them, and don't frame results as wins ("matches at fewer tokens" unless actually better).
- **Overleaf:** the user edits on Overleaf at the same time as the session. Pull before any paper edit. When they
  are editing, give them the fix to apply themselves rather than pushing, to avoid conflicts.
- **Dictation:** the user dictates. Interpret transcription errors sensibly ("Office 5.5" = Opus 5.5, "MLCC" =
  MLSys, "overlive" = Overleaf, "Irkut" = IRCoT).
- **Reporting:** show what changed in a readable HTML diff when edits are large. Report body length and page fit
  after every paper build. Never state a number you didn't read from a file.
- **Pronouns:** use they/them for the user.
- **Guardrails (the user is relaxing them):** the container guardrails were written for earlier models, and the
  user now considers them over-cautious. The ones about **external shared systems** still matter until the user
  says otherwise:
  - never start Duo or other auth flows on CSAIL/ORCD;
  - ORCD reservation 23493337 is off limits;
  - EC2 instance `i-0918b8f73e9a0fe03` was marked "never touch". Check whether that is the host you're now on;
  - run `squeue` right before any `scancel`, and kill jobs only by PID;
  - never edit a running job's script or recipe;
  - never copy `optimizer.pt`.

  Ask the user which of the remaining rules in `IB/CANON/ROOT_CANON.md` and the end of `IB/STATE.md` still apply.
