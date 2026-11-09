# source: /workspace/IB/TMP/PAPER_SYSTEMS/README.md
# lines: 1-3 20-22 59-64
# extracted: nl -ba | sed -n (protocol common to both setups: one request at a time, batch 1, bf16, single RTX PRO 6000)
# git commit: n/a: git-ignored IB/TMP file, not in HEAD 258fce6adf6e59ad54594b82863d7cb6169404fa

----- lines 1,3 -----
     1	# Systems bench: single-request latency and memory of Activation Contexts (k = 20 passages)
     2	
     3	Status: RUNNING. The numbers, summary tables and filled-in checklist go at the bottom once the runs finish.
----- lines 20,22 -----
    20	## Exact commands
    21	
    22	The node is `s45-ec2f` (1× RTX PRO 6000 Blackwell Server Edition, 96 GB). The repo is mirrored at `/root/sky_workdir`. The kits and data are rsynced to `/root/systems/{kit,data,pairs}` with:
----- lines 59,64 -----
    59	**Common to both setups:**
    60	- One request at a time (batch 1 at the request level), bf16, a single RTX PRO 6000.
    61	- `torch.cuda.synchronize()` before every stage start and after every stage end; Python garbage collection is paused during a request.
    62	- Warm-up requests are run per arm and discarded; the arm order is shuffled per item with a fixed seed.
    63	- 32 forced new tokens, greedy, EOS ignored.
    64	
