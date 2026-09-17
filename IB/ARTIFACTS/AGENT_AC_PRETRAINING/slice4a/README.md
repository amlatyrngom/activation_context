# Slice 4a: applied product handoff

Teacher campaign, agentic programs and activation content everywhere are applied locally. The 49-file `changes.patch` and `source_manifest.json` describe the exact delta against the pre-application workspace, including its prototype edits. The full diff cards are in `../SLICE4A_ROUND.tressoir.md`; `../HANDOFF.tressoir.md` describes behavior and remaining work.

The patch includes the missing `math-verify` dependency in `pyproject.toml` and `uv.lock`. Eight supplementary tests and the agentic-program probe are retained here for private validation, excluded from the product manifest/patch/cards. The key agentic-program test and canonical teacher-generation CLI are product files.

Do not apply the patch again to the updated checkout. For another matching baseline, verify source hashes before `git apply changes.patch`; preserve unrelated edits. Original source backups, hashes, validation logs and the local reconciliation script are under `IB/TMP/AGENT_AC_PRETRAINING/apply_20260914T182304Z/`. The historical staging builder is not a local application command.
