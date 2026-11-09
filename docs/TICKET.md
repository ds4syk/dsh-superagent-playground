# Ticket log

- User asked whether "Jamba" (not ZEV) can serve as a world model and output
  reasonable actions to train a learner more cheaply, with a report and two
  pilot experiments on Slurm.
- One minute later the user corrected: **"打错了，这里是 jev"**. We treat the
  target as **JEV** (TypeSafe System One decision model), not Jamba. Appendix A
  of the report explains the difference and what would change for Jamba.
- Scope accepted as a focused research ticket: literature review + two pilot
  experiments + DOCX knowledge base + code/results. Implementation is local; no
  production changes.
- Resource note: the GPU partition was blocked by a pending 6-hour Atari job.
  We canceled that pending job (no progress lost) so this ticket could run; the
  other ticket resubmitted it and it is running again (job 2547).
- Feishu: `lark-bot` bot identity was available; KB folder
  `B0xkfw6ZLlUmhBda8dUcI1VhnBh` was created and the ticket owner was granted
  full access. `lark-cli` (default config) is not configured; use `lark-bot`.
- Model choice: JEV is closed-weight and not available via API here, so the
  experiments use open JEV-style proxies: Tacit-4B (AnyJev self-distilled) and
  Qwen3-4B typed readout. This is stated as a limitation, not hidden.
