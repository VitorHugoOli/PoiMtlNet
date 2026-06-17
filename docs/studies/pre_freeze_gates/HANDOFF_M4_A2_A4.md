# HANDOFF — A2 + A4 pre-freeze gates · M4 Pro

> Created 2026-06-16. Machine: **M4 Pro (MPS, fp32, no AMP — slower)**. Position: Level-1 pre-freeze
> gates (feed `closing_data` G0.2). Runs in parallel with `mtl_frontier` R4–R9 (A40) and C1 (M2 Pro).
>
> **Read first:** [`AGENT_PROMPT.md`](AGENT_PROMPT.md) (A2/A4 specs), `docs/research/baseline_gap_analysis.md`
> (A2 rationale), `docs/research/evaluation_protocol_review.md §4.1` (A4 rationale).

## ⚠ Substrate setup FIRST (the comparability constraint — do not skip)

Both gates must run on the **canonical frozen v14 substrate** (`check2hgi_design_k_resln_mae_l0_1`), the
SAME artifact the rest of the board uses. Two different needs:

- **A2 → SYNC, do NOT rebuild.** A2 compares arms on the frozen base; a substrate *rebuilt* on a different
  machine (MPS vs the build machine) can produce a numerically non-identical embedding, which silently
  breaks "same frozen base." **rsync the existing v14 artifacts** for FL/AL/AZ from the M2 Pro
  (`output/check2hgi/<state>/` + the v14 engine dir + seeded per-fold log_T) — do not regenerate them.
- **A4 → rebuild is the point, but pin the recipe.** A4 retrains the substrate per-fold on train-users-only,
  so it DOES need the v14 **build pipeline** recreated here (this is the "recreate check2hgi newest version"
  step). Pin the v14 build recipe identically (`scripts/.../regen_emb_t3.py` + design_k+resln+mae flags per
  `CANONICAL_VERSIONS.md §v14`). A4's **full-corpus baseline arm must be the canonical synced artifact**
  (not a fresh full rebuild) — and before trusting the A4 delta, **validate** your rebuilt full-corpus
  substrate matches the canonical (hash / downstream-metric spot-check); a mismatch means your build
  environment diverges and the train-only-vs-full delta would be confounded with the rebuild.

## A2 — feature-concat control (do FIRST — light, decisive)

- **Question:** is Check2HGI's +14–29 pp next-category lift the hierarchical-infomax *learning*, or just
  *feature injection*? (Check2HGI's nodes carry category one-hot + hour/dow sin/cos; HGI does not.)
- **Spec:** HGI (and/or POI2Vec) embedding ⊕ the same raw per-visit features → matched heads (`next_gru`
  cat / `next_stan_flow` reg), STL, FL+AL+AZ (AL/AZ are minutes on MPS; FL is slower — budget). No
  embedding retraining → light.
- **Gate (interpretation, G0.2):** if the concat closes most of the cat gap → reframe the substrate claim
  in the new paper (honest); if not → the claim is strengthened. Either way write the G0.2 row.

## A4 — transductivity bound (SECOND — heavy; defer to A40 if MPS too slow)

- **Question:** how much does training v14 on the full corpus (incl. validation-fold check-ins) inflate
  downstream numbers? (The substrate is transductive.)
- **Spec:** retrain v14 per-fold on **train users only**, FL × 1 seed; compare downstream cat/reg vs the
  canonical full-corpus v14. This is 5 substrate builds on MPS — **slow**; if it won't fit the MPS budget,
  **defer A4 to the A40** once R4–R9 finish, and ship A2 alone from the M4.
- **Gate (disclosure, G0.2):** small delta → one-paragraph defusal in the paper; large → report it +
  re-anchor headline numbers (and it motivates inductive Check2HGI as future-work).

## Guardrails

MPS: fp32, no AMP. Per-fold per-seed train-only priors + freshness preflight. Multi-seed {0,1,7,100} for
A2's headline (or AL/AZ multi-seed + FL seed-0 if budget-bound); single-seed pilot for A4. Paired Wilcoxon,
n and p. Pin `--canon`. Branch `study/pre-freeze-gates`; do not commit to `main`.

## Note on "escalate G0.2"

The G0.2 escalation of **C1** (per the user) is a doc action already done centrally in
`PRE_FREEZE_PROGRAM.md` (gate ledger) — the M4's job is to *run A2/A4* and write their G0.2 rows on close.
A2/A4 are already gates in the ledger; closing them is what advances the freeze.

## Hand-off

Each gate closes with a verdict + a G0.2 row in `closing_data/PLAN.md` + `PRE_FREEZE_PROGRAM.md`, plus
`STATE`/`log.md` rows. A2 and A4 are interpretation/disclosure gates (they change what the paper *claims*,
not the frozen numbers) but must resolve pre-freeze so the RUN_MATRIX records the right caveats.
