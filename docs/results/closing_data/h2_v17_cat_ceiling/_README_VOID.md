# ⚠ VOID for the headline — rejected "matched-knob" STL cat ceiling (kept as a labeled ablation only)

These JSONs are the **rejected matched-knob H2 arm**: the STL `next_gru` cat ceiling forced to the MTL's knobs
(**bs8192 @ cat-lr 1e-3**). Despite the canonical-sounding tags (`<state>_s<seed>_stl_cat_ceiling_v17`), **these are
NOT the v17 ceiling**. An advisor panel (2026-07-03, unanimous) ruled the matched-knob comparison **baseline
sabotage**: it handicaps the STL below its own optimum (AL 53.58 vs the true ceiling 56.82) and inflates Δcat to a
bogus +10.96.

- **The real v17 ceilings** (best-vs-best, per-state max, n=20) live in
  `docs/studies/closing_data/v17_completion/CEILINGS_N20_FINAL.md` + `cat_ceiling_sweep/sweep_results/`.
- Keep these files ONLY as the **iso-budget ablation** record ("what if both arms share knobs") — never cite them as
  the dedicated-model ceiling in any table.

(Marker added 2026-07-08, post-merge audit of PR #58 — finding F3.)
