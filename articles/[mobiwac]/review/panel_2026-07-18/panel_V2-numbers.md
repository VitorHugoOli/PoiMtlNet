(1) MISMATCHES

1. **figs/fig4_deltas.tex:7 — stale OLD-footing caption (renders in the built PDF as Fig. 3 caption).**
   Current: "the region gain (Acc@10) rises across the five U.S. states (TX and CA: one seed) and is also positive at Istanbul."
   Expected: no "(TX and CA: one seed)" parenthetical — all six joint cells are now n=20 (four seeds x five folds; the fig4_deltas.py STATES comment and Table III footnote "All joint and dedicated entries average four seeds over five folds" both say so). This is the single-seed-at-CA/TX old footing the sweep was meant to catch, and it directly contradicts Table III's footnote in the same built PDF (main.pdf p.7).

2. **Comment-only staleness (does not render, but contradicts the current footing; will mislead the next editor):**
   - sections/06_results.tex:6 — header comment "CA/TX joint = seed-0 n=5 provisional (reg CIs entirely above +2)". Stale; body prose (lines 77-92) correctly says n=20 all six.
   - figs/fig4_deltas.py:15-16 — module docstring "AL/AZ/FL/Istanbul are n=20 on both arms; CA/TX joint cells are seed-0 provisional (A1 pending)". Stale; contradicted by the corrected in-code comment at lines 38-41 ("ALL SIX datasets n=20 on both arms (CA/TX A1 landed 2026-07-11)").

3. **Minor observations (not errors vs ground truth):**
   - AZ category delta: Table III's rounded cells give 65.79 − 56.43 = 9.36, while prose (06_results.tex:61), fig4_deltas.py:44, and the ground truth all say +9.35 (unrounded means). Rounding artifact only; consistent with ground truth as given.
   - 01_introduction.tex:33 "even though each dedicated model is tuned per dataset": §6.2 (06_results.tex:43) states only the dedicated *category* model is tuned per dataset (region uses the strongest fixed configuration). The intro sentence sits inside the category claim so it is defensible, but a strict reader could take "each dedicated model" to cover region.

(2) ALL-CLEAR (verified against the ground truth and against Table III; source AND built `pdftotext -layout main.pdf` text, which is newer than every source file)

- **Table III cells (tbl3_results.tex:41-46):** all 12 joint cells, 12 dedicated cells, and all externals (HMT-GRN, ReHDM, STAN, POI-RGNN, Markov-K) match ground truth exactly, including ± sds in the provenance comment (lines 14-17).
- **Table III markers:** bold on all six joint category cells; region bold+↑ only at Ist/FL/TX/CA; AL/AZ region joint NOT bold, marked ≈ — matches the verdict map (never-upgrade-AZ respected: "no change, 0.00" and "a match, not a gain" in §6.2).
- **Table III footnotes:** †STAN partial folds (TX 4/5, CA 2/5, seed 0), ‡ReHDM single seed at TX/CA, "All joint and dedicated entries average four seeds over five folds", "±: sd across seeds" — all present and correct.
- **Abstract:** "+28 to +40" (substrate gaps 27.63-39.62, "about" carries AZ's rounding), "+5 to +9" (5.33-9.35), "four of the six", softened TOST wording ("matching it (statistically, within two points)", no acronym), "Istanbul... slightly ahead on region" (+0.19).
- **Intro contributions:** "+28 to +40", "+5 to +9", "four of the six", monotone region trend across states with CA largest (+2.20), "average four random initializations".
- **§6.1 vs Table II:** +29.31/+27.63/+39.62/+37.95/+37.47/+28.09 all match tbl2; band phrasing "+28 to +29 smaller / +37 to +40 large"; CTLE 33.45/29.69 vs FL place 35.53 ("about two points below" = 2.08) and "below our 75.15" = Table II FL check-in cell.
- **§6.2:** delta list +5.33..+9.35 (smallest FL, largest AZ, Ist +8.58); region +0.71 FL / +2.11 TX / +2.20 CA / +0.19 Ist, match at AL (−0.41) and AZ (0.00); trend "−0.41 to +2.20"; all six CIs exact (AL −0.63..−0.20, AZ −0.08..+0.07, FL +0.67..+0.76, Ist +0.15..+0.23, TX +2.10..+2.13 above margin, CA +2.19..+2.21 above margin); n=20/pair-per-seed-means n=4/Holm p<0.001; the new joint-best convention disclosure with diag-best as the ≤0.06/≤0.11 robustness check (old two-epoch convention fully inverted, no residue).
- **Floors:** stride-1 Markov region floor "51 to 72" (51.23-72.47) and margin "4.9 to 10.3" (min FL 77.41−72.47=4.94, max Ist 75.35−65.06=10.29) — both correct at all six; the old "43-65" floor is gone.
- **§6.3 Istanbul:** +8.58 (54.74 vs 63.32), +0.19 (75.16 vs 75.35) with CI>0 + TOST, "four seeds over five folds", externals 69.33/61.86/60.4 and 24.55/30.12 — all match.
- **§7:** "65.69 percent... ten regions of 8,501" matches Table III CA joint reg; "over 500 times better" checks out (65.69/0.1176 ≈ 558).
- **§8 conclusion:** "four of the six"; "at least 4 Acc@10 points over the strongest region reference" — min gap is AL joint 69.70 vs ReHDM 65.38 = 4.32, holds at all six (Ist 6.02, AZ 6.46, FL 4.42 vs STAN, TX 5.39, CA 7.17); "at least 33 macro-F1 over POI-RGNN" — min is Istanbul 63.32−30.12 = 33.20, holds everywhere.
- **fig4_deltas.py + fig4_deltas.pdf:** STATES values are exactly the ground-truth deltas; the rendered PDF (regenerated 18:03, after the .py edit) shows +8.6/+7.7/+9.3/+5.3/+7.5/+6.5 and +0.2/−0.4/+0.0/+0.7/+2.1/+2.2 — correct 1-dp roundings, no single-seed asterisks left in the plot.
- **§5.3:** sd range "0.01 to 0.18" is consistent with the ground-truth CIs (back-solved via t(3, 90%): CA≈0.01 ... AL≈0.18); "pass a margin as small as one point at Alabama and Arizona" holds (both CIs within ±1).
- **§5.1/§5.2 vs Table I:** ~114k check-ins/1,109 regions AL, ~3.2M/8,501 CA; Food majority ~25% FL to 34% AL (24.7/34.2 in Table I).
- **Old-footing sweep of the built text:** no hits for 9.40, 5.34, 8.59, +0.28, −0.31, +0.72, 2.12, 7.72, 6.44, "0.04 to 0.15", "half a point", "43 to 65" (the cascade sentence now says "a quarter of a point"). The two remaining "single seed" mentions are the intentional disclosures (§7 shortlist analysis; ReHDM footnote).
- **Cross-references:** no "??" in the built text; Fig. 1 = dataflow, Fig. 2 = model, Fig. 3 = deltas (fig3_embquality correctly cut, numbering consistent); Tables I/II/III and Sections II-B, III, IV-A/B, V-C, V-D, VI-A/B all resolve to the right targets; 32 bibitems, no undefined/multiply-defined warnings in main.log; main.pdf (18:03) is newer than every .tex/.py source.

(3) COULD NOT VERIFY (no ground truth provided; checked only for internal consistency)

- Table II substrate cells themselves (54.65/55.87/57.13/75.15/69.95/70.26 and HGI arms) — prompt gave no substrate ground truth; prose matches the table and the table's provenance comments.
- §6.1 auxiliaries: silhouette 0.57/0.00, purity 0.98/0.78, "64 to 90 percent" averaging control, CTLE frozen gaps +37.8/+37.0/+28.7.
- §6.2 freeze-control "within 0.3"; cascade "quarter of a point"; §4.2 parameter counts (4.2M vs 1.1M AL, 5.2 vs 2.0 CA); §7 shortlist distances (3-8 km vs 20-241 km); §5.2 A4 leak numbers (−0.33..+0.01 / 0.00..+0.29, 67-87% coverage) and the 13-27-point prior-leak figure; Table I statistics. All would need their source JSONs/board docs (docs/studies/closing_data/RESULTS_BOARD.md et al.) to confirm; none conflicts with anything in the provided ground truth.

Key files: /Users/vitor/Desktop/mestrado/ingred/articles/[mobiwac]/src/figs/fig4_deltas.tex (the one rendering mismatch, line 7), /Users/vitor/Desktop/mestrado/ingred/articles/[mobiwac]/src/sections/06_results.tex (stale header comment, line 6), /Users/vitor/Desktop/mestrado/ingred/articles/[mobiwac]/src/figs/fig4_deltas.py (stale docstring, lines 15-16).