# PLAN — 8-page (fee-free) variant

> **Goal.** Cut `src/` from 10 to **8 pages** (the MobiWac fee-free size) without losing a single
> claim, number, test-bound verb, or disclosure. The submitted 10-page version is frozen at
> [`src_v1/`](src_v1/VERSION.md) — if the 8-page cut degrades the paper, `src_v1` remains the paper.
> Working copy: `src/`. Every wave ends with a full compile gate (pages / 0 undefined / 0 overfull /
> refs count) and a render spot-check of the load-bearing phrases.

## 0 · Measured baseline (2026-07-09, the frozen submission)

| Where | Measure |
|---|---|
| Body text | 7,134 words across §1–§8 (≈1,100 words/page two-column) |
| §1 876w · §2 1,017w · §3 319w · §4 772w · §5 1,570w · §6 1,836w · §7 575w · §8 169w | per-section |
| Floats | fig1+fig2 (TikZ), fig3 (3.26×1.79 in), fig4 (3.16×2.05 in), tbl1, tbl2, tbl3 (full-width, p9 top) |
| References | 37 entries, ≈1 full page (p10 both columns after §8) |
| Page map | p1 abs+§1 · p2 §2 · p3 §3(+fig1) · p4 §4(+fig2/tbl1) · p5 §5(+tbl2) · p6–8 §6(+fig3/fig4) · p9 §7+tbl3 · p10 §8+refs |

## 1 · The 2-page budget (where it comes from)

| Source | Target saving | How |
|---|---|---|
| **Text** | ≈1.2–1.3 pages (≈1,300 words, −18%) | per-section rewrite waves (§2 below) |
| **Figures** | ≈0.35 page | fig3/fig4 −20–25% height; fig1+fig2 TikZ compaction (smaller nodes/fonts, tighter spacing); consider merging fig3+fig4 into one two-panel figure (saves one caption block) |
| **Tables** | ≈0.1 page | tbl1 column compression; tbl2/tbl3 caption+footnote tightening (already tight — small) |
| **References** | ≈0.3 page (8–10 entries) | prune multi-cite clusters, never orphan a claim (§3 below) |

## 2 · Text-cut rules (HARD constraints for every cutter)

**NEVER cut or weaken (the survival list):**
1. Every number, CI, p-value, seed/fold count, and its test-bound verb ("outperforms" = CI>0/Holm;
   "match" = TOST). GLOSSARY.md §6 stays law (no em-dash, category/region naming, American English).
2. The disclosure chain: CA/TX provisional (abstract→§1→§6.2→tbl3→§7), the leak audit + the
   13-to-27-point confession + per-fold prior story, the AL deficit clause, FL/Ist sub-margin clause,
   Markov "indicative rather than protocol-matched", Acc@10 error convention, the two-protocol
   baseline pre-training statement, the cost paragraph (4.2M vs 1.1M), the figshare data statement,
   the repo URL, "collected 2009 to 2011".
3. The mobility bridge cites (Moura/Loureiro — author decision; Silva survey; Vielhaus; Bastug;
   Song) and the §3 scoping ("not cell association or handover"; "We build and evaluate no such
   service here").
4. The architecture truth sentences (§4.2 streams, exchange-not-shared-layers, region-node vectors).
5. All hidden `%` comments (response-letter arsenal) — they cost no page space; keep them.

**Cut targets by section (words, measured → target):**
- §1 876→≈660: compress the motivation walk (¶1–2), merge contribution preambles; keep all three
  bullets + glosses.
- §2 1,017→≈720: §2.1 compress the DGI→HGI history to two sentences; §2.2 one clause each for the
  novelty defusals (DRRGNN, KGTB); §2.3 compress the balancer negative-result discussion.
- §3 319→≈270: trim transitions; keep both scoping sentences.
- §4 772→≈630: tighten §4.1 graph prose; §4.2 keep streams/cost, compress connective tissue.
- §5 1,570→≈1,180: the biggest donor — windows/split tighten, leak paragraph compress (keep all six
  audit numbers + three grounds), metrics/TOST justification tighten (keep pre-registration + both
  margin arguments in shorter form), baselines paragraph compress (keep every deviation disclosure).
- §6 1,836→≈1,480: calibration sentence shorter; CTLE block compress (keep 33.45/29.69/75.15 +
  margins); cascade block compress (keep tie + both qualifications); keep §6.2 stats paragraph
  nearly intact (it is the paper's spine).
- §7 575→≈440: sketch tighten (keep CA 65.69/8,501 enrichment + 3–8 km vs 20–241 km); limitations
  keep all four topics, fewer words.
- §8 169→≈140.

## 3 · Reference prune candidates (verify each cite's sentence survives without it)

Keep-list is implicit (everything not below). Candidates, in prune order:
1. Balancer cluster (§2.3 cites 5 for one sentence): keep PCGrad + Nash-MTL + xin2022domtl; drop
   `chen2018gradnorm`… only if GradNorm's name is dropped from the sentence too; drop
   `kurin2022defense` (xin covers the negative result).
2. Next-place list (§2.2 cites 5): keep DeepMove + STAN + GETNext; drop `liu2016strnn`,
   `yang2020flashback` (names leave the sentence with them).
3. Cascade lineage (4 cites): keep ye2013 (the lineage root) + CSLSL + CatDM; drop `he2017lbpr`.
4. `senushkin2023aligned` / `liu2023famo` (if cited in the balancer sentence only) — same rule as 1.
5. `luca2021mobilitysurvey` — only if §2.2's grid-cell formulation sentence is reworded to stand
   without it (risky; last resort).
NEVER prune: all mobility-bridge cites, caruana1997multitask, both data-source cites, all baseline
method cites (they anchor Table 3), huang2023hgi, velickovic2019dgi, lin2021ctle, silva2025mtlnet.

## 4 · Execution waves (each gated by a full compile + render spot-check)

1. **W1 floats:** fig3/fig4 heights; TikZ compaction; table captions. Gate: pages ≤ 9.5-ish.
2. **W2 text:** apply the eight section rewrites (agent-proposed, closer-audited line by line
   against the survival list before applying). Gate: ≤ 8 pages + survival-list render check.
3. **W3 refs:** apply the prune with per-sentence verification. Gate: 8 pages, 0 undefined,
   bibtex clean.
4. **W4 verify:** claims-preservation audit (agent: diff src vs src_v1 rendered text; assert every
   survival-list item present; digit audit: every number in src ∈ src_v1 unchanged).
5. **W5 reviews:** fresh MobiWac panel on the 8-page build (per author instruction).

## 5 · Abort criteria

If W2+W3 cannot reach 8 pages without touching the survival list, stop and report: the 10-page
`src_v1` stays the submission and this plan closes as "attempted, not viable". The fee is cheaper
than a weakened paper.
