# Overlapping-window validation probe — results (2026-06-03)

Validates the pipeline-audit HIGH finding (non-overlapping stride=9 windows cap both ceilings).
Isolated harness (`scripts/mtl_improvement/overlap_probe.py`): builds cat (next_gru+logit-adjust τ=0.5)
and reg (next_stan ≡ next_stan_flow α=0) at stride {9 control, 1/2 overlap} through ONE identical loop
matching next_cv/p1; user-grouped 5-fold seed=42, 50ep, bs2048. Control arm reproduces the frozen
ceiling (within ~+1pp harness optimism), so the control→overlap delta is the windowing effect alone.

| task | state | control (non-overlap, s9) | overlap | LIFT | data multiple | frozen ceiling |
|---|---|---|---|---|---|---|
| cat | AL | 50.73 ± 2.41 | 60.23 ± 3.74 (s1) | **+9.50** | 8.5× (12.7k→108k) | 49.97 |
| cat | FL | 71.10 ± 0.68 | 72.40 ± 0.65 (s2) | **+1.30** | 4.5× (159k→693k) | 69.97 |
| reg | AL | 63.99 ± 1.78 | 68.96 ± 3.11 (s1) | **+4.97** | 8.5× (12.7k→108k) | 62.88 |

## Findings
- **The lift is data-scarcity-dependent**: LARGE at small states (AL cat +9.5, reg +5.0), MODEST at
  large (FL cat +1.3). At FL (159k seqs) the model is near data-saturation; at AL (12.7k) 8.5× more
  (history→next) supervision helps enormously.
- Lifts are well beyond σ (AL cat ~2.5σ, reg ~1.9σ; FL cat small but σ=0.65). Overlapping windows are
  genuinely different prediction pairs (shifted history + different target), NOT duplicates.
- **Leak-safe**: StratifiedGroupKFold(userid) keeps a user's windows in one fold → overlap can't cross
  train/val. Val rows within a held-out user are correlated → wider σ (seen: overlap σ > control σ),
  a variance effect, not upward bias.
- Harness control reproduces the frozen ceilings within ~+1pp (50.73≈49.97 / 63.99≈62.88 / 71.10≈69.97)
  — the isolated comparison is faithful; absolute overlap numbers need real-pipeline re-validation.

## Caveats / open
- FL cat used stride=2 (memory); stride=1 might add a little. FL reg + AZ/GE overlap not yet run.
- Numbers are from the isolated harness, ~+1pp optimistic vs the real trainers. The LIFT is the signal.
- Untested: does overlap help MTL too, or only STL? If it lifts STL more than MTL at small states, the
  STL→MTL gap WIDENS (relevant to the regime finding + the dual-tower story). Needs an MTL-with-overlap run.

## Strategic implication (FOUNDATIONAL — paper-touching)
The frozen (c)/(d) ceilings, the MTL board, the per-fold log_T, and the v11 PAPER CANON are all on
non-overlapping windows → the small-state ceilings (and the small↔large scale story) are capped by
~5-9.5pp. Adopting overlap is a major rebuild that changes the paper. Decision needed: adopt-as-canon
(rebuild everything) vs document-as-headroom/future-work vs investigate-further (real-pipeline + MTL).

## Real-pipeline confirmation (2026-06-03) — caveat resolved
Built an isolated probe engine `check2hgi_dk_ovl` (v14 embeddings re-windowed stride=1; embeddings/
region symlinked from v14; frozen substrate untouched) and ran the ACTUAL trainers at AL:
| task | frozen (non-overlap) | overlap (REAL pipeline) | lift | isolated harness |
|---|---|---|---|---|
| cat | 49.97 | **59.74 ± 3.57** | **+9.77** | 60.23 |
| reg | 62.88 | **68.01 ± 4.22** | **+5.13** | 68.96 |
The real pipeline reproduces the isolated-harness lift exactly → the ~1pp harness-optimism caveat is
RESOLVED. Overlapping windows genuinely lift the AL STL ceilings by ~+5 to +9.8pp in the real trainers.
Build: `scripts/mtl_improvement/build_overlap_probe_engine.py`; engine-aware region seq (seq_engine).

## STILL OPEN — the decisive question: does overlap help MTL too?
If overlap lifts the MTL model as much as STL → "rising tide" (rebuild for higher numbers, regime
finding unchanged). If it lifts STL MORE than MTL at small states → the STL→MTL gap WIDENS, which
*strengthens* the dual-tower/regime story. Needs an MTL-with-overlap run (engine-aware MTL fold
creator + log_T-KD-off control). NEXT.

## MTL-with-overlap (2026-06-03) — THE DECISIVE RESULT: cat rising-tide, reg gap WIDENS
Real MTL pipeline (cross-attn, next_gru cat + next_stan reg, static_weight cw=0.75, KD OFF, AL 5f/50ep,
joint_best basis). Control = v14 non-overlap, overlap = check2hgi_dk_ovl — IDENTICAL recipe, only windowing differs.

| task | MTL control (v14 n/o) | MTL overlap | MTL lift | STL lift (for ref) |
|---|---|---|---|---|
| cat F1 | 46.30 | **55.21** | **+8.92** | +9.77 |
| reg Acc@10 | 54.54 | **55.05** | **+0.50** | +5.13 |

**cat = RISING TIDE**: overlap lifts cat ~equally in STL (+9.8) and MTL (+8.9) — the cat MTL pathway
exploits the extra data. **reg = GAP WIDENS**: overlap lifts STL reg (+5.1) but MTL reg barely moves
(+0.5). The STL→MTL reg gap goes 62.88−54.54=**8.34** (non-overlap) → 68.01−55.05=**12.96** (overlap).
The shared-backbone reg pathway (the architectural bottleneck) CANNOT use the extra data; STL reg does.

**Interpretation:** overlap does NOT undermine the regime finding — it STRENGTHENS it. More data makes the
reg architectural bottleneck MORE visible (bigger STL→MTL gap), which is direct evidence FOR the dual-tower
motivation. Cat benefits across STL+MTL (a clean rising tide). Caveats: AL only, single seed, KD-off +
prior-free reg recipe (control uses the SAME recipe so the windowing delta is valid; absolute≠board).

## Strategic options
1. ADOPT overlap as canon — rebuild substrate inputs + log_T + re-freeze ceilings + re-run the board at
   all states. Lifts cat (STL+MTL) + STL reg; strengthens the reg regime story. Major rebuild, changes paper.
2. DOCUMENT as a key finding + future-work — keep the consistent non-overlap canon (all within-study
   comparisons valid), add overlap as a "data-formation headroom + it sharpens the regime finding" result.
3. CONFIRM-FIRST — run the cat-rising-tide / reg-gap-widens pattern at AZ/GE/FL before deciding (the AL
   result is single-state, single-seed).

## HGI-overlap STL reg (2026-06-03) — the composite substrate also benefits, stays tied
Overlapping sequences + HGI region embeddings (--engine-override check2hgi_dk_ovl --region-emb-source hgi):
HGI-overlap reg AL = **68.47 ± 4.12** vs non-overlap HGI-α0 63.58 = **+4.89** (≈ v14's +5.13). With overlap,
v14 (68.01) ≈ HGI (68.47) for STL reg — they stay tied (consistent with T1.4 "v14≈HGI once α=0-hardened").
So overlap is a uniform STL-reg lift across BOTH substrates; the composite (d) reg arm rises to ~68.5 too.

## DECISION (user 2026-06-03): document as a key finding + future-work; KEEP the non-overlap canon
Rationale: consistency of the whole study. The frozen (c)/(d), the MTL board, the per-fold log_T, and the
v11 paper canon are all non-overlapping; every within-study comparison is internally valid on that canon.
Overlap is recorded as (1) a validated data-formation headroom finding and (2) evidence that SHARPENS the
regime conclusion (reg gap widens with more data). NOT adopted as canon (would require a full multi-state
rebuild + re-paper). Future-work memo: docs/future_works/overlapping_windows.md.

## Full AL picture (validated, real pipeline)
| metric | non-overlap | overlap | lift |
|---|---|---|---|
| STL cat | 49.97 | 59.74 | +9.77 |
| STL reg (v14) | 62.88 | 68.01 | +5.13 |
| STL reg (HGI) | 63.58 | 68.47 | +4.89 |
| MTL cat joint/disjoint | 46.30 / 46.52 | 55.21 / 55.90 | +8.92 / +9.39 |
| MTL reg joint/disjoint | 54.54 / 53.47 | 55.05 / 54.46 | +0.50 / +1.00 |
