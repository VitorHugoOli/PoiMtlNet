# Audit — T1.4 ceiling + Tier S (STL head search) — 2026-06-03

**Scope.** Independent audit of the completed Tier-1.4 (tuned-incumbent ceiling) + Tier-S (STL head
search) work, commissioned after "we haven't found any interesting finds." Method: read the result-of-
record (`TIER01_RESULTS.md`, `HANDOFF.md`, INDEX S.1–S.4, `overlapping_windows.md`), two parallel
sub-agents (Tier-S fairness audit + SOTA-head literature check), code inspection, and an advisor pass.

**One-line call.** The head-null is *expected and sound* (the regime finding predicted the head is not
the lever) — so this is **not a failed study**; the value is in three by-products. Of them, **one is a
genuine reviewer-exposable flaw that should be re-opened, not shelved (non-overlapping windowing)**, one
is **a real finding that is currently under-evidenced (α=0 "prior is a drag")**, and the Tier-S negative
is **conservative but has named, cheap-to-close cracks.** None of this invalidates the chain; it sharpens
what to confirm before the paper leans on it.

---

## 0. Correcting the premise: the head-null was *expected*, not a dead end
The regime finding (the MTL→STL reg gap is architectural) predicts the **head is not the lever**. So
"no head beats the tuned incumbent" is the *expected* result, not a disappointment. The interesting
finds are the by-products surfaced *while* establishing that null:

1. **The HGI-reg edge was a prior artifact** (genuine positive, well-evidenced — §2c). Once both
   substrates are α=0-hardened, v14 ≈ HGI STL-reg at all four states (every Δ ≤ 0.70pp, <0.5σ). HGI's
   apparent reg-substrate advantage **was the log_T prior, not the substrate** → the substrate axis is
   exhausted → *strengthens* the regime/architecture thesis. This is paper-relevant and is the
   best-evidenced new result of the tier.
2. **Non-overlapping windowing caps every ceiling** (§1 — the loose point that needs a user decision).
3. **A silently-ignored `--cat-head` bug** had mis-pinned the cat ceiling by up to +8pp (caught + fixed;
   containment check in §4).

---

## 1. ⚠ FINDING (re-open, don't shelve): non-overlapping windows under-supervise by ~8×

**What.** `generate_sequences` defaults `stride = window = 9` → **one (history→next) target per 9
check-ins**. AL: 113,846 check-ins → only 12,709 training windows. The validated probe (AL, real
pipeline) lifts **STL cat +9.77 / STL reg +5.13 / MTL cat +8.92** with dense (stride-1) supervision.

**Why it's a real flaw, not a discretionary lever (SOTA agent).** The field segments non-overlapping
(GETNext: 24h trajectories; STAN/DeepMove/LSTPM/Flashback) **but supervises *densely*** — prefix-
expansion, ~one target per position (~L−1 pairs per segment). So standard practice gets stride-1-
*equivalent* target density even with non-overlapping segmentation. **Our outlier is the *combination*
of non-overlapping segmentation AND one-target-per-window** — that is what throws away ~8× of the
supervision. Do **not** cite GETNext's daily segmentation as precedent for sparse targets — a reviewer
who knows GETNext catches that (it segments non-overlapping but supervises densely).

**Cost — do NOT sell this as cheap (advisor).** The ~8× lift *is* the extra (history→next) pairs; you
cannot get it without them, and those pairs invalidate `next_region.parquet`, the per-fold **log_T**
(built from those pairs), the **frozen (c)/(d) ceilings**, the **MTL board**, and **§0.1**. Only the
substrate/embeddings are reusable (symlink, as the `check2hgi_dk_ovl` probe already does). So it is a
**log_T + ceilings + board + paper-canon rebuild**, not a from-scratch substrate rebuild — worth it
*because it fixes a genuine sub-standard-practice flaw*, but not free.

**What it threatens / what survives (advisor — sharpen this for the user).**
- **Survives:** every *internal Δ* (v14-vs-canonical, MTL-vs-STL gap) is non-overlap on both arms →
  apples-to-apples → unaffected.
- **Exposed:** (a) absolute-number credibility vs other Foursquare papers (our baselines look ~5–10pp
  low at small states); (b) the reviewer attack *"your reg gap is just under-supervision."*
- **Partial pre-rebuttal we already hold:** the AL probe shows more data makes the MTL reg gap *widen*
  (8.34→12.96), not close — so the gap is not merely under-supervision. **But that rebuttal is n=1
  state, single seed.**

**The bigger point I initially missed (advisor): windowing is *upstream of the regime finding itself*.**
The headline claim — "the MTL reg gap is architectural" — was measured **entirely under the
under-supervised regime**. The AL overlap probe supports it (gap widens with more data) but is one
state, one seed. **Before the paper leans on the architectural-gap headline, re-confirm the regime
finding under dense supervision at AZ/GE/FL, multi-seed.** Frame the re-open not as "is there a better
head" but as **"does our headline survive correct supervision."** That is both the risk and the
opportunity, and it is the real loose point of the whole study.

**Recommended decision (user's call).** Either (i) **adopt dense targets** (rebuild log_T + ceilings +
board + §0.1; expect large STL/MTL cat lifts + STL reg lifts at small states, the MTL reg gap widening
— report the widening as a headline, not a regression), or (ii) **keep non-overlap but pre-empt the
reviewer** with a dense-supervision robustness appendix (the regime re-confirm above) + an explicit
methodological justification. The current "shelve as future-work for internal consistency" is the one
choice I'd advise against — it leaves the headline measured only in the regime a reviewer will question.

---

## 2. FINDING (real but under-evidenced): α=0 "the transition prior is a drag"

**What.** The T1.4 reg tune found **α=0 (log_T prior fully OFF) wins STL reg at every state**
(62.88/55.11/58.45/73.31 vs default-prior 62.32/…/70.28).

**(a) Plausible — consistent with the field (SOTA agent).** The literature has moved *off* fixed 1st-
order Markov priors: FPMC (WWW'10) → FOSSIL (ICDM'16, relaxes 1st-order) → GETNext (SIGIR'22) *learns*
the transition map and adds it as a residual; STAN injects spatio-temporal relation bias *inside
attention*, not as an additive Markov logit prior. So "a fixed additive 1st-order Markov logit prior is
redundant/harmful once a strong self-attention encoder + contextual embeddings are present" is *not*, on
its face, a red flag.

**(b) But the fairness of the comparison is suspect — settle by INSPECTION before claiming it.**
Code inspection (`next_stan_flow/head.py:101-104`): "prior-on" uses a **learnable** `alpha =
nn.Parameter(init 0.1)`; "prior-off" freezes α to 0. **A learnable α can always set itself to 0**, so
prior-on (62.32) should be ≥ frozen-α=0 (62.88) — yet it is 0.56pp *worse*. That means α did **not**
converge to its optimum (or the encoder co-adapted to a nonzero prior). So "α=0 wins" is at least partly
an **optimization artifact**, not a clean "embeddings subsume transitions." **Action (a read, not a
GPU run):** pull the saved α from the prior-on run dirs — if α≈0 yet still scored 62.32, the gap is an
unexplained artifact (e.g. weight-decay on α / the `alpha*0.0` path); if α stayed ≈0.1, then the model
*couldn't/didn't* drop the prior — a weaker claim than "the prior is useless."

**(c) The well-evidenced corollary IS a genuine finding — promote it.** Once both substrates are
α=0-hardened, HGI's reg edge **vanishes** (v14≈HGI, every Δ≤0.70pp <0.5σ). HGI's reg-substrate
advantage **was the prior artifact** → substrate axis exhausted → strengthens the regime thesis. This
is better-evidenced than the α=0 tuning claim and is paper-relevant. Lead with it.

**(d) The MTL-vs-STL "contradiction" is coherent once named.** In MTL the log_T enters as a **KD loss on
the shared representation** (helps a starved backbone, +2.4–5pp); in STL it enters as an **additive
logit bias at inference** (hurts a head that already fits transitions). Different interventions on
different objects — name the mechanism and the apparent inconsistency dissolves. Do **not** write "the
same log_T helps in MTL and hurts in STL" without that distinction.

**Two cheap diagnostics to make the claim airtight:** (1) standalone log_T Acc@10 (prior alone, no
encoder) — decent-but-additive-nothing ⇒ redundant; poor ⇒ just a bad prior; (2) log_T row-coverage at
small states (a 1k–9k-class train-only matrix is sparse → "bad prior at low data" is the null to rule
out). Reframe the written claim to **"the fixed additive prior is not needed here,"** not "transition
priors don't help."

---

## 3. Tier-S negative: conservative, but with named cracks (not yet airtight)

**Sound:** the dominated clusters lose by margins no LR sweep closes — CNNs −13 to −17, transformers
−8, Mamba −8 (both tasks), STAN-as-cat −7.4 *even fully tuned*. The two most plausible challengers
(next_lstm cat, STAN-for-cat) got genuine post-hoc per-arch tunes and still lost/tied. "The head is not
the lever" stands.

**Cracks (all cheap to close):**
- **Per-arch LR mini-sweep (hard rule 7) was SKIPPED in the screen** — every challenger ran at registry
  default, single LR. Only STAN-for-cat + next_lstm got real sweeps, post-hoc. So the negative is a
  *conservative promotion test*, not an equal-budget tournament (the study labels it as such — INDEX
  S.3 honest-labelling rule).
- **The one crack with reviewer traction: `next_lstm` cat.** Nominal single-seed wins at **GE +0.51
  (clears the 0.5pp gate), AZ +0.48**, FL +0.14 — but its fair tune was run only at **AL (its single
  losing state, −0.21)**. The multi-seed the gate *requires* was never run where it wins → this is
  "failed to demonstrate a win," not "demonstrated no win." **Close it:** multi-seed `next_lstm` (and
  `next_single`, which wins cat at GE +1.45) at GE/AZ.
- **`next_hybrid` unaccounted** — in the cat-screen driver loop, passes the unit gate, but absent from
  the 7-row result TSV while INDEX claims 8 encoders. Reporting gap (results/ is A40-only).
- **2 hierarchical-softmax reg heads (`*_hsm`) never GPU-screened** (need a prebuilt `hierarchy_path`;
  deferred, not bit-rot).
- **The unit gate does not verify prior-consumption** — it tests shape/finiteness only; "did a
  prior-aware head actually consume log_T" rests on p1 wiring, not the gate (the silent-kwarg-drop trap
  is only *indirectly* closed for next_stan_flow via its distinct on/off numbers).

---

## 4. Smaller loose points

- **Single-seed (seed=42) frozen ceilings** are the immutable yardstick for all of T2–T5 — this
  contradicts the track's own n≥10 statistical-gate rule. Most acute: **FL (c)-cat (69.97) sits BELOW
  the MTL diagnostic-best it bounds (70.26)** — the *same symptom class* (ceiling < the thing it
  bounds) that exposed the cat-head bug. Do not wave it off as "within σ"; **multi-seed FL cat is the
  cheap confirm.**
- **Cat-bug containment.** The STL ceiling was re-pinned (next_single → real next_gru). Verify the fix
  **propagated** to the **(d) composite cat arm** and the **S.4 huge-picture table** (anything computed
  pre-fix). The MTL board cat is fine (`--cat-head` works on the MTL track; the bug is `--task next`
  only).

---

## 5. Recommended actions (ranked; all bounded)
1. **Decide windowing (§1) — the only one needing a user call.** Re-confirm the regime finding under
   dense supervision at AZ/GE/FL multi-seed *before* the paper leans on the architectural-gap headline.
   Then adopt dense targets (rebuild) or write the robustness appendix. Do **not** leave it shelved.
2. **Settle α=0 by inspection (§2b)** — read the saved α from prior-on runs; if needed, the 2 cheap
   diagnostics (standalone log_T, coverage). Reframe the claim. **Promote the HGI-prior-artifact
   corollary (§2c) as the real finding.**
3. **Close the Tier-S cracks (§3)** — multi-seed `next_lstm`/`next_single` cat at GE/AZ; account for
   `next_hybrid`; note the `*_hsm` deferral. Cheap; converts "failed to show a win" → airtight.
4. **Multi-seed FL (c)-cat (§4)** — resolve the ceiling-below-MTL inversion; single-seed yardstick is a
   standing risk.

**None of this blocks Tier 2 (the dual-tower).** Items 1 (re-confirm) and 4 (multi-seed FL cat) are the
two that, if skipped, leave a reviewer opening; item 1 is also the one most likely to *change paper
numbers*. The head search itself can be closed as a conservative negative once item 3 lands.

---

## 6. Decisions + action items (2026-06-03)

### Windowing — DECISION (user, 2026-06-03): defer the rebuild to a dedicated follow-up study
The first article's numbers are all non-overlapping; internal Δs (regime finding, v14-vs-canonical) are
apples-to-apples under non-overlap and **hold for this paper's comparative claims**. A mid-paper dense-
supervision rebuild (log_T + ceilings + board + §0.1) is disruptive for no comparative gain. **So: close
this study on consistent non-overlap numbers; spin the dense-supervision rebuild + the multi-state regime
re-confirm into a dedicated follow-up study** (seed: `future_works/overlapping_windows.md`).
**Required caveat (do NOT ship silent):** the current paper must carry a short **limitations note** —
"sequences use non-overlapping windows (one target / 9 check-ins); standard next-POI practice supervises
densely, so absolute numbers are conservative" — **plus the AL dense-supervision probe** (more data →
the MTL reg gap *widens* 8.34→12.96, not closes) as a pre-emption of the "your reg gap is just under-
supervision" reviewer attack. Honest residual: the architectural-gap headline is then multi-state-
validated only under non-overlap + n=1 (AL) under dense supervision — state that explicitly; the follow-up
study removes it.

### Closed THIS session (here, no A40 needed)
- **Cat-bug propagation swept.** Frozen (c)/(d) + T1.4 + S.4 already use the corrected next_gru cat
  (49.97/51.01/58.12/69.97). Two stale spots carrying the bugged `next_single` cat were found + annotated
  in `INDEX.html`: the **T1.1 pre-hardening table** (39.13/65.88 — now flagged INVALID, cat column ran
  next_single) and the **T0.3 multi-seed cat claim** (65.88 — now flagged: relative v14≈v11 holds, absolute
  is not the next_gru ceiling). No live ceiling or Δ used the bugged numbers (freeze-sanity asserts
  arch=NextHeadGRU).
- **α=0 fairness diagnosed by code inspection** (`next_stan_flow/head.py:101-104`): "prior-on" uses a
  **learnable** `alpha=Parameter(init 0.1)` → it *can* set itself to 0, yet scored 0.56pp WORSE than
  frozen-α=0. So "α=0 wins" is at least partly an optimization artifact, not a clean "embeddings subsume
  transitions." The saved α value would confirm — but it is NOT in the committed `docs/results/P1/*.json`
  (only on the A40 run dirs) → see open item O1.

### Close-out status (2026-06-04)
| item | status | result |
|---|---|---|
| **O1** | ✅ CLOSED | Both hypotheses FALSIFIED — α converges LARGE (0.45→1.09), prior-ON still 0.56–3.03pp < α=0. The fixed additive prior is a net drag on the STL-reg ceiling (mechanism not isolated; train/val-gap most-likely). See below + `docs/results/mtl_improvement/TIER01_RESULTS.md §O1`. |
| **O2** | ✅ CLOSED | next_lstm ties at all 4 states multi-seed (single-seed wins evaporate); next_single GE +1.54±0.17 (robust, GE-specific) → T5.2 candidate, fails ≥2-band gate, does NOT re-open (c). Multi-band negative HOLDS. |
| **O3** | ✅ CLOSED | FL (c)-cat 69.96±0.08 validates seed42 69.97; inversion vs MTL-diag 70.26 PERSISTS multi-seed (−0.30) but tiny + explained (oracle-epoch + small FL cat transfer), not a bug. |
| **O4** | ✅ CLOSED | next_hybrid AL cat 49.34 (< 49.97 floor; was a reporting omission) + `*_hsm` deferral noted. INDEX §S.1/S.2. |
| **O5** | ✅ CLOSED | Limitation (vi) added to `PAPER_DRAFT.md §7 Beat 3` (windowing + AL rebuttal); dense rebuild deferred. |

### OPEN — for the A40 to execute / close (ranked)
- **O1 (cheap, settles §2). — ✅ CLOSED 2026-06-04.** Re-ran the prior-ON config and read the converged
  learned α (faithful re-run; the model was never persisted). **Result overturns BOTH framings in this item:**
  α did NOT converge ≈0 and did NOT stay ≈0.1 — it converges **large (AL +0.454 / AZ +0.789 / GE
  +0.944 / FL +1.095; larger at higher-coverage states, n=4 — suggestive only)**, i.e. the model actively *leans
  into* the prior, yet prior-ON still scores 0.56–3.03pp BELOW α=0 at every state. The replication is exact
  (reproduces 62.32/70.28/52.87/55.81) and the standalone log_T prior (50.86/66.15 at AL/FL) validates against
  the authoritative Markov-1-region floors (47.01/65.05), so the prior carries real signal. **Phenomenology: the
  fixed additive log_T prior is a net drag on the STL-reg ceiling** (a learnable α leans in yet generalizes worse
  than α=0). **Mechanism NOT isolated** — most-likely cause is the train-only prior's train/val generalization
  gap, but the probe does not discriminate it from additive scale-mismatch or transition double-counting (advisor
  2026-06-04). **Reframed claim: "the fixed additive log_T prior is a net drag on the STL reg ceiling"** — NOT
  "embeddings subsume transitions," NOT a stuck-α artifact, NOT the co-adaptation mechanism asserted as proven.
  Leak audit: NONE. The §2c HGI-prior-artifact corollary stands and is strengthened. Full write-up + table:
  `docs/results/mtl_improvement/TIER01_RESULTS.md §O1`; JSON `o1_alpha_probe.json`.
- **O2 (cheap, closes the Tier-S crack §3). — ✅ CLOSED 2026-06-04.** Multi-seed {0,1,7,100} `next_lstm` +
  `next_single` cat at GE+AZ. **next_lstm: the single-seed nominal wins EVAPORATE** (AZ +0.48→+0.25, GE
  +0.51→+0.18; with AL −0.21 / FL +0.14 → tie at all 4 states) → the crack closes: "shown no win." **next_single:
  GE win is REAL and robust** (+1.45 single → **+1.54 ± 0.17** multi-seed) but GE-SPECIFIC (AL −8.11, AZ −0.03)
  → fails the ≥2-band gate. Per this item's rule ("clears ≥0.5pp → real T5.2 candidate") it ENTERS the T5.2
  candidate set as a state-conditional option (re-judged under MTL); does NOT re-open frozen (c) (moving-baseline
  guard — (c) GE-cat stays next_gru 58.12). **Net: the multi-band Tier-S cat negative HOLDS.** `TIER01_RESULTS.md §O2`.
- **O3 (cheap, §4). — ✅ CLOSED 2026-06-04.** Multi-seed FL (c)-cat = **69.96 ± 0.08** → validates the seed42
  frozen 69.97 (agree to 0.01pp; the single-seed yardstick is representative at FL). **The inversion vs MTL
  diag-best 70.26 PERSISTS multi-seed (−0.30pp) — NOT the single-seed artifact hypothesised.** Tiny (~0.35σ) +
  explained: (c) bounds the *deployable* MTL cat (69.96 ≫ 66.73); the diag-best is an oracle epoch + a small FL
  cat transfer (board Δcat≈0). Not a bug (arch=NextHeadGRU; freeze-sanity hard checks pass). Honest caveat
  retained in the (c) footnote. `TIER01_RESULTS.md §O3`.
- **O4 (reporting, §3).** Account for `next_hybrid` (in the cat-screen driver loop + passes the unit gate,
  but absent from the result TSV — ran-and-dropped vs failed is unresolvable from the repo). Note the 2
  `*_hsm` reg heads were never GPU-screened (need a prebuilt `hierarchy_path`; deferred, not bit-rot).
- **O5 (follow-up study, not this paper).** Stand up the dense-supervision / overlapping-windows study:
  dense (per-position / stride-1) targets → rebuild log_T + ceilings + board + §0.1, multi-state +
  multi-seed; **re-confirm the regime finding under correct supervision** (the headline-validation item).
  Carry the limitations note + AL rebuttal into the current paper now (above).

**Containment confirmed:** the cat-head bug was `--task next` only; the MTL board cat is unaffected
(`--cat-head` works on the MTL `is_check2hgi_track` path). No live yardstick used a bugged number.
