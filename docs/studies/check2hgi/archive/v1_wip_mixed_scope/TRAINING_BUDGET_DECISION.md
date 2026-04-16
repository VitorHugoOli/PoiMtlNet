# Training-budget decision (advisor #1 + reviewer #3)

**Status:** OPEN — needs your call before CH01 can be validated.

## The problem

Advisor flagged check2HGI's 500-epoch training as possibly undertrained:

| Epoch | Best loss (Alabama) |
|---|---|
| 1 | 1.40 |
| 49 | 0.89 |
| 99 | 0.73 |
| 199 | 0.60 |
| 299 | 0.53 |
| 399 | 0.49 |
| 493 | 0.47 |

0.60 → 0.47 over 300 epochs is still decreasing — not plateaued.

Critical reviewer elevated this to a **falsifiability** concern: if CH01 (check2HGI > HGI) refutes in P2, we cannot distinguish "contextual embeddings don't help" from "contextual embeddings were undertrained." CH01 becomes non-falsifiable without budget matching.

## Three options

### Option 1 — Match FLOPs / matched budget (most rigorous)

Extend check2HGI to plateau (~1000 epochs, maybe more) AND retrain HGI at the same wall-clock / FLOPs / optimizer-steps budget. Then CH01 measures representation quality, not training investment.

- **Cost:** ~40 min check2HGI extension (FL + AL) + potentially re-running HGI which depends on how its current checkpoint was obtained.
- **Benefit:** CH01 becomes cleanly falsifiable. Paper claim defensible.
- **Risk:** if HGI's existing checkpoint is from a long prior run, matching "FLOPs" may be ambiguous — we'd need to find its original training log.

### Option 2 — Extend check2HGI only; don't touch HGI (asymmetric)

Run check2HGI to 1000 epochs on FL + AL. HGI stays at whatever it was. This makes check2HGI the "harder opponent" — if it still fails CH01, the refutation is stronger; if it passes, reviewers may ask why HGI wasn't also retrained.

- **Cost:** ~40 min total.
- **Benefit:** partial — resolves the "check2HGI undertrained" concern but introduces the opposite asymmetry.
- **Risk:** reviewer still flags the comparison as unfair.

### Option 3 — Document 500 epochs as scope caveat, run CH01 anyway

Ship 500-epoch check2HGI. Add a paper limitation: "Check2HGI was trained for 500 epochs (loss still decreasing); results may be a lower bound on attainable performance under this encoder." If CH01 fails, the limitation explains it; if CH01 succeeds, it's a conservative claim.

- **Cost:** 0.
- **Benefit:** fastest path to P2.
- **Risk:** reviewer #3 says this makes CH01 **non-falsifiable**. For BRACIS specifically, reviewers are forgiving of scope caveats but hostile to confounded comparisons. This route is weaker than Options 1–2.

## My recommendation

**Option 1** — but gated on whether we know HGI's original training schedule. If HGI was also trained to 500 epochs on identical optimizer/batch settings, Options 1 and 3 collapse to the same thing. If HGI was trained longer, Option 1 requires re-running HGI.

### Action before P2

1. Inspect `pipelines/embedding/hgi.pipe.py` and any surviving HGI training log to establish its epoch count and optimizer.
2. If HGI ran 500 epochs with matched optimizer → CH01 comparison is already at matched budget; ship Option 3.
3. If HGI ran longer → extend check2HGI to match, then ship Option 1.

**Wall-clock impact:** at the observed 1.5–2 s/epoch rate, an extra 500 epochs per state is ~15 min. Both states ~30 min. Negligible vs the rest of the P2–P7 plan.

## Related: class-weighted CE for Florida (advisor #2)

`--use-class-weights` CLI flag now added (scripts/train.py, task 30). When set, the CrossEntropyLoss on each head receives balanced class weights. Recommended default for FL runs given the 22.5% majority-class region.

Decision to make: should we enable class weights by default on the check2HGI track? The legacy track already defaults `use_class_weights=True` on MTL configs (see `ExperimentConfig.default_mtl`), but the factory for the check2HGI track inherits from that — so we inherit the default. Current behaviour: **class weights ON by default** on both tracks.

If you want FL to run *without* class weights (to see the unmitigated imbalance behaviour), add `--no-class-weights` explicitly.
