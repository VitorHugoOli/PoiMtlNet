# CTLE frozen-below-floor — AL diagnosis (BASELINE_M4 Task 1, CRITICAL PATH)

> **Verdict (2026-06-24, M4 Pro / MPS): REAL CTLE frozen-substrate weakness — NOT a pipeline/leak bug.**
> The H100 may proceed with FL CTLE-SC. Present **CTLE-E2E** (~21, fine-tuned) as the headline CTLE number,
> CTLE-SC as the matched-frozen-capacity companion (frozen-SC undersells deep models — the honest reading).

## The question
Recorded frozen CTLE-SC cat @ AL = **17.77** macro-F1, *below* the bigram floor (~19.5). Before the H100 spends FL
compute on CTLE-SC, confirm whether 17.77 is a true CTLE weakness or an artifact (embedding not feeding the head /
degenerate vectors / silent check2hgi reuse / min_seq–fold desync).

## Evidence (all on the committed AL artifacts; no re-run needed — the per-fold trace already exists)

**1 · The frozen CTLE embedding is real and non-degenerate.**
`output/board_baselines/ctle/alabama/s0_f0/embeddings.parquet` — 113,846 check-in rows, 11,848 POIs, 64-d.
- Leak-safe: `CTLE_FOLD.txt` = "encoder pretrained on fold-0 TRAIN users only", vocab=11850; `LEAK_MARKER.txt` =
  "TRAIN-ONLY per fold; row-align + val-user disjoint asserted".
- 0 / 64 zero-std cols, 0 all-zero rows, per-col std 1.17–5.06. Not zeros, not constant.

**2 · The substrate was actually swapped (CTLE ≠ check2hgi).** Row-aligned to `output/check2hgi/alabama/embeddings.parquet`
(placeid match = 1.000, both 113,846 rows):
- mean per-row cosine(CTLE, check2hgi) = **0.0105** (≈ orthogonal); 0% of rows are near-identical (|cos|>0.9).
- magnitude: CTLE mean|·|=2.93 vs check2hgi 0.38 (~7×). Distinct vector spaces — CTLE is not silently reusing check2hgi.

**3 · The head demonstrably learns — the comparand proves the pipeline.** The IDENTICAL head, same rows / same 5
folds, on the check2hgi substrate (`alabama_check2hgi_sc.json`) → cat **55.59** (per-fold 57.1/57.8/54.9/54.3/53.8).
CTLE on the same pipeline (`alabama_ctle.json`) → cat **17.77** (per-fold 18.5/17.9/16.9/15.8/19.7). Only the substrate
changed. ⇒ the head consumes its input embedding (else both would match), and it can learn to 55.6 — so 17.8 is the
substrate, not the pipeline.

## Reading
The three artifact-bug hypotheses are each falsified (real vectors; genuinely swapped substrate; head provably uses
its input and reaches 55.6 on check2hgi). **17.77 is CTLE's true frozen-SC category signal at AL** — Check2HGI's
hierarchical substrate beats it by **+37.8 pp** under matched capacity (the W3 representation-isolation gate).
Frozen-SC < bigram floor is expected for a deep model whose strength is fine-tuning (CTLE-E2E ~21); frozen-SC
undersells it. **Decision: real — H100 proceeds with FL CTLE-SC.**
