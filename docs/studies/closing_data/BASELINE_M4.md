# BASELINE handoff — **M4 Pro** (24 GB, MPS) · self-contained · 2026-06-24

> **You are the M4 Pro. Read ONLY this file, then execute.** Phase = baselines. Jobs: **(1) diagnose the CTLE
> frozen-below-floor — CRITICAL PATH, it gates the H100's FL CTLE-SC run (do FIRST); (2b) re-run the Istanbul
> champion-G at stride-1 to unify the §6.3 box (LOW priority).** (TX HMT-GRN — the old Task 2 — is ✅ DONE, PR #38.)
> Decisions: `../../../articles/[mobiwac]/BASELINE_HANDOFF.md`; results →
> [`RESULTS_BOARD.md`](RESULTS_BOARD.md); cross-machine map → [`BASELINE_DISTRIBUTION.md`](BASELINE_DISTRIBUTION.md).
>
> **Protocol:** seed 0 × 5 folds (n=5), gated stride-1 overlap, leak-free per-fold train-only priors, user-disjoint
> folds. MPS = fp32 (no fp16 confound); HMT-GRN on MPS is validated == CPU within 0.06 pp (PR #38). Keep ≤2–3
> concurrent processes under `scripts/closing_data/ram_watchdog.sh` (4+ → OOM-reboot zone; CA kNN once rebooted the box).

## 0 · Setup (once)
```bash
cd <M4 repo>; git checkout main && git pull
export PYTHONPATH=src
```

## 1 · Task 1 — ✅ DONE (2026-06-24): CTLE frozen-below-floor is a REAL weakness, not a bug — **H100 FL CTLE-SC CLEARED**
> **Verdict:** the AL CTLE-SC cat 17.77 (below the ~19.5 bigram floor) is **CTLE's true frozen-SC category signal**,
> NOT a pipeline/leak artifact. Full evidence: [`../../results/closing_data/baseline_compare/alabama_ctle_DIAGNOSIS.md`](../../results/closing_data/baseline_compare/alabama_ctle_DIAGNOSIS.md).
> Three artifact hypotheses falsified on the committed AL artifacts (no re-run needed — the per-fold trace already exists):
> (1) the frozen CTLE embedding is real/non-degenerate (0/64 zero-std cols, std 1.17–5.06); (2) the substrate was
> actually swapped — row-aligned cosine(CTLE, check2hgi)=0.01, ~7× magnitude apart, NOT silent reuse; (3) the head
> demonstrably learns — IDENTICAL head/rows/folds on check2hgi → **55.59**, on CTLE → **17.77** (comparand
> `alabama_check2hgi_sc.json`). Only the substrate differs. **H100 proceeds with FL CTLE-SC**; the paper presents
> CTLE-E2E (~21, fine-tuned) as the headline with CTLE-SC the matched-frozen-capacity companion (frozen-SC undersells deep models).

<details><summary>original Task 1 spec (for reference)</summary>

The recorded frozen **CTLE-SC cat at AL = 17.77** is *below* the bigram floor (~19.5). Before the H100 spends FL
compute on CTLE-SC, confirm **whether that is a real CTLE weakness or a pipeline/leak-fix artifact** (e.g. the
frozen CTLE embedding not actually feeding the head, or a min_seq/fold desync). Diagnose on **AL** (small, fast):
```bash
# 1. confirm the per-fold leak-clean CTLE embedding exists + is the frozen CTLE vectors (not zeros / not c2hgi):
ls output/board_baselines/ctle/alabama/s0_f0/embeddings.parquet   # + LEAK_MARKER.txt = "TRAIN-ONLY per fold"
#    spot-check the 64 vector cols are non-degenerate and differ from check2hgi's (substrate actually swapped).
# 2. re-run the SC comparison and read the per-fold trace:
python scripts/closing_data/mac_baseline_compare.py --state alabama --baseline ctle --cells-root output --folds 5 --heads cat
#    -> docs/results/closing_data/baseline_compare/alabama_ctle.json  (expect cat ~17.8 if real)
# 3. cross-check: does the SAME head on check2hgi give ~55.6 on the same rows/folds? (comparand) — if yes, the
#    pipeline is fine and 17.8 is CTLE's true frozen weakness; if the head can't learn on ANY substrate, it's a bug.
```
**Report the verdict** (real CTLE weakness vs pipeline bug) to the orchestrator/H100 **before** the H100 runs FL
CTLE-SC. If it's a bug, identify it (the H100 fixes + reruns); if it's real, the H100 proceeds — and the paper
presents **CTLE-E2E** (~21, the fine-tuned strength) as the headline CTLE number with CTLE-SC as the
matched-frozen-capacity companion (frozen-SC undersells deep models; that's the honest reading).

## 2 · Task 2 — ✅ DONE: TX HMT-GRN (PR #38, reg 53.85 / cat 25.81)
~~Finish TX HMT-GRN.~~ **Complete** — all 5 Gowalla states + Istanbul now have HMT-GRN (AL 57.1 / AZ 43.7 / FL 63.7
/ CA 49.6 / **TX 53.9** / Istanbul 60.4), all well below our MTL ~65–69. The region-native safety-net row is closed.

## 2b · Task 2 (NEW) — Istanbul champion-G @ stride-1 (make the §6.3 box internally consistent)
> **⏸ DEFERRED (2026-06-24, user decision) — LOW priority; the 4-seed set-a champion is already a valid datapoint.**
> Two prerequisites the next runner must handle (investigated this session — don't re-discover):
> 1. **The stride-1 champion base is NOT on disk.** `output/check2hgi/istanbul/input/` currently holds the **set-a**
>    base (58,297 rows, `stride=null`, `min_seq=5`; provenance utc 2026-06-24T10:18) — the baselines built their
>    stride-1 windowing transiently and did not persist a 271,666-row base here. Build it with
>    `generate_next_input_from_checkins('istanbul', check2hgi, stride=1, min_sequence_length=10)` +
>    `build_next_region_for('istanbul', check2hgi)` (the same calls `mac_baseline_compare.build_inputs` uses) —
>    this **overwrites the set-a base in place** (recoverable: the set-a numbers are recorded in
>    `PHASE_V_ISTANBUL_S0.md` + `RESULTS_BOARD §1`, and the set-a base regenerates from `generate_next_region_input`).
> 2. **Rebuild the per-fold log_T AFTER the base** (the on-disk `region_transition_log_seed0_fold{1..5}.pt` are
>    STALE — older than `next_region.parquet`); verify mtime > `next_region.parquet` (standing trap).
> 3. **The exact validated champion command is not on this box** — the set-a run was on the H100 and its rundir
>    manifest (`results/check2hgi/istanbul/mtlnet_..._190800_105170`) is gitignored / absent locally. The recipe
>    below has elided flags (`...`); reconstructing risks the "wrong MTL flags drop a head 10–30pp" trap. Recover
>    the exact command (manifest on the SSD, or paste it) before the multi-hour run; smoke 1 fold × 2 ep first
>    (sanity: cat ≈ 60 / reg ≈ 66–70 — the SC-stride-1 ceilings cat 54.52 / reg 66.16 are the matched gap-to-ceiling targets).

**Why:** the Istanbul *baselines* (CTLE-SC, Check2HGI-SC ceiling, HMT-GRN) are all at **stride-1** and live here on
the M4 (PR #38 built the 271,666-row stride-1 Phase-V base). The Istanbul *champion-G* is still at **set-a** (PR #33
Phase V, 4 seeds, cat +8.06 / reg −0.58). So champion-vs-baseline on Istanbul currently mixes windowings. Re-run the
champion at stride-1 to unify the box. **This is the M4's job, NOT the A40's** — the champion is from-scratch-trained
(device-sensitive, the HMT lesson), so it must be on the SAME device as the Istanbul baselines (all here) for a clean
Δ. Istanbul-scale champion-G is MPS-proven (the original NYC/Istanbul dry-run ran it on MPS).
```bash
# the stride-1 Phase-V base already exists (PR #38). Build the per-fold seeded log_T for it if absent:
python scripts/compute_region_transition.py --state istanbul --engine check2hgi --per-fold --seed 0 --n-splits 5
# champion-G MTL on the stride-1 Phase-V Istanbul base (engine check2hgi = Phase-V mahalle substrate, NOT dk_ovl):
#   the champion recipe at Istanbul = the second_dataset dry-run invocation + the frozen v16 recipe, --state istanbul
#   --device mps. Mirror scripts/second_dataset/phase_v_substrate.py's champion call; build at --stride 1.
#  -> score vs the Istanbul STL cat/reg ceilings (Check2HGI-SC, already on the M4); report gap-to-ceiling / lift only.
```
**Priority: LOW** — Istanbul is the provisional §6.3 external-validity box (gap-to-ceiling/lift, not absolute), and
the 4-seed set-a champion is already a valid replication datapoint; this run only buys a windowing-consistent table.
Do it after Task 1 (the critical-path CTLE diagnosis). Keep the 4-seed set-a result too — note both windowings.
Expected: champion cat beats its STL ceiling, reg matches/near (the non-US replication), and champion > CTLE-SC
(+~26 cat) and > HMT-GRN (60.4 reg) at stride-1.

## 3 · (Optional) CSLSL @ AL/AZ — only if the A40 stays busy
`python scripts/baselines/b4_cascade.py --state {alabama,arizona} --seed 0 --folds 5 --epochs 50 --device mps` —
MPS from-scratch training is trustworthy here (HMT validated MPS==CPU); **label the result `[M4/MPS]`** and
cross-check one fold vs CPU. Otherwise leave CSLSL to the A40 (CUDA, cleaner vs the champion).

## 4 · Traps + outputs
- ≤2–3 concurrent; `ram_watchdog.sh` kills the newest proc < 4 GB free. Do NOT run a CA/TX embedding-quality kNN here (that's what OOM-rebooted the box).
- All paths train-only per fold (vocab/prior/OOD/fold-split); val users disjoint from train (asserted) — don't "optimise" away.
- Outputs: CTLE diagnosis JSON (`baseline_compare/alabama_ctle.json`, NOT gitignored) + a one-line verdict; TX HMT numbers into `MACS_BOARD_RESULTS.md`. **n=5 provisional.**
- Do **NOT** run the FL block (H100) or any dropped baseline (CTLE-SC ladder / region-SC).
