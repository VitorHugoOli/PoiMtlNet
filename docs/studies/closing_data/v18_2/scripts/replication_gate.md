# The replication gate — run before any bulk rented work

Numbers from a different GPU are not automatically poolable with local ones. An n=20 cell
pools 4 seeds x 5 folds into a single mean and SD; if hardware varies inside a dataset,
hardware noise becomes indistinguishable from seed noise.

## Procedure

1. On the rented H100, run one cheap cell — **alabama dedicated-cat, seed 0**:

   ```
   bash scripts/run_lane.sh alabama 0 check2hgi_v18 \
        check2hgi_design_k_resln_mae_l0_1 /tmp/gate
   ```

   About 1 min at the projected 4.81x (322 s on the A40).

2. Compare against the A40 value, which is **30.7654** macro-F1
   (`data/alabama_s0_cat.json`, produced 2026-08-09 on the A40 at bs8192 / lr0.0025 /
   tau 0.5 / fp32).

3. Decide:

   | outcome | meaning | action |
   |---|---|---|
   | within the A40 fold SD | kernels agree; hardware is not a variance source | rented cells may be pooled, subject to the per-dataset purity rule |
   | outside fold SD | kernel selection differs materially | rented box takes **whole datasets only**; declare the hardware split in the methodology |

4. Record the observed wall time. That gives the **true** speed multiplier, replacing the
   4.81x bandwidth-ratio projection, before the bulk of the budget is committed.

## Why this cell

Alabama dedicated-cat is the cheapest cell in the study that exercises the full stack
(engine read, windowing, logit-adjusted loss, the scorer). It costs about a minute and
five dollars of judgment.
