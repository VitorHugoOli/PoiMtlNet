# Data Splits & Fold Protocol

The cross-validation split protocol for single-task and multi-task experiments. New agents must understand this before touching `src/data/folds.py` or running any experiment that compares MTL to STL.

> **Source-of-truth implementation**: [`src/data/folds.py::_create_single_task_folds`](../../src/data/folds.py) and [`src/data/folds.py::_create_mtl_folds`](../../src/data/folds.py).
> This doc describes the *protocol* the implementation realises; if there's ever a discrepancy, the code wins.

## Hard invariant

**A user's check-ins never appear in both training and validation of the same fold.** This is enforced by `StratifiedGroupKFold(groups=userid)`. The MTL backbone seeing a user's check-ins via one task in training would otherwise leak that user's behavioural patterns into the other task's validation.

## Configuration

```python
from sklearn.model_selection import StratifiedGroupKFold
StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
```

- **5 folds** (project default)
- **Random seed = 42** (fold-id seed; reproducible across re-runs)
- **Stratification**: by primary class (category for the cat task; region for the next task)
- **Group**: by `userid` (the user-isolation invariant)

## Single-task vs multi-task

### STL (single-task)

`_create_single_task_folds` runs `StratifiedGroupKFold` on the task-specific dataset:
- For cat: stratify by category; group by userid.
- For next/region: stratify by region; group by userid.

Outputs per-task fold tensors at `output/<engine>/<state>/folds/fold_indices_<task>.pt`.

### MTL (multi-task)

`_create_mtl_folds` is the harder case. The cat dataset and the next dataset are different (different rows, different sizes), but the MTL backbone trains on both jointly. Naive zipping of independent fold splits would let user U appear in the cat-train of fold k and the next-val of fold k → leak.

**Solution: MTL fold pairing on a shared user partition.**
1. Compute a single `StratifiedGroupKFold` user partition (5 folds × users).
2. For each fold k, split the cat dataset by "is this row's user in fold-k validation users?" and likewise the next dataset.
3. Both tasks get fold-k cat-train + fold-k next-train (training set is union of users in folds {1..5}\{k}).
4. The fold-k cat-val and fold-k next-val share the same val-user set.

Outputs at `output/<engine>/<state>/folds/fold_indices_mtl.pt`.

## Frozen-fold registry

`docs/archive/fusion-study/results/P0/folds/frozen.json` is a per-engine fold-pair registry that records which fold tensors are "frozen" (immutable) for paper-grade work. For new states / engines, freeze-folds is run via `scripts/study/freeze_folds.py` and the entry is appended.

## Per-fold log_T (region task only)

The region task uses a transition prior `α·log T` where `T` is the fold-specific 1-region transition matrix (a Markov-1 prior over region-to-region transitions, computed on the training partition only).

**Critical leak-prevention:** the transition matrix MUST be computed on the per-fold training partition only — using the full-dataset transition matrix as the prior would let validation-fold transition edges leak into training. The fix landed 2026-04-29 (see `findings/F50_T4_C4_LEAK_DIAGNOSIS.md` and `findings/F50_DELTA_M_FINDINGS_LEAKFREE.md`).

Per-fold transition matrices live at `output/check2hgi/<state>/region_transition_log_seed42_fold{1..5}.pt` and are built by `scripts/build_phase3_per_fold_transitions.sh`.

The legacy single-file `region_transition_log.pt` (full-dataset prior) is **leak-inflated by 13–27 pp on FL-style states**. All paper-grade results from 2026-05-01 onward use per-fold logs.

## Multi-seed pooling

For paper-grade Wilcoxon (n=20), the protocol is:

1. ~~**Same fold partition across seeds** — fold-id seed=42 always; only the model-init seed varies.~~
   🔴 **FALSO. Corrigido 2026-08-27 — ver o bloco abaixo.** O correto: **cada semente sorteia uma
   partição diferente**; a mesma semente governa a partição **e** a inicialização.
2. **Seeds {0, 1, 7, 100}** for the standard 4-seed pool. Combined with 5 folds → n=20 paired (seed, fold) tuples.
3. FL multi-seed extends to {42, 0, 1, 7, 100} → n=25 (the FL Δm-MRR Pareto-positive cell).

~~Multi-seed runs reuse the same fold tensors (no per-seed fold regeneration).~~ 🔴 **FALSO, mesma
correção.** Pairing is on `(fold_idx, seed_idx)`.

> ## 🔴 CORREÇÃO 2026-08-27 — a partição NÃO é congelada entre sementes
>
> **`--seed` alimenta o `random_state` do divisor. Cada semente produz uma partição de usuários
> diferente**, e a mesma semente também semeia a inicialização — as duas estão **confundidas**, o que
> é um limite declarado, não uma escolha.
>
> **Sete linhas de evidência, duas empíricas:**
>
> | | |
> |---|---|
> | código | `src/data/folds.py:1159, 1247, 1453` — `StratifiedGroupKFold(..., random_state=self.seed)`. O `1453` é o caminho conjunto entregue |
> | código | `scripts/compute_region_transition.py:304` — mesmo contrato, arquivo diferente |
> | cadeia | `scripts/train.py:573` (`--seed`) → `:1375` (`replace(config, seed=...)`) → `:1874` (`FoldCreator(seed=config.seed)`) → `folds.py:1071` |
> | drivers | `docs/studies/closing_data/v18/run_wave.sh:132,175,193` passam `--seed`; `--folds-path`, `--no-folds-cache` e `--per-fold-seed` aparecem **0 vezes** |
> | logs | todo log v18 registra **miss** do cache de folds; as contagens de usuários por fold **diferem entre sementes** |
> | 🟢 empírico | rodar o divisor no `next.parquet` entregue de Alabama: **nenhum par de sementes partilha o fold-0**; Jaccard 0,096–0,134 |
> | 🟢 empírico | os `region_transition_log_seed{S}_fold{N}.pt` em disco têm **md5 distinto por semente** para o mesmo índice de fold |
>
> ⚠ **E a reconstrução fecha:** re-dividir o `next.parquet` entregue **só a partir da semente**
> reproduz os tamanhos de fold dos quatro logs entregues, **fold a fold, nos vinte números** — e a
> variante com `userid` cru **não** casa. É falsificável e passou.
>
> **O volume entregue já estava certo** (`5_mobiwac/05_setup.tex:117`: *"each seed produces a
> different division of the users"*). **Era este documento que estava errado.**
>
> ⚠ **Por que sobreviveu:** a varredura de correção de 2026-08-04 escopou-se a
> `articles/dissertacao/**` e **nunca olhou para `docs/context/`** — enquanto 19 arquivos citam este
> como fonte. **Busca com escopo errado, resultado vazio, vazio lido como ausência.**

## What a new agent must verify before claiming "MTL beats STL"

1. **Same fold partition** for the MTL and STL runs being compared. Check `manifest.json` in each run dir.
2. **Same epoch-extraction protocol** (F51 canonical: per-fold max for ep ≥ 5). See `METRICS.md`.
3. **Per-fold leak-free log_T** for any region-task comparison.
4. **n_pairs ≥ 20** for headline significance claims (n=5 is screening only).

If any of those is violated, the comparison is at best illustrative and at worst leak-confounded.

## See also

- `METRICS.md` — paired Wilcoxon details + sample-size guidance
- `TASKS.md` — task definitions
- `DATASETS.md` — per-state dataset sizes (regions, users, check-ins)
- `../issues/check2hgi/FOLD_LEAKAGE_AUDIT.md` — historical record of the leakage diagnosis (FIXED + VERIFIED)
- `../archive/check2hgi-generic-issues-pre-promotion/SPLIT_PROTOCOL.md` — the original 2026-04 split-protocol spec (now implemented + canonical in code)
