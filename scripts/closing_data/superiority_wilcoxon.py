"""Pre-registered superiority tests (STATISTICAL_PROTOCOL.md §2, §5.2): paired
one-sided Wilcoxon signed-rank + Holm-Bonferroni for the headline family-(A) cells.

- Category (all 6 states): superiority MTL champion-G cat > STL dedicated cat ceiling.
- Region (FL/CA/TX, the "beats"): superiority MTL reg > STL dedicated reg ceiling.
  (The small-state region "matches" AL/AZ/Istanbul are NON-inferiority -> region_match_tost.py, not here.)

Paired per fold on the SAME frozen overlap folds (seed 0 x 5). Reads the committed
matched-score JSONs + the P1 region-head STL ceilings. n=5 (seed 0) provisional;
{1,7,100} -> n=20 is post-deadline.
"""
import json
from scipy import stats

# ---- sources (board §3 file map) ------------------------------------------
MTL = {  # champion-G matched-score JSONs (cat_per_fold + reg_per_fold)
    'AL': 'docs/results/closing_data/h100/alabama_s0_mtl_fp32_matched_score.json',
    'AZ': 'docs/results/closing_data/h100/arizona_s0_mtl_fp32_matched_score.json',
    'FL': 'docs/results/closing_data/h100/florida_s0_mtl_fp32_5f_matched_score.json',
    'CA': 'docs/results/closing_data/h100/california_s0_mtl/california_s0_mtl_final_score.json',
    'TX': 'docs/results/closing_data/a40/tx_ba2_fp32_s0.json',
    'Istanbul': 'docs/results/second_dataset/istanbul/istanbul_stride1_s0_mtl_fp32_matched_score.json',
}
STL_CAT = {
    'AL': 'docs/results/closing_data/h100/alabama_s0_stl_cat_ceiling.json',
    'AZ': 'docs/results/closing_data/h100/arizona_s0_stl_cat_ceiling.json',
    'FL': 'docs/results/closing_data/h100/florida_s0_stl_cat_ceiling.json',
    'CA': 'docs/results/closing_data/h100/california_s0_stl_cat_ceiling.json',
    'TX': 'docs/results/closing_data/a40/tx_stl_cat_ceiling_s0.json',
    'Istanbul': 'docs/results/second_dataset/istanbul/istanbul_stride1_s0_stl_cat_ceiling.json',
}
STL_REG = {  # next_stan_flow per-fold top10 (the dedicated reg ceiling)
    'FL': 'docs/results/P1/region_head_florida_region_5f_50ep_florida_ovl_stl_reg_s0.json',
    'CA': 'docs/results/P1/region_head_california_region_5f_50ep_ca_ovl_stl_reg_s0.json',
    'TX': 'docs/results/P1/region_head_texas_region_5f_50ep_tx_ovl_stl_reg_s0.json',
}


def mtl_cat(p):
    d = json.load(open(p))
    return d['mtl_cat_per_fold'] if 'mtl_cat_per_fold' in d else d['cat_per_fold']


def mtl_reg(p):
    d = json.load(open(p))
    return d['mtl_reg_per_fold'] if 'mtl_reg_per_fold' in d else d['reg_per_fold']


def stl_cat(p):
    d = json.load(open(p))
    return d['cat_per_fold'] if 'cat_per_fold' in d else d['cat_macro_f1_per_fold']


def stl_reg(p):
    d = json.load(open(p))
    for h, hv in d.get('heads', {}).items():
        if isinstance(hv, dict) and 'per_fold' in hv:
            return [x['top10_acc'] * 100 for x in hv['per_fold']]
    raise KeyError(p)


def superiority(mtl, stl, label):
    d = [m - s for m, s in zip(mtl, stl)]
    n = len(d)
    pos = sum(1 for x in d if x > 0)
    mean_d = sum(d) / n
    # one-sided paired Wilcoxon, alternative: MTL > STL (greater); exact for small n
    w = stats.wilcoxon(mtl, stl, alternative='greater', zero_method='wilcox', mode='exact')
    return dict(label=label, n=n, mean_d=round(mean_d, 3), pos=f"{pos}/{n}",
                d=[round(x, 3) for x in d], p=w.pvalue)


def holm(items, alpha=0.05):
    """items: list of (key, p). Returns dict key-> (p_adj, reject) under Holm-Bonferroni."""
    m = len(items)
    order = sorted(items, key=lambda kv: kv[1])
    out, running = {}, 0.0
    for i, (k, p) in enumerate(order):
        adj = min((m - i) * p, 1.0)
        running = max(running, adj)          # enforce monotonicity
        out[k] = round(running, 5)
    # stepwise reject: reject p(i) while p(i) <= alpha/(m-i)
    rej = {}
    cont = True
    for i, (k, p) in enumerate(order):
        thr = alpha / (m - i)
        if cont and p <= thr:
            rej[k] = True
        else:
            cont = False
            rej[k] = False
    return {k: (out[k], rej[k]) for k, _ in items}


# ---------------------------------------------------------------------------
print("=== CATEGORY superiority (MTL champ-G > STL dedicated ceiling), n=5 ===")
cat_rows = {s: superiority(mtl_cat(MTL[s]), stl_cat(STL_CAT[s]), s) for s in MTL}
cat_holm = holm([(s, cat_rows[s]['p']) for s in cat_rows])
for s, r in cat_rows.items():
    padj, rej = cat_holm[s]
    print(f"  {s:9s} Δ={r['mean_d']:+6.2f}  folds+={r['pos']}  p1={r['p']:.5f}  "
          f"Holm_adj={padj:.4f}  reject@.05={rej}")

print("\n=== REGION superiority (MTL > STL ceiling) — FL/CA/TX (the beats), n=5 ===")
reg_rows = {s: superiority(mtl_reg(MTL[s]), stl_reg(STL_REG[s]), s) for s in STL_REG}
reg_holm = holm([(s, reg_rows[s]['p']) for s in reg_rows])
for s, r in reg_rows.items():
    padj, rej = reg_holm[s]
    print(f"  {s:9s} Δ={r['mean_d']:+6.2f}  folds+={r['pos']}  p1={r['p']:.5f}  "
          f"Holm_adj={padj:.4f}  reject@.05={rej}")

# ---- pooled across-state aggregate (descriptive: is the direction unambiguous?) ----
print("\n=== POOLED aggregate (descriptive, NOT the per-state family test) ===")
cat_all_m = [m for s in MTL for m in mtl_cat(MTL[s])]
cat_all_s = [m for s in MTL for m in stl_cat(STL_CAT[s])]
rc = superiority(cat_all_m, cat_all_s, 'cat-pooled')
print(f"  category pooled: n={rc['n']} fold-pairs, folds+={rc['pos']}, "
      f"mean Δ={rc['mean_d']:+.2f}, one-sided Wilcoxon p={rc['p']:.2e}")
reg_all_m = [m for s in STL_REG for m in mtl_reg(MTL[s])]
reg_all_s = [m for s in STL_REG for m in stl_reg(STL_REG[s])]
rr = superiority(reg_all_m, reg_all_s, 'reg-pooled')
print(f"  region(FL/CA/TX) pooled: n={rr['n']} fold-pairs, folds+={rr['pos']}, "
      f"mean Δ={rr['mean_d']:+.2f}, one-sided Wilcoxon p={rr['p']:.2e}")

print("\n[note] n=5 one-sided Wilcoxon floor = 1/2^5 = 0.03125 (5/5 folds positive). "
      "Holm-FWER across the family needs n=20 ({1,7,100} top-up, post-deadline) to clear 0.05.")
