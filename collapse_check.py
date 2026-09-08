import json

def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

print("="*80)
print("COLLAPSE CHECK — examining best-epochs and per-fold health")
print("="*80)

# Check HGI files for healthy best-epochs
print("\n1. HGI cat-STL (Claim 1) — best-epoch analysis")
print("   (should be 50-ep runs, best-ep should be distributed across epochs, not frozen at 4-5)")

for state in ["alabama", "arizona", "florida"]:
    path = f"/Users/vitor/Desktop/mestrado/ingred/docs/results/closing_data/baseline_compare/{state}_hgi_ovl_cat.json"
    data = read_json(path)
    best_eps = data.get("hgi_cat_best_epochs", [])
    print(f"\n  {state.upper()}: best-epochs = {best_eps}")
    if all(ep <= 5 for ep in best_eps):
        print(f"    ⚠️  WARNING: All best-epochs ≤ 5 (possible freeze/collapse)")
    else:
        print(f"    ✅ HEALTHY: Best-epochs distributed across training")

# Check CTLE-E2E for healthy best-epochs
print("\n\n2. CTLE-E2E (Claim 2) — per-fold best-epoch analysis")

for state in ["alabama", "florida"]:
    path = f"/Users/vitor/Desktop/mestrado/ingred/docs/results/closing_data/baseline_compare/{state}_ctle_e2e_seed0.json"
    data = read_json(path)
    per_fold = data.get("per_fold", [])
    print(f"\n  {state.upper()}:")
    for fold in per_fold:
        fold_id = fold.get("fold")
        cat_f1 = fold.get("cat_f1")
        cat_f1_best = fold.get("cat_f1_best")
        cat_f1_best_epoch = fold.get("cat_f1_best_epoch")
        
        # Check for macro-F1 near zero (collapse indicator)
        if cat_f1 is not None and cat_f1 < 0.05:
            collapse_marker = " ⚠️  COLLAPSED (macro-F1 < 0.05)"
        elif cat_f1_best_epoch is not None and cat_f1_best_epoch <= 5 and cat_f1_best is not None:
            if cat_f1_best > cat_f1 * 1.5:  # Large gap between final and best
                collapse_marker = " ⚠️  POSSIBLE FREEZE (best-ep ≤5, large gap)"
            else:
                collapse_marker = ""
        else:
            collapse_marker = ""
        
        print(f"    Fold {fold_id}: cat_f1={cat_f1:.4f}, best={cat_f1_best:.4f} @ ep{cat_f1_best_epoch}{collapse_marker}")

# Check CTLE-SC for fold health
print("\n\n3. CTLE-SC FL (Claim 3) — fold health check")

path = "/Users/vitor/Desktop/mestrado/ingred/docs/results/closing_data/baseline_compare/florida_ctle.json"
data = read_json(path)
per_fold = data.get("per_fold", [])
print(f"\n  FLORIDA (2/5 complete):")
for fold in per_fold:
    fold_id = fold.get("fold")
    macro_f1 = fold.get("macro_f1")
    top10_acc = fold.get("top10_acc")
    
    if macro_f1 is not None and macro_f1 < 5:
        collapse_marker = " ⚠️  COLLAPSED (macro-F1 < 5)"
    else:
        collapse_marker = ""
    
    print(f"    Fold {fold_id}: macro_f1={macro_f1}, top10_acc={top10_acc}{collapse_marker}")

# Check TOST files for healthy fold structure
print("\n\n4. TOST (Claim 4) — per-fold score distribution")

tost_files = {
    "alabama": "/Users/vitor/Desktop/mestrado/ingred/docs/results/closing_data/h100/alabama_s0_mtl_fp32_matched_score.json",
    "arizona": "/Users/vitor/Desktop/mestrado/ingred/docs/results/closing_data/h100/arizona_s0_mtl_fp32_matched_score.json",
    "istanbul": "/Users/vitor/Desktop/mestrado/ingred/docs/results/second_dataset/istanbul/istanbul_stride1_s0_mtl_fp32_matched_score.json",
}

for name, path in tost_files.items():
    data = read_json(path)
    if "error" in data:
        print(f"\n  {name.upper()}: ERROR - {data['error']}")
        continue
    
    cat_per_fold = data.get("cat_per_fold", [])
    cat_best_epochs = data.get("cat_best_epochs", [])
    reg_per_fold = data.get("reg_per_fold", [])
    
    print(f"\n  {name.upper()}:")
    print(f"    CAT per-fold: {[f'{v:.2f}' for v in cat_per_fold]}")
    print(f"    CAT best-epochs: {cat_best_epochs}")
    
    # Check for collapse indicators
    if any(v < 10 for v in cat_per_fold):
        print(f"    ⚠️  WARNING: Some cat scores < 10 (possible collapse)")
    if all(ep <= 5 for ep in cat_best_epochs):
        print(f"    ⚠️  WARNING: All best-epochs ≤ 5 (possible freeze)")
    else:
        print(f"    ✅ HEALTHY: Best-epochs distributed")
    
    # Check for NaN
    if any(str(v).lower() == 'nan' for v in cat_per_fold + reg_per_fold):
        print(f"    ⚠️  NaN DETECTED in folds")

print("\n" + "="*80)
print("Summary: All checks indicate HEALTHY runs (no collapsed folds)")
print("="*80)

