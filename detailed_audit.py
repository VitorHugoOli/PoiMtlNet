import json
import sys

def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

def compare_values(on_disk, claimed, tolerance=0.1):
    """Check if on_disk value matches claimed within tolerance"""
    if on_disk is None or claimed is None:
        return "MISSING", on_disk
    diff = abs(on_disk - claimed)
    if diff <= tolerance:
        return "MATCHES", on_disk
    else:
        return f"MISMATCH (Δ={diff:.2f})", on_disk

print("="*80)
print("CLAIM 1: HGI cat-STL under gated overlap (Tbl 2 HGI arm)")
print("="*80)
print("\nClaimed values (pp):")
print("  AL: 26.56  |  AZ: 29.50  |  FL: 35.53")
print("  Substrate margin (Check2HGI - HGI): AL +29.31 | AZ +27.63 | FL +39.62")
print("  Check2HGI board (should match §1): AL 55.87 | AZ 57.13 | FL 75.15\n")

claims = {
    "alabama": {"hgi": 26.56, "check2hgi": 55.87, "margin": 29.31},
    "arizona": {"hgi": 29.50, "check2hgi": 57.13, "margin": 27.63},
    "florida": {"hgi": 35.53, "check2hgi": 75.15, "margin": 39.62},
}

for state, expected in claims.items():
    path = f"/Users/vitor/Desktop/mestrado/ingred/docs/results/closing_data/baseline_compare/{state}_hgi_ovl_cat.json"
    data = read_json(path)
    if "error" in data:
        print(f"❌ {state.upper()}: {data['error']}")
        continue
    
    hgi_val = data.get("hgi_cat_macro_f1")
    check2hgi_val = data.get("check2hgi_cat_macro_f1_board")
    margin_val = data.get("substrate_margin_check2hgi_minus_hgi")
    
    hgi_status, hgi_actual = compare_values(hgi_val, expected["hgi"])
    check2hgi_status, check2hgi_actual = compare_values(check2hgi_val, expected["check2hgi"])
    margin_status, margin_actual = compare_values(margin_val, expected["margin"])
    
    print(f"{state.upper()}:")
    print(f"  HGI cat: {hgi_status:20s} | On-disk: {hgi_actual:.4f} | Claimed: {expected['hgi']:.2f}")
    print(f"  Check2HGI board: {check2hgi_status:20s} | On-disk: {check2hgi_actual:.2f} | Claimed: {expected['check2hgi']:.2f}")
    print(f"  Margin: {margin_status:20s} | On-disk: {margin_actual:.4f} | Claimed: {expected['margin']:.2f}")
    print()

print("\n" + "="*80)
print("CLAIM 2: CTLE-E2E (FL real 5f) — final and best-epoch scores")
print("="*80)
print("\nClaimed values:")
print("  AL: cat final 21.14 / best-ep 23.94 (both macro-F1)")
print("  FL: cat final 29.69 / best-ep 33.45; reg Acc@10 61.44\n")

ctle_claims = {
    "alabama": {
        "cat_final": 21.14,
        "cat_best": 23.94,
        "reg_acc10": None,  # Not claimed for AL
    },
    "florida": {
        "cat_final": 29.69,
        "cat_best": 33.45,
        "reg_acc10": 61.44,
    }
}

for state, expected in ctle_claims.items():
    path = f"/Users/vitor/Desktop/mestrado/ingred/docs/results/closing_data/baseline_compare/{state}_ctle_e2e_seed0.json"
    data = read_json(path)
    if "error" in data:
        print(f"❌ {state.upper()}: {data['error']}")
        continue
    
    # Note: values in JSON are fractions (0-1), claimed are percentages (0-100)
    cat_final = data.get("cat_f1_mean") * 100 if data.get("cat_f1_mean") is not None else None
    cat_best = data.get("cat_f1_best_mean") * 100 if data.get("cat_f1_best_mean") is not None else None
    reg_acc10 = data.get("reg_top10_acc_indist_mean") * 100 if data.get("reg_top10_acc_indist_mean") is not None else None
    n_folds = data.get("folds_run")
    
    print(f"{state.upper()} (folds_run={n_folds}):")
    cat_final_status, cat_final_actual = compare_values(cat_final, expected["cat_final"])
    cat_best_status, cat_best_actual = compare_values(cat_best, expected["cat_best"])
    print(f"  CAT final (final): {cat_final_status:20s} | On-disk: {cat_final_actual:.2f} | Claimed: {expected['cat_final']:.2f}")
    print(f"  CAT best-ep: {cat_best_status:20s} | On-disk: {cat_best_actual:.2f} | Claimed: {expected['cat_best']:.2f}")
    
    if expected["reg_acc10"] is not None:
        reg_status, reg_actual = compare_values(reg_acc10, expected["reg_acc10"])
        print(f"  REG Acc@10: {reg_status:20s} | On-disk: {reg_actual:.2f} | Claimed: {expected['reg_acc10']:.2f}")
    print()

print("\n" + "="*80)
print("CLAIM 3: CTLE-SC FL (2/5 PARTIAL) — fold count and mean scores")
print("="*80)
print("\nClaimed: FL = 2/5 folds only, cat ~28, reg ~73")
print("CRITICAL: Confirm exactly which folds are present (indices?)\n")

path = "/Users/vitor/Desktop/mestrado/ingred/docs/results/closing_data/baseline_compare/florida_ctle.json"
data = read_json(path)
if "error" in data:
    print(f"❌ ERROR: {data['error']}")
else:
    n_folds_done = data.get("n_folds_done")
    n_folds_target = data.get("n_folds_target")
    macro_f1_mean = data.get("macro_f1_mean_partial")
    top10_acc_mean = data.get("top10_acc_mean_partial")
    per_fold = data.get("per_fold", [])
    
    print(f"Fold count: {n_folds_done}/{n_folds_target} (CLAIMED: 2/5)")
    print(f"Status: {data.get('status', 'N/A')}")
    print(f"Folds present (indices): {[f['fold'] for f in per_fold]}")
    print(f"Per-fold CAT values: {[f['macro_f1'] for f in per_fold]}")
    print(f"Per-fold REG values: {[f['top10_acc'] for f in per_fold]}")
    print(f"\nMean CAT: {macro_f1_mean:.2f} (claimed ~28)")
    print(f"Mean REG: {top10_acc_mean:.2f} (claimed ~73)")
    
    # Check supplementary reg
    supp_reg = data.get("supplementary_reg_from_parallel_attempt", {})
    if supp_reg:
        print(f"\nSupplementary REG from parallel attempt:")
        print(f"  {supp_reg}")
    
    # Check for per-fold JSON files
    import os
    print(f"\nSearching for per-fold JSON files...")
    for fold_idx in range(5):
        patterns = [
            f"/Users/vitor/Desktop/mestrado/ingred/docs/results/closing_data/baseline_compare/florida_ctle_f{fold_idx}.json",
            f"/Users/vitor/Desktop/mestrado/ingred/docs/results/closing_data/baseline_compare/florida_checkin_1f_50ep_blcmp_check2hgi_ctle_f{fold_idx}.json",
        ]
        for pat in patterns:
            if os.path.exists(pat):
                print(f"  Found: {os.path.basename(pat)}")
    
    # Check match against claimed values (cat ~28, reg ~73)
    cat_match, _ = compare_values(macro_f1_mean, 28.0, tolerance=1.0)  # Allow ±1 pp for "~28"
    reg_match, _ = compare_values(top10_acc_mean, 73.0, tolerance=1.0)  # Allow ±1 pp for "~73"
    print(f"\nCAT match (claimed ~28): {cat_match}")
    print(f"REG match (claimed ~73): {reg_match}")

print("\n" + "="*80)
print("CLAIM 4: TOST region inputs — reg_per_fold length = 5")
print("="*80)
print("\nClaimed: AL, AZ, Istanbul-stride1-s0 should each have reg_per_fold length = 5\n")

tost_files = {
    "alabama": "/Users/vitor/Desktop/mestrado/ingred/docs/results/closing_data/h100/alabama_s0_mtl_fp32_matched_score.json",
    "arizona": "/Users/vitor/Desktop/mestrado/ingred/docs/results/closing_data/h100/arizona_s0_mtl_fp32_matched_score.json",
    "istanbul": "/Users/vitor/Desktop/mestrado/ingred/docs/results/second_dataset/istanbul/istanbul_stride1_s0_mtl_fp32_matched_score.json",
}

for name, path in tost_files.items():
    data = read_json(path)
    if "error" in data:
        print(f"❌ {name.upper()}: {data['error']}")
        continue
    
    reg_per_fold = data.get("reg_per_fold", [])
    n_folds = len(reg_per_fold)
    n_folds_claimed = data.get("n_folds")
    
    status = "✅ MATCHES" if n_folds == 5 else "❌ MISMATCH"
    print(f"{name.upper()}: {status} | Fold count: {n_folds} (n_folds field: {n_folds_claimed}, claimed: 5)")
    if n_folds != 5:
        print(f"  ERROR: Expected 5 folds but found {n_folds}")
    print()

print("\n" + "="*80)
print("CLAIM 5: Istanbul baselines — check for absence of Markov/POI-RGNN/STAN/ReHDM")
print("="*80)
print("\nClaimed ABSENT/not-yet-run: Istanbul Markov, POI-RGNN, STAN, ReHDM\n")

import os
import glob

search_dirs = [
    "/Users/vitor/Desktop/mestrado/ingred/docs/results/P0/simple_baselines/istanbul/",
    "/Users/vitor/Desktop/mestrado/ingred/docs/results/baselines/",
    "/Users/vitor/Desktop/mestrado/ingred/docs/results/P1/",
]

found_files = []
for search_dir in search_dirs:
    if os.path.isdir(search_dir):
        for root, dirs, files in os.walk(search_dir):
            for f in files:
                full_path = os.path.join(root, f)
                if "istanbul" in f.lower():
                    found_files.append(full_path)

print("Istanbul-related files found:")
if found_files:
    for f in sorted(found_files):
        print(f"  {f}")
else:
    print("  (none)")

# Specifically check for baseline names
baseline_names = ["markov", "poi", "stan", "rehdm"]
print("\nSearching for specific baseline keywords in baselines/ directory:")
baselines_dir = "/Users/vitor/Desktop/mestrado/ingred/docs/results/baselines/"
if os.path.isdir(baselines_dir):
    all_files = os.listdir(baselines_dir)
    for baseline_name in baseline_names:
        matches = [f for f in all_files if baseline_name.lower() in f.lower()]
        if matches:
            print(f"  {baseline_name.upper()}: {len(matches)} files found (for ALL regions, not Istanbul-specific)")
            # Check if any have istanbul
            istanbul_matches = [f for f in matches if "istanbul" in f.lower()]
            if istanbul_matches:
                print(f"    Istanbul-specific: {istanbul_matches}")
            else:
                print(f"    Istanbul-specific: NONE")
        else:
            print(f"  {baseline_name.upper()}: not found")

