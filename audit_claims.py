import json
import sys

def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

def analyze_json(data, name):
    print(f"\n{'='*60}")
    print(f"File: {name}")
    print(f"{'='*60}")
    
    if "error" in data:
        print(f"ERROR: {data['error']}")
        return
    
    # Check structure
    print(f"Top-level keys: {list(data.keys())}")
    
    # Look for fold information
    if "cat_per_fold" in data:
        folds = data["cat_per_fold"]
        print(f"Fold count (cat_per_fold): {len(folds)}")
        print(f"cat_per_fold values: {folds}")
        if "cat_mean" in data:
            print(f"cat_mean (claimed): {data['cat_mean']}")
    
    if "reg_per_fold" in data:
        folds = data["reg_per_fold"]
        print(f"Fold count (reg_per_fold): {len(folds)}")
        print(f"reg_per_fold values: {folds}")
        if "reg_mean" in data:
            print(f"reg_mean (claimed): {data['reg_mean']}")
    
    if "best_ep_cat_per_fold" in data:
        folds = data["best_ep_cat_per_fold"]
        print(f"Fold count (best_ep_cat_per_fold): {len(folds)}")
        print(f"best_ep_cat_per_fold values: {folds}")
        if "best_ep_cat_mean" in data:
            print(f"best_ep_cat_mean: {data['best_ep_cat_mean']}")
    
    if "best_ep_reg_per_fold" in data:
        folds = data["best_ep_reg_per_fold"]
        print(f"Fold count (best_ep_reg_per_fold): {len(folds)}")
        print(f"best_ep_reg_per_fold values: {folds}")
    
    # Check for final scores
    for key in ["cat", "reg", "final_cat", "final_reg"]:
        if key in data:
            print(f"{key}: {data[key]}")
    
    # Check for NaN or collapse indicators
    print(f"\nRaw data: {json.dumps(data, indent=2)}")

# Claim 1: HGI cat-STL under gated overlap
print("\n" + "="*80)
print("CLAIM 1: HGI cat-STL under gated overlap")
print("="*80)

for state in ["alabama", "arizona", "florida"]:
    path = f"/Users/vitor/Desktop/mestrado/ingred/docs/results/closing_data/baseline_compare/{state}_hgi_ovl_cat.json"
    data = read_json(path)
    analyze_json(data, f"{state}_hgi_ovl_cat.json")

# Claim 2: CTLE-E2E (FL real 5f)
print("\n" + "="*80)
print("CLAIM 2: CTLE-E2E")
print("="*80)

for state in ["alabama", "florida"]:
    path = f"/Users/vitor/Desktop/mestrado/ingred/docs/results/closing_data/baseline_compare/{state}_ctle_e2e_seed0.json"
    data = read_json(path)
    analyze_json(data, f"{state}_ctle_e2e_seed0.json")

# Claim 3: CTLE-SC FL (2/5 PARTIAL)
print("\n" + "="*80)
print("CLAIM 3: CTLE-SC FL")
print("="*80)

path = "/Users/vitor/Desktop/mestrado/ingred/docs/results/closing_data/baseline_compare/florida_ctle.json"
data = read_json(path)
analyze_json(data, "florida_ctle.json")

# Claim 4: TOST region inputs
print("\n" + "="*80)
print("CLAIM 4: TOST region inputs")
print("="*80)

for state in ["alabama", "arizona"]:
    path = f"/Users/vitor/Desktop/mestrado/ingred/docs/results/closing_data/h100/{state}_s0_mtl_fp32_matched_score.json"
    data = read_json(path)
    analyze_json(data, f"{state}_s0_mtl_fp32_matched_score.json")

path = "/Users/vitor/Desktop/mestrado/ingred/docs/results/second_dataset/istanbul/istanbul_stride1_s0_mtl_fp32_matched_score.json"
data = read_json(path)
analyze_json(data, "istanbul_stride1_s0_mtl_fp32_matched_score.json")

