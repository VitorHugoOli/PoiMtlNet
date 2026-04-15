# MTL Task Replacement Proposal (Check2HGI-Oriented)

## 1. Decision

Replace the current MTL secondary task:

- From: **Unknown POI Category Classification** (cold-start POI label)
- To: **Trajectory-coupled auxiliary task(s)** that use check-in context

Reason:

- Unknown-POI category is a different problem family from next-checkin trajectory modeling.
- It creates evaluation/interpretation conflicts when combined with Check2HGI features.
- Check2HGI naturally encodes temporal + mobility context, so auxiliary tasks should exploit this.

---

## 2. Recommended Task Candidates

### A) Next Region Prediction (recommended first)

Predict the region (e.g., tract/region_idx) of the next check-in.

Why it fits best:

- Check2HGI already produces region structure and region embeddings.
- Strongly aligned with mobility dynamics.
- Classification setup reuses existing metric/logging patterns.

### B) Next Time-Gap Prediction

Predict `delta_t = t(next) - t(last_history_event)` as:

- regression (`minutes`) or
- binned classification (`<1h`, `1-6h`, `6-24h`, `>24h`)

Why it fits:

- Check2HGI encodes temporal context at check-in level.
- Complements next-category without duplicating it.

### C) Revisit vs Explore

Binary target: whether next POI was already seen in user history window.

Why it fits:

- Pure behavior signal from trajectories.
- Cheap label engineering.

---

## 3. Why Next Region Is a Strong First Choice

1. Minimal conceptual change: still a classification target like next-category.  
2. Strong spatial supervision: close to what HGI/Check2HGI hierarchy models.  
3. Lower risk than POI-ID prediction (smaller label space, less sparsity).  
4. Uses existing Check2HGI outputs:
   - graph mappings in `checkin_graph.pt`
   - `region_embeddings.parquet` for optional future fusion

---

## 4. Scratch Implementation Plan (Next Region)

## 4.1 Data/Label Engineering

Goal: build next-task dataset with label `next_region` instead of (or alongside) `next_category`.

Suggested steps:

1. Build `placeid -> region_idx` mapping:
   - from Check2HGI graph artifact: `IoPaths.CHECK2HGI.get_graph_data_file(state)`
   - keys needed: `placeid_to_idx`, `poi_to_region`
2. In sequence conversion, map target POI to region:
   - for POI-level path: `convert_sequences_to_poi_embeddings(...)`
   - for checkin-level path: `convert_user_checkins_to_sequences(...)`
3. Save output with configurable label column:
   - today: `next_category`
   - proposed: `next_region` (or both columns)

Code touchpoints:

- `src/data/inputs/core.py`
- `src/data/inputs/builders.py`

---

## 4.2 Input Schema Update

Current next parquet schema:

- feature columns `0..N-1`
- `next_category`
- `userid`

Proposed:

- feature columns `0..N-1`
- `next_region` (required)
- `next_category` (optional, for dual-task experiments)
- `userid`

Practical note:

- `save_next_input_dataframe(...)` in `src/data/inputs/core.py` is currently hard-coded for `next_category`; make this label-column configurable.

---

## 4.3 Fold/Loader Support

Extend `load_next_data(...)` in `src/data/folds.py`:

- support `target_col` argument (`next_category` default, `next_region` optional)
- encode region labels to contiguous class IDs (if needed)

Then wire target selection via config flag in train/eval entrypoints.

---

## 4.4 Model/Training Integration

Two rollout options:

1. **Replace task** (simple):
   - keep 2-head MTL
   - category head = POI category task
   - next head target = `next_region`

2. **Add third head** (bigger change):
   - keep next-category + add next-region
   - true 3-task MTL

Start with option 1 for fast iteration.

---

## 4.5 Metrics

For next-region (classification):

- Macro-F1 (primary)
- Accuracy
- Per-class F1 (diagnostic)

Keep existing reporting style for consistency.

---

## 5. Suggested Experiment Sequence

1. Implement `next_region` label generation and loader toggle.
2. Run single-task next-region baseline.
3. Run 2-task MTL: `POI category + next_region`.
4. Compare against current `POI category + next_category` setup.
5. If gains are stable, optionally add time-gap as the next auxiliary task.

---

## 6. Scope Boundaries

- This proposal is about **task fit for MTL with Check2HGI**.
- It does not replace the need for leakage-safe split protocols when reporting inductive claims.
- Unknown-POI category can still be kept as a **separate benchmark track**, not as the default Check2HGI-MTL paired task.
