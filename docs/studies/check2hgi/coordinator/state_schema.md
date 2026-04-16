# State.json Schema — Check2HGI Study

The state-file schema is **shared** with the fusion study. See `docs/studies/fusion/coordinator/state_schema.md` for the full specification (top-level fields, per-phase fields, per-test fields, lifecycle transitions).

## Check2HGI-specific fields

Beyond the shared schema, `docs/studies/check2hgi/state.json` includes:

- **`sibling_studies`** — array documenting parallel studies (currently `[{name: "fusion", path: "docs/studies/fusion", ...}]`). Cosmetic; the coordinator does not read these but humans do.

## Per-test `observed` fields for this study

| Field | When reported | Unit |
|---|---|---|
| `next_poi_acc1`, `next_poi_acc5`, `next_poi_acc10` | all runs that include the next_poi head | float ∈ [0, 1] |
| `next_poi_mrr`, `next_poi_ndcg5`, `next_poi_ndcg10` | all runs that include next_poi | float ∈ [0, 1] |
| `next_poi_f1` | reported for completeness; noisy at 10K-class cardinality | float ∈ [0, 1] |
| `next_region_acc1`, `next_region_acc5`, `next_region_acc10` | all runs that include next_region head | float ∈ [0, 1] |
| `next_region_mrr` | all next_region runs | float ∈ [0, 1] |
| `next_region_f1` | reported for completeness; noisy at 1K–5K class cardinality | float ∈ [0, 1] |
| `joint_lift` | MTL runs only | float (unitless ratio — mean of per-head Acc@1 / majority) |
| `joint_acc1` | reported alongside joint_lift but NOT the monitor | float ∈ [0, 1] |

**No category metrics** (`cat_f1`, `cat_accuracy`, etc.) — the POI-category task does not live in this study.

## Per-test `verdict` values

Inherited from the shared schema:

- `matches_hypothesis`, `partial_match`, `no_match`, `surprising`, `pending`, `invalid`.

## Per-test `config` fields specific to this study

- `task_set` — the TaskSet preset name, e.g. `check2hgi_next_poi_region`, `check2hgi_next_region` (legacy mixed-scope, not used in the new study).
- `dual_stream` — bool; whether the run used the Option A region-embedding concat input (P3+ only).
- `cross_attention` — bool; whether the run used MTLnetCrossAttn (P4 only).
- `use_class_weights` — bool; `--use-class-weights` CLI flag value.
