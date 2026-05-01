# Knowledge Snapshot — Check2HGI Study (2026-04-16, updated)

## Study in one paragraph

Standalone study: under joint MTL on Check2HGI check-in-level embeddings, do the two tasks `{next_category, next_region}` help each other (**bidirectional** — both heads must improve over single-task baselines)? No cross-engine comparison. Baselines: internal single-task (both tasks) + simple-baselines floor (majority/random/Markov-1-region) + external literature (POI-RGNN for next-category, HMT-GRN/MGCL for hierarchical-MTL concept). Runs on Gowalla AL + FL.

## What we believe (hypotheses, updated 2026-04-16)

1. **Bidirectional MTL benefit.** Joint {next_category, next_region} training improves both heads — MTL category F1 > single-task category F1, AND MTL region Acc@10 > single-task region Acc@10. Maps to CH01 + CH02.
2. **Per-task input modality is the right architectural choice.** `category_encoder` gets check-in emb sequences; `next_encoder` gets region emb sequences. The shared backbone bridges them. Supersedes the earlier concat framing. Maps to CH03.
3. **The region head needs GRU-family, not transformer.** Confirmed in P1: `next_gru` wins under both check-in and region-emb inputs; transformer (`next_mtl`) collapses at best_epoch=1 regardless of LR. Maps to CH05.
4. **Input modality dominates head choice.** P1: same head goes from 20% → 53% Acc@10 (Markov-1-region floor 47%) by swapping check-in emb → region emb. Implication: check-in embeddings have weak explicit region signal. Maps to CH03.
5. **Expert-gating MTL architectures may not help here.** Both tasks are sequential from the same input structure. FiLM may suffice. Maps to CH06.

## Critical open questions

1. **Does per-task input modality actually produce bidirectional MTL gains?** P1 answered the single-task side; P3 tests the joint case. If MTL with (check-in, region) streams beats both single-task baselines, thesis validated.
2. **Champion MTL architecture under per-task modality.** CGC/MMoE/DSelectK/PLE are now TaskSet-ready (ported 2026-04-16); P2 runs the arch × optim grid with region-side input = region emb.

## What's done (2026-04-16)

- Check2HGI embeddings for AL + FL + AZ: ✅
- next_region labels for AL + FL + AZ: ✅
- Simple baselines (next_category + next_region, **including region-level Markov-1/2/3**): ✅
- CH14 (preprocessing shortcut audit): ✅ `confirmed_by_construction` — no POI2Vec
- Single-task next-category on AL: **38.67% macro-F1** (5f × 50ep)
- Single-task next-region on AL (region-emb input, `next_gru` default, **5f × 50ep**): **56.94% ± 4.01 Acc@10** (headline single-task number). Scaled variant (hd=384, nl=3, ls=0.1, 1f×50ep) = 54.68% — no net benefit from scaling beyond fold variance.
- P1 head ablation: 5 heads × check-in input + top-2 × region/concat inputs — `next_gru` wins both; transformer out. (Remaining 3 heads × region still pending.)
- **Markov baseline corrected**: old POI-level Markov (21.3% AL / 45.9% FL) was a degenerate 50%-fallback-to-top-k baseline. Corrected region-level Markov-1: 47.01% AL / 65.05% FL.
- **CH04 retired as a gate** — neural head is 1.16× Markov-1-region, not 2×. Demoted to reported comparison.
- P2 pre-req: CGC/MMoE/DSelectK/PLE ported to TaskSet-aware (2026-04-16).
- End-to-end MTL smoke with `check2hgi_next_region` preset: ✅ exit 0.
- All infrastructure code: ✅ (legacy MTLnet regression tests 17/17 pass after P2 port).

## What's next

1. P1 tail: remaining 3 heads × region-emb input + their scaled variants (low priority, near Markov-informed ceiling).
2. P4 prerequisite: extend FoldCreator to produce per-task input tensors (~80 LOC in the data pipeline), plus `--task-a-input-type` / `--task-b-input-type` flags.
3. P2: full arch × optim grid under the per-task-modality config.
4. P3: multi-seed n=15 bidirectional headline.
