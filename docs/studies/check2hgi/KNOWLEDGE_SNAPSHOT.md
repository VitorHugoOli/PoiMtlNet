# Knowledge Snapshot — Check2HGI Study (2026-04-16)

## Study in one paragraph

Standalone study: does adding a **next-region auxiliary task** under MTL improve **next-POI-category prediction** on Check2HGI check-in-level embeddings? No cross-engine comparison. Baselines: internal (single-task Check2HGI is the reference) + simple-baselines floor (majority/random/Markov) + external literature (POI-RGNN for next-category, HMT-GRN/MGCL for the region-auxiliary concept). Runs on Gowalla-AL + FL.

## What we believe (hypotheses, not yet confirmed)

1. **Next-region as auxiliary helps next-category F1** — region-level supervision structures the shared backbone. Maps to CH01.
2. **The region head needs its own architecture** — the next-category transformer may not be optimal for ~1K-class coarser spatial prediction. HMT-GRN uses GRU. Maps to CH05.
3. **Expert-gating MTL architectures may not help here** — both tasks are sequential from the same input (unlike fusion's flat+sequential pair). Simpler sharing (FiLM) might suffice. Maps to CH06.
4. **Region embeddings as explicit input help more on FL than AL** — probe showed FL's check-in embeddings carry weaker implicit region signal. Maps to CH03, CH08.

## Critical open question

**Which MTL configuration works best for {next_category, next_region} on check-in-level embeddings?** The fusion study's champion (DSelectK + Aligned-MTL) was optimised for {POI-category, next-category} on POI-level fused embeddings — different task pair, different embedding granularity. We need a fresh ablation. This drives the P1 → P2 → P3 pipeline.

## What's done

- Check2HGI embeddings for AL + FL + AZ: ✅
- next_region labels for AL + FL + AZ: ✅
- Simple baselines (next_category + next_region): ✅
- CH14 (preprocessing shortcut audit): ✅ confirmed_by_construction — no POI2Vec
- Single-task next-category on AL: **38.67% macro-F1** (5f × 50ep, `next_single` head)
- End-to-end MTL smoke with `check2hgi_next_region` preset: ✅ exit 0
- All infrastructure code: ✅ (860 tests pass)

## What's next

P1: validate region head + head ablation → P2: parameterise all MTL archs, run full ablation → P3: multi-seed headline.
