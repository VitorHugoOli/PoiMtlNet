# Check2HGI Follow-ups Tracker

**Created:** 2026-04-23. **Updated:** 2026-04-24 — post-F27 cat-head swap + F21c matched-head finding.

**Champion (post-F27):** `mtlnet_crossattn + static_weight(category_weight=0.75) + next_gru (task_a) + next_getnext_hard (task_b) d=256, 8h`. See [`NORTH_STAR.md`](NORTH_STAR.md) and [`PAPER_STRUCTURE.md`](PAPER_STRUCTURE.md). FL scale-dependence flag on the task_a head is open pending F33 (Colab FL 5f).

**Convention.** One row per open work item. Status ∈ `{pending, in_progress, blocked, done, gated, retired}`. Priority P1 (blocker) → P4 (low). Owner defaults to `m4_pro` (this machine); `m2_pro`, `4050`, `any` also valid. Cost is wall-clock estimate at the stated hardware.

**Read before working:**
1. [`NORTH_STAR.md`](NORTH_STAR.md) — champion config + re-evaluation triggers.
2. [`PAPER_STRUCTURE.md`](PAPER_STRUCTURE.md) — paper scope + baselines + tables.
3. [`review/2026-04-23_critical_review.md`](review/2026-04-23_critical_review.md) — analytical state.

---

## 1 · Ready-now (P1/P2 — paper-blocking)

| # | Pri | Item | Obj | Owner | Cost | Acceptance criterion | Why |
|---|:-:|---|:-:|:-:|:-:|---|---|
| **F21c** | **P1** | STL GETNext-hard 5f per state (strict matched-head reg baseline) ⭐ | 2b | m4_pro | done 2026-04-24 | **AL: 68.37 ± 2.66 Acc@10 · AZ: 66.74 ± 2.11 Acc@10.** STL-with-graph-prior **dominates MTL-B3 by 12-14 pp on reg** at both ablation states. Paper-reshaping finding in `research/F21C_FINDINGS.md`. FL 5f not launched — awaiting headline-path decision. |
| **F27** | **P1** | Cat-head ablation + 5-fold validation on AZ | — | m4_pro | **done 2026-04-24** | 7-config 1f screen (research/F27_CATHEAD_FINDINGS.md) → `next_gru` winner → AZ 5f × 50ep paired Wilcoxon p=0.0312 on cat F1 / cat Acc@1 / reg MRR (5/5 folds +). B3's cat head **swapped `next_mtl` → `next_gru`**. AZ cat F1 0.4362 → 0.4581. |
| **F31** | **P1** | **B3 (post-F27) AL 5f × 50ep validation** | — | m4_pro | ~15 min MPS | Re-run AL with `--cat-head next_gru`. Pass: cat F1 ≥ 0.3928 (old B3) AND reg Acc@10 ≥ 0.5633 (old B3), both within σ. | Verify the F27 cat-head choice generalises off AZ before adopting it universally. |
| **F32** | **P2** | **B3 (post-F27) FL 1f × 50ep re-check** | — | m4_pro | **done 2026-04-24 04:05** | Flipped sign vs AL/AZ: cat F1 = 0.6572 (−0.93 pp vs pre-F27 mean), cat Acc@1 = 0.6860 (−0.43 pp), reg Acc@10 = 0.6526 (−0.93 pp). All within n=1 noise but direction is opposite. **Scale-dependence flag raised** — see `research/F27_CATHEAD_FINDINGS.md §Validation outcomes`. |
| **F33** | **P1** | **FL 5f × 50ep B3+next_gru (decisive F27 FL resolution)** ⭐ | 2b | **colab** | ~6 h Colab T4 | Run MTL-B3 on FL with `--cat-head next_gru` at 5 folds. Compare to the two pre-F27 FL n=1 points (cat F1 0.6623 / 0.6706). Pass: cat F1 5f-mean within σ of 0.6623-to-0.6706 envelope → Path A (universal next_gru). Fail: cat F1 below envelope → Path B (scale-dependent cat head). | Scale-dependence was flagged by F32. FL is the paper's headline state — 5-fold σ is the binding criterion for committing the B3 champion universally. Colab run (not M4 Pro) to avoid the OOM/kill history at FL scale. |
| **F34** | **P1** | **CA upstream pipeline + CA 1f × 50ep B3+next_gru** | headline | **colab** | ~6–12 h Colab T4 (pipeline + 1f train) | Steps: (a) Check2HGI embedding training on CA, (b) `compute_region_transition.py --state california`, (c) `create_inputs_check2hgi.pipe.py --state california`, (d) B3+gru 1f×50ep training. Land `results/headline/california/fl_1f50ep_b3_gru.json`. | First CA data point. Also tests whether F27's cat-head choice is scale-stable beyond FL (CA ~500K check-ins). |
| **F35** | **P1** | **TX upstream pipeline + TX 1f × 50ep B3+next_gru** | headline | **colab** | ~6–12 h Colab T4 (pipeline + 1f train) | Same as F34 but TX. | First TX data point. |
| **F3** | **P2** | AZ HGI STL cat (5f × 50ep, fair folds) | 1 | m4_pro | ~3 h MPS | HGI region embeddings through cat pipeline on AZ. Δ = F1(Check2HGI) − F1(HGI) with σ. Land `results/P1_5b/next_category_arizona_hgi_5f_50ep_fair.json`. | Extends CH16 from n=1 to n=2 states — cheapest Objective-1 fix. |
| **F22** | — | (retired — merged into F34) CA upstream pipeline | — | — | — | CA upstream moved to F34 (Colab). | |
| **F23** | — | (retired — merged into F35) TX upstream pipeline | — | — | — | TX upstream moved to F35 (Colab). | |
| **F24** | **P2** | CA 5f headline (after F34 1f confirms config) | headline | colab/any | ~20–25 h | Full baselines + STL + MTL at 5f × 50ep seed 42. Gated on F34. | Part of the paper's primary table; launch only after F34 1f shows the config works on CA data. |
| **F25** | **P2** | TX 5f headline (after F35 1f confirms config) | headline | colab/any | ~20–25 h | Same as F24 but TX. Gated on F35. | |
| **F4** | **P1** | FL MTL-B3 5-fold (clean re-run) | headline | m4_pro / any | ~6–8 h MPS | Replace F17 partial result with a clean n=5 FL run of B3. With F20 per-fold persistence, partial progress survives crashes. | FL headline needs real σ. F17 attempt was user-killed; retry. |
| **F9** | **P3** | FL HGI STL cat 5f | 1 | any | ~5–6 h MPS | Δ vs Check2HGI STL on cat F1 with σ on FL. | Completes CH16 across headline states. Run after F3 confirms AZ pattern. |
| **F37** | **P1** | **STL `next_gru` cat 5f per state (matched-head for post-F27 B3)** | 2b | **4050** | ~30min (AL) + 45min (AZ) + 2h (FL-1f) = $\sim$3h | Run `scripts/run_stl_next_gru_cat.sh`. Produces `results/check2hgi/<state>/next_lr1.0e-04_bs2048_ep50_*/summary/full_summary.json`; archive to `results/P1_5b_post_f27/`. Acceptance: MTL-B3 cat F1 post-F27 remains $>$ STL `next_gru` cat F1 at 5f × 50ep (matched-head). If STL `next_gru` $\ge$ MTL-B3, the cat-side MTL-over-STL claim collapses into matched-head comparison and needs re-framing. | Fecha o \textit{matched-head} STL para cat pós-F27 sem asterisco arquitetural. Hoje STL cat usa `next_mtl` (matched com a \emph{antiga} config). |
| **F38** | **P1** | **Exp A --- diagnostic\_task\_best re-análise de B3 AL/AZ 5f JSONs** | 2b | m4\_pro | **done 2026-04-24** | **REFUTADO.** Δ (task-best − joint-best) em reg Acc@10 = **−0{,}01 pp AL**, **−0{,}40 pp AZ**. Fator 2 (seleção de \textit{checkpoint}) \emph{não} é \textit{load-bearing} em AL/AZ sob B3. Mecanismo: `val_joint_geom_lift` naturalmente escolhe a época reg-best porque cat F1 já estabiliza em $\sim$ep 42 e só reg ainda melhora. Análise completa em `research/F38_CHECKPOINT_SELECTION.md`. Atribuição agora foca em Fator 1 (loss weight) e Fator 3 (upstream cross-attn). |
| **F39** | **P2** | **Exp B --- cat\_weight sweep em B3 (cat=0.50, cat=0.25) em AL+AZ 5f** | 2b | m4\_pro | $\sim$1h AL + $\sim$1.5h AZ $\times$ 2 \textit{weights} = $\sim$5h | Rodar B3 em AL e AZ 5f$\times$50ep com `--category-weight 0.50` e `--category-weight 0.25`. Monitor: reg Acc@10 joint-best. Acceptance: se reg Acc@10 $\uparrow$ $\ge 3$ pp quando cat\_weight $\downarrow$, confirma Fator 1 (peso da \textit{loss}) como \textit{load-bearing}. Se reg Acc@10 não mover, o peso não é o culpado e a análise deve escalar pro Fator 3 (upstream cross-attn). | Isola Fator 1 do Fator 2. Já temos F2 a n=1 em FL mas sob pcgrad; este replica sob static em AL/AZ. |
| **F40** | **P3** | **Exp C --- scheduled handover: cat\_weight $0{,}75 \to 0{,}25$ linear em 50 épocas** | 2b | m4\_pro | $\sim$1h dev + $\sim$3h treino (AL+AZ 5f) | Implementar \texttt{ScheduledStaticWeight} (nova \textit{loss class} em \texttt{src/losses/scheduled\_static/}). Interpolar linearmente \texttt{category\_weight} de $0{,}75$ (ep 0) a $0{,}25$ (ep 49). Mantém \textit{late-stage handover} mas passa o \textit{budget} gradiente pro reg na segunda metade. Acceptance: cat F1 $\ge$ B3 current -1 pp AND reg Acc@10 $>$ B3 current +3 pp --- \textit{Pareto-lift}. | Combina Fatores 1+2: mantém vantagem do handover (cat converge cedo) enquanto libera gradiente pro reg tarde. Pode fechar $\sim$metade do \textit{gap} F21c. |
| **F41** | **P1** | **Exp D --- STL \texttt{next\_getnext\_hard} com pré-encoder MTL (MLP 64$\to$256)** | 2b | m4\_pro (em execução) | $\sim$3h treino (AL + AZ 5f) | **Código landed 2026-04-24**: \texttt{p1\_region\_head\_ablation.py} estendido com \texttt{--mtl-preencoder / --preenc-hidden / --preenc-layers / --preenc-dropout}; classe \texttt{\_MTLPreencoder} espelha \texttt{MTLnet.\_build\_encoder}; smoke test 1f$\times$2ep OK. Launcher: \texttt{scripts/run\_f41\_stl\_mtl\_preencoder.sh}. Acceptance: diferença $\le 2$ pp vs STL puro (F21c) $\Rightarrow$ Fator 3 não é \textit{load-bearing}; $\ge 5$ pp $\Rightarrow$ upstream MTL lava o sinal da \textit{prior}; intermediário $\Rightarrow$ precisa D-2 com cross-attn self-attention. | Isola Fator 3 (distribuição de entrada do head). **De volta a P1** após F42 refutar o Fator 5 --- arquitetura upstream é o único suspeito restante para CH18. |
| **F42** | **done 2026-04-24** | **Exp E --- \textit{epoch budget} em B3 AL 5f $\times$ 150ep** | 2b | m4\_pro | ~31 min | **HIPÓTESE REFUTADA NA DIREÇÃO OPOSTA.** 150ep \emph{piora}: reg Acc@10 $59{,}60 \to 56{,}14$ ($-3{,}46$ pp), cat F1 $42{,}71 \to 40{,}68$ ($-2{,}03$ pp); σ \emph{inalterada} (4.09 $\to$ 4.00). Per-fold reg-best ep migrou para [20..27] (dentro do warmup do OneCycleLR esticado). **Fator 5 refutado.** Mecanismo: OneCycleLR com mesmo \texttt{max\_lr=3e-3} estica o cronograma proporcional ao budget; em 150ep o \textit{peak LR} acai em ep~45 mas o modelo já pica antes (ep 20) e degrada depois. σ é intrínseca ao acoplamento MTL+OneCycleLR, não de \textit{under-training}. **Implicação CH18:** Fator 3 (upstream arch) é agora único suspeito; F41 promovido a P1. Paper \textit{protocol} fica com 50ep justificado. | Descoberta \textit{worth reporting}: MTL + OneCycleLR é budget-sensitivo mesmo. Se o paper quiser explorar \textit{budget}, teria que variar \texttt{max\_lr} junto OR usar \textit{early stopping} --- isso é território de follow-up. |

---

## 2 · Gated

| # | Pri | Item | Gate |
|---|:-:|---|---|
| **F5** | P4 | FL MTL-GETNext-soft 5-fold | Only if the paper explicitly needs to compare soft vs B3 at FL 5-fold. Currently covered by the F2 B3-vs-soft comparison at n=1 (B3 Pareto-dominates); 5f replication is nice-to-have, not blocking. |

---

## 3 · Deferred (follow-up paper or post-champion-freeze)

| # | Item | Cost | When to revisit |
|---|---|:-:|---|
| **F8** | **Multi-seed n=3 on champion configs** | **~20 h MPS total, parallelisable** | **Deferred by user 2026-04-23.** Held until headline (FL/CA/TX) completes + B3 is frozen. Seeds {42, 123, 2024} × 5 folds × headline configs. |
| F12 | Per-fold transition matrix (leakage-safe GETNext) | ~4 h impl + 2 h reruns | Camera-ready, if reviewer asks for per-fold protocol. |
| F13 | GETNext with true flow-map Φ = (Φ₁1ᵀ + 1Φ₂ᵀ) ⊙ (L̃ + J) | ~30 min impl + rerun | Follow-up paper. |
| F14 | PIF-style user-specific region frequency prior | ~3–4 h | Follow-up paper. |
| F15 | TGSTAN + STA-Hyper full reproductions | ~2–3 weeks | Follow-up paper. |
| F16 | Encoder enrichment (P6: temporal / spatial / graph features) | multi-week track | Post-paper research direction. |
| F26 | AZ HGI STL cat for CH16 | ~3 h | Nice-to-have at headline states (CA/TX). Deferred unless reviewer asks. |
| **F36** | **CH15 under graph-prior head — HGI STL `next_getnext_hard` on AL** | **~3 h** | Extends CH15 (Check2HGI ≈ HGI on reg) from `next_tcn_residual` to the F21c matched-head baseline. Substrate-agnostic *a priori* (the hard prior reads `region_transition_log.pt`, not embeddings) but STAN co-adapts differently. Run only if reviewer asks whether CH16 replicates on reg under graph-prior. |
| **F21a** | **DROPPED** FL STL STAN 5f | — | Dropped 2026-04-24 by user decision — STAN is our reg baseline reference, not an ablation data point we need to publish. |
| **F21b** | Archived — STL GETNext (soft) 5f per state | ~12 h total | User-deprioritised 2026-04-24. Would give a "graph prior alone" reference vs hard, but not blocking any headline claim. Revisit if reviewer asks specifically about soft-vs-hard at the STL level. |
| **F27** | Cat-head ablation (try `next_gru`, `next_stan`, ensemble, etc. as task_a heads on MTL-B3) | ~1 h 1-fold AZ sweep + 5f runs for any winner | Follow-up / camera-ready if a reviewer asks "did you try other cat heads". Current `CategoryHeadTransformer` has not been head-ablated. Cost-benefit today: cat isn't the bottleneck, reopening the cat-head choice could break the committed north-star. |

---

## 4 · Done (audit trail, this push 2026-04-21 → 2026-04-23)

| Item | Finished | Evidence |
|---|:-:|---|
| `MTL_PARAM_PARTITION_BUG` fix + 6 contaminated reruns | 2026-04-22 | commits `5668856` `c1c7f3e`; `results/P5_bugfix/SUMMARY.md` |
| `CROSSATTN_PARTIAL_FORWARD_CRASH` fix | 2026-04-22 | commit `8afc9ac`; `tests/test_regression/test_mtlnet_crossattn_partial_forward.py` |
| B5 hard-index GETNext implementation + AL + AZ 5f retraining + FL 1f | 2026-04-22 | commits `6a2f808` `ea65fb3`; `results/B5/*.json`; `research/B5_RESULTS.md` |
| ALiBi × GETNext on AL (B7) | 2026-04-22 | `research/B7_ALIBI_GETNEXT_FINDINGS.md` |
| α-inspection (GETNext / TGSTAN / STA-Hyper, AL+AZ) | 2026-04-21 | `research/GETNEXT_FINDINGS.md §α inspection` |
| PCGrad-vs-static attribution on GETNext | 2026-04-22 | `research/ATTRIBUTION_PCGRAD_VS_STATIC.md` |
| North-star committed initially to `GETNext-soft` | 2026-04-23 | `NORTH_STAR.md` (pre-F2) |
| Critical review post-B5 | 2026-04-23 | `review/2026-04-23_critical_review.md` |
| **F2** FL-hard training-pathology diagnosis (4 phases, ~2h37m) | 2026-04-23 03:18 | `research/B5_FL_TASKWEIGHT.md` · B3 identified as new champion candidate · Pareto-dominates soft at FL n=1 on all 4 joint metrics |
| **F1** AZ Wilcoxon B-M9b soft vs B-M9d hard | 2026-04-23 | `research/B5_AZ_WILCOXON.md` · Every region metric p=0.0312 (n=5 minimum) · `scripts/analysis/az_wilcoxon.py` |
| **F11** MTLoRA post-fix tables update | 2026-04-23 | `results/BASELINES_AND_BEST_MTL.md` + `results/RESULTS_TABLE.md` |
| **F7** POI-RGNN protocol audit | 2026-04-23 | `docs/baselines/POI_RGNN_AUDIT.md` |
| **F10** CLAIMS_AND_HYPOTHESES dashboard reconciled | 2026-04-23 | `CLAIMS_AND_HYPOTHESES.md §Summary dashboard` |
| **F18** B3 5f AL | 2026-04-23 03:39 | `results/B3_validation/al_5f50ep_b3.json` · cat F1 +0.78 pp over B-M6e · no regression |
| **F19** B3 5f AZ | 2026-04-23 04:05 | `results/B3_validation/az_5f50ep_b3.json` · cat F1 +1.40 pp vs B-M9d, +1.54 pp vs STL STAN · Pareto-dominates soft (+0.80 cat, +6.10 reg) |
| **F19-followup** B3 Wilcoxon vs STL at AZ | 2026-04-23 | `research/B3_AZ_WILCOXON_VS_STL.md` · cat F1 +1.65 pp p=0.0312 (strict MTL-over-STL) · reg macro-F1 +3.75 pp p=0.0312 · reg Acc@10 tied · reg MRR −7.94 pp (STL wins) |
| **F20** per-fold persistence in `MLHistory` | 2026-04-23 06:35 | `src/tracking/storage.py::save_fold_partial` + `src/tracking/experiment.py::step` hook · 4 regression tests + 186 full-suite green · validated in vivo (F17 fold 1 persisted despite user-kill) |
| Doc cleanup — archive pre-B3 framing + research, create `PAPER_STRUCTURE.md` | 2026-04-23 | `archive/pre_b3_framing/`, `archive/research_pre_b3/`, `archive/phases_original/`, new `PAPER_STRUCTURE.md` |

---

## 5 · Progress snapshot (2026-04-23 evening)

| Objective | AL | AZ | FL | CA | TX | What closes |
|---|:-:|:-:|:-:|:-:|:-:|---|
| **Obj 1** CH16 (Check2HGI > HGI cat) | 🟢 +18.30 pp σ-clean | 🔴 not run | 🔴 not run | 🔴 | 🔴 | F3 + F9 (CA/TX optional) |
| **Obj 2a** MTL > baselines (cat) | 🟢 | 🟢 | 🟢 n=1 | 🔴 | 🔴 | F24 + F25 |
| **Obj 2a** MTL > baselines (reg) | 🟢 +10.95 pp | 🟢 +10.29 pp | 🟡 Markov saturation (approach a) | 🔴 | 🔴 | F24 + F25 |
| **Obj 2b** MTL > STL (cat) | 🟡 tied | 🟢 +1.65 pp p=0.0312 | 🟢 n=1 +0.22 | 🔴 | 🔴 | F24 + F25 + F4 |
| **Obj 2b** MTL > STL (reg) | 🟡 σ-tied | 🟢 reg F1 +3.75 pp p=0.0312, Acc@10 tied | 🟡 n=1 +5.20 pp | 🔴 | 🔴 | F4 + F6 + F24 + F25 |
| **NEW** MTL coupling > head alone | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 | **F21 per state** |

---

## 6 · Recommended execution order (P1 first)

**On m4_pro (this machine):**
1. **F21 AL + AZ + FL** — matched-head STL baselines (~3+3+6 = ~12 h sequential). Highest information gain per hour.
2. **F3 AZ HGI STL cat** (~3 h) — cheap CH16 extension.
3. **F4 FL MTL-B3 5f clean** (~6–8 h) — headline FL σ.
4. **F6 FL STL STAN 5f** (~5–6 h) — region ceiling row at FL.

**On m2_pro (in parallel, from kickoff):**
1. **F22 CA pipeline** (~4–8 h) → **F24 CA headline** (~20–25 h).
2. **F23 TX pipeline** (~4–8 h) → **F25 TX headline** (~20–25 h).

**On 4050 (if available):** overflow from m4_pro queue.

The m2_pro path is the critical-path wall-clock: CA and TX upstream + headline = ~50–70 h each. m4_pro fills in FL-specific runs + matched-head STLs + Obj 1 replication in parallel.
