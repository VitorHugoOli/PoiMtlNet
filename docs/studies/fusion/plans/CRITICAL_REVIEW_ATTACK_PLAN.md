# Plano de Ataque — Critical Review do Report

**Objetivo:** resolver os pontos do `docs/REPORT_CRITICAL_REVIEW.md` antes da submissão ao BRACIS (2026-04-20).

**Filosofia:** ser ruthless sobre o que é blocking vs. nice-to-have. Cada experimento custa tempo de máquina; cortamos o que não move uma claim do paper.

---

## Tier 0 — Bloqueadores absolutos (faz ou desiste da claim)

### T0.1 — Single-task-fusion baseline
**Por quê é blocker:** sem isso, "MTL helps when configured right" é uma claim sem evidência. Se single-task-fusion já atinge ~80% Cat F1, a tese inteira do paper desmorona porque a melhoria é da fusão, não do MTL.

**Experimentos:**
- `python scripts/train.py --task category --state alabama --engine fusion --epochs 50 --folds 5 --model mtlnet_dselectk --embedding-dim 128`
- `python scripts/train.py --task next --state alabama --engine fusion --epochs 50 --folds 5 --model mtlnet_dselectk --embedding-dim 128`
- Repetir os dois para Florida (na outra máquina)

**Tempo:** Alabama ~45 min total; Florida ~8 h total.

**Critério de decisão:**
- Se MTL-fusion > single-task-fusion (ambas tarefas): **claim confirmada**, paper segue como está.
- Se MTL-fusion ≈ single-task-fusion (delta < 1 p.p.): **reframe necessário** — o ganho é da fusão+otimizador, e o MTL não traz benefício adicional. A tese vira "fusão multi-fonte com cirurgia de gradiente é o que faz MTL funcionar para POI" em vez de "MTL bem configurado supera single-task".
- Se MTL-fusion < single-task-fusion: **abandonar a claim de MTL**, virar paper de "fusion+gradient surgery for POI prediction" sem MTL como hook.

**Se não der tempo:** suavizar todas as claims de "MTL beneficia" para "framework multitarefa atinge SOTA" (descritivo, não comparativo).

---

### T0.2 — Matched-batch equal_weight fusion
**Por quê é blocker:** o gap de 25% entre ca/al e eq/db/uw pode ser parcialmente artefato de batch size (4096 vs 8192). Sem o controle, a claim "gradient surgery is essential" fica vulnerável.

**Experimentos:**
- Rodar `equal_weight + DSelectK + Fusion` no Alabama com `--gradient-accumulation-steps 1` (effective batch 4096, mesmo dos otimizadores ca/al)
- Mesma config em 2f×15ep (rápida) e 5f×50ep (confirmação)

**Tempo:** ~25 min total no Alabama.

**Critério de decisão:**
- Se equal_weight matched-batch ainda perde por >15% para ca/al: **claim "gradient surgery essential" confirmada** com matched-batch.
- Se equal_weight matched-batch fecha a maior parte do gap (<5% de diferença): **reframe necessário** — o ganho era de batch size, não de cirurgia de gradiente. Vira "smaller effective batches favor MTL on fusion" (achado menos bonito mas honesto).

---

### T0.3 — HGI-only no regime matched (5f × 50ep)
**Por quê é blocker:** a "Evolução do MTLnet" do report compara HGI@2f/15ep com Fusion@5f/50ep — apples-to-oranges. HGI a 50 épocas pode subir ~0.02 joint, encolhendo o gap.

**Experimentos:**
- `python scripts/train.py --task mtl --state alabama --engine hgi --epochs 50 --folds 5 --model mtlnet_dselectk --mtl-loss aligned_mtl --embedding-dim 64 --gradient-accumulation-steps 1 --model-param num_experts=4 --model-param num_selectors=2 --model-param temperature=0.5`
- E também o "antigo HGI champion": `mtlnet_cgc(s2,t2) + equal_weight` a 5f×50ep para a tabela evolutiva.

**Tempo:** ~45 min total no Alabama.

**Critério de decisão:**
- Independente do resultado, atualizar a tabela evolutiva com regime matched. Não muda a tese principal — só fortalece a comparação.

---

## Tier 1 — Honestidade científica (faz se possível)

### T1.1 — Std dev em todas as claims
**Por quê:** "78,3% de Cat F1" sem ± é cientificamente fraco. Já temos os per-fold metrics nos run dirs.

**Experimentos:** nenhum novo. Só recomputar.

**Tempo:** ~30 min de script.

**Output:** atualizar todas as tabelas do report e do paper com mean ± std.

### T1.2 — Multi-seed do champion
**Por quê:** PAPER_FINDINGS.md Finding 10 mostra std entre seeds ~0.005 — pequeno, mas BRACIS_GUIDE recomenda múltiplos seeds.

**Experimentos:**
- Champion config (DSelectK+Aligned-MTL+fusion) com seeds {123, 2024} no Alabama, 5f×50ep.

**Tempo:** ~45 min total (2 × 22 min).

**Critério de decisão:**
- Se std entre seeds > 0.01 no joint, reportar honestamente como variabilidade. Senão, o resultado fica blindado.

### T1.3 — Suavizar comparação com HAVANA/POI-RGNN
**Por quê:** HAVANA usa grafo bruto; nós usamos embeddings pré-treinados. Comparação direta é injusta.

**Experimentos:** nenhum.

**Output:** adicionar 1-2 frases no paper qualificando a comparação. Trocar "supera HAVANA" por "outperforms published HAVANA results, noting different input representations". Manter os números (são reais).

### T1.4 — Definir "25%" e "4,8%" precisamente
**Experimentos:** nenhum.

**Output:** no paper, escrever explicitamente: "the top-10 mean joint score is 25% higher than the bottom-15 mean (0.485 vs 0.342, normalized)" ou similar. Sem ambiguidade.

### T1.5 — Dropar ou derivar a decomposição 73%/7%/20%
**Por quê:** os efeitos são interativos (não-aditivos), então a decomposição é metodologicamente suspeita.

**Decisão:** **dropar** do paper. Manter só a claim qualitativa "embedding choice contributes the largest share, followed by optimizer (which only matters with fusion), then architecture".

---

## Tier 2 — Reforço de evidência (se sobrar tempo)

### T2.1 — Gradient cosine diagnostic
**Por quê:** a claim mecanicista "fusion creates source-level gradient conflict" é empírica mas não medida.

**Experimentos:**
- Adicionar instrumentação para medir cosine entre gradientes do encoder de categoria por fonte (Sphere2Vec dims vs HGI dims) durante 1 run.
- Comparar com cosine no HGI-only.

**Tempo:** ~2 h (instrumentação + 2 runs).

**Output:** uma figura mostrando que o cosine-de-fonte é menor que o cosine-de-tarefa na fusão. Forte para a Section 5.

### T2.2 — California e Texas com a config champion
**Por quê:** strengthens the cross-state generalization claim.

**Experimentos:**
- `--state california --engine fusion` com champion config (5f×50ep)
- `--state texas --engine fusion` com champion config

**Tempo:** ~6 h cada (na outra máquina, em paralelo).

**Output:** tabela cross-state expandida no paper.

### T2.3 — Wall-clock / FLOPs vs CBIC
**Por quê:** o CBIC reportou MTL = 4× single-task em wall-clock. Se nosso novo MTL ainda é 4× mais lento que single-task-fusion, o achado do CBIC ainda vale parcialmente.

**Experimentos:** nenhum novo (extrair dos logs do T0.1 e do Stage 3).

**Output:** uma frase no paper sobre o trade-off de compute.

---

## Tier 3 — Não fazer (corte ruthless)

### T3.1 — Reproduzir HAVANA / PGC nosso pipeline
**Por quê não:** ~2 dias de trabalho, baixo retorno. O número da Florida (54.23% reproduzido vs 62.9% paper) já está em BASELINE.md como nota — basta citar.

### T3.2 — Comparar todos os 8 engines
**Por quê não:** já está em PAPER_FINDINGS.md (Finding 6 e 8). Vira material de journal, não BRACIS.

### T3.3 — 19 otimizadores benchmark completo
**Por quê não:** Phase 3 já foi rodado. Não precisa repetir para BRACIS. Cita o resultado existente.

### T3.4 — Florida HGI-only equal_weight para "Time2Vec scaling"
**Por quê não:** a claim "Time2Vec beneficia da base maior" foi removida do report (já estamos honestos). Não precisa do controle.

---

## Cronograma sugerido (5 dias até 2026-04-20)

### Dia 1 (hoje, 04-13 noite — Alabama)
- T0.1 single-task-fusion Alabama (cat + next): 45 min
- T0.2 matched-batch equal_weight Alabama: 25 min
- T0.3 HGI-50ep Alabama (2 configs): 45 min
- T1.2 multi-seed (1 seed): 22 min
- **Total: ~2.5 h** — termina antes de dormir

### Dia 2 (04-14 — Florida na outra máquina + Alabama complementar)
- T0.1 single-task-fusion Florida: 8 h em background
- T1.2 multi-seed (2º seed Alabama): 22 min
- T2.1 gradient cosine instrumentação + run: 2 h
- T1.1 recompute std devs: 30 min
- **Total Alabama: ~3 h ativos**

### Dia 3 (04-15)
- Análise dos resultados de T0.1 → decisão sobre framing do paper
- T2.2 California fusion start (na outra máquina)
- Começar redação do paper em LaTeX (estrutura)

### Dia 4 (04-16)
- T2.2 Texas fusion + análise
- Redação contínua

### Dia 5 (04-17–04-19)
- Finalização do paper, revisão, submissão

---

## Resumo executivo

| Tier | Itens | Tempo total | Decisão |
|------|-------|-------------|---------|
| 0 (Blocker) | T0.1, T0.2, T0.3 | ~10 h (com Florida em paralelo) | **Obrigatório** |
| 1 (Honestidade) | T1.1, T1.2, T1.3, T1.4, T1.5 | ~1.5 h ativas | **Recomendado** |
| 2 (Reforço) | T2.1, T2.2, T2.3 | ~14 h (Florida/CA/TX em paralelo) | **Se der tempo** |
| 3 (Cortes) | T3.1, T3.2, T3.3, T3.4 | — | **Não fazer** |

**Mínimo viável para BRACIS:** Tier 0 + T1.1 + T1.3 + T1.4 + T1.5 = ~10 h compute + ~3 h escrita.

**Risco residual mesmo com tudo feito:** se T0.1 mostrar que single-task-fusion ≈ MTL-fusion, todo o paper precisa ser reframed em ~2 dias. Plano B: tornar o paper sobre fusão+otimizador (sem o ângulo MTL), ainda publicável mas perde o hook do CBIC.
