# Plano de Ablação — Cabeças de Tarefa com a Nova Arquitetura

**Objetivo:** investigar sistematicamente quais cabeças funcionam melhor com o backbone campeão (DSelectK + Aligned-MTL + Fusão), refinando o achado parcial do Stage 2 do estudo principal.

**Motivação:** o Stage 2 testou apenas 3 combinações de cabeças (DCN, TCN, ambas) sobre os 3 melhores backbones, em 2 folds × 15 épocas. Isso foi suficiente para um sinal preliminar (DCN ajuda em treino curto), mas não é uma ablação completa. Para o paper de journal (Opção B) precisamos de uma comparação mais robusta.

---

## Hipóteses a testar

1. **H1 (do Stage 2):** A vantagem do DCN em treino curto desaparece em treino longo (DCN coadapta lentamente; cabeça padrão eventualmente alcança).
2. **H2 (não testada):** Existe uma cabeça que supera a padrão em treino longo se for explicitamente projetada para lidar com a fusão (ex.: cabeça com gating por fonte).
3. **H3 (relevante para next):** Cabeças sequenciais alternativas (TCN, Conv-Attn, Transformer com pos relativa) podem superar o Transformer padrão na presença do Time2Vec, que torna a posição na janela menos informativa.
4. **H4 (controle):** Cabeças mais simples (linear, MLP único) podem ser competitivas se o backbone DSelectK já for expressivo o suficiente.

---

## Configuração fixa

- **Embedding:** Fusão (Sphere2Vec+HGI para categoria; HGI+Time2Vec para next), 128D
- **Arquitetura:** DSelectK (e=4, k=2, temp=0.5)
- **Otimizador:** Aligned-MTL (com `gradient_accumulation_steps=1`)
- **Estado:** Alabama (rápido para iteração), depois Florida para os melhores
- **Validação cruzada:** triagem 1f×10ep; promoção 2f×15ep; confirmação 5f×50ep
- **Seed:** 42 (multi-seed apenas para os top-3 finais)

---

## Fase A — Cabeça de Categoria (varia categoria, fixa next padrão)

**Candidatos (9):**
| Cabeça | Descrição | Razão para incluir |
|--------|-----------|-------------------|
| `category_default` (CategoryHeadMTL) | Ensemble 3-MLP padrão | Baseline |
| `category_dcn` | Deep & Cross Network | Vencedor isolado em HGI; testado parcialmente no Stage 2 |
| `category_residual` | MLP residual | Top-2 isolada em HGI |
| `category_gated` | MLP com gating | Top-3 isolada em HGI |
| `category_single` | MLP único | Controle simples |
| `category_deep` | MLP profundo (5+ camadas) | Capacidade máxima |
| `category_linear` | nn.Linear único | Probe diagnóstico (mede o que o backbone aprendeu) |
| `category_attention` | Pooling atencional (sem grafo) | Variante atencional |
| `category_transformer` | Transformer encoder (default isolado, mas pior em MTL) | Sanity check do paradoxo |

**Triagem:** 9 runs × 1f×10ep ≈ 7 min total no Alabama.

**Promoção:** top-3 a 5f×50ep ≈ 70 min.

---

## Fase B — Cabeça de Next (varia next, fixa categoria padrão)

**Candidatos (10):**
| Cabeça | Descrição | Razão para incluir |
|--------|-----------|-------------------|
| `next_default` (NextHeadMTL) | Transformer 4-camadas, 8-heads | Baseline |
| `next_tcn_residual` | TCN com blocos residuais | Vencedor isolado em HGI |
| `next_temporal_cnn` | CNN temporal simples | Top-2 isolada |
| `next_single` | MLP único sobre concat | Top-3 isolada |
| `next_conv_attn` | CNN + atenção | Híbrida |
| `next_transformer_relpos` | Transformer com pos relativa | Específica para sequências curtas |
| `next_gru` | GRU bidirecional | Recorrente clássica |
| `next_lstm` | LSTM | Recorrente clássica |
| `next_mlp_pool` | MLP sobre pooling | Controle simples |
| `next_attention_pool` | Apenas pooling atencional | Probe diagnóstico |

**Triagem:** 10 runs × 1f×10ep ≈ 8 min.

**Promoção:** top-3 a 5f×50ep ≈ 70 min.

**Nota:** algumas dessas cabeças podem não estar implementadas no registry (ex.: `next_gru`, `next_lstm`, `next_attention_pool`). Verificar `src/models/next/` antes de rodar e cortar as ausentes do escopo.

---

## Fase C — Combinações dos vencedores (sinergias)

**Candidatos (3-5):**
- Top-1 cat × top-1 next
- Top-1 cat × default next
- Default cat × top-1 next
- Top-1 cat × top-2 next (caso top-1 next seja borderline)

**Triagem:** já temos default×default do estudo principal. As 3-5 novas combinações a 2f×15ep ≈ 30 min, depois top-2 a 5f×50ep ≈ 45 min.

---

## Fase D — Multi-seed dos finalistas

Top-3 combinações finais com seeds {123, 2024} a 5f×50ep ≈ 90 min.

---

## Fase E — Cross-state Florida (apenas top-1)

Melhor combinação na Florida 5f×50ep ≈ 4 h.

---

## Cronograma total

| Fase | Runs | Tempo | Acumulado |
|------|------|-------|-----------|
| A triagem | 9 | 7 min | 7 min |
| A promoção | 3 | 70 min | 77 min |
| B triagem | 10 | 8 min | 85 min |
| B promoção | 3 | 70 min | 155 min (~2,5 h) |
| C combinações | 5 | 75 min | ~3,8 h |
| D multi-seed | 6 | 90 min | ~5,3 h |
| E Florida | 1 | 4 h | ~9,3 h |
| **Total** | **37 runs** | | **~9-10 h** |

Doable em 1-2 dias se a outra máquina estiver disponível para Florida.

---

## Critérios de decisão

- **Se Fase A produzir uma cabeça > default por >2 p.p. no joint a 5f×50ep:** cabeças importam para fusão. Update do paper em Section 5.3.
- **Se Fase A confirmar default ≥ alternativas:** confirma a hipótese de coadaptação (PAPER_FINDINGS Finding 3) sob fusão também. Forte para discussão.
- **Se Fase B mostrar TCN ou GRU > default:** revoga o achado preliminar do Stage 2 (que foi com 2f×15ep apenas) e abre caminho para um achado novo.
- **Se Fase C revelar sinergia (combinação > soma das partes):** seria um insight legítimo sobre coadaptação par a par.

---

## O que essa ablação NÃO faz

- Não testa hyperparams das próprias cabeças (ex.: profundidade do DCN, kernel size do TCN). Isso é deliberado — variar muitas dimensões simultaneamente diluiria o sinal.
- Não testa cabeças em backbones outros que DSelectK. Isso é deliberado — já sabemos do Stage 2 que a interação é específica (CGC21+DCN piorou).
- Não testa em todos os estados — só Alabama+Florida.

---

## Output esperado

Uma tabela única para o paper (Section 5.3 ou Appendix):

| Cat Head | Next Head | Joint (5f×50ep) | $\Delta$ vs default×default |
|----------|-----------|-----------------|------------------------------|
| default | default | 0.540 ± 0.015 | — |
| ... | ... | ... | ... |

E uma figura com: cabeças isoladas (em treinamento single-task) ranqueadas vs ranqueamento na combinação MTL, mostrando a inversão (Finding 3).
