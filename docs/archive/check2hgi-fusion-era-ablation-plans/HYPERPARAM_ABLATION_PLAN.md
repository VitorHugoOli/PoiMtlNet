# Plano de Ablação — Hiperparâmetros da Configuração Campeã

**Objetivo:** entender a sensibilidade da configuração campeã (DSelectK + Aligned-MTL + Fusão) aos seus hiperparâmetros principais, identificar se há ganhos relevantes a obter por tuning, e blindar o paper contra a pergunta "por que esses valores específicos?".

**Filosofia crítica:** muitos hiperparâmetros podem ser variados, mas só alguns realmente importam. Vamos atacar os de maior alavancagem teórica e cortar os de baixa probabilidade de impacto.

---

## Configuração base (a partir da qual variamos um eixo de cada vez)

- Embedding: Fusão (128D)
- Arquitetura: DSelectK (e=4, k=2, temp=0.5)
- Otimizador: Aligned-MTL
- LR: 1e-4, batch: 4096, dropout: 0.2 (do código)
- Shared backbone: 4 blocos residuais, 256-dim
- Encoder: 256-dim, 2 camadas
- Estado: Alabama (rápido), seed 42

---

## Eixo 1 — Hiperparâmetros do DSelectK (alta prioridade)

**Por que importa:** os valores `e=4, k=2, temp=0.5` são padrões arbitrários do paper original. Reviewers vão perguntar.

### 1.1 Número de experts (`num_experts`)
**Valores:** {2, 4, 6, 8}
**Hipótese:** mais experts = mais capacidade de roteamento mas mais parâmetros. Pode haver platô em e=4.
**Runs:** 4 a 1f×10ep (triagem) ≈ 4 min. Top-2 a 5f×50ep ≈ 45 min.

### 1.2 Número de seletores (`num_selectors`)
**Valores:** {1, 2, 3, 4}
**Hipótese:** k=1 vira gating "hard"; k>2 pode ser overkill.
**Runs:** 4 × 1f×10ep ≈ 4 min. Top-2 a 5f×50ep ≈ 45 min.

### 1.3 Temperatura (`temperature`)
**Valores:** {0.1, 0.3, 0.5, 0.7, 1.0}
**Hipótese:** baixa temperatura = roteamento mais decisivo; alta = mais soft. Pode interagir com o conflito de gradiente.
**Runs:** 5 × 1f×10ep ≈ 5 min. Top-2 a 5f×50ep ≈ 45 min.

**Sub-total Eixo 1:** ~13 runs de triagem + 6 runs de confirmação ≈ 2,5 h.

---

## Eixo 2 — Tamanho do backbone compartilhado (CRÍTICO para a tese)

**Por que importa:** PAPER_FINDINGS.md Finding 4 mostra que o backbone compartilhado é apenas 10% dos parâmetros do modelo --- a parte que faz o trabalho de MTL é a menor. Se aumentar o backbone melhorar significativamente, isso é um achado novo e relevante para o paper.

### 2.1 `shared_layer_size`
**Valores:** {128, 256, 384, 512}
**Hipótese:** o atual 256 pode estar saturando. Backbone maior pode capturar mais transferência de conhecimento.
**Runs:** 4 × 1f×10ep ≈ 4 min. Top-2 a 5f×50ep ≈ 45 min.

### 2.2 `num_shared_layers`
**Valores:** {2, 4, 6, 8}
**Hipótese:** mais profundidade pode ajudar a desemaranhar a fusão.
**Runs:** 4 × 1f×10ep ≈ 4 min. Top-2 a 5f×50ep ≈ 45 min.

### 2.3 Combinação ótima (depth × width)
Após achar os melhores 2.1 e 2.2 isoladamente, testar a combinação a 5f×50ep ≈ 22 min.

**Sub-total Eixo 2:** 8 runs triagem + 5 runs confirmação ≈ 2 h. **Maior valor potencial para o paper.**

---

## Eixo 3 — CAGrad temperature (validação cruzada do otimizador)

**Por que importa:** CAGrad usa hyperparam `c=0.4` (default do paper). Se variar c não muda nada, blindamos a comparação ca-vs-al.

### 3.1 CAGrad `c`
**Valores:** {0.2, 0.4, 0.6, 0.8}
**Hipótese:** c controla quão averso ao conflito o algoritmo é. c→0 vira equal_weight; c→1 vira pure conflict-averse.
**Runs:** 4 × 1f×10ep ≈ 4 min. Top-1 a 5f×50ep ≈ 22 min.

**Sub-total Eixo 3:** ~30 min. Confirma que CAGrad é robusto às configurações.

---

## Eixo 4 — Treinamento (LR, batch, schedule)

**Por que importa:** o batch-size confound da review. Se mostrarmos que o resultado é estável em torno do batch atual, blindamos a claim.

### 4.1 Learning rate
**Valores:** {5e-5, 1e-4, 2e-4, 5e-4}
**Runs:** 4 × 1f×10ep ≈ 4 min. Top-1 a 5f×50ep ≈ 22 min.

### 4.2 Batch size (com matched grad accumulation)
**Valores:** {2048, 4096, 8192, 16384}
**Hipótese:** isso ataca diretamente o batch-size confound. Se champion (Aligned-MTL) for estável em todos os batches, prova que o ganho não é de batch.
**Runs:** 4 × 1f×10ep ≈ 4 min. Top-2 a 5f×50ep ≈ 45 min.

### 4.3 Dropout
**Valores:** {0.1, 0.2, 0.3, 0.4}
**Runs:** 4 × 1f×10ep ≈ 4 min. Top-1 a 5f×50ep ≈ 22 min.

**Sub-total Eixo 4:** ~2 h. **4.2 é importante para a critical review.**

---

## Eixo 5 — Window size para a tarefa next (baixa prioridade)

**Por que considerar:** janela atual de 9 check-ins é arbitrária.

### 5.1 `slide_window`
**Valores:** {5, 7, 9, 11, 15}
**Hipótese:** janela maior = mais contexto temporal, mas mais padding em usuários esparsos.
**Runs:** Requer regerar inputs de next para cada janela. **Caro.** Pular para o paper de conferência; deixar para journal.

**Decisão:** **NÃO incluir no plano principal.**

---

## Eixo 6 — Cortes (não fazer)

| Hiperparâmetro | Por que cortar |
|----------------|---------------|
| `seq_length` | Já fixo pela natureza da tarefa |
| `embedding_dim` (Time2Vec, Sphere2Vec) | Cada um é treinado separadamente; varia o pipeline upstream |
| `weight_decay` | Pouco impacto, default AdamW funciona |
| `eps` AdamW | Microparâmetro |
| `OneCycleLR` parameters | Schedule OK como está |
| Aligned-MTL hparams | Não tem (essa é uma característica que vendemos) |

---

## Cronograma total

| Eixo | Triagem | Confirmação | Total |
|------|---------|-------------|-------|
| 1 (DSelectK) | 13 × ~1 min = 13 min | 6 × 22 min = 132 min | ~2,5 h |
| 2 (Backbone size) | 8 × ~1 min = 8 min | 5 × 22 min = 110 min | ~2 h |
| 3 (CAGrad c) | 4 × ~1 min = 4 min | 1 × 22 min = 22 min | ~30 min |
| 4 (LR/batch/dropout) | 12 × ~1 min = 12 min | 4 × 22 min = 88 min | ~1,7 h |
| **Total** | **~37 min triagem** | **~5,9 h confirmação** | **~7 h** |

---

## Critérios de decisão

### Para o paper de BRACIS (curto prazo):
- **Eixo 4.2 é prioritário:** mostra que o ganho do Aligned-MTL não é batch-size artifact.
- **Eixo 1.1, 1.2, 1.3 são suficientes para sanity check** do DSelectK. Mostrar uma curva de sensibilidade no appendix.
- Outros eixos: opcional, vão para o journal.

### Para o paper de journal (Opção B):
- Todos os eixos. A tabela de sensibilidade é exatamente o tipo de conteúdo que diferencia conferência de journal.

---

## Output esperado

### Para a conferência (1 figura no appendix):
Curvas de sensibilidade do DSelectK (e, k, temp) e do batch size, mostrando que o champion é robusto.

### Para o journal (Section "Hyperparameter Sensitivity"):
Tabela completa com todos os eixos, std entre seeds, e identificação dos hiperparâmetros mais sensíveis.

---

## Riscos

1. **Risco baixo, alto valor:** se o backbone maior (Eixo 2) trouxer +0.02 joint, é um achado novo grátis. Vale a pena rodar.

2. **Risco médio:** se variar batch (Eixo 4.2) mudar significativamente o ranking entre otimizadores, é uma má notícia para o paper — significa que a tese "gradient surgery essential" é parcialmente batch-dependent. **Importante rodar.**

3. **Risco alto, baixo retorno:** ablar o número de camadas e dimensão do encoder por tarefa. Pode mexer em muita coisa sem ganho proporcional. Não fazer.

---

## Resumo

**Mínimo viável (BRACIS):** Eixo 1 + Eixo 4.2. ~3 h. Mostra robustez e ataca o batch-size confound da review.

**Completo (journal):** todos os eixos, ~7 h. Conteúdo de appendix técnico forte.

**Prioridade absoluta dentro deste plano:** Eixo 4.2 (batch size sensitivity), porque ataca diretamente um ponto da critical review.
