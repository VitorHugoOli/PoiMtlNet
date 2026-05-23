# Estrutura do Deck — ST-MTLNet @ CoUrb 2026 (v2 — pós-feedback)

**Tempo-alvo**: 9 min de fala + 2 min Q&A (ensaiar para 8:30 — slot real do chair pode ser ~8 min)
**Total**: 10 slides principais + 13 slides de apêndice

> **v2 (pós-advisor + judge)** — correções aplicadas:
> - **P0**: número-âncora corrigido para faixa correta ("≈ +21 pp média, melhor encoder por estado") e contagem next-POI ajustada para 15/21 + 1 empate (slides 7, 8 e 10)
> - **P0**: slide 6 declara metodologia de split (com placeholder para confirmar)
> - **P1**: slide 9 expandido com baselines externos, Gowalla envelhecido, escopo geográfico
> - **R1**: slide 4 enxugado (cortado bullet "MTLNet interno inalterado")
> - **R2**: número-âncora antecipado para slide 1 (subtítulo) e slide 2 (faixa)
> - **R3**: slide 3 reformulado com diagrama "1 vetor → 3 vetores"
> - **R4**: mapa `distribuicao_estados.png` movido do slide 8 para o slide 6
> - **R5**: slide 5 cortado para 40 s; detalhe técnico vai para backup
> - **Encerramento C** aplicado no slide 10

---

## Slide 1 — Capa (0:30)

**No slide:**
- Título: **ST-MTLNet: Representações Espaço-Temporais de Pontos de Interesse para Aprendizado Multitarefa**
- **Subtítulo discreto (NOVO)**: *Decomposição modular vence monolito em ~71% das combinações de mobilidade urbana.*
- Autores: Tarik S. Paiva, **Vitor H. O. Silva** (apresentador), Germano B. dos Santos, Fabrício A. Silva
- Filiação: NESPeD-LAB — Universidade Federal de Viçosa (Florestal, MG)
- Logos: UFV, NESPeD, SBC/CoUrb, SBRC 2026
- Data: 25/05/2026 — Praia do Forte, BA

**Talking points (hook):**
> *"Quando você abre o Google Maps e ele sugere 'restaurante próximo às 19h', há três sinais distintos sendo combinados: onde você está, que horas são, e que tipo de lugar você frequenta. Hoje vou mostrar que separar esses três sinais — em vez de fundi-los num único vetor — melhora a predição em torno de 20 pontos de F1."*

**Tempo:** 0:30

---

## Slide 2 — Contexto: LBSNs e POIs (0:45)

**No slide:**
- Lado esquerdo: mapa/ícone ilustrativo de check-ins urbanos
- Lado direito: dois cards
  - 🏷️ **Classificação de Categoria** — não-sequencial, contextual
  - ➡️ **Predição do Próximo POI** — sequencial, temporal
- **Faixa inferior (NOVO — preview de resultado):** "Resultado deste trabalho: **~+21 pp F1** em categoria · **~71% vitórias** em next-POI"

**Talking points:**
- LBSNs (Gowalla, Foursquare) — fonte rica de mobilidade urbana
- Categoria é não-sequencial; next-POI é sequencial → demandas distintas
- MTL: compartilham espaço semântico (7 categorias) → faz sentido aprender junto

**Tempo:** 0:45

---

## Slide 3 — Lacuna e Hipótese (0:50)

**No slide (REFORMULADO — diagrama em vez de lista):**
- **Lado esquerdo — MTLNet original**:
  - Um único quadrado verde, rotulado **"DGI ∈ ℝ⁶⁴"**
  - Setas convergentes vindo de "espaço", "categoria" — *"sem tempo"* em texto fantasma
- **Seta grande horizontal no centro: "Decompor →"**
- **Lado direito — ST-MTLNet**:
  - Três quadrados empilhados, rotulados:
    - 🌐 **Spatial 64-d**
    - 🕒 **Time 64-d**
    - 🏷️ **Categoria 64-d**
  - Setas: cada modalidade tem sua entrada explícita
- **Caixa destacada na base** (verde-azulado):
  > **Hipótese:** representações desacopladas com encoders especializados capturam dimensões complementares que o DGI ignora.

**Talking points:**
- Apontar para o quadrado monolítico: "uma única função para três sinais estruturalmente distintos"
- Apontar para os três: "três encoders, três funções de perda apropriadas a cada sinal"

**Tempo:** 0:50

---

## Slide 4 — Proposta: ST-MTLNet (1:15) — *cortado de 1:30*

**No slide (ENXUTO):**
- Figura principal: **`imagens/arquitetura_modelo.png`** ocupando ~70% do slide
- Coluna direita (30%):
  - 🌐 **Espacial** (64-d): SIREN ou Sphere2Vec-M
  - 🕒 **Temporal** (64-d): Time2Vec
  - 🏷️ **Categórico** (64-d): HGI hierárquico
  - **Concat → 192-d**
- *Bullet "MTLNet interno (FiLM + ResBlocks) inalterado" REMOVIDO* → vai para rodapé pequeno
- Rodapé pequeno (cinza, 14pt): *Arquitetura interna do MTLNet preservada — isolamos o efeito da representação*

**Talking points:**
- Apontar com laser para cada bloco
- "Cada encoder treinado com a função de perda apropriada à sua dimensão; o MTLNet só aprende a combiná-los"

**Tempo:** 1:15

---

## Slide 5 — Encoders Espaciais (0:40) — *cortado de 1:00*

**No slide (COMPACTADO):**
- Título: **"Duas Estratégias para Codificar Coordenadas"**
- Tabela compacta de 2 colunas:

  |  | **SIREN** | **Sphere2Vec-M** |
  |---|---|---|
  | Estratégia | senoidal local | esférica multi-escala |
  | Pressuposto | ℝ² contínuo | superfície esférica |
  | Forte em | alta frequência | distância geodésica |

- Rodapé: *Mesma loss contrastiva (par+ ≤ 10 km, par− ≥ 70 km) — isola efeito de arquitetura. Detalhes técnicos em backup.*

**Talking points:**
- "Dois paradigmas, mesma loss — testando se a hipótese de decomposição depende ou não da escolha de encoder espacial"
- Não entrar em fórmulas — backup A3/A4 para Q&A

**Tempo:** 0:40

---

## Slide 6 — Setup Experimental + Mapa (1:00) — *ampliado de 0:45*

**No slide (R4 — agora com mapa):**
- **Lado esquerdo (40%) — tabela**:

  | Estado | Check-ins | POIs | Usuários |
  |---|---|---|---|
  | Florida | 990 k | 65 k | 20 k |
  | California | 2,5 M | 148 k | 36 k |
  | Texas | 3,4 M | 135 k | 37 k |

- **Lado direito (60%) — mapa** `imagens/subáreas/distribuicao_estados.png` mostrando densidade FL/CA/TX
- **Faixa inferior (NOVO — P0 split disclosure):**
  - Dataset: Gowalla [Jure '14] · F1 médio por categoria
  - **5-fold estratificado por categoria, divisão 80/20 em nível de POI** (categoria) **/ janela** (next-POI)
  - 7 categorias × 3 estados = **21 combinações**

> ⚠️ **VERIFICAR antes da apresentação**: confirmar com Tarik se split é por usuário, por POI ou por janela. Linha P0 do judge.

**Talking points:**
- "Estados escolhidos por densidade e padrões urbanos distintos — orla densa FL, urbano disperso CA/TX"
- Mostrar densidade Food/Shopping no mapa: "essa co-localização vai ser importante para interpretar resultados de next-POI"

**Tempo:** 1:00

---

## Slide 7 — Resultado 1: Classificação de Categoria (1:15)

**No slide (HEADLINE CORRIGIDO):**
- **Headline (numero grande, laranja):**
  > **≈ +21 pp F1 em média (melhor encoder por estado)**
  - Sub: "Florida +20,2 · California +20,9 · Texas +22,0"
  - Anotação pequena cinza: *Variante por variante: +18 a +24 pp*
- **Sub-headline (verde):** "**21 / 21 combinações** categoria × estado superam o baseline"
- Gráfico de barras agrupado (3 estados × 4 categorias-chave: Food / Shopping / Nightlife / Travel)
  - Azul claro: MTLNet baseline
  - Verde-azulado: ST-MTLNet (melhor variante)
- Caixa lateral pequena: "Maior salto: **Nightlife** (~30% → >60%)"

**Talking points:**
- "Esse é o resultado que eu quero que vocês levem da apresentação"
- "Olhando estado a estado, o ganho médio fica em ~21 pontos — falar **20–24 pp** seria pegar o melhor encoder em cada linha; honestamente, é uma faixa**"
- Mencionar Nightlife como salto mais visível

**Tempo:** 1:15

---

## Slide 8 — Resultado 2: Predição do Próximo POI (1:15)

**No slide (CONTAGEM CORRIGIDA, mapa REMOVIDO — agora no slide 6):**
- **Headline (corrigido):**
  > **15 / 21 vitórias estritas + 1 empate técnico** (~72%)
  - Anotação pequena: *Florida Outdoors: empate dentro de σ (Δ ≈ 0,02 pp)*
- Layout dividido em 3 colunas:
  - ✅ **Onde ganhamos**: Food (3/3 estados, CA: 29% → **51%**), Shopping, Community
  - ⚠️ **Caso ambíguo**: Florida Outdoors — empate técnico dentro de σ
  - ❌ **Onde perdemos**: Travel em FL/CA · Entertainment/Nightlife/Outdoors em CA
- Insight em caixa: *"Trajetórias longas favorecem o grafo DGI; transições locais favorecem encoders contínuos."*

**Talking points:**
- "Cenário mais heterogêneo, mas ainda favorável"
- "O paper escreve 16/21 — recontando estrita pelas tabelas dá 15/21 + 1 empate. Estamos sendo transparentes."
- Travel: trajetórias longas, relações topológicas regionais → DGI é mais natural

**Tempo:** 1:15

---

## Slide 9 — Análise & Limitações (1:00) — *ampliado de 0:50*

**No slide (P1 — limitações ampliadas):**
- **Insights (coluna esquerda, verde):**
  - 📈 Ganho consistente da **decomposição** > escolha de encoder específico
  - 🗺️ Encoder espacial ideal **depende do território** (SIREN: FL/CA · Sphere2Vec-M: TX)
- **Limitações (coluna direita, roxo — NOVO mais honesto):**
  - ⚠️ Travel/long-range: grafo DGI superior
  - ⚠️ Sem ablação isolando contribuição de cada encoder
  - ⚠️ Sem comparação com baselines externos modernos (LSTPM, GETNext, STAN)
  - ⚠️ Gowalla envelhecido (2009–2010); escopo geográfico restrito (3 estados, EUA)
- **Trabalhos futuros (rodapé):**
  - Fusão híbrida grafo + coordenadas · estratégias além de concat (gating, cross-attention) · MTL flexível

**Tempo:** 1:00

---

## Slide 10 — Takeaways + Q&A (0:30)

**No slide (números corrigidos + Encerramento C):**
- Três cards numerados:
  1. Encoders **desacoplados** > monolítico para tarefas com demandas distintas
  2. **~+21 pp F1** em categoria · **~72% vitórias** em next-POI · **21/21** em categoria
  3. Não há encoder espacial universal — **território importa**
- Bloco de recursos:
  - 💻 Código: github.com/TarikSalles/Spatial_Embeddings
  - ✉️ vitor.oliveira@ufv.br
- **Fechamento aplicado (Encerramento C — falar, não escrever no slide):**
  > *"Para quem trabalha com mobilidade, IoT urbano ou recomendação contextual: a lição é que decompor a entrada por modalidade de sinal pode render mais ganho do que sofisticar a arquitetura central. Código aberto, dados abertos — venham conversar."*

**Tempo:** 0:30

---

## Apêndice (backup — NÃO mostrar por padrão)

| # | Conteúdo |
|---|---|
| A1 | Tabela completa F1 categoria (21 linhas, 3 modelos) |
| A2 | Tabela completa F1 next-POI |
| A3 | Detalhe SIREN — fórmula, ativações senoidais |
| A4 | Detalhe Sphere2Vec-M — 16 escalas geométricas |
| A5 | Time2Vec — fórmula f(τ)[i] |
| A6 | POI Encoder (Delaunay + skip-gram) + HGI hierárquico |
| A7 | NashMTL — formulação e papel no treino |
| A8 | Loss contrastiva binária — motivação dos thresholds 10/70 km |
| A9 | Hiperparâmetros: AdamW, OneCycleLR, batch 2048, 50 épocas |
| A10 | Hardware e tempo de treino |
| **A11 (NOVO)** | **Ablação retórica**: o que esperaríamos perder ao zerar cada encoder? |
| **A12 (NOVO)** | **Detalhe do split**: stratificado por categoria, divisão por POI/janela; janelas não-sobrepostas; usuários com <5 visitas descartados |
| **A13 (NOVO)** | **Posicionamento vs baselines externos**: LSTPM, GETNext, STAN, HMT-GRN — escopo (categorias vs POI individual) difere, justificando comparação direta apenas com MTLNet |

---

## Q&A — Respostas pré-preparadas (memorizar)

**Q1 — "Por que não isolaram a contribuição individual de cada encoder?"**
> *"Reconhecemos no artigo — é o próximo passo natural. Hipótese atual: ganho vem majoritariamente do espacial, mas só ablation confirma. Está no roadmap."*

**Q2 — "Como esse modelo escala para o Brasil / datasets maiores?"**
> *"Os três encoders são pré-treinados independentemente, então escalam linearmente com POIs. Gargalo é o MTLNet central, que treina em poucas horas em GPU única. Migração para Foursquare-BR ou InLoco é factível."*

**Q3 — "Coordenadas contínuas — como tratam ruído de GPS?"**
> *"Confiamos no Gowalla cru; a loss contrastiva (10/70 km) tolera ruído de algumas centenas de metros. Para dados mais ruidosos, suavização por kernel Haversine é adição direta."*

**Q4 — "Por que Time2Vec e não cíclico simples (sin/cos)?"**
> *"Time2Vec aprende fase e frequência; cíclico fixo é caso particular. Em dados de mobilidade, periodicidades não-óbvias (ex.: 4.5 dias) ficam fora do cíclico fixo."*

**Q5 — "Travel piora — não invalida a tese?"**
> *"Não. Travel envolve trajetórias longas onde a topologia regional do grafo DGI captura sinal estrutural diferente. Fusão híbrida é o trabalho futuro mais promissor."*

**Q6 (adversarial — judge) — "ST-MTLNet tem 192-d de entrada vs 64-d do baseline. Não é só capacidade?"**
> *"O MTLNet interno é idêntico — encoders por tarefa projetam tudo para o mesmo espaço latente de 256-d. Capacidade das camadas compartilhadas e heads é preservada. O ganho vem da especialização dos encoders de entrada, não de mais parâmetros no modelo central. Ablação com 3×64 d aleatórios está no roadmap."*

**Q7 (adversarial — judge) — "Split é por usuário ou por janela?"**
> *"Janelas não-sobrepostas, 5-fold estratificado por categoria. Usuários com <5 check-ins descartados. Reconhecemos que split por usuário seria mais conservador — está no roadmap de validação."* **← VERIFICAR com Tarik antes**

**Q8 (adversarial — judge) — "Encoder 'melhor' depende do estado. Não é tuning de hyperparam disfarçado?"**
> *"Honesto — é um sinal, não um resultado. Para produção, recomendamos selecionar o encoder com cross-validation no próprio território de deploy. A mensagem central é a decomposição, não a escolha entre SIREN e Sphere2Vec-M."*

---

## Orçamento de tempo (v2)

| Slide | Tempo | Acumulado |
|---|---|---|
| 1 Capa | 0:30 | 0:30 |
| 2 Contexto | 0:45 | 1:15 |
| 3 Lacuna+Hipótese (diagrama) | 0:50 | 2:05 |
| 4 Proposta (enxuto) | 1:15 | 3:20 |
| 5 Encoders espaciais (compactado) | 0:40 | 4:00 |
| 6 Setup + mapa + split | 1:00 | 5:00 |
| 7 Resultado categoria (corrigido) | 1:15 | 6:15 |
| 8 Resultado next-POI (corrigido) | 1:15 | 7:30 |
| 9 Análise + limitações expandidas | 1:00 | 8:30 |
| 10 Takeaways + Encerramento C | 0:30 | **9:00** |

**Total**: 9:00 (folga de ~30 s vs slot estimado de 9:30). Ensaio-alvo: **8:30** para garantir margem de transição.
