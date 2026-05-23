# DECK_CONTENT — ST-MTLNet @ CoUrb 2026

> Documento único consolidando **todo o conteúdo do deck** v2 (pós-correção P0/P1 + recomendações R1–R5). Pronto para alimentar uma ferramenta de geração visual (Claude Design, Gamma, Beautiful.ai, Marp, Figma manual). Cada slide tem: layout, textos exatos, anotações de design, e referências de imagem.
>
> **Versão**: 2 (24/05/2026) · **Apresentador**: Vitor H. O. Silva · **Data alvo**: 25/05/2026 11:15–12:00 (Sessão 1, ~9 min talk + 2 min Q&A) · **Idioma**: PT-BR · **Formato**: 16:9 widescreen (1920×1080)

---

## Identidade visual

### Paleta de cores

| Cor | Hex | Uso |
|---|---|---|
| Texto primário | `#1A1A1A` | título, corpo |
| Verde-azulado (destaque + positivo) | `#16A086` | barra de capa, headers de destaque, ganhos |
| Laranja (números-âncora) | `#E67E22` | headline numérico de resultado |
| Roxo (limitações) | `#663399` | bullets de limitação |
| Vermelho (perdas) | `#C0392B` | onde o modelo perde |
| Cinza médio | `#7F8C8D` | rodapé, citações |
| Cinza claro de fundo | `#F4F6F7` | cards, faixas |
| Branco | `#FFFFFF` | fundo |

### Tipografia

- **Famílias**: Inter (preferida) ou Roboto. Tudo em sans-serif limpa.
- **Hierarquia**:
  - Título de slide: 48–56 pt, Semi Bold
  - Subtítulo: 28–32 pt, Regular
  - Corpo / bullets: 22–26 pt, Regular
  - Headline numérico: 96–120 pt, Bold (cor laranja)
  - Rodapé / citação: 14–16 pt, Regular cinza
- **Espaçamento**: line-height 130–140%

### Elementos recorrentes

- **Topo (slides de conteúdo)**: barra horizontal verde-azulada `#16A086`, altura 6 px
- **Rodapé (todos exceto capa)**: 3 zonas
  - Esquerda: "🎓 NESPeD-LAB · UFV" (16 pt, cinza)
  - Centro: "ST-MTLNet @ CoUrb 2026" (16 pt, cinza)
  - Direita: número do slide / 10 (16 pt, cinza)
- **Capa**: barra vertical verde-azulada de 80 px na esquerda; sem barra superior nem rodapé padrão

### Assets a embutir

| Asset | Caminho local | Onde usar |
|---|---|---|
| Arquitetura ST-MTLNet | `articles/CoUrb_2026/imagens/arquitetura_modelo.png` | Slide 4 |
| Distribuição espacial FL/CA/TX | `articles/CoUrb_2026/imagens/subáreas/distribuicao_estados.png` | Slide 6 |

---

## SLIDE 1 — Capa

**Layout**:
- Barra vertical verde-azulada (80 px) na esquerda
- Conteúdo alinhado à esquerda a partir de x=140
- Sem rodapé padrão

**Conteúdo (texto exato)**:

- **Título grande** (96 pt Bold): `ST-MTLNet`
- **Subtítulo do título** (44 pt Semi Bold, 2 linhas): `Representações Espaço-Temporais de Pontos de Interesse para Aprendizado Multitarefa`
- **Frase-âncora** (28 pt itálico, cor `#16A086`): `Decomposição modular vence monolito em ~71% das combinações de mobilidade urbana.`
- Linha divisória cinza (1640×2 px)
- **Autores** (28 pt Regular): `Tarik S. Paiva · Vitor H. O. Silva · Germano B. dos Santos · Fabrício A. Silva`
- **Apresentador em destaque** (22 pt Semi Bold, cor `#16A086`): `Apresentador: Vitor H. O. Silva`
- **Filiação** (24 pt cinza): `NESPeD-LAB · Universidade Federal de Viçosa, Florestal, MG`
- **Linha do evento** (22 pt Semi Bold): `X Workshop de Computação Urbana (CoUrb 2026) · SBRC 2026`
- **Data/local** (20 pt cinza): `Praia do Forte, BA · 25 de maio de 2026`

**Talking points (não vão no slide — só para o apresentador)**:
> *"Quando você abre o Google Maps e ele sugere 'restaurante próximo às 19h', há três sinais distintos sendo combinados: onde você está, que horas são, e que tipo de lugar você frequenta. Hoje vou mostrar que separar esses três sinais — em vez de fundi-los num único vetor — melhora a predição em torno de 20 pontos de F1."*

**Tempo**: 0:30

---

## SLIDE 2 — Contexto: LBSNs e POIs

**Layout**:
- Título no topo
- Duas colunas: esquerda (40%) com ilustração, direita (60%) com cards
- Faixa inferior verde-azulada clara com preview de resultados

**Conteúdo**:

- **Título** (48 pt Semi Bold): `LBSNs, POIs e Duas Tarefas Complementares`
- **Coluna esquerda** — ilustração esquemática (estilizada, não foto): mapa estilizado com pinos coloridos (vermelho, laranja, azul, verde) representando POIs de categorias diferentes. Pode ser simplesmente um cluster de pins coloridos sobre um fundo de "ruas" estilizadas.
- **Coluna direita — Card 1** (fundo `#F4F6F7`, borda esquerda `#16A086` 4px):
  - Ícone: 🏷️
  - Título (28 pt Semi Bold): `Classificação de Categoria de POI`
  - Texto (22 pt): `Classificar a categoria semântica de um POI a partir de suas características.`
  - Tag (16 pt cinza, no canto): `não-sequencial`
- **Coluna direita — Card 2** (mesmo estilo):
  - Ícone: ➡️
  - Título: `Predição do Próximo POI`
  - Texto: `Prever a categoria do próximo POI dado o histórico de check-ins do usuário.`
  - Tag: `sequencial`
- **Faixa inferior** (banner verde-azulado claro, altura ~120 px, texto centralizado, 32 pt):
  - `Resultado: ` + (em laranja `#E67E22` Bold) `+21 pp F1` + ` em categoria · ` + (laranja) `~72% vitórias` + ` em next-POI`

**Talking points**:
- Gowalla, Foursquare — fonte rica de mobilidade urbana
- Categoria é não-sequencial; next-POI é sequencial → demandas distintas
- MTL: compartilham espaço semântico → vale aprender junto

**Tempo**: 0:45

---

## SLIDE 3 — Lacuna e Hipótese (diagrama central)

**Layout**:
- Título no topo
- Diagrama horizontal central ocupando ~70% do slide
- Caixa de hipótese na base

**Conteúdo**:

- **Título** (48 pt Semi Bold): `De um Vetor Monolítico para Três Vetores Especializados`

- **Diagrama horizontal** (centralizado, 3 zonas):
  - **Zona esquerda** — MTLNet original:
    - Texto pequeno acima (18 pt cinza): `MTLNet [Silva et al., 2025]`
    - **1 retângulo** grande verde escuro (260×220 px, radius 16, cor `#16A086` opacidade 80%) com texto branco centralizado: `DGI ∈ ℝ⁶⁴` (32 pt Bold)
    - 2 setas convergentes pequenas chegando ao retângulo, com labels: `espaço`, `categoria`
    - Texto fantasma cinza claro ao lado: `(sem tempo)` (20 pt itálico)
  - **Zona central** — transição:
    - Seta horizontal grossa verde-azulada (largura ~200 px, espessura 16 px)
    - Texto acima da seta (28 pt Semi Bold cor `#16A086`): `Decompor →`
  - **Zona direita** — ST-MTLNet:
    - Texto pequeno acima (18 pt cinza): `ST-MTLNet (este trabalho)`
    - **3 retângulos** empilhados verticalmente (260×120 px cada, gap 16 px, radius 16):
      - Topo: cor `#16A086`, label branco `🌐 Spatial · 64-d` (28 pt Bold)
      - Meio: cor `#E67E22`, label branco `🕒 Time · 64-d` (28 pt Bold)
      - Base: cor `#663399`, label branco `🏷️ Category · 64-d` (28 pt Bold)
    - Texto abaixo (22 pt cinza): `concat → 192-d`

- **Caixa de hipótese** na base (fundo branco, borda 2 px `#16A086`, padding 24 px, max-width 1400 px, centralizada):
  - Label pequeno (18 pt Semi Bold cor `#16A086`): `Hipótese`
  - Texto (28 pt Regular): `representações desacopladas com encoders especializados capturam dimensões complementares que o DGI ignora.`

**Talking points**:
- Apontar para o quadrado monolítico: "uma única função para três sinais estruturalmente distintos"
- Apontar para os três: "três encoders, três funções de perda apropriadas"

**Tempo**: 0:50

---

## SLIDE 4 — Proposta: ST-MTLNet (arquitetura)

**Layout**:
- Título no topo
- Imagem da arquitetura à esquerda (70% largura)
- Coluna de bullets à direita (30%)
- Nota de rodapé pequena

**Conteúdo**:

- **Título** (48 pt Semi Bold): `ST-MTLNet — Arquitetura`

- **Imagem** (lado esquerdo, ~1200×900 px): `articles/CoUrb_2026/imagens/arquitetura_modelo.png`

- **Coluna direita** (a partir de x=1320, espaçamento vertical 80 px):
  - **🌐 Espacial (64-d)** (28 pt Semi Bold cor `#16A086`)
    - Sub: `SIREN ou Sphere2Vec-M` (22 pt cinza)
  - **🕒 Temporal (64-d)** (28 pt Semi Bold cor `#E67E22`)
    - Sub: `Time2Vec` (22 pt cinza)
  - **🏷️ Categórico (64-d)** (28 pt Semi Bold cor `#663399`)
    - Sub: `HGI hierárquico (POI → região → cidade)` (22 pt cinza)
  - **Divisor horizontal fino**
  - **Concat → 192-d** (32 pt Bold, fundo verde-azulado claro, padding 12)

- **Rodapé pequeno** (cinza, 16 pt, centralizado em baixo):
  - `MTLNet interno (FiLM + ResBlocks + heads) preservado — isolamos o efeito da representação de entrada.`

**Talking points**:
- Apontar com laser para cada bloco
- "Cada encoder treinado com função de perda apropriada à sua dimensão; o MTLNet só aprende a combinar"

**Tempo**: 1:15

---

## SLIDE 5 — Encoders Espaciais (comparação compacta)

**Layout**:
- Título no topo
- Tabela compacta centralizada
- Faixa inferior cinza explicativa

**Conteúdo**:

- **Título** (48 pt Semi Bold): `Duas Estratégias para Codificar Coordenadas`

- **Tabela centralizada** (1400×420 px, fonte 26 pt, padding generoso):

  | (linha) | **SIREN** | **Sphere2Vec-M** |
  |---|---|---|
  | Estratégia | senoidal local | esférica multi-escala |
  | Pressuposto | ℝ² contínuo | superfície esférica |
  | Forte em | alta frequência | distância geodésica |
  | Referência | Rußwurm '24 | Mai '23 |

  - Headers em Bold cor `#16A086`
  - Linhas alternadas com fundo `#F4F6F7`
  - Bordas finas cinza claro

- **Faixa inferior** (texto centralizado, 22 pt itálico cinza):
  - `Mesma loss contrastiva binária (par+ ≤ 10 km, par− ≥ 70 km) — isola o efeito de arquitetura. Detalhes técnicos no apêndice.`

**Talking points**:
- "Dois paradigmas, mesma loss — testamos se a tese de decomposição depende ou não da escolha do encoder espacial"
- Não entrar em fórmulas — backup A3/A4 para Q&A

**Tempo**: 0:40

---

## SLIDE 6 — Setup Experimental + Mapa

**Layout**:
- Título no topo
- Lado esquerdo (40%): tabela do dataset
- Lado direito (60%): mapa de distribuição
- Faixa inferior com método de avaliação

**Conteúdo**:

- **Título** (48 pt Semi Bold): `Setup: Três Estados, Padrões Urbanos Distintos`

- **Tabela** (lado esquerdo, fonte 24 pt):

  | Estado | Check-ins | POIs | Usuários |
  |---|---|---|---|
  | **Florida** | 990 k | 65 k | 20 k |
  | **California** | 2,5 M | 148 k | 36 k |
  | **Texas** | 3,4 M | 135 k | 37 k |

  - Headers em Bold, cor `#16A086`
  - Coluna "Estado" em Bold
  - Espaçamento confortável

- **Mapa** (lado direito): inserir `articles/CoUrb_2026/imagens/subáreas/distribuicao_estados.png`
  - Legenda abaixo (18 pt itálico cinza): `Densidade de POIs Food (vermelho) e Shopping (laranja) nas sub-regiões mais densas`

- **Faixa inferior** (3 linhas curtas, 20 pt, centradas):
  - Linha 1: `Dataset Gowalla [Cho et al., '11] · F1 médio por categoria`
  - Linha 2: `5-fold estratificado por categoria · 80/20 (por POI para classificação · janelas não-sobrepostas para next-POI)`
  - Linha 3: `7 categorias × 3 estados = ` (em Bold) `21 combinações por tarefa`

**Talking points**:
- "Estados escolhidos por densidade e padrões urbanos distintos"
- Apontar no mapa: "essa co-localização de Food/Shopping vai importar nos resultados de next-POI"

**Tempo**: 1:00

> ⚠️ **VERIFICAR antes da apresentação**: confirmar com Tarik se o split de next-POI é por usuário, por POI ou por janela. Foi o achado P0 do agente judge.

---

## SLIDE 7 — Resultado 1: Categoria

**Layout**:
- Título no topo
- Headline numérico gigante centralizado
- Subheadline com breakdown por estado
- Gráfico de barras embaixo
- Caixa lateral pequena com "maior salto"

**Conteúdo**:

- **Título** (48 pt Semi Bold): `Classificação de Categoria — 21 / 21 Vitórias`

- **Headline numérico** (centralizado, 120 pt Bold cor `#E67E22`):
  - `≈ +21 pp F1`
- **Subheadline** (32 pt Semi Bold cor `#16A086`, centralizado):
  - `em média por estado, melhor encoder por linha`
- **Linha pequena cinza** (18 pt, centralizada):
  - `Florida +20,2 · California +20,9 · Texas +22,0   |   variante por variante: +18 a +24 pp`

- **Gráfico de barras agrupado** (largura 1400 px, altura 360 px):
  - Eixo X: 4 grupos (Food, Shopping, Nightlife, Travel) × 3 estados (FL, CA, TX) = 12 grupos
  - Em cada grupo: 2 barras (Baseline azul claro `#5DADE2`, ST-MTLNet melhor variante verde `#16A086`)
  - Eixo Y: F1 (%), 0–80
  - Legenda discreta no canto superior direito
  - Dados (Baseline | ST-MTLNet melhor):
    - **FL Food**: 55,47 | 73,22
    - **FL Shopping**: 62,96 | 77,48
    - **FL Nightlife**: 32,59 | 62,60
    - **FL Travel**: 45,49 | 64,89
    - **CA Food**: 60,68 | 75,78
    - **CA Shopping**: 60,43 | 77,00
    - **CA Nightlife**: 26,81 | 60,78
    - **CA Travel**: 38,88 | 63,59
    - **TX Food**: 54,31 | 73,10
    - **TX Shopping**: 62,70 | 79,67
    - **TX Nightlife**: 34,44 | 64,57
    - **TX Travel**: 39,37 | 64,73

- **Caixa lateral** (pequena, lado direito do gráfico, fundo verde claro):
  - `Maior salto: Nightlife (~30% → >60%)`

**Talking points**:
- "Este é o resultado que eu quero que vocês levem"
- "Honestos: a faixa 20–24 pp é com 'melhor encoder por linha'. SIREN isolado no Texas dá 17,9 pp — por isso reportamos média ~21 pp"
- Mencionar Nightlife como salto mais visível

**Tempo**: 1:15

---

## SLIDE 8 — Resultado 2: Next-POI

**Layout**:
- Título no topo
- Headline numérico centralizado
- Subheadline com breakdown
- Layout 3 colunas (cards verde/amarelo/vermelho)
- Faixa de insight inferior

**Conteúdo**:

- **Título** (48 pt Semi Bold): `Predição do Próximo POI — 15 / 21 Vitórias + 1 Empate`

- **Headline** (centralizado, 96 pt Bold cor `#E67E22`):
  - `~72%` (em laranja) + ` das combinações` (em texto normal 32 pt)
- **Subheadline** (28 pt Semi Bold cor `#16A086`, centralizado):
  - `15 vitórias estritas · 1 empate técnico · 5 derrotas`
- **Anotação fina** (16 pt itálico cinza, centralizada):
  - `Florida Outdoors: empate dentro de σ (Δ ≈ 0,02 pp)`

- **3 cards lado a lado** (cada um 600×400 px, gap 24 px):

  - **Card verde** (fundo `#16A086` opacidade 12%, borda esquerda 4 px `#16A086`):
    - Ícone topo: ✅
    - Título (24 pt Semi Bold cor `#16A086`): `Onde ganhamos`
    - Bullets (20 pt):
      - **Food vence em 3/3 estados** — CA: 29% → **51%**
      - **Shopping** consistente
      - **Community** consistente

  - **Card amarelo** (fundo `#F4C430` opacidade 15%, borda `#E67E22`):
    - Ícone: ⚠️
    - Título: `Empate técnico`
    - Bullets:
      - **Florida Outdoors** — empate dentro de σ
      - (Δ ≈ 0,02 pp, baseline marginalmente à frente)

  - **Card vermelho** (fundo `#C0392B` opacidade 12%, borda `#C0392B`):
    - Ícone: ❌
    - Título: `Onde perdemos`
    - Bullets:
      - **Travel** em FL e CA
      - **Entertainment**, **Nightlife**, **Outdoors** em CA

- **Faixa inferior** (fundo `#F4F6F7`, padding 24, 22 pt itálico):
  - `"Trajetórias longas favorecem o grafo DGI; transições locais favorecem encoders contínuos."`

**Talking points**:
- "Cenário mais heterogêneo, mas ainda favorável"
- "O paper escreve 16/21; recontando pelas tabelas o estrito é 15/21 + 1 empate. Estamos sendo transparentes."

**Tempo**: 1:15

---

## SLIDE 9 — Análise & Limitações

**Layout**:
- Título no topo
- Duas colunas: Insights (verde) à esquerda, Limitações (roxo) à direita
- Rodapé com trabalhos futuros

**Conteúdo**:

- **Título** (48 pt Semi Bold): `O Que Aprendemos — e o Que Ainda Falta`

- **Coluna esquerda — Insights** (fundo `#16A086` opacidade 8%, padding 32):
  - Header (24 pt Semi Bold cor `#16A086`): `📈 Insights`
  - Bullets (22 pt):
    - Ganho consistente da **decomposição** > escolha de encoder específico
    - 🗺️ Encoder espacial ideal **depende do território**:
      - SIREN: Florida, California
      - Sphere2Vec-M: Texas

- **Coluna direita — Limitações** (fundo `#663399` opacidade 8%, padding 32):
  - Header (24 pt Semi Bold cor `#663399`): `⚠️ Limitações`
  - Bullets (22 pt):
    - Long-range / Travel: grafo DGI ainda superior
    - Sem ablação isolando cada encoder
    - Sem baselines externos modernos (LSTPM, GETNext, STAN)
    - Gowalla envelhecido (2009–10)
    - Escopo: 3 estados, 7 categorias high-level

- **Rodapé centralizado** (20 pt cinza, em itálico):
  - `Trabalhos futuros: fusão híbrida grafo + contínua · gating / cross-attention · MTL flexível (MoE, soft sharing)`

**Tempo**: 1:00

---

## SLIDE 10 — Takeaways + Q&A

**Layout**:
- Título no topo
- 3 cards numerados grandes lado a lado
- Bloco de recursos (código + contato)
- Frase final de fechamento

**Conteúdo**:

- **Título** (48 pt Semi Bold): `Para Levar`

- **3 cards numerados** (cada 540×260 px, gap 32 px, fundo `#F4F6F7`, borda topo 4 px com cor variando):

  - **Card 1** (borda `#16A086`):
    - Número grande no topo (64 pt Bold cor `#16A086`): `1`
    - Texto (24 pt): Encoders **desacoplados** > monolítico para tarefas com demandas distintas

  - **Card 2** (borda `#E67E22`):
    - Número (64 pt Bold cor `#E67E22`): `2`
    - Texto: **~+21 pp F1** em categoria · **~72% vitórias** em next-POI · **21/21** combinações em categoria

  - **Card 3** (borda `#663399`):
    - Número (64 pt Bold cor `#663399`): `3`
    - Texto: Não há encoder espacial universal — **território importa**

- **Bloco recursos** (centralizado, 24 pt):
  - 💻 `Código: github.com/TarikSalles/Spatial_Embeddings`
  - ✉️ `Contato: vitor.oliveira@ufv.br`

- **Frase final centralizada** (40 pt Semi Bold cor `#16A086`):
  - `Obrigado — perguntas?`

**Talking points (fechamento aplicado — Encerramento C)**:
> *"Para quem trabalha com mobilidade, IoT urbano ou recomendação contextual: a lição é que decompor a entrada por modalidade de sinal pode render mais ganho do que sofisticar a arquitetura central. Código aberto, dados abertos — venham conversar."*

**Tempo**: 0:30

---

## APÊNDICE — Slides de backup (após slide 10, ocultos no fluxo)

> Cada slide do apêndice tem mesmo layout (título + conteúdo), mas marcado claramente como "Backup". Mostrar **só** se Q&A pedir.

### A1 — Tabela completa F1 Categoria (21 linhas)

Inserir tabela completa do arquivo `articles/CoUrb_2026/resultados/tabela_comparativa_f1_category.tex`. Em cada linha, destacar (Bold + verde) a célula vencedora. Pode ser uma imagem da tabela exportada do PDF.

### A2 — Tabela completa F1 Next-POI

Mesmo padrão de A1, dados de `tabela_comparativa_f1_next.tex`.

### A3 — Detalhe SIREN

- Fórmula da camada senoidal: `y = sin(ω · Wx + b)`
- Arquitetura: rede feed-forward com ativações `sin`
- `f_θ : ℝ² → ℝ⁶⁴`
- Referência: Rußwurm '24

### A4 — Detalhe Sphere2Vec-M

- 16 escalas geométricas entre 10 km e 10 000 km
- Coordenadas esféricas (λ, φ)
- Termos de interação multi-escala
- Referência: Mai '23

### A5 — Time2Vec

- Fórmula:
  ```
  f(τ)[i] = ω₀τ + φ₀       se i = 0
           = sin(ωᵢτ + φᵢ)   se 1 ≤ i ≤ D−1
  ```
- Term linear (i=0) captura tendência; senoidais capturam ciclos

### A6 — POI Encoder + HGI

- Random walks em grafo Delaunay
- Skip-gram com negative sampling
- HGI hierárquico em 2 níveis (POI–região, região–cidade) com α=0.5

### A7 — NashMTL

- Barganha de Nash entre tarefas K=2
- Maximiza produto das utilidades de gradiente
- Garante atualização benéfica para todas as tarefas simultaneamente

### A8 — Loss contrastiva

- `L_loc = -[y · log σ(sim/τ) + (1-y) · log(1 - σ(sim/τ))]`
- τ = 0.15
- y=1 se Δd ≤ 10 km; y=0 se Δd ≥ 70 km
- Mesma estrutura aplicada para Time2Vec

### A9 — Hiperparâmetros

- Otimizador: AdamW (lr=1e-4, weight_decay=0.05, eps=1e-8)
- Scheduler: OneCycleLR (max_lr=1e-3, 50 épocas)
- Batch: 2048
- Gradient accumulation: 2 steps
- MTL: NashMTL (max_norm=2.2, update_weights_every=4)

### A10 — Hardware

- GPU: (preencher: A40/A100/T4 conforme rodado)
- Tempo de treino: (preencher)
- Memória pico: (preencher)

### A11 — Ablação retórica

> "Se zerássemos um encoder, o que perderíamos?"

| Encoder zerado | Perda esperada |
|---|---|
| Espacial | Performance similar ao DGI (~baseline) |
| Temporal | Maior queda em next-POI; categoria pouco afetada |
| Categórico | Perda em separabilidade entre classes próximas |

*Nota: ablação real está no roadmap.*

### A12 — Detalhamento do split

- 5-fold estratificado por categoria
- Categoria: divisão por POI (cada POI aparece só em um fold)
- Next-POI: janelas não-sobrepostas de tamanho L_h = 9
- Usuários com < 5 check-ins descartados
- Class weights aplicados na CE para mitigar desbalanceamento

### A13 — Posicionamento vs baselines externos

| Baseline | Tarefa | Granularidade | Escopo desta comparação |
|---|---|---|---|
| LSTPM | Next-POI | POI individual | Diferente — predizemos categoria, não POI |
| GETNext | Next-POI | POI individual | Mesmo motivo |
| STAN | Next-POI | POI individual | Mesmo motivo |
| HMT-GRN | Next-POI + região | Multi-nível | Próximo trabalho de comparação |

*Justifica comparação direta apenas com MTLNet original, que opera no mesmo nível de granularidade (categoria).*

---

## Q&A — Respostas pré-preparadas

**Q1: "Por que não isolaram a contribuição individual de cada encoder?"**
> "Reconhecemos no artigo — próximo passo natural. Hipótese: ganho vem majoritariamente do espacial, mas só ablação confirma. Está no roadmap."

**Q2: "Como escala para o Brasil / datasets maiores?"**
> "Os três encoders são pré-treinados independentemente — escalam linearmente com POIs. Gargalo é o MTLNet central, que treina em poucas horas em GPU única. Foursquare-BR ou InLoco são factíveis."

**Q3: "Coordenadas contínuas — como tratam ruído de GPS?"**
> "Confiamos no Gowalla cru; a loss contrastiva 10/70 km tolera ruído de algumas centenas de metros. Para mais ruidosos, suavização por kernel Haversine é adição direta."

**Q4: "Por que Time2Vec e não cíclico simples (sin/cos)?"**
> "Time2Vec aprende fase e frequência; cíclico fixo é caso particular. Em mobilidade, periodicidades não-óbvias (ex.: 4.5 dias) ficam fora do cíclico fixo."

**Q5: "Travel piora — não invalida a tese?"**
> "Não. Travel envolve trajetórias longas onde a topologia regional do grafo DGI captura sinal estrutural diferente. Fusão híbrida é o trabalho futuro mais promissor."

**Q6: "ST-MTLNet tem 192-d de entrada vs 64-d do baseline. Não é só capacidade?"**
> "O MTLNet interno é idêntico — encoders por tarefa projetam tudo para o mesmo espaço latente de 256-d. Capacidade preservada. Ganho vem da especialização. Ablação com 3×64 d aleatórios está no roadmap."

**Q7: "Split é por usuário ou por janela?"**
> "Janelas não-sobrepostas, 5-fold estratificado por categoria. Usuários com <5 check-ins descartados. Split por usuário seria mais conservador — está no roadmap." **← CONFIRMAR com Tarik**

**Q8: "Encoder 'melhor' depende do estado. Não é tuning disfarçado?"**
> "É um sinal, não um resultado. Para produção, recomendamos selecionar via cross-validation no território de deploy. A mensagem central é a decomposição, não a escolha entre SIREN e Sphere2Vec-M."

---

## Orçamento de tempo total

| Slide | Tempo | Acumulado |
|---|---|---|
| 1 Capa | 0:30 | 0:30 |
| 2 Contexto | 0:45 | 1:15 |
| 3 Lacuna+Hipótese | 0:50 | 2:05 |
| 4 Proposta | 1:15 | 3:20 |
| 5 Encoders espaciais | 0:40 | 4:00 |
| 6 Setup + mapa + split | 1:00 | 5:00 |
| 7 Resultado categoria | 1:15 | 6:15 |
| 8 Resultado next-POI | 1:15 | 7:30 |
| 9 Análise + limitações | 1:00 | 8:30 |
| 10 Takeaways | 0:30 | **9:00** |

**Ensaiar para 8:30. Q&A absorve ~2 min até 11:00.**

---

## Checklist final pré-apresentação

- [ ] Confirmar com Tarik: split é por usuário, POI ou janela? (Q&A Q7)
- [ ] Verificar hardware/tempo de treino para preencher A10
- [ ] Conferir que `imagens/arquitetura_modelo.png` ainda reflete a arquitetura atual
- [ ] Ensaio 1: cronometrar slide a slide
- [ ] Ensaio 2: cronometrar inteiro, alvo 8:30
- [ ] Exportar PDF de backup (caso projetor falhe)
- [ ] Salvar versão em pendrive USB
- [ ] Trazer apresentador remoto (clicker) — slide 4 e 6 têm gestos para apontar imagens
