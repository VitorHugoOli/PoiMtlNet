# Prompt para Figma MCP — v2 (pós-correção P0/P1/R1-R5)

Cole este prompt em uma sessão Claude com o MCP **plugin:figma:figma** ativo. Carregue a skill `figma-generate-design` antes (mandatória pelo servidor).

---

## Briefing

Crie um deck de apresentação acadêmica em **português brasileiro**, formato **16:9 widescreen**, com **10 slides principais + 13 slides de apêndice**, para apresentação oral de 9 minutos no workshop **CoUrb 2026** (X Workshop de Computação Urbana, SBRC 2026, Praia do Forte, Bahia, 25/05/2026).

### Identidade visual

- **Estilo**: acadêmico minimalista, profissional, sóbrio
- **Paleta**:
  - Texto primário: `#1A1A1A`
  - Verde-azulado (destaque positivo): `#16A086`
  - Laranja (números-âncora): `#E67E22`
  - Roxo (limitações/avisos): `#663399`
  - Vermelho (perdas): `#C0392B`
  - Cinza médio (rodapé/citações): `#7F8C8D`
  - Fundo: `#FFFFFF`
- **Tipografia**: Inter ou Roboto
  - Títulos: 36–40 pt, peso 600
  - Corpo: 22–24 pt, peso 400
  - Citações/rodapé: 14 pt, peso 400
- **Rodapé** (todos exceto capa): logo NESPeD/UFV (esq.) + "ST-MTLNet @ CoUrb 2026" + nº do slide
- **Topo**: barra fina horizontal verde-azulada (~3 px) nos slides de conteúdo

### Assets para upload

- `articles/CoUrb_2026/imagens/arquitetura_modelo.png` (slide 4)
- `articles/CoUrb_2026/imagens/subáreas/distribuicao_estados.png` (slide 6)

---

## Slide 1 — Capa

- Fundo branco, barra verde-azulada vertical na esquerda (~80 px)
- **Título grande** (centralizado): "ST-MTLNet: Representações Espaço-Temporais de Pontos de Interesse para Aprendizado Multitarefa"
- **Subtítulo discreto, itálico, verde-azulado**: *"Decomposição modular vence monolito em ~71% das combinações de mobilidade urbana."*
- Linha divisória fina
- **Autores**: Tarik S. Paiva¹, **Vitor H. O. Silva¹** (negrito + sublinhado discreto), Germano B. dos Santos¹, Fabrício A. Silva¹
- **Filiação**: ¹NESPeD-LAB — Universidade Federal de Viçosa, Florestal, MG
- Rodapé: logos UFV + NESPeD-LAB (esquerda), logos SBC + CoUrb 2026 (direita)
- Data: "Praia do Forte, BA — 25 de maio de 2026"

## Slide 2 — Contexto: LBSNs e POIs

- Título: **"LBSNs, POIs e Duas Tarefas Complementares"**
- Lado esquerdo (40%): ícone ilustrativo de mapa com pinos coloridos representando categorias
- Lado direito (60%): dois cards verticais
  - Card 1 (ícone 🏷️): **Classificação de Categoria de POI** — "Classificar a categoria semântica de um POI." Tag: "não-sequencial"
  - Card 2 (ícone ➡️): **Predição do Próximo POI** — "Prever a categoria do próximo POI dado o histórico." Tag: "sequencial"
- **Faixa inferior (banner verde-azulado claro com texto destacado em laranja):**
  > "Resultado: **~+21 pp F1** em categoria · **~72% vitórias** em next-POI"

## Slide 3 — Lacuna & Hipótese (REFORMULADO — diagrama)

- Título: **"De um Vetor Monolítico para Três Vetores Especializados"**
- Layout horizontal em 3 zonas:
  - **Zona esquerda (30%)** — MTLNet original:
    - Texto pequeno em cima: "MTLNet [Silva et al., 2025]"
    - Um único bloco verde grande rotulado **"DGI ∈ ℝ⁶⁴"**, com setas convergentes externas rotuladas "espaço", "categoria"
    - Texto fantasma cinza ao lado: *(sem tempo)*
  - **Zona central (10%)** — seta horizontal grande verde-azulada com texto "Decompor →"
  - **Zona direita (60%)** — ST-MTLNet:
    - Três blocos verticais empilhados:
      - 🌐 **Spatial 64-d**
      - 🕒 **Time 64-d**
      - 🏷️ **Categoria 64-d**
    - Cada bloco tem sua própria seta de entrada à esquerda
    - Texto pequeno: "concat → 192-d"
- **Caixa destacada na base** (bordas verde-azulado, fundo branco):
  > **Hipótese:** representações desacopladas com encoders especializados capturam dimensões complementares que o DGI ignora.

## Slide 4 — Proposta: ST-MTLNet (ENXUTO)

- Título: **"ST-MTLNet — Arquitetura"**
- Inserir imagem `arquitetura_modelo.png` ocupando 70% do slide à esquerda
- Coluna direita (30%):
  - 🌐 **Espacial** (64-d) — SIREN ou Sphere2Vec-M
  - 🕒 **Temporal** (64-d) — Time2Vec
  - 🏷️ **Categórico** (64-d) — HGI hierárquico
  - Linha de destaque: `concat → 192-d`
- **Rodapé pequeno** (cinza, 14 pt): *MTLNet interno preservado — isolamos o efeito da representação de entrada*

## Slide 5 — Encoders Espaciais (COMPACTADO)

- Título: **"Duas Estratégias para Codificar Coordenadas"**
- Tabela compacta centralizada (3 linhas × 2 colunas):

  |  | **SIREN** | **Sphere2Vec-M** |
  |---|---|---|
  | Estratégia | senoidal local | esférica multi-escala |
  | Pressuposto | ℝ² contínuo | superfície esférica |
  | Forte em | alta frequência | distância geodésica |

- Faixa inferior (cinza): *Mesma loss contrastiva binária (par+ ≤ 10 km, par− ≥ 70 km) — isola o efeito de arquitetura. Detalhes em backup.*

## Slide 6 — Setup Experimental + Mapa

- Título: **"Setup: Três Estados, Padrões Urbanos Distintos"**
- **Lado esquerdo (40%)** — tabela:

  | Estado | Check-ins | POIs | Usuários |
  |---|---|---|---|
  | Florida | 990 k | 65 k | 20 k |
  | California | 2,5 M | 148 k | 36 k |
  | Texas | 3,4 M | 135 k | 37 k |

- **Lado direito (60%)** — inserir `subáreas/distribuicao_estados.png`, legenda pequena: *"Densidade de POIs Food (vermelho) e Shopping (laranja) nas sub-regiões mais densas"*
- **Faixa inferior** com 3 linhas curtas:
  - Dataset **Gowalla** [Jure '14] · F1 médio por categoria
  - **5-fold estratificado por categoria**, divisão 80/20 (por POI para categoria · por janela não-sobreposta para next-POI)
  - 7 categorias × 3 estados = **21 combinações por tarefa**

## Slide 7 — Resultado 1: Categoria (HEADLINE CORRIGIDO)

- Título: **"Classificação de Categoria — 21/21 Vitórias"**
- **Headline gigante (90 pt, cor laranja `#E67E22`):**
  > **≈ +21 pp F1**
- Subheadline (verde-azulado, 32 pt): "em média por estado, melhor encoder por linha"
- Linha pequena cinza: "FL +20,2 · CA +20,9 · TX +22,0 · (variante por variante: +18 a +24 pp)"
- Gráfico de barras agrupado (3 estados × 4 categorias-chave: Food / Shopping / Nightlife / Travel)
  - Azul claro: MTLNet baseline
  - Verde-azulado: ST-MTLNet (melhor variante)
- Caixa lateral pequena: "Maior salto: **Nightlife** (~30% → >60%)"

## Slide 8 — Resultado 2: Next-POI (CONTAGEM CORRIGIDA, sem mapa)

- Título: **"Predição do Próximo POI — 15/21 Vitórias + 1 Empate"**
- **Headline (laranja, 72 pt):**
  > **~72%** das combinações
- Subheadline (32 pt): "15 vitórias estritas · 1 empate técnico · 5 derrotas"
- Pequena anotação em cinza: *"Florida Outdoors: empate dentro de σ (Δ ≈ 0,02 pp)"*
- Layout em 3 colunas (cards):
  - Card verde ✅ **Onde ganhamos**: "Food (3/3 estados — CA: 29% → **51%**), Shopping, Community"
  - Card cinza/amarelo ⚠️ **Empate técnico**: "Florida Outdoors"
  - Card vermelho ❌ **Onde perdemos**: "Travel em FL/CA · Entertainment/Nightlife/Outdoors em CA"
- Faixa inferior: *"Trajetórias longas favorecem o grafo DGI; transições locais favorecem encoders contínuos."*

## Slide 9 — Análise & Limitações (EXPANDIDO)

- Título: **"O Que Aprendemos — e o Que Ainda Falta"**
- **Coluna esquerda (Insights — verde):**
  - 📈 Ganho consistente da **decomposição** > escolha de encoder
  - 🗺️ Encoder espacial ideal **depende do território**
    - SIREN: FL e CA
    - Sphere2Vec-M: TX
- **Coluna direita (Limitações — roxo, lista mais honesta):**
  - ⚠️ Long-range / Travel: grafo DGI superior
  - ⚠️ Sem ablação isolando contribuição de cada encoder
  - ⚠️ Sem comparação com baselines externos modernos (LSTPM, GETNext, STAN)
  - ⚠️ Gowalla envelhecido (2009–10) · escopo de 3 estados, 7 categorias high-level
- Rodapé pequeno: "**Futuro**: fusão híbrida grafo+contínua · gating/cross-attention · MTL flexível (MoE)"

## Slide 10 — Takeaways

- Título: **"Para Levar"**
- Três cards numerados grandes, lado a lado:
  - **1.** Encoders **desacoplados** > monolítico para tarefas com demandas distintas
  - **2.** **~+21 pp** em categoria · **~72%** vitórias em next-POI
  - **3.** Não há encoder espacial universal — **território importa**
- Linha divisória
- Bloco "Recursos":
  - 💻 Código: github.com/TarikSalles/Spatial_Embeddings
  - ✉️ Contato: vitor.oliveira@ufv.br
- Em letras grandes embaixo, centralizado, verde-azulado: "Obrigado — perguntas?"

---

## Slides de apêndice (após slide 10, ocultos por padrão)

- **A1**: Tabela completa F1 categoria (21 linhas — MTLNet vs SIREN vs Sphere2Vec-M)
- **A2**: Tabela completa F1 next-POI (21 linhas)
- **A3**: Detalhe SIREN — fórmula + arquitetura senoidal
- **A4**: Detalhe Sphere2Vec-M — 16 escalas geométricas
- **A5**: Time2Vec — fórmula f(τ)[i] linear + senoidal
- **A6**: POI Encoder (Delaunay + skip-gram) + HGI hierárquico (POI–região–cidade)
- **A7**: NashMTL — barganha de Nash para balanceamento de gradientes
- **A8**: Loss contrastiva binária — motivação dos thresholds 10 km / 70 km
- **A9**: Hiperparâmetros: AdamW (lr=1e-4, wd=0.05), OneCycleLR, batch 2048, 50 épocas
- **A10**: Hardware e tempo de treino
- **A11**: Ablação retórica — "se zerássemos um encoder..."
- **A12**: Split detalhado — estratificado por categoria, janelas não-sobrepostas, <5 visitas descartados
- **A13**: Posicionamento vs baselines externos (LSTPM, GETNext, STAN, HMT-GRN)

---

## Após gerar

1. Compartilhe o link do arquivo Figma com o apresentador
2. Apresentador exporta como PDF (File → Export → PDF)
3. Mantém versão Figma editável para refinamentos visuais finais
