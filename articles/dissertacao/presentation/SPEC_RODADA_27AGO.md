# Spec — rodada de revisão de 27/08

Escrito pelo `gate`. Destinatário da execução: `ppt` (único que edita `slides/main.tex`).
Cada mudança abaixo está ancorada por **título de slide**, não por número — os números
já se moveram três vezes.

Estado do deck medido no início desta rodada: **47 impressos**, PDF de 104 páginas.

---

## Reconciliação de numeração (ler antes de executar)

O autor pediu por número. Três dos quatro bateram; um não.

| pedido do autor | título hoje | veredito |
|---|---|---|
| slide 31 — Check2HGI | `The architecture: sharing by exchange` | ❌ **não bate.** O conteúdo descrito (64-d, one-hot de categoria, seno do horário, tempo desde a última visita) está no **impresso 30, `What each visit contributes`** |
| slide 34 — OOD discounted | `The protocol, in four steps` (2 · what is measured) | ✅ bate, a linha OOD está lá |
| slides 38 e 39 | `Result 2: one model, two tasks` / `The verdict, dataset by dataset` | ✅ batem |
| slides 44 e 45 | `Limitations and next steps (1 of 2)` / `(2 of 2)` | ✅ batem |

**Premissa assumida no item 1:** o alvo é o impresso **30**. O conteúdo é inequívoco;
o número não. Se o autor quis outro slide, este item volta.

---

## ITEM 1 · `What each visit contributes` (impresso 30)

### 1a · O que o autor pediu, e o que o volume sustenta

Duas das três coisas pedidas entram. A terceira não, e é conceitual.

| pedido | o que o Apêndice D do volume principal entrega | entra? |
|---|---|---|
| dimensão 64 dos embeddings de saída | `apx_h_check2hgi_joint_model.tex:29` — "export separate **64-dimensional** check-in and region representations"; `:133` — $\mathbf{x}_i\in\mathbb{R}^{64}$; `:439` — "width 64" | ✅ **sim** |
| one-hot da categoria, seno do dia/horário, tempo desde a última visita | `:95` e `:437` — "seven category indicators, **sine and cosine** for hour of day, **sine and cosine** for day of week, and **four** elapsed-time values; **width 15**" | ✅ sim, **com duas correções** (abaixo) |
| "isso acontece no **pre-trained encoder**, antes do resto do Check2HGI" | — | ❌ **não. Ver 1b.** |

**Correção 1.** É seno **e cosseno**, e para **hora do dia e dia da semana** — quatro valores,
não dois. O slide 30 hoje já diz `through sine and cosine`, e está certo. Não trocar por "seno".

**Correção 2.** São **quatro** valores de tempo decorrido (desde a visita anterior, desde a
primeira visita, o intervalo dentro do mesmo dia, e um indicador de primeira visita), não um.
O slide 30 hoje já lista os quatro. Não reduzir a "intervalo desde a última visita".

### 1b · Por que "pre-trained encoder" não pode entrar

Os 15 valores **não são produzidos por encoder nenhum**. São lidos direto do check-in bruto.
Quem mapeia 15 → 64 é a **primeira camada de convolução de grafo do próprio Check2HGI**:

> "The first layer maps the 15 input features to 64 dimensions and applies layer normalization,
> PReLU, and dropout. The second layer maps 64 to 64 and is added" — `apx_h:130-133`
> "Check2HGI & Encoders and pooling & **Two check-in graph-convolution layers**…; width 64" — `apx_h:439`

O que **é** pré-treinado no Check2HGI existe, mas é outra coisa, em outro andar:

> "a 64-dimensional place table **initialized from the pretrained place representation**" — `apx_h:167`
> "Check2HGI & Auxiliary learning & …; **pretrained place-table anchor**" — `apx_h:440`

É uma **âncora auxiliar no nível do lugar**, não o produtor das features do check-in.
(O `[Pretrained Category Encoder]` que o autor tem em mente está no desenho do **HGI**
— `hgi_draw.txt:16` — onde alimenta as features iniciais do **lugar**. Herança do HGI,
nível diferente.)

**Escrever "pre-trained encoder" sobre esses três grupos afirma o que a dissertação não diz.**
É a mesma classe do defeito do slide 47 (cross-attention) e do bullet 3 do slide 38 (abaixo):
conteúdo do autor que perde o escopo do capítulo ao migrar para a tela.

### 1c · O que executar

O que o autor quer que a plateia entenda — *"um vetor é montado a partir do check-in bruto,
antes do resto"* — é **verdade** e pode ser dito. Só não com a palavra "pre-trained".

**(i)** Trocar o `\framesubtitle`:

```latex
% de:
\framesubtitle{The node features, and one design principle}
% para:
\framesubtitle{Assembled from the raw check-in $\cdot$ 15 values in, 64 out}
```

**(ii)** Acrescentar uma linha única depois do `itemize`, antes do `\bigskip`:

```latex
    \vspace{1mm}
    {\footnotesize These 15 values are read off the check-in itself --- no encoder before them.
    Check2HGI's first check-in layer maps them to the \textbf{64 dimensions} it exports.\par}
```

Custo: 1 linha de texto + `\framesubtitle` já existente. Nenhum bullet novo.
Se não couber, o `\framesubtitle` sozinho entrega o 15/64 e a linha vira fala.

---

## ITEM 2 · `The protocol, in four steps` — 2 · what is measured (impresso 34)

### Veredito: o **rótulo** sai, a **substância** fica — dentro do bullet do Acc@10.

Medido:

- `OOD-discounted` aparece **uma vez na tela** e **nunca mais é referenciado** no deck.
- No volume o rótulo é definido em `2_fundamentals.tex` e **não aparece** em
  `5_mobiwac/06_results.tex` nem nas tabelas — nenhum número reportado usa esse nome.
- O Cap. 5 descreve a mesma quantidade **sem rótulo, dentro da definição do Acc@10**:

> "For the region task, we report accuracy at ten (Acc@10). This measure is the fraction of test
> visits for which the true region is among the model's ten highest-scoring predictions.
> **If the true region does not occur in the training data for that fold, the visit counts as an
> error.**" — `5_mobiwac/05_setup.tex:111`

Como bullet separado, a tela **implica duas métricas de região**. O Cap. 5 reporta **uma**.
Fundir corrige a implicação, tira um termo órfão (glossário fail-closed) e devolve altura.

**Executar** — apagar o terceiro item e fundir a cláusula no segundo:

```latex
% APAGAR:
        \item \textbf{OOD-discounted Acc@10} --- a region absent from the training fold counts as
              an error;

% e o bullet do Acc@10 passa a ser:
        \item \textbf{Region: Acc@10} --- the share of test visits whose true region is among the
              ten highest-scoring predictions; \emph{it does not separate first place from tenth},
              and \textbf{a region absent from the training fold counts as an error};
```

Ganho: −1 bullet (≈ 2ex de `itemsep` + a linha), zero perda de divulgação.

---

## ITEM 3 · `Result 2` (38) e `The verdict` (39)

Medido no PDF: **195 e 174 palavras**. Abaixo das tabelas, ~99 e ~103 palavras de **frases**
— e frase é exatamente o que a plateia do minuto 40 não lê. As tabelas ficam intactas:
número escaneia.

### 3a · `Result 2: one model, two tasks` — o bullet 3 é defeito, não densidade

O bullet 3 diz:

> "**no established protocol exists for next region** over an administrative partition.
> **This work fixes one.**"

O volume diz:

> "To our knowledge, fine-grained region as an end target of equal standing, rather than an
> auxiliary coarse grid cell, **is underexplored**. **The nearest exceptions do not study our
> exact pairing**: DRRGNN… ; a generative recommender…" — `5_mobiwac/02_related.tex:94-99`

Três desvios, e eles se compõem:

1. **"underexplored" → "no established protocol exists".** Alegação hedged promovida a alegação
   categórica de ausência — e o volume **nomeia duas exceções por citação** na frase seguinte.
2. **"This work fixes one."** O volume nunca reivindica consertar a lacuna do campo. A conclusão
   escreve "the final design and evaluation protocol developed **in this dissertation**"
   (`6_conclusion.tex:200`) — escopo em si mesma, não no campo.
3. **Trocou o eixo da alegação.** A do volume é sobre a **formulação da tarefa** (região como alvo
   final de igual estatura). O slide a converteu em alegação sobre **protocolo de avaliação** —
   que é outra coisa, e não está em lugar nenhum.

**E o deck já faz a alegação certa, hedged, em dois lugares** — então cortar não perde nada:

- `main.tex:273` — "**Among the works reviewed in this dissertation**, none treats the next category
  and the next region as co-equal end targets of one joint model that does not also predict the next place."
- `main.tex:1016` — "the standard formulation targets a **grid cell**; here, official
  neighborhood-scale units. In MCMG and HMT-GRN, category and region are **auxiliary**…"

O bullet 3 é a **única versão sem escopo** dessas três. **Cortar.**

> ⚠ Clayson Celes (ITA, externo, mobilidade) é precisamente quem pergunta *"protocolo estabelecido
> por quem?"*. Este bullet é convite.

Os bullets 1 e 2 ficam — a fala do 38 já os cobre inteiros (`main.tex:1332-1345`), então
encurtar na tela não perde argumento.

**Executar:**

```latex
    {\footnotesize
    \begin{itemize}\setlength{\itemsep}{0pt}
        \item \textbf{Above every external system we ran, on both tasks} --- by at least
              \textbf{3.06} macro-F1 on next category;
        \item \textbf{above the first-order Markov floor by $+4.1$ to $+10.0$}, at every dataset ---
              \alert{HMT-GRN is below that floor at all six}; STAN at four, ReHDM at three.
    \end{itemize}\par}
```

Rodapé (provenance dos externos, R6 — fica, só aperta):

```latex
    {\scriptsize HMT-GRN: same data, folds and seeds (primary comparison).
    $^\dagger$STAN: our re-implementation, output adapted to regions; TX 4/5, CA 2/5 folds, seed 0.
    $^\ddagger$ReHDM: its own published protocol, single seed on CA/TX. Ties in bold on both.\par}
```
(inalterado — é rótulo de dado, não frase)

**Balanço:** 99 → ~52 palavras de prosa, −1 bullet, e sai a alegação insustentável.

### 3b · `The verdict, dataset by dataset` — cortar o que os Fundamentos já entregaram

O autor está certo sobre a repetição. Medido, o slide **36** (`4 · how it is decided`) já imprime:

| slide 36, `main.tex:1289-1291` | slide 39 repete |
|---|---|
| "Analysis plan written **before any result was read**" | "registered **before any result was read**" |
| "**Non-inferiority** → next region, **two-point margin**" | "stay within the **two-point margin**" |
| "Paired $t$ · 90% CI · **Holm** across the six datasets" | "**Holm-corrected**" |

Isso é ~25 palavras de metodologia duplicada, três slides depois.

**O que NÃO pode sair** (cada uma tem lei atrás, e as três estão no volume):

- **"The two region gains are secondary results, outside the registered plan."**
  `05_setup.tex:113` — *"The plan did not define a superiority test for next-region prediction.
  Therefore, the two next-region gains … are secondary results outside the plan."*
  A fala também diz (`main.tex:1417`). **A duplicação é deliberada** — é uma divulgação
  auto-incriminatória colada a um número na tela, a mesma classe da AUT-26. Fica.
- **"all four are deficits, not ties"** — lei anti-`match`. INTOCÁVEL.
- **"the other five are unresolved"** — a palavra de veredito da categoria.
- **a ressalva de que "within half a point" não vem de teste.**
  `05_setup.tex:113` — *"On next category the plan registered **no equivalence margin**, so a
  difference that fails the superiority test is reported as **unresolved rather than as a match**."*

**Executar** — tabela intacta; o rodapé de três parágrafos vira:

```latex
    {\scriptsize
    {\color{primaryshade}$\blacktriangle$}~\textbf{Outperforms the dedicated model}, Holm-corrected
    --- region: Texas ($p$ 0.00013) $\cdot$ California ($p<10^{-4}$), 20 of 20 folds;
    category: Florida ($p$ 0.011), 19 of 20.
    \textbf{The two region gains are secondary results, outside the registered plan.}\par
    \textbf{Region} --- the other four \alert{are deficits, not ties}: all four intervals lie
    entirely below zero.\par
    \textbf{Category} --- the other five are \alert{unresolved}; all six within half a point,
    read off the intervals. \textbf{The plan registered no equivalence margin on category.}\par}
```

**Balanço:** 103 → ~68 palavras. Todo o corte cai sobre o que o slide 36 já entregou;
nenhuma divulgação sai; e a última frase agora cita o volume em vez de parafrasear.

---

## ITEM 4 · `Limitations and next steps` (44 e 45)

Pendente — a varredura de trabalhos futuros nas documentações ainda está rodando.
Entra em spec separada.

---

## Procedência

Tudo acima foi lido no volume entregue (`src/`), não em resumo:
`chapters/apx_h_check2hgi_joint_model.tex` (29, 95, 130-133, 167, 437, 439, 440);
`chapters/2_fundamentals.tex` (441-452, 703-712);
`chapters/5_mobiwac/02_related.tex` (92-99); `chapters/5_mobiwac/04_method.tex` (17-22);
`chapters/5_mobiwac/05_setup.tex` (111, 113); `chapters/6_conclusion.tex` (200).
Deck medido em `slides/main.pdf` por `pdftotext`, páginas 44 e 45.

---
---

# ITEM 4 · `Limitations and next steps` (44 e 45) — spec

Escrito depois da varredura. Substitui a nota "pendente" acima.

## 4.0 · O que a varredura das documentações devolveu, e por que quase tudo foi rejeitado

O autor pediu: *"para o 45 consulte a dissertação e as demais documentações para outras
propostas de trabalhos futuros"*. Consultei. **87 candidatos**, de três famílias.

| origem | candidatos | entram? |
|---|---:|---|
| volume entregue (`6_conclusion.tex` §Future work + §Limitations) | 11 | ✅ **todos** — é o núcleo defensável, cada um amarrado 1:1 a uma limitação |
| lista do próprio autor (`wrapup/Questions_author.md` §Ideias futuras / §Gama de trabalhos futuros) | 13 | ✅ os que sobrevivem ao teste "já entregue?" |
| memos do código (`docs/future_works/`, `docs/studies/*/FINAL_SYNTHESIS.md`) | 63 | ❌ **nenhum.** Ver abaixo |

**Por que os 63 memos do código não podem subir nesta tela** — três motivos, e cada um sozinho basta:

1. **Alguns concedem a tese.** `composite_two_substrate_engine.md:11` propõe rotear categoria
   para um checkpoint e região para outro; `part2_mtl_dual_substrate_routing.md:23` idem; o
   roteamento C1 idem. Todos quebram *"um modelo, uma passagem, N tarefas"* — que é a
   propriedade que o Cap. 5 defende. Propor isso como trabalho futuro na defesa é oferecer
   ao arguidor a desistência da tese.
2. **Alguns já são o que se faz hoje.** `mtl_frontier/FINAL_SYNTHESIS.md:138` propõe
   "acoplamento cat↔reg por cross-attention". **É o tronco atual.** É exatamente o defeito
   que já apareceu no slide 47 e foi corrigido — a mesma armadilha, de outra fonte.
3. **O resto é de outro gênero.** FAMO, DSelect-K, cross-stitch, RLW, PCGrad, log_T, BayesAgg-MTL,
   rank efetivo, GRM/Memory-Soup: vocabulário que **não está no `GLOSSARY.md`** (§8.11 é
   fail-closed) e que o deck nunca introduziu. E `evaluation_protocol_cleanup.md` propõe
   trocar o protocolo estatístico (CV aninhada, bootstrap no lugar do n=20) — numa tela de
   defesa isso não lê como trabalho futuro, lê como *"nosso protocolo está errado"*.

> Conclusão da varredura, que é a resposta ao pedido: **as documentações do código não têm
> trabalho futuro de dissertação.** Têm backlog de pesquisa do repositório. As duas fontes
> legítimas para o slide 45 são o volume e a lista do autor.

## 4.1 · O que está faltando hoje

**A limitação 6 não está no deck principal.** O volume abre a seção com *"**Six** limitations
bound the scope of these conclusions"* (`6_conclusion.tex:204`) e o deck imprime 1…5. A sexta é
**"The task-pair confound"** — o par de tarefas mudou junto com a representação e a topologia,
então nenhuma ablação isola qual mudança produziu o ganho ao longo dos três capítulos.

Ela existe na **Série B** (`Task pair`, `U6`, hypertarget `u6`, `main.tex:2635-2645`), com a
defesa completa. Mas a tabela do 44 numera 1…5, o que afirma enumeração completa. Quem estiver
com o volume na mão vê a lacuna — e a sexta é justamente a que limita o arco inteiro.

**Três trabalhos futuros do volume não estão no deck:**

| proposta | fonte no volume | por que importa |
|---|---|---|
| ablação do acoplamento entre a tabela de lugares e a tabela pré-treinada de inicialização | `6_conclusion.tex:402-405` | separa o que a inicialização contribui do que a hierarquia contribui — experimento cético contra o próprio método |
| alvo estático que a representação **não carregue como feature de entrada** | `6_conclusion.tex:448` | é o único caminho para fechar a limitação 6 |
| experimento controlado separando **número de regiões** do **volume de dados** | `6_conclusion.tex:145-147` | o slide `Closing` já diz que a resposta depende da **escala do problema**; este é o experimento que a fixaria |

**Mais dois, confirmados pela varredura contra o texto entregue:**

| proposta | fonte | veredito |
|---|---|---|
| testar o Check2HGI **dentro de outras arquiteturas** de predição de mobilidade | `6_conclusion.tex:126` | ainda futura — todo externo avaliado (STAN, HMT-GRN, POI-RGNN, ReHDM, CTLE) roda com as próprias representações |
| **controle de dimensão igual** para o Cap. 4 (192 vs 64) | `6_conclusion.tex:79` | o próprio Cap. 4 o pede, para separar a contribuição semântica dos encoders do efeito da largura |

**Um da lista do autor, listado por ele duas vezes, ausente do deck:**

- **"Unificar o Check2HGI com o MTL"** (`Questions_author.md:95` e `:127`) — treinar representação
  e modelo conjunto ponta a ponta. Hoje são dois estágios: *"train Check2HGI, **then export**
  separate 64-dimensional check-in and region representations"* (`apx_h:29`). **Não entregue.**

## 4.2 · Executar — slide 44 vira só limitações

Título: `Limitations` (some o "and next steps (1 of 2)"). Uma tabela, seis linhas, duas colunas.

```latex
\begin{frame}{Limitations}
    \vspace{0pt}% ⚠ GUARDA -- NAO REMOVER.
    % ⚠ "a field-wide constraint" FICA -- quatro palavras, unico item com defesa embutida.
    {\footnotesize
    \setlength{\tabcolsep}{6pt}
    \renewcommand{\arraystretch}{1.5}\begin{tabular}{@{}p{4.4cm} p{8.8cm}@{}}
        \toprule
        \textbf{1 $\cdot$ Data vintage}
            & Gowalla 2009--2011 $\cdot$ Istanbul 2012--2018 --- \alert{a field-wide constraint} \\
        \textbf{2 $\cdot$ Taxonomy coarseness}
            & seven top-level classes \\
        \textbf{3 $\cdot$ Transductive representation}
            & no unseen places or users without retraining \\
        \textbf{4 $\cdot$ No next-place task}
            & conclusions cover next category and next region only \\
        \textbf{5 $\cdot$ Geographic coverage}
            & outside the United States, one city \\
        \textbf{6 $\cdot$ The task-pair confound}
            & the pair changed together with the representation and the topology;
              \textbf{Chapter~4 is the fixed-pair control} \\
        \bottomrule
    \end{tabular}\par}
\end{frame}
```

⚠ **A linha 6 precisa do botão para a Série B** (`\hyperlink{u6}{...}`), no padrão dos outros 53.
`ppt` sabe a macro; eu não a especifico para não errar a assinatura.

A meia-frase `Chapter 4 is the fixed-pair control` é o que impede que a linha 6 leia como
rendição — é literal da Série B (`main.tex:2641`) e do volume.

## 4.3 · Executar — slide 45 vira só trabalhos futuros

Título: `Future work`. Três grupos, itens em **frase nominal, não sentença** — é o que a
regra de gênero permite escanear no minuto 43. A etiqueta `L<n>` preserva o pareamento 1:1
com o slide anterior, que era o que a tabela de duas colunas fazia.

```latex
\begin{frame}{Future work}
    \vspace{0pt}% ⚠ GUARDA -- NAO REMOVER.
    {\footnotesize\setlength{\tabcolsep}{6pt}
    \renewcommand{\arraystretch}{1.35}
    \begin{tabular}{@{}p{2.5cm} p{10.9cm}@{}}
        \toprule
        \textbf{Representation}
            & an \textbf{inductive variant} --- unseen places and users, no retraining~{\scriptsize(L3)};
              a \textbf{hypergraph} formulation, one edge per session~{\scriptsize(L3)};
              an \textbf{ablation of the pretrained place-table coupling}~{\scriptsize(L3)};
              \textbf{Check2HGI inside other mobility architectures} \\
        \addlinespace
        \textbf{Joint model}
            & the \textbf{exact next place as a third target} --- own head, or a cascade~{\scriptsize(L4)};
              a shared trunk with \textbf{MMoE}, as an alternative to cross-attention;
              \textbf{one end-to-end stage} --- representation and joint model trained together \\
        \addlinespace
        \textbf{Evidence}
            & \textbf{newer, denser traces}~{\scriptsize(L1)} $\cdot$
              \textbf{finer taxonomies}~{\scriptsize(L2)} $\cdot$
              \textbf{cities outside the United States}~{\scriptsize(L5)};
              a \textbf{static target the representation does not carry as a feature}~{\scriptsize(L6)};
              a \textbf{controlled experiment separating region count from data volume} \\
        \bottomrule
    \end{tabular}\par}
\end{frame}
```

O bloco `Further work, by area` que estava no rodapé do 45 **é absorvido** — seus quatro itens
viraram linhas dos grupos acima, exceto um (ver 4.4).

**Onde cada item se ancora**, para o caso de alguém perguntar de onde saiu:

| item | fonte |
|---|---|
| inductive variant · hypergraph · place-table coupling | `6_conclusion.tex:399-409` (todos L3) |
| Check2HGI inside other mobility architectures | `6_conclusion.tex:126` |
| next place as third target, head or cascade | `6_conclusion.tex:439-443` + `Questions_author.md:125-126` |
| MMoE | `Questions_author.md:123` |
| end-to-end | `Questions_author.md:95, :127`; hoje é dois estágios (`apx_h:29`) |
| traces · taxonomies · cities | `6_conclusion.tex:397-398, :446` |
| static target | `6_conclusion.tex:448` |
| region count vs data volume | `6_conclusion.tex:145-147` |

## 4.4 · Correções aplicadas depois da varredura, e o que fica para o autor

A varredura terminou depois que eu escrevi a spec acima e **derrubou dois itens meus**.
Auditei os dois contra o texto entregue antes de aceitar; um caiu, o outro sobreviveu por
causa de uma qualificação que já estava na frase.

### CAÍDO · "more features on the check-in nodes" — **removido do slide 45**

O Check2HGI entregue **já fez uma rodada disso**: o nó foi de 11 para 15 colunas com o grupo
de tempo decorrido (`apx_h:95`, `:437`; `2_fundamentals.tex:703-712`), e esse grupo é um dos
pontos de design que o próprio capítulo destaca — está no slide 30. Como linha nua de trabalho
futuro, lê como se os nós fossem pobres em features, quando a última rodada de adicioná-las **é**
a contribuição.

A forma honesta nomearia o obstáculo que o autor anota (*"Problema no infomax"*: o objetivo
troca fatores entre si quando se acrescentam features). Mas **esse mecanismo não está no volume**
— vive só num memo de `docs/studies/`, que é exatamente a classe de fonte rejeitada em 4.0.
Sem o obstáculo é um to-do enganoso; com ele, cita o que a dissertação não diz. **Sai.**

### SOBREVIVEU · o item do MMoE — e a pergunta 1 antiga está **respondida**

Eu ia perguntar ao autor o que era "remover a camada embedding" (`Questions_author.md:123`).
Não é preciso: é o miolo compartilhado do MTLnet — **task embedding + FiLM** — e o **Cap. 5 já o
removeu**:

> "MTLnet uses residual layers conditioned by FiLM as its shared middle. **The joint model
> replaces that component with cross-attention blocks**" — `2_fundamentals.tex:796-806`
> o tronco é um "cross-attention stack of two blocks", **"not by owning hidden layers in common"**
> — `5_mobiwac/04_method.tex:28-30`

Ou seja: a proposta do autor era "remover a camada embedding, ter uma camada compartilhada
(**mmoe ou cross-attention**)" — e **o ramo cross-attention foi tomado**. Só o ramo MMoE é futuro.

A linha da spec já está certa porque carrega a qualificação: *"a shared trunk with **MMoE**,
**as an alternative to cross-attention**"*. **Não acrescentar** um item separado de "remover a
camada embedding" — essa metade está entregue. É a quarta vez que esta armadilha aparece
(slide 47, bullet 3 do slide 38, o "pre-trained encoder" do item 1, e agora esta).

### Fica para o autor — não bloqueia nada

1. ✅ **FECHADO (autor, 27/08 — `AUT-36`). O controle de dimensão igual do Cap. 4 fica FORA.**
   *"É trabalho futuro do Cap. 4 de um artigo que já mudamos muita coisa; não acho que vale
   voltar nisso."* O item é real e permanece no volume (`6_conclusion.tex:79`); sai só da tela.
   ⚠ A ressalva de que a comparação do Cap. 4 **não é width-matched** (192 contra 64) passa a
   existir **apenas no volume** — não há apoio na tela nem na fala se a banca perguntar.
2. **"fold the POI encoder into the HGI"** (`Questions_author.md:119`, mecanismo: *"propagar os
   erros das features"*). Estava no bloco antigo do 45. Saiu junto com "more features", porque é a
   mesma manobra vista do outro lado e cai pelo mesmo argumento. Se o autor a quiser de volta,
   ela precisa de uma forma que o volume sustente — e eu não achei uma.

---
---

# ITEM 5 · `Future work` (45) — refazer para ser OUVIDO

Pedido do autor, via `ppt`: *"Esse tem de ficar bem mais didático e simples. Nessa altura
ninguém estará lendo, só escutando."* É a `AUT-24` aplicada ao slide que eu acabei de encher.

**O `ppt` está certo e eu errei o gênero.** A tabela de três grupos com etiqueta `L<n>` é boa
**para ser lida**. No minuto 45 ninguém cruza uma etiqueta de ouvido com a tela anterior.

## 5.1 · Três, e por que estas três

A tela passa de **11 itens / ~90 palavras** para **3 / ~34**. O critério não foi importância
científica — foi: *o que responde "e agora, o quê?" para quem só escuta*.

As três espelham a espinha que o próprio deck já declarou. O slide `Closing` diz que a resposta
depende de **três coisas: a representação de entrada, a topologia de compartilhamento e a escala
do problema**. Os trabalhos futuros devolvem uma para cada:

| a tela | o que fecha | por que esta |
|---|---|---|
| tornar a representação **indutiva** | L3 | é a única que destrava uso real — cidade que cresce |
| o **próximo lugar exato** como terceira tarefa | L4 | estende o alcance sem trocar a representação |
| **separar escala de dados** | a condicional do `Closing` | TX e CA são os dois que ganham **e** os dois com mais regiões **e** os dois com mais dados. Eu devo esse experimento |

A terceira é a que se auto-incrimina, que é a postura estabelecida do deck (*"quatro limites que
eu ofereço antes de alguém pedir"*).

```latex
\begin{frame}{Future work}
    \vspace{0pt}% ⚠ GUARDA -- NAO REMOVER.
    \vfill
    \begin{itemize}\setlength{\itemsep}{3.0ex}
        \item {\large \textbf{Make the representation inductive} --- new places and new users,
              without retraining.\par}
        \item {\large \textbf{Add the exact next place} as a third target, on the same
              representation.\par}
        \item {\large \textbf{Separate scale from data} --- a controlled experiment on why
              Texas and California gain.\par}
    \end{itemize}
    \vfill
\end{frame}
```

Sem etiqueta `L<n>` (ninguém as cruza de ouvido) e **sem botão para a reserva** — o autor
acabou de mandar remover o do item 6 do slide 44, então a trilha principal não aponta para lá.

## 5.2 · Os oito que saem da tela — e onde ficam

⚠ **É o custo que o `ppt` declarou, e ele é real:** cinco destes eu recuperei do volume hoje,
e a varredura existiu para isso. **Eles não se perdem — mudam de veículo.**

| proposta | vai para |
|---|---|
| formulação em hipergrafo (L3) | fala |
| ablação do acoplamento da place-table (L3) | fala |
| Check2HGI dentro de outras arquiteturas | fala |
| treino ponta a ponta (representação + conjunto) | fala |
| tronco compartilhado com MMoE | reserva |
| alvo estático que a representação não carregue (L6) | fala |
| taxonomias mais finas (L2) · traces mais recentes (L1) | fala |
| cidades fora dos Estados Unidos (L5) | fala |

**Um slide de reserva `B-FUTURE`** com as onze vale a página — o `ppt` ofereceu construí-lo.
✅ **Sim, construa.** Sem botão na trilha principal; é alcançável por navegação, para o caso de a
banca perguntar *"e o que mais?"*. A tabela de três grupos da §4.3 serve como conteúdo dele
tal como está.

## 5.3 · A fala, que hoje não existe

O `ppt` reporta que o `Future work` ficou **mudo** — as duas falas antigas foram ambas para o
slide de limitações. Segue o texto. ~150 palavras, ~64 s.

> "Três coisas, e a primeira é a que destrava uso real. A representação de hoje é **transdutiva**:
> ela não representa um lugar nem um usuário que não estava no grafo, sem retreinar. Uma variante
> indutiva remove isso, e é o que uma cidade que cresce precisa.
>
> A segunda estende o alcance: o **próximo lugar exato**, como terceira tarefa, sobre a mesma
> representação. Muda o pipeline de entrada e sai uma saída a mais — não uma representação nova.
>
> A terceira é a que **eu devo**. O Texas e a Califórnia são os dois conjuntos que ganham, e são
> os dois com mais regiões. Mas são também os dois com **mais dados**. Separar as duas explicações
> exige um experimento controlado, e ele não está feito.
>
> E há mais, que eu não ponho na tela: uma formulação em hipergrafo, uma ablação da âncora da
> tabela de lugares, um alvo estático que a representação não carregue como entrada, o Check2HGI
> dentro de outras arquiteturas, e treinar representação e modelo conjunto num estágio só."

⚠ **O último parágrafo é o que salva os cinco que a varredura recuperou.** Se a fala for cortada
por tempo, corte-o por último — sem ele, cinco propostas do volume somem da defesa inteira.

## 5.4 · Uma lição de instrumento, do quase-acidente do `ppt`

Ele registrou: substituir um bloco de frames **apagou as falas que estavam entre eles**
(limites 4 e 5, 57 palavras), e *"isso não dispara em varredura nenhuma"* — só apareceu na
conferência manual.

**Regra nova, para o `HANDOFF_GATE.md`:** a fala vive em `% FALA:`, **fora** do `\begin{frame}`.
Toda operação que substitui um intervalo de linhas do `.tex` leva junto as falas do intervalo,
e **nenhuma verificação de PDF a detecta** — comentário não renderiza. Antes de substituir um
bloco de frames, contar os `% FALA:` do intervalo; depois, contar de novo.

---
---

# ITEM 6 · O preprocessing na tela — `HGI` (20) e `Check2HGI: a fourth level below the place` (29)

Pedido do autor: uma **linha visual curta** em cada slide, mostrando como os dados chegam ao modelo.
Fonte primária mandada por ele: `considerations.md`. Confrontado com o volume e com o código.

**Numeração: os dois batem exatamente** (20 = `HGI`, 29 = `Check2HGI: a fourth level below the place`).

## 6.0 · A decisão de forma, e por que ela não é neutra

Ele ofereceu duas formas: fluxo ASCII vertical, ou cadeia horizontal com setas. **Cadeia horizontal**,
por três razões medidas e não sentidas:

1. o slide 20 já carrega a chapa `hgi_flow` **sem `width=`** (tamanho final da `tikz`) — altura é o
   recurso escasso, e um fluxo vertical cobra altura enquanto o horizontal cobra largura, que sobra;
2. uma varredura da esquerda para a direita é **um movimento de olho**, e no oral é o que se pode
   pedir; um fluxo vertical pede leitura;
3. as duas chapas já são fluxos horizontais — a faixa fica **consistente com o que está acima dela**.

🛑 **E a forma tem de ser a MESMA nos dois slides**, porque o valor real desta mudança não é
documentar dois pipelines: é **fazer o contraste se entregar sozinho** (ver 6.3).

## 6.1 · HGI (20) — o que o `Category Encoder` realmente faz

⚠ **Aqui o registro do autor e o código divergem, e a divergência é material.**

O desenho dele (`hgi_draw.txt:14-18`) diz:

```
POI Categories → [Pretrained Category Encoder] → POI Category Embeddings
```

E o Cap. 2 concorda: *"A pretrained category encoder supplies the initial POI features"*
(`2_fundamentals.tex:441`). **Mas isso descreve o ARTIGO do HGI, não o que este repositório executa.**

Medido no código e no artefato:

| | o que é |
|---|---|
| o módulo | `research/embeddings/hgi/poi2vec.py` — duas tabelas `nn.Embedding`, xavier init. **Nenhum texto é processado**, não é modelo de linguagem |
| o vocabulário | **não são as 7 categorias.** É a coluna `spot`, **284 a 365 valores por estado** |
| o treino | skip-gram sobre caminhos de Node2Vec, `logsigmoid` de produto interno (`:158-168`) |
| a saída | `poi_emb[i] = fclass_emb[fclass[i]]` (`:484-487`) — **lookup puro** |
| medido no artefato real | `poi2vec_poi_embeddings_Alabama.csv`: **11.848 lugares → 284 vetores distintos → 41,7 lugares por vetor** |
| quando roda | fase 3b–3d do `hgi.pipe.py:141`, **antes** do HGI, e o resultado é lido de disco depois |

🛑 **Consequência para o desenho: escrever `POI Categories` como entrada da caixa seria falso** — a
entrada é a classe fina do lugar, e as 7 categorias são o *achatamento* dela
(`src/etl/gowalla/stage_1.py:162`, `category = spot.map(super_categories_dict)`).

⚠ **E o termo exato está BLOQUEADO.** `GLOSSARY.md:100` restringe **`fine class`** a *"Appendix B
§B.5 only"* e diz **"NEVER write `fclass` in prose"**. §8.11 é fail-closed. Isto é o `Q18`, já aberto
no registro para o slide 24. **Contornei por paráfrase, usando os exemplos da própria entrada do
glossário** — que são melhores numa tela do que o termo seria.

### A faixa do slide 20

```
the place's type          a 64-d table, trained             one vector per TYPE
Airport · Coffee Shop  →  before HGI runs, then frozen  →   every place of that type shares it
```

## 6.2 · Check2HGI (29) — e por que a caixa NÃO pode chamar-se "pre-trained encoder"

O desenho do autor (`considerations.md:3128-3135`) nomeia a caixa
**`Preprocessing/Pretraining Encoder`**, com `Check-in` a entrar e `Check-in Feature Embedding` a sair.
**A metade `Preprocessing/` é exata. A metade `Pretraining` não descreve esta caixa.**

Medido:

- os 15 valores são **lidos do check-in** — 7 indicadores de categoria, 4 de tempo cíclico
  (seno **e cosseno** de hora do dia **e** dia da semana), 4 de tempo decorrido (`apx_h:95`, `:437`);
- quem mapeia 15 → 64 é a **primeira camada de convolução do próprio Check2HGI** (`apx_h:130-133`),
  que no desenho do autor é o `same GCN`, **depois** da caixa;
- o que **é** pré-treinado existe, mas é outro andar e outro papel: a **âncora da tabela de lugares**
  (`apx_h:167`, `:440`), termo auxiliar do objetivo, a jusante do pooling.

**Nada pré-treinado está no caminho check-in → feature.** Escrever "pre-trained encoder" ali é a
mesma classe do defeito do slide 47 e do bullet 3 do slide 38: um nome que a dissertação não sustenta.
A caixa chama-se **`Preprocessing`**, que é a primeira palavra do próprio autor.

### A faixa do slide 29

```
the check-in itself              category · cyclical time · elapsed time      15 values, one vector per VISIT
                              →  read off the visit, nothing pretrained   →   different at every visit
```

## 6.3 · 🛑 O que estas duas faixas fazem juntas — e é isto que justifica a mudança

Elas não são duas documentações. Lado a lado, **elas entregam o argumento do capítulo**:

| | entrada do modelo |
|---|---|
| **HGI** | **um vetor por TIPO** — todo lugar daquele tipo partilha o mesmo |
| **Check2HGI** | **um vetor por VISITA** — diferente a cada visita |

É literalmente a frase do Cap. 5: *"Both produce one vector per place, so **two visits to the same
coffee shop look identical**"* (`5_mobiwac/02_related.tex:57`). E o slide 28 já se chama
*"Why a per-visit representation is new in this line"*.

> **Por isso as duas caixas finais têm de ter a mesma forma e o contraste em versalete
> (`per TYPE` × `per VISIT`).** É o único par de palavras da faixa que a plateia precisa de reter.
> Se a `ppt` tiver de cortar algo por espaço, **corta as caixas 1 e 2 antes da 3.**

## 6.4 · A frase do slide 20 — 🛑 ela é obrigatória, não opcional

O autor escreveu: *"acredito que podemos inclusive remover a frase textual que está atualmente no
slide"*. **Antes de remover, ele precisa de saber o que o registro dele diz.**

`considerations.md:2712-2713`, sobre este slide, lista as duas linhas que sobrevivem e conclui:

> *"a **1** é o que **explica** o resultado do Cap. 4 e a **2** é a **ressalva obrigatória pela §8.6**"*

**A linha que está hoje na tela é a 2** — a ressalva obrigatória. *(Nota lateral: a linha **1**,
`the place-level output already reflects the region the place belongs to`, **já caiu do slide em
algum momento** e o registro a dava como sobrevivente. Não é o pedido de hoje, mas fica anotado.)*

**Recomendação: comprimir, não remover.** O núcleo obrigatório é a última cláusula.

```latex
% de (30 palavras):
\textbf{Why it is used here, and the limit} --- built for \textbf{urban region representation};
its place-level output is \textbf{repurposed} here for sequential prediction,
\alert{a use the original evaluation does not cover}.

% para (11 palavras):
\textbf{Repurposed here} --- \alert{a use the original evaluation does not cover}.
```

Liberta ~2 linhas para a faixa e mantém a ressalva na tela. **O contexto que sai
(*"built for urban region representation"*) vai para a fala**, onde já está.

⚠ Se mesmo assim o autor quiser a frase inteira fora, é decisão dele e registra-se como `AUT`; mas
então a ressalva **tem de aparecer noutro sítio da trilha principal**, porque a §8.6 não é preferência
de forma.

## 6.5 · A colisão com o `ITEM 1` desta mesma spec

O `ITEM 1` pôs no slide **30** um subtítulo `15 values in, 64 out` e uma linha de rodapé sobre o
mesmo assunto. **Com a faixa no 29, isso passa a estar duas vezes em telas adjacentes.**

**Resolver assim** — o 29 fica com o fluxo, o 30 faz o *zoom* nos três grupos:

- **no 29**: entra a faixa (6.2), que passa a ser o único sítio onde `15` aparece;
- **no 30**: **remover a linha de rodapé** do `ITEM 1 (ii)`, e o `\framesubtitle` volta a descrever o
  conteúdo do slide — sugestão: `The three groups, and one design principle`. Os três bullets ficam.

Fica melhor do que estava: o 29 diz **de onde vem**, o 30 abre **o que tem dentro**.

## 6.6 · Forma, para a `ppt` medir

Faixa centrada, corpo `\footnotesize`, três estágios separados por `$\rightarrow$`, com régua fina
acima e abaixo para ler como diagrama e não como parágrafo. **A mesma macro nos dois slides** — se
divergirem em corpo, cor ou espaçamento, o contraste da 6.3 deixa de funcionar.

Orçamento: **~22 palavras no 20** (que ganha ~2 linhas com o corte da 6.4) e **~20 no 29** (que hoje é
só imagem com `\vfill` dos dois lados). Eu digo o conteúdo; **quantas linhas ocupa é medição sua.**
Se não couber, corte pela ordem da 6.3 — as caixas 1 e 2 antes da 3.

---
---

# RODADA DE 28/08 — seis itens de um ensaio novo

Numeração conferida contra o PDF: **todos os oito slides citados batem com o título.**
`6 = Related work: POI prediction` · `8 = MTL Fundamentals` · `17 = The null result` ·
`27 = Next region…` · `31 = The architecture: sharing by exchange` · `32 = The private spatial path` ·
`34/35 = The protocol, in four steps` (overlays 2 e 3).

Itens **A (slide 6)**, **E (54 h)** e **F (slide 27)** dependem de validação em curso e entram depois.
Abaixo, os três que já fecham.

---

## ITEM B · `MTL Fundamentals` (8) — acrescentar Soft parameter sharing

**Validado, e os dois portões abrem:**

| portão | resultado |
|---|---|
| o volume define? | ✅ `2_fundamentals.tex:942`, `\begin{definition}[Soft parameter sharing]` — **é a Def. 2.11** (2.10 hard, 2.11 soft, 2.12 negative transfer, que é o que o slide já cita) |
| o glossário permite? | ✅ `GLOSSARY.md:194` + `:217-218` — *"Two rows added 2026-08-03 **on the author's authorization**: `soft parameter sharing` and `negative transfer`"*. §8.11 satisfeita |

Texto entregue, verbatim: *"Soft parameter sharing gives each task its own complete network and
couples the networks by **penalizing differences between their parameters**"*.

### 🛑 A restrição de espaço é dura e está registrada

O comentário do próprio slide (`main.tex`, bloco do frame 8) mede: com a frase do TME o frame estoura
**28,49 pt** e *"NÃO cabe nem zerando todo o respiro"*. **O slide 8 está no teto.** Portanto a
inclusão tem de ser **quase de graça**, que é o que o autor pediu.

### Executar — não é bullet novo, é o par dentro do bullet existente

```latex
% de (16 palavras):
        \item \textbf{Hard parameter sharing} (Def.~2.10): every task passes through one shared
              trunk before branching, and separates only at its own output;

% para (24 palavras — líquido +8):
        \item \textbf{Hard parameter sharing} (Def.~2.10) --- one shared trunk; tasks separate
              only at their own output.\\
              \textbf{Soft parameter sharing} (Def.~2.11) --- one network per task, coupled by a
              penalty on their differences;
```

A metade `hard` **encolhe de 16 para 10 palavras**, então o custo real do acréscimo é **+8 palavras**
e a quebra de linha explícita dá o pareamento visual que ele pediu, sem bullet novo e sem cromo de
ambiente.

### Por que isto conecta com o slide 17 melhor do que só nomear o termo

O volume tem a ponte pronta, e ela é a razão de o MoE existir:

> *"**Because hard sharing is rigid and soft sharing requires many parameters**, several
> architectures explore intermediate topologies. Cross-stitch units… The multi-gate
> mixture-of-experts…"* — `2_fundamentals.tex:965-970`

O slide 17 diz *"One shared block may be too restrictive; soft sharing or Mixture-of-Experts models
might fit better"*. Com as **duas pontas** estabelecidas no 8, o 17 deixa de introduzir dois termos e
passa a apontar para **o meio de um eixo que a plateia já viu**. ⚠ **Essa ponte é FALA, não tela** —
o 8 não tem espaço, e uma frase sobre topologias intermediárias no minuto 8 é conteúdo do 17.

---

## ITEM C · `The architecture` (31) e `The private spatial path` (32) — virar sequência

### C1 · Os títulos

```latex
% 31:  {The architecture: sharing by exchange}      →  {The architecture: sharing by exchange (1/2)}
% 32:  {The private spatial path}                   →  {The architecture: sharing by exchange (2/2)}
```

O autor tem razão no diagnóstico: o 32 continua a explicar a arquitetura, e o título atual anuncia
um subtema que é só o **primeiro** dos seus três itens.

### C2 · 🛑 O primeiro item do 32 — cortar a cláusula que ele nomeou, NÃO o item inteiro

Ele pediu para remover *"A branch inside the own model, not a second model."* **Essa cláusula é a
primeira metade do item; a segunda metade é a definição do caminho espacial privado**, e ela não
pode sair junto, por uma razão que só aparece quando se lê o rodapé do slide:

> *"The evidence does not separate the contributions of the shared trunk and **the private spatial
> path**…"*

Esse `alertblock` é **uma das quatro cláusulas contra si mesma que a trilha principal mantém na tela**
(§3, Classe 8 do `HANDOFF_GATE.md`). Com o **título** a deixar de dizer `private spatial path` e o
**item** removido inteiro, o termo apareceria na ressalva **sem referente em tela nenhum**.

```latex
% de (24 palavras):
        \item \textbf{A branch inside the one model}, not a second model --- bypasses the shared
              trunk, feeds the region output only; \alert{the category task never touches it};

% para (16 palavras):
        \item \textbf{The private spatial path} --- bypasses the shared trunk, feeds the region
              output only; \alert{the category task never touches it};
```

**Ganha-se o corte que ele pediu (−8 palavras), a cláusula `not a second model` vai para a fala como
ele quis, e o referente da ressalva volta — agora no item, já que saiu do título.**

### C3 · A caixa `joint-best` migra do 35 para o 32 — e a fala tem de migrar com ela

O argumento dele é bom: `S_joint` é a regra que decide **qual checkpoint é salvo**, portanto é decisão
de treino, e fica melhor ao lado do peso fixo de perda e do logit adjustment.

Mover o bloco (`main.tex:1313-1316`) tal como está:

```latex
    \begin{block}{The joint-best convention $\cdot$ one saved model per fold}
        \centering
        $S_{\mathrm{joint}} = \sqrt{\mathrm{MacroF1} \times \mathrm{Acc@10}}$
    \end{block}
```

🛑 **E aqui está a armadilha, que é a Classe 6 do handoff e o quase-acidente da `ppt` de ontem:
a fala do 35 explica esta caixa e ela NÃO está dentro do frame** — vive no `% FALA:` acima dele.
Mover o bloco sem mover a fala deixa o 35 a falar de uma caixa que já não está lá.

**A fala parte-se em duas, e a divisão não é arbitrária:**

| trecho da fala do 35 | vai para |
|---|---|
| *"os dois resultados saem de um único modelo salvo por partição, escolhido pela média geométrica das duas métricas"* | **fala do 32**, com a caixa. É a **definição** |
| *"a convenção alternativa… é mais favorável ao modelo conjunto, e transformaria mais quatro células de categoria e mais duas de região em melhorias"* | 🛑 **FICA no 35** |

**Por que a ressalva não viaja:** ela é autoincriminatória (Classe 8) e fala em **células que se
tornariam melhorias** — no slide 32 ainda não se viu célula nenhuma, e a frase seria ininteligível.
Ela é sobre **consequência**, e consequência mora onde os resultados estão. O comentário do
`main.tex:1311` já registra que essa ressalva *"continua FALADA (verificado antes de cortar)"*: ela
nunca esteve na tela, então mover a caixa não a desaloja — só não a leve junto.

---

## ITEM D · `2 · what is measured` (34) — macro-F1

**Nenhuma ação. Já está como ele quer.** Verificado no PDF renderizado (página 40): o marcador
`Category: macro-F1` está intacto, com o piso de classe majoritária, o caso da Flórida (24,7% / 5,7)
e a razão de não ser acurácia simples. A única mudança de hoje nesse slide foi a fusão do OOD dentro
do marcador do `Acc@10`, que não tocou no de categoria.

---

## ITEM E · o custo experimental — `3 · what is compared` (impresso 35)

**Autor: seguir.** Ancorado por CONTEÚDO: a linha das sementes está no **35**. *(Ele escreveu "37"
uma vez; o 37 é a tabela do `Result 1`. O 35 é onde `Four seeds {0,1,7,100} × five folds` vive, e foi
onde a primeira mensagem dele o colocou.)*

Acrescentar ao marcador das sementes:

```latex
        \item Four seeds \textbf{\{0, 1, 7, 100\}} $\times$ five folds $=$ \alert{20 fitted models}
              --- {\footnotesize$\approx$ \textbf{60 h} of GPU time (A40 $+$ H100/A100),
              joint model and the two dedicated models};
```

### Por que não `54 h [A40]`

| o que ele queria | o que está registrado |
|---|---|
| `≈ 54 h` | **59,75 h** — `docs/studies/closing_data/v18/PROGRESS.md:21`, quadro completo 72/72 células |
| — | o `54,95 h` é a **mesma linha 21 no `mtlcheck`**, um snapshot de **64/72**: faltavam CA e TX nas sementes 7 e 100 |
| `[NVIDIA A40]` | hardware **misto por desenho**: 39 células A40 (44,44 h), 19 H100, 8 A100-40GB, 2 A100-80GB. **O A40 cobre 74%** |

Duas ressalvas que a redação acima já respeita, e por isso ela diz o que diz:

1. **escopo** — os 59,75 h cobrem **o conjunto + os dois dedicados** (6 datasets × 4 sementes ×
   5 folds × 3 famílias = 72 células). **Não** incluem baselines externas, construção do substrato,
   nem as waves descartadas (que somam mais ~74 h). Por isso a frase nomeia o escopo;
2. **`≈ 60` e não `59,75`** — `measured wall-clock total` é `sum(wall_seconds)/3600`, **soma por
   célula, não relógio de calendário**. Estados pequenos correram 2-wide e as lanes alugadas em
   paralelo, então o tempo decorrido foi **menor**. Um número redondo com `≈` não promete precisão
   que a medição não tem; `59,75` prometeria.

---

## ITEM F · `Next region` (27) — o bloco `The Task` sai, e a frase muda de argumento

**Autor: *"realmente estávamos errados; region ainda é uma tarefa difícil, não à toa usamos
Acc@10."*** A conclusão dele está certa. ⚠ **Mas a razão que ele deu não pode ir para a tela.**

### Por que o argumento do Acc@10 não entra

O volume **não** justifica o Acc@10 pela dificuldade. Ele o define
(`2_fundamentals.tex`, eq. `acc10`) e declara os limites dele — *"It does not distinguish first place
from tenth and does not measure the probability assigned to the true [region]"*. A razão registrada
para uma métrica de lista é **operacional**: *"A mobility-aware service acts on **which region will
be busy**, rather than on a single position in the ranking"* (`05_setup.tex:119`).

🛑 **E dizer "usamos Acc@10 porque é difícil" lê-se como concessão** — *"escolhemos uma métrica
generosa"* — imediatamente antes dos slides de resultado. Há ainda uma decisão registrada contra
reabrir isto: `2_fundamentals.tex:1483`, **"Do not reintroduce a second metric"**. E a ressalva
honesta **já está na tela**, no slide 34: *"it does not separate first place from tenth"*.

### Executar

**Remover o marcador `The task`** inteiro. O intervalo `520 … 8.501` não se perde: as contagens de
região já estão na **tabela de datasets** (verificado, `8,501` aparece lá). Em seu lugar, uma linha:

```latex
        \item \textbf{The target} --- a region covers a larger area than a place. It is an easier
              target than the exact place, but the task is not easy;
```

**Por que esta forma passa nas três leis:**

- **contra o volume:** concede exatamente o que o volume concede — `01_introduction.tex:14`
  (*"Two coarser questions are usually enough"*) e `03_problem.tex:13` (*"both properties are easier
  to learn"*). A formulação antiga (*"that does not make it easier"*) **contradizia as duas**;
- **contra os dados:** não usa contagem de classes, que está refutada — região é espaço **menor** que
  lugares nos seis conjuntos (520 vs 29.816 em Istambul; 8.501 vs 169.145 em CA). Há precedente: a
  razão `8.501/520` já foi **deletada do volume** por ser aritmética em prosa;
- **contra a regra de leitura** (`WRITING_LAW`, o teste do autor: *"pode um leitor não-nativo
  absorvê-la numa única leitura?"*): duas frases curtas, uma ideia cada, sem travessão dentro de
  oração completa e **sem `coarser`**, que ele próprio apontou como palavra difícil no oral.

**Onde fica a evidência:** o **piso de Markov-1** (`51 a 72 Acc@10`) chega no **34** e o pagamento
chega no **38** (*"HMT-GRN stays below that floor at all six"*). O 27 planta a ideia, os outros dois
entregam o número. **Não repetir o piso no 27** — seria a única ocorrência antecipada e o autor pediu
brevidade.

---

## ITEM A · `Related work: POI prediction` (6)

**Autor: *"vamos deixar como tá e deixar essa ideia, só tirando o HST-LSTM."*** A tabela de unidade de
representação **não entra**. Só a correção factual:

```latex
% de:
              \textbf{dominant target in the field} --- recurrent: ST-RNN, DeepMove, HST-LSTM,
              Flashback; attention: STAN, GeoSAN, GETNext;
% para:
              \textbf{dominant target in the field} --- recurrent: ST-RNN, DeepMove,
              Flashback; attention: STAN, GeoSAN, GETNext;
```

**Por que ele sai:** o slide afirma que todo modelo ali nomeado prediz *"the **exact
establishment**"*. O HST-LSTM não prediz — ele prediz uma **AOI**, que o artigo define como
*"a functional zone that offers same geographical function … which covers certain area on digital
maps and **contains various individual POIs**"* (IJCAI 2018, §3), sobre **9.000 AOIs**. Com ele na
lista, a frase é falsa. Sem ele, é verdadeira para os seis restantes — verificado um a um.

---
---

# ITEM 7 · o preprocessing volta a ser TEXTO — `HGI` (20) e `Check2HGI` (29)

**Autor: a faixa-diagrama sai.** *"Minha intenção original era algo bem mais simples: adicionar
apenas mais um item textual… quero aproveitar o espaço do slide e manter a arquitetura principal como
elemento visual dominante."* **Remover o `\pipefaixa` dos dois slides.**

## 7.1 · A sequência do HGI, validada — e uma correção ao pressuposto

⚠ O autor avisou para não assumir que `Delaunay`, `Node2Vec`, `skip-gram` e `hierarchical category
loss` pertencem à mesma etapa. **Conferi, e pertencem — todas as quatro estão DENTRO do encoder.**

O ponto que eu próprio tinha errado antes: o Cap. 2 diz que a convolução sobre o grafo de Delaunay
vem **depois** do encoder (`2_fundamentals.tex:442`), o que sugeria Delaunay fora dele. **Mas os
passeios do Node2Vec correm SOBRE o grafo de Delaunay** — `poi2vec.py:224`,
`edges_file: Path to edges.csv (Delaunay graph)`. **Delaunay aparece duas vezes**: uma como o grafo
onde os passeios andam (dentro do encoder), outra como o grafo que a GCN do HGI convolve (depois).

**A fonte de record é o Cap. 4, que descreve este pipeline em prosa e com as citações**
(`4_courb/methodology.tex:160-215`), literal:

> *"The POIs are organized into a spatial graph built by **Delaunay triangulation** over the
> geographic coordinates… Over this graph, random walks are executed following the **Node2Vec**
> methodology `\cite{grover2016node2vec}`. Each walk is converted into a sequence of secondary
> categories, the fine classes… The model learns the embeddings using the **skip-gram** strategy with
> **negative sampling** `\cite{mikolov2013word2vec,mikolov2013negsampling}`… The implementation
> incorporates a **hierarchical regularization term** `\cite{Xu2023}` between category and fine
> class… In the end, the resulting embedding is **generated per category and remapped to each POI**."*

*(A última cláusula confirma o que eu medi no artefato: 11.848 lugares → 284 vetores distintos.)*

## 7.2 · As citações — o padrão existe e é `Autor et al., Ano`

O deck **já cita**, inline e entre parênteses: `(Song et al., 2010)` na trilha principal,
`Standley et al. (ICML 2020)` e `Karpathy (2019)` na Série B. **Mesmo padrão.**

| método | citação | chave no volume |
|---|---|---|
| Node2Vec | **Grover & Leskovec, 2016** | `grover2016node2vec` |
| skip-gram + negative sampling | **Mikolov et al., 2013** | `mikolov2013word2vec`, `mikolov2013negsampling` |
| hierarchical category loss | **Xu et al., 2023** | `Xu2023` (TME) |
| HGI | **Huang et al., 2023** | `huang2023hgi` |

🛑 **Delaunay NÃO recebe citação, e é deliberado.** **Não existe entrada de Delaunay no
`references.bib`**, e o volume usa o termo **sem citar** nos quatro sítios onde aparece
(`3_cbic:23`, `4_courb:163`, `2_fundamentals:442`, `apx_h:80`). Citar aqui inventaria uma referência
que a dissertação não tem. O pedido do autor foi *"utilize prioritariamente as referências já
presentes na dissertação"* — e para Delaunay não há nenhuma.

## 7.3 · Executar — slide 20

**Sai o `\pipefaixa`. Entra um marcador**, ao lado da ressalva já comprimida:

```latex
        \item \textbf{Preprocessing} --- places $\to$ \textbf{Delaunay} graph $\to$
              \textbf{Node2Vec} walks {\scriptsize(Grover \& Leskovec, 2016)} $\to$
              \textbf{skip-gram} with negative sampling {\scriptsize(Mikolov et al., 2013)}
              $+$ a \textbf{hierarchical category loss} {\scriptsize(Xu et al., 2023)} $\to$
              one \textbf{64-d} vector per place type $\to$ \textbf{HGI};
```

✅ **Isto devolve o slide ao orçamento REGISTRADO** — `considerations.md:2709`: *"a figura + duas
linhas"*. Passa a ter exatamente duas: esta e a ressalva obrigatória da §8.6.

💰 **E provavelmente paga a dívida da `T9`.** A `ppt` reduziu a chapa `hgi_flow` para **0,70** para
caber a faixa, com autorização do autor e dívida declarada. **Com a faixa fora, medir se a chapa
volta a 1,0** — se voltar, a dívida fecha sozinha e os traços recuperam os 30% de espessura.

## 7.4 · Executar — slide 29

**Sai o `\pipefaixa`.** O slide volta a ser a chapa mais **um** marcador:

```latex
        \item \textbf{Preprocessing} --- the check-in itself $\to$ category indicator $+$
              \textbf{sine and cosine} of hour of day and day of week $+$ four
              \textbf{elapsed-time} values $=$ \textbf{15 values per visit} $\to$
              forward-only visit graph $\to$ \textbf{Check2HGI} exports one \textbf{64-d} vector
              per visit;
```

🛑 **E aqui NÃO há citações a acrescentar, e a razão é substantiva.** O autor pediu referências
*"quando forem métodos provenientes da literatura"*. **A featurização do check-in não é da
literatura** — `apx_h:75-100` descreve os 15 valores e **não cita ninguém**. É construção deste
trabalho. Inventar uma citação aqui seria pior do que não ter nenhuma.

*(O que é da literatura no Check2HGI é o **objetivo** infomax, herdado de `huang2023hgi` /
`velickovic2019deep` — mas isso é o modelo, não o preprocessing, e o slide 28 já o situa.)*

⚠ **`forward-only` está na linha de propósito:** é o dispositivo antivazamento do v18 e o slide 30
o desenvolve. Aqui ele só nomeia; a explicação fica onde está.

## 7.5 · O que o par continua a fazer

Mesmo em texto, o contraste sobrevive e é o motivo de os dois terem a mesma forma:
**`one 64-d vector per place type`** contra **`one 64-d vector per visit`**. É a frase do Cap. 5
(*"two visits to the same coffee shop look identical"*) reduzida a duas metades simétricas.
**Se algo tiver de encolher, encolha o meio das duas linhas, não as pontas.**

---
---

# ITEM 8 · auditoria de nomes e referências — trilha principal

Pedido do autor, tarefa separada. **Auditei os 46 slides numerados da trilha principal.**
*(A Série B é do `extra`; mando-lhe o método e o inventário, não especifico edições lá.)*

## 8.1 · O que foi medido

**35 nomes de método/modelo aparecem em tela.** Extraí-os por varredura de siglas, CamelCase e
nomes hifenizados sobre o PDF **renderizado** (não a fonte — a fonte mente por quebra de linha).

✅ **Os 35 têm chave no `references.bib` do volume. Zero risco de referência inventada.**
*(Quatro pareciam ausentes e eram falso-negativo do meu casamento: `DGI` está sob
`velickovic2019deep` — o título é "Deep Graph Infomax", sem a sigla; `HMT-GRN` sob `Lim2022`;
`SIREN` sob `sitzmann2020implicit`; `Nash-MTL` sob `nash`. Um quinto falso-negativo foi **erro de
sintaxe meu** — usei `\|` num `grep -E`, onde é literal.)*

🔴 **E os dois exemplos que o autor deu são exatamente os dois piores casos do deck:**

| sigla | ocorrências em tela | vezes que o deck a expande |
|---|---:|---:|
| **HGI** | **122** | **0** |
| **DGI** | **9** | **0** |
| GAT | 1 | 0 |
| FiLM | 3 | 0 |
| GCN | 4 | 1 ✓ |
| MTL | 131 | 24 ✓ |
| CTLE | 2 | 1 ✓ |
| LBSN | 1 | 1 ✓ |

**Cento e vinte e duas ocorrências de `HGI` e a defesa nunca diz o que a sigla significa** — num deck
cujo Capítulo 4 inteiro é uma comparação contra ela.

✅ **E o padrão de citação já existe e está vivo**: `(Song et al., 2010)` na trilha principal,
`(Grover & Leskovec, 2016)` · `(Mikolov et al., 2013)` · `(Xu et al., 2023)` no slide 20 desde o
`ITEM 7`, e `Standley et al. (ICML 2020)` na Série B. **Não é preciso inventar formato.**

## 8.2 · TIER 1 — entra (três edições, e é o que o autor pediu literalmente)

| slide | hoje | passa a ser |
|---|---|---|
| **13** (título `DGI`, chapa) | `DGI` | `Deep Graph Infomax (DGI)` + `(Veličković et al., 2019)` |
| **20** (título `HGI`, chapa) | `HGI` | `Hierarchical Graph Infomax (HGI)` + `(Huang et al., 2023)` |
| **19** (`Architecture or representation?`) | *"one monolithic 64-dimensional place embedding (DGI)"* | fica — a expansão já terá acontecido no 13 |

⚠ **Eu digo o CONTEÚDO; onde ele cabe é medição da `ppt`.** Os dois são slides de chapa e o título é
de uma linha — o registro mede que **um título de duas linhas custa 3,7 mm**. Se não couber no
título, o `\framesubtitle` resolve. **Não especifico a composição** (`Classe 11`).

**`Check2HGI` não precisa de expansão** — não é inicialismo, e o `GLOSSARY` já o define como
*"extends the place→region→city hierarchy with a fourth check-in level"*. A primeira ocorrência
(slide 28) já vem com essa explicação ao lado.

## 8.3 · TIER 2 — recomendo, e é barato

**Os quatro sistemas externos que têm NÚMERO em tela** (slides 38 e 39). Um número atribuído a um
sistema sem referência é a pergunta mais fácil de fazer, e o rodapé do 38 **já os nomeia aos quatro**
— só faltam os anos:

```
HMT-GRN (Lim et al., 2022) · ReHDM (Li et al., 2025) · STAN (Luo et al., 2021)
POI-RGNN (Capanema et al., 2023)
```

**Os três codificadores do slide 21** (`Why these encoders`), que são componentes da contribuição do
Cap. 4 e não paisagem: `SIREN (Sitzmann et al., 2020)` · `Sphere2Vec (Mai et al., 2023)` ·
`Time2Vec (Kazemi et al., 2019)`.

## 8.4 · TIER 3 — 🛑 recomendo NÃO fazer, e a razão é de desenho

Restam **~25 nomes**, todos em **enumerações de paisagem**: o slide 6 (`ST-RNN, DeepMove, Flashback,
STAN, GeoSAN, GETNext`), o 7 (`CatDM, DRRGNN, CSLSL, HMRM, MCMG`), o 8 (`PCGrad, Nash-MTL, GradNorm,
DWA, FAMO, MGDA, CAGrad, Aligned-MTL, MCARNN, HAMTL`) e o 9 (`GCN, GAT, DeepWalk, GraphSAGE`).

**Todos têm chave. Nenhum deve ser citado em tela**, por três razões que se somam:

1. **Eles não sustentam alegação nenhuma.** Existem para mostrar que a paisagem é densa. O argumento
   do slide 6 é *"nenhum é baseline direto"*; o do 8 é *"são duas classes"*. Trocar oito nomes por
   oito nomes-com-ano não fortalece nem um nem outro;
2. **O custo é proibitivo e é medido.** O slide 8 está a **+1,59 pt** e dez citações dobram o
   marcador. O registro já mostra que uma frase de 27,2 pt não coube nesse slide;
3. **Quebraria o próprio padrão.** O deck cita **onde o nome carrega peso** — Song para o limite de
   previsibilidade, as quatro do preprocessing. Citar tudo apaga essa distinção.

> **O critério que proponho, e é o que separa os três níveis:** *cita-se o nome de que uma
> alegação da defesa depende* — porque a representação vem dele (DGI, HGI, os três codificadores) ou
> porque há um número na tela atribuído a ele (os quatro externos). **Nome de paisagem não se cita.**

## 8.5 · Série B — rota, não especificação

O mesmo inventário aplica-se lá, e a Série B tem mais nomes por slide. **Mando ao `extra` o método,
a tabela de chaves e o critério da 8.4**, para ele decidir com as medições dele. Não especifico
edições fora da trilha principal.

## 8.6 · Uma lição de instrumento, minha, nesta auditoria

Extraí o deck para um `deck.pkl` no início e raciocinei sobre ele enquanto a `ppt` aplicava o
`ITEM 7`. **Concluí que o slide 20 tinha perdido o número do frame** — não tinha; o meu instantâneo
é que era anterior à edição.

> **Um cache de um artefato que outra sessão está a editar é um artefato diferente.** Se a auditoria
> demora mais do que um lote do par, re-extraia antes de concluir — ou trabalhe sobre o PDF, sempre.

---

## ITEM 8b · o autor derrubou o TIER 3 — referência em TUDO

*"Temos que colocar referência em tudo, MTL, trabalhos relacionados e afins, tudo que tem na
dissertação."* **A triagem da 8.4 fica registrada como recomendação minha rejeitada. Executa-se tudo.**

### Resposta à pergunta dele: sim, agora todos

A primeira passada cobriu os **47 numerados**. Refiz sobre as **104 páginas**, incluindo os **57 da
Série B**. A Série B **não acrescenta métodos da literatura** além de `MMoE`, `GRU`, `AdamW`, `ReLU`,
`GELU` — o resto dos nomes novos lá são ficheiros, apêndices e códigos de slide.

### A tabela completa — 43 nomes, todos verificados na entrada do `.bib`

⚠ **Nenhum ano foi derivado da chave.** Seis entradas não têm campo `year` no sítio esperado
(`chen2018gradnorm`, `liu2023famo`, `yu2020pcgrad`, `senushkin2023aligned`, `velickovic2019deep`,
`kazemi2019time2vec`) e foram lidas linha a linha.

| slide | nome | citação | chave |
|---|---|---|---|
| 2 | **MTL** | Caruana, 1997 | `caruana1997multitask` |
| 2 | LBSN | Yang et al., 2015 | `yang2015tsmc` |
| 6 | ST-RNN | Liu et al., 2016 | `liu2016strnn` |
| 6 | DeepMove | Feng et al., 2018 | `feng2018deepmove` |
| 6 | Flashback | Yang et al., 2020 | `yang2020flashback` |
| 6 | STAN | Luo et al., 2021 | `luo2021stan` |
| 6 | GeoSAN | Lian et al., 2020 | `lian2020geosan` |
| 6 | GETNext | Yang et al., 2022 | `yang2022getnext` |
| 7 | HMT-GRN | Lim et al., 2022 | `Lim2022` |
| 7 | CatDM | Yu et al., 2020 | `yu2020catdm` |
| 7 | DRRGNN | Zhu et al., 2022 | `zhu2022drrgnn` |
| 7 | CSLSL | Huang et al., 2024 | `huang2024cslsl` |
| 7 | POI-RGNN | Capanema et al., 2023 | `capanema2023poirgnn` |
| 7 | ReHDM | Li et al., 2025 | `li2025rehdm` |
| 7 | Markov | Gambs et al., 2012 | `gambs2012mmc` |
| 8 | GradNorm | Chen et al., 2018 | `chen2018gradnorm` |
| 8 | DWA | Liu et al., 2019 | `liu2019dwa` |
| 8 | FAMO | Liu et al., 2023 | `liu2023famo` |
| 8 | MGDA | Sener & Koltun, 2018 | `sener2018mgda` |
| 8 | PCGrad | Yu et al., 2020 | `yu2020pcgrad` |
| 8 | CAGrad | Liu et al., 2021 | `liu2021cagrad` |
| 8 | Nash-MTL | Navon et al., 2022 | `nash` |
| 8 | Aligned-MTL | Senushkin et al., 2023 | `senushkin2023aligned` |
| 8 | MCARNN | Liao et al., 2018 | `Liao2018` |
| 8 | HAMTL | Wang et al., 2025 | `wang2025hamtl` |
| 9 | DeepWalk | Perozzi et al., 2014 | `perozzi2014deepwalk` |
| 9 | GCN | Kipf & Welling, 2017 | `kipf2017gcn` |
| 9 | GAT | Veličković et al., 2018 | `velivckovic2017graph` |
| 9 | GraphSAGE | Hamilton et al., 2017 | `hamilton2017graphsage` |
| 10 | Massive-STEPS | Wongso et al., 2025 | `wongso2025massivesteps` |
| **13** | **Deep Graph Infomax (DGI)** | **Veličković et al., 2019** | `velickovic2019deep` |
| 14 | FiLM | Perez et al., 2018 | `perez2018film` |
| **20** | **Hierarchical Graph Infomax (HGI)** | **Huang et al., 2023** | `huang2023hgi` |
| 20 | Node2Vec | Grover & Leskovec, 2016 | ✅ já em tela |
| 20 | skip-gram | Mikolov et al., 2013 | ✅ já em tela |
| 21 | Time2Vec | Kazemi et al., 2019 | `kazemi2019time2vec` |
| 21 | SIREN | Sitzmann et al., 2020 | `sitzmann2020implicit` |
| 21 | Sphere2Vec | Mai et al., 2023 | `mai2023sphere2vec…` |
| 21 | Space2Vec | Mai et al., 2020 | `mai2020multiscale…` |
| 27 | MCMG | Sun et al., 2024 | `sun2024mcmg` |
| 28 | CTLE | Lin et al., 2021 | `lin2021ctle` |
| B | MMoE | Ma et al., 2018 | `ma2018mmoe` |
| B | PLE | Tang et al., 2020 | `tang2020ple` |

**HMRM** aparece no slide 7 e é o único cujo mapeamento eu não fechei — as candidatas no volume são
`Halder2022`, `Xia2020`, `Zhang2020`, `chen2020modeling`, `zeng2019next`. **Deixo-o sem citação até o
`ppt` ou eu resolvermos qual é**, em vez de arriscar a errada. Ou sai da lista, se o autor preferir.

### 🛑 O custo, medido — e a decisão de FORMA que ele precisa de tomar

O formato completo custa **~18 caracteres por nome**. Nos quatro slides densos:

| slide | nomes | custo em forma completa | estado atual |
|---|---:|---:|---|
| **8** | 10 | **~180 car.** | **+1,59 pt** — já no teto |
| 6 | 6 | ~108 car. | — |
| 7 | 6 | ~108 car. | — |
| 9 | 4 | ~72 car. | — |

**Recomendação de forma, e é só forma:** completa onde o nome carrega peso; **só o ano** nas quatro
enumerações — `PCGrad (2020) · Nash-MTL (2022) · GradNorm (2018)`. Custa **~7 caracteres** em vez de
18, mantém a lista escaneável, e o ano é ancoragem suficiente para quem tem o volume na mão.

⚠ **Isto é decisão do autor, não minha.** Se ele quiser a forma completa nos quatro, **alguma coisa
sai do slide 8** — ele está a +1,59 pt e o registro já mostra que uma frase de 27,2 pt não coube lá.
**Eu digo o conteúdo; a `ppt` mede; o autor escolhe o que sacrifica.**

---

## ITEM 8c · MAPA FINAL DE CITAÇÕES — depois da auditoria adversarial

**Veredito da auditoria (Fable, instruído a REFUTAR):** das 41 triplas,
**zero chaves inexistentes, zero autores errados, zero anos errados, zero artigos trocados.**
*(Testou explicitamente a troca `velickovic2019deep` ⇄ `velivckovic2017graph` — DGI e GAT **não**
estão invertidos.)* Dois defeitos de mérito e três correções ao meu mapa de primeira ocorrência.

### 🔴 D1 · `LBSN` estava com a chave errada — segundo a prática do próprio volume

Eu mapeei `yang2015tsmc`. **Existe e bate**, mas é o paper do benchmark Foursquare NYC/TKY, que o
volume usa **só** como *"another common benchmark"* — um dataset que a dissertação **não usa**
(`2_fundamentals.tex:1522`, com o comentário `CONTEXT ONLY` na 1568).

A frase do volume que **define** LBSN — a mesma que o slide 2 espelha (*"its record is the check-in:
a user, a place, and a time"*) — cita **`silva2019urbancomputing`** (`2_fundamentals.tex:27-33`).

> **LBSN → (Silva et al., 2019)**, não Yang. Não era falso; era a chave errada.

### 🟡 D2 · `skip-gram with negative sampling` — dois papers, uma citação renderizada

O bib regista, com verificação: *"Negative sampling is introduced **HERE** (arXiv 1310.4546), **not
in** `mikolov2013word2vec`… The citing sentence at `4_courb.tex:208` claims skip-gram WITH negative
sampling, **so it needs both**."*

✅ **Mas em forma autor-ano os dois renderizam `(Mikolov et al., 2013)`** — a citação em tela cobre
os dois e **não há defeito a corrigir**. Fica registado para a arguição: *"os dois papers de 2013 —
skip-gram no primeiro, negative sampling no segundo."*

### 🟡 D3 · `HMRM` resolvido, com uma armadilha de ano

**`chen2020modeling`** — *"Modeling Spatial Trajectories with Attribute Representation Learning"*,
Chen, Zhao, Liu, Yu, Zheng, TKDE. O volume amarra-o em quatro pontos (`3_cbic/intro.tex:26`,
`3_cbic/results.tex:120,125,145`).

⚠ **O campo `year` do bib é 2022** (tiragem impressa do TKDE), **mas a prosa do Cap. 3 escreve
"Chen et al. (2020)"** e o DOI é `…/TKDE.2020.…`. No volume o estilo numérico esconde a divergência;
**num slide autor-ano ela fica visível.**

> **Citar como (Chen et al., 2020)** — é o que a banca leu no Cap. 3 e o que o DOI diz.
> **Registado aqui para ninguém "corrigir" para 2022 depois.**

### 🔵 D4 · Três erros meus de primeira ocorrência

**`node2vec` e `skip-gram` estreiam no impresso 9**, não no 20 — slide 9, segundo marcador:
*"skip-gram, DeepWalk, node2vec — dense vectors whose geometry reflects relationships in the data"*.
**O meu padrão era sensível a maiúsculas e não casou `node2vec` minúsculo.**

🛑 **Consequência, e ela é a favor do espaço:** pela regra do autor, as citações pertencem ao **9**,
e as duas que estão hoje no slide 20 **saem**. O 20 fica só com `(Xu et al., 2023)`, cuja primeira
ocorrência é lá.
💰 **E isso pode devolver altura ao 20** — a chapa está a 0,80 com dívida `T9` declarada. **Medir se
volta a subir depois de tirar as duas citações.**

**`MMoE`** não aparece em slide numerado nenhum — só no extra (pdf 89). **Sai da trilha principal.**

### 🟢 D5 · Onze nomes que eu tinha esquecido

| nome | slide | citação | nota |
|---|---|---|---|
| Song et al., 2010 | 1 | ✅ já em tela | `song2010limits` |
| Xu et al., 2023 | 20 | ✅ já em tela | `Xu2023` |
| **MHA+PE** | 7, 16 | (Zeng et al., 2019) | `zeng2019next` — baseline do Cap. 3 |
| **iMTL** | 8 | (Zhang et al., 2020) | `Zhang2020` |
| **uncertainty weighting** | 8 | (Kendall et al., 2018) | `kendall2018uncertainty` |
| **Gowalla** | 10, 44 | (Cho et al., 2011) | `cho2011gowalla` (+`jure2014snap`) |
| **Holm** | 36 | (Holm, 1979) | `holm1979` |
| **Wilcoxon** | 36 | (Wilcoxon, 1945) | `wilcoxon1945` |
| Mixture-of-Experts | 17 | — sem chave dedicada | a mais próxima é `ma2018mmoe` |
| **Delaunay** | 9, 13, 20 | 🛑 **sem citação** | **não existe no bib, e o volume também não cita** |
| **Logit adjustment** | 32 | 🛑 **sem citação** | **não existe no bib; o volume usa sem citar** (`04_method.tex:38`) |

**Delaunay e logit adjustment ficam SEM citação, e é decisão fundamentada, não lacuna:** citar
inventaria referência que a dissertação não tem. Se o autor quiser as duas, é acrescentar entradas ao
`references.bib` do volume — decisão dele, e não na véspera.

### O mapa a executar, por slide

| slide | citações que entram |
|---|---|
| **2** | MTL (Caruana, 1997) · LBSN **(Silva et al., 2019)** |
| **6** | ST-RNN (Liu et al., 2016) · DeepMove (Feng et al., 2018) · Flashback (Yang et al., 2020) · STAN (Luo et al., 2021) · GeoSAN (Lian et al., 2020) · GETNext (Yang et al., 2022) |
| **7** | HMT-GRN (Lim et al., 2022) · CatDM (Yu et al., 2020) · DRRGNN (Zhu et al., 2022) · CSLSL (Huang et al., 2024) · POI-RGNN (Capanema et al., 2023) · ReHDM (Li et al., 2025) · Markov (Gambs et al., 2012) · HMRM **(Chen et al., 2020)** · MHA+PE (Zeng et al., 2019) |
| **8** | uncertainty weighting (Kendall et al., 2018) · GradNorm (Chen et al., 2018) · DWA (Liu et al., 2019) · FAMO (Liu et al., 2023) · MGDA (Sener & Koltun, 2018) · PCGrad (Yu et al., 2020) · CAGrad (Liu et al., 2021) · Nash-MTL (Navon et al., 2022) · Aligned-MTL (Senushkin et al., 2023) · MCARNN (Liao et al., 2018) · iMTL (Zhang et al., 2020) · HAMTL (Wang et al., 2025) |
| **9** | **skip-gram (Mikolov et al., 2013)** · DeepWalk (Perozzi et al., 2014) · **node2vec (Grover & Leskovec, 2016)** · GCN (Kipf & Welling, 2017) · GAT (Veličković et al., 2018) · GraphSAGE (Hamilton et al., 2017) |
| **10** | Gowalla (Cho et al., 2011) · Massive-STEPS (Wongso et al., 2025) |
| **13** | DGI ✅ já aplicado |
| **14** | FiLM (Perez et al., 2018) |
| **20** | HGI ✅ já aplicado · Xu ✅ · 🛑 **remover** Grover e Mikolov (repetem o 9) |
| **21** | Time2Vec (Kazemi et al., 2019) · SIREN (Sitzmann et al., 2020) · Sphere2Vec (Mai et al., 2023) |
| **27** | MCMG (Sun et al., 2024) |
| **28** | CTLE (Lin et al., 2021) |
| **36** | Holm (1979) · Wilcoxon (1945) |

**Slides de resultado (37–47): nenhuma citação.** Tudo já foi citado antes — é o ganho da regra de
não repetir.

⚠ **Os dois slides mais apertados são o 8 (+1,59 pt, doze citações) e o 21 (+3,10 pt, três).**
A `ppt` mede os treze **antes** de aplicar em qualquer um e devolve a lista dos que não comportam.

---
---

# RODADA DOS ENSAIOS — quatro pontos onde a plateia parou para perguntar

Numeração conferida no PDF atual: `6 = Related work: POI prediction` · `11 = The metric all three
studies share` · `14 = MTLnet` · `22 = Architecture or representation?` ·
`31 = The architecture: sharing by exchange (1/2)` · `34 = 2 · what is measured`. **Os seis batem.**

Itens **11** e **14/22/31** dependem de validação em curso (como os dedicados são construídos, e a
arquitetura real de cada head nos três estudos). Abaixo, os dois que fecham.

## ENSAIO-A · `2 · what is measured` (34) — cortar, e o Acc@10 fica direto

**Verificado antes de cortar (`Classe 8`):** a fala do 34 carrega **tudo** o que sai da tela —
a Flórida (*"vinte e quatro vírgula sete por cento… e ainda assim marca cinco vírgula sete"*), o
*"acurácia simples não é a métrica aqui"*, o *"não separa o primeiro lugar do décimo"* e o
*"região ausente do treino conta como erro"*. **E não há decisão registrada prendendo nada à tela**
— o exemplo da Flórida é **explicativo**, não uma ressalva contra o trabalho, então a fala é destino
legítimo por essa mesma classe.

```latex
    \begin{itemize}\setlength{\itemsep}{2ex}
        \item \textbf{Category: macro-F1} --- the mean of the per-category F1; reference point the
              \textbf{majority-class floor}, \alert{5.7 to 7.3};
        \item \textbf{Region: Acc@10} --- \textbf{the model returns a Top 10; we measure how often
              the true region is in it}. \emph{It does not separate first from tenth}, and
              \textbf{a region unseen in training counts as an error};
        \item \textbf{Reference points for region} --- the \textbf{dedicated model} and the
              \textbf{Markov-1 floor}, \alert{51 to 72}.
    \end{itemize}
```

**~95 → ~62 palavras.** O que sai é o exemplo da Flórida (o maior bloco) e a razão da macro-F1,
ambos vivos na fala. **O que FICA são as duas divulgações auto-limitantes** — *não separa o primeiro
do décimo* e *região não vista conta como erro*. Essas não descem para a fala.

A formulação do autor foi adotada com um encurtamento: *"The model predicts a Top 10; we measure how
often the correct region is within that Top 10"* → **`the model returns a Top 10; we measure how
often the true region is in it`**. Diz o mesmo em 15 palavras em vez de 20, e mantém **`Top 10`**
literal, que era o ponto — foi essa a dúvida do ensaio.

## ENSAIO-C · `Related work: POI prediction` (6) — definir POI e check-in

Fonte no volume, não paráfrase:

> *"states that a user visited a place, or **point of interest (POI)**, at a given time"*
> — `1_introduction.tex:39`
> **Definição de check-in** (`2_fundamentals.tex:84-91`): $x_i=(u,p_i,t_i,c_i,r_i)$ — *"where $p_i$
> is the **visited POI**, $t_i$ is its **timestamp**, $c_i$ is its **category**, and $r_i$ is its
> **region**"*

Entra **antes** do `itemize`, porque define o vocabulário que os marcadores usam:

```latex
    {\footnotesize
    \textbf{POI} --- a place, with a category and a region.\qquad
    \textbf{Check-in} --- one recorded visit: a user, a place, a time.\par}
    \medskip
```

**A oposição é a carga:** `a place` × `one recorded visit`. Ambas as metades saem direto da
definição entregue — a tupla do check-in **contém** o POI, que é exatamente a relação que o autor
quer tornar visível. **Duas linhas, 18 palavras. Não vira bloco.**

⚠ **Se não couber numa linha, quebra entre as duas definições — nunca dentro de uma.** A oposição
lê-se pelo paralelismo; partida ao meio, ela desaparece.

---

## ENSAIO-B · `The metric all three studies share` (11) — definir o Dedicated Model

**A lembrança do autor era *"a cabeça do MTL isolada"*. Validado nos três capítulos: `parcialmente`,
e a nuance é o que faz a definição sobreviver à pergunta seguinte.**

| cap. | tem dedicado? | como é construído |
|---|---|---|
| **3** (MTLnet) | sim, `Single` | a **mesma cabeça**, treinada sozinha **direto sobre o embedding de 64 d** — sem encoder por tarefa, sem FiLM, sem tronco. ⚠ Mesmo *desenho*, hiperparâmetros diferentes (next 4 camadas vs 2; categoria token_dim 16 vs 64) |
| **4** (ST-MTLNet) | 🛑 **NÃO EXISTE** | o capítulo não constrói, não treina e não reporta braço single-task nenhum. As tabelas têm três colunas — MTLnet, ST-MTLNet_SIREN, ST-MTLNet_Sphere2Vec-M. O eixo é entrada monolítica × decomposta **dentro da mesma arquitetura conjunta** |
| **5** (conjunto) | sim, dois | **região: literalmente sim** — a torre privada é réplica fiel (`NextHeadSTAN`, mesmo `d_model=128`, 4 cabeças, dropout 0,3); o código diz que processa *"exactly as the STL reg head does"*. **categoria: não** — o dedicado é `next_gru` sobre entrada de 64, sem nada antes |

**A formulação segura — verdadeira no 3 e no 5, e que não afirma nada sobre o 4:**

```latex
        \textbf{Dedicated model} --- the task's own head, trained alone on the same input,
        without the shared trunk.
```

✅ **`Two reference points` → `Three reference points`** funciona: o bloco **não** alega ser dos três
estudos. Medido — `majority` e `markov` aparecem **só no Cap. 5**; `dedicated/single-task` aparece no
**3 e no 5**. Ou seja, dos três pontos, o dedicado é o **mais** compartilhado, não o menos.

🛑 **Nunca escrever *"é a cabeça do MTL isolada"* sem o resto.** No Cap. 3 os hiperparâmetros diferem;
no Cap. 5 vale para região e **não** para categoria. A frase acima é verdadeira nos dois porque diz
*"the task's own head"* (o desenho) e não *"the same module"* (a instância).

## ENSAIO-D · as arquiteturas das heads — slides 14, 22 e 31

⚠ **Os dois primeiros agentes divergiram, e a divergência tinha causa:** o Cap. 3 **não descreve as
cabeças** (`3_cbic/method.tex:90-91` diz só *"dedicated, unshared task-specific heads"*; a seção que
o faria **nunca existiu** — o rótulo `sec:method:single_task_heads` cai na subseção de Dataset,
errata registrada). Quem descreve o MTLnet em prosa é o **Cap. 4**. Um agente leu o texto, o outro
leu artefatos de rodada da era CBIC. **O texto entregue vence.**

**Fonte única para 14 e 22** — `4_courb/methodology.tex:34-37`, verbatim:
> *"The **category classification module** uses a set of **three parallel MLPs with varying depths
> (2, 3, and 4 layers)**, whose outputs are concatenated and projected to the 7 POI classes.
> The **next-POI head** employs a **Transformer encoder with 8 attention heads and 4 layers**, a
> **causal mask** for autoregressive processing, and **attention-weighted pooling**."*

### Slide 14 (`MTLnet`)
```latex
        \item \textbf{Heads} --- category: \textbf{three parallel MLPs} (2, 3, 4 layers),
              concatenated; next category: \textbf{Transformer encoder}, 8 heads, 4 layers,
              causal mask;
```

### Slide 22 (`Architecture or representation?`)
```latex
        \item \textbf{Heads unchanged from MTLnet} --- only the input moves.
```
**Uma linha, e ela é o argumento do slide**, não um detalhe: o Cap. 4 congela a arquitetura de
propósito (`methodology.tex:90`). Repetir as duas arquiteturas aqui contradiria a mensagem.

### Slide 31 (`The architecture: sharing by exchange (1/2)`)
```latex
        \item \textbf{Heads} --- category: \textbf{4-layer GRU}; region: \textbf{two STAN routes},
              one on the raw region window, one on the shared context;
```
Fonte: `apx_h_check2hgi_joint_model.tex` (cabeça de categoria `next_gru`; as duas rotas STAN, a
privada citando `\cite{luo2021stan}`).
🛑 **Sem citação de STAN aqui** — a primeira ocorrência é o slide 6 e a regra do autor é não repetir.

### ⚠ Uma nota sobre o `SIREN` do slide 21, já em tela

O volume cita SIREN **de duas formas, de propósito**, e a linha que o mostra é
`4_courb/related.tex`: *"SIREN methodology (Sinusoidal Representation Networks)
`\cite{sitzmann2020implicit}`, **applied to the geographic context**
`\cite{russwurm2024geographiclocationencodingspherical}`"*. A subseção §SIREN do Cap. 4
(`methodology.tex:119`) cita **Rußwurm**.

**O que está em tela (`Sitzmann et al., 2020`) é o paper do método e não está errado.** Registo a
duplicidade para a arguição: *"Sitzmann é o SIREN; Rußwurm é o SIREN aplicado a coordenadas
geográficas, que é o uso deste capítulo."* **Não mexer na véspera por causa disto.**

---
---

# ITEM 9 · `4 · how it is decided` (36) — δ explícito, o quinto item sai, e o que faltava

## 9.1 · Os 2 pp: confirmados, e o autor tinha razão em mandar conferir

⚠ **A primeira coisa que encontrei no código era a errada, e é exatamente a armadilha que ele
antecipou.** `scripts/finalize_phase3.py:267` passa `tost=0.02` — mas essa chamada compara
**Check2HGI contra HGI** (`c2_reg` vs `h_reg`), que é o eixo do `Result 1`, **não** o teste de
não-inferioridade do veredito. Mesmo número, outra etapa.

**A confirmação que vale é o volume entregue**, `5_mobiwac/05_setup.tex:119`, literal:

> *"The non-inferiority claim states that the joint model is no worse than the dedicated model by
> more than a **two-point margin**. We test this claim with the **two one-sided tests (TOST)**
> procedure~`\cite{lakens2017tost}`. **The analysis plan also fixed the two-point margin in
> advance**."*

Corroborado em `wrapup/ESTUDOS_DEFESA.md:561, 2813, 2847, 2862` — e a 2813 acrescenta o que importa:
**"margem registrada SÓ para região"**.

## 9.2 · O quinto item sai — verificado antes

O autor: *"não é utilizada nem discutida nos resultados que apresentamos posteriormente"*. **Medido:
`Wilcoxon` aparece na trilha principal UMA vez — neste slide — e nunca volta.** A outra ocorrência é
a Série B (`The unit of the test`), que é o lugar certo de um desvio de protocolo.

**E a divulgação não desaparece da defesa:**
- **a fala di-la inteira** — *"E um desvio declarado: o plano registrava Wilcoxon, e com quatro
  sementes o Wilcoxon exato não desce abaixo de 0,0625. Ele não podia decidir nada"*;
- **o rodapé fica** — *"The statistical evaluation protocol was later refined based on the
  literature"*. Com o item fora, **ele passa a ser o traço único em tela de que o protocolo mudou.**
  🛑 **Não o removas junto.**

*(`Classe 8`: a cláusula qualifica uma **escolha de método** sem número associado à tela, e o
handoff registra que essa classe pode viver na fala e na reserva. Não é o caso do `optimistic`.)*

## 9.3 · O que faltava, e é a revisão crítica que ele pediu

**Falta a ASSIMETRIA do plano — e ela é a causa das duas ressalvas que o slide 39 já mostra.**
`05_setup.tex:113`, literal:

> *"**The plan did not define a superiority test for next-region prediction.** Therefore, the two
> next-region gains … are **secondary results outside the plan**. On next category **the plan
> registered no equivalence margin**, so a difference that fails the superiority test is reported as
> **unresolved rather than as a match**."*

**Hoje a plateia encontra as duas consequências no 39** — *"secondary results, outside the registered
plan"* e *"the other five are unresolved"* — **sem nunca ter ouvido a causa.** É o caso exato do que
ele pediu: informação metodológica essencial para interpretar os resultados posteriores.

## 9.4 · Executar

```latex
    \begin{itemize}
        \item Analysis plan written \textbf{before any result was read};
        \item \textbf{Superiority} $\rightarrow$ next category \textbf{only};
        \item \textbf{Non-inferiority} $\rightarrow$ next region \textbf{only} --- TOST
              {\scriptsize(Lakens, 2017)}, \textbf{$\delta = 2$ pp}, fixed in advance;
        \item Paired $t$ $\cdot$ 90\% CI $\cdot$ \textbf{Holm} {\scriptsize(1979)} across the six
              datasets;
    \end{itemize}

    \vspace{1mm}
    {\footnotesize\alert{The plan is asymmetric on purpose: no superiority test for region,
    no equivalence margin for category.}\par}
```

**Balanço: sai o item de ~35 palavras, entra a linha de ~18. O slide ENCOLHE ~17 palavras e ganha a
metodologia que faltava.** O rodapé do asterisco fica intacto.

Três coisas deliberadas:
- **`δ = 2 pp`** em notação, como ele pediu — é mais rápido de identificar que *"two-point margin"*;
- **os dois `only`** carregam a assimetria já nos marcadores; a linha em `\alert` diz a consequência;
- **`TOST (Lakens, 2017)`** nomeia o teste. Ele quer que a plateia identifique *"qual teste, com que
  critério"* em segundos — e `Non-inferiority` sozinho não nomeia o procedimento.

⚠ **`Holm (1979)` e a citação do TOST vêm do `ITEM 8c`** — este slide já estava na lista dos treze.
**Aplica as duas de uma vez, não em duas passagens.** O `Wilcoxon (1945)` que estava previsto para
este slide **cai junto com o item**, e não deve ser reintroduzido.

---
---

# ITEM 10 · `Future work` — reorganizado sobre a espinha da própria tese

Autor: *"hoje o slide tá bem pobre e mal organizado… de forma bem convidativa, didática, que mostre
paixão e que há espaço para muitas melhorias."*

⚠ **Isto reverte a redução de 11 → 3 de ontem, e é decisão dele.** Mas **não** revoga a `AUT-24`:
aquela regra diz *"converter frase em etiqueta… o conteúdo não é cortado, volta para a fala"*. Aqui
a **tela** ganha nove etiquetas e a **fala** continua com três + um fecho (~40 palavras contra as ~34
de hoje). **A carga falada não muda; a paisagem visível muda.**

## 10.1 · O princípio, e ele é do deck, não importado

**Os três determinantes da resposta — e o deck já disse quanto de cada um está fechado.**
Verificado no PDF renderizado, slide de contribuições:

```
▶ Input representation: established by controlled ablation (Chapter 4 is that control);
▶ Architecture: suggested, not isolated;
▶ Scale: a possible condition, not an established cause.
```

E o `Closing`, dois slides depois, responde *"what does the answer depend on?"* com exatamente
**the input representation, the sharing topology, and the scale of the problem**.

> **É daqui que sai o "há muito espaço", e sem inflar nada: dois dos três determinantes o próprio
> deck classifica como apenas SUGERIDOS.** O slide deixa de ser lista de pendências e vira o mapa do
> que a resposta ainda não fixou — dito com as palavras que o deck já usou, e ecoado no slide
> seguinte.

## 10.2 · A tela — três `exampleblock` em `columns`

**Por que `exampleblock`:** é o dispositivo de tom positivo do template (`nesped.sty:512-513`,
título sobre `secondaryshade`) e **o menos gasto do deck — 5 usos, contra 63 de `block` e 22 de
`alertblock`**. No minuto 45 a mudança de cor sinaliza sozinha: *isto já não é limite, isto é
presente*. `columns` é nativo do deck (25 usos).

| **Input representation** | **Sharing topology** | **Problem scale** |
|---|---|---|
| **An inductive Check2HGI** — new places and users, no retraining | **The exact next place** — a third head, same representation | **Region count vs. data volume** — the controlled experiment |
| a hypergraph over sessions | deeper sharing — one trunk, per-task heads | newer traces · finer taxonomies · new cities |
| attention-based graph encoders | one training stage — representation and model together | the geographic size of a miss |

Linha de fecho, **sem botão** (o autor tirou os links da trilha principal para a reserva):

```latex
{\footnotesize Eleven items in the volume, each tied to a named limitation.\par}
```

**Nove portas na tela é o que faz "há muito espaço" ser VISTO em vez de afirmado.** Os seis itens
quietos não precisam de ser lidos — existem para a paisagem.

## 10.3 · Procedência de cada item

| item | fonte | amarra |
|---|---|---|
| inductive Check2HGI | `6_conclusion.tex:400-404` | L3 |
| hypergraph over sessions | `6_conclusion.tex:407-409` + autor `Questions_author.md:120` | L3 |
| attention-based graph encoders | autor, `Questions_author.md:98, 121` | — |
| the exact next place | `6_conclusion.tex:439-446` + autor `:94, 125-126` | L4 |
| deeper sharing (one trunk) | autor `:93` e a metade MMoE de `:123`; MMoE no volume `2_fundamentals.tex:967-969` | — |
| one training stage | autor `:95, 127` | — |
| region count vs. data volume | `6_conclusion.tex:145-147` | — |
| newer traces · taxonomies · cities | `6_conclusion.tex:397-399, 447-448` | L1, L2, L5 |
| **the geographic size of a miss** | `5_mobiwac/07_discussion.tex:77` — *"the geographic size of the error is the quantity that would matter to such a service… left to future work"* | 🆕 **novo, e é o mais ligado a serviço de todos** |

### 🛑 Duas correções de forma a itens do autor, com a intenção preservada

**`GSM++` não vai à tela pelo nome.** A nota dele (`:98, :121`) mistura dois trabalhos e diz
*"no DGI ao invés do GNN"* — o DGI é o **objetivo** de treino; o que se troca é o **encoder** dentro
dele. A intenção que sobrevive e entra: **substituir o encoder convolucional por um baseado em
atenção**. O deck já introduziu `GCN`/`GAT`/`GraphSAGE`; `GSM++` nunca — e §8.11 é fail-closed.

**`deeper sharing` tem licença do próprio volume**, e é a melhor frase de venda do slide:
> *"Nothing here says the gradients **stay orthogonal in a model that shares more of its depth**,
> couples the tasks in a cascade, or shares…"* — `apx_f_cosine.tex:610`

**A tese declara, por escrito, que o achado de gradientes ortogonais não cobre compartilhamento mais
profundo.** É uma porta que o próprio texto deixou aberta — e citá-la na fala é honestidade que
recruta.

## 10.4 · Os três que abrem — o critério é recrutamento, não importância

1. **An inductive Check2HGI** — vende um futuro concreto (*a cidade que cresce*), e é o favorito
   declarado do autor na fala do slide de limitações (*"a que eu mais gostaria de ver feita"*);
2. **The exact next place** — vende proximidade: a tarefa mais visível da literatura está **a uma
   cabeça de distância** da representação que já existe. É o convite mais barato de aceitar;
3. **One training stage** — vende uma pergunta bonita: unificar os dois estágios e descobrir **se o
   task-agnostic sobrevive** quando os gradientes das tarefas chegam à representação. É o item mais
   dele de todos — aparece duas vezes nas notas, uma com `?` no fim.

⚠ **`Region count vs. data volume` fica na tela como líder do eixo C, mas NÃO abre.** O momento
*"a terceira é a que eu devo"* é dos mais honestos da fala e permanece — **mas dívida não recruta,
convite recruta.**

## 10.5 · O que ficou de fora, para poder defender as ausências

- **já entregue:** *"remover a camada embedding"* e *"…ou cross-attention"* (`:123`) — o Cap. 5 fez
  as duas; *"mais features nos nós"* na forma crua (`:118`) — o entregue foi 11 → 15 colunas;
- **concede a tese:** motor composto de dois substratos e roteamento dual — contradizem *"um modelo,
  uma passagem, N tarefas"*;
- **backlog de repositório:** treze memos de `docs/future_works/` com vocabulário que o deck nunca
  introduziu (FAMO, DSelect-K, cross-stitch, RLW, log_T…);
- **lê como conserto:** `evaluation_protocol_cleanup.md` e a busca de hiperparâmetros em TX/CA —
  na tela, *"vamos mudar o protocolo"* lê como *"o nosso estava errado"*. Ficam na reserva.
