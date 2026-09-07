# Revisões MobiWac 2026 — leitura organizada e verificada

> ### 🔴 O ACHADO MAIS URGENTE, e não custa uma palavra do artigo
>
> O R3 escreveu: *"I believe that a researcher working in this area would be unable to reproduce the
> results shown in the paper with the information given in it."* A resposta do artigo a essa acusação
> é a nota de rodapé da página 1, que promete o código.
>
> ✔ **Verifiquei o ramo prometido.** `origin/mobiwac`, HEAD `f9c50218` (2026-08-07):
> **159 ficheiros referem a geração antiga, ZERO referem a nova**, e o `README.md` abre com
> *"MobiWac 2026 — **Anonymous** Code Release"*, numa conferência de revisão simples e com o artigo
> aceite desde 2026-08-26.
>
> **Ou seja: o artigo reporta a geração corrigida e o ramo publica a que tinha o vazamento.** Um
> leitor que siga a nota de rodapé — exactamente o leitor que o R3 diz que falharia — encontra os
> números que retirámos. A promessa era verdadeira quando foi escrita e **tornou-se falsa por acção
> nossa**, ao corrigir os números sem republicar o pacote.
>
> Custa **zero palavras** no artigo e é o único ponto dos três pareceres que se fecha sem tocar no
> texto. Deve ser feito antes de 2026-09-07.


**Artigo.** *Predicting the Next Category and Region of a Visit: A Check-in-Level Multi-Task Study on Mobility Data*
**EDAS.** #3 (1571313639) · The 23rd International Symposium on Mobility Management and Wireless Access — Regular Paper
**Autores.** Vitor Hugo De Oliveira Silva, Germano dos Santos, Fabrício Aguiar Silva (UFV)
**Estado.** `Accepted` · sessão *Session 1: Mobility Management*, seg. 26 Out 2026, 08:30 CET (1.º artigo, 30 min)
**Fonte.** `articles/[mobiwac]/EDAS.TXT` (56 linhas; carimbo do próprio EDAS: *Sun, 06 Sep 2026 15:08:40 -0300*; ficheiro criado no repositório às 21:58:34 de 2026-09-06 — `stat -f "%Sm %N" "articles/[mobiwac]/EDAS.TXT"`)

### A grelha de notas

| Dimensão | R1 | R2 | R3 | média |
|---|---|---|---|---|
| Originality and Impact | *Good Contribution* **4** | *Has merit but mostly incremental* **3** | *Has merit but mostly incremental* **3** | 3,33 |
| Presentation | *Good* **4** | *Readable* **3** | *Some work needed* **2** | **3,00** |
| Technical Correctness | *Good* **5** | *Flaws but Easy to Correct* **3** | *Flaws but Easy to Correct* **3** | 3,67 |
| Relevance and timeliness | *Good* **4** | *Acceptable* **3** | *Acceptable* **3** | 3,33 |
| Technical content and scientific rigour | *Solid work of notable importance* **4** | *Valid work but limited contribution* **3** | *Valid work but limited contribution* **3** | 3,33 |
| Reviewer Familiarity | *Familiar with this area* **3** | *Familiar with this area* **3** | *Familiar with this area* **3** | 3,00 |
| **Recommendation** | *Likely accept (Top 20% but not top 10%)* **4** | *Accept if room (Top 30% but not top 20%)* **3** | *Likely reject (Top 50%, but not top 30%)* **2** | 3,00 |

Duas leituras que a grelha impõe:

1. **Presentation é a dimensão mais fraca** (média 3,00, a única com um 2) **e a única sobre a qual os três revisores efectivamente escrevem.** R1 elogia-a, R2 não a comenta em prosa, R3 dedica-lhe três frases e a acusação de reprodutibilidade.
2. **Technical Correctness é a mais alta (3,67) e a mais exposta.** O 5 do R1 — a nota mais alta de todo o conjunto — foi atribuído a números que a correcção do vazamento invalidou. Ver §0.

---

## 0 · A ressalva que governa a leitura toda

### 0.1 O que os revisores leram

Os três pareceres foram escritos contra a **v1**: o commit **`0834419b`** (2026-07-10 22:56:32 -0300, *"mobiwac 8p: seed caveat out of the abstract…"*), **8 páginas**.

```
git show 0834419b:"articles/[mobiwac]/src/main.pdf" > /tmp/v1.pdf && pdfinfo /tmp/v1.pdf | grep Pages
   → Pages: 8
pdfinfo "articles/[mobiwac]/src/main.pdf" | grep Pages
   → Pages: 11
```

Prosa viva (comentários, comandos, matemática e flutuantes removidos; extractor em `scratchpad/measure.py`): **v1 = 5 442 palavras · actual = 8 267 palavras (+51,9 %)**. O `CONSOLIDATION_PLAN.md:862` mede 8 274 pelo seu próprio critério — a diferença de 7 palavras é ruído de extractor, não divergência.

### 0.2 O que mudou nos números — e a assimetria que ninguém nomeia

Tabela II (representação, categoria macro-F1). `git show 0834419b:"…/tables/tbl2_substrate.tex"` vs. o ficheiro em disco:

| Dataset | v1 check-in | v1 place | v1 Δ | actual check-in | actual place | actual Δ |
|---|---:|---:|---:|---:|---:|---:|
| Istanbul | 54,65 | 26,56 | +28,09 | 35,35 | 29,07 | **+6,29** |
| AL | 55,87 | 26,56 | +29,31 | 30,77 | 29,15 | **+1,62** |
| AZ | 57,13 | 29,50 | +27,63 | 34,51 | 31,93 | **+2,58** |
| FL | 75,15 | 35,53 | +39,62 | 37,36 | 37,13 | **+0,23** |
| TX | 69,95 | 32,48 | +37,47 | 36,32 | 35,33 | **+0,99** |
| CA | 70,26 | 32,31 | +37,95 | 35,62 | 34,74 | **+0,88** |

Tabela III (o modelo conjunto contra o dedicado). **Repare-se na coluna que *não* mudou:**

| Dataset | cat. ded. v1 → hoje | cat. conj. v1 → hoje | reg. **ded.** v1 → hoje | reg. conj. v1 → hoje | Δ região v1 → hoje |
|---|---|---|---|---|---|
| Istanbul | 54,74 → 35,34 | 63,33 → 35,42 | 75,16 → **75,16** | 75,44 → 75,08 | +0,28 → **−0,08** |
| AL | 56,82 → 30,77 | 64,54 → 30,59 | 70,11 → **70,12** | 69,80 → 69,24 | −0,31 → **−0,87** |
| AZ | 56,43 → 34,57 | 65,84 → 34,57 | 59,46 → **59,48** | 59,56 → 59,04 | +0,10 → **−0,44** |
| FL | 74,51 → 37,35 | 79,85 → 37,55 | 76,70 → **76,69** | 77,42 → 76,54 | +0,72 → **−0,16** |
| TX | 69,79 → 36,33 | 77,23 → 36,19 | 64,95 → **64,94** | 67,07 → 66,15 | +2,12 → **+1,21** |
| CA | 70,60 → 35,63 | 77,04 → 35,63 | 63,49 → **63,48** | 65,69 → 64,54 | +2,20 → **+1,06** |

**O tecto dedicado de região é o mesmo a ±0,02 em todos os seis datasets. O modelo conjunto de região caiu entre 0,36 e 1,15.** A inversão de sinal no eixo da região não veio de o baseline subir; veio de o modelo conjunto descer. Isto é coerente com dois factos já registados — `sections/06_results.tex` diz *"the dedicated region models use one fixed configuration throughout"*, e `CAMERA_READY.md:303` (§S6) regista que *"no substrato entregue a via de região está desligada do codificador por construção (`Check2HGIModule.py:653`)"* — mas **não está dito em lado nenhum do artigo**, e é a explicação mais limpa disponível para porque é que a correcção mordeu um braço e não o outro. Vale a pena o autor confirmá-la antes de a usar: se um revisor comparar as duas versões, é a primeira coisa que salta.

**A recomeça foi maior do que "tirar o vazamento".** `CAMERA_READY.md:59-66`, citando `CHARTER_COMPLIANCE.md §2.1`: da receita antiga *"nada disso foi usado na regeneração final"*; um *sweep* de 103 braços re-afinou **os dois** braços; `category_weight` 0,75 → 0,50; pesos de classe substituídos por *logit adjustment* τ = 0,5. A formulação honesta é ali mesmo: *"vazamento removido **e** ambos os braços re-afinados"*, não *"o mesmo modelo, medido sem o vazamento"*.

### 0.3 O que isso faz aos pareceres

**Dissolvidos (dois pontos, ambos do R2).** A "lei" da contagem de regiões foi retirada e o confundimento é nomeado em três sítios; e a premissa factual do R2 — *"rests most heavily on the two single-seed datasets"* — **deixou de ser verdadeira**: TX e CA passaram a quatro sementes × cinco folds em ambos os braços (`tables/tbl3_results.tex`: *"Joint and dedicated entries: four seeds $\times$ five folds"*). Detalhe em §1.2.

**Sobrevivem intactos.** Os quatro pontos do R1 e os quatro do R3. Nenhum deles foi tocado na direcção pedida.

**Agravados.** A relevância para o venue (R2a), a motivação do modelo único (R1-2), a narratividade e a densidade de negações (R3), e a reprodutibilidade (R3) — todos por adição de prosa sem adição de estrutura.

### 0.4 O elogio que já não se sustenta — a parte desconfortável

Isto tem de ser dito sem amortecimento, porque é o eixo em que a versão final é lida.

- **R2 aceitou com base em duas proposições nomeadas:** *"The core empirical claims (check-in-level context substantially improves category prediction; joint training does not hurt and often helps region prediction) are well-supported by the evidence given."* A primeira passou de +27,63…+39,62 para +0,23…+6,29 macro-F1, com a Florida a não atingir significância (`tables/tbl2_substrate.tex`, nota: *"significant under a paired test at every dataset except FL ($p=0.07$)"*). A segunda passou de quatro ganhos + dois *matches* TOST para **dois ganhos e quatro défices resolvidos**, que o próprio artigo se recusa a chamar empates: *"every one of the four intervals lies entirely below zero … but none of them is a tie."* (`sections/06_results.tex`).
- **R3 fez uma única afirmação positiva sobre os resultados:** *"The macro-averaged F1 for next category prediction shows good results for the proposed model."* Hoje as células de categoria conjuntas são 30,59–37,55 e o modelo conjunto bate o dedicado **num** dataset, por +0,19. A frase do R3 só continua verdadeira contra o piso de classe maioritária (5,7–7,3 macro-F1, `sections/06_results.tex`) — que é exactamente o item 16 da lista de cortes da Fase 2 (*"a frase «For scale» · ≈45"*).
- **R1 deu 5 em Technical Correctness** — a nota mais alta do conjunto — a um conjunto de números retirado. `CAMERA_READY.md:99` regista que nenhuma versão que os revisores pudessem ter lido declarava o canal de vazamento.

**Consequência prática:** o objectivo realista da versão final **não é preservar o 5 do R1** — esse não é defensável. É **impedir que o 2 do R3 se converta numa objecção formal**, e é por isso que a §3 (medições) e a §4 (reprodutibilidade) são as secções operacionais deste documento.

### 0.5 O plano de corte vivo foi escrito às cegas

`CONSOLIDATION_PLAN.md:885` (secção *Notas de proveniência deste plano*):

> **Os relatórios dos revisores do MobiWac não existem em lado nenhum do repositório** (`CAMERA_READY §10.1`). Todo o juízo deste plano é feito contra o registo interno, sem saber o que os revisores pediram.

Cronologia (`stat -f "%Sm %N"`): `CAMERA_READY.md` 2026-09-06 **15:20** · `CONSOLIDATION_PLAN.md` **16:33** · `EDAS.TXT` **21:58**. **A Fase 2 é anterior aos pareceres em cinco horas e meia.** E isso importa, porque três dos seus itens cortam exactamente o que dois revisores pedem para expandir:

| Item da Fase 2 | Palavras | Colide com |
|---|---:|---|
| 6 · §VII ¶4, *shortlist* de serviço → 2 frases | ≈80 | **R2(a)** — é a única ponte medida para o venue |
| 9 · §III ¶2, mantendo o escopo *census tract* | ≈85 | **R1-3** — é o parágrafo cuja elaboração ele pede |
| 16 · §VI-B, a frase *"For scale"* | ≈45 | **R3** — é o único enquadramento em que o elogio dele fica verdadeiro |

Nenhum destes cortes é errado em si; todos foram decididos sem esta informação. É a primeira coisa a rever.

### 0.6 Correcções às leituras anteriores desta sessão

Duas afirmações que circularam no mapeamento e que **não se confirmam**:

1. **"O órfão da §II-C está no PDF e o `gate_v17.sh` não o testa."** Ambas as metades caíram.
   - O defeito **existiu** (o PDF imprimia *"…not a general rule. (Istanbul, Alabama, Arizona, and Florida), is equivalent to zero…"*) e **foi reparado durante esta sessão**: `stat` dá `sections/02_related.tex` modificado a **2026-09-06 22:21:38**; o ficheiro tem agora a cabeça da frase viva na linha 184 e um comentário de reparação nas linhas 182-183 (*"The comment above had swallowed the head of the next sentence… repaired"*). Reconstruí o PDF (22:23) e `pdftotext -layout main.pdf` imprime a frase inteira: *"pair of tasks, not a general rule. The same quantity, measured on the reported joint model at four of the six datasets (Istanbul, Alabama, Arizona, and Florida), is equivalent to zero…"*.
   - O portão **tem** a verificação, e nomeia este caso pelo nome. `gate_v17.sh` linhas 112-160: a guarda `(f)` apanha linha viva → comentário; a guarda `(f2)`, escrita depois, apanha o espelho e o comentário cita textualmente *"…not a general rule. (Istanbul, Alabama, Arizona, and Florida)…"*. **`./gate_v17.sh` sai hoje com `rc=1`**, com uma única falha, um falso positivo em `main.tex:137` (a linha de *keywords*).
2. **"O portão tem um teste de pé firme: `grep -n "kilomet" sections/07_discussion.tex` tem de dar 0."** Não tem. `grep -c "kilomet" gate_v17.sh` → **0**. Os quilómetros estão de facto fora do texto (`grep -o -i "kilomet"` na prosa viva: v1 = 2, actual = 0) e a razão está registada em `CONSOLIDATION_PLAN.md:490-491` (item 3.47: *"é uma medição v17 viva dentro de um texto v18"*), mas **não há guarda automática**. Se a decisão for mantê-los fora, o teste tem de ser escrito.

---

## 1 · Os pontos, um a um

### 1.1 R1 — *Likely accept, top 20 %* (4/4/5/4/4/3/4)

Contexto que pesa: R1 é o revisor que carregou a aceitação e o único que leu o artigo como forte — *"The paper is sound and well written. Related works are discussed in an in-depth manner, which helps to position the paper's contribution with the state-of-the-art works."* Os quatro comentários dele estão **todos em aberto** e um **piorou**.

---

#### R1-1 · *"It is not clear from where the 7 category classification is coming from."*

| | |
|---|---|
| **v1** | `03_problem.tex` (0834419b): *"The first is its \emph{category}, one of seven fixed labels: Community, Entertainment, Food, Nightlife, Outdoors, Shopping, and Travel. Istanbul's source collection maps its places onto the same seven labels (Section~\ref{sec:setup-data})."* Sem proveniência. |
| **Actual** | `sections/03_problem.tex:12-13` — a mesma lista, mas **o ponteiro para a frente desapareceu**: *"Istanbul's source collection maps its places onto the same seven labels."* E `sections/05_setup.tex:9` remete para trás: *"All six datasets use the seven place categories defined in Section~\ref{sec:problem}. The source labels for Istanbul are mapped to these categories."* A §V aponta para a §III, a §III não aponta para lado nenhum, e nenhuma diz de onde vêm. |
| **Verificação** | Na prosa viva (comentários removidos, texto achatado): `taxonom` **0** · `Foursquare` **0** · `figshare` **1** (a nota de rodapé de dados) · `root categor` **1** (a legenda da Tabela I: *"All share the same seven root categories"*). Idêntico em v1. |
| **Veredicto** | **POR RESOLVER** — e é o único ponto em que um revisor relata *confusão*, não um desejo. |

**A proveniência existe e é documental.** `data/gowalla/gowalla_category_structure.json` (ficheiro do próprio depósito, `ls -la` dá data de 2011) tem **exactamente 7 entradas de topo**, cada uma com o seu URL `gowalla.com`:

```
Community /categories/934 · Entertainment /categories/12 · Food /categories/7
Nightlife /categories/4 · Outdoors /categories/5 · Shopping /categories/6 · Travel /categories/3
```

Espelhadas em `src/configs/globals.py:26-35` (`CATEGORIES_MAP`; a 8.ª entrada `'None'` é inútil). O que o projecto acrescenta é só o mapa fino→topo: `src/etl/gowalla/stage_1.py:96-108` funde a estrutura do depósito com `callback_categories.json` e `extra_categories.json`. Istambul: folhas Foursquare v1 → 10 raízes FSQ → as mesmas 7, por `scripts/second_dataset/build_category_map.py` (`FSQ_ROOT_TO_GOWALLA`, linha 32) e `docs/studies/second_dataset/category_map.md`.

**Recomendação.** Uma frase na §III, logo a seguir à lista: *"These are Gowalla's own seven top-level spot categories, distributed with the dataset (footnote~\ref{fn:code}); Istanbul's Foursquare labels are collapsed onto the same seven roots through their top-level parents."* **Não** repor o ponteiro `(Section~\ref{sec:setup-data})` — a frase nova torna-o desnecessário. **Custo: +35 palavras.** É a correcção com melhor relação valor/palavra de todo o conjunto.

---

#### R1-2 · *"The motivation and advantages of using a single multi-task model instead of 2 different models could be better elaborated since it is one of the main contributions of the paper."*

| | |
|---|---|
| **v1** | Havia **duas** respostas. Na Introdução — o sítio onde se procura — a resposta era **exactidão**: *"In one forward pass, it outperforms a dedicated single-task category model at every dataset (by about $+5$ to $+9$ points of macro-F1), even though each dedicated model is tuned per dataset."* E no `04_method.tex`: *"What the single model provides is operational rather than arithmetic: one artifact to train, version, and deploy, and one forward pass whose one set of inputs produces both answers at once."* |
| **Actual** | A resposta de exactidão **desapareceu da Introdução e nada a substituiu ali**. `sections/01_introduction.tex:8` só enquadra a partilha como custo (*"Sharing one representation across tasks has a cost…"*) e a linha 6 põe a questão sem estaca: *"we study whether one model should learn both at once."* A resposta operacional sobrevive **palavra por palavra** em `sections/04_method.tex` (págs. 4 do PDF), com as contagens actualizadas (4,2 M contra 1,9 M em AL; 5,2 contra 2,8 em CA). |
| **Veredicto** | **AGRAVADO** — e agravado precisamente no sítio onde o revisor olha. |

**O ponto duro:** a frase que sobrevive é **exactamente a que R1 já leu e considerou insuficiente**. Repeti-la não fecha nada; tem de **mudar de sítio**. E a aritmética é contra o artigo, que o diz honestamente: o modelo conjunto é **maior** do que os dois dedicados somados. Acresce que a §VII agora retira até a atribuição à partilha na Califórnia (*"a dedicated region model of the same size scores above the joint model's region result, so the advantage there cannot be attributed to sharing between the tasks"*). Sobra o argumento operacional, e tem de ser feito explicitamente.

**Recomendação.** Içar para a §I ¶3, logo a seguir à frase do custo: *"The reason to want one model anyway is operational: one artifact to train, version, and deploy, and one forward pass that returns both answers. The question is what that costs per task."* (34 p.), e reduzir a versão da §IV-B a um ponteiro (10 p. em vez de 32). **Custo: +12 líquidas.** Não tentar reconstruir o argumento de exactidão.

---

#### R1-3 · *"The needs and advantages of predicting the next visit (type and area) could be better elaborated."*

| | |
|---|---|
| **v1** | `03_problem.tex` ¶2: *"the category says what type of place comes next, hence what a user will want; the region says where, hence where to prepare… A census tract is a neighborhood, not a radio cell, so we scope our claims to neighborhood-level preparation…"* |
| **Actual** | `sections/03_problem.tex:15-26` — o mesmo argumento, repontuado em três frases (*"A census tract is a neighborhood, not a radio cell. We therefore scope our claims… Cell association and handover are radio-level decisions and remain out of scope."*). 258 → **268 palavras**: cresceu dez palavras sem acrescentar motivação. |
| **Veredicto** | **POR RESOLVER** (o mais fraco dos quatro: é um desejo, não um defeito). |

**Restrição.** Não repor os *"3 to 8 kilometers … against 20 to 241 kilometers"* da v1 (`git show 0834419b:"…/07_discussion.tex"`). `CONSOLIDATION_PLAN.md:490-491` mata-os por serem *"uma medição v17 viva dentro de um texto v18"*.

**Recomendação.** Trocar as duas abstracções por duas acções, sem custo: *"the category says what type of place comes next, so a service knows which content to stage; the region says where, so it knows which neighborhood to provision before demand arrives."* **Custo: 0 a +15.** E **não** aplicar o corte 9 da Fase 2 (≈85 p.) a este parágrafo — recuaria contra dois revisores ao mesmo tempo.

---

#### R1-4 · *"it would be interesting to see a short discussion in how the model can adapt to capture sporadic special events that could lead to changes in behaviour and mobility of people in an area."*

| | |
|---|---|
| **v1** | Nada. |
| **Actual** | **Nada.** Contagens na prosa viva achatada, v1 → actual: `event` 0→0 · `festival` 0→0 · `holiday` 0→0 · `season` 0→0 · `drift` 0→0 · `stationar` 0→0 · `distribution shift` 0→0 · `sporadic` 0→0 · `over time` 0→0. `adaptive` 2→2, e são a citação do *handover* (`vielhaus2022handover`), as *"adaptive strategies"* do Moura, e a ressalva do *"adaptive weighting scheme"*. `retrain` 1→1, e é `pretrains`. A §VII lista hoje **cinco** limites — nenhum sobre mudança temporal. |
| **Veredicto** | **POR RESOLVER** |

**O antídoto já está pago.** O texto actual acrescentou dois factos que a v1 não tinha (`sections/05_setup.tex:14`): *"The median time from the last visit in a window to its target ranges from 0.4 hours in Florida to 5.5 hours in Istanbul. Across the datasets, 5 to 27 percent of the targets occur more than 3 days later."* E o protocolo é conhecido: `docs/research/evaluation_protocol_review.md` descreve o *split* como `StratifiedGroupKFold(shuffle=True, groups=userid)` — **por utilizador, não temporal**.

**Recomendação.** Um sexto limite, declarado, sem experiência nova: *"Sixth, the folds are drawn over users, not forward in time, and the representation is built once, offline. Nothing here measures what happens when behavior changes: a special event or a seasonal shift moves the distributions both models were fitted on. Testing that needs a temporally split evaluation, which we do not run."* **Custo: +52.** Responde exactamente ao pedido e fecha uma lacuna de protocolo que o artigo hoje não divulga em lado nenhum.

---

### 1.2 R2 — *Accept if room, top 30 %* (3/3/3/3/3/3/3)

R2 é o revisor que resume o artigo com precisão e que nomeia três reservas: (a) relevância para o venue, (b) a tendência da contagem de regiões, (c) três detalhes metodológicos.

---

#### R2(b)-i · *"the region-count-vs-gain trend, one of the three headline contributions, is acknowledged to be confounded with corpus size"*

| | |
|---|---|
| **v1** | Abstract: *"Across the five U.S. states, the gain on region grows with the number of regions."* Contribuição 3: *"…with the state with the most regions (California) showing the largest region gain."* |
| **Actual** | A lei foi retirada. O abstract não tem a cláusula (`main.tex`, bloco `abstract`, verificado por `sed -n '/begin{abstract}/,/end{abstract}/p'`). Contribuição 3 (`sections/01_introduction.tex:37-40`): *"The grouping is reported as an observation rather than a law: the ordering does not hold inside the pair, since California has more regions than Texas and a slightly smaller gain, and region count co-varies with corpus size across these states."* Repetido em `sections/06_results.tex:87-89`. |
| **Veredicto** | **JÁ RESOLVIDO** |

**Recomendação: nenhuma acção.** A alegação a que R2 se opôs deixou de existir, e o confundimento que ele nomeou está escrito em três sítios. Não voltar a fortalecer, mesmo com os cinco deltas americanos a serem monótonos em quatro passos consecutivos (−0,87 · −0,44 · −0,16 · +1,21 · +1,06): a inversão terminal é real. **Custo: 0.**

---

#### R2(b)-ii · *"…and rests most heavily on the two single-seed datasets"*

| | |
|---|---|
| **v1** | *"California and Texas are measured at a single random initialization, the other four datasets at four."* |
| **Actual** | **A premissa morreu.** `tables/tbl3_results.tex`, nota: *"Joint and dedicated entries: four seeds $\times$ five folds; $\pm$: sd across seeds."* E `sections/05_setup.tex:59`: *"The final evaluation uses four seeds… This design produces $4\times5=20$ fitted models per configuration."* Os dois ganhos de região sobrevivem à correcção de Holm: TX corrigido *p* = 0,00013, CA *p* < 10⁻⁴, 20/20 folds em cada (`sections/06_results.tex:107-109`). |
| **Veredicto** | **JÁ RESOLVIDO** |

**Recomendação.** Nenhuma alteração no manuscrito. **É a frase mais forte disponível para uma nota aos chairs ou carta de resposta**: os dois datasets que R2 marcou como a perna fraca passaram ao mesmo protocolo de 20 folds do resto, e ambos os ganhos sobrevivem à correcção de multiplicidade. **Custo: 0.**

---

#### R2(b)-iii · Exposição **nova** criada pela reescrita: quem contou sementes vai contá-las na Tabela II

| | |
|---|---|
| **v1** | A Tabela II corria ao mesmo protocolo do resto, e as magnitudes (+27,63…+39,62) estavam muito fora de qualquer ruído de semente. |
| **Actual** | `tables/tbl2_substrate.tex`, legenda: *"…only the input representation differs (seed 0, five folds)"*. E as magnitudes desceram uma ordem de grandeza (+0,23…+6,29), com FL a *p* = 0,07. O controlo de capacidade da §VII também é semente 0 e o Texas não tem controlo de tamanho igualado. |
| **Veredicto** | **AGRAVADO** — a exposição a semente única migrou do eixo que R2 inspeccionou para dois eixos que ele ainda não inspeccionou. |

**Recomendação.** As divulgações já estão correctas e no sítio; **não acrescentar mais ressalvas**. Antecipar na nota aos chairs, com as palavras do próprio artigo: a direcção é unânime em todos os folds nos seis datasets e significativa em cinco de seis, que é o que um desenho emparelhado de uma semente estabelece. **Custo: 0 no artigo.**

---

#### R2(b)-iv · Costura residual: a §VI-B ainda dá o mecanismo que a §VII contradiz

`sections/06_results.tex:85-86`: *"The two datasets where it outperforms are the two with the largest region vocabularies, which is where the region task is hardest **and where the dedicated model has the most to gain from an auxiliary signal**."* Três frases adiante, no mesmo parágrafo: *"…at California, a dedicated region model widened to the joint model's size scores above the joint model … so the gain at that dataset is not attributed to sharing between the tasks."*

A segunda metade da primeira frase é um **mecanismo de partilha**; o controlo diz que na Califórnia não é partilha. **Veredicto: DISCUTÍVEL** — o portão licencia a forma da §VI, e não proponho reabrir isso; proponho cortar só a metade do mecanismo. **Custo: −13 palavras** (poupança).

---

#### R2(b)-v · Residual: a quebra da tendência na ponta **baixa** está na Figura 4 e não é nomeada

v1 usava-a a favor (*"Istanbul, the dataset with the fewest regions, is also ahead on region"*, +0,28). Hoje Istambul é **−0,08** — o menor défice dos quatro, apesar de ter **menos** regiões que Alabama (−0,87) e Arizona (−0,44). A Figura 4 *"plots the signed differences ordered by region count"*, portanto a não-monotonia está na página; o texto confessa só a quebra de cima.

**Recomendação.** Estender a confissão que já existe: *"…nor at the low end, where Istanbul has the fewest regions and the smallest deficit of the four."* **Custo: +18.** **Veredicto: POR RESOLVER.**

---

#### R2(c)-i · *"region-freezing ablation's loss-weight handling … need[s] clarification"*

| | |
|---|---|
| **v1** | `06_results.tex`: *"As a control, we freeze the region pathway at the start of training so it can neither learn nor teach the category task, yet the full category gain survives at Alabama, Arizona, and Florida (within $0.3$ of the joint model…). We therefore attribute the category gain to a stronger shared trunk … we report this as a finding, not a hypothesis."* A perda era *L = 0,75 L_cat + 0,25 L_reg* e o texto nunca dizia o que acontecia ao termo 0,25 com a via congelada — que é exactamente a pergunta do R2. |
| **Actual** | **A experiência desapareceu da prosa entregue.** O único rasto em toda a árvore é um comentário em `sections/02_related.tex:178-181`: *"…the pointer resolved, in the accepted text, to the freeze-region control, which is withdrawn (DP-4); Section VI-B carries no mechanism test now, so the pointer was dangling."* E a perda é hoje *L = 0,5 L_cat + 0,5 L_reg* (`sections/04_method.tex:43`), logo a assimetria em que a pergunta assentava também deixou de existir. |
| **Veredicto** | **PARCIALMENTE RESOLVIDO** — a substância está resolvida; **a supressão é silenciosa**. |

**A razão honesta já está registada.** `CAMERA_READY.md:303` (§S6): *"ENFRAQUECIDA; apagada da dissertação — no substrato entregue a via de região está desligada do codificador por construção (`Check2HGIModule.py:653`), logo «no transfer» é tautologia lida como achado"*.

**Recomendação.** Uma frase, na §VII, no parágrafo que já diz que a contribuição do tronco está por resolver: *"An earlier version of this work reported a control that froze the region pathway; on the representation delivered here that pathway does not read the check-in encoder, so the control cannot measure transfer and is not repeated."* Converte um desaparecimento numa retractação declarada. **Custo: +34. Não repor a ablação.**

---

#### R2(c)-ii · *"per-class category breakdown … need[s] clarification"*

| | |
|---|---|
| **v1** | Nada além da definição da métrica: *"macro-averaged F1 (macro-F1), which averages the per-class F1 so each of the seven categories counts equally"*. |
| **Actual** | Nada, e menos: `per-class` na prosa viva v1 = 1 (era esta definição), actual = **0** — a §V foi reescrita para *"It is the mean of the F1 scores for the seven categories, so each category has equal weight."* `confusion` 0→0; `Nightlife`/`Travel` aparecem uma vez cada, só na enumeração da §III. |
| **Veredicto** | **POR RESOLVER** |

**Recomendação: não acrescentar tabela por classe.** Duas razões independentes. (i) Espaço: 7 classes × 6 datasets × 2 modelos não cabe num artigo a 11 páginas contra um tecto de 10. (ii) Substância: com diferenças conjunto-vs-dedicado de ±0,19 macro-F1, uma decomposição por classe mostraria ruído e convidaria a uma pergunta pior. A resposta barata e verdadeira é um ponteiro — o código de treino escreve um relatório por classe por fold (`src/training/runners/mtl_validation.py`, `src/training/shared_evaluate.py`, persistido em `folds/fold{N}_next_report.json`). Anexar à última frase da §V: *"…together with a per-class report for every fold."* **Custo: +9.** É o mais baixo dos três detalhes do R2 — se o orçamento fechar a vermelho, é este que cai.

---

#### R2(c)-iii · *"unseen-place handling … need[s] clarification"*

| | |
|---|---|
| **v1** | Uma frase nítida em `05_setup.tex`: *"The remaining visits are the one residual this measurement cannot reach, because the representation cannot score a place never seen in training."* E em `07_discussion.tex`: *"our representation is trained once over all places; visits to places never seen during training are the single effect that we cannot fully isolate. A planned follow-up trains it on each fold's training places only…"* — uma promessa, sem números. |
| **Actual** | **A promessa foi cumprida**: `sections/07_discussion.tex:48-50` — *"a per-fold rebuild from training users only changed the results by at most $0.33$ Acc@10 and $0.29$ macro-F1 across three datasets at one seed"*. Mas **a frase nítida perdeu-se**: `sections/05_setup.tex:48-49` diz apenas *"These windows cover 67 to 87 percent of the validation data. The comparison does not cover information specific to each visit or places unseen in training."* `unseen` ocorre **uma vez** em todo o texto vivo. |
| **Veredicto** | **PARCIALMENTE RESOLVIDO** |

**O risco concreto:** com o número 67–87 % sem dizer a que se aplica, um leitor pode concluir que 13 a 33 % da avaliação **reportada** envolve lugares que o modelo não sabe pontuar. É falso.

**Recomendação.** Uma oração onde o número já está: *"…cover 67 to 87 percent of the validation data. That coverage is a property of this control only: the reported results use a representation trained once over all places, so every visit they score has a vector."* **Custo: +31.**

---

#### R2(a) · *"the paper's relevance to a wireless/mobile-systems venue rests on caching/capacity-planning framing that is asserted but never connected to an actual system-level measurement, which the paper itself concedes"*

| | |
|---|---|
| **v1** | O enquadramento (abstract, §I, §III) **mais** duas quantidades medidas na §VII: *"…over 500 times better than picking ten at random"* e *"the ten shortlisted regions are about 3 to 8 kilometers from the shortlist's center …, against 20 to 241 kilometers between two regions drawn at random from the same map."* |
| **Actual** | O enquadramento sobrevive quase palavra por palavra (a §III ainda promete que a §VII *"quantifies what these predictions would give such a service"*). As duas quantidades **desapareceram**: fica a taxa de acerto (*"at California, ten regions out of $8{,}501$ contain the true next region $64.54$ percent…"*) e uma retirada explícita — *"the geographic size of the error … requires the per-visit predictions that the evaluation path does not retain, so it is left to future work."* |
| **Veredicto** | **AGRAVADO.** As duas supressões foram correctas na evidência (a razão 500× era sem fonte; os quilómetros eram v17). Mas nada as substituiu, portanto a queixa exacta do R2 — *asserted, never measured* — é hoje **mais literalmente verdadeira** do que quando ele a escreveu. |

**Recomendação — gastar dois números que o artigo já imprime e nunca usa:**
1. **Tempo de antecipação** (`sections/05_setup.tex:14`), que é o horizonte em que uma cache teria de agir: acrescentar à frase da *shortlist* — *"…and within a lead time this data measures rather than assumes: a median 0.4 hours at Florida to 5.5 hours at Istanbul from the last visit to its target."*
2. **A alternativa mais barata** (`sections/06_results.tex:129-131`): *"The first-order Markov region floor … reaches $51$ to $72$ Acc@10 across the datasets; the joint model exceeds it by $4.1$ to $10.0$ points."* Uma tabela de transições é o que um serviço faria sem modelo — nomeá-la como referência de serviço no mesmo parágrafo.
3. **Corrigir a promessa a mais:** na §III, `quantifies` → `states`. A §VII já não quantifica a metade geográfica.

**Custo: +58 para (1) e (2); 0 para (3).** Não tentar recuperar um número de erro espacial: `CAMERA_READY.md` regista-o como não mensurável no caminho de avaliação v18.

---

#### R2 · o resumo do método: *"trained with fixed-weight joint loss"*

A receita mudou por baixo do revisor: 0,75/0,25 → 0,5/0,5, pesos de classe → *logit adjustment* τ = 0,5, batch/LR re-afinados em **ambos** os braços (`CAMERA_READY.md:59-66`). O manuscrito descreve-se correctamente e **não precisa de edição**. Mas a nota aos chairs tem de dizer *"o vazamento foi removido **e** ambos os braços foram re-afinados"*. **Veredicto: DISCUTÍVEL · Custo: 0 no artigo.**

---

### 1.3 R3 — *Likely reject, top 50 %* (3/**2**/3/3/3/3/2)

É o voto oscilante e a nota mais baixa do conjunto. Quatro queixas de apresentação + a frase da reprodutibilidade (tratada na §4).

---

#### R3-1 · *"Section III, Problems and tasks … This section may be too short, so it may be merged with Section IV."*

| | |
|---|---|
| **v1** | `03_problem.tex` (0834419b): **258 palavras** de prosa viva, uma `\section`, dois parágrafos. |
| **Actual** | `sections/03_problem.tex`: **268 palavras** — dez a mais. Ainda uma `\section`, dois parágrafos, sem subsecções. E a §IV cresceu de 578 → **853** palavras, portanto a desproporção que R3 notou é **maior** hoje. |
| **Veredicto** | **POR RESOLVER** (e proporcionalmente agravado). |

**Recomendação.** Fazer a fusão à letra: em `03_problem.tex`, `\section{Problem and Tasks}` → `\section{Problem, Tasks, and Method}` + `\subsection{Problem and tasks}`; em `04_method.tex`, apagar `\section{Method}` (as duas `\subsection` passam a viver sob o cabeçalho fundido). **Custo: ≈0 palavras, com recuperação de um cabeçalho de secção e da cláusula de roteiro da §I** (que o item 10 da Fase 2 já quer cortar). É a concessão mais barata de todo o conjunto e é o **único pedido estrutural** que qualquer revisor fez — o que conta quando a nota de Presentation é 2. **Não** usar a fusão como cobertura para cortar o parágrafo de motivação (ver R1-3).

---

#### R3-2 · *"The text is full of negative sentences, which make the reading tedious."*

Medições completas na §3. O resumo: **a densidade não mudou (1,56 → 1,57 por 100 palavras) mas a contagem absoluta subiu 53 % (85 → 130) e a *composição* ficou mais pesada.** A forma auxiliar+`not` (*"does not"*, *"is not"*, *"cannot"*) passou de **14 para 55**; *"rather than"* de 11 para 25; e a forma curta *", not Y"* — a única que se lê depressa — **caiu de 21 para 11**. Trocou-se um negativo curto por um negativo longo, que é exactamente o que se lê como tedioso.

**Veredicto: AGRAVADO.**

**Recomendação.** Atacar as 55 construções auxiliar+`not`, não a negação em si (o significado negativo é ligado ao veredicto estatístico; o *enquadramento perifrástico* não é). Ordem de densidade: §V (17), §VII (11), §VI (10), §VIII (7). Exemplos verificados no texto actual:

| Actual | Reescrita | Δ |
|---|---|---|
| *"The evidence here does not separate their contributions."* | *"The evidence here leaves their contributions entangled."* | −1 |
| *"A non-significant difference does not provide evidence of a match."* | *"Evidence of a match requires its own test."* | −3 |
| *"…that a screen of that size cannot distinguish from noise, and it was not run at the two datasets…"* | *"…within noise for a screen of that size, and it omits the two datasets…"* | −4 |

Converter ~30 das 55 baixa a densidade auxiliar+`not` de 0,67 para ~0,30 por 100 e poupa ≈100 palavras. **Alavanca maior, mesma queixa:** o bloco *"Five limits"* da §VII tem **611 palavras** (`sections/07_discussion.tex`, de *"Five limits qualify these results."* até *"Our representation is a fixed per-visit vector"*), com 13 negações e 7 auxiliar+`not`. Passá-lo a `\begin{itemize}` de cinco entradas corta-o para ~330. **Não** reduzir negação hedgeando menos — o registo de honestidade é estrutural.

**Custo: −100 (reescritas) a −350 (com a lista).**

---

#### R3-3 · *"There is also an abuse of parentheses."*

**É a única queixa do R3 em que o texto actual está melhor do que o que ele leu:** 120 → 105 parênteses, **2,21 → 1,27 por 100 palavras** (−43 %). Mas a melhoria é invisível onde o revisor forma a impressão, e apareceram dois casos novos:

- **§I não melhorou:** 3,66 → **3,14** por 100 palavras, e a percentagem de frases com parêntese **subiu de 57,1 % para 63,0 %**. Das 24, **14 são referências cruzadas puras** — `(Section )`, `(Table )`, `(Fig. )`, `(Sections and )` — empilhadas no fim das frases.
- **§II tem um parêntese de 41 palavras** que a v1 não tinha (o maior em §II na v1 tinha **5**): *"(four seeds each on four Gowalla states: Alabama, Arizona and Florida, which are three of the five U.S. datasets reported here, and Georgia, which is not among the datasets this study reports; the largest per-dataset mean in absolute value is …)"*.
- **§VI tem 12 parênteses de estatística** da forma `($+1.21$; $+1.13$ to $+1.29$; corrected $p=0.00013$)`, contra **4** na v1.

**Veredicto: PARCIALMENTE RESOLVIDO.**

**Restrição que exclui a solução óbvia.** `GLOSSARY.md:194`: *"No em-dash ("—"); use commas, parentheses, semicolons, or short sentences."* Ambas as versões têm **zero** travessões (medido). O parêntese é a substituição sancionada — é a causa mecânica da queixa do R3. As saídas legais são vírgula, ponto e vírgula, frase nova, ou **célula de tabela**.

**Recomendação, por valor decrescente:** (a) mover os 12 triplos `(Δ; IC; p corrigido)` da §VI para duas colunas novas da Tabela III (a tabela já existe, não se paga flutuante novo); (b) na §I, manter as quatro referências que o leitor precisa e apagar as outras dez; (c) na §II, cortar a cláusula da Geórgia — nomeia um dataset que o artigo não reporta. **Custo: ≈−100 palavras**, e leva o total para ~85 parênteses (≈0,99 por 100), menos de metade do que R3 leu.

---

#### R3-4 · *"Although the work is interesting, the paper is written in a very narrative way."*

**É a queixa que mais piorou, e a razão é estrutural, não estilística.**

| | v1 | Actual |
|---|---:|---:|
| Prosa viva | 5 442 | **8 267** (+51,9 %) |
| Equações | 1 | **1** |
| `itemize` | 1 | **1** |
| Figuras | 3 (fig1, fig2, fig4) | **3** (as mesmas) |
| Tabelas | 3 | **3** |
| §VII | 295 | **1 227** (×4,2) |
| §VIII | 101 | **506** (×5,0) |

*(inventário de flutuantes verificado por `grep -n "input{figs\|input{tables" main.tex` em ambas as versões — a fig3 já estava comentada na v1)*

**Acrescentaram-se 2 825 palavras de prosa e zero objectos estruturados.** O registo de narração pessoal até melhorou (densidade de *"we"* na §VI caiu de 0,52 para 0,19 por 100), mas o volume foi todo para o outro lado — incluindo um bloco de **611 palavras** (*Five limits*) e um parágrafo de **159 palavras** na §V que narra a cobertura da busca sem imprimir um único valor:

> *"Configuration search. The search was not uniform across the three model families, so its scope is stated here. For the dedicated category model, batch size was searched at all six datasets, over five folds at Istanbul, Alabama and Arizona and on a single fold at Texas, and the learning rate at four of them…"*

**Veredicto: AGRAVADO.**

**Recomendação — três conversões que pagam a queixa e a factura de páginas com a mesma edição:**
(i) o parágrafo *Configuration search* (159 p.) torna-se uma tabela de configuração que **imprime os valores** (ver §4) — ≈−60 líquidas;
(ii) o bloco *Five limits* (611 p.) torna-se `itemize` de cinco entradas, com os números do controlo de capacidade (largura 352, 5 014 942 parâmetros, 97,4 %, +0,41, *p* = 0,010, os braços de 174,8 % e 170,5 %) numa tabela de 4 linhas — ≈−280;
(iii) a §VIII volta para perto do seu tamanho v1: a 506 palavras está a repetir a §VII — ≈−250.

**Custo: −500 a −600 palavras + dois objectos estruturados (~150 palavras de espaço) ⇒ líquido −350 a −450.**

---

## 2 · O que os revisores concordam entre si

Só há **duas** concordâncias reais. São a prioridade, porque são as únicas em que a versão final move mais de um voto.

### 2.1 «Porquê um modelo, e porquê este venue» — R1 (duas vezes) + R2(a)

- R1: *"The motivation and advantages of using a single multi-task model instead of 2 different models could be better elaborated **since it is one of the main contributions of the paper**."*
- R1: *"The needs and advantages of predicting the next visit (type and area) could be better elaborated."*
- R2: *"the paper's relevance to a wireless/mobile-systems venue rests on caching/capacity-planning framing that is asserted but never connected to an actual system-level measurement, **which the paper itself concedes**."*

São a mesma pergunta em dois registos. **E é mais difícil hoje do que na v1**, porque o texto actual concede (a) que o modelo conjunto é maior que os dois dedicados somados e (b) que na Califórnia um modelo dedicado do mesmo tamanho o ultrapassa. O argumento de exactidão morreu; o de partilha está desarmado na Califórnia. **Sobra o argumento operacional, e tem de ser dito explicitamente e cedo.**

Pacote mínimo: R1-2 (+12 líq.) + R1-1 (+35) + R2(a) (+58) + a troca sem custo de R1-3. **≈+105 palavras.**

⚠ **Conflito directo com a Fase 2**: os itens 6 (≈80 p., *shortlist* de serviço) e 9 (≈85 p., §III ¶2) cortam precisamente estes dois sítios. **Não os aplicar.**

### 2.2 «Não se consegue reconstruir o método» — R2(c) + R3

- R2: *"a few methodological details (region-freezing ablation's loss-weight handling, per-class category breakdown, unseen-place handling) need clarification"*
- R3: *"I believe that a researcher working in this area would be unable to reproduce the results shown in the paper with the information given in it."*

Tratada por inteiro na §4. O ponto de concordância é o mesmo objecto: **uma tabela de configuração** responde às duas metades, e é o único acréscimo que *reduz* narratividade em vez de a aumentar.

### 2.3 Concordância tácita nas notas, sem prosa: R2 e R3 dão **3** em Originality e **3** em Rigour

Dois revisores chamaram a contribuição incremental quando ela eram **três** alegações às magnitudes da v1. Hoje são **duas**, e uma delas está desatribuída no seu dataset mais forte. **Não há edição que suba uma nota de originalidade** — não gastar palavras a tentar. Gastar no único enquadramento que sobrevive, que a §VI-B já enuncia: a representação é o efeito dominante na categoria (*"up to thirty-two times the largest difference the choice of architecture makes"*) e o modelo conjunto está *"at least $3.06$ points above the strongest external baseline at every dataset"*. **Proteger ambas as frases no corte.**

---

## 3 · As medições

Método: comentários LaTeX removidos linha a linha; ambientes `figure`/`table`/`tabular`/`tikzpicture` descartados; `$…$` substituído por um símbolo; comandos removidos preservando os argumentos de `\emph`/`\textbf`. Script em `scratchpad/measure.py`. Fronteiras de frase com protecção de `U.S.`, `Fig.` e iniciais. v1 = blobs de `0834419b`; actual = `articles/[mobiwac]/src/` às 22:23 de 2026-09-06 (já com a reparação do órfão).

### 3.1 Negações — por 100 palavras de prosa viva

| Secção | palavras v1 → act. | negações v1 → act. | **por 100** v1 → act. | **aux+not** v1 → act. | *rather than* | *", not Y"* |
|---|---|---|---|---|---|---|
| §I introduction | 710 → 765 | 5 → 10 | 0,70 → **1,31** | 1 → 3 | 0 → 1 | 1 → 0 |
| §II related | 746 → 916 | 12 → 13 | 1,61 → 1,42 | 2 → 3 | 4 → 3 | 3 → 3 |
| §III problem | 258 → 268 | 4 → 3 | 1,55 → 1,12 | 1 → 1 | 0 → 0 | 1 → 1 |
| §IV method | 578 → 853 | 11 → 14 | 1,90 → 1,64 | 1 → 3 | 2 → 2 | 4 → 4 |
| §V setup | 1 246 → 1 660 | 20 → 28 | 1,61 → 1,69 | 5 → **17** | 1 → 3 | 3 → 0 |
| §VI results | 1 508 → 2 072 | 26 → 29 | 1,72 → 1,40 | 2 → **10** | 3 → 9 | 8 → 2 |
| §VII discussion | 295 → 1 227 | 5 → 25 | 1,69 → **2,04** | 2 → **11** | 0 → 7 | 1 → 1 |
| §VIII conclusion | 101 → 506 | 2 → 8 | 1,98 → 1,58 | 0 → **7** | 1 → 0 | 0 → 0 |
| **TOTAL** | **5 442 → 8 267** | **85 → 130** | **1,56 → 1,57** | **14 → 55** | **11 → 25** | **21 → 11** |

Frases com pelo menos uma negação: **33,3 % → 32,7 %** no total, mas **§VII 33,3 % → 54,1 %** e **§VIII 50,0 % → 43,8 %**.

**Leitura.** A densidade global é praticamente idêntica — o que significa que a queixa do R3, tal como ele a formulou, continua exactamente tão válida como quando a escreveu, aplicada a 52 % mais texto. O que mudou é a *forma*: **auxiliar+`not` quase quadruplicou** (14 → 55; 0,26 → 0,67 por 100) e a forma curta *", not Y"* quase desapareceu (21 → 11). **A §VII é o epicentro**: era 295 palavras com 5 negações e é hoje 1 227 com 25, mais de metade das suas frases negativas.

### 3.2 Parênteses — por 100 palavras

| Secção | parênteses v1 → act. | **por 100** v1 → act. | % frases com parêntese v1 → act. | palavras dentro de parênteses |
|---|---|---|---|---|
| §I introduction | 26 → 24 | 3,66 → **3,14** | 57,1 → **63,0** | 78 → 66 |
| §II related | 9 → 10 | 1,21 → 1,09 | 29,6 → 31,2 | 18 → **61** |
| §III problem | 4 → 3 | 1,55 → 1,12 | 25,0 → 14,3 | 8 → 10 |
| §IV method | 6 → 9 | 1,04 → 1,06 | 28,6 → 24,2 | 34 → 37 |
| §V setup | 20 → 8 | 1,61 → **0,48** | 36,4 → 7,3 | 78 → 8 |
| §VI results | 49 → 38 | 3,25 → 1,83 | 59,6 → 33,8 | 218 → 191 |
| §VII discussion | 5 → 10 | 1,69 → 0,81 | 33,3 → 21,6 | 20 → 38 |
| §VIII conclusion | 1 → 3 | 0,99 → 0,59 | 25,0 → 18,8 | 3 → 17 |
| **TOTAL** | **120 → 105** | **2,21 → 1,27** | **42,1 → 23,5** | **457 → 428** |

Casos novos, medidos: **§II — parêntese mais longo 5 → 41 palavras** (a cláusula da Geórgia); **§VI — parênteses de estatística `(Δ; IC; p)` 4 → 12**; **§I — 14 dos 24 são referências cruzadas puras**. Travessões: **0 em ambas as versões** (legislado em `GLOSSARY.md:194`). Ponto e vírgula: 52 → 64.

### 3.3 Comprimento da §III

**258 → 268 palavras (+10).** A §IV cresceu 578 → 853 (+275). Nenhuma subsecção foi criada. **O único pedido estrutural feito por qualquer revisor não foi executado, e a desproporção que o motivou aumentou.**

### 3.4 Um efeito lateral não pedido, mas real: a §V ficou telegráfica

Comprimento médio de frase: **v1 28,3 → actual 15,2 palavras** na §V (contra 27,9 → 24,6 no artigo inteiro). Frases actuais como *"The selected learning rate differs by dataset."*, *"This design produces 4×5 = 20 fitted models per configuration."*, *"A non-significant difference does not provide evidence of a match."* É a secção mais curta em frase e a que tem **17 das 55** construções auxiliar+`not`. Foi reescrita para ser clara e o resultado é uma sequência de asserções curtas maioritariamente negativas — que é a queixa do R3 na sua forma mais concentrada.

### 3.5 A reescrita piorou o R3?

**Sim, em três dos quatro pontos, e melhorou num.**

| Ponto do R3 | Veredicto medido |
|---|---|
| §III curta demais | **POR RESOLVER**, e proporcionalmente pior (+10 p. contra +275 na §IV) |
| Frases negativas | **AGRAVADO** — densidade igual, contagem +53 %, auxiliar+`not` ×3,9 |
| Abuso de parênteses | **PARCIALMENTE RESOLVIDO** — −43 % em densidade, mas §I pior e dois focos novos |
| Escrita muito narrativa | **AGRAVADO** — +2 825 palavras de prosa, **zero** objectos estruturados novos |

---

## 4 · A acusação de reprodutibilidade

> *"I believe that a researcher working in this area would be unable to reproduce the results shown in the paper with the information given in it."*

**A acusação era justa contra a v1 e continua justa hoje — mas a razão mudou, e mudou para pior.**

### 4.1 O que a página nunca imprimiu, e continua a não imprimir

Contagens sobre a prosa viva achatada (`sed` a remover comentários, `tr` a achatar, `grep -o -i | wc -l`), v1 → actual:

`AdamW` 0→0 · `Adam` 0→0 · `optimizer`/`optimiser` 0→0 · `dropout` 0→0 · `GRU` 0→0 · `LSTM` 0→0 · `GPU` 0→0 · `scheduler` 0→0 · `TIGER` 0→0 · `shapefile` 0→0 · `admin` 0→0. Único acerto de `cosine`: um diagnóstico de gradientes na §II, não o nosso *scheduler*.

**O artigo nunca imprimiu um único hiperparâmetro de treino, e nem sequer nomeia as suas duas cabeças de tarefa.** Tudo é delegado numa frase (`sections/05_setup.tex:79`): *"The released code provides the full training configuration for every model (footnote~\ref{fn:code})."* — quase idêntica à delegação da v1.

Pior: o texto actual **narra** o que foi ajustado sem dizer com que valor. `batch size` 1→6 ocorrências, `learning rate` 1→8. A cobertura da busca está escrita em **cinco** sítios (§V ×2, §VI, §VII ×2 — o item 2 da Fase 2 confirma-o) e não imprime um número. **Dizer ao leitor que a afinação importou e reter todos os valores é a pior das duas opções.**

### 4.2 A delegação era válida para a v1. Hoje é falsa.

O ramo público é `origin/mobiwac`, HEAD **`f9c50218`** (2026-08-07), **762 ficheiros**:

```
git ls-tree -r --name-only origin/mobiwac | grep -c mobiwac_v18   →  0
git ls-tree -r --name-only origin/mobiwac | grep mobiwac_v17      →  docs/reproducibility/mobiwac_v17/…
```

**O ramo documenta e fixa a receita v17. O camera-ready reporta v18.** Três consequências verificáveis em minutos por quem seguir a nota de rodapé:

1. **A equação (1) contradiz todos os scripts publicados.** O artigo imprime *L = 0,5 L_cat + 0,5 L_reg* (`sections/04_method.tex:43`). No ramo: `scripts/closing_data/p3_board.sh:230`, `run_catx_v17_n20.sh:42`, `run_catx_v17_seed0_5f.sh:27`, `run_istanbul_champion_stride1.sh:33` — todos com `--mtl-loss static_weight --category-weight 0.75`, que é **exactamente a equação da v1**. O README do ramo repete 0,75 no bloco *"exact paper recipe"*.
2. **O elemento de método mais visível do artigo não está em nenhuma receita publicada.** `logit adjustment` com τ = 0,5 ocupa um parágrafo e uma citação (`menon2021logitadjustment`) em `sections/04_method.tex:45-48`. `--logit-adjust-tau` existe no ramo em três ficheiros — `scripts/train.py`, `src/training/runners/mtl_cv.py`, `scripts/p1_region_head_ablation.py` — e em **nenhum script de execução**. O valor por omissão é 0,0, portanto **todas as receitas publicadas correm CE simples**.
3. **O README continua a descrever-se como anónimo.** Abre com *"# MobiWac 2026 — Anonymous Code Release"* e diz *"This repository is the anonymized reviewer snapshot: code only… the development repository will be made public after acceptance."* O artigo foi aceite a 2026-08-26; o ramo não é tocado desde 2026-08-07.

**Veredicto: AGRAVADO.** Um leitor que siga a nota de rodapé não *falha* em reproduzir — encontra uma receita documentada que **contradiz a equação impressa**.

### 4.3 O que melhorou, e é preciso dizê-lo

Não é tudo mau, e ignorá-lo seria tão desonesto como ignorar o resto. Face à v1, a especificação legível na página melhorou substancialmente: a equação (1) está exacta e justificada; o *logit adjustment* está definido, parametrizado e citado; a construção *forward-only* do grafo está declarada; o janelamento e a construção dos folds são invulgarmente precisos; as métricas e a convenção de erro do Acc@10 são exactas; e a secção estatística (plano pré-registado, dois desvios nomeados, famílias de Holm, margem TOST, convenção de selecção de época) é melhor do que a da maioria dos artigos aceites.

### 4.4 O que fazer, por ordem de custo

| # | Acção | Custo no artigo |
|---|---|---|
| 1 | **Republicar `origin/mobiwac` com o bundle v18 e um README desanonimizado.** Os scripts v18 têm de fixar `--category-weight 0.5` e passar `--logit-adjust-tau 0.5` **só na cabeça de categoria**. `CAMERA_READY.md:591` já lista este passo (D10). | **0** |
| 2 | **Uma tabela de configuração** (12-14 linhas, coluna única): optimizador + *weight decay*; *scheduler* e LR máximo por cabeça; épocas; batch; precisão; cabeça de categoria e cabeça de região; blocos de *cross-attention*; codificador da representação; sementes; GPU. **Financiada** pelo item 2 da Fase 2 (≈120 p. da cobertura da busca dita em cinco sítios). | +150 de espaço, −120 de prosa ⇒ **≈+30** |
| 3 | **Os valores das sementes, inline**, onde a frase já está: *"…uses four seeds ({0, 1, 7, 100}), each of which…"* (o 42 é a semente de desenvolvimento e está deliberadamente excluído — o README do ramo di-lo, o artigo não). | **+6** (ou 0, se for para a tabela) |
| 4 | **A proveniência das fronteiras de região**: *"…a census tract in the U.S. datasets (Census TIGER 2022 boundaries) and a mahalle in Istanbul (OpenStreetMap admin_level 8)."* As contagens de *tracts* mudam entre vintages, portanto 1 109 em AL e 8 501 em CA não são reconstruíveis sem isto. | **+11** |
| 5 | **Repor as quatro palavras que caíram**: *"a location-based social network **(collected 2009 to 2011)**"*. Medido: `2009` v1 = 1, actual = **0**. É como um reprodutor confirma que descarregou o *dump* certo — e passou a importar mais desde que o README do ramo atribui erradamente esse *dump* ao SNAP. | **+5** |

**Não** responder ao R3 acrescentando prosa sobre reprodutibilidade: pioraria os pontos 2 e 3 dele e custaria palavras que o artigo não tem.

---

## 5 · O que isto custa em palavras

### 5.1 O constrangimento, medido

- **Build actual: 11 páginas** (`pdfinfo main.pdf`; `Output written on main.pdf (11 pages` em `main.log`).
- **Orçamento do venue**: *"8 páginas; tecto 10 com taxa por página"* (`CAMERA_READY.md:396`). **A 11 páginas o artigo não é submissível de todo.**
- **Taxa por página em euros e prazo exacto do camera-ready: não verificáveis neste repositório.** `CAMERA_READY.md §10.2` regista que o dossier do venue (congelado 2026-06-19) diz **"TBA"** para prazo, número de páginas e tabela de taxas. Os valores «2 extra a 120 EUR» e «2026-09-07 AoE» vêm do enunciado desta tarefa, não do registo — **confirmar no EDAS antes de decidir**.
- **Densidade medida: ≈1 042 palavras-página** (`CONSOLIDATION_PLAN.md:864`; enchimento p8 1 095 · p9 1 034 · p10 1 082 · p11 956).
- ⚠ **A conta é pior do que parece:** `grep -c "IEEEpubid" main.tex` → **0**. Não há `\IEEEpubid`, nem agradecimentos, nem bloco de copyright. Todos consomem espaço na página 1 e **entram antes de medir**.

### 5.2 Todas as recomendações, com custo

| # | Ponto | Recomendação | Palavras |
|---|---|---|---|
| 1 | R1-1 | Proveniência das 7 categorias, §III | **+35** |
| 2 | R1-2 | Içar o argumento operacional para a §I; ponteiro na §IV-B | **+12** |
| 3 | R1-3 | Duas acções em vez de duas abstracções, §III ¶2 | **0 a +15** |
| 4 | R1-4 | Sexto limite: folds por utilizador, não temporais | **+52** |
| 5 | R2(a) | Tempo de antecipação + piso de Markov como referência de serviço, §VII | **+58** |
| 6 | R2(a) | `quantifies` → `states`, §III | **0** |
| 7 | R2(b)-iv | Cortar *"and where the dedicated model has the most to gain from an auxiliary signal"* | **−13** |
| 8 | R2(b)-v | Nomear a quebra na ponta baixa (Istambul) | **+18** |
| 9 | R2(c)-i | Retractação declarada do controlo de congelamento | **+34** |
| 10 | R2(c)-ii | Ponteiro para o relatório por classe do código | **+9** |
| 11 | R2(c)-iii | Âmbito da cobertura 67–87 % | **+31** |
| 12 | R3-1 | Fundir §III em §IV (subsecção) | **≈0** (+ um cabeçalho) |
| 13 | R3-2 | Reescrever ~30 das 55 auxiliar+`not` | **−100** |
| 14 | R3-2/4 | *Five limits* (611 p.) → `itemize` + tabela do controlo | **−280** |
| 15 | R3-3 | 12 triplos estatísticos → colunas da Tabela III | **−57** |
| 16 | R3-3 | §I: cortar 10 das 14 referências cruzadas parentéticas | **−25** |
| 17 | R3-3 | §II: cortar a cláusula da Geórgia (41 p.) | **−20** |
| 18 | R3-4 | §VIII (506 p.) de volta para perto do tamanho v1 | **−250** |
| 19 | R3-repro | Tabela de configuração (−120 da cobertura da busca) | **+30** |
| 20 | R3-repro | Sementes {0, 1, 7, 100} | **+6** |
| 21 | R3-repro | TIGER 2022 / OSM admin_level 8 | **+11** |
| 22 | R3-repro | Repor *"collected 2009 to 2011"* | **+5** |

**Somas.** Acréscimos **+306** · cortes **−745** ⇒ **líquido −439 palavras**.

### 5.3 A aritmética de páginas, sem embelezar

| | palavras-página |
|---|---:|
| Necessário para 11 → **10 páginas** (mínimo submissível) | ≈**1 042** |
| Necessário para 11 → **8 páginas** (grátis) | ≈**2 700 a 2 900** (`CONSOLIDATION_PLAN.md:866`) |
| Disponível: pacote dos revisores (líquido) | **−439** |
| Disponível: Fase 2 do `CONSOLIDATION_PLAN` | **−1 870** |
| Sobreposição a descontar (itens 3, 5, 11, 12, 13 da Fase 2 tocam nas mesmas passagens dos ## 13/14/15/18) | ≈**+450** de duplicação |
| Disponível: Figura 4 duplica os 12 deltas da Tabela II | ≈**−260** (¼ de página) |

**Resultado.** Pacote dos revisores + Fase 2 (líquida da sobreposição) ≈ **−1 860 palavras** ⇒ **10 páginas com folga confortável, e não 8**. Chegar a 8 exige, além disto, os flutuantes todos (Figura 4 fora, Figura 1 a largura de coluna, Tabela III compacta) e ainda assim fica apertado — e o `\IEEEpubid` por pôr come parte da margem.

**A decisão que isto força, e é do autor:**

- **Rota A — 10 páginas, pagar duas.** Executa-se todo o pacote dos revisores, a Fase 2 fica a meio, nada do que dois revisores pediram é cortado, e o artigo continua a poder proteger o piso de classe maioritária, o parágrafo de motivação e a *shortlist* de serviço.
- **Rota B — 8 páginas, grátis.** Exige largar ≈2 800 palavras. Aos preços medidos, isso significa **cortar passagens que R1 e R2 pediram para expandir** (itens 6 e 9 da Fase 2) e a frase que sustenta o único elogio do R3 (item 16). O `PLAN_8PAGES.md` já nomeia o seu próprio critério de aborto — *"pagar a taxa em vez de"* enfraquecer o artigo — e esse plano está morto por outra razão (cabeçalho: **"DOCUMENTO DE 2026-07-09, MANTIDO POR PROVENIÊNCIA. NÃO O SIGA."**), mas o critério não envelheceu.

**Se só couber uma coisa:** R1-1 (+35 palavras). É factual, é a mais barata, e é o único ponto em que um revisor relata confusão em vez de exprimir um desejo.

### 5.4 Três coisas que não custam palavras e que têm de ser feitas

1. **Republicar `origin/mobiwac` em v18, desanonimizado** (§4.2). Enquanto não o for, a nota de rodapé do artigo é falsa de uma maneira que não era quando os revisores a leram.
2. **`./gate_v17.sh` sai a `rc=1`** com um falso positivo em `main.tex:137`. Ou se limpa, ou se documenta, antes de qualquer sessão de corte — porque o corte é exactamente a operação em que a guarda `(f)`/`(f2)` volta a disparar a sério.
3. **Escrever a nota aos chairs** (Program Chair Rodolfo W. L. Coutinho): o vazamento foi removido **e** ambos os braços foram re-afinados; as duas proposições que R2 nomeou como bem apoiadas mudaram de magnitude e de sinal; em contrapartida TX e CA deixaram de ser de semente única e existe agora um controlo de capacidade que a versão aceite não tinha.

### 5.5 Riscos residuais que o autor deve verificar antes de assinar

- **Tabela III vs. texto, arredondamento.** Três das seis diferenças de categoria diferem em 0,01 entre a subtracção directa das células e o valor impresso no texto (FL: tabela 37,55 − 37,35 = +0,20, texto **+0,19**; AL: −0,18 vs **−0,19**; TX: −0,14 vs **−0,13**). É o artefacto esperado de médias por semente arredondadas antes da subtracção, e os valores do texto batem certo com `CAMERA_READY.md §3.1`. Não é um erro — mas um revisor cuidadoso subtrai as células, e vale a pena saber a resposta de antemão.
- **A coluna do tecto dedicado de região não mudou a ±0,02 entre a v1 e hoje** (§0.2), apesar de `CHARTER_COMPLIANCE.md` dizer que **ambos** os braços foram re-afinados. A explicação existe e é boa (configuração fixa + via de região desligada do codificador), mas **está fora do artigo** e é a primeira pergunta de quem compare as duas tabelas.

---

*Este documento é a leitura organizada e verificada dos pareceres. Não é o plano de ataque.*