# LISTA FINAL DE CORTES — MobiWac 2026, camera-ready

**Ficheiro de trabalho: uma cópia de `/tmp/edascheck`. Não editar `articles/[mobiwac]/src/` (outra sessão é a dona).**

> ⚠ **AVISO DE NUMERAÇÃO — ler antes de abrir qualquer ficheiro de tabela.** Os nomes dos ficheiros estão trocados em relação aos números impressos. Confirmado em `main.aux`:
> - `\ref{tab:datasets}` = **Tabela I** = `tables/tbl1_datasets.tex` (página 7)
> - `\ref{tab:results}` = **Tabela II**, a escada de 24 células = `tables/tbl3_results.tex` (página 7)
> - `\ref{tab:substrate}` = **Tabela III**, a representação = `tables/tbl2_substrate.tex` (página 8)
>
> Nesta lista **tudo é referido por etiqueta e por ficheiro**, nunca por número. Um dos caçadores usou a numeração invertida; a lista-lei e o PDF usam esta.

---

## 1. A aritmética — e a resposta

**As 8 páginas são alcançáveis, com margem de ~15%, sem tocar num único item da lista-lei nem nas decisões em aberto do autor.**

| | col-pt | equivalente |
|---|---|---|
| **Necessário** (conteúdo além da página 8, medido no PDF) | **1 562,1** | **~1 270 palavras** |
| Prosa aceite (§I, II, IV, V, VI, VII, VIII) — **1 246 palavras** | 1 532,6 | 1 246 palavras |
| Equação (1) em display → em linha (M18) — 0 palavras, 3 linhas de coluna | ~34 | ~28 |
| Tabelas / figuras / bibliografia / resumo (T1–T10), líquido de re-empacotamento | ~225 | ~183 |
| **Total libertado** | **~1 792** | **~1 457** |
| **Margem** | **+230** | **+187 palavras (~0,17 página)** |

Três números que sustentam isto, todos medidos e não estimados: o conteúdo além da página 8 são 1 562,1 col-pt (uma página = 1 344 col-pt de coluna; 1 palavra de corpo = 1,23 col-pt, ~1 092 palavras/página); os seis caçadores mediram 1 530 palavras de prosa cortável, das quais **rejeito 284** (secção 3) e aceito **1 246**; e o caçador da §IV verificou empiricamente que só as 252 palavras dele já levam o build de 10 para 9 páginas.

**A margem de +187 palavras não é gordura, é seguro.** O LaTeX iguala o fundo das colunas: uma poupança só se converte em página quando ultrapassa uma linha inteira de corpo, e os floats voltam a empacotar-se a cada passagem (o caçador dos floats mediu ~30 col-pt de T6 a serem reabsorvidos enquanto a prosa não entra). Contar com margem zero é contar com pagar 120 EUR.

Se ainda assim faltar, há **reserva medida de ~145 palavras + 0,13 página** (secção 3, fim) que o autor pode libertar com uma decisão só dele. Não a inclui aqui.

---

## 2. Os cortes — por ficheiro, ordenados por rendimento

Rendimento = palavras ÷ peso do risco (nenhum = 1, baixo = 1,5, médio = 3). Trabalhe de cima para baixo dentro de cada ficheiro; se ficar sem tempo, o que ficar por fazer é o de menor rendimento.

---

### 2.1 `sections/07_discussion.tex` — 155 palavras (fazer primeiro: é o maior bloco único do artigo)

**D1 — 85 palavras — risco baixo — rend. 56,7**
Parágrafo de abertura (linhas ~23-31). Apagar as **quatro primeiras** frases; a quinta, protegida ("The four region results inside that margin..."), fica.
*De:* `A single model serves both tasks in one forward pass, on next region it outperforms ... registered before any result was read. On next category it outperforms the dedicated model at Florida, and the five remaining differences are equivalent to zero within half a point. Throughout, the region output keeps its own private spatial path (Table~\ref{tab:results}). The practical reading is that one model replaces two while staying inside the margin ... largest region vocabularies.`
*Para:* `One model serves both tasks in one forward pass, above two dedicated models on next region at Texas and California and inside the registered two-point margin at the other four (Table~\ref{tab:results}).`
*Perde-se:* o boletim de resultados repetido à cabeça da Discussão. A frase 4 já repetia as frases 1-2 dentro do mesmo parágrafo; o resto está em §VI-B (linhas 112-118), no §VIII (parágrafo 2) e no resumo.
*Condição dura:* **não** combinar com nenhum corte ao segundo parágrafo do §VIII — uma das duas cópias tem de sobreviver. C15 (abaixo) só tira o aposto, não o boletim: é compatível.

**D2 — 27 palavras — risco baixo — rend. 18,0**
Fim do quinto limite. Apagar: `What remains on the region axis is the operational result: one model produces both predictions in one forward pass, at no cost beyond the bounds reported above.`
*Perde-se:* a aterragem positiva do parágrafo, que passa a fechar em "and at Texas the question is open" — que é onde uma limitação deve fechar. A propriedade de modelo único é dita na abertura da §VII (sobrevive em D1) e na última frase da Conclusão. Nada do núcleo D4 é tocado.

**D3 — 20 palavras — risco baixo — rend. 13,3**
Parágrafo do tronco partilhado, **só a oração final**. Apagar: `: this design produces a joint region output above two dedicated models at the two datasets with the largest region vocabularies` (e o `What can be stated is a claim about the model rather than about transfer between its tasks` fica, terminando com ponto final).
*Perde-se:* a quarta enunciação da superioridade Texas/Califórnia — que além do mais lê mal contra o quinto limite, dois parágrafos abaixo, onde um modelo dedicado de tamanho igualado ganha +0,41 na Califórnia. Todas as divulgações do parágrafo (não-separação, ablação dentro do ruído, não corrida onde vive a vantagem) ficam intactas.

**D4 — 23 palavras — risco MÉDIO — rend. 7,7 — exige renumeração; fazer com a checklist**
Terceiro limite. Apagar: `Third, we do not build or evaluate a mobility-aware service; it is background motivation, and this paper's claims are the prediction results themselves.`
*Perde-se:* a segunda cópia do aviso; a primeira, `This remains motivation, not a measured service result.`, está oito linhas acima, dentro da passagem protegida do serviço, e não se toca.
*Checklist obrigatória depois de apagar:* `Five limits` → `Four limits`; `Fourth,` → `Third,`; `Fifth,` → `Fourth,`. Depois correr `grep -n "Five limits\|Fifth\|Fourth\|Third" sections/*.tex` — nenhuma outra secção pode citar os limites por número — e `./gate_v17.sh`.

---

### 2.2 `sections/06_results.tex` — 263 palavras

> **Dependência dura:** R2 + R3 juntos tiram da prosa **tanto** a unanimidade de folds **como** o teste emparelhado. Ambos passam a viver só na nota de rodapé de `tables/tbl2_substrate.tex`. **Essa nota não pode ser apagada** (ver rejeição #19). Se por alguma razão a nota sair, R2 e R3 revertem.

**R1 — 50 palavras — baixo — rend. 33,3**
§VI-B, o parágrafo `Table~\ref{tab:results} gives the signed differences on both axes...`
*De:* as três frases completas.
*Para:* `On region (Acc@10), the joint model outperforms the dedicated model at Texas and California.`
*Perde-se:* a terceira impressão dos quatro estimadores pontuais. −0,19 (AL) e +0,19 (FL) reaparecem oito linhas abaixo na lista de ICs de categoria; +1,21 (TX) e +1,06 (CA) na lista de região; "Florida is the only one the tests resolve" é reformulado logo a seguir. Os seis ICs, o limite 0,334 e os quatro défices ficam intocados. A oração mantida é o antecedente de que o par protegido precisa ("The two datasets where it outperforms are...").

**R2 — 32 palavras — baixo — rend. 21,3**
§VI-A. Apagar: `A paired test over the five matched folds separates the two representations at every dataset except Florida, where the direction is unanimous across folds but the difference does not reach significance ($p=0.07$).`
*Perde-se:* nada — a nota de `tbl2_substrate.tex` diz o mesmo teste, a mesma excepção, o mesmo p, e cobre também a unanimidade.

**R3 — 27 palavras — baixo — rend. 18,0**
§VI-A. Apagar: `The difference ranges from $+0.23$ macro-F1 at Florida to $+6.29$ at Istanbul, and all five folds favor the check-in-level representation at every one of the six datasets.`
*Perde-se:* nada. A amplitude está na coluna Δ de `\ref{tab:substrate}` e é repetida em §VI-B ("moves the same metric by +0.23 to +6.29 points", onde faz trabalho); a unanimidade fica na nota, no item 1 das contribuições e no §VIII.

**R4 — 24 palavras — baixo — rend. 16,0**
§VI-A, abertura.
*De:* `The comparison is controlled: target, single-task model, training configuration, windowing, epoch budget, logit adjustment, and folds are identical; only the input representation changes (a vector per visit versus a vector per place), so the difference isolates the representation.`
*Para:* `Only the input representation differs, a vector per visit versus a vector per place.`
*Perde-se:* a enumeração em prosa. A legenda de `\ref{tab:substrate}`, imediatamente por baixo, tem os mesmos sete itens pela mesma ordem — e **essa legenda não é alterada por esta lista** (ver rejeição #21), portanto o sinal de justiça é substituído, não apagado.

**R5 — 24 palavras — baixo — rend. 16,0**
§VI-B. Apagar a frase inteira: `Table~\ref{tab:results} reports the single joint model against the dedicated single-task ceilings (the level a joint model is usually expected, at best, to reach).`
*Perde-se:* a palavra "ceiling", que ocorre **exactamente uma vez em todo o artigo** — a frase introduz um termo, define-o e nunca mais o usa. O parágrafo passa a abrir em `Every reported model is one saved model per fold`.

**R6 — 20 palavras — baixo — rend. 13,3**
§VI-A. Cortar a segunda metade da frase: manter `What the comparison establishes is therefore a consistent direction rather than a large effect.` e apagar `: the input representation is one condition that affects performance here, and it acts the same way everywhere it was measured`.
*Perde-se:* uma ressalva repetida dentro da própria frase.

**R7 — 18 palavras — baixo — rend. 12,0**
§VI-A.
*De:* `The check-in-level column is the same dedicated single-task category model reported in Table~\ref{tab:results}, read at seed 0 so that the two columns share one fold partition; the place-level column is the same model with its input swapped.`
*Para:* `The check-in-level column is the dedicated category model of Table~\ref{tab:results} at seed 0; the place-level column swaps its input.`
*Perde-se:* a razão do seed 0 — que a legenda já diz ("over five matched folds, seed 0"). A identidade entre tabelas, que a legenda **não** diz e o leitor precisa (30,77 com sd 1,16 numa e 0,07 na outra), fica.

**R8 — 17 palavras líquidas — baixo — rend. 11,3**
§VI-B. Apagar `The tests behind those verdicts are reported next, each entry giving the point estimate and its 90\% confidence interval.` **e**, obrigatoriamente, etiquetar a primeira entrada: `...survives the Holm correction across the six ($+0.19$; 90\% CI $+0.14$ to $+0.25$; corrected $p=0.011$)` (+2 palavras).
*Perde-se:* o anúncio do formato. O "90%" não se perde.

**R9 — 14 palavras — baixo — rend. 9,3**
§VI-B. Apagar a cauda `, which is the sense in which one model can replace two on this task` (a frase fica em `...the smallest of the three effects at work here.`).

**R10 — 9 palavras — baixo — rend. 6,0**
§VI-B. Apagar `, its interval reaching to within two thousandths of zero` — é o $-0.002$ impresso três linhas acima, escrito por extenso. `resolves three of the four; Istanbul's is not resolved` fica.

**R11 — 9 palavras — baixo — rend. 6,0**
§VI-B, parágrafo dos quatro défices. Apagar `, which is what the pre-registered analysis asked of them,`. **`Each of the four stays inside the margin ... but none of them is a tie.` fica palavra por palavra**, e o registo do plano continua na segunda frase do mesmo parágrafo ("the registered two-point margin").

**R12 — 8 palavras — baixo — rend. 5,3**
§VI-C. Apagar `, which uses the same representation, windows, and folds` — glosa do termo usado duas palavras antes.

**R13 — 11 palavras — risco MÉDIO — rend. 3,7**
§VI-B. Apagar `Section~\ref{sec:setup-windows} states the search coverage per model, knob, and dataset.`
*Perde-se:* um de **três** ponteiros para o mesmo sinal de justiça. Ficam dois: a §V-B (o enunciado completo, por botão e por dataset) e a nota de rodapé de `tables/tbl3_results.tex`, que **mantemos** (rejeição #20). A cobertura continua dita mais do que uma vez, e no ponto de uso.

---

### 2.3 `sections/05_setup.tex` — 339 palavras (o ficheiro mais rentável)

**S1 — 66 palavras — baixo — rend. 44,0 — corrige também um erro de facto**
§V-A, segundo bloco.
*De:* `Five datasets come from Gowalla, a location-based social network, collected from 2009 to 2011, ... The U.S. datasets range from about 114,000 check-ins and 1,109 regions in Alabama to about 3.2 million check-ins and 8,501 regions in California. All six datasets use the seven place categories defined in Section~\ref{sec:problem}. The source labels for Istanbul are mapped to these categories. A region is a census tract in the U.S. datasets and a mahalle in Istanbul.`
*Para:* `Five come from Gowalla, collected from 2009 to 2011, and cover the U.S. states of Alabama, Arizona, Florida, Texas, and California~\cite{cho2011gowalla}. The sixth contains check-ins from Istanbul in the Massive-STEPS collection~\cite{wongso2025massivesteps}, which lets us examine the findings outside the United States.`
*Perde-se:* a amplitude, as sete categorias, o mapeamento de Istambul e a unidade de região — todos ditos na §IV-A, que **fica intacta por decisão desta lista** (ver rejeição #1). A frase da amplitude vale por si: lida como amplitude de check-ins está **errada** — a Tabela I dá 4 089 892 ao Texas contra 3 171 380 à Califórnia, logo "to about 3.2 million ... in California" não nomeia o extremo superior. Cortá-la remove 66 palavras e um erro.

**S2 — 37 palavras — baixo — rend. 24,7**
§V-B. Fundir os dois parágrafos corridos (`\emph{Integrity of the representation.}` e `\emph{Whole-dataset training.}`) num só, colocado onde está hoje o segundo (depois do parágrafo da busca de configuração):
*Para:* `\emph{Whole-dataset training.} The prediction models are trained without the validation users, but Check2HGI is trained once on the whole dataset and has seen their visits. Its training objective uses neither the next-category nor the next-region label. To check whether this gives the prediction models an advantage, we built a new representation for each fold using only its training users.`
*Perde-se:* a pergunta feita uma vez em vez de três. **Os números do whole-dataset training e o âmbito "on an earlier build" são a frase seguinte e não se tocam.** A etiqueta `\label{sec:setup-windows}` da subsecção não se move (a §VII e a nota da Tabela II citam-na).

**S3 — 18 palavras — risco NENHUM — rend. 18,0**
Primeira frase da §V. Apagar: `This section describes the datasets, data splits, evaluation metrics, statistical tests, and comparison models used in the experiments.` — é a lista dos quatro títulos de subsecção que se seguem na mesma página.

**S4 — 25 palavras — baixo — rend. 16,7**
§V-B, `\emph{Windows.}`
*De:* `Near the end of a user's history, several start positions can produce shorter, padded windows with the same final visit as the target. We remove these duplicates and keep only the full-length window that ends at this target.`
*Para:* `Where several start positions share a target, we keep only the full-length window.`

**S5 — 23 palavras — baixo — rend. 15,3**
§V-D, `\emph{Representation controls.}`
*Para:* `Two controls test whether the category gain comes from our representation design rather than from contextualization in general or from additional features: CTLE~\cite{lin2021ctle}, the closest prior contextual check-in representation, and the standard place embedding (HGI~\cite{huang2023hgi}) concatenated with raw per-visit features (Section~\ref{sec:results-part1}).`
*Perde-se:* onde o CTLE foi afinado e onde foi congelado, e o que o controlo de concatenação concatena — ambos ditos com mais rigor na §VI-A, onde o leitor encontra os números. **O HGI continua nomeado na §V-D**, para que a frase DP-6 seguinte ("HGI is pre-trained once ... while CTLE is trained for each fold") mantenha o antecedente.

**S6 — 22 palavras — baixo — rend. 14,7**
§V-B, `\emph{Splitting.}`
*De:* `We stratify the split by the next-category label because the seven classes are imbalanced. Food is the majority class in every dataset, ranging from about 25 percent of visits in Florida to 34 percent in Alabama (Table~\ref{tab:datasets}).`
*Para:* `We stratify the split by the next-category label because the seven classes are imbalanced (Table~\ref{tab:datasets}).`
*Perde-se:* a quota da classe maioritária, que é uma coluna da Tabela I (Majority %). **Não é a frase "For scale" protegida** — essa está na §VI-B, carrega o piso de macro-F1 (5,7–7,3) e não é tocada.

**S7 — 20 palavras — baixo — rend. 13,3 — compressão, não supressão**
§V-B, parágrafo `\emph{Configuration search.}` A frase actual repete a mesma cláusula de cobertura de folds duas vezes.
*Para:* `The search was not uniform across the three model families. For the dedicated category model, batch size was searched at all six datasets and the learning rate at four of them, in both cases over five folds at Istanbul, Alabama and Arizona and on single folds at Texas. Florida and California carry the large-dataset value, which only the single-fold Texas screen tested, and the selected learning rate differs by dataset. For the joint model, batch size and the category-head learning rate were searched at Istanbul, Alabama, and Arizona over five folds and screened at Florida, with Texas and California carrying the configuration transferred from those searches. The dedicated region model uses one configuration throughout, with only the logit-adjustment strength tested and set to zero.`
*Perde-se:* nada. Os sete factos ficam, pela mesma ordem. A lei da substituição não é accionada porque nada é suprimido.

**S8 — 16 palavras — baixo — rend. 10,7**
§V-D. Apagar a última frase de `\emph{Task comparisons.}`: `ReHDM~\cite{li2025rehdm} is run on our data under its published chronological split and reported as a reference.` **e** mover a citação para a frase de abertura do mesmo parágrafo, quatro linhas acima, que já diz o mesmo: `... and ReHDM~\cite{li2025rehdm} under its published chronological split.`

**S9 — 16 palavras — baixo — rend. 10,7**
§V-D, último parágrafo. Apagar: `A final control uses the standard place-level HGI representation~\cite{huang2023hgi} to measure what the per-visit representation adds.` A frase seguinte (`All other choices, including the nine-visit window, were fixed during development.`) corre a seguir ao parágrafo anterior. **Fazer S5 antes de S9**, para que o HGI continue nomeado na §V-D.

**S10 — 15 palavras — baixo — rend. 10,0**
§V-B. Apagar `The comparison does not cover information specific to each visit or places unseen in training.` — as duas frases anteriores dizem o mesmo em concreto e com a cobertura 67–87%. A ressalva seguinte ("Within this coverage...") fica.

**S11 — 15 palavras — baixo — rend. 10,0**
§V-C. Apagar `Its reference point is the majority-class floor, obtained by always predicting the most common category.` — a frase protegida "For scale", na §VI-B, define o piso ao mesmo tempo que lhe dá o valor.

**S12 — 9 palavras — risco NENHUM — rend. 9,0**
§V-C, as duas definições de métrica passam a apostos:
`For the category task, we report macro-averaged F1 (macro-F1), the mean of the seven per-category F1 scores, so each category has equal weight.` … `For the region task, we report accuracy at ten (Acc@10), the fraction of test visits whose true region is among the model's ten highest-scoring predictions.`

**S13 — 13 palavras — baixo — rend. 8,7**
§V-B. Apagar `The prediction horizon is usually short, but some time gaps are much longer.` — as duas frases seguintes dizem-no com números (mediana 0,4 h–5,5 h; 5–27% acima de 3 dias) e ficam.

**S14 — 10 palavras — baixo — rend. 6,7**
§V-D. `and, for region prediction, a Markov-1 floor over region transitions on the same sliding windows and folds as our models` → `and, for region prediction, a Markov-1 floor over region transitions`. A §VI-B diz "computed under our windows and folds" onde os números do piso aparecem, que é onde importa.

**S15 — 10 palavras — baixo — rend. 6,7**
§V-C, parágrafo do plano estatístico. Apagar a frase de abertura `A claimed gain and a claimed match require different tests.` — o parágrafo executa-a explicitamente a seguir ("assigned a superiority test to next-category prediction and a non-inferiority test to next-region prediction"). **O plano, o desvio Wilcoxon→t emparelhado e o Holm m=6 não são tocados.**

**S16 — 9 palavras — baixo — rend. 6,0**
§V-B. `We then form overlapping sliding windows of nine visits and use the next visit as the target. A new window starts at each visit, which provides more examples for both tasks.` → `We then form overlapping sliding windows of nine visits, one starting at each visit, and use the next visit as the target.` ("overlapping" **tem** de ficar: a frase do POI-RGNN na §V-D refere-se a ela.)

**S17 — 9 palavras — baixo — rend. 6,0**
§V-C. Apagar `These assignments apply to each task across the datasets.` — o Holm dois parágrafos abaixo diz o mesmo com precisão ("across the six next-category comparisons and, separately, across the six next-region comparisons") e a legenda de `\ref{tab:results}` repete-o.

**S18 — 6 palavras — baixo — rend. 4,0**
§V-D, HMT-GRN. Apagar apenas `, which we do not study,`: `We remove its graph components and hierarchical beam search, which support exact next-place prediction, so the resulting model predicts region as one of its original targets but is not a reproduction of the complete published system.` **A cláusula protegida final fica literal.**

---

### 2.4 `sections/04_method.tex` — 233 palavras + a equação

**M1 — 30 palavras — baixo — rend. 20,0**
§IV-C. Apagar `This is standard fixed-weight joint training~\cite{caruana1997multitask}, kept simple by design so that any improvement over the dedicated single-task models comes from the shared representation, not from an adaptive weighting scheme.` — a §II-C tem o mesmo argumento com a mesma citação **e** o rastreio dos dezanove balanceadores; esta é a versão fraca. A citação sobrevive na §II-C.

**M2 — 22 palavras — baixo — rend. 14,7**
§IV-B, fim. `From the trained graph we extract one 64-dimensional vector per check-in (its region nodes have trained vectors as well), so a model sees a sequence of per-visit vectors rather than repeated per-place ones.` → `From the trained graph we extract one 64-dimensional vector per check-in.` (O parêntesis é dito duas frases adiante, na §IV-C, onde é usado; a cauda é a quinta enunciação do contraste visita/lugar.)

**M3 — 20 palavras — baixo — rend. 13,3**
§IV-B, abertura. `We want each visit, not each place, to have its own vector, and we call the resulting representation Check2HGI (Section~\ref{sec:related-embeddings}). Fig.~\ref{fig:dataflow} shows the data flow from check-ins to the two outputs.` → `We call this representation Check2HGI (Section~\ref{sec:related-embeddings}); Fig.~\ref{fig:dataflow} shows the data flow.`

**M4 — 19 palavras — baixo — rend. 12,7**
§IV-A. Apagar `We do not predict the exact next place; both properties are easier to learn and, for most uses, enough.` — dito na §I ¶2 e na §II-B/II-C; e, além disso, "easier to learn" está em tensão com a §II-B ("not to make the task easier"). Cortar remove 19 palavras **e** uma quase-contradição.

**M5 — 33 palavras — risco MÉDIO — rend. 11,0 — já estava no plano do autor**
§IV-C. Apagar `Class weighting, tested on both outputs, lowered both metrics, and logit adjustment replaces it on the category output only, because macro-F1 weights every category equally whereas Acc@10 rewards ranking the frequent regions well.`
*Perde-se:* um resultado negativo e a razão da assimetria; não estão em mais lado nenhum ("class weight" não aparece fora deste ficheiro). É o item 2 da Fase 2 do próprio `CONSOLIDATION_PLAN.md:614`. A frase anterior mantém a mecânica; a frase de justiça seguinte (`The dedicated category model receives the same adjustment`) **fica**.

**M6 — 16 palavras — baixo — rend. 10,7**
§IV-B. `We train the graph mainly with an infomax objective, under which each vector learns to match its real neighborhood and reject a shuffled one~\cite{...}, plus...` → `We train the graph mainly with an infomax objective~\cite{velickovic2019dgi,huang2023hgi}, plus two small label-free auxiliary terms (weights 0.3 and 0.1): a masked reconstruction of each place's aggregated category features, and an anchor to a place embedding pre-trained on the same data.` (A §II-A dá a glosa com as mesmas duas citações. Os pesos 0.3/0.1 ficam; `The training never sees the next category or the next region.` fica.)

**M7 — 15 palavras — baixo — rend. 10,0**
§IV-C. `a small branch inside the one model that reads the spatial input window and bypasses the shared trunk. The category task does not touch this branch.` → `a small branch inside the one model that bypasses the trunk.` ("private" já significa isso, e a Fig. 2 desenha-o. **`inside the one model` fica** — é o guarda da propriedade de modelo único.)

**M8 — 9 palavras — baixo — rend. 6,0**
§IV-A. `Region, unlike category, is a large label set: we frame next-region as classification over candidate regions, from 520 classes (Istanbul) to 8,501 (California; Table~\ref{tab:datasets}).` → `We frame next-region as classification over the dataset's regions, 520 to 8,501 of them (Table~\ref{tab:datasets}).`

**M9 — 9 palavras — baixo — rend. 6,0**
§IV-B. `Same-place visits connect through their shared place node, and nearby places are linked at the place level.` → `Nearby places are linked at the place level.` (a primeira metade é dedução directa de "Edges connect each level to the one above it").

**M10 — 16 palavras — risco MÉDIO — rend. 5,3**
§IV-B. `together with four elapsed-time features: the time since the user's previous visit and since the user's first visit, both on a logarithmic scale, the gap to the previous visit within the same day, and an indicator for a user's first visit.` → `together with four elapsed-time features: the log-scaled time since the user's previous visit and since their first visit, the same-day gap, and a first-visit indicator.` (As quatro continuam nomeadas uma a uma; a frase forward-only protegida que se segue continua a cobri-las com o mesmo "Each".)

**M11 — 5 palavras — risco NENHUM — rend. 5,0**
§IV-C. Apagar ` Inference uses the unadjusted logits.` — a mesma frase já diz `during training only`, quinze palavras antes.

**M12 — 15 palavras — risco MÉDIO — rend. 5,0**
§IV-C. Apagar `, so the tasks share by exchanging information rather than by owning hidden layers in common` — a oração anterior já o diz em termos de arquitectura ("keeping its own feed-forward weights") e a legenda da Fig. 2 repete-o.

**M13 — 4 — nenhum:** `a weight that decays as the time gap between the visits grows` → `a weight that decays with the time gap`.
**M14 — 4 — nenhum:** `We then train one model that reads a window` → `One model reads a window`.
**M15 — 11 — MÉDIO:** parágrafo dos parâmetros. `The joint model includes the shared trunk and both task outputs, so it is larger than either dedicated model, and a forward pass costs more compute than running the two: about 4.2 million...` → `The joint model is larger than either dedicated model and than the two combined, and a forward pass costs more compute: about 4.2 million parameters at Alabama against 1.9 million for the two combined (5.2 against 2.8 at California).` **Verificar depois de aplicar: 4.2 / 1.9 / 5.2 / 2.8 presentes, "larger than either dedicated model" presente, "than the two combined" presente, "operational rather than arithmetic" intocado.** Só sai a mobília.
**M16 — 3 — nenhum:** `the place's region (a census tract), and the city` → `the place's region, and the city` (definido quatro frases antes, na §IV-A, que fica).
**M17 — 2 — nenhum:** `(the semantic stream), and the region task reads the same window` → `(the semantic stream), the region task the same window`.

**M18 — 0 palavras, ~28 equivalentes (≈3 linhas de coluna) — risco NENHUM — o melhor rendimento do artigo**
§IV-C. Passar a equação (1) a linha:
*De:* `\begin{equation}\label{eq:loss}` … `\end{equation}` `where`
*Para:* `$L = 0.5\,L_{\mathrm{cat}} + 0.5\,L_{\mathrm{reg}}$, where`
`\ref{eq:loss}` não é citado em lado nenhum (só a própria `\label`). Zero palavras de texto, três linhas de coluna.

---

### 2.5 `sections/01_introduction.tex`, `02_related.tex`, `08_conclusion.tex` — 256 palavras

**C1 — 43 palavras — baixo — rend. 28,7** (`01_introduction.tex`, item 1 das contribuições)
Apagar a primeira frase do item: `We build a representation in which each check-in carries its own vector, describing that visit in its own context, rather than sharing one fixed vector per place: a fourth level, the visit, beneath the place, region, and city levels of hierarchical graph infomax.` O item passa a abrir na evidência (`Under a controlled comparison ... it improves next-category prediction at every one of the six datasets and in every one of their folds, by $+0.23$ to $+6.29$ points of macro-F1`; só `this` → `it`). O parágrafo seis linhas acima diz o mesmo, palavra por palavra, incluindo o quarto nível.

**C2 — 20 palavras — baixo — rend. 13,3** (`08_conclusion.tex`, último parágrafo)
`The differences between one model and two are bounded on both axes, the representation is the larger effect on category, and whether the exchange of information between the tasks adds anything beyond that is not separated by the evidence here (Section~\ref{sec:discussion}).` → `Whether the exchange of information between the tasks adds anything beyond the representation is not separated by the evidence here (Section~\ref{sec:discussion}).` (As duas orações cortadas são o resumo do resumo, a três e a oito linhas.)

**C3 — 18 — baixo:** §I item 1. Apagar `A single vector per place cannot separate places that serve several purposes, and per-visit context recovers that distinction.` — a §II-A diz o mesmo ("two visits to the same coffee shop look identical to the model").
**C4 — 17 — baixo:** §I item 2. Apagar `It shares semantic context between the tasks while keeping a private spatial path for the region task.` — terceira enunciação da arquitectura em meia coluna. A propriedade de modelo único (frase anterior, adição A3) **não se toca**.
**C5 — 15 — baixo:** §VIII ¶1. `On the category task the input representation moves the score more than the choice between one model and two at every one of the six datasets: giving each visit its own vector, instead of one fixed vector per place, improves next-category prediction at every dataset and in every fold, by...` → `On the category task the input representation moves the score more than the choice between one model and two: at every one of the six datasets and in every fold it improves next-category prediction, by $+0.23$ to $+6.29$ macro-F1 points.` (Nenhum número muda; a cobertura era dita duas vezes na mesma frase.)
**C6 — 14 — baixo:** §I ¶3. Apagar `one model does several jobs at once by sharing most of its parts, so` — o início da frase já é a definição. A sigla e `\cite{caruana1997multitask}` ficam.
**C7 — 14 — baixo:** §II-A. Apagar a glosa do DGI `, training them to tell the real network from a copy with shuffled node features`. (Nome, citação e descrição da rede ficam.)
**C8 — 13 — baixo:** §II-A ¶2. Apagar `The novelty is this specific combination: per-visit context inside a hierarchical graph-infomax representation.` — terceira formulação do mesmo contraste em cinco linhas. A frase seguinte, com os controlos, fica.
**C9 — 12 — baixo:** §I ¶3. `where sharing helps, what it costs, and how to share so the gains hold and the cost stays small.` → `where sharing helps and what it costs.` (A terceira cláusula promete um "como partilhar" que a §VII e a §VIII dizem que a evidência não isola.)
**C10 — 12 — baixo:** §II-A. Apagar `Our representation, Check2HGI, also works at the check-in level but through a different construction.` e mover o nome para a frase seguinte: `... our representation, Check2HGI, remains a network model, keeping the hierarchical place-to-region-to-city network ...`
**C11 — 7 — NENHUM:** §II-C. Apagar `The reason is visible in the gradients.` — a frase seguinte anuncia-se sozinha.
**C12 — 10 — baixo:** §II-C. `and we drop the next-place target entirely, so neither task is an intermediate step toward a third.` → `and we drop the next-place target entirely.` (As duas frases anteriores, sobre CSLSL e CatDM, já fixam o contraste. **Nenhum dos cinco sistemas nomeados é tocado.**)
**C13 — 9 — baixo:** §I ¶2. Apagar `(a neighborhood-scale unit such as the U.S. census tract)` — primeira de três, e a mais fraca ("such as", sem Istambul). A §IV-A fica como casa definicional.
**C14 — 8 — baixo:** §I ¶2. `The first captures intent, the second captures geography; we study whether one model should learn both at once.` → `We study whether one model should learn both at once.`
**C15 — 8 — baixo:** §VIII ¶2. Apagar `, the two datasets with the largest region vocabularies,`. Texas e Califórnia ficam nomeados; o veredito e a margem registada ficam literais. **Compatível com D1** — o boletim do §VIII sobrevive.
**C16 — 8 — baixo:** §II-B. `Predicting over a partition of the map is also the standard formulation in the human-mobility literature, with a grid cell as the target~\cite{luca2021mobilitysurvey}; our next-region task substitutes official neighborhood-scale units for grid cells.` → `Predicting over a partition of the map is standard in the human-mobility literature, with a grid cell as the target~\cite{luca2021mobilitysurvey}; we substitute official neighborhood-scale units.`
**C17 — 15 — MÉDIO:** §I item 2, ressalva anti-causal. Substituir (não apagar) por: `The pairing of the two region gains with the two largest region vocabularies is an observation and not a law (Section~\ref{sec:results-part2}).` **Não apagar a ressalva por inteiro** — o item continua a dizer "outperforms ... at the two datasets with the largest region vocabularies" e uma agrupação sem guarda, na parte mais lida do artigo, é exactamente o que a cultura do texto proíbe. A ressalva completa (não-monotonia + co-variação com o tamanho do corpus) fica em §VI-B, colada ao par protegido.
**C18 — 7 — baixo:** §II-C. Apagar `that re-weight the tasks' updates during training` (PCGrad e Nash-MTL continuam nomeados com as citações).
**C19 — 6 — baixo:** §II-B. Apagar `and found no consistent multi-task gain.` — a §I diz o mesmo e diz mais. **A correcção da errata B.1 (classificação estática + próxima-categoria, sem tarefa de região) fica intacta.**

---

### 2.6 Tabelas, figuras, resumo, preâmbulo, bibliografia — ~225 col-pt líquidos (≈183 equivalentes) — **FAZER POR ÚLTIMO**

**T1 — 125,5 col-pt (0,093 página) — baixo — o maior item não-prosa**
`references.bib` + `main.tex`. Acrescentar ao .bib:
```
@IEEEtranBSTCTL{IEEEexample:BSTcontrol,
  CTLuse_forced_etal = "yes",
  CTLmax_names_forced_etal = "3",
  CTLnames_show_etal = "1"
}
```
e `\bstctlcite{IEEEexample:BSTcontrol}` imediatamente a seguir a `\maketitle`. Entradas com quatro ou mais autores passam a "A. Navon *et al.*"; as de três ou menos ficam iguais. **Correr `bibtex` + duas passagens.** Desaparecem 14 linhas impressas. Nenhuma chave, número ou referência cruzada muda.

**T2 — 17 palavras — baixo** — resumo, quatro compressões (`main.tex`):
(a) `We give each check-in its own vector, describing that visit in its own context, instead of giving every place one fixed vector.` → `We give each check-in its own vector instead of giving every place one fixed vector.`
(b) `Under a controlled comparison, with the same model, folds, windows, and training configuration and only the input changed,` → `With the same model, folds, windows, and training configuration, and only the input changed,`
(c) `Five of the datasets are U.S. states and one is a non-U.S. city.` → `Five datasets are U.S. states and one is a non-U.S. city.`
(d) `We then ask whether one model, trained on that representation, can answer both questions in a single forward pass.` → `We then ask whether one model can answer both questions in a single forward pass.`
**A quinta compressão do caçador (apagar o fecho `One model therefore serves both tasks at a cost bounded on both axes.`) NÃO entra** — é a tese de fecho do resumo. Todos os números protegidos ficam literais.

**T3 — 22 palavras / 18,0 col-pt — NENHUM** — legenda da Fig. 1 (`main.tex`):
→ `From a check-in sequence to two predictions. Each visit is embedded in a four-level graph trained without task labels; sliding nine-visit windows of the resulting per-visit and per-region vectors feed a single model.` (Os quatro níveis, os dois conjuntos de vectores e as duas saídas estão desenhados como rótulos dentro da própria figura.)

**T4 — 24 palavras / 17,9 col-pt — NENHUM** — `tables/tbl1_datasets.tex`, legenda, duas primeiras frases:
→ `Dataset statistics, ordered by check-in count. Windows: ...`
**Condição:** a coluna *Source* tem de ficar (passa a ser a única declaração de proveniência dentro da tabela), e a §IV-A tem de ficar intacta (rejeição #1). Verificação depois de S1+T4: `grep -n "same seven" sections/04_method.tex` tem de devolver a frase do mapeamento de Istambul.

**T5 — 17 palavras / 17,9 col-pt — NENHUM** — `tables/tbl3_results.tex` (**Tabela II**, a escada), legenda: apagar `The region counts in the second column do not follow that order.` e `Category is macro-F1; region is Acc@10.` (esta última está impressa nos cabeçalhos de grupo mesmo por baixo); `..., ordered by check-in count (Table~\ref{tab:datasets}).` → `..., in the order of Table~\ref{tab:datasets}.` **A legenda de ↑/negrito com Holm, a legenda de ≈ com a margem registada e os dois testes unilaterais, e `all four are small deficits` ficam palavra por palavra.**

**T6 — 12 palavras / 17,9 col-pt — baixo** — `tables/tbl1_datasets.tex`, segunda passagem (**só depois de T4**): apagar `Windows: sliding nine-visit inputs plus the next-visit target; ` e ` (Food in every dataset)`. A definição da coluna *Majority* fica. *Nota honesta: parte desta poupança é reabsorvida pelo re-empacotamento até a prosa entrar — por isso as tabelas são as últimas.*

**T7 — 7 palavras / 9,0 col-pt — NENHUM** — legenda da Fig. 2: apagar `One model, one forward pass, two predictions.` (é um rótulo desenhado dentro da própria figura, mais o item da §I e duas frases da §VI/§VII). **`the only place where the two tasks interact` e o caminho espacial privado ficam.**

**T8 — 15,6 col-pt — baixo** — `main.tex`: `\renewcommand{\arraystretch}{0.95}` → `{0.90}`. Passo de linha 8,5 → 8,1 pt nas três tabelas; altura de glifo 7,1 pt, folga confortável. Nenhuma célula muda. Reversível num carácter.

**T9 — 7,0 col-pt — baixo** — `tables/tbl2_substrate.tex`: `\centering\small` → `\centering`. O `\small` explícito estava a subir o corpo desta tabela acima das outras duas (passo 9,5 contra 8,5 pt). O `\sd` é `\scriptsize` absoluto: **a coluna de sd de folds não muda de tamanho**.

**T10 — 9,0 col-pt — NENHUM** — `references.bib`: apagar o campo `url` da entrada `navon22` (a entrada mantém "Proceedings of the 39th ICML, ser. PMLR vol. 162, pp. 16 428–16 446") e `journal = {The Journal of Supercomputing}` → `{J. Supercomput.}` (todos os outros periódicos já estão abreviados ao estilo IEEE — isto também corrige a inconsistência).

---

## 3. O que REJEITO — e porquê

**Colisões (aplicar os dois destruía o facto):**

1. **§IV-A `(for Istanbul, the mahalle, a municipal neighborhood)` (7) e `We map Istanbul's source labels onto the same seven.` (9).** Colidem com S1 e T4, que apagam as cópias da §V-A e da legenda da Tabela I. Só uma casa pode ficar e é a §IV-A, que é a definicional. **Não cortar.**
2. **Nota de rodapé de `tables/tbl2_substrate.tex` (`All five folds favor... p=0.07`, 24 palavras de legenda).** Colide com R2. Só uma cópia pode sair: a prosa vale ~39 col-pt, a nota ~20, e a nota preserva **as duas** afirmações (unanimidade em todos os datasets **e** a excepção FL). Sai a prosa, fica a nota. **Depois de R2+R3, esta nota é a única fonte da unanimidade em prosa/legenda — apagá-la seria um erro grave.**
3. **Nota de rodapé de `tables/tbl3_results.tex` com a cobertura da busca (33).** É o sinal de justiça no ponto de uso, mesmo por baixo da escada. Sai antes o ponteiro de prosa (R13). Fica dita duas vezes (§V-B + nota), o que cumpre "stated once" com folga.

**Toca na lista-lei ou no seu perímetro imediato:**

4. **§VI `The intervals themselves carry what can be said about magnitude: the widest of them` (9).** Existe e é redundante, mas reescreve a abertura da frase protegida do limite 0,334. Fora de questão.
5. **§VII, braço de 174,8% do orçamento (17) e Texas 170,5%/+0,21 (14).** Fora do núcleo D4 à letra, mas 31 palavras não pagam esvaziar de números o parágrafo do controlo de capacidade — é o parágrafo que um árbitro lê com mais atenção, e a saturação é o que mostra que o emparelhamento a 97,4% caiu no planalto e não num tamanho de sorte.
6. **§VII, reflow da frase "geographic size of a miss" (10).** Está dentro da passagem protegida do R2(a). Dez palavras não pagam entrar lá.
7. **§VIII `The central concern was that sharing could help one task while hurting the other.` (14).** Encosta directamente na frase protegida P2 (verificado: é a frase imediatamente anterior).
8. **§V-C `The analysis plan also fixed the two-point margin in advance.` (10).** É a declaração de pré-registo dentro do parágrafo que define o teste de não-inferioridade — o sítio exacto onde se procura p-hacking.

**Custa mais em credibilidade do que poupa:**

9. **§VI `and the dedicated and joint models are above it at every dataset` (12).** O piso de Markov não está em nenhuma tabela; o leitor não pode re-derivar que o modelo dedicado também o supera.
10. **§VI, descritores das comparações externas (`the primary region-native comparison`, `trained from the raw check-ins`, `reference`) (11).** A hierarquia primário/referência é sinal de justiça, e território de decisão do autor.
11. **§VI-A, contraste do CTLE `whereas our graph reads each visit's category and time of day as input features` (14).** Faz trabalho retórico exactamente onde está.
12. **§VI `(an epoch is one pass over the training data)` (9).** A política do glossário para esta audiência é do autor (o mesmo glossário que baniu "arm").
13. **§II-A `The order of a user's visits still enters the network, as links between consecutive check-ins` (17).** É a réplica imediata à objecção "modelo de rede ignora a ordem", num venue de mobilidade; a divulgação forward-only fica uma página adiante.
14. **§II-C `This is a finding for this pair of tasks, not a general rule.` (13).** Guarda anti-generalização sobre os balanceadores.
15. **§I `To our knowledge, this is the first work to treat fine-grained region as an end target of equal standing` (21).** É posicionamento nas contribuições, não redundância técnica. Decisão do autor.
16. **§V-C `the standard deviation of the paired difference ranges from 0.02 to 0.16 points` (18).** Decisão COD-006a do autor, que substituiu "well powered" por precisão observada. Não se reabre uma decisão fechada na véspera.
17. **§V-C, reescrita da definição de seed (10)** e **§VII, reescrita do segundo limite (10)**: texto trancado por errata (2026-08-04 e COD-006). Vinte palavras não pagam churn em prosa já corrigida uma vez.
18. **§V-D `The released code provides the full training configuration for every model` (12).** Compromisso de reprodutibilidade mais largo do que o que a nota 1 enumera. (Se faltarem 7 palavras no fim: fundir na frase anterior — `All other choices, including the nine-visit window, were fixed during development and are released with the code (footnote~\ref{fn:code}).`)
19. **§IV-A `one of the seven top-level categories that the Gowalla data distributes with its places` (3).** Prosa que responde ao ponto 1 do R1. Três palavras.
20. **`tables/tbl2_substrate.tex`: tirar os seis `\underline{}` e `and underline the second` (9).** Muda as marcas de uma tabela protegida por 9 palavras. Decisão do autor, não do escritor.
21. **Fundir as Figuras 1 e 2 num float de dois painéis (171 col-pt = 0,13 página).** Rótulos medidos a 5,19 pt e 5,51 pt em impressão a preto e branco. **Alavanca de emergência, só se as 8 páginas falharem por um triz e o autor decidir.**

**Reserva medida, se faltar (não recomendada, por ordem de preferência):** reescrita completa do parágrafo do tronco na §VII (+22 além de D3) · ressalva do vocabulário na §I apagada em vez de encurtada (+22 além de C17) · reivindicação de primazia na §I (21) · sd 0,02–0,16 (18) · ordem das visitas na §II-A (17) · contraste do CTLE (14) · "central concern" no §VIII (14) · código libertado, fundido (7) · sublinhados da Tabela III (9). **Total ~145 palavras + a fusão de figuras (0,13 página).**
**Item não medido, para o dono da §VI avaliar se faltar:** §VI linhas 193-196 repetem a conclusão do quinto limite **e** a frase protegida "Texas has no control of matched size", já com ponteiro para a §VII — o caçador estimou "~30 palavras", **estimativa, não medição**. Medir antes de usar.

---

## 4. Ordem de execução e onde medir

```bash
cp -a /tmp/edascheck /tmp/edascut && cd /tmp/edascut
./gate_v17.sh                      # verde ANTES de tocar em nada
```

**Ordem (prosa primeiro, floats no fim — os floats voltam a empacotar-se a cada mudança de prosa e medir antes disso engana):**

| # | Ficheiro | Itens | Palavras | Nota |
|---|---|---|---|---|
| 1 | `sections/07_discussion.tex` | D1, D2, D3, **D4 por último** | 155 | D4 exige a checklist de renumeração |
| 2 | `sections/06_results.tex` | R1 → R13 | 263 | R2+R3 dependem da nota de `tbl2_substrate` ficar |
| 3 | `sections/05_setup.tex` | S1 → S18 | 339 | **S5 antes de S9** |
| 4 | `sections/04_method.tex` | M1 → M18 | 233 | M18 (equação) é grátis: fazer primeiro deste bloco |
| 5 | `01_introduction`, `02_related`, `08_conclusion` | C1 → C19 | 256 | C15 é compatível com D1; nenhum outro corte ao §VIII ¶2 |
| 6 | tabelas + `main.tex` + `references.bib` | T1 → T10 | ~225 col-pt | **só agora**; T4 antes de T6; T1 exige `bibtex` |

**Medir a seguir a CADA ficheiro (não só no fim):**

```bash
pdflatex -interaction=nonstopmode main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null
# quando o .bib mudar (T1/T10):  pdflatex; bibtex main; pdflatex; pdflatex
pdfinfo main.pdf | grep Pages                      # alvo: 8
grep -c "Overfull" main.log                        # alvo: 0
grep -c "undefined" main.log                       # alvo: 0
N=$(pdfinfo main.pdf | awk '/Pages/{print $2}')
pdftotext -f $N -l $N main.pdf - | wc -w           # ~1090 palavras = página cheia; é o medidor de enchimento
./gate_v17.sh                                      # PORTÃO VERDE
```

**Verificações finais, antes de copiar para `articles/[mobiwac]/src/`:**

```bash
grep -c "0.334\|+0.41\|p = 0.010\|97.4\|no control of matched size" sections/07_discussion.tex sections/06_results.tex
grep -n "same seven" sections/04_method.tex        # o mapeamento de Istambul TEM de existir (S1+T4)
grep -n "All five folds favor" tables/tbl2_substrate.tex   # a nota TEM de existir (R2+R3)
grep -n "Configuration search" tables/tbl3_results.tex     # a nota TEM de existir (R13)
grep -n "Five limits\|Fifth\|Fourth\|Third" sections/07_discussion.tex   # coerente depois de D4
pdftotext main.pdf - | grep -c "±"                 # as duas colunas de sd continuam impressas
```

Se ao fim do passo 5 o `pdfinfo` já disser 8 páginas, os itens T4/T6/T8/T9/T10 tornam-se folga: aplique-os na mesma (custam minutos e compram margem contra a repaginação final), mas **T1 e T2 valem sempre a pena** e **T3/T5/T7 são risco nenhum**.