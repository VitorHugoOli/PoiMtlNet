# CAMERA_READY.md — o estado do artigo MobiWac, e o que muda a partir daqui

> **Artigo:** *Predicting the Next Category and Region of a Visit: A Check-in-Level Multi-Task
> Study on Mobility Data.* MobiWac 2026 (23.º Intl. Symposium on Mobility Management and Wireless
> Access), Paris, 26–30 out 2026. EDAS **#1571313639**, Regular Paper, single-blind.
> **ACEITE em 2026-08-26** (registo: `articles/dissertacao/CLAUDE.md:130`, decisão AUT-35).
>
> **Este ficheiro é o ponto de entrada da campanha de camera-ready.** Foi escrito em 2026-09-04 a
> partir de uma varredura de 17 agentes sobre toda a pasta `[mobiwac]/`, sobre o estudo `v18` e
> sobre o Capítulo 5 da dissertação. Substitui, **em matéria de números e vereditos**, tudo o que
> `CLAUDE.md`, `GLOSSARY.md §1`, `PAPER_PLAN.md §3` e `ERRATA.md` dizem sobre resultados.

---

## 0 · Como usar este ficheiro

**Ordem de leitura obrigatória para qualquer agente que vá tocar no artigo:**

1. Este ficheiro, §1 a §7.
2. [`GLOSSARY.md`](GLOSSARY.md) — **continua a ser a lei de escrita** (nomes, verbos, palavras
   proibidas, registo). O que nela está morto são os **números** e a **escada de vereditos** do §1;
   as regras de linguagem valem inteiras.
3. Os `.tex` em [`src_fix/`](src_fix/) — o texto de trabalho.
4. Tudo o resto **só se alguém pedir**.

**Regra de precedência.** Onde este ficheiro e outro documento da pasta discordarem sobre um número
ou um veredito, **este ficheiro ganha**. Onde discordarem sobre uma regra de escrita, `GLOSSARY.md`
ganha. Não há terceiro caso.

**Regra do dedo queimado.** Qualquer macro-F1 de próxima-categoria na casa dos **50, 60 ou 70** é
um número da geração com vazamento. Pare. Os valores válidos vivem entre **30 e 38**.
⚠ A regra só protege o eixo da **categoria**. A região mexeu menos de 2 pp entre gerações, portanto
um número de região **não** se denuncia pelo valor: confira o caminho do ficheiro.

---

## 1 · O facto central, numa página

O artigo foi aceite com os números da geração **v17**. Esses números estão inflacionados por um
**vazamento de rótulo no grafo de visitas consecutivas**: o pré-processador emitia as arestas nos
**dois sentidos**, portanto o vector de uma visita era convolvido sobre uma vizinhança que continha
a visita **seguinte** — e a categoria é uma *feature* de entrada do nó. A cabeça de categoria via
uma feature do seu próprio alvo. Em Alabama isso vale **28,63 macro-F1** (56,86 → 28,23).
A região quase não é tocada pelo vazamento: a torre de região lê constantes por região, que o
vazamento não alcança. Em Alabama o resultado conjunto de região moveu-se **+0,01 pp** sob a leitura
estrita, e o estudo fixou **2 pp** como limiar de investigação.
*Prova:* `docs/studies/closing_data/v18/METHODOLOGY.md:34-37`.

> ⚠ **"≤ 0,33 pp" não é o movimento absoluto da região** — é um número que circula e que engana. Os
> 0,33 são o deslocamento do **delta** conjunto−dedicado entre gerações (`V18_RESULTS.md:53-60`,
> coluna *shift*: −0,14 … −0,33; resumido em `CHARTER_COMPLIANCE.md:22`), e coincidem por acaso com o
> resíduo do controlo de transdutividade (−0,33 … +0,01). Em absoluto, o braço **dedicado** de região
> mexeu ≤ **0,03 pp** entre v17 e v18, mas o braço **conjunto** mexeu até **1,15 pp**
> (CA 65,69 → 64,54; FL −0,87; TX −0,91).

O estudo **v18** corre sobre um grafo **forward-only** (`src < tgt`, no treino e na leitura), mais
4 colunas de tempo decorrido nas features de nó (`in_channels = 15`). **Não é uma mudança de
arquitectura** — o modelo, as cabeças e o selector são os do v17.

> ⚠ **Mas a receita mudou, e isso tem de constar da nota aos chairs.** `CHARTER_COMPLIANCE.md §2.1`
> é explícito: da receita do charter (`--category-weight 0.75`, `--cat-lr 1e-3`, tiers de batch/LR do
> `CEILINGS_N20_FINAL.md`) *"nada disso foi usado na regeneração final"*. Um *sweep* de 103 braços
> re-afinou **os dois** braços — deliberadamente, porque afinar só o baseline enviesaria o Δcat
> contra o MTL. O que mudou: batch **8192 em todos**; max_lr **AL 0,0025 · AZ/IST 0,0005 ·
> FL/CA/TX 0,005**; `category_weight` **0,75 → 0,50**; cat-lr do MTL **1e-3 pequenos / 2e-3 grandes**;
> pesos de classe **substituídos por *logit adjustment* τ = 0,5** (região τ = 0).
> A formulação honesta é *"vazamento removido **e** ambos os braços re-afinados"*, não *"o mesmo
> modelo, medido sem o vazamento"*.
> ⚠ `METHODOLOGY.md` e `V18_RESULTS.md` (o cabeçalho de cada um) ainda dizem *"the frozen v17 recipe"* — são anteriores ao
> `CHARTER_COMPLIANCE.md` (2026-08-11) e estão desactualizados nesse ponto.

| Alegação | Como foi aceite (v17) | Sob v18 (sem vazamento) |
|---|---|---|
| Categoria, conjunto vs. dedicado | **supera em 6/6**, +5,33 … +9,35 | supera em **1/6** (Florida +0,19, Holm p = 0,011). **Quatro** das outras cinco são negativas |
| Região, conjunto vs. dedicado | supera em **4/6** (Ist +0,19, FL +0,71, TX +2,11, CA +2,20) | supera em **2/6** (TX +1,21, CA +1,06). As outras quatro são **défices**; três resolvem-se no sentido negativo |
| Representação (Tab. 2) | **+27,63 … +39,62** macro-F1 | **+0,23 … +6,29** |
| Células absolutas de categoria | 54,7 – 79,8 | 30,6 – 37,6 (queda de −27,9 a −42,3 no braço **conjunto**; −19,4 a −37,2 no **dedicado**) |
| Lei de escala com o n.º de regiões | monótona nos cinco estados | **morta** (CA tem mais regiões do que TX e ganho menor) |

**O detalhe agravante — e ao contrário do que se supôs.** O manuscrito **aceite NÃO declarava este
canal**, e isso está agora verificado contra o PDF real.

> ✅ **RESOLVIDO 2026-09-06 — o autor forneceu o PDF submetido, e ele está identificado.**
> O artigo enviado ao EDAS é **byte-a-byte** o commit **`0834419b`** (2026-07-10 22:56), 8 páginas,
> `md5 dce01efe60db68a50f531cdd0a88dbde`. Provado por duas vias independentes: o md5 do blob do git
> é igual ao do PDF, e o `CreationDate` do PDF (`Fri Jul 10 22:56:09 2026 -03`) coincide com a data
> do commit. Etiquetado como **`mobiwac/submetido-EDAS`**.
>
> ⚠ **Duas identificações anteriores estavam erradas, as duas minhas.** Citei primeiro o `f66f8a73`
> (2026-07-20 — nove dias *depois* do prazo) e depois o `97f01a50` (2026-07-11 00:28). E não eram
> "dois candidatos": ✔ são **dezasseis** blobs distintos de 8 páginas entre 07-09 e 07-20, dos quais
> **treze** elegíveis pelo prazo. O teste que eu usava — *"o abstract casa com o bloco do
> `EDAS_SUBMISSION.md`"* — **não distingue nenhum deles**.
>
> ✔ **A alegação que isto sustentava aguenta-se, e agora está testada à exaustão:** procurei
> `absorb the category`, `linear probe`, `consecutive-visit edge` e `node input feature` nos **treze**
> builds elegíveis — **zero ocorrências em todos os treze**. O texto submetido abre a §7 com
> *"**Three** limits qualify these results"*, confirmado no próprio PDF do autor. **Nenhuma versão que
> os revisores pudessem ter lido declarava o canal de vazamento.**
>
> A quarta limitação (*"the vector of an
earlier visit could absorb the category of the next one"*) entrou em **2026-08-05**, no commit
`0b472205` — quase um mês depois do prazo de 2026-07-11. É uma das 17 correcções pós-submissão que
os revisores nunca viram (§2, §6 D8).
**Os revisores aceitaram o artigo sem a ameaça nomeada**, o que torna a razão para não deixar os
*chairs* de fora **mais forte**, não mais fraca — ver **D1**.

---

## 2 · As três árvores, e qual é a base

| Árvore | Números | Páginas | Último toque | O que é |
|---|---|---|---|---|
| [`src/`](src/) | **v17 (com vazamento)** | 9 (8 no PDF enviado) | 2026-08-06 | O texto aceite, **mais 17 correcções pós-submissão que os revisores nunca viram** |
| [`src_fix/`](src_fix/) | **v18** | **10** | 2026-08-12 | A reescrita v18 completa, já aplicada (18 commits em 2026-08-11/12) |
| `articles/dissertacao/src/chapters/5_mobiwac/` + `src/tables/mobiwac/` | **v18 + correcções de Setembro** | n/a | 2026-09-02 | O mais recente. Carrega ~10 correcções que o `src_fix/` não tem |

**Base recomendada: `src_fix/`, com as correcções de Setembro trazidas da dissertação, ficheiro a
ficheiro** (lista em §8). A `articles/dissertacao/CLAUDE.md:190` já regista: *"o paper de referência
é `src_fix/`, não `src/`"*.

⚠ **A instrução de re-sincronização em `ERRATA.md` aponta no sentido errado.** Ela manda
re-sincronizar a dissertação a partir de `src/`. Hoje **a dissertação está um mês à frente**.

⚠ **Ledgers de adaptação mortos:** `articles/dissertacao/src_utils/adaptation_ledgers/5_mobiwac_ADAPTATION_LEDGER.md`
está congelado em 2026-07-23 e afirma coisas falsas desde 2026-08-06. Não usar.
O ledger vivo de divergência é `articles/dissertacao/src/tables/mobiwac/errata_scope.tex`.

---

## 3 · Os números canónicos (v18, convenção **joint-best**)

**Convenção de registo:** *joint-best* = **um** checkpoint guardado por fold, selector de validação
`geom_simple`, `min_best_epoch 0`, as duas tarefas lidas nessa mesma época. É o modelo que se
serve. n = 20 (4 sementes {0,1,7,100} × 5 folds); ± é o desvio-padrão **entre as quatro médias por
semente**.
**Teste:** *t* pareado unilateral sobre as quatro médias por semente (n = 4, gl = 3), **Holm dentro
de cada família de tarefa, m = 6**; IC a 90 %. Não-inferioridade: **TOST a 2 pp, pré-registado
apenas para o eixo da região**.

**Fonte única e verificada:** `articles/dissertacao/wrapup/evidence/ladder_recompute.json`
(reproduzida célula a célula nesta sessão) · células conjuntas por fold em
`docs/results/closing_data/v18/joint_best_perfold.json` · células dedicadas em
`docs/studies/closing_data/v18/data/v18_results.json`.

### 3.1 Próxima categoria (macro-F1)

| Dataset | Dedicado | Conjunto | Δ | IC 90 % | p (Holm) | Veredito |
|---|---:|---:|---:|---|---:|---|
| Istanbul | 35,34 | 35,42 | **+0,08** | +0,011 … +0,149 | 0,181 | **não resolvido** |
| Alabama | 30,77 | 30,59 | **−0,19** | −0,334 … −0,043 | 1,000 | não resolvido (**défice**, IC exclui 0) |
| Arizona | 34,57 | 34,57 | **−0,00** | −0,042 … +0,035 | 1,000 | não resolvido |
| Florida | 37,35 | **37,55** | **+0,19** | +0,140 … +0,249 | **0,0107** | **supera ↑** |
| Texas | 36,33 | 36,19 | **−0,13** | −0,186 … −0,076 | 1,000 | não resolvido (**défice**, IC exclui 0) |
| California | 35,63 | 35,63 | **−0,00** | −0,029 … +0,020 | 1,000 | não resolvido |

### 3.2 Próxima região (Acc@10)

| Dataset | Dedicado | Conjunto | Δ | IC 90 % | p (Holm) | TOST (δ=2 pp) | Veredito |
|---|---:|---:|---:|---|---:|---:|---|
| Istanbul | 75,16 | 75,08 | **−0,08** | −0,156 … −0,002 | 1,000 | 0,0000 | não-inferior ≈ (défice não resolvido) |
| Alabama | 70,12 | 69,24 | **−0,87** | −1,003 … −0,746 | 1,000 | 0,0001 | não-inferior ≈ (**défice resolvido**) |
| Arizona | 59,48 | 59,04 | **−0,44** | −0,622 … −0,252 | 1,000 | 0,0001 | não-inferior ≈ (**défice resolvido**) |
| Florida | 76,69 | 76,54 | **−0,16** | −0,187 … −0,126 | 1,000 | 0,0000 | não-inferior ≈ (**défice resolvido**) |
| Texas | 64,94 | **66,15** | **+1,21** | +1,125 … +1,287 | **0,00013** | — | **supera ↑** |
| California | 63,48 | **64,54** | **+1,06** | +1,033 … +1,082 | **0,0000063** | — | **supera ↑** |

**Três de doze células são vitórias. Oito das doze diferenças têm estimativa pontual negativa**
(seis com magnitude ≥ 0,08; as de AZ e CA na categoria, −0,004 cada, imprimem-se como −0,00).
As quatro células "match" da região são **défices**, nunca empates. Nunca escrever "no difference".

### 3.3 Tabela 2 — a representação (categoria, semente 0, 5 folds)

| Dataset | Nível de check-in | Nível de lugar (HGI) | Δ | folds a favor | p (t pareado) |
|---|---:|---:|---:|---:|---:|
| Istanbul | 35,35 | 29,07 | **+6,29** | 5/5 | 0,00003 |
| Alabama | 30,77 | 29,15 | **+1,62** | 5/5 | 0,0034 |
| Arizona | 34,51 | 31,93 | **+2,58** | 5/5 | 0,0004 |
| Florida | 37,36 | 37,13 | **+0,23** | 5/5 | **0,067 (n.s.)** |
| Texas | 36,32 | 35,33 | **+0,99** | 5/5 | 0,00003 |
| California | 35,62 | 34,74 | **+0,88** | 5/5 | 0,0005 |

Três factos que nenhuma das duas tabelas diz e que o camera-ready não deve perder:
1. **A coluna "check-in" É a fatia da semente 0 da coluna "Dedicado categoria" da Tabela 3.** É por
   isso que AZ lê 34,51 aqui e 34,57 ali. Não é erro.
2. **Os dois braços mexeram desde o v17.** O braço check-in caiu (removeu-se o vazamento); o braço
   de lugar **subiu** (Istanbul 26,56 → 29,07) só porque foi re-medido sob a receita v18. Quem
   substituir apenas a coluna do check-in calcula margens erradas.
3. **A contribuição 1 é n = 5 (uma semente); a contribuição 2 é n = 20.** O desvio-padrão entre
   folds em FL (0,42) é maior do que o gap de FL (+0,23).

### 3.4 Referências externas (Tabela 3) — **inalteradas pelo v18**

O vazamento vivia **apenas** dentro dos nossos vectores por visita. POI-RGNN, STAN e ReHDM treinam
a partir de entradas cruas; Markov-K conta rótulos; HMT-GRN lê só as colunas de POI-id e rótulo e
aprende a sua própria tabela de embeddings. As janelas e os folds são idênticos byte a byte entre
o board v17 e o `check2hgi_v18` (retenção 1,0000 em todos os estados, `docs/studies/closing_data/v18/AUDIT.md`).

| Dataset | Regiões | Markov-K (cat) | POI-RGNN (cat) | HMT-GRN (reg) | ReHDM (reg) | STAN (reg) |
|---|---:|---:|---:|---:|---:|---:|
| Istanbul | 520 | 24,55 | 30,12 | 60,4 | 69,33 | 61,86 |
| Alabama | 1 109 | 20,50 | 23,80 | 57,05 | 65,38 | 60,72 |
| Arizona | 1 547 | 23,92 | 27,64 | 43,70 | 53,00 | 49,86 |
| Florida | 4 703 | 29,74 | 34,49 | 63,74 | 64,49 | 72,99 |
| Texas | 6 553 | 28,67 | 33,03 | 53,85 | 48,81 ‡ | 61,67 † |
| California | 8 501 | 27,58 | 31,78 | 49,61 | 50,26 ‡ | 58,52 † |

† STAN folds parciais: TX 4/5, CA 2/5 (semente 0). ‡ ReHDM em TX e CA: uma semente.

**Margens do conjunto sobre as externas (recalculadas):** categoria ≥ **+3,06** (mínimo em FL,
sobre POI-RGNN); região ≥ **+3,55** (mínimo em FL, sobre STAN). Piso Markov-1 stride-1 (51,23 …
72,47): excedido por **+4,07 a +10,02**.
⚠ **O modelo dedicado também está acima de todas as externas em todos os datasets** (cat mín.
+2,86; reg mín. +3,27). "Acima de todas as externas" **não** é uma propriedade só do conjunto.
⚠ A margem externa **inclui** a vantagem da representação, portanto **não é prova sobre MTL** e não
pode liderar as contribuições nem o resumo.

---

## 4 · NUNCA CITAR — o que era v17, e o que o substitui

| v17 (NUNCA CITAR) | Substituto v18 |
|---|---|
| Categoria conjunto **63,32 / 64,51 / 65,79 / 79,84 / 77,24 / 77,05** | 35,42 / 30,59 / 34,57 / 37,55 / 36,19 / 35,63 |
| Categoria dedicado **54,74 / 56,82 / 56,43 / 74,51 / 69,79 / 70,60** | 35,34 / 30,77 / 34,57 / 37,35 / 36,33 / 35,63 |
| Região conjunto **75,35 / 69,70 / 59,46 / 77,41 / 67,06 / 65,69** | 75,08 / 69,24 / 59,04 / 76,54 / 66,15 / 64,54 |
| Δcat **+5,33 … +9,35**; "supera em todos os datasets" | **−0,19 … +0,19**; supera **só em Florida** |
| Região "supera em quatro de seis"; Ist +0,19 / FL +0,71 / TX +2,11 / CA +2,20 | supera em **dois**: TX +1,21, CA +1,06 |
| "matches em Alabama e Arizona" (região) | matches (TOST) em **Istanbul, AL, AZ, FL** — as quatro são **défices** |
| Representação **+27,63 … +39,62**; "cerca de 28 a 40 pontos" | **+0,23 … +6,29** |
| Coluna de lugar 26,56 / 26,56 / 29,50 / 35,53 / 32,48 / 32,31 | 29,07 / 29,15 / 31,93 / 37,13 / 35,33 / 34,74 |
| Âncora de Florida **75,15**, e toda a coluna de check-in da Tab. 2: **54,65 / 55,87 / 57,13 / 75,15 / 69,95 / 70,26** | 35,35 / 30,77 / 34,51 / 37,36 / 36,32 / 35,62 (FL: 37,36 na Tab. 2, 37,35 na Tab. 3) |
| Deltas CTLE **+37,8 / +37,0 / +28,7** | **não existe comparando v18**. Só a ordenação sobrevive |
| "roughly **64 to 90** percent of the gain" (`src/sections/06_results.tex`; atribuição no comentário do `src/main.tex` e no bullet 2 da `01_introduction.tex`). ⚠ A variante "64–72 % / 89–90 %" só existe no `PAPER_PLAN.md`, **nunca no artigo** — procurar pela cadeia errada deixa a frase viva | **retirado** (já feito no `src_fix`, substituído pelo controlo de concatenação) |
| "pelo menos 4 Acc@10" e "pelo menos 33 macro-F1" sobre as externas | **3,55** e **3,06** |
| Markov excedido por "4,9 a 10,3" (`src/sections/06_results.tex`, o parágrafo do piso de Markov) | **4,1 a 10,0**. ⚠ **O piso 51–72 NÃO muda** — a árvore aceite já o imprime. A gama "43–65" é do `src_v1`, retirada em 2026-07-18; não está no texto aceite |
| Taxa de acerto CA **65,69 %** | **64,54 %** |
| Convenção de época: "no máximo 0,06 (cat) e 0,11 (reg)" | **0,17 / 0,90** (média por fold) ou **0,23 / 0,93** (pior semente) |
| Perda `L = 0,75 L_cat + 0,25 L_reg`, CE simples nas duas | **0,50 / 0,50**, com *logit adjustment* τ = 0,5 só na cabeça de categoria |
| "1,1 M / 2,0 M" para os dois dedicados somados — **prosa viva** (`src/sections/04_method.tex:46-47`, ainda viva em `src_fix/…:55-57`). "+5 % de parâmetros" só sobrevive num comentário que já a retira; "duas respostas ao preço de uma" não está em árvore nenhuma | **falso, e ao contrário** — ver §5, S1 |
| "a via de região é várias vezes o tamanho do modelo dedicado" — ⚠ **é uma frase v18**, de `src_fix/sections/07_discussion.tex` — ⛔ **árvore apagada a 2026-09-06** na fusão das três; recuperar com `git show 422c2d37`. **Não existe em `src/`** | **1,34× (CA) a 2,36× (Istanbul)** |
| "a lei de escala com o n.º de regiões" | **morta** — ver §5, C4 |
| Cosseno de gradiente "em sete datasets, positivo em todos" | **quatro datasets, equivalente a zero** |
| Silhueta 0,53 — ⚠ **não é v17**: é um comentário obsoleto do `src_fix/main.tex` (⛔ árvore apagada a 2026-09-06; `git show 422c2d37`). O `src/` imprime **0,57** na prosa da §VI-A e no comentário do `src/main.tex` | 0,53 não tem fonte nenhuma. 0,55 (AL/AZ/FL) ou 0,57 (cinco estados) — e **ambas são pré-v18** |

**Listas de nunca-citar anteriores, ainda válidas:** colapso v4 do STAN (AL 34,46 / AZ 38,96); linha
v2 do ReHDM (66,06 / 54,65 / 65,68); *outlier* HMT-GRN AL 62,37; o rótulo "STAN infeasible"
(retirado); todas as células fp16/bf16 marcadas VOID; **todo** o `docs/studies/closing_data/RESULTS_BOARD.md`
(é v17 e está morto).

---

## 5 · Ledger de alegações — o que sobrevive, o que morreu

Verbos ligados a testes (lei do `GLOSSARY §1`, inalterada):
**"outperforms"** só onde um teste de superioridade sobrevive a Holm dentro da sua família — ou seja
**exactamente três células**: região em TX e CA, categoria em FL. Em mais lado nenhum.
**"matches" / "statistically non-inferior within a two-point margin (TOST)"** só no **eixo da
região**, onde δ = 2 pp está pré-registado: Istanbul, AL, AZ, FL. **TOST é proibido no resumo.**
**Nunca promover o Arizona a ganho.**

### O terceiro estado: *unresolved*, e a lacuna de registo

`docs/studies/closing_data/v17_completion/STATISTICAL_PROTOCOL.md` §3.2 fixa a margem de
equivalência de 2 pp **só para o eixo da região**, e o §0 **proíbe explicitamente** reutilizar uma
margem entre eixos. **Nunca se registou um δ para a categoria.** Sob v18, cinco de seis células de
categoria falham a superioridade — sem uma regra, tornar-se-iam silenciosamente "matches" sobre uma
margem que nunca foi registada. Três saídas honestas, e só três:

| Rota | O que diz | Custo |
|---|---|---|
| **A. "Unresolved"** (é o que `src_fix` traz hoje) | "The other five category differences are not resolved in either direction by a superiority test. The equivalence margin is registered for the region axis, and we do not transfer it to category." | A mais segura e a mais fraca. Sub-declara AL e TX, que **são** resolúveis no sentido negativo |
| **B. Limite derivado** (é o que a dissertação traz) ← **recomendada** | Imprimir os seis intervalos e ler o limite deles: **0,334 pp** por dataset (extremo do IC de Alabama), **0,489 pp** simultâneo sobre os seis (Bonferroni), pior TOST p = 0,008 em Alabama. "Porque nada é escolhido, não há margem *post hoc* a justificar nem desvio a registar" | Três linhas. Exige imprimir os seis intervalos |
| **C. Registar δ_cat agora** | Margem *post hoc* com justificação própria, registada como desvio de protocolo | Reabre a errata de 2026-07-25 que o artigo acabou de fechar. **Não recomendada** |

⚠ **O limite derivado é por eixo.** Na região é **1,287 pp** por dataset / 1,372 simultâneo — cerca
de três vezes o da categoria. **"Within half a point" é FALSO no eixo da região.**
⚠ **O limite restringe magnitude, não sinal.** **Quatro** dos seis deltas de categoria são negativos.
"At least as good as the dedicated model on category everywhere" **não** é licenciado.

⚠ **Fuga entre eixos já presente no texto.** O resumo ("Across both tasks and all six datasets, no
difference from a dedicated model exceeds two points") e a conclusão aplicam a margem da região ao
eixo da categoria — exactamente o que a §6.2 recusa quarenta linhas antes. **Os dois têm de ser
reescritos.**

### Alegações, uma a uma

| # | Alegação | Estado sob v18 | O que fazer |
|---|---|---|---|
| **C1** | A representação por check-in melhora a categoria | **ENFRAQUECIDA** (sobrevive em direcção, colapsa uma ordem de grandeza) | "+0,23 a +6,29 pontos, com todos os cinco folds a favor em todos os seis datasets; significativa em todos excepto Florida (p = 0,07)" |
| **C1b** | *Mecanismo* do ganho da representação | **MORTA** | O controlo de concatenação recupera **111 % / 68 % / 490 %** do gap em AL/AZ/FL (Q13, 2026-08-16). Não reivindicar decomposição de causas. **Não adoptar** a inversão mais forte: o autor recusou-a em 2026-09-02 |
| **C1c** | Geometria (silhueta / pureza kNN) como explicação | **NÃO RESOLVIDA** | Constantes de um CSV de **2026-06-24**, no motor **v14 `design_k`**, e **agregadas por POI** embora a prosa diga "per-visit vectors". Não existe nenhuma medição de geometria v18. Re-medir (A2), delimitar por escrito, ou apagar — **e o resumo apoia-se nisto** |
| **C2** | O conjunto supera o dedicado em **categoria** | **MORTA como escrita; sobrevive só em Florida** | Ver §3.1 e a rota B acima |
| **C3** | O conjunto supera o dedicado em **região** | **ENFRAQUECIDA** (quatro vitórias → duas) | Nunca dizer a região sem a partição, nem as duas vitórias sem os quatro défices |
| **C3b** | Atribuição ao *trunk* partilhado | **MORTA** (retirada em 2026-07-28; não reabrir) | Todos os seis contrastes de trunk pareados por peso são não-resolvidos. Dizer que a atribuição fica em aberto |
| **C3c** | ⚠ **Confusão com capacidade** | **A LINHA MAIS PERIGOSA DO LEDGER** | Ver **D4** |
| **C4** | Lei de escala com o n.º de regiões | **MORTA como lei; sobrevive como agrupamento delimitado** | Sob joint-best a ordenação é −0,874 / −0,437 / −0,156 / **+1,206** / **+1,057**: a California tem mais regiões e ganho menor. Sob diag-best seria monótona — foi este o preço da convenção |
| **S1** | Custo / propriedade de modelo único | **MORTA como aritmética; sobrevive como propriedade operacional** | **Medido:** conjunto 4 197 621 (AL) e 5 151 189 (CA); os dois dedicados somados **1 850 980** e **2 804 548**. O conjunto é **2,27× / 1,84× MAIOR**. `src_fix` imprime 1,1 M e 2,0 M, que fecham contra uma contagem retirada. A poupança é operacional: um modelo para treinar, guardar e servir; uma passagem para a frente em vez de duas |
| **S2** | Piso de Markov na região | **ENFRAQUECIDA** (re-derivada) | "+4,1 a +10,0". **Importar a reconciliação da dissertação:** o piso stride-1 fica **acima** do HMT-GRN em 6/6, do STAN em 4 e do ReHDM em 3 — um revisor lê isso como baselines mal treinados. A dissertação explica (persistência de região) e o artigo não |
| **S3** | Domínio sobre as externas | **ENFRAQUECIDA** (as externas não mudaram; a nossa distância a elas colapsou) | ≥ +3,06 cat / ≥ +3,55 reg. **Não pode liderar** o resumo nem as contribuições |
| **S4** | Validade externa (Istanbul) | **MORTA como ganho; sinal invertido** | cat +0,08; reg **−0,08**, não-inferior. O padrão que se repete é a paridade, não o ganho |
| **S5** | Comparação com a cascata | **NÃO RESOLVIDA** | Medida no substrato v17 e **nunca re-corrida**. É a única medição com peso de alegação a atravessar a fronteira v17→v18 sem re-derivação. O autor apagou o parágrafo da dissertação em 2026-08-06 |
| **S6** | Controlo *freeze-region* | **ENFRAQUECIDA; apagada da dissertação** | No substrato entregue a via de região está desligada do codificador por construção (`Check2HGIModule.py:653`), logo "no transfer" é tautologia lida como achado |
| **S7** | Rastreio de *balancers* (§2) | **NÃO RESOLVIDA / auto-contraditória** | `02_related.tex` nunca foi re-auditado para v18 (mas **não** é o que os revisores leram — ver §7.10). Dois problemas: a nossa própria perda **é agora** peso igual (0,50/0,50), e a frase diz que dois balancers batem o peso igual; e os valores são pré-v18 (AL equal_weight 53,57) |
| **S8** | Cosseno de gradiente | **MORTA como escrita** | "quatro datasets, equivalente a zero, com médias por dataset a menos de dois milésimos de zero" |
| **S9** | Auto-posicionamento na §2 | **MORTA** | `02_related.tex:50` diz "on which sharing helps instead of hurting". Sob v18 a partilha não ajuda em 5/6 (cat) e 4/6 (reg). É o último sítio do artigo onde o mapa de vereditos antigo sobrevive em prosa |
| **S10** | Descrição do protocolo estatístico (§5.3) | **ENFRAQUECIDA e em parte não cumprida** | `05_setup.tex:43` ainda promete "We also report the registered test" e **não aparece nenhum Wilcoxon**. E a família Holm da região passou de m=4 (desvio D-4) para m=6 sem registo de desvio |
| **S11** | Cobertura da busca de hiperparâmetros | **MORTA (sobre-declaração)** | Correcto: *batch size* nos seis; *learning rate* em **quatro dos seis**. FL e CA levam o valor de dataset grande (0,005), testado só pelo rastreio de um fold do **Texas**. Importa porque o argumento de conservadorismo do artigo assenta em o braço dedicado ter tido a busca mais larga |
| **S12** | Convenção de época | **ENFRAQUECIDA; a versão do `src_fix` é materialmente incompleta** | Acrescentar o pior desvio por semente (0,23 / 0,93) **e** o facto de a convenção alternativa transformar **mais quatro células de categoria e mais duas de região** em melhorias que sobrevivem ao mesmo Holm. É isso que torna a escolha da convenção estrita principiada em vez de arbitrária — e o autor já a defendeu na arguição |
| **S13** | Grafo forward-only | **INVERTIDA; e a construção falta no Método** | A única mudança que produziu todo o colapso v18 é afirmada apenas dentro da quinta limitação da §7. A §4.1 descreve as arestas sem direcção. **A Figura 1 ainda desenha o grafo pré-v18** |
| **S14** | Integridade da representação (§5.2) | **NÃO RESOLVIDA** | Medida no substrato **pré-v18 (v14)** e nunca re-corrida. É a resposta do artigo à objecção de vazamento que afundou a submissão anterior: defende um build v18 com evidência pré-v18 |
| **S15** | Esboço de serviço (geografia da *shortlist*) | **NÃO RESOLVIDA / provavelmente não mensurável** | A dissertação retirou a cláusula: medir o erro geográfico exige predições por visita que o caminho de avaliação não retém |
| **S16** | "We propose two enhancements" | **SOBREVIVE com três palavras** | "…two enhancements **to that earlier model**" (B-23, aprovado pelo autor 2026-09-02) |

### O título honesto — quatro enquadramentos, ordenados

**#1 (recomendado) — paridade operacional, com a representação como a vitória medida.**
> Um modelo, uma passagem para a frente, duas respostas. Em seis datasets, toda a diferença face a
> dois modelos dedicados é pequena: dentro de meio ponto na categoria e dentro da margem
> pré-registada de dois pontos na região. A representação ao nível do check-in é o que torna a
> tarefa de categoria aprendível, melhorando-a em todos os datasets e em todos os folds.

É o único enquadramento em que **cada frase com peso é licenciada por um teste que o artigo
realmente correu**, e onde nenhuma experiência em aberto pode derrubar o título.

**#2 — a região onde a tarefa de região é mais difícil.** É a vitória de exactidão mais inequívoca e
a estatística mais forte do artigo (Holm p 1,3e-04 e 6,3e-06). Mas é a única alegação que uma
experiência **já existente e não publicada** mina (D4). Usável como resultado de apoio **com o
limite de capacidade dito no mesmo fôlego**; não usável como título enquanto o Q14 estiver por
reportar.

**#3 — representação primeiro.** Um revisor que compare os +28–40 submetidos com os +0,2–6,3 lê um
título construído sobre isto como o enquadramento mais fraco possível da maior retracção possível.

**#4 — custo / implantação.** Último: os números correm **ao contrário** (o conjunto é 1,84–2,27×
os dois dedicados somados) e colide directamente com a confusão de capacidade.

**Estrutura recomendada:** #1 como espinha, #2 como resultado de apoio *com* o limite de capacidade,
#3 como contribuição 1 com o mecanismo explicitamente não reivindicado, #4 como uma cláusula na
discussão.

---

## 6 · As decisões que só o autor toma

### D1 — O que reporta o camera-ready, e o que se diz aos *chairs*?

> ### ✅ DECIDIDO PELO AUTOR — 2026-09-06
>
> > *"eu não acho que seja necessario avisar nada a banca e ao chair, para mim vamos fazer a correção
> > dos texto e deixar na melhor forma possivel e re-enviamos"*
>
> **Rota (a): corrigir o texto, deixá-lo o melhor possível, reenviar. Sem nota aos chairs.**
> A preocupação foi levantada antes da decisão e o autor manteve-a depois de a ouvir; **está
> encerrada e não se reabre.** As opções abaixo ficam por proveniência.
>
> **O que isto transfere para o texto.** Sem nota de processo, **é o próprio artigo que tem de ser
> honesto sobre a mudança** — e isso é matéria de redacção, não de processo, portanto continua no
> nosso âmbito. Três coisas que o texto passa a ter de fazer sozinho:
> 1. dizer que a representação é construída com o grafo **forward-only** (hoje só aparece na §7 do
>    `src_fix`, e falta ao Método — item S13);
> 2. não deixar sobreviver nenhum número v17 (é o portão do item 3.62b);
> 3. declarar o limite de capacidade na região (D4), que é o que impede a leitura de que a vitória
>    de região é partilha.
>
> *(Opções consideradas, mantidas por registo:)*

| | O que significa | Consequência |
|---|---|---|
| **(a)** Publicar o v18 e não dizer nada | O camera-ready substitui em silêncio todos os números do título | **Indefensável.** Os revisores aceitaram outro artigo. A dissertação é pública e carrega os mesmos números v18 com a narrativa do vazamento; quando aparecer, lê-se como ocultação. E o `src_fix` **não contém uma única frase a reconhecer que algum número mudou** |
| **(b)** Publicar o v18 **com aviso aos chairs** | Nota escrita: o defeito, a magnitude, que alegações caem, quais sobrevivem | A opção defensável por omissão |
| **(c1)** Publicar o texto v17 aceite | | **Indisponível.** Sabe-se que os números estão inflacionados pelo vazamento |
| **(c2)** Retirar o artigo | Retirada formal antes do camera-ready | Custa a publicação, e mais nada |
| **(c3)** Avisar primeiro e deixar os chairs decidirem | Enviar a nota, propor (b), oferecer (c2) explicitamente, pedir que decidam | **Recomendada.** É a única rota em que a decisão sobre um conjunto de resultados materialmente diferente é tomada por quem é dono dela |

**O que se deve normalmente a um PC chair quando um resultado muda entre a aceitação e o
camera-ready:** avisar por escrito e depressa, **antes** do prazo do camera-ready e **antes** de
assinar o formulário de copyright (o formulário é uma declaração sobre o conteúdo); declarar o
defeito, a magnitude e o que invalida, sem retórica; **oferecer a retirada explicitamente** — é isso
que faz de um aviso um aviso e não uma negociação; deixar os chairs escolherem o remédio; uma só
mensagem ao Program Chair (**Rodolfo W. L. Coutinho**, Concordia) com cópia ao General Chair
(**Jun Zhang**, Southeast University) — fonte: `docs/MOBIWAC_CONFERENCE_GUIDE.md:193-194`.
O que estas políticas tratam como má conduta é a ocultação, não o erro.

Duas erratas **já dizem, no repositório**, que deviam acompanhar a próxima revisão enviada ao
MobiWac — a má atribuição do CBIC (2026-07-24) e os rótulos do protocolo estatístico (2026-07-25).
Nunca foram enviadas. **Nenhuma** das 17 correcções da `ERRATA.md` chegou aos revisores: o prazo de
submissão foi 2026-07-11 e a primeira correcção é de 2026-07-24.

> ⚠ **Escrever a nota só depois de D4 estar decidida** — D4 decide se alguma vitória de exactidão
> sobrevive.

### D2 — Qual árvore é a base? Ver §2. Recomendação: `src_fix/` + as correcções de Setembro.

### D3 — Orçamento de páginas: cortar para 8, ou pagar 9–10? **[bloqueia toda a prosa]**
Orçamento livre do MobiWac: **8 páginas**; tecto 10 com taxa por página. O PDF enviado tinha **8**.
`src/main.pdf` hoje: **9**. `src_fix/main.pdf`: **10**. A reescrita v18 acrescentou ~976 palavras de
corpo (6 359 → 7 335) num build que já estava acima. Cortar de 10 para 8 é largar ~2 200 palavras —
mais do que toda a adição v18, e as adições v18 são precisamente as ressalvas que tornam o texto
honesto. O `PLAN_8PAGES.md` nomeia o seu próprio critério de aborto: **pagar a taxa em vez de
enfraquecer o artigo** — e a sua lista de sobrevivência foi escrita para o conjunto de alegações
v17 e tem de ser re-derivada.
⚠ **A tarja de copyright do IEEE (`\IEEEpubid`) não existe em nenhuma árvore e vai consumir espaço
na página 1. Acrescentar ANTES de medir o corte.**

### D4 — O camera-ready diz que o ganho de região é capacidade, não partilha? **[decide se sobrevive alguma vitória]**

O controlo dedicado pareado em capacidade **foi corrido em 2026-08-13** — logo a frase de
`src_fix/sections/07_discussion.tex:105-109`, *"a capacity-matched dedicated region model … has not
been run"*, é hoje **falsa no texto entregue**.

Parâmetros e `% do conjunto` reproduzidos nesta sessão com
`research/reproducibility/mobiwac_v18/param_counts.py` (que **só** imprime contagens); Acc@10, delta
e p vêm de `docs/results/P1/region_head_{california,texas}_…_capmatched*_s0.json` contra
`docs/results/closing_data/v18/joint_best_perfold.json`:

| dataset | largura | parâmetros | % do conjunto | Acc@10 | vs. conjunto | p |
|---|---:|---:|---:|---:|---:|---:|
| California | 352 (**a única paridade real**) | 5 014 942 | **97,4 %** | 64,910 | **+0,406** | 0,010 (5/5) |
| California | 528 | 9 004 686 | 174,8 % | 64,931 | +0,428 | 0,008 (5/5) |
| Texas | 544 | 8 354 882 | 170,5 % | 66,330 | +0,214 | 0,116 (4/5) |

> ⚠ **Protocolo: semente 0, 5 folds (n = 5) — NÃO é a convenção do §3.** O comparador é a célula
> conjunta de **semente 0** (`joint_best_perfold.json`: `california_s0_joint` = **64,5034**,
> `texas_s0_joint` = **66,1168**), não os 64,54 / 66,15 de quatro sementes do §3.2 — quem subtrair
> contra o §3.2 obtém +0,37, não +0,406. O teste é um *t* pareado **bilateral** sobre os cinco folds,
> **sem Holm** (`errata_Q14_capacity_region.tex`). Os três braços são de uma só semente. Isto não
> anula o achado — anula a possibilidade de o citar como se estivesse na escada do §3.

**A curva satura em 352.** Em California, um modelo dedicado com 97,4 % do orçamento do conjunto
**está acima** do conjunto. O Texas **não tem braço em paridade**: o seu único braço leva 1,7× o
orçamento e o seu +0,214 não se separa de zero.

Opções: (1) declarar, delimitado à California — a errata já redigida está em
`articles/dissertacao/wrapup/erratas/errata_Q14_capacity_region.tex`; consequência: **o artigo fica
sem nenhuma vitória de exactidão**, e a alegação reduz-se a paridade operacional mais a margem
externa. (2) Correr primeiro o braço de paridade do Texas (~2 h/semente) e declarar o que os dois
suportarem. (3) Apagar a limitação e não dizer nada — **indisponível**: a frase no texto é hoje uma
afirmação que o autor sabe ser falsa.

### D5 — Eixo da categoria: registar margem, dizer "unresolved", ou o limite derivado? Ver §5. Recomendação: **B**.

### D6 — Existe um piso de significância prática? O gerador atribui "beats" a deltas de **+0,04 pp**,
porque emparelhar nos mesmos folds colapsa a variância. `GAPS.md §7` regista isto como
**"Not yet decided"**. Só o resultado de região em TX/CA sobrevive a qualquer piso razoável.

### D7 — Quanto se declara da convenção de época? Ver §5, S12. Custo: uma frase.

### D8 — Quais das 17 erratas entram no camera-ready e quais entram na nota aos chairs?
Cinco *commits* adicionais tocaram `src/` e **não estão no ledger** (`84e4a703`, `54c8e0b4`,
`fd94d0a2`, `e8bd4575`, `f8050d3c`) — um deles acrescentou uma alegação viva na §2 sem entrada.

### D9 — Quem corrige os ficheiros de lei, e quando? Ver §7. **No mesmo commit.**

### D10 — A nota de rodapé de disponibilidade do código.
A página 1 promete `github.com/VitorHugoOli/PoiMtlNet/tree/mobiwac`. Verificado vivo em 2026-09-04:
público, o *branch* resolve, HEAD `f9c50218` (2026-08-07). Mas o *branch* carrega **apenas
artefactos `mobiwac_v17`**, e o seu README ainda diz *"Anonymous Code Release … will be made public
after acceptance"* — errado para um venue single-blind e obsoleto depois da aceitação. A §5.3
afirma ainda *"The code release includes both tests and documents this departure from the plan"* —
verdadeiro para v17. Se o camera-ready reportar v18, a nota é falsa outra vez, de uma maneira nova.

### D11–D20 — decisões de menor bloqueio

| | Decisão |
|---|---|
| D11 | Figura 3 (geometria): re-medir em v18, delimitar ao motor arquivado, ou largar. Está cortada do build desde 2026-07-09, **mas o resumo apoia-se nela** |
| D12 | Ablação do *trunk* em TX/CA à força do board (~22 h) — a única experiência que licenciaria uma frase de atribuição |
| D13 | Escala da silhueta: 0,55 (AL/AZ/FL) vs. 0,57 (cinco estados). Hoje três sítios discordam: prosa 0,55, comentário do `main.tex` 0,53, ficheiro commitado 0,57 |
| D14 | Convenções das tabelas: ordem das linhas (n.º de regiões vs. n.º de check-ins); convenção de ênfase B-27 (negrito/sublinhado = magnitude, ↑/≈ = veredito); repor a coluna ± da Tabela 2 |
| D15 | Formulação de estado. Regra do autor 2026-09-02: *"pode tratar a versão atual como a submetida; o conteúdo da atual não vai mudar numa versão revisada para o MobiWac"* |
| D16 | Números de integridade da §5.2: re-medir em v18 ou levar com uma frase de escopo assinada |
| D17 | Parágrafo da cascata + controlo *freeze-region*: manter (como em `src_fix`) ou apagar (o autor apagou ambos da dissertação em 2026-08-06). Apagar custa também a referência antecipada da §2 |
| D18 | Geografia da *shortlist* ("3 a 8 km contra 17 a 176 km") — retirada na dissertação |
| D19 | Reconciliação piso-Markov vs. externas: importar os dois parágrafos da dissertação (~12 linhas)? |
| D20 | Agradecimentos/financiamento e bloco de copyright IEEE — **não existe nenhum em nenhuma árvore**. A lista de três autores **já existe** (`src_fix/main.tex:54`) e coincide com o registo EDAS |

---

## 7 · Armadilhas — o que vai reinjectar o v17 se ninguém o impedir

1. **⚠ OS FICHEIROS DE LEI AINDA CODIFICAM A ESCADA v17.** `GLOSSARY.md §1` e §6, `PAPER_PLAN.md §3`,
   o cabeçalho de guardrails da `ERRATA.md` e `[mobiwac]/CLAUDE.md` §2/§3 nomeiam todos
   {Istanbul, Florida, Texas, California} como vitórias de região e citam células de categoria na
   casa dos 63–80. A regra de precedência declarada é que **a lei ganha sobre um plano**. Este é o
   caminho mais provável pelo qual um número com vazamento entra num artigo publicado.
   **Corrigir no mesmo commit da primeira edição de prosa.** A cláusula que sobrevive intacta:
   **nunca promover o Arizona**.
2. **O `src_fix/main.pdf` commitado É o PDF v17.** No HEAD do git é byte a byte igual ao
   `src/main.pdf` (9 páginas). O build v18 de 10 páginas existe **apenas** como modificação não
   commitada na *working tree* — e mesmo esse está dois commits atrasado (imprime silhueta 0,53 onde a
   fonte diz 0,55). Um *checkout* limpo + um upload apressado produz o PDF aceite-mas-invalidado.
   **Reconstruir e commitar antes de qualquer coisa.**
3. **`RESULTS_BOARD.md` chama-se a si próprio fonte única de verdade e está morto.** É v17.
4. **`V18_RESULTS.md §1` tem uma escada de vereditos própria que não é a do artigo**: é
   *diag-best*, com p pooled ao nível do fold e **sem Holm**, e atribui "beats" a Δ de +0,04.
   As colunas *joint-best* aparecem como `—` porque o `score_all.py` lê o nível errado do JSON.
   **`joint_best_perfold.json` é a única fonte de joint-best.**
5. **A Figura 1 contradiz a premissa do próprio artigo.** As etiquetas TikZ ainda dizem
   *"edges: consecutive visits by a user"* e *"features: category, hour, weekday"* — o grafo
   pré-v18. A §7 afirma que o grafo é forward-only; a Figura 1 mostra que não é; a §4 não diz nada.
6. **A Tabela 3 não sobrevive à subtracção.** O arredondamento independente por coluna faz a tabela
   discordar da prosa e da Fig. 4 em cinco células (AL cat −0,18 vs −0,19; FL cat +0,20 vs +0,19;
   TX cat −0,14 vs −0,13; AL reg −0,88 vs −0,87; FL reg −0,15 vs −0,16). A deltas de um quinto de
   ponto, subtrair as colunas é a **primeira** coisa que um revisor faz.
7. **`review/gate_revision_plan_2026-08-11.md` não está no git.** É o único artefacto de revisão
   escrito contra o v18, com nove bloqueadores e oito perguntas ao autor. **Commitar.**
8. **Os três JSONs do controlo de capacidade estão fora do git.** O `docs/results/P1/` **está**
   versionado (536 ficheiros rastreados); o padrão `results` em `.git/info/exclude:9` só esconde
   ficheiros **novos**, nunca um já rastreado. O que nunca foi adicionado são os três artefactos do
   controlo (`…california_reg_capmatched_s0.json`, `…_capmatched528_s0.json`,
   `…texas_reg_capmatched544_s0.json`): **`git add -f` nos três.** A prova narrativa está rastreada
   (`P1_capacity_region.md`, `fig_P1_capacity.png`, `errata_Q14_capacity_region.tex`); falta o
   artefacto numérico.
9. **Não existe protocolo estatístico v18 nem entrada de desvio v18.** O `log.md` pára em D-1…D-4
   (2026-07-25). O plano v17 continua a ser o plano de registo para uma escada calculada em Agosto,
   **depois** de a escada v17 ter sido lida por inteiro. A família Holm da região mudou de m=4 para
   m=6 sem entrada.
10. **Nenhum dos dois é byte a byte igual ao submetido, e só um deles nunca foi re-auditado.**
    `02_related.tex`: idêntico entre `src/` e `src_fix/` **hoje** (md5 `aa6259c3…`), sem uma única
    alteração desde `da97ecf7`, o commit que ramificou o `src_fix` — **nunca re-auditado para v18**.
    É onde vivem **S7** e **S9**. Mas seis commits tocaram `src/sections/02_related.tex` **depois**
    do último build de 8 páginas (`ef3b2e55`, `d1911c0a`, `232befd5`, `fecc7fb1`, `84e4a703`,
    `fd94d0a2`), e foi um deles que acrescentou a frase *"seven datasets … positive at every one"*
    que a §8 manda corrigir: **essa frase nunca esteve no PDF que os revisores viram** (conta para
    a D8).
    `05_setup.tex`: **difere** de `src/` (md5 `ad7b3a11…` vs `b6d75b98…`) e **foi** re-auditado para
    v18 em 2026-08-11 (`826aeaee`, `0a27b356`, `0ebbbceb`) — "the four next-region gains" → "the
    two", o Holm da região de "four" → **"six next-region comparisons"** (linha 45), o
    desvio-padrão pareado 0,01–0,18 → 0,02–0,16. **S10 e S14 continuam por resolver lá, mas o que
    falta não é a re-auditoria.**

---

## 8 · Plano de edição, ficheiro a ficheiro

Caminhos relativos a `articles/[mobiwac]/src_fix/`.
Legenda: **[C]** correcção que tem de entrar em qualquer decisão · **[B]** trazer da dissertação ·
**[P]** apresentação · **[D#]** dependente de uma decisão.

### Fase 0 — antes de tocar em prosa
1. **Reconstruir e commitar `main.pdf`** (`pdflatex` → `bibtex` → `pdflatex ×2`), **depois** medir páginas.
2. **Corrigir os ficheiros de lei** (§7.1), no mesmo commit da primeira edição.
3. **Responder a D1 e D4.** O resumo e a §7 dependem das duas.
4. **Recuperar os pareceres dos revisores do EDAS** (ver §10).

### Fase 1 — correcções que têm de entrar sob qualquer decisão

| Ficheiro | Edição | |
|---|---|---|
| `figs/fig1_dataflow.tex:64` | Etiquetas ainda desenham o grafo pré-correcção. Passar a `… (forward in time)` e `…, elapsed time`, como em `articles/dissertacao/src/figures/mobiwac/fig1_dataflow.tex:69` | **[C]** |
| `sections/04_method.tex:22` | Acrescentar a frase do forward-only. A única mudança que produziu todo o colapso v18 **não está no Método** | **[C]** |
| `sections/07_discussion.tex:105-109` | *"has not been run"* é **falso**. E *"several times the size"* é 1,34×–2,36× | **[C] [D4]** |
| `sections/04_method.tex:55-57` | Somas de parâmetros erradas: 1,1 M → **1,9 M**; 2,0 M → **2,8 M** | **[B]** |
| `sections/02_related.tex` | "seven datasets … positive at every one" → **quatro datasets, equivalente a zero** (`apx_f_cosine.tex` (o bloco do cosseno de gradientes)``: as quatro médias caem dentro de dois milésimos de zero, contra uma margem de 0,05). ⚠ **Não colar a frase nova ao lado da antiga:** as linhas 99-106 já dizem "four Gowalla states … $+0.001$ … $+0.0032$". Fundir as duas ou apagar a última — senão o parágrafo diz "quatro datasets" duas vezes | **[B]** |
| `sections/02_related.tex:50` | *"on which sharing helps instead of hurting"* — falsificado. `GLOSSARY §9.3` permite exactamente uma frase de auto-posicionamento na §2, logo **reescrever, não apagar** | **[C]** |
| `sections/02_related.tex:92-98` | A frase dos *balancers* diz que dois batem o peso igual — e o peso igual é agora **a nossa própria perda** | **[C]** |
| `sections/06_results.tex:61`, `tables/tbl3_results.tex:49`, `sections/07_discussion.tex:32-33,99-101` | Cobertura da busca: *batch size* nos seis, *learning rate* em **quatro** | **[B]** |
| `sections/05_setup.tex` | *"We also report the registered test"* — nenhum Wilcoxon aparece. Regra do autor 2026-09-02: retirar a promessa, manter o registo + a razão do desvio + a declaração da release. ⚠ A cadeia "primary analysis" **não existe** no `src_fix` — nada a renomear; se se importar o parágrafo da dissertação (`5_mobiwac/05_setup.tex`), ele já traz "The reported analysis" | **[B]** |
| `sections/08_conclusion.tex:11` | 3,5 / 3,0 → **3,55 / 3,06**. E "on either task" transfere a margem da região para a categoria | **[B] [D5]** |
| `main.tex:85-87` (resumo) | O mesmo problema de margem entre eixos | **[D5]** |
| `sections/07_discussion.tex:90-92`, `sections/05_setup.tex:30` | *"A planned follow-up"* — foi corrido | **[B]** |
| `main.tex:131-134` | Comentário da Fig. 3 diz 0,53; a prosa diz 0,55 | **[C]** |

### Fase 2 — importações que reforçam o artigo (todas já com decisão do autor na dissertação)

| Ficheiro | Edição |
|---|---|
| `sections/06_results.tex` | Imprimir os **seis** IC de categoria, não só o de Florida (rota B do §5) |
| `sections/06_results.tex:99-105` | Assimetria de declaração de défice: a §6.2 declara-a para a região e não para as duas células de categoria cujos IC também excluem zero (AL, TX) |
| `sections/06_results.tex:66-79` | Convenção de época → pior semente 0,23/0,93 **e as inversões de veredito** (D7) |
| `sections/06_results.tex:21` | Declarar que a coluna de check-in da Tab. 2 é a fatia da semente 0 do braço dedicado da Tab. 3 |
| `sections/06_results.tex:25-37` | Controlo de concatenação: imprimir os três ganhos ao lado dos três gaps. **Não adoptar** a inversão mais forte (o autor recusou-a: o Q13 não reproduz a Tab. 2 fold a fold em AZ/FL) |
| `sections/06_results.tex` | Importar a reconciliação piso-Markov vs. externas (D19) |
| `sections/05_setup.tex` — fim do parágrafo *"A claimed gain and a claimed match require different tests…"*. ⚠ **A linha 125 é a da dissertação**; no `src_fix` a 125 é `\subsection{Baselines}` | "On next category the plan registered no equivalence margin, so a difference that fails the superiority test is reported as unresolved rather than as a match" |
| `sections/04_method.tex:43-45` | A razão da assimetria da métrica para o *logit adjustment* só na categoria |
| `sections/01_introduction.tex:19` | **B-23:** "…two enhancements **to that earlier model**" |
| `tables/tbl1_datasets.tex` | Cláusula na legenda: check-ins/utilizadores/POIs são do corpus cru, Windows é pós-filtro. **Nenhum número muda** |
| `tables/tbl2_substrate.tex` | Repor a coluna ± (AL 1,16/0,84 · AZ 1,15/1,08 · Ist 0,89/0,63 · FL 0,42/0,41 · CA 0,47/0,41 · TX 0,51/0,48). O gap de FL (+0,23) cabe dentro do seu desvio de 0,42 — o leitor deve ver isso |
| `sections/06_results.tex:25` | CTLE: **`src_fix` está À FRENTE aqui** (já largou a magnitude). A dissertação ainda diz "about two points". **Não importar** |

### Fase 3 — apresentação (dependente de D14)
Ordem das linhas; convenção de ênfase B-27; a discordância entre colunas arredondadas e prosa
(§7.6 — ou reimprimir com mais precisão, ou acrescentar uma coluna Δ explícita); a legenda da
Tab. 3 não diz que as duas ↑ de região são **resultados secundários fora do plano**, embora a §5.3
o diga; `figs/fig4_deltas.py:81-82` tem comentários de cor obsoletos.

### Fase 4 — build e submissão
Acrescentar `\IEEEpubid` e o bloco de agradecimentos → **depois** medir páginas (D3) → reconstruir
(0 Overfull, 0 refs indefinidas, 32 `\bibitem`) → actualizar o resumo e a ordem das *keywords* no
EDAS → republicar o *branch* `mobiwac` com artefactos v18 e um README desanonimizado (D10) →
formulário de copyright IEEE / validação de PDF.

### Já correcto — não tocar
`sections/03_problem.tex` · `figs/fig2_model.tex` (o v18 **não** é mudança de arquitectura) ·
números da `tables/tbl1_datasets.tex` · `references.bib` · todas as células externas da Tabela 3 ·
os dados de `figs/fig4_deltas.py` (os seis pares verificados a 4 casas contra os artefactos).

---

## 9 · Backlog de medição

**Nada aqui restaura o título aceite.** O colapso da categoria é real.

### Tier A — mudaria o que o camera-ready pode dizer

| # | Medição | Decide | Custo |
|---|---|---|---|
| **A1** | **Dedicado de região pareado em capacidade no Texas** (~97 % do orçamento, 5 folds) | Se o ganho de região em TX é partilha ou capacidade. Hoje só CA tem braço em paridade — e **inverte** a alegação lá | ~2 h/semente. Comando em `docs/studies/closing_data/v18/POSTPONED.md:17-21` — mas é o braço da **California** (`d_model=352`); a largura de paridade do Texas (~97 % de 4 899 897) ainda tem de ser derivada, e o `P1_capacity_region.md` não traz comando nenhum. **O de maior valor da lista** |
| **A2** | **Geometria de embedding re-medida em `check2hgi_v18` vs `hgi_dk_ovl`** | A única alegação de mecanismo que sobrevive à contribuição 1 — e o resumo apoia-se nela | Só leitura + métrica, **sem treino**. Barato. Decidir primeiro a granularidade |
| **A3** | **Auditoria de integridade / transdutividade (§5.2) no substrato forward-only** | A resposta do artigo à objecção de vazamento | Moderado. Alternativa: uma frase de escopo assinada |
| **A4** | **Ablação do trunk em TX/CA à força do board** | Licenciaria ou proibiria uma frase de atribuição | **~22 h** |

### Tier B — fecha uma lacuna declarada ou uma assimetria de protocolo
**B1 · Piso Markov-K de *categoria* em stride-1** — o bloco de categoria mistura protocolos:
POI-RGNN e o piso Markov-K correm janelas **stride-9 não sobrepostas** (~11 % das nossas linhas),
enquanto as nossas células correm stride-1. Inofensivo a +46 pp, **exposto a +3,06**. Custo:
segundos. **Fazer este.**
B2 · POI-RGNN em janelas stride-1 · B3 · CTLE emparelhado com a receita v18 · **B4 · Tabela 2 de
n=5 para n=20** (a contribuição 1 é uma semente, e Florida não é significativa) · B5 · diagnosticar
a fidelidade do Q13 · B6 · *logit adjustment* τ=0,5 na California (nunca medido; CA carrega metade
do resultado de região do título) · B7 · segunda semente para o resultado de peso de perda em FL.

### Já medido — não repetir
Wilcoxon sobre a escada v18 (`research/reproducibility/mobiwac_v18/wilcoxon_v18.py`, 2026-09-01;
concorda com o *t* nas doze células a α=0,05; fino só numa: categoria Istanbul Holm 0,0599 vs 0,181) ·
contagens de parâmetros (`param_counts.py`, 2026-09-02) · controlo de concatenação à escala da
Tab. 2 (Q13) · piso Markov-1 stride-1 · piso de classe maioritária por dataset (**o máximo é o
Alabama, 7,28 — nunca impresso no material entregue, mas já tabelado em
`docs/baselines/next_category/comparison.md:34` e `docs/baselines/README.md:231`**) · as 24 células
conjuntas v18.

---

## 10 · O que falta no repositório

### 10.1 ⚠ OS PARECERES DOS REVISORES DO MOBIWAC NÃO EXISTEM. EM LADO NENHUM.

Uma varredura de toda a árvore por `Reviewer 1`, `R1:`, `meta-review`, `overall evaluation`,
`detailed comments to the author`, `parecer` e cadeias do EDAS devolve **apenas** três tipos de
resultado: *personas* simuladas nossas (`review/panel_2026-07-10.md`, `review/panel_2026-07-18/`,
`archive/REVIEW_PANEL.md`, `review/gate_revision_plan_2026-08-11.md`), revisões feitas por outros
modelos (`review/fable.md`, `codex.md`, `agy.md` — "externo" ali significa externo aos nossos
próprios agentes), e as preocupações dos revisores da submissão anterior (BRACIS), transcritas em
segunda mão no `PAPER_PLAN.md §9`.

O único vestígio da decisão do MobiWac é **uma data**: "accepted 2026-08-26", registada só na árvore
da dissertação. A pasta `[mobiwac]` parou em 2026-08-12 e os seus documentos de estado ainda dizem
"submitted, under review".

**Consequência: tudo o que está catalogado neste ficheiro é auto-iniciado.** Se os revisores pediram
alguma coisa, está por fazer nas três árvores. **O autor tem de ir buscar a notificação e os
pareceres ao EDAS ou ao email antes de o camera-ready ser escrito** — para os satisfazer, e porque
se algum revisor tocou no canal de vazamento — que a §7 aceite **não** declarava — isso muda
materialmente a nota de D1.

### 10.2 Também em falta
- **O prazo do camera-ready, o número de páginas permitido e a tabela de taxas.** O dossier do venue
  (congelado em 2026-06-19) diz "TBA".
- **Todo o fluxo de camera-ready IEEE**: formulário de copyright, validação de PDF, `\IEEEpubid`,
  bloco ISBN/DOI. Nenhuma das três árvores tem bloco de copyright.
- **Que PDF foi de facto enviado ao EDAS, e quando.** O `EDAS_SUBMISSION.md` (2026-07-10) diz que o
  Passo 3 estava pendente; o histórico do git mostra `src/main.pdf` com **8 páginas** de 2026-07-09
  a 2026-07-20, com edições justificadas contra o orçamento de 8 páginas a aterrar em 2026-07-13 e
  2026-07-20 — ou seja, **depois** do prazo de 2026-07-11 registado. Ou o prazo mudou (não
  registado), ou essas edições não estão no PDF revisto.
- **Se o resumo registado no EDAS chegou a ser substituído.** O registado é o rascunho pré-v17; o
  bloco de substituição que o `EDAS_SUBMISSION.md` fornece é v17 ("+28 to +40 macro-F1"), que
  contradiz o resumo v18 da forma mais forte possível.
- Bloco de agradecimentos / financiamento (CAPES / FAPEMIG / CNPq).
- **Material de reprodutibilidade v18 no branch público** — `origin/mobiwac` carrega só `mobiwac_v17`.
- **O lado do artigo do ledger de divergência.** O único que existe é
  `articles/dissertacao/src/tables/mobiwac/errata_scope.tex` (sete linhas, actualizado 2026-09-02,
  já nomeado no §2) — e é unilateral: vive na árvore da dissertação e regista o que o **capítulo**
  mudou face ao artigo. A pasta `[mobiwac]/` não tem equivalente.

---

## 11 · Riscos em aberto

- **Ética de publicação (o maior).** Publicar os números v17 não está disponível; substituí-los sem
  avisar entrega um artigo materialmente diferente sob uma aceitação dada a outra coisa. Só a
  terceira rota sobrevive a escrutínio. **O calendário importa: antes do prazo, antes das taxas de
  página, antes do formulário de copyright.**
- **O resultado pode não sobreviver a D4.** Se o Texas se comportar como a California, o artigo fica
  **sem vantagem de exactidão** sobre modelos dedicados em qualquer dos eixos, e a alegação reduz-se
  a paridade operacional mais a margem externa. É um resultado real, mas um artigo muito mais
  pequeno — e a nota aos chairs não deve ser escrita antes de isto estar decidido.
- **Os ficheiros de lei vão reinjectar alegações v17** (§7.1).
- **O PDF errado é enviado** (§7.2).
- **O orçamento de páginas decidido tarde** — duas páginas de corte cairiam exactamente sobre o
  material que torna o texto v18 defensável, e o `GLOSSARY §9.4` proíbe cortar a auditoria de
  vazamento da §5.2 abaixo do seu piso de evidência.
- **A §5.2 defende um build v18 com evidência pré-v18.**
- **Exposição estatística.** As duas superioridades de região que sobrevivem não estão em nenhuma
  família de correcção registada — a §5.3 chama-lhes "secondary results outside the plan" e a legenda
  da Tab. 3 não o repete. E o comparador dedicado foi afinado na mesma CV, o que faz de Δcat uma
  estimativa conservadora de uma vantagem e **optimista de um défice** — direcção que importa agora
  que três células de região e duas de categoria são défices.
- **Lacunas de proveniência.** 30 de 72 células têm `commit_sha` = `"unknown"` (sementes 7 e 100,
  fechadas como irrecuperáveis). O hardware é misto e não declarado. A equivalência de leitura é
  `n/a` exactamente no Texas e na California.
- **As duas árvores estão a divergir.** Sem ledger, o registo depositado e o registo publicado vão
  contradizer-se na margem do CTLE, na cobertura da busca, nas contagens de parâmetros e na
  limitação de capacidade.

---

## 12 · LOG

> Uma entrada por alteração com efeito a jusante. Data absoluta, o que mudou, a prova, e o que
> passa a ser falso noutro sítio. Entradas mais recentes em cima.

### 2026-09-04 — este ficheiro criado; recontextualização completa
Varredura de 17 agentes sobre `[mobiwac]/` (todos os `.md` e `.tex`), `docs/studies/closing_data/v18/`,
`docs/results/closing_data/v18*/` e o Capítulo 5 da dissertação, seguida de três redutores
(números / alegações / plano de trabalho). Verificado por mim, de forma independente e directa:

- a escada entregue reproduz célula a célula a partir de `ladder_recompute.json`;
- `param_counts.py` corre e confirma que o conjunto é **2,27× (AL) e 1,84× (CA)** os dois dedicados
  somados, e que **só `california/352` é um controlo em paridade** (97,4 %);
- `src/main.pdf` = **9 páginas**, `src_fix/main.pdf` = **10 páginas** (`pdfinfo`);
- o `src/main.pdf` teve 8 páginas de `113cf8ce` (2026-07-09) a `f66f8a73` (2026-07-20) e passou a 9
  em `e36b3194` (2026-07-30);
- **não existe nenhum parecer de revisor do MobiWac em todo o repositório.**

Nada foi alterado em `src/`, `src_fix/` ou na dissertação nesta sessão.

**Bloqueadores que aguardam o autor:** D1 (o que se diz aos chairs), D4 (o controlo de capacidade),
D3 (orçamento de páginas), e a recuperação dos pareceres do EDAS.

**Ronda de falsificação (mesma data).** Sete verificadores tentaram derrubar este ficheiro bloco a
bloco, mais um crítico de completude. Resultado: **nenhum número do §3 caiu** — as 24 células da
escada, os 24 extremos de IC, os 12 p de Holm, os 6 TOST, as seis linhas da Tabela 2, as margens
externas, os pisos e os limites derivados foram todos recalculados de raiz a partir dos arrays por
fold e reproduzem. Caíram **cinco afirmações de processo**, todas corrigidas acima:

1. **A mais grave, e estava invertida.** Eu tinha escrito que o manuscrito aceite já declarava o
   canal de vazamento. **Não declarava.** O PDF submetido abre a §7 com *"Two limits"*; a quarta
   limitação entrou em `0b472205`, 2026-08-05. Isto reforça a obrigação de avisar os chairs, não a
   enfraquece.
2. **"a mesma receita v17" era falso.** O `CHARTER_COMPLIANCE.md §2.1` diz que da receita do charter
   *"nada disso foi usado"*: um sweep de 103 braços re-afinou os dois braços. Acrescentado ao §1.
3. **"região não tocada, ≤ 0,33 pp"** confundia o deslocamento do *delta* com o movimento absoluto;
   o braço conjunto mexeu até 1,15 pp.
4. **`docs/results/P1/` não está escondido do git** (536 ficheiros rastreados); o que falta são três
   JSONs nunca adicionados.
5. **`02_related.tex` / `05_setup.tex`**: a afirmação estava errada nas duas metades e em sentidos
   opostos. E `sections/05_setup.tex:125` era a linha da **dissertação** — seguir a instrução teria
   colado uma frase de margem de equivalência dentro de `\subsection{Baselines}`.

Mais quatro correcções de proveniência na lista de nunca-citar: "43–65", "várias vezes o tamanho",
"64–72 % / 89–90 %" e "silhueta 0,53" não são valores v17 e procurar por eles deixaria as frases
vivas em paz.
