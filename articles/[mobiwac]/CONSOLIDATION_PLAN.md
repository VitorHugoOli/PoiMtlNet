# PLANO DE CONSOLIDAÇÃO — `articles/[mobiwac]/`

> **Companheiro obrigatório:** [`CAMERA_READY.md`](CAMERA_READY.md) — os números, os vereditos e as
> decisões D1–D20. Este ficheiro é o **como**; aquele é o **quê**. Onde discordarem sobre um número,
> o `CAMERA_READY §3` ganha.
>
> **Papéis (autor, 2026-09-06):** a sessão `mobiwac-writer` redige e tem veto de qualidade; esta
> sessão (`mobiwac`) filtra, audita e propõe.
>
> ---
>
> ## ✅ DECISÕES DO AUTOR — 2026-09-06 (em primeira pessoa)
>
> | | Decisão | Consequência |
> |---|---|---|
> | **D4** | **Declarar que o ganho de região parece relacionado com capacidade.** | O artigo fica **sem vitória de exactidão** no eixo da região como propriedade de partilha. O título passa a ser paridade operacional + a margem externa (`CAMERA_READY §5`, enquadramento #1). Trava os bullets 2 e 3 da introdução e a contagem de limites da §7. |
> | **D3** | **8 páginas**, como a primeira versão enviada. | **É o maior item de trabalho do plano.** Ver o aviso a seguir. |
> | **DP-1** | **A `WRITING_LAW` governa o camera-ready.** ✅ **FECHADO em primeira pessoa** — a cláusula de excepção do `WRITING_LAW.md:8` foi removida pelo autor, commit `964cc50c`. | Resolve também a **D5** por consequência: a superfície da categoria é *"equivalent to zero within half a point"*. `matches` sai dos dois eixos. O texto do Cap. 5 já está assim escrito. |
>
> ### ⚠ O que a D3 custa, medido
>
> | | palavras (abstract + corpo) | páginas |
> |---|---:|---:|
> | Build submetido (`97f01a50` / `f66f8a73`) | **5 732 / 5 680** | 8 |
> | `src_fix` hoje | **7 341** | 10 |
> | Depois do porte recomendado | **≈ 7 680** | 10 |
>
> **Há que largar ≈ 1 980 palavras — 26 % do artigo.** Isto **não** é a aparagem que a §4 descrevia
> (os candidatos que ela lista somam ~550). É uma decisão estrutural, e é o segundo item que ainda
> precisa do autor: **o quê sacrificar.** As ~1 630 palavras que o v18 acrescentou são precisamente
> as ressalvas, os intervalos e as declarações que tornam o texto defensável — e a D4 **acrescenta**
> mais (79 a 284 palavras, consoante a forma).
>
> ### Estado da execução
>
> - **Bloco 0 (Preservação): FEITO** — ramo `mobiwac-camera-ready`, commit `422c2d37`, três tags.
>   O `src_fix/main.pdf` foi **reconstruído** antes de commitar (o da working tree estava dois
>   commits atrasado e imprimia silhueta 0,53): 10 páginas, 0 refs indefinidas, 0 overfull, 0,53
>   ausente do texto extraído.
> - **Item 3.0:** resolvido pela via honesta — em vez de afirmar qual build foi submetido, **os dois
>   candidatos ficaram etiquetados com nomes neutros** (`mobiwac/build-8pp-prazo`,
>   `mobiwac/build-8pp-jul20`). A identificação continua por fazer e só se resolve no EDAS.
> - Tudo a partir do Bloco A continua **por executar**.

**Estado:** proposta para aprovação do autor. Nada aqui foi executado.
**Base factual:** doze auditorias (oito secções, proveniência das árvores + assets, triagem da pasta, herança das leis), verificadas contra os ficheiros. Verificações próprias desta sessão assinaladas com ✔.
**Árvore de referência:** working tree em `HEAD = e8f5a1b8` (2026-09-06). ✔ Confirmado que desde a linha de base das auditorias só três commits tocaram material MobiWac (`0b0bcb8c` comentários em `07_discussion.tex`, `0c4f6f2f` citações do logit em `04_method.tex`, `30621bce` a oitava linha da `errata_scope.tex`) — todos já cobertos pelas auditorias. **Não há deriva por reconciliar.**
**Aviso de execução:** os números de linha citados são de hoje. O escritor deve casar por **texto da frase**, nunca por número de linha — qualquer edição desloca-os.

---

## 1. O QUE EXISTE HOJE

### 1.1 As três árvores (quatro, na verdade)

| Árvore | Conteúdo | Geração | PDF |
|---|---|---|---|
| `articles/[mobiwac]/src/` | Texto submetido **mais** ~19 commits pós-prazo (correcções de Jul-Ago). Não é byte-idêntico ao que os revisores leram. | v17 | committed 9 pp (`md5 6a32e1e30b7b285e19e4b857cf49c0ae`) |
| `articles/[mobiwac]/src_fix/` | A reescrita v18 de 2026-08-11/12, depois parada. 26 commits. | v18 | **committed = 9 pp e byte-idêntico ao de `src/` (é o build v17)**; o build v18 de 10 pp existe **só na working tree**, por commitar |
| `articles/[mobiwac]/src_v1/` | Cópia congelada de `src/` no commit `57235720` (2026-07-09) + um `VERSION.md` de 5 linhas. | v17 pré-corte | 10 pp |
| `articles/dissertacao/src/chapters/5_mobiwac/` + `.../tables/mobiwac/` | O Capítulo 5: mesmos números v18, mais as correcções de 2026-08-13 a 2026-09-04 que `src_fix` não tem. Muito mais longo. | v18 + Setembro | dentro de `dissertacao.pdf` |

### 1.2 A pergunta de uma linha

> **`src_fix` é superconjunto estrito de `src`? Sim.**
> As duas árvores eram **byte-idênticas** no commit de ramificação: `git diff da97ecf7:"articles/[mobiwac]/src" da97ecf7:"articles/[mobiwac]/src_fix"` devolve **vazio**.
> E `src/` **não recebeu um único commit depois disso**: `git log da97ecf7..HEAD -- "articles/[mobiwac]/src/"` devolve **vazio**.
> ✔ A working tree confirma: o único ficheiro sujo nas três árvores é `src_fix/main.pdf`.
> **Logo: apagar `src/` não perde trabalho commitado.** O que se perde é a *proveniência*, e isso resolve-se com uma tag (item 3.3).

### 1.3 O que a auditoria confirmou sobre a contaminação cruzada

**Confirmado:** agentes da dissertação editaram a fonte do artigo. 13 dos 19 commits pós-prazo em `src/` tocam também `articles/dissertacao/` — `e36b3194` (31 ficheiros da dissertação), `9be36e8e` (13), `d1911c0a` (11), `cd975826` (10). Os assuntos dizem-no: `ef3b2e55` "in BOTH dissertation Ch.5 and MobiWac paper source (author-authorized cross-boundary)".
**Refutado na direcção inversa:** nenhum commit de `src_fix` tocou `src/`. A reescrita v18 esteve limpa.

### 1.4 O build submetido

`97f01a50` (✔ 2026-07-11 00:28:46 -0300): `src/main.pdf` com **8 páginas**, e o abstract renderizado casa palavra a palavra com o bloco de `EDAS_SUBMISSION.md`, incluindo "about +28 to +40 macro-averaged F1". `src/main.pdf` esteve em 8 pp de `113cf8ce` (07-09) a `c178f35a` (07-29) e passou a 9 em `e36b3194` (07-30).
**Consequência:** `src_v1/VERSION.md` (“uploaded to EDAS on 2026-07-09 … 10-page fee variant”) é contradito pelo `EDAS_SUBMISSION.md`, um dia mais novo, que a 2026-07-10 ainda regista o Step 3 pendente e descreve um manuscrito de 8 pp sem taxa. **O `VERSION.md` está errado.**

### 1.5 Ficheiros críticos fora do git (✔ verificado nesta sessão)

```
?? articles/[mobiwac]/CAMERA_READY.md                          717 linhas, o ledger de registo
?? articles/[mobiwac]/review/gate_revision_plan_2026-08-11.md  492 linhas, o único artefacto de revisão v18
 M articles/[mobiwac]/CLAUDE.md        (banner STOP → CAMERA_READY, por commitar)
 M articles/[mobiwac]/GLOSSARY.md      (banner STOP → CAMERA_READY, por commitar)
 M articles/[mobiwac]/src_fix/main.pdf (o único build v18, 10 pp, por commitar)
```
✔ `git tag -l` não tem nenhuma tag MobiWac. Um `git clean -fd` destrói os dois primeiros de forma irrecuperável; um `git checkout -- .` repõe leis v17 e o PDF v17 dentro da pasta chamada "artigo de registo".

---

## 2. A ÁRVORE ÚNICA

> ### ⚠ SIMPLIFICAÇÃO DO AUTOR — 2026-09-06 (relatada por `knowladge`)
>
> > *"os textos que estão hoje no src, src_fix e afins não importam muito, o que importa é o texto da
> > dissertação, vamos passar ele para lá e depois fazemos as adaptações, acho que não vale ficar
> > preocupando com o que está em cada um desses src, até pq eles usam o v17."*
>
> **Instrução: o Capítulo 5 vai inteiro, e adapta-se depois.** Não se reconcilia árvore contra
> árvore. A §2.2 abaixo deixa de ser uma tabela de decisão e passa a ser o que é: **o Capítulo 5 é a
> base de tudo**, mais uma lista curta de excepções nomeadas.
>
> **Uma correcção de facto à premissa, e é a razão de as excepções existirem.** ✔ Verificado nesta
> sessão pela Tabela 3 de cada árvore: o `src/` imprime categoria conjunta **63,32 / 64,51 / 65,79 /
> 79,84 / 77,24 / 77,05** — v17, com vazamento. O `src_fix/` imprime **35,42 / 30,59 / 34,57 / 37,55
> / 36,19 / 35,63** — **v18, idêntico ao capítulo célula a célula**. Ou seja: `src/` e `src_v1/` são
> v17 e podem ser ignorados sem custo, mas **o `src_fix/` não é v17** — é a reescrita v18, e tem
> quatro passagens que o capítulo perdeu.
>
> ### As excepções — REVISTAS a 2026-09-06, de quatro para duas
>
> ⚠ **Duas das quatro que este plano listou não sobreviveram à verificação, e o erro foi meu.**
> Construí-as sobre os relatórios por secção sem verificar o lado da **ausência** — isto é, aceitei
> "o capítulo perdeu X" sem procurar X sob outras redacções no capítulo. É a mesma armadilha que
> este plano avisa noutros sítios. A sessão `knowladge` contestou duas; verifiquei as quatro.
>
> **CAI — #1 "single saved model".** A propriedade **não** se perde. ✔ Contagens no texto vivo
> (comentários removidos): `one model` 14× no capítulo contra 12× no `src_fix`; `single forward`
> 3× contra 2×; e a frase mais forte da propriedade — *"operational rather than arithmetic: one
> artifact to train, version, and deploy, and one forward pass…"* — **está nas duas árvores**
> (`04_method.tex:75` no capítulo, `:58` no `src_fix`). O que difere é a frase *"reading both
> answers from a single saved model"* no bullet 2 da introdução. E o bullet 2 do capítulo é
> **melhor**: nomeia Texas e California, usa a superfície da lei (*"a two-point margin registered
> before any result was read"*) e escreve *"equivalent to zero within half a point"*, enquanto o do
> `src_fix` diz *"within a fifth of a point"*, que é magnitude e não a superfície de registo.
> **Não restaurar.** Quando muito, acrescentar quatro palavras ao bullet do capítulo — opcional.
>
> **CAI — #2 unanimidade dos folds.** ✔ O capítulo carrega o facto, só o coloca noutro sítio:
> `06_results.tex` (*"all five folds favor"*), `tables/mobiwac/representation.tex`, e
> `08_conclusion.tex` (*"every fold favor"*). Falta só no bullet 1 da introdução. **Não é perda.**
>
> **FICA — a divulgação do prior de transição. É a única perda real, e é substância.**
> ✔ Procurada sob cinco redacções (`13 to 27`, `inflated region accuracy`, `Only the HMT-GRN`,
> `uses this prior`, `whole dataset inflated`): existe **só** em `src_fix/sections/05_setup.tex:30`
> e tem **zero** ocorrências no capítulo. É a frase *"A version built from the whole dataset
> inflated region accuracy by 13 to 27 points. Only the HMT-GRN comparison model uses this prior;
> our joint and dedicated models do not."* Protege a comparação com o baseline de região primário —
> um árbitro que conheça o HMT-GRN pergunta por isto. **Migrar como acrescento.**
>
> **FICA, mas em versão mais fraca — a cláusula de Alabama.** Não é "falsa", é **ambígua**.
> ✔ O capítulo diz *"Alabama's is not, and it is the dataset with the largest region difference"*.
> Dentro da frase o conjunto são os quatro datasets dentro da margem, e aí Alabama (−0,874) **é** o
> maior. Lido contra a Tabela 3, onde o Texas está a +1,206, é falso. O `src_fix` diz *"the interval
> reaches just past one point, so there the two-point margin is the operative one"* — que enuncia o
> mecanismo em vez de um superlativo e não se presta à leitura errada. **Preferir a do `src_fix`.**
>
> ⚠ **Este item saiu do âmbito da migração e está com o `knowladge` (2026-09-06).** A frase é texto
> de artigo reproduzido numa dissertação a dias do depósito, portanto o defeito existe **também**
> do lado de lá. O `knowladge` levou-o ao autor como errata pequena e opcional, com a recomendação
> de acrescentar o escopo (*"the largest of these four"*), que não muda número, veredicto nem
> intervalo. **Não agir deste lado:** se o autor aceitar, precisa de uma linha na Tabela B.5 e da
> re-execução do portão `src_utils/count_errata_rows.py`, e isso tem de acontecer na dissertação
> primeiro, senão os dois textos voltam a divergir.
>
> **CAI — o parágrafo de integridade.** A versão do capítulo é a **mais cuidadosa**: tem a ressalva
> de escopo (*"This check covers three datasets at one seed"*), o mecanismo de porque região e
> categoria diferem, a cobertura de 67–87 %, e fecha com *"Within this coverage, whole-dataset
> training did not meaningfully change the results"* — onde o `src_fix` afirma *"which is within
> fold noise"*, que é a alegação menos suportada. **Fica a do capítulo.**
>
> **REFORMULADA — #4, a §8 Conclusão.** A objecção do `mobiwac-writer` mostrou que a base `src_fix`
> custa **oito** edições, não quatro, e três delas são defeitos de lei (ver o aviso na §2.2). Com o
> capítulo já escrito segundo a lei que o autor escolheu, **a recomendação inverte-se: base = o
> capítulo**, e portam-se do `src_fix` as três cláusulas que faltam, como acrescento. Custa +175
> palavras (0,18 pág.) mas não custa nenhuma edição de conformidade — e é consistente com a
> instrução do autor de levar o capítulo inteiro.
>
> Tudo o resto do trabalho de reconciliação que a §2.2 e a §3 descrevem **fica como proveniência,
> não como lista de tarefas**. A §3 continua útil pelas **conversões mecânicas (CONV-1..8)** e pelo
> **inventário de quebras**, que são sobre o que parte quando o texto sai da dissertação — e isso
> não muda com a simplificação.


**Alvo:** `articles/[mobiwac]/src/` — uma só árvore, construída sobre o texto da dissertação (decisão do autor, 2026-09-06), com dois desvios justificados (§3 e §8, onde a dissertação nada acrescenta).
**Método de arranque:** copiar `src_fix/*` sobre `src/*` (é superconjunto estrito, §1.2) e só depois substituir ficheiro a ficheiro. Isto preserva `main.tex`, `figs/`, `.gitignore`, `IEEEtran.*` e o `references.bib` do lado do artigo.
`src_fix/` e `src_v1/` desaparecem da working tree **depois** de 3.60 (repontagem), recuperáveis pela tag de 3.3.

### 2.1 As sete conversões mecânicas (CONV)

Aplicam-se a **todo** o texto vindo da dissertação. Definidas aqui uma vez; os itens da §3 referem-nas pelo código.

| Código | Conversão |
|---|---|
| **CONV-1** | Cabeçalho: `% !TeX root = ../../main.tex` → `../main.tex`; apagar o bloco de proveniência do split de 2026-07-28 e o ponteiro "NORTH_STAR section 4"; **restaurar** o cabeçalho de planeamento do artigo que estava em `src_fix` |
| **CONV-2** | Namespace de labels: tirar `mobiwac:` de **todos** os `\label` e `\ref`. **Excepção que não é um strip:** `tab:mobiwac:representation` → **`tab:substrate`** (não `tab:representation`) |
| **CONV-3** | Chaves de citação: `velickovic2019deep`→`velickovic2019dgi`; `Lim2022`→`lim2022hmtgrn`; `nash`→`navon2022nashmtl`. **Não** acrescentar as entradas alias ao `references.bib` — foram removidas de propósito (nota em `src_fix/references.bib:148-150` e `:278-279`) |
| **CONV-4** | Ortografia e dêixis: `multitask`→`multi-task`; `Figure~\ref`→`Fig.~\ref`; `this chapter`→`this paper`; `The remainder of this chapter`→`The rest of the paper`. ⚠ **A primeira é uma escolha, não uma correcção** (levantada pelo `mobiwac-writer`): ✔ o Cap. 5 escreve `multitask` 10 vezes e `multi-task` 0; o `src_fix` escreve `multi-task` 10 vezes e 0 sem hífen. **Recomendação: manter `multi-task` no artigo**, por fidelidade ao título registado no EDAS — *"A Check-in-Level **Multi-Task** Study on Mobility Data"* (`EDAS_SUBMISSION.md:19`), que não pode mudar. Pergunta de uma linha ao autor; o default é o hífen |
| **CONV-5** | Flutuantes: apagar todo `\input{tables/mobiwac/...}`, todo `\providecommand{\sd}` duplicado e **todos** os ambientes `figure` — no artigo os flutuantes vivem em `main.tex` |
| **CONV-6** | Léxico do GLOSSARY do artigo (§3/§4): `arm(s)` → "both models / the joint and the dedicated model"; `recipe` → "training configuration"; `checkpoint` → **"saved model"**. Instância a instância (`frozen` com glosa é legal, `GLOSSARY.md:139`). ⚠ **Contradição interna corrigida (2026-09-06):** o item 3.43 mandava restaurar de B a frase *"read from one saved checkpoint"*, que a própria CONV-6 proíbe. **Resolvido a favor de "saved model"** — é a forma que a §1 de B já usa (*"a single saved model"*) e que as duas leis aceitam. Aplica-se também à conclusão de B, que diz *"reads both answers from one saved checkpoint"* |
| **CONV-7** | Comentários: **manter** os que explicam prosa (as decisões datadas); apagar os que são só da dissertação (`COD-*` de layout, probes `R13-*`, ponteiros para `PENDENCIAS`) |

> **CONV-8 (verbos de veredicto) — DESBLOQUEADA em 2026-09-06, e é preservação, não trabalho.**
> O autor decidiu que a `WRITING_LAW` governa o camera-ready (citação literal em §7, DP-1), portanto
> `matches` sai dos dois eixos: região diz *"stays within the two-point margin"*, categoria diz o
> limite derivado. **O texto do Capítulo 5 já está assim escrito** — a migração herda a regra. A
> instrução é: ao portar, **não "corrigir" a superfície da dissertação de volta para a do artigo**.
> Só é preciso agir nas secções cuja base seja o `src_fix` (§3 e §8), onde há quatro sítios a rever.

### 2.2 Composição, ficheiro a ficheiro

Legenda: **A** = `src/` v17 · **B** = `src_fix/` v18 · **C** = `dissertacao/src/chapters/5_mobiwac/` (ou `.../tables/mobiwac/`, `.../figures/mobiwac/`).

| Ficheiro alvo | Base | Ports / desvios |
|---|---|---|
| `sections/01_introduction.tex` | **C** | CONV-1..5. **Restaurar de B:** "and reading both answers from a single saved model" e "with every fold favoring it at every dataset" (+ ressalva Florida p=0,07). **Não importar** a frase da geometria. Corrigir escopo das sementes no bullet 3. Sujeito a **D5** |
| `sections/02_related.tex` | **C** | CONV-1..5. **Desambiguar o segundo "four"** → "at four of the six datasets". S9 (auto-posicionamento) e S7 (balanceadores) são **prosa nova**, não portes. Ponteiro da cascata: **DP-3** |
| `sections/03_problem.tex` | **B (excepção)** | **Nenhum port.** A e B são byte-idênticos (`md5 efde0dc31c57965c964579098c032836`); C difere em três tokens de label. Ficheiro fica intocado |
| `sections/04_method.tex` | **C** | CONV-1..5 (apagar o `figure` de C:78-91; a Fig. 2 vive em `main.tex:118-124`). **Restaurar de B** o guarda-costas COST de 2026-07-08 (B:8-12). Apagar o comentário COD-017. Sums 1.1/2.0→**1.9/2.8**. Citação Menon (ver bib) |
| `sections/05_setup.tex` | **C, com enxerto de B** | CONV-1..5. **Manter o parágrafo de integridade de B** (três canais: prior de transição por fold com a inflação de 13–27 pontos, e os âmbitos de treino HGI/CTLE) — sujeito a **DP-6**. **Manter a cláusula de Alabama de B** (a de C é falsa). Portar o par Wilcoxon+sementes **junto**. Portar a frase "no equivalence margin on next category". Parágrafo "Configuration search": **DP-7** |
| `sections/06_results.tex` | **C** | CONV-2,3,5 (apagar 2 `\input`, 2 `\providecommand`, 2 `figure`). **Não retro-portar 4 passagens de C** (ver 3.33). Freeze e cascata continuam apagados (**DP-3/DP-4**). Ponteiro da Fig. 3 só se **D13** a repuser. **O banner de 2026-08-12 do cabeçalho tem de viajar com os comentários** |
| `sections/07_discussion.tex` | **C** | CONV-1..6. **Restaurar de B:** "and both answers are read from one saved **model**" (⚠ **não** "checkpoint" — ver CONV-6) e "at one forward pass instead of two". ⚠ **NÃO restaurar** o "(TOST, $\pm2$~pp)" como superfície de veredicto: sob a lei escolhida pelo autor isso é o nome técnico e só pode aparecer onde o teste está a ser nomeado. Partir o comma splice de abertura de C. Contagem de limites: **D4** |
| `sections/08_conclusion.tex` | **B (excepção)** | B tem 238 palavras contra 413 de C e três cláusulas que C perdeu. ⚠ **NÃO são quatro edições cirúrgicas — são oito.** Ver o aviso abaixo. Sujeito a **D5** |

> ### ⚠ A §8 com base B custa mais do que o plano dizia (corrigido 2026-09-06)
>
> Levantado pelo `mobiwac-writer`; ✔ verifiquei lendo a conclusão de B inteira. O erro de método foi
> meu e é o mesmo de antes: **inventariei a superfície de veredicto procurando a cadeia `match`, e a
> superfície é um conceito, não uma cadeia.** Sob a `WRITING_LAW`, o nome técnico *"non-inferior
> (TOST, ±2 pp)"* só pode aparecer onde o teste está a ser nomeado; como **superfície de veredicto**
> tem de ser *"stays within the two-point margin"*.
>
> As oito edições que a conclusão de B precisa:
> 1. `3.5` → **`3.55`** · 2. `3.0` → **`3.06`** · 3. separação de eixos (*"on either task"* aplica a
>    margem da região à categoria) · 4. a magnitude `+0,23` a `+6,29`
> 5. *"remains **non-inferior (TOST, ±2 pp)** at the other four"* → a superfície da lei
> 6. *"Joint training therefore **pays** where the region task is **hardest**"* — **dois** defeitos
>    numa frase: é a leitura C4 **morta** (`CAMERA_READY §5 C4`), e "pays" é metáfora de dinheiro
>    banida pela `GLOSSARY §8`. O mesmo em `07_discussion` de B.
> 7. *"several of them **resolvable**"* — a direcção tem de viajar (`WRITING_LAW:221-225`): são
>    défices **resolvidos** em AL, AZ e FL, não diferenças "resolúveis"
> 8. *"reads both answers from one saved **checkpoint**"* → "saved model" (CONV-6)
>
> **A estimativa de +17 a +37 palavras está errada** e tem de ser refeita depois de 5–8.
> **O teste do Bloco J deixa de ser um grep por `match`**: tem de procurar `TOST`, `hardest`, `pays`,
> `resolvable`, `checkpoint` e `either task`. Um grep por `match` passa a verde com sete defeitos
> vivos.
| `tables/tbl1_datasets.tex` | **B** | Portar **só** a cláusula factual da legenda de C: "Check-ins, users, and POIs are counts over the full corpus, before the filter that keeps users with at least ten visits; the Windows column is computed after it" |
| `tables/tbl2_substrate.tex` | **B + coluna de A/C** | **Repor a coluna `±` de desvio-padrão por fold** que `src_fix` largou: AL 1.16/0.84, AZ 1.15/1.08, Istanbul 0.89/0.63, FL 0.42/0.41, CA 0.47/0.41, TX 0.51/0.48 (`dissertacao/src/tables/mobiwac/representation.tex:15-20`), e repor o `\providecommand{\sd}`. Ordem das linhas: **DP-8** |
| `tables/tbl3_results.tex` | **B** | Portar a nota de rodapé factual da cobertura da busca. **Não tocar** no comentário de proveniência do POI-RGNN nem nos valores impressos 34.49/33.03/31.78. Convenção de ênfase e ordem das linhas: **DP-8** |
| `figs/fig1_dataflow.tex` | **B + correcção de 1 linha** | Substituir o rótulo pela versão de `dissertacao/src/figures/mobiwac/fig1_dataflow.tex:69`: "(forward in time)" e ", elapsed time". **Não** copiar o `\usepackage{newtxtext,newtxmath}` |
| `figs/fig2_model.tex` | **B, intocado** | Não está desactualizado; a dissertação nem sequer tem fonte, só o PDF construído a partir desta |
| `figs/fig3_embquality.{py,tex,pdf}` | **B, fora do build** | Continua cortado em `main.tex:131-134`. Valores pré-v18. Reposição: **D13**. O comentário do `main.tex:132` (0.53/0.00, "AL/AZ/FL") não casa com o próprio `.py` (0.5668/0.0003, cinco estados) — corrigir ou apagar |
| `figs/fig4_deltas.{py,tex}` | **B** | Dados **idênticos** aos da dissertação. Corrigir os dois comentários mortos em `fig4_deltas.py:81-82`. Legenda: portar a de C (que encaminha cada afirmação para o seu teste) — ligado a **DP-8** |
| `references.bib` | **B** | **Acrescentar uma entrada:** `menon2021logitadjustment`, copiada verbatim de `dissertacao/src/references.bib:751`, **sem DOI** (a Crossref não tem nenhum registado). Mais nada muda: `src/references.bib` e `src_fix/references.bib` são byte-idênticos (30.397 bytes, 46 entradas) |
| `main.tex` | **B** | Só o que os itens da §3 mandarem (`\IEEEpubid`, eventual `\input` da Fig. 3) |

---

## 3. MIGRAÇÃO FICHEIRO A FICHEIRO

Ordem de execução. Marcas: **[MEC]** mecânico · **[ESC]** o escritor decide · **[AUT]** precisa de decisão do autor (§7).
Todos os caminhos são relativos à raiz do repositório. Citar sempre as aspas por causa dos parênteses rectos.

### Bloco 0 — Preservação (antes de tudo o resto)

> ### ⚠ 3.0 [AUT/MEC] — RESOLVER QUAL É O BUILD SUBMETIDO, ANTES DE 3.3 (Fable, 2026-09-06)
>
> Este plano marcava `97f01a50` como o build submetido com um ✔. **Não estava provado.** ✔ Verifiquei:
> existem **dois** builds de 8 páginas, e diferem no que importa.
>
> | commit | data | Tabela 3, categoria conjunta | §7 abre com |
> |---|---|---|---|
> | `97f01a50` | 2026-07-11 00:28 (dia do prazo) | 63,33 / 64,54 / 65,84 / 79,85 (diag-best) | *"**Three** limits"* |
> | `f66f8a73` | 2026-07-20 (**nove dias depois**) | 63,32 / 64,51 / 65,79 / 79,84 (joint-best) | *"**Two** limits"* |
>
> A evidência que este plano usava — *"o abstract casa com o bloco do `EDAS_SUBMISSION.md`"* — é
> **circular**: esse bloco foi refrescado a 2026-07-10 e o ficheiro nunca regista o Step 3 como feito.
> O `CAMERA_READY §10.2` já dava isto por não resolvido.
>
> **Consequência:** a tag `mobiwac/submetido-EDAS`, o README de 3.5 e sobretudo a **nota aos
> chairs (D1)** descreveriam o texto errado como "o que os revisores aceitaram".
> **Ação, dois minutos:** descarregar o manuscrito do EDAS #1571313639 e comparar por `pdftotext`
> contra os dois commits. **3.3 não corre antes disto.**
> ✔ O que **não** depende da resposta: `grep -c "absorb the category"` devolve **0 nos dois**, logo
> nenhum dos candidatos declarava o canal de vazamento e o argumento de D1 aguenta-se de qualquer forma.

**3.1 [MEC] Commitar os dois ficheiros untracked.**
`git add "articles/[mobiwac]/CAMERA_READY.md" "articles/[mobiwac]/CONSOLIDATION_PLAN.md" "articles/[mobiwac]/review/gate_revision_plan_2026-08-11.md"`
⚠ **Este plano também está untracked** e o risco 1 (`git clean -fd`) destrói-o junto com os outros dois (Fable). O `-f` é desnecessário: nenhum dos três está ignorado.
*Evidência:* ✔ `git status --porcelain -- "articles/[mobiwac]/"` mostra os dois como `??`.
*Teste:* o mesmo comando deixa de devolver linhas com `??`.

**3.2 [MEC] Commitar as três modificações da working tree.**
`articles/[mobiwac]/CLAUDE.md`, `GLOSSARY.md` (os banners STOP) e `src_fix/main.pdf` (o único build v18).
*Teste:* `git show HEAD:"articles/[mobiwac]/src_fix/main.pdf" | pdfinfo -` diz **10 páginas** e o md5 deixa de ser `6a32e1e30b7b285e19e4b857cf49c0ae`.

**3.3 [MEC] Duas tags, antes de qualquer fusão.**
`git tag -a mobiwac/submetido-EDAS 0834419b` ✅ **FEITO** (identificado contra o PDF do autor) e `git tag mobiwac/tres-arvores <commit de 3.2>`.
*Teste:* `git show mobiwac/submetido-EDAS:"articles/[mobiwac]/src/main.pdf" | pdfinfo -` diz **8 páginas**.
*Porquê:* é isto que torna reversível tudo o que vem a seguir, e é a única coisa que preserva o texto submetido depois de `src/` passar a conter o camera-ready.

> ### 🔒 PASSAGENS PROTEGIDAS DO CORTE — não separar, não aparar (2026-09-06)
>
> A D3 vai obrigar a largar ~2 000 palavras numa segunda fase. Estas três passagens são frágeis ao
> corte de maneiras que não se veem a olho: **cortar metade de cada uma inverte o que ela diz.**
>
> **P1 · A unidade de duas frases da região** (`5_mobiwac/06_results.tex:216-224`). A primeira frase
> nomeia um mecanismo — *"which is where the region task is hardest and where the dedicated model has
> the most to gain from an auxiliary signal"*. A segunda é o que a torna honesta: nomeia os dois
> confundidores (não-monotonia dentro do par; o número de regiões co-varia com o tamanho do corpus) e
> declara o conjunto *"an observation about where the benefit appears rather than as a law"*.
> ⚠ **Migram juntas e cortam-se juntas.** Se o corte levar a segunda e deixar a primeira, o capítulo
> passa de **observar** a **afirmar**, e a teoria do autor — que o texto hoje sustenta de forma
> honesta — vira uma alegação que a evidência não carrega. (Levantado pelo `knowladge`.)
>
> **P2 · A recusa explícita da explicação por partilha** (`5_mobiwac/08_conclusion.tex:12-13`):
> *"the evidence shows that **sharing alone does not explain the outcome** because the result also
> depends on what the model represents"*. ✔ **Existe só no capítulo — zero ocorrências em todo o
> `src_fix`.** É a defesa mais forte da posição do autor e não é uma ausência que se argumenta: é uma
> frase positiva no texto entregue. Sem ela, a §8 fica a dever a recusa que a D4 vai exigir.
>
> **P3 · A frase forward-only no Método.** Ainda **não existe** em nenhuma árvore do artigo — no
> `src_fix` a propriedade só é afirmada dentro da quinta limitação da §7. Com a D1 decidida (sem nota
> aos chairs), **é o texto que tem de ser honesto sozinho**, e o Método é onde um leitor a procura.
> Entra como acrescento, e é a única passagem desta lista que ainda tem de ser escrita.

### Bloco A — Estrutura

**3.4 [MEC] Arrancar a árvore única.** Copiar `articles/[mobiwac]/src_fix/*` sobre `articles/[mobiwac]/src/*` (ver §1.2: superconjunto estrito).
*Teste:* `diff -r "articles/[mobiwac]/src" "articles/[mobiwac]/src_fix"` fica vazio **excepto** o
`.DS_Store` de `src/` (só apagado na §5.1) e o `REVISION_PLAN.md` (ver aviso).
⚠ **`cp src_fix/* src/` traz o `src_fix/REVISION_PLAN.md`** — 386 linhas, rastreado, cujo cabeçalho
diz *"Status: PROPOSAL … `src/` is the submitted version of record and stays untouched"*, o que
passa a ser falso dentro da própria árvore do camera-ready. **Mover para `archive/` no mesmo passo**
(Fable, 2026-09-06; a §5 e a §6 não o mencionavam).

**3.5 [MEC] Criar `articles/[mobiwac]/src/README.md`** com três linhas: esta árvore é o camera-ready v18; o texto submetido está na tag `mobiwac/submetido-EDAS`; a dissertação cita caminhos desta pasta como doador verificado — ver ERRATA.
*Porquê:* **25 sítios** do `.tex` da dissertação apontam para esta pasta (23 para `src/`, 2 para `src_fix/`). ✔ **Verificado nesta sessão: os 25 estão dentro de comentários `%`** — nenhum é texto renderizado, portanto **nada quebra no build da dissertação**. O que se quebra é a semântica: depois de 3.4 esses ponteiros descrevem o camera-ready e não o doador que citam. É defeito de registo, não de compilação — o que baixa a urgência mas não dispensa o README.

### Bloco B — §1 Introdução

**3.6 [MEC]** Copiar `articles/dissertacao/src/chapters/5_mobiwac/01_introduction.tex` → `src/sections/01_introduction.tex` e aplicar CONV-1..5. Mapa completo dos 14 `\ref` + 2 `\label`: `sec:mobiwac:X`→`sec:X` para todos, e **`tab:mobiwac:representation`→`tab:substrate`**.
*Teste:* `grep -rn "mobiwac:" "articles/[mobiwac]/src" --include=*.tex` → 0 ocorrências; build sem `undefined`.

**3.7 [ESC]** Restaurar de B, no bullet 1, "with every fold favoring it at every dataset" **junto com** a ressalva de Florida (`p = 0,07`), e no bullet 2 "and reading both answers from a single saved model" a seguir a "In one forward pass,".
*Evidência:* 5/5 folds nos seis datasets (`CAMERA_READY.md §3.3`); a redacção prescrita em `§5 C1`; a propriedade operacional em `§5 S1`.
*Teste:* a frase da unanimidade e a ressalva aparecem no mesmo período.

**3.8 [AUT — ver DP-9]** **Não** importar "The advantage tracks the geometry of the per-visit vectors…". Se o autor a quiser, tem de viajar com a delimitação de `dissertacao/.../06_results.tex:39-40`.
*Evidência:* `CAMERA_READY.md §5 C1c` (NÃO RESOLVIDA; constantes de um CSV de 2026-06-24 sobre o motor v14, agregadas por POI). A mesma afirmação está viva no abstract, `src/main.tex:79`.

**3.9 [AUT — D5]** A frase do eixo da categoria muda em **quatro** sítios, não dois: `sections/01_introduction.tex`, `sections/07_discussion.tex`, `sections/08_conclusion.tex` e `main.tex:85-87`.
*Teste:* `grep -rn "a fifth of a point" "articles/[mobiwac]/src"` → 0, ou 4, nunca um número intermédio.

**3.10 [ESC]** Corrigir o escopo das sementes: "All joint and dedicated results average four random initializations" → forma ligada à Tabela 3 (o braço do bullet 1 é semente 0, n=5).
*Evidência:* `CAMERA_READY.md §3.3` facto 3; `src/tables/tbl2_substrate.tex:12-13`.

### Bloco C — §2 Trabalho relacionado

**3.11 [MEC]** Copiar C → `src/sections/02_related.tex`, CONV-1..5. Nota: A e B são **byte-idênticos** aqui (`md5 aa6259c35bb4c4b3afd28af18d92d544`) — não há eixo A-vs-B.

**3.12 [ESC]** A frase do cosseno de gradiente: usar a versão de C ("measured on the reported joint model at four datasets, is equivalent to zero at every one of them, with per-dataset means within two thousandths of zero") **e desambiguar o segundo "four"** → "at four of the six datasets" (ou nomeá-los: Istanbul, Alabama, Arizona, Florida).
*Evidência:* `dissertacao/src/chapters/apx_f_cosine.tex:339-341` e `:194-196`; a versão antiga está na lista NUNCA CITAR, `CAMERA_READY.md:235`.
*Teste:* o parágrafo não diz "four" duas vezes sobre conjuntos diferentes (o primeiro "four Gowalla states" inclui a Geórgia e exclui Istambul).

**3.13 [AUT]** Reescrever, não portar, "on which sharing helps instead of hurting". C tem a **mesma** cláusula, logo não há port possível. `GLOSSARY.md:377` permite exactamente uma frase de auto-posicionamento na §2 → **reescrever, não apagar**.
*Evidência:* `CAMERA_READY.md §5 S9` (MORTA); sob v18 a partilha não ajuda em 5/6 (cat) e 4/6 (reg).

**3.14 [AUT]** Reescrever, não portar, o parágrafo dos balanceadores (idêntico em B e C, palavra a palavra, incluindo `0.68` e `0.19`). Decidir ao mesmo tempo se PCGrad continua nomeado (recomendação REV-011, preservada só em `dissertacao/.../02_related.tex:144-156`, nunca aplicada).
*Evidência:* `CAMERA_READY.md §5 S7`: a nossa própria perda **é agora** peso igual (0,50/0,50) e a frase diz que dois balanceadores batem o peso igual; os valores são pré-v18 (âncora AL equal_weight 53,57 contra 30,59 em v18).

**3.15 [AUT — DP-3]** O ponteiro da cascata ("Since the cascade is the pattern that the field uses, we test the choice directly") existe em B e **não** em C: o autor apagou-o em `5be3458b` (2026-08-06), no mesmo commit que apagou 142 linhas de `06_results.tex` e **removeu uma linha da `errata_scope.tex`**.

**3.16 [MEC]** Actualizar o bloco de comentário que hoje afirma que os dois textos são idênticos e que a frase diz "positive"/"seven" — falso desde `65ada3cd` (2026-08-20).
*Evidência:* `dissertacao/.../02_related.tex:199-208`; o gémeo em `src_fix/sections/02_related.tex:112-121`.

### Bloco D — §3 Problema

**3.17 [MEC] Não fazer nada.** Manter `src/sections/03_problem.tex` tal como veio de 3.4.
*Evidência:* A ≡ B (`md5 efde0dc31c57965c964579098c032836`); C difere em 3 tokens de label; contagem de palavras 271 vs 271. `CAMERA_READY.md §8` já o lista em "Já correcto — não tocar".
*Teste:* `md5 -q "articles/[mobiwac]/src/sections/03_problem.tex"` = `efde0dc31c57965c964579098c032836`.
*Armadilha:* se alguém copiar C mesmo assim, o `\label{sec:problem}` renomeado parte **três** referências noutros ficheiros (`01_introduction.tex:46`, `05_setup.tex:22`, `05_setup.tex:47`).

### Bloco E — §4 Método

**3.18 [MEC]** Copiar C → `src/sections/04_method.tex`, CONV-1..5 (apagar o `figure` de C:78-91 e o comentário COD-017), **restaurar de B o bloco de guarda COST de 2026-07-08** (B:8-12: "the old '+5 percent' and the INV2 numbers … must NEVER return to the prose").
*Teste:* `grep -n "+5 percent\|INV2" "articles/[mobiwac]/src/sections/04_method.tex"` volta a encontrar o guarda-costas.

**3.19 [MEC] A construção forward-only entra no Método.** Inserir a frase de C:18 verbatim: "The consecutive-visit edges run in one direction only, from an earlier visit to a later one, for the same reason: a target is predicted from a user's past, so the representation is built from the past alone." E "…preceding it." → "…preceding it, never anything that follows."
*Evidência:* `CAMERA_READY.md §5 S13` e `§8 Fase 1 [C]` — obrigatório sob qualquer decisão. Em `src_fix` a propriedade só aparece na §7 (`07_discussion.tex:111`); verificado sob seis redacções diferentes.
*Teste:* acompanhado obrigatoriamente de 3.30 (Figura 1), ou prosa e figura contradizem-se na mesma página.

**3.20 [MEC] Somas de parâmetros.** `1.1 million` → `1.9 million`; `2.0 at California` → `2.8 at California`. Levar o comentário de decisão de C:56-69.
*Evidência:* errata_scope linha 4; `CAMERA_READY.md §5 S1`: conjunto 4.197.621 (AL) / 5.151.189 (CA); dedicados somados 1.850.980 / 2.804.548.
*Teste:* os valores 4.2 e 5.2 **não** mudam; só os dois somatórios.

**3.21 [ESC — D3]** Inserir a frase da assimetria das métricas (C:48-50, 36 palavras) e "so the comparison **between them** is not affected by it".
*Evidência:* `CAMERA_READY.md §8 Fase 2` (reforço, não obrigatório). **É a primeira candidata a corte se D3 mandar 8 páginas.**

### Bloco F — §5 Setup

**3.22 [MEC]** Copiar C → `src/sections/05_setup.tex`, CONV-1..5, apagar o `\input{tables/mobiwac/datasets}` de C:24 (o `main.tex:124` já lá o mete), e converter `\cite{Lim2022}` (CONV-3).
*Teste:* build sem `Citation ... undefined`.

**3.23 [AUT — DP-6] Manter o parágrafo de integridade de B, não o de C.** B tem 219 palavras e três canais nomeados; C tem 410 e um canal. C perdeu duas coisas que o artigo precisa: o prior de transição por fold com "A version built from the whole dataset inflated region accuracy by 13 to 27 points. Only the HMT-GRN comparison model uses this prior", e "HGI is pre-trained once on the whole dataset … while CTLE is trained for each fold using training users only". Enxertar em B o que C traz de bom: "This check covers three datasets at **one seed**" e a explicação do vector-por-lugar na categoria.
*Ausência verificada:* nove redacções para o prior ("13 to 27", "inflated region", "transition prior", …) e seis para os âmbitos de treino — zero em toda a `articles/dissertacao/src/`.

**3.24 [MEC] Manter a cláusula de Alabama de B.** C diz "it is the dataset with the largest region difference" — **falso**: Alabama −0,874 contra Texas +1,206 e Califórnia +1,057.
*Evidência:* `dissertacao/wrapup/evidence/ladder_recompute.json`; `CAMERA_READY.md §3.2`.

**3.25 [ESC] Portar o par Wilcoxon+sementes JUNTO** (C:135 e C:137). C mudou o "$4\times5=20$ fitted models" de um parágrafo para o outro; portar só um duplica ou apaga a afirmação. Levar o comentário de decisão C:127-134.
*Evidência:* errata_scope linha 3; `CAMERA_READY.md §8 Fase 1`.
*Armadilha:* **não** fazer find-and-replace de "primary analysis" — essa expressão não existe em `src_fix`.

**3.26 [ESC]** Acrescentar ao fim do parágrafo do plano de análise: "On next category the plan registered no equivalence margin, so a difference that fails the superiority test is reported as unresolved rather than as a match." **Sem linha na errata_scope** → criar uma (3.62).
*Evidência:* `CAMERA_READY.md §8 Fase 2`.

**3.27 [AUT — DP-7]** O parágrafo "Configuration search" (206 palavras, C:39-75) **não** entra no Setup do artigo. A correcção factual entra onde o plano de registo a põe: `sections/06_results.tex`, `sections/07_discussion.tex` (2 sítios) e a nota de rodapé da Tabela 3 — ver 3.36.
*Nota:* em C o parágrafo está **encravado no meio do argumento de integridade** (C:32-37 põe a pergunta, C:39-75 fala de busca, C:77-89 responde) sob um título de subsecção que não o cobre.

**3.28 [AUT — DP-9]** Antes de publicar o parágrafo de integridade em qualquer forma: acrescentar uma frase de escopo dizendo que a auditoria de transdutividade correu no substrato **pré-v18** e numa só semente, ou correr A3.
*Evidência:* `docs/studies/pre_freeze_gates/A4_RESULTS.md:9` ("rebuild the v14"), linhas rotuladas "(seed0)"; `CAMERA_READY.md §5 S14`.

**3.29 [MEC]** Apagar os comentários mortos de `src/sections/05_setup.tex:98-100` (diz que o artigo está em revisão e que as duas árvores têm de ficar idênticas) e `:101-121` (diz "the four reg gains" e "m=4 family", contra a própria prosa que já diz dois e seis).

### Bloco G — Figuras e bibliografia (partilhados)

**3.30 [MEC] Figura 1.** Em `src/figs/fig1_dataflow.tex:64`, pôr "edges: consecutive visits by a user **(forward in time)**" e "features: category, hour, weekday, **elapsed time**". **Não** copiar as linhas `\usepackage{newtxtext,newtxmath}`.
*Evidência:* `diff` completo contra `dissertacao/src/figures/mobiwac/fig1_dataflow.tex` dá exactamente estas duas strings mais o pacote de fontes e a linha de TeX root.
*Teste:* o rótulo renderizado não contradiz 3.19; verificar o tamanho do texto depois do `\resizebox{0.66\textwidth}` do `main.tex`.

**3.31 [MEC] Figura 4.** Corrigir os dois comentários mortos em `src/figs/fig4_deltas.py:81-82` ("deep blue: category (always up)" — quatro dos seis deltas são negativos; "brick red: region (crosses the band)" — o maior é 1,2059 contra `NI_MARGIN = 2.0`). Os **dados** são idênticos aos da dissertação; nada renderizado muda.

**3.32 [AUT — D13] Figura 3.** Ou repor (descomentar `main.tex:131-134`, portar o ponteiro de 45 palavras de C:93-96, e reconciliar o silhouette) ou manter cortada e sem ponteiro. As duas metades andam juntas (regra L4: todo o flutuante é referido).
*Conflito a resolver:* `src/sections/06_results.tex:25` diz 0.55/0.79 (AL/AZ/FL), a dissertação diz 0.57/0.78 (cinco estados), o `main.tex:132` diz 0.53, e o **asset construído desenha os cinco estados** (`fig3_embquality.py:49-50`: `[0.5668, 0.9827]` vs `[0.0003, 0.7750]`).

**3.33 [MEC] Bibliografia.** Acrescentar `@inproceedings{menon2021logitadjustment,...}` a `src/references.bib`, copiado verbatim de `dissertacao/src/references.bib:751` **incluindo a ausência de DOI**. Citar em `sections/04_method.tex` ("cross-entropy loss with logit adjustment~\cite{menon2021logitadjustment}:") e em `sections/06_results.tex:21`.
*Evidência:* errata_scope linha 8; `ERRATA.md:393-419`. Ausência confirmada sob `menon`, `logit`, `Long-tail`, `2007.07314`.
*Armadilha:* **não** inventar um DOI de agregador — a Crossref não tem nenhum, e o incidente `holm1979` está registado como precedente.
*Teste:* `grep -c '\\bibitem' src/main.bbl` passa de 32 para 33.

### Bloco H — §6 Resultados

**3.34 [MEC]** Copiar C → `src/sections/06_results.tex`, CONV-2/3/5. Apagar os dois `\input`, os dois `\providecommand{\sd}` e os dois ambientes `figure` (o `main.tex:124-135` já mete a Tabela 2, a Tabela 3 e a Figura 4).
*Teste:* build sem `Label ... multiply defined` para `tab:substrate`, `tab:results`, `fig:deltas`.

**3.35 [MEC] Quatro passagens de C que NÃO se retro-portam:**
 (a) "about two points below the place embedding" — quantidade v17; sob v18 é 37.13 − 33.45 = **3.68**. Usar a forma de B, sem número.
 (b) "The comparable quantity is the gain over the ceiling" (Istambul) — usar a de B, "the difference against the dedicated model", porque a diferença de Istambul é −0,08, uma perda.
 (c) o mecanismo das vitórias de região — manter a cobertura de B ("consistent with the reading that…"), por causa de **D4**.
 (d) o fecho "The U.S. result repeats on a different continent and region unit" — resto de v17; mas o de B ("within the two-point margin … on either task") também não serve, que é fuga entre eixos. **Frase nova.**

**3.36 [MEC] A cobertura da busca em TRÊS sítios, num só passo:** `sections/06_results.tex`, `sections/07_discussion.tex` (2 ocorrências) e **a nota de rodapé da `tables/tbl3_results.tex`**, que repete a frase demasiado geral e que o plano de registo não lista.
*Texto alvo:* batch size nos seis datasets, learning rate em quatro; a busca conjunta cobre quatro dos seis.
*Evidência:* `CAMERA_READY.md §5 S11`; `dissertacao/.../05_setup.tex:63-76`.

**3.37 [ESC] Identidade da coluna check-in.** Acrescentar a frase de C: a coluna check-in-level da Tabela 2 é a fatia de semente 0 da coluna dedicada de categoria da Tabela 3, e o braço place-level é o mesmo modelo com o input trocado.
*Porquê:* sem ela, AZ 34.51 na Tabela 2 contra AZ 34.57 na Tabela 3 lê-se como erro aritmético em páginas contíguas. Verificado: semente 0 = 35.3539/30.7654/34.5080/37.3630/36.3225/35.6208 contra as médias de quatro sementes 35.3430/30.7750/34.5740/37.3538/36.3251/35.6330.

**3.38 [ESC] Convenção de época.** Portar a versão alargada de C: o desvio pior-por-semente (0,23 / 0,93) **e** que a convenção alternativa transformaria mais quatro células de categoria e mais duas de região em melhorias sobreviventes ao mesmo Holm.
*Evidência:* `CAMERA_READY.md §5 S12` chama à versão de B "materialmente incompleta".

**3.39 [AUT — D5]** Se D5 escolher a rota B (limite derivado): imprimir **os seis** intervalos de confiança da categoria, não só o de Florida, modelado em `dissertacao/.../06_results.tex:228-238` (o endpoint mais largo é 0,3338 em Alabama → 0,34).
*Aviso aritmético:* seis IC de 90% independentes não dão um limite simultâneo; a rota B carrega **dois** números por isso, 0,334 por dataset e 0,489 sob Bonferroni. A conclusão sobrevive (0,489 < 0,5) mas a derivação mostrada não licencia "at once".

**3.40 [ESC — D3] A reconciliação do piso de Markov** (três parágrafos, 258 palavras, C:287-313). Verificada de forma independente: HMT-GRN abaixo do piso em 6/6, ReHDM 3/6, STAN 4/6, persistência da região 32,91% (31.704 de 96.326).
*Evidência:* `CAMERA_READY.md §5 S2` pede a importação. Todas as cláusulas de protocolo que cita já estão na subsecção de baselines do artigo.

**3.41 [MEC] O banner de 2026-08-12 tem de viajar com os comentários.** C mantém cinco blocos longos cujas células e veredictos são v17 (a escada 63.32/64.51/…/77.05, a regra "Honesty rules (do NOT relax)", a nota da convenção com a perda 0.75/0.25, os 54.65/26.56/+28.09 da Tabela 2). Dentro da dissertação, o banner de cabeçalho neutraliza-os. **Copiar o corpo sem o banner mete no artigo uma instrução escrita para restaurar v17.**
*Alternativa:* apagar os cinco blocos.

**3.42 [MEC] Verbo da Tabela 2.** "The check-in-level representation **outperforms**…" → "reaches a higher next-category macro-F1 than", porque o teste emparelhado falha em Florida (p=0,07) e `GLOSSARY.md §1` liga "outperforms" a um teste de superioridade sobrevivente.

### Bloco I — §7 Discussão

**3.43 [MEC]** Copiar C → `src/sections/07_discussion.tex`, CONV-1..6. **Restaurar de B** as três cláusulas: "and both answers are read from one saved checkpoint"; "Isolating the trunk at Texas and California, at the full four-seed protocol, is the experiment that would attribute the gain, and it remains open."; "at one forward pass instead of two". Partir o comma splice de abertura de C.

**3.44 [MEC]** Corrigir as duas ocorrências de "arms" que a reescrita de 2026-08-12 reintroduziu em C (CONV-6). O ficheiro regista duas remoções anteriores da mesma palavra.

**3.45 [ESC]** Primeiro limite: usar a versão medida de C ("a per-fold rebuild from training users only changed the results by at most $0.33$ Acc@10 and $0.29$ macro-F1 across three datasets at one seed") em vez da promessa retirada de B ("A planned follow-up…"), **e reformular o remate da referência cruzada**: a qualificação registada na secção apontada é sobre a metade da **categoria**, não da região.

**3.46 [AUT — D4] Contagem de limites.** Três opções em §7. Enquanto D4 não for respondida, não fechar os bullets 2 e 3 da introdução nem o mecanismo de 3.35(c).
*Facto que decide:* controlo emparelhado por capacidade corrido em 2026-08-13 — na Califórnia um modelo dedicado com 97,4% do orçamento do conjunto (5.014.942 contra 5.151.189) faz 64,910 contra 64,5034, **+0,406, p=0,010, 5/5 folds**. O Texas não tem braço de paridade. `CAMERA_READY.md §D4` marca a opção "não dizer nada" como **indisponível**.
*Frase que hoje é falsa em B:* "a capacity-matched dedicated region model … has not been run" (`07_discussion.tex:105-109`), e "several times the size" que é 1,34× a 2,36×.

**3.47 [MEC] Confirmar que a cláusula dos quilómetros morre.** A base C já a substituiu por "the geographic error … requires the per-visit predictions that the evaluation path does not retain". Verificar que não sobreviveu no ficheiro final.
*Porquê:* os 3–8 km / 17–176 km foram produzidos na receita **campeã v17**, no motor `check2hgi_dk_ovl`, na época diagnostic-best (`analysis/near_miss_RESULTS.md:14-16`) — é uma medição v17 viva dentro de um texto v18.
*Teste:* `grep -n "kilomet" "articles/[mobiwac]/src/sections/07_discussion.tex"` → 0.

### Bloco J — §8 Conclusão

**3.48 [MEC]** Manter a base B. Substituir `$3.5$`→`$3.55$` e `$3.0$`→`$3.06$`.
*Verificação:* 76,54 − 72,99 = 3,55 (região, FL, sobre STAN); 37,55 − 34,49 = 3,06 (categoria, FL, sobre POI-RGNN). **Esta é a única secção do artigo onde estas margens aparecem.**

**3.49 [AUT — D5]** Separar os eixos: apagar "and elsewhere its cost stays inside the margin: … none exceeds two points" e a cláusula final "…on either task"; escrever região e categoria em separado.
*Evidência:* `CAMERA_READY.md §5` nomeia a conclusão como um dos dois sítios com fuga entre eixos; a própria §6 do artigo recusa a transferência quarenta linhas antes.

**3.50 [ESC]** Acrescentar a magnitude com os valores literais que o artigo já imprime: "…at every dataset and in every fold, by $+0.23$ to $+6.29$ macro-F1 points." **Não** usar "a quarter of a point" de C (arredonda 0,23 para cima).

**3.51 [ESC]** Levar as quatro mensagens de commit que governam este parágrafo para um comentário (ou para o `ERRATA.md`): `5a22eeaa`, `918b1d7a`, `94fb6a66`, `7b0b7464`. Nada no `.tex` regista porque é que a frase tem a forma que tem — foi estreitada quatro vezes em catorze horas.

### Bloco K — Tabelas

**3.52 [MEC]** `tables/tbl1_datasets.tex`: acrescentar a cláusula corpo-vs-filtro à legenda (texto em §2.2). Nenhum número muda.

**3.53 [MEC]** `tables/tbl2_substrate.tex`: repor `\providecommand{\sd}` e a coluna `±` com os seis pares de valores listados em §2.2.
*Porquê:* é o único sítio onde `src/` (v17) e a dissertação estão **à frente** de `src_fix`. E é sustentante: o Δ de Florida (+0,23) cabe dentro do próprio sd (0,42).

**3.54 [MEC]** `tables/tbl3_results.tex`: a nota de rodapé da cobertura da busca (parte de 3.36). **Não tocar** no comentário de proveniência do POI-RGNN nem nos números 34.49 / 33.03 / 31.78.

**3.55 [AUT — DP-8]** Ordem das linhas (contagem de regiões vs contagem de check-ins) e convenção de ênfase (negrito só para melhoria sobrevivente a Holm, como no artigo; ou negrito/sublinhado por magnitude com ↑ e ≈ a carregar a estatística, como na dissertação — errata_scope linha 5). Decide também a legenda da Figura 4.

### Bloco L — Leis, ponteiros e fecho

> ### ⭐ 3.62b [MEC] — O PORTÃO ANTI-v17. É o item de maior valor do plano (Fable, 2026-09-06)
>
> O `§8` classifica *"um número v17 reentrar num artigo publicado"* como risco 3 — e era o **único
> risco sem teste**. O 3.63 dizia "validar por um caminho diferente" e não nomeava ferramenta
> nenhuma. Uma frase mal escrita um revisor apanha; **um número v17 publicado só se descobre quando
> alguém subtrai duas colunas** — que é exactamente a falha que produziu esta pasta.
>
> **Script, ~30 linhas, precondição de 3.64 E da nota aos chairs (D1):**
> 1. `pdftotext main.pdf -` e afirmar que **nenhuma** cadeia da lista de nunca-citar do
>    `CAMERA_READY §4` sobrevive;
> 2. afirmar que **todas** as células da escada do `CAMERA_READY §3.1/§3.2/§3.3` aparecem, **com o
>    sinal certo**;
> 3. sobre os `.tex` **com os comentários incluídos** (é lá que vivem os cinco blocos v17 do item
>    3.41, um dos quais é uma instrução escrita chamada *"Honesty rules (do NOT relax)"*), afirmar
>    zero ocorrências de:
>    `63.32|77.05|54.65|28.09|Honesty rules|has not been run|kilomet|fifth of a point`
>
> ⚠ **`fifth of a point` está na lista por uma razão medida.** O teste do 3.9 dizia "→ 0, ou 4, nunca
> intermédio". ✔ Contei: são **9 ocorrências em 6 ficheiros** (`main.tex:84`, `01:35`, `06:44`,
> `06:83`, `07:22`, `08:11`, `fig4_deltas.tex:10`, mais 2 em comentários) — não 4. Depois do porte a
> contagem aterra em 3–4 **por coincidência**, portanto **o teste passava a verde com o abstract por
> tocar**. Um teste que aceita o estado de falha é pior do que nenhum.

**3.56 [MEC] `articles/[mobiwac]/GLOSSARY.md`:** corrigir quatro sítios — a linha do "seed" (`:127`, hoje factualmente falsa; usar a redacção exacta de `ERRATA.md:245-261`, não uma paráfrase) e as três ocorrências dos pesos 0.75/0.25 (`:310`, `:356`, `:394`, esta última prescreve a equação). Estender o banner para nomear §3, §9.2 e §9.3, não só §1 e §6.

**3.57 [MEC] `articles/[mobiwac]/ERRATA.md`:** reescrever o bloco de guarda do cabeçalho — o estado ("submitted / under review" contra a aceitação de 2026-08-26, `:4`), a escada v17 (`:16-18`) e a seta de re-sync invertida (`:20`, aponta de `src/` para a dissertação quando a dissertação está um mês à frente).

**3.58 [MEC] `articles/[mobiwac]/CLAUDE.md`:** corrigir a linha do ledger "FL region cell | FL +0.57 stays a beat" (`:151`) — sob v18 é um défice resolvido dentro da margem (−0,16). Corrigir as duas contagens de bibliografia (`:33` "37", `:266` "31 of 38") para o medido (32 renderizadas de 46 entradas), com o comando ao lado.

**3.59 [MEC — lado da dissertação] `articles/dissertacao/AGENT_GUARDRAILS.md`:** repontar §2 N1 e §3 C1 de `articles/[mobiwac]/PAPER_PLAN.md §3` + `[mobiwac]/CLAUDE.md §3` para `CAMERA_READY.md §3` e `§5`.
*Porquê:* `CAMERA_READY.md §7.1` nomeia esses dois ficheiros como o caminho mais provável para um número v17 fugido chegar a um artigo publicado — e o ponteiro está dentro do âmbito da dissertação.

**3.60 [MEC] Repontar os seis documentos que citam `src_fix`,** antes de a árvore desaparecer: `dissertacao/CLAUDE.md:231`, `wrapup/REVISAO_BANCA_PDF.md:1107,:1159`, `wrapup/evidence/README.md:14`, `wrapup/REVISION_PLAN.md` (nove sítios), `wrapup/NEW_VERSION.md:785`. E `NORTH_STAR.md`: `:204` (aponta para a árvore errada), `:206-208` (escada v17 numa instrução viva — **marcar, não reescrever**, pela regra do próprio ficheiro), `:74`, `:210`, `:419` (estado "under review", que a AUT-35 nunca alcançou aqui).
*Teste:* `grep -rn "src_fix" articles/dissertacao --include=*.md` → só ocorrências históricas conscientes.

**3.61 [MEC] `CAMERA_READY.md`:** corrigir `:636` ("sete linhas" → oito, verificado pelo `count_errata_rows.py`), tirar `references.bib` da lista "Já correcto — não tocar" (3.33 toca-lhe), corrigir o ponteiro `dissertacao/CLAUDE.md:190` → `:231`, e acrescentar um **§13** com quatro heranças: a receita de reprodução v18 (`src/configs/canon.py` ainda fixa `DEFAULT_CANON="v17"`; copiar o comando de `cell_joint()` em `docs/studies/closing_data/v18/run_wave.sh`), as três armadilhas de medição do V10 (md5 não prova conteúdo em PDF; comentários LaTeX inflacionam qualquer grep; `git check-ignore` mente em caminho rastreado), a obrigação de declaração de uso de IA, e um ponteiro para `AGENT_GUARDRAILS §4b`.

**3.62 [AUT] Declarar as divergências que a `errata_scope.tex` não tem.** A tabela tem oito linhas e **não é exaustiva**: a sua própria política (`:10-20`) só guarda correcções "demasiado elaboradas para dobrar num artigo em revisão", e essa premissa caiu com a aceitação a 2026-08-26 — o que a linha 8 já diz. Em falta, no mínimo: a construção forward-only, a assimetria das métricas, a cláusula da legenda da Tabela 1, a nota de rodapé da cobertura da busca, a coluna `±` da Tabela 2, a supressão da cascata (a linha do controlo de congelamento foi **removida** em `5be3458b` em vez de acrescentada), a retirada dos quilómetros, e a reformulação do primeiro limite. Corrigir também a linha 1 (descreve um quarto fundamento que está comentado e não aparece no PDF entregue) e a justificação da linha 2 (ainda diz que a frase cita um apêndice — já não cita).

**3.63 [MEC] Fecho: construir e provar.**
```
cd "articles/[mobiwac]/src" && pdflatex -pdf main.tex
grep -Ei "undefined (reference|citation)|There were undefined" main.log      # -> vazio
pdftotext main.pdf - | grep -n "??"                                          # -> vazio
grep -rn "mobiwac:" . --include=*.tex                                        # -> vazio
pdfinfo main.pdf | grep Pages
```
E, por um **caminho diferente do que produziu o ficheiro**: reextrair todos os números da secção acabada e compará-los com `CAMERA_READY.md §3`, não com o ficheiro que se acabou de editar.

**3.64 [AUT] Só depois de 3.60 e 3.63:** remover `src_fix/` e `src_v1/` da working tree (recuperáveis pela tag de 3.3), registando no commit o comando de recuperação.

---

## 4. O CUSTO EM PÁGINAS

> ⚠ **A aritmética anterior deste plano não limitava nada (Fable, 2026-09-06).** Dizia "~970 palavras
> por página marginal", inferido de um único passo 9→10 — o que não é um limite, é uma média de uma
> amostra. **O número que decide é o quanto a página 10 já está cheia.** ✔ Medido com `pdftotext`:
> a **página 10 do `src_fix` tem 577 palavras** (só referências) e a **página 9 tem 1.144**.
> **Folga real até rebentar para a página 11: ≈ 567 palavras.**
>
> **E isso torna o orçamento um portão sobre as decisões em aberto, não um cálculo à parte:**
> o cenário recomendado (+337) mais a `\bibitem` do Menon, o `\IEEEpubid` e os agradecimentos dá
> **≈ +430 a +470** — aterra em 10 páginas com **~100 palavras de margem**. A partir daí, **cada**
> item [AUT] rebenta para 11: D4 opção 3 (+284), D13 Fig. 3 (+45 e um flutuante), DP-3/DP-4 (+429),
> DP-7 (+206). O "cenário C ≈ 11 pp" da tabela abaixo **não é um cenário, é uma violação do tecto**.

**Calibração antiga (mantida por proveniência):** `src` 6.142 palavras de corpo → `src_fix` 7.114 = **+972 palavras**, e o build passou de 9 para 10 páginas. Orçamento: **8 páginas grátis, tecto 10 com taxa por página** (`CAMERA_READY.md §D3`). O submetido tinha 8. O `src_fix` de hoje tem **10** (working tree; o commitado tem 9 e é v17).

| Secção | B (palavras) | C (palavras) | Port recomendado | Δ |
|---|---|---|---|---|
| §1 Introdução | 736 | 722 | C + duas cláusulas de B | **+11** |
| §2 Relacionado | 898 | 893 | C + desambiguação; S9/S7 por escrever | **+0 a +37** |
| §3 Problema | 271 | 271 | nenhum | **0** |
| §4 Método | 789 | 868 | C completo (obrigatório: +43) | **+79** |
| §5 Setup | 1.423 | 1.654 | só correcções (mantém parágrafo de B) | **+45 a +70** |
| §6 Resultados | 1.932 | 2.007 | C sem os flutuantes nem o ponteiro da Fig. 3 | **+75** |
| §7 Discussão | 814 | 980 | C com as compressões de registo | **+60 a +80** |
| §8 Conclusão | 238 | 413 | base B + 4 edições | **+17 a +37** |
| **Total** | **7.101** | **7.808** | | **≈ +337 (intervalo +255 a +410)** |

**Conversão:** +337 palavras ≈ **+0,35 página**. Mais 3 linhas para a nova `\bibitem` (32→33).

### Cenários

| Cenário | Conteúdo | Δ palavras | Δ páginas | Resultado |
|---|---|---|---|---|
| **A — mínimo obrigatório** | só a Fase 1: forward-only, Fig. 1, somas 1.9/2.8, frase do cosseno, cobertura da busca, 3.55/3.06, separação de eixos, Wilcoxon, Menon | **≈ +100** | +0,10 | ~10 pp |
| **B — recomendado** | tudo o que está na §3 sem marca [AUT] | **≈ +337** | +0,35 | ~10 pp |
| **C — B + tudo o que está em aberto** | + quinto limite Q14 (+284) + Fig. 3 (+45 e um flutuante) + freeze e cascata de volta (+429) | **≈ +1.095** | +1,1 | ~11 pp |

**Onde cortar, se D3 mandar 8 páginas** (é preciso largar ~2.200 palavras **mais** o que se acrescentar, logo ~2.540 no cenário B):
1. reconciliação do piso de Markov, §6 — 258 palavras (0,26 pág.) — mas `§5 S2` avisa que sem ela um árbitro lê um piso não aprendido acima de três sistemas publicados e desconta a comparação externa toda;
2. assimetria das métricas, §4 — 36 palavras (é o único item de Fase 2 desta secção);
3. quatro passagens de registo de dissertação na §7 — 90 a 110 palavras;
4. cauda de motivação de mobilidade, §3 — 165 das 271 palavras da secção (**mas** é o que ancora o enquadramento no venue; requer decisão do autor).

**Não medido ainda:** `\IEEEpubid` (consome espaço na página 1) e o bloco de agradecimentos. Medir **depois** de os acrescentar, nunca antes.

---

## 5. LIMPEZA DA PASTA — PROPOSTA (nada executado)

### 5.1 DELETE — 20 ficheiros, todos untracked e confirmadamente gitignored

| Ficheiro(s) | Justificação |
|---|---|
| 5 × `.DS_Store` (raiz, `docs/`, `docs/exemples/`, `src/`, `src_v1/`) — **untracked** | Lixo do Finder; regra em `.gitignore:21` e nos `.gitignore` locais das árvores |
| 3 × `analysis/__pycache__/*.cpython-312.pyc` — **untracked** | Bytecode regenerável; `.gitignore:16` |
| 12 × `main.{aux,bbl,blg,log}` em `src/`, `src_fix/`, `src_v1/` — **untracked** | Resíduo de build regenerável; `src/.gitignore:1`, `src_fix/.gitignore:2` |

**Cuidado:** nesta pasta `*.pdf` **não** está ignorado, de propósito. Nenhum glob largo (`git clean -fdX` variantes, `rm *.???`).

**Além disso, tracked e recuperável por tag, só depois de 3.60/3.63:** as árvores `src_fix/` e `src_v1/`. Nenhuma delas é citada por um `.tex` entregue (só `src/` é). `src_v1` está condicionado a **DP-2** (o seu `VERSION.md` afirma um facto contradito pelo registo).

### 5.2 ARCHIVE — mover para `articles/[mobiwac]/archive/`, com um banner de uma linha

| Ficheiro(s) | Justificação |
|---|---|
| `docs/exemples/` (7 ficheiros, READMEs de proceedings anteriores) | Única referência no repositório é um relatório de rascunho em `tmp/`. Precedente da dissertação: backup verificado **antes** de apagar (`NORTH_STAR.md:91`) |
| `docs/BEST_PAPERS_ANALYSIS.md`, `docs/SOURCES.md`, `docs/SUBMISSION_CHECKLIST.md` | Material de preparação da submissão; encerrado com a aceitação |
| `PLAN_8PAGES.md` | A sua condição de falha ("src_v1 stays the submission") nunca disparou — o corte para 8 pp resultou no mesmo dia. Dobra em **D3** |
| `CLOSER_HANDOFF.md`, `IMPROVEMENTS_BACKLOG.md` | Dobram em `CAMERA_READY §9` |
| `EDAS_SUBMISSION.md` | Dobra em `CAMERA_READY §10` + Fase 4. **É o documento que refuta o `src_v1/VERSION.md`** — arquivar, nunca apagar |
| `J1_JOINT_SCORE_RUNBOOK.md` | Dobra no parágrafo da convenção de `CAMERA_READY §3` |
| `RELATED_WORK_TRIAGE.md`, `REVIEW_GERMANO.md` | Proveniência pré-submissão |
| `PAPER_PLAN.md` | **Só depois de 3.59.** Hoje é citado por infra-estrutura viva de revisores da dissertação (`AGENT_GUARDRAILS.md:72`, `reviewers/07_claim_honesty_auditor.md:26`, `reviewers/README.md:106`). As cinco proibições ainda vivas da §3 dobram em `CAMERA_READY §4` primeiro |

**Aviso registado pelo próprio projecto:** `AGENT_GUARDRAILS §4b` diz que não se arquiva uma auditoria com base na tabela de resultados dela própria. Dobrar as cláusulas vivas → verificar que chegaram → só então mover.

### 5.3 KEEP — nada disto se toca

| Item | Porquê |
|---|---|
| `src/` | A árvore única. **E** é citada por `.tex` **entregue** da dissertação em ~20 sítios e por ~30 linhas de proveniência da bibliografia entregue |
| `CAMERA_READY.md` | O ledger de registo (untracked hoje — item 3.1) |
| `GLOSSARY.md`, `ERRATA.md`, `CLAUDE.md` | Lei de escrita, ledger de divergências, router |
| `analysis/` (scripts + RESULTS.md + JSON) | Proveniência: `07_discussion.tex` de **ambas** as árvores cita `analysis/{near_miss,shortlist_compactness}_` num comentário de chaveta que um grep por nome de ficheiro não apanha |
| `science/` (4 ficheiros) | Registo de decisão dos desvios D-1..D-4 do protocolo estatístico; a alegação S10 ainda está aberta contra ele |
| `review/` (22 ficheiros, inclui o gate v18 **untracked**) | Registos de revisão; o de 2026-08-11 é o único artefacto de revisão da era v18 |
| `archive/` | Já é o destino |
| `docs/MOBIWAC_CONFERENCE_GUIDE.md` | Única fonte dos nomes dos chairs, de que depende a decisão bloqueante **D1** |
| `mobility/*.pdf` | Substrato de verificação de `moura2025mobilityaware`, nomeado pelas **duas** bibliografias entregues |
| `MOBILITY_PLAN.md` | Citado por **código de treino vivo**: `src/training/runners/mtl_cv.py:859`, `:2207`, `mtl_eval.py:211` |
| `BRIDGING_METRICS.md`, `MOBILITY_SCIENCE_BRIDGE_PLAN.md` | Citados por `.tex` **entregue** da dissertação; o segundo é a fonte de um defeito removido — o antídoto cita o veneno |

**Fora da pasta, a contabilizar antes de declarar a fusão feita:** existem cópias completas das três árvores em `.claude/worktrees/majority-class-floor/articles/[mobiwac]/` (branch `worktree-majority-class-floor`, último commit 2026-08-25) e uma cópia do capítulo em `.temp/term_audit/sab/chapters/5_mobiwac/`. Apagar na `main` não lhes toca; um merge posterior ou uma auditoria por grep ressuscita-as.

---

## 6. DOCUMENTOS QUE FICAM

Quatro documentos de topo, contra os quinze de hoje.

| Documento | Papel | O que dobra nele |
|---|---|---|
| `CLAUDE.md` | **Router fino**: o que é a pasta, mapa dos documentos, as linhas ainda vivas do ledger §3, e o ponteiro para o `CAMERA_READY`. É o entrypoint auto-carregado — **não pode ser apagado sem substituto** | as linhas vivas de `PAPER_PLAN §3`; o mapa de dados da §2b reescrito sobre as fontes v18 |
| `CAMERA_READY.md` | **O estado de registo**: números, veredictos, decisões, plano de edição, armadilhas, backlog, e o novo §13 | `PLAN_8PAGES` → D3; `CLOSER_HANDOFF` + `IMPROVEMENTS_BACKLOG` → §9; `EDAS_SUBMISSION` → §10 e Fase 4; `J1_JOINT_SCORE_RUNBOOK` → convenção da §3; as cinco proibições de `PAPER_PLAN §3` → §4 |
| `GLOSSARY.md` | **A lei de escrita do artigo**, corrigida por 3.56 | — |
| `ERRATA.md` | **O ledger de divergências nas duas direcções**, espelho do lado do artigo da `errata_scope.tex`; cabeçalho reescrito por 3.57 | as mensagens de commit que governam a conclusão (3.51) |

Mantêm-se como subpastas de proveniência: `analysis/`, `science/`, `review/`, `archive/`, `docs/` (reduzida ao guia da conferência), `mobility/`.
Mantêm-se por citação externa: `MOBILITY_PLAN.md`, `BRIDGING_METRICS.md`, `MOBILITY_SCIENCE_BRIDGE_PLAN.md`, e `PAPER_PLAN.md` até 3.59.

---

## 7. DECISÕES QUE FALTAM

### Bloqueiam a escrita

**D4 — o controlo emparelhado por capacidade.**
*Facto:* na Califórnia um modelo dedicado de região com 97,4% do orçamento do conjunto (5.014.942 contra 5.151.189) faz 64,910 Acc@10 contra 64,5034, **+0,406, p=0,010, 5/5 folds** (2026-08-13). O Texas não tem braço de paridade.
*Opções:* **(0) correr primeiro o braço de paridade do Texas** (~2 h/semente) e declarar o que os
dois datasets suportarem — é a opção (2) do `CAMERA_READY §D4` e o item **A1** do backlog, que o
ledger marca como *"o de maior valor da lista"*. ⚠ Uma versão anterior deste plano **perdeu-a**,
transformando uma decisão científica numa decisão de redacção (Fable, 2026-09-06); (1) declarar,
limitado à Califórnia, e reescrever as vitórias de região como resultado de apoio com o limite de
capacidade dito no mesmo fôlego; (2) manter o limite de B corrigido (tirar "has not been run", que o autor sabe ser falso; trocar "several times the size" pelo medido 1,34×–2,36×), custo 79 palavras; (3) adoptar o quinto limite da errata Q14, custo 284 palavras (0,29 pág.).
*Indisponível:* apagar o limite e não dizer nada — é o que a dissertação faz hoje, e `§D4` marca-o como indisponível para o artigo.
*Consequência:* decide se **alguma** vitória de exactidão sobrevive, e por isso trava os bullets 2 e 3 da introdução, o mecanismo da §6 e a contagem de limites da §7.

**D5 — o eixo da categoria.**
*Opções:* (A) "não resolvidas em nenhuma direcção" (o que `src_fix` faz hoje); (B) o limite derivado, "bounded within half a point of zero" (o que a dissertação traz; **recomendado** pelo ledger) — **exige** imprimir os seis IC na §6; (C) registar agora uma margem (não disponível a posteriori).
*Consequência:* muda **quatro** sítios (`01_introduction`, `07_discussion`, `08_conclusion`, `main.tex:85-87`), não dois.

> **Correcção a este plano (2026-09-06), levantada pelo `mobiwac-writer` e verificada por mim.** Uma
> versão anterior deste item mandava escrever *"bounded within half a point of zero"* e **não**
> *"equivalent to zero"*, com o argumento de que num eixo sem margem registada "equivalent" seria
> linguagem de equivalência indevida. **Estava errado, e a lei já responde ao argumento na mesma
> frase.** A `WRITING_LAW.md:208-211` diz que o limite é *"derived, not chosen — two one-sided tests
> at margin d pass exactly when the 90 % CI falls inside ±d, so it is read off the intervals. Because
> nothing is chosen there is no post-hoc margin to justify"*, e a `:228` fixa a superfície como
> *"equivalent to zero within half a point"*, **"a bounded-magnitude claim whose bound is read off
> the intervals rather than chosen, never as a match"**. Não é linguagem de equivalência escolhida: é
> um limite de magnitude derivado, e a lei diz isso explicitamente.
> **A superfície de registo é `equivalent to zero within half a point`, e é essa que se usa.**
> O que continua verdadeiro, e é da própria lei (`:214-219`): **meio ponto é falso no eixo da
> região**, onde o limite derivado é 1,372 pp simultâneo (1,287 por dataset) — escrever "meio ponto"
> ali é uma sobre-afirmação de três vezes. Nunca citar o limite de um eixo no outro.

**DP-1 — WRITING_LAW contra GLOSSARY do artigo.**

> ### ⚠ ACTUALIZAÇÃO 2026-09-06 — o autor decidiu; falta só a confirmação de uma linha
>
> A sessão `mobiwac-writer` pôs o conflito ao autor exactamente nestes termos (citando a cláusula, e
> notando que o TOST licenciaria "matches") e cita a resposta dele, literal:
>
> > *"Vale a lei da dissertação, WRITING_LAW governa o camera-ready, apesar de já termos feitos os
> > testes estatiscos e podermos usar o matches. Mas como eu disse o agent mobiwac vai te filtrar,
> > auditar e te passar os conteudos e vcs vão alinhado"*
>
> **Ou seja: WRITING_LAW ganha, "matches" sai, e sai por escolha informada e não por lapso.** A
> mesma citação corrobora a divisão de papéis nas palavras que o autor usou comigo.
> Fica pendente **uma linha** de confirmação do autor — não por dúvida, mas porque a decisão tem
> duas consequências que só ele autoriza (a seguir).
>
> **Consequência boa, e é grande:** o texto do Capítulo 5 **já está escrito segundo a WRITING_LAW**.
> Toda a superfície de veredicto lá diz *"stays within the two-point margin"*; a única ocorrência de
> "match" em todo o capítulo é a recusa em `06_results.tex:230`. **A regra herda-se com a migração
> em vez de custar uma reescrita** — a CONV-8 deixa de ser trabalho e passa a ser preservação.
>
> **Consequência a autorizar:** o `WRITING_LAW.md:8` passa a contradizer a decisão do autor
> (continua a entregar a prosa do capítulo MobiWac ao GLOSSARY do artigo). Ninguém lhe toca por
> indicação de um par: entra como item do autor, no mesmo commit da primeira edição de prosa, como o
> `CAMERA_READY §7.1` já exige para os ficheiros de lei. O `GLOSSARY.md:70` do artigo e o
> `CLAUDE.md:148` (que manda "matches" verbatim no abstract) precisam da mesma correcção.
>
> **A superfície a rever no `src_fix`, se alguma secção acabar baseada nele** — quatro sítios, não
> dois: `main.tex` (abstract), `sections/06_results.tex:87` (*"non-inferior match (TOST, ±2 pp)"*),
> `:101`, e `tables/tbl3_results.tex:26` (a legenda). Legais e a preservar: `06_results:105`
> (negação), `01_introduction:33` (já na forma nova) e todo o `matched folds`, que é vocabulário de
> emparelhamento e não de veredicto.

*Enunciado neutro do conflito, como estava antes da decisão (mantido para proveniência):*
- `articles/dissertacao/WRITING_LAW.md:226` **proíbe** "matches" nos dois eixos e manda "stays within the two-point margin" (região) e "equivalent to zero within half a point" (categoria); o checklist repete em `:388`.
- `articles/[mobiwac]/GLOSSARY.md:70` **licencia** "matches" como o verbo de equivalência ligado ao TOST, e `:73` prefere a forma curta; `articles/[mobiwac]/CLAUDE.md:148` **manda-a verbatim no abstract**.
- **Nenhuma reversão está registada.** Procurado sob: `AUT-3[6-9]`, `AUT-4[0-9]`, "paper GLOSSARY wins", "GLOSSARY do artigo", "reverte", "inverteu", "reversão", "revogad", "passa a ganhar", "prevalec", "preced", "conflit", "wins", "ganha", e as datas 2026-09-05 / 2026-09-06 — nos oito documentos de lei da dissertação, em `wrapup/`, `src_utils/`, nos cinco documentos do artigo, e em `git log --all --since=2026-09-01`. O **único** commit que tocou algum dos três ficheiros é `42175509` (2026-09-02), e tocou **uma linha** de `articles/dissertacao/GLOSSARY.md` (a promessa Wilcoxon).
- **Pode ser um erro de categoria:** `WRITING_LAW.md:3` limita-se a "every sentence written in this dissertation" e a sua ressalva de `:7-8` entrega o *capítulo* MobiWac ao GLOSSARY do artigo; nenhuma das duas cláusulas alcança o camera-ready, e `CAMERA_READY.md:26-28` já diz que no artigo o GLOSSARY ganha nas regras de escrita.
- **O statu quo do texto que vai ser portado é WRITING_LAW:** `dissertacao/.../06_results.tex:229-231` escreve "by the bound their intervals support rather than as matches" e `08_conclusion.tex:25-27` traz as duas superfícies. Em todo o Capítulo 5 sobrevivem 2 usos de "match", um deles a recusa.
*A pergunta a fazer ao autor, na forma que os factos suportam:* o camera-ready vai ser rebaseado no Capítulo 5, cujas frases de veredicto estão escritas segundo a WRITING_LAW; o GLOSSARY do artigo e o ledger do abstract pedem "matches it (statistically, within two points)". **As frases portadas mantêm a superfície da dissertação, ou voltam à do artigo?**
*As quatro frases que mudam:* `src/main.tex:82-83`, `sections/06_results.tex:87`, `:101`, `:105`.
*Consequência processual:* seja qual for a decisão, é uma divergência declarável — linha nova na `errata_scope.tex` (se ganhar a superfície do artigo) ou linha nova no `ERRATA.md` (se ganhar a da dissertação).

### Decidem conteúdo

**DP-3 — a cascata.** Manter o ponteiro da §2 **e** o parágrafo da §6 (`src/sections/06_results.tex:155-163`), ou apagar ambos como a dissertação fez a 2026-08-06?
*Evidência:* `§5 S5` marca NÃO RESOLVIDA — medida no substrato v17 e nunca recorrida, "a única medição com peso de alegação a atravessar a fronteira v17→v18 sem re-derivação". O ponteiro de B já está pendurado: a secção que aponta nunca menciona uma cascata (verificado sob dez redacções nas duas árvores).
*Consequência de apagar:* perde-se a única resposta do artigo à objecção cadeia-versus-paralelo do CSLSL. De manter: uma medição v17 a carregar peso de alegação num artigo v18. **Nota:** `CAMERA_READY` é **silencioso** sobre se o parágrafo fica (zero ocorrências de "cascade/chain/CSLSL"), logo isto não está decidido em lado nenhum.

**DP-4 — o controlo de congelamento da região.** A dissertação apagou-o; `§5 S6` marca-o ENFRAQUECIDO (no substrato entregue a via de região está desligada do codificador check-in por construção, logo "sem transferência" é uma tautologia lida como achado). B já concede que o braço "belongs to a separate study" e recusa reportar as pontuações. **Recomendado: sair.** Custo de o repor: 248 palavras.

**DP-5 — o controlo de concatenação (Q13).** B declara o braço fora de escala e recusa pôr os números ao lado da Tabela 2; C (2026-09-02, aprovado pelo autor, errata linha 6) põe os dois triplos lado a lado sem afirmar rácio, limitado ao eixo da categoria. Não são reconciliáveis como estão.
*Risco de C:* um árbitro calcula 123 / 66 / 348 por cento e lê o controlo como mostrando que a representação quase nada acrescenta na categoria. *Risco de B:* perde-se a correcção Q13 que o autor já aprovou.

**DP-6 — a frase do prior de transição de região.** Está no texto **aceite** e em B ("A version built from the whole dataset inflated region accuracy by 13 to 27 points. Only the HMT-GRN comparison model uses this prior"). Foi apagada da dissertação em `467753c7`, no mesmo commit que trazia a instrução do autor de 2026-08-04 — "não mencionar o episódio do vazamento em lado nenhum" — instrução que foi **medida e delimitada** a cláusulas só da dissertação. Esta é um vazamento diferente e está no artigo aceite.
*Consequência de apagar:* remove-se a revelação de que um prior sobre o dataset inteiro inflaccionaria o baseline de região primário. Um árbitro que conheça o HMT-GRN pergunta.

**DP-7 — onde vai a cobertura da busca de configuração.** Setup (como na dissertação, 206 palavras) ou só as três correcções em Resultados/Discussão/rodapé da Tabela 3 (como o plano de registo manda, ~10 palavras)? **Recomendado: o segundo**, por causa do orçamento e porque em C o parágrafo parte o argumento de integridade ao meio.

**DP-8 — Tabelas 2 e 3: ordem das linhas e convenção de ênfase.** Ver 3.55. Arrasta a legenda da Figura 4.

**DP-9 — evidência pré-v18 a defender um build v18.** Duas instâncias: a auditoria de transdutividade do parágrafo de integridade (substrato v14, semente 0) e a frase da geometria (silhouette/pureza de um CSV de 2026-06-24 sobre o motor v14, agregado por POI enquanto a prosa diz "per-visit vectors").
*Opções:* declarar o escopo numa frase; correr A3 no build forward-only; ou retirar a afirmação.
*Consequência:* é a linha mais atacável do artigo, e é atacável por quem leia a dissertação ao lado do artigo — `§5 S14` chama-lhe "a resposta do artigo à objecção de vazamento que afundou a submissão anterior".

**D13 — escopo do silhouette e destino da Figura 3.** 0,55/0,79 (AL/AZ/FL, prosa do artigo) ou 0,57/0,78 (cinco estados, dissertação e **o asset construído**)? E o `main.tex:132` diz 0,53, que não concorda com nenhum dos dois. Repor a figura custa ~0,25–0,30 página.

### Decidem processo

**D1 — a nota aos chairs** (bloqueante). Único sítio com os nomes: `docs/MOBIWAC_CONFERENCE_GUIDE.md:192-193`.
**D3 — 8, 9 ou 10 páginas.** Ver §4. Medir só **depois** de acrescentar `\IEEEpubid` e os agradecimentos.
**D10 — o pacote de reprodução.** O ramo público leva só artefactos v17, e o repositório fixa `DEFAULT_CANON="v17"`; o artigo imprime uma promessa de código na página 1.
**DP-2 — `src_v1/`.** O seu `VERSION.md` afirma ser o que foi para o EDAS; o `EDAS_SUBMISSION.md`, um dia mais novo, di-lo de 8 páginas e com o upload pendente. Opções: corrigir/anotar o `VERSION.md` e manter; arquivar; apagar (recuperável por `git show 57235720:...`). **Recomendado: corrigir primeiro, decidir depois** — é prova numa matéria que `§10.2` dá por não resolvida.
**DP-10 — declaração de uso de IA.** `AGENT_GUARDRAILS §6` regista a Portaria CNPq nº 2.664/2026 e a exigência dos grandes editores. O plano do camera-ready não tem nenhum item disto; a única ocorrência de "CNPq" no `CAMERA_READY` é um placeholder de financiamento.
**DP-11 — confirmar o caminho alvo.** O autor disse "uma só `src/`". Consequência aceite: ~20 ponteiros de `.tex` entregue passam a resolver para o camera-ready e não para o texto doador. Mitigação: 3.3 (tag) + 3.5 (README) + uma linha no `ERRATA.md`. Se preferir outro caminho, diz-se agora.

---

> ### ⚠ 3.65 [MEC] — REPONTAR OS DOIS DOCUMENTOS DE ENTRADA (Fable, 2026-09-06)
>
> O 3.60 procura ponteiros `src_fix` só em `articles/dissertacao`. **Faltam os do próprio artigo**, e
> são os dois ficheiros que um agente lê primeiro: `[mobiwac]/CLAUDE.md:13` diz *"o paper de
> referência é `src_fix/`"*, o `CAMERA_READY.md` menciona `src_fix` **27 vezes** (incluindo o §0
> item 3, a recomendação do §2 e o §8 inteiro, cujos caminhos são *"relativos a `src_fix/`"*), e o
> `ERRATA.md` duas. Depois do 3.64 os três apontam para uma árvore apagada.
> *Teste:* `grep -rn "src_fix" "articles/[mobiwac]" --include="*.md"` devolve só linhas históricas
> marcadas como tal.

> ### ⚠ 3.66 [ESC] — UM DONO POR FICHEIRO DURANTE A MIGRAÇÃO (Fable, 2026-09-06)
>
> Entre 3.4 e 3.63 os `src/sections/*.tex` são um híbrido, e três sessões estão a trabalhar nesta
> pasta ao mesmo tempo. O plano não diz quem pode tocar em quê e quando. **Regra do projecto: um só
> dono por ficheiro.** Proposta: o `mobiwac-writer` é dono exclusivo de `src/**` a partir do 3.4;
> esta sessão não escreve lá, só audita e propõe.

---

## 8. RISCOS — do mais grave para o menos

> ### ⚠ Re-ordenação proposta (Fable, 2026-09-06)
>
> O topo actual (1 e 2, os ficheiros fora do git) está certo como **sequência** — fecham com dois
> commits — mas errado como **gravidade**. Correcções:
> - **O risco 3 (um número v17 num artigo publicado) é o mais grave e era o único sem teste.** Passa
>   a ter o portão do item 3.62b. É o que justifica a ordem.
> - **O risco 10 (orçamento de páginas) está subvalorizado.** Não é uma preocupação, é um **tecto
>   duro com ~100 palavras de margem** no cenário recomendado, e é ele que decide quais das opções
>   [AUT] estão sequer disponíveis (ver §4).
> - **Três riscos em falta:** (a) o build submetido está por identificar e é a base da nota aos
>   chairs (item 3.0); (b) o prazo do camera-ready é **desconhecido** (`docs/MOBIWAC_CONFERENCE_GUIDE.md:38`
>   diz "TBA em 19/06/2026" e não há nada mais recente) e os pareceres não existem no repositório,
>   portanto a rota (c3) da D1 pode acabar em **retirada** — e o plano não diz quais itens mantêm
>   valor de qualquer forma (Bloco 0, tags, 3.56–3.61) e quais são contingentes (Blocos B–K);
>   (c) autoria concorrente (item 3.66).
> - **A instrução do autor de levar o capítulo inteiro AUMENTA os riscos 7 e 9** (cópia literal, e
>   referências que não resolvem), porque passa a haver menos filtragem manual. As CONV-5, 3.41 e
>   3.62 passam a ser as únicas guardas.

1. **Perder o `CAMERA_READY.md` e o gate de revisão v18.** Estão fora do git. Um `git clean -fd` ou um clone limpo destrói-os **de forma irrecuperável**, e com eles a única escada de veredictos v18, a lista D1–D20 e a ronda de falsificação. Uma sessão de limpeza é exactamente quando esses comandos se escrevem. → **item 3.1, antes de tudo.**
2. **Perder o único PDF v18 e regredir as leis.** `HEAD` guarda o build v17 (9 pp, `md5 6a32e1e3…`) dentro da pasta chamada "artigo de registo"; os banners STOP que mandam ler o `CAMERA_READY` estão por commitar. Um `git checkout -- .` repõe leis v17 e o PDF v17 num só gesto. → **item 3.2.**
3. **Um número v17 reentrar num artigo publicado.** Os vectores estão todos identificados e nenhum está fechado: `PAPER_PLAN §3` e `[mobiwac]/CLAUDE.md §3` (para onde a `AGENT_GUARDRAILS` ainda encaminha os agentes da dissertação), quatro cláusulas dentro das secções que o próprio banner do GLOSSARY certifica como vivas, a escada v17 no `ERRATA.md`, o `NORTH_STAR §4`, e — o pior — os cinco blocos de comentário v17 dentro do `06_results.tex` da dissertação, um dos quais é uma instrução escrita chamada "Honesty rules (do NOT relax)". → **itens 3.41, 3.56–3.59.**
4. **Publicar uma frase que o autor sabe ser falsa.** `src/sections/07_discussion.tex:105-109` diz hoje que o controlo emparelhado por capacidade "has not been run". Correu a 2026-08-13 e ganhou ao modelo conjunto na Califórnia. → **D4.**
5. **Referências e citações que não resolvem, e o aviso morre no log.** O projecto já teve um "Appendix ??" com o aviso por ler. Os pontos concretos: `tab:mobiwac:representation` que **não** é um strip (vai para `tab:substrate`); `\label{sec:problem}` que se for renomeado parte três `\ref` em ficheiros que ninguém abre; `\cite{Lim2022}`, `\cite{nash}`, `\cite{velickovic2019deep}` que foram removidos de propósito; `\ref{fig:mobiwac:embquality}` que aponta para um flutuante que o `main.tex` não faz `\input`. → **CONV-2/3, item 3.63.** Validar pelo log e pelo `pdftotext`, nunca pelo diff.
6. **Medições v17 vivas dentro de texto v18.** Três, todas ainda em `src_fix`: os quilómetros do near-miss (receita campeã v17, época diagnostic-best), a magnitude "about two points" do CTLE (sob v18 são 3,68), e as constantes da geometria (motor v14). É exactamente a classe de defeito que afundou a submissão anterior. → **3.35(a), 3.47, DP-9.**
7. **Copiar a dissertação literalmente.** Traz `\input` de tabelas que duplicam a Tabela 1/2/3, ambientes `figure` que duplicam as Figuras 1/2/4 com labels multiplamente definidas, `\includegraphics` sem `width` a transbordar uma coluna IEEE, caminhos `figures/mobiwac/*.pdf` que não existem, e um `\usepackage{newtxtext}` dentro de um build IEEEtran. → **CONV-5.**
8. **Apagar `src_fix` antes de repontar.** Seis documentos da dissertação citam caminhos `src_fix`, dois deles por ficheiro e linha, e o `dissertacao/CLAUDE.md:231` chama-lhe "o artigo de registo". → **3.60 antes de 3.64.**
9. **Tratar a `errata_scope.tex` como exaustiva.** Tem oito linhas e a sua própria política só guarda o que era "demasiado elaborado para dobrar num artigo em revisão" — premissa que caiu com a aceitação. Uma das maiores divergências desta pasta foi feita **removendo** uma linha (`5be3458b`), não acrescentando. Um agente que rederive a lista de portes só a partir dela reproduz os números submetidos "+28 a +40". → **3.62.**
10. **O orçamento de páginas.** O build está em 10 contra 8 grátis; o porte recomendado acrescenta ~0,35 página e as decisões em aberto podem acrescentar mais 1. Cortar de 10 para 8 exige largar ~2.540 palavras, e os candidatos naturais são precisamente as passagens que o ledger diz que protegem o artigo (o piso de Markov, a cauda de mobilidade). → **D3, medida depois do `\IEEEpubid`.**
11. **Confirmar o merge relendo o ficheiro que se acabou de produzir.** Isso prova só que a edição foi aplicada. Validar por um caminho diferente: o log de build para as referências, o `CAMERA_READY §3` para os números. → **3.63.**

---

### Notas de proveniência deste plano

- Tudo o que está marcado **[MEC]** foi verificado por pelo menos um auditor com o caminho e a linha; tudo o que está marcado **[AUT]** está explicitamente por decidir no registo, não por investigar.
- As três afirmações verificadas nesta sessão (✔) são: as três árvores existem e não há nenhuma tag MobiWac; os cinco ficheiros críticos estão untracked/sujos; `97f01a50` é de 2026-07-11 00:28:46 -0300 e não houve deriva no material MobiWac desde a linha de base das auditorias.
- **Não foi rederivado nenhum número experimental.** Todos os valores de resultados citados vêm de `articles/[mobiwac]/CAMERA_READY.md §3`/`§4`/`§5` ou de `articles/dissertacao/wrapup/evidence/ladder_recompute.json`, pelas auditorias que os recomputaram célula a célula.
- **Os relatórios dos revisores do MobiWac não existem em lado nenhum do repositório** (`CAMERA_READY §10.1`). Todo o juízo deste plano é feito contra o registo interno, sem saber o que os revisores pediram.