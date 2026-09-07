# VEREDITOS.md — o que já está decidido, e não se reabre

> **Para que serve.** Este ficheiro responde por **pergunta**, não por documento. Antes de acusar
> o texto de um defeito, de copiar um número, ou de apagar um ficheiro, procure a pergunta aqui.
> Se ela estiver nesta lista, **a questão está encerrada** e a prova está na linha.
>
> **Ordem de leitura obrigatória:** este ficheiro → `CLAUDE.md` §0 → os documentos vivos →
> `src_utils/_round*`, `_review*`, `_specialists*` **só se alguém pedir explicitamente**.
> Tudo sob um directório com `_` à frente é **rodada encerrada e sem autoridade**: serve como
> registo de proveniência do que foi discutido, nunca como fonte de um facto corrente.
>
> **Como o ledger é escrito.** Cada verdete traz as **paráfrases** pelas quais a pergunta costuma
> chegar (para o `grep` a encontrar), o veredito, a **data** em que fechou, a **prova**, e o
> documento que costuma induzir ao contrário. Não repita o vocabulário do erro fora da linha
> `Induz ao contrário` — cada repetição é mais um acerto falso na busca do próximo agente.

---

## V1 · O capítulo 3 (ou a representação, ou o Check2HGI) tem vazamento de dados?

*Também chega como:* data leak, leakage, vazamento de rótulo, canal transdutivo, o grafo de visitas
consecutivas vê o futuro, a vizinhança inclui a visita seguinte.

**Veredito: existiu, e está FECHADO desde a geração v18. Não é defeito do texto entregue.**
Fechado em **2026-08-13** (auditoria de proveniência); a correcção é anterior.

O `v18` fechou o vazamento de rótulo no grafo de visitas consecutivas: passou a ser **forward-only**
(`src < tgt`), no treino **e** na leitura. Em Alabama o vazamento valia **28,63 macro-F1**, e é por
isso que toda a coluna de categoria se moveu tanto. A região mexeu menos de 2 pp.

**Prova:** `CLAUDE.md` §0.2 · `docs/studies/closing_data/v18/METHODOLOGY.md` ·
a auditoria estudo-a-estudo em `wrapup/open_points/AUDITORIA_PRE_LEAK.md`, que percorre os nove
estudos pré-correcção e diz, por estudo, se a contaminação alcança alguma alegação viva.

**Induz ao contrário:** `src_utils/_review_v1/09_stats_leakage_skeptic_report.md` (2026-07-24).
Relatório de persona, 4.817 palavras, escopo exactamente `src/chapters/3_cbic`. Apoia-se **sete
vezes** numa fonte que já estava morta (ver **V2**). É citado pelo `.tex` entregue como proveniência
de uma decisão deliberada (`src/chapters/5_mobiwac/07_discussion.tex:245`), por isso continua no
repositório — mas as suas conclusões não valem.

---

## V2 · Posso usar o `RESULTS_BOARD.md` como fonte de um número?

*Também chega como:* qual é a fonte de verdade do capítulo 5, onde estão as células da Tabela 3,
o board de resultados.

**Veredito: NÃO. Está morto para a dissertação.** Fechado em **2026-08-20**.

`docs/studies/closing_data/RESULTS_BOARD.md` chama-se a si próprio fonte única de verdade, e é uma
— para a geração **v17**, cujas células de categoria estão infladas em 25 a 45 pontos. Foi tocado
pela última vez em 2026-07-20 e **nunca menciona v18**.

**A fonte correcta** é o próprio ficheiro de tabela, `src/tables/mobiwac/*.tex`: cada valor impresso
carrega a proveniência num comentário ao lado, e é esse comentário que o portão de factos segue.
O mapa completo está em `CLAUDE.md` §0.1.

**Induz ao contrário:** vinte e um ficheiros ainda se apoiam nele, e três deles são **instruções
vivas** que mandam o leitor ir lá conferir. Estão listados em **V9**.

---

## V3 · Este macro-F1 de categoria está certo?

*Também chega como:* 63,56 · 64,51 · 77,05 · 79,85 · o valor de categoria parece alto demais,
os números não batem com o que eu li noutro sítio.

**Veredito: um macro-F1 de próxima-categoria fora da faixa 30 a 38 é número pré-v18. Pare.**

Os valores entregues são **AL 30,59 · FL 37,55 · CA 35,63**. Qualquer coisa na casa dos 60 ou 70
pertence à geração com vazamento.

⚠ **A verificação de faixa protege só o eixo da categoria.** A região mexeu menos de 2 pp entre as
gerações, portanto um número de região **não** se denuncia pelo valor: confira o caminho do ficheiro.

**Prova:** `CLAUDE.md` Regra 2 (§0) · `AGENT_GUARDRAILS.md` §N1.

---

## V4 · Posso apagar `src_utils/_round6` … `_round14`?

*Também chega como:* estas pastas parecem trabalho velho, dá para limpar as rodadas, o `src_utils`
está cheio de lixo.

**Veredito: NÃO. São de carga.** Verificado por três caminhos independentes.

1. `src_utils/check.sh:345` **executa** `_round9/35_wave_a_render_check.py`.
2. `check_audit_claims.py` **lê os `.md` como dados** — tem uma tabela de expressões regulares que
   casa contra `_round9/37_reviewer_gate_round9.md`, `_round9/47_applied_check.md` e outros.
   Apagar um deles faz o portão de auditoria **falhar, não avisar**.
3. O `.tex` entregue cita caminhos de rodada **quarenta vezes**, em comentários de proveniência.

Também não tocar: `_fixtures/check_verify_list/{clean,dirty}/src_utils/_round6/VERIFY_LIST.md` são
cópias-fixture que o próprio `check.sh` compara.

**Prova:** `ACHADOS.md` §A5 · `README.md` §"parecem pastas de trabalho velhas e não são" ·
`CLAUDE.md` linha 150.

---

## V5 · Um `python scripts/train.py --task mtl` reproduz a dissertação?

*Também chega como:* qual é a receita canónica, como reproduzo uma célula, o comando de treino.

**Veredito: NÃO.** `src/configs/canon.py` ainda fixa `DEFAULT_CANON = "v17"`, sobre o substrato
com vazamento.

Para reproduzir uma célula entregue, copie o comando **literalmente** de `cell_joint()` em
`docs/studies/closing_data/v18/run_wave.sh`.

⚠ **O bloco de receita B9 do `/CLAUDE.md` da raiz do repositório tem três gerações de atraso** e é
carregado automaticamente em toda sessão. Ignore-o para qualquer coisa da dissertação.

**Prova:** `CLAUDE.md` §0.2, aviso final.

---

## V6 · A categoria supera o modelo dedicado em todos os seis conjuntos?

*Também chega como:* +5,3…+9,4, category outperforms everywhere, o modelo conjunto ganha em todos.

**Veredito: NÃO. Supera em Florida e só em Florida** (+0,19, Holm p 0,011). As outras cinco são
**não-resolvidas**. Superado em **2026-08-20**.

A região é o eixo que se sustenta: **não-inferior nos seis**, com TX +1,21 e CA +1,06.

**Prova:** banner de `NORTH_STAR.md` linhas 15-27 · marcadores `[SUPERADO 2026-08-20]` na linha 67.

---

## V7 · O ganho da representação ao nível do check-in é +28…+40 macro-F1?

*Também chega como:* o salto da representação, check-in-level bate place-level por quanto.

**Veredito: NÃO. A faixa real é +0,23…+6,29.** Superado em **2026-08-20**.

O `+28…+40` é número de substrato **pré-v18**. Não é uma tese anterior que foi revista: é uma
geração que foi **invalidada**. Não há leitura em que seja citável.

**Prova:** `NORTH_STAR.md` linha 67, marcador `[SUPERADO 2026-08-20]` · `ACHADOS.md` §A4.

---

## V8 · Que estudos pré-v18 contaminam alegações vivas do texto?

**Veredito: seis foram rastreados um a um; um só está exposto.** Fechado em **2026-08-13**.

O critério é limpo: uma medição só é contaminada se **lê vectores**. Quem lê apenas rótulos, o
stream de check-ins, ou o mapa POI-para-região não passa por convolução nenhuma e é imune.

| estudo | veredito |
|---|---|
| `markov_floor_stride1` | **IMUNE** (não lê vectores) |
| `h2_v17_cat_ceiling`, `catx_v17_n20` | **IMUNE** |
| `capacity_matched_stl_cat` | **EXPOSTO** — e a razão do Apêndice G é de outra geração |
| `apxi_v18` | **VÁLIDO** (medido na preparação actual) |
| `baseline_compare` | **VÁLIDO** (o texto declara que rodam nos próprios embeddings) |
| `v18_place_level` | **VÁLIDO por desenho** (é o braço de comparação, e o texto nomeia-o) |

**Prova:** `wrapup/open_points/AUDITORIA_PRE_LEAK.md`.

---

## V9 · O `make check` saiu com código diferente de zero. Fui eu?

**Veredito: provavelmente não. A esteira já sai vermelha por defeito**, e imprime verde em vários
portões enquanto isso. **Compare a assinatura das linhas de portão, nunca o código de saída.**

⚠ **A assinatura depende de `src/build/main.pdf` existir.** Sem ele, o portão do Wave A dá `SKIP`
e 4 comandos do `VERIFY_LIST` falham por falta do render. Com ele, esses portões passam a correr —
e a assinatura muda sem que ninguém tenha tocado no texto. Meça a sua própria baseline antes de
mexer, e compare contra ela.

Assinatura medida **2026-09-06**, com o `main.pdf` de 2026-09-04 presente: `rc=1`, com

- **`FAIL FAB-12 new absent (wanted present)`** no portão do Wave A. As duas formas, a nova e a
  velha, estão **ambas ausentes**: a frase está numa terceira redacção. É a reversão que o autor
  mandou fazer (`_round9/45_author_rulings.md`), e a construção está registada como **decisão do
  orientador, não defeito** (`_round9/42_excellence_r9b.md` §3 item 8). **Não "corrigir" isto sem
  falar com o autor** — o portão está a assinalar uma escolha deliberada que ninguém reconciliou
  com ele.
- 16 alegações registadas como `APPLIED` que não estão no documento: `R8-head`, `R8-head2`,
  `A22-11`, `A23-R6`, `R13-s1pcgrad`, `R13-s2base`, `R13-s2base2`, `RTV-08b`, `R13-aut37{,b,c}`,
  `R13-foldseed{4,6,7,8}`, `R13-mech-soft`.

Sem o `main.pdf` (baseline de 2026-09-02) eram 6 `FAIL` e um `SKIP`; as outras 49 linhas são iguais.

Nunca escreva num commit que a esteira passou sem ter lido a última linha: `AGENT_GUARDRAILS.md`
regista **quatro** ocorrências de mensagens de commit a afirmar `rc=0` sobre execuções que saíram 1.

---

## V9b · Ponteiros pendurados pela limpeza de 2026-09

Dezanove ficheiros de rodada foram removidos: uns por afirmarem números da geração com vazamento,
outros por serem relatórios de portão superados. Alguns eram citados **por outros ficheiros de
rodada**, e essas citações já não resolvem. **É esperado, e não se conserta indo procurar o
ficheiro.**

O caso a conhecer: `_round9/37_reviewer_gate_round9.md:44` cita `reviews/06_number.md`, removido.
Esse relatório **validava as células v17 como correctas** (*"AL 64.51 … CA 77.05 — every cell
matching"*), portanto o seu veredito não vale; a linha da tabela fica como registo de que o portão
correu. O `check_audit_claims.py` valida expressões **dentro** do agregador, não a existência dos
caminhos que ele cita — por isso a esteira não mudou (verificado por A/B).

Tudo continua recuperável: `git show <sha>^:<caminho>`.

---

## V10 · Armadilhas de ferramenta que já custaram caro

| pergunta | veredito |
|---|---|
| `git check-ignore` diz que o ficheiro está ignorado. Está? | **Mente em caminho já rastreado.** Use `--no-index` ou `git status --ignored`. Quase custou o `slide_final.pdf` |
| Dois PDFs com o mesmo `md5` têm o mesmo conteúdo? | **Não prova.** Sem `SOURCE_DATE_EPOCH` cada reconstrução muda o hash. Compare `pdftotext` ou o render |
| O `grep` achou N ocorrências no `.tex`. São todas texto? | **Não.** Comentários LaTeX inflam qualquer `grep`. Filtre as linhas `%` |
| Um `.md` sob `_round*`/`_review*` afirma X. Vale? | **Não.** Ver o cabeçalho deste ficheiro: rodada encerrada é proveniência, não fonte |

**Prova:** `ACHADOS.md` §A6.

---

## V11 · Quanto do numero de categoria vem do grafo TREINADO?

*Tambem chega como:* a representacao treinada vale alguma coisa na categoria; o Check2HGI aprende
mesmo alguma estrutura; um encoder aleatorio daria o mesmo.

**Veredito: quase nada, NO EIXO DA CATEGORIA — cerca de 0,2 pontos.** Medido em **2026-09-06/07**,
depois da defesa. Um encoder **por treinar**, sem um unico passo do optimizador, chega a **30,60**
(Alabama) e **34,31** (Arizona) contra os **30,77** e **34,51** entregues. A diferenca e **menor que
a dispersao entre sementes do proprio braco treinado**.

Contra o nivel do lugar (29,15 / 31,93), o encoder por treinar ja recupera **+1,45 dos +1,62** e
**+2,38 dos +2,58**. A vantagem da representacao decompoe-se em **cerca de nove partes de
granularidade da entrada para uma parte de grafo aprendido**.

**Isto NAO contradiz o texto depositado; quantifica-o.** O Capitulo 5 ja diz *"most of the category
difference is therefore already available in the raw per-visit features… this control does not
separate it"*. A frase continua verdadeira: aquele controlo, de facto, nao separava. Este separa.

⚠ **DUAS RESSALVAS QUE VIAJAM COM O NUMERO, SEMPRE.** (1) **Eixo da categoria e mais nada** — a
regiao nao foi tocada, e e onde vivem as alegacoes mais fortes (Texas +1,21, California +1,06,
nao-inferioridade nos seis); *"o encoder estava por treinar e nada mudou"* nao e uma afirmacao sobre
o substrato. (2) **Nao diz que a representacao nao vale nada** — diz que o *componente aprendido*
acrescenta pouco *neste eixo*; a estrutura ao nivel do check-in continua a fazer +1,45/+2,38.

**Porque e que isto nao esta no texto:** decisao do autor, 2026-09-07. Um apendice seria peso para
um controlo que confirma o que o texto ja afirma, e um ponteiro do Capitulo 5 para um apendice da
dissertacao **feriria o principio de que os capitulos de artigo se sustentam sozinhos** — o texto
depositado cita-se so a si proprio e ao repositorio (`CLAUDE.md` linha 97), e um ponteiro desses e
das coisas que partem quando o capitulo sai da dissertacao. Fica aqui, pronto a mostrar a quem
perguntar. **Para a versao final do MobiWac e outra conversa**: la o texto esta a ser reescrito e o
numero transforma o "most" numa quantidade.

**Prova:** [`wrapup/post_submission_studies/Q15_untrained_encoder_floor.md`](wrapup/post_submission_studies/Q15_untrained_encoder_floor.md)
— desenho, oito celulas, tres verificacoes de integridade com valores, comandos exactos, e a
ressalva da semente 42 (a mesma da inicializacao do substrato entregue: cosseno 0,80/0,85 contra ~0
das outras tres, logo **as sementes 0/1/7 sao o piso limpo** e a conclusao aguenta-se descontando-a).

---

## Como acrescentar um verdete

Um verdete entra aqui quando a questão está **fechada com prova**, não quando alguém tem uma
opinião forte. Copie a forma: pergunta, paráfrases, veredito, data, prova com caminho, e o
documento que induz ao contrário. Se não conseguir escrever a linha da prova, ainda não é verdete.
