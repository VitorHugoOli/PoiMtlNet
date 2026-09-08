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

**Veredito: NÃO** — e continua NÃO depois da correcção de 2026-09-08, mas por outra razão.

Até 8 de Setembro a razão era esta: `src/configs/canon.py` fixava `DEFAULT_CANON = "v17"`, e o
*bundle* v17 pina `check2hgi_design_k_resln_mae_l0_1` — o substrato v14, com vazamento. Uma corrida
nua escolhia-o **sem o utilizador escrever `--engine`**.

Isso foi corrigido: o default é agora `v18`, sobre o substrato sem vazamento, e a receita v18 sobre
qualquer outro substrato passou a ser **recusa dura** (não honra `MTL_STRICT=0`). **Mas o veredicto
não muda**, porque um *bundle* é uma lista estática de tokens e três coisas da receita entregue não
cabem lá: o `--cat-lr` varia por estado (1e-3 nos pequenos, 2e-3 em FL/CA/TX), a precisão fp32 só
existe por variável de ambiente (`MTL_DISABLE_AMP=1`, não há flag de CLI), e as células reportadas
são as sementes {0,1,7,100} enquanto uma corrida nua leva a 42.

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

**Dezoito** ficheiros foram removidos, no commit `2075e70e` — **16 de rodada** (`_review_v2` ×7,
`_review_v3` ×2, `_round6` ×2, `_round9/reviews`, `_gates`, `_specialists_v1`, `_archive/reviews_v1`,
`science/fundamentals/_review`) mais **2 do `archive/`**. Uns saíram por afirmarem números da geração
com vazamento, outros por serem relatórios de portão superados. Alguns eram citados **por outros
ficheiros de rodada**, e essas citações já não resolvem. **É esperado, e não se conserta indo
procurar o ficheiro.**

> *[Contagem corrigida 2026-09-08: dizia "dezanove ficheiros de rodada". Eram dezoito ficheiros, e
> só 16 deles de rodada. Medido com `git show --diff-filter=D --name-only 2075e70e`. O erro é
> pequeno mas é do género que este ledger existe para não ter — uma contagem em prosa que não bate
> com o que descreve.]*

O caso a conhecer: `_round9/37_reviewer_gate_round9.md:44` cita `reviews/06_number.md`, removido.
Esse relatório **validava as células v17 como correctas** (*"AL 64.51 … CA 77.05 — every cell
matching"*), portanto o seu veredito não vale; a linha da tabela fica como registo de que o portão
correu. O `check_audit_claims.py` valida expressões **dentro** do agregador, não a existência dos
caminhos que ele cita — por isso a esteira não mudou (verificado por A/B).

**Verificado 2026-09-08, antes da fusão para a main:** nenhum dos dezoito é citado por **caminho**
em ficheiro vivo nenhum, e **zero** são citados pelo texto entregue — varri os 18 caminhos e os 18
nomes de ficheiro contra todo o `src/**/*.tex`. As únicas referências que sobram são por nome, e
todas dentro de outros ficheiros de rodada ou dos seus `README`, que é precisamente o caso que este
verdete declara esperado. **A regra do `CLAUDE.md` linha 225 — "do not prune the underscore dirs" —
não foi violada**: ela protege o que o `check.sh` executa e o que o texto entregue cita por caminho,
e nenhum dos dezoito é uma coisa nem outra.

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

## V12 · Como se decide se um documento com números mortos sai ou fica

*Também chega como:* isto ainda vale? dá para apagar? o git guarda, restaura-se depois.

**Veredito: quatro perguntas, por esta ordem.** Adoptado 2026-09-08 como critério do passe de
validade, depois de fechar com 27 candidatos, 11 tarjados, **10 falsos positivos e zero apagados**.

1. **Afirma factos mortos como correntes?** → candidato a sair
2. **Menciona-os para avisar?** → **fica: é a defesa**
3. **A carga é uma decisão, uma medição, ou um dicionário de dados?** → fica, independentemente
   dos números
4. **Algo de carga cita-o?** (a toolchain, o `.tex` entregue, uma lista de conferência do autor)
   → **veto: fica**, mesmo que falhe as três primeiras

### O teste que separa a 1 da 2, e é onde quase toda a gente erra

**Teste a função do número na frase, não a presença dele.** Um número que serve de **comparando,
piso, ou alvo a bater** está a ser *mencionado* — mesmo sem tarja, mesmo sem aspas, mesmo numa
tabela.

O `_round9/reviews/06_number.md` saiu porque dizia *"every cell matching"*: asserção de que as
células estavam **certas**. O `handoff/ch5_mechanism_evidence.md` ficou apesar de imprimir `63.56`
e `77.05` sem tarja, porque a linha lê *"77.05 (−7.17 from the matched arm)"* — o número está lá
para ser batido.

⚠ **O antídoto contém o veneno textualmente.** O `AGENT_GUARDRAILS.md §N1`, o `CLAUDE.md §0` e o
V9b deste ficheiro citam `77.05` **para o nomear como errado**. Uma varredura cega por esse número
apaga exactamente a defesa. Classifique cada ocorrência antes de agir.

**Filtro rápido que decide a maioria:** o documento **declara a que geração pertence**? Se declara,
quase de certeza é menção. Foi essa regra que salvou os 37 ficheiros do `docs/` — ver abaixo.

### Onde "está no git, restaura-se" falha

O git preserva bytes, **não descobribilidade**. **O caso que o provou, no mesmo dia:** a
proveniência dos `+2,0 / +1,7 / +0,8` **já submetidos** no MobiWac só se traçou até ao A2 porque o
`docs/studies/pre_freeze_gates/A2_RESULTS.md` estava vivo e apareceu num `grep` pelos valores. Era um
relatório de portão resolvido, de um estudo fechado — perfil exacto de candidato a poda pela regra 1.
Sem ele a resposta honesta teria sido *"ninguém sabe"* sobre um número publicado. Apagar é seguro quando a ausência se **auto-anuncia**
(um portão fica vermelho, um link dá 404 na revisão) e inseguro quando é silenciosa. O teste antes
de agir: *"o que parte alto se isto sair?"* — se a resposta for "nada", isso **não** é prova de
segurança, é o sinal de perigo. Cinco casos medidos nesta árvore:

- o ponteiro sobrevive ao ficheiro (`check.sh` **executa** `_round9/35_wave_a_render_check.py`);
- o `.tex` entregue cita caminhos de ronda **40 vezes** como proveniência, e num documento
  depositado a proveniência É o artefacto;
- o caso circular: para saber que deve restaurar o `_aut_closed_blocks.md` precisaria do que está
  escrito **dentro** dele (que 32 números não existem noutro sítio);
- o dicionário de dados: apagar o `_round7/gradient_cosine_tests6_README.md` deixa o CSV a
  significar outra coisa em silêncio (**sete** datasets, e Georgia não é dos seis);
- restaurar exige o caminho e o sha, e ninguém procura um ficheiro que não sabe que existiu.

---

## V13 · Uma medição acusou muita coisa. Confio nela?

**Veredito: NÃO. Suspeite da medição primeiro.** Em 2026-09-05/08, **quatro** sondas acusaram em
massa e **nas quatro o ficheiro estava bem** — o erro era sempre da sonda.

| sonda | acusou | verdade |
|---|---|---|
| regex de `v1x` para achar geração | 6 sem rótulo | descreviam-se por motor e data |
| janela de 25 linhas no cabeçalho | `docs/baselines/README.md` mudo | declara **por linha**, mais abaixo |
| resolução de caminhos relativos | 130 alvos inexistentes | **129 existiam** |
| resolver `log.md` pela pasta irmã | ponteiro fora do ficheiro | há **18** `log.md`; era o do assunto, não o irmão |

### E o inverso: detecção estrutural subestima o dano

Verificação **estrutural** dos guardas (o alvo existe? a linha existe?) encontrou **1 ponteiro
partido em 308**. Verificação **do conteúdo**, feita à mão, encontrou **12 em 17 a apontar para o
sítio errado — cerca de 70%**. O ficheiro existia e a linha existia; o conteúdo é que se tinha
mexido por baixo.

**Portanto: não automatize a auditoria de guardas.** Uma varredura estrutural dá dois achados
triviais e uma sensação de cobertura, que é pior do que não ter nenhuma — falha exactamente a
espécie que motiva a auditoria. O remédio é de **construção, não de detecção**: ancorar por
**conteúdo**, nunca por número de linha (`ACHADOS.md` §A4 — *"aponta por conteúdo, que não apodrece
quando o ficheiro se mexe"*). A conversão dos oito ficheiros-guarda fechou 2026-09-08 com zero
ponteiros de linha.

### Números desta série, para quem a repetir

- `docs/` + `articles/[mobiwac]`: **40** ficheiros com células v17, **37 declaram a geração** no
  cabeçalho. O `docs/` já pratica a convenção; **a dissertação era o valor atípico** (1 dos 262
  tinha cabeçalho de estado). Não generalize a poda a partir da excepção.
- Peneira `afirma|denuncia` sobre `63.56|79.85|77.05|64.51|RESULTS_BOARD`: **2,6× de
  sobre-reporte**. É triagem, nunca veredicto — cada acerto tem de ser lido antes de contar.

---

## V14 · Um guarda podre é só um documento podre a mais?

*Também chega como:* o aviso está desactualizado, corrijo depois; é só um ponteiro.

**Veredito: NÃO — é pior, e de forma assimétrica.** Um documento superado engana **quem o lê**. Um
guarda superado engana quem o lê **e quem o obedece**.

**O caso, de 2026-09-08.** Um aviso do `CLAUDE.md` sobre o `NORTH_STAR.md` apodreceu (citava cinco
números de linha que já apontavam para outro conteúdo). Ao ser substituído, o substituto afirmou
*"nenhuma dessas frases existe no ficheiro, zero ocorrências"*. **Existiam, duas vezes cada**, e
**correctamente tarjadas** `[SUPERADO 2026-08-20]`.

⚠ **Repare na direcção do dano.** O guarda podre só podia causar desconfiança inútil de um ficheiro
que estava bem. O substituto podia causar uma **deleção**: quem lesse "zero ocorrências" e
encontrasse uma leria registo histórico marcado como contaminação fresca e "corrigi-la-ia"
apagando — destruindo exactamente o que a tarja existe para proteger. **A medição que se escreve no
substituto de um guarda é ela própria um guarda.**

**A causa não foi grafia: foi o ficheiro errado.** Há **dois** `NORTH_STAR.md` — `docs/` (a receita
campeã) e `articles/dissertacao/` (a tese e o mapa dos capítulos), documentos diferentes. Mediu-se um
para responder sobre o outro. Não é uma busca que não podia encontrar; é **a busca certa no sítio
errado**, e nenhuma disciplina sobre grafias a apanha. Os dois ficheiros levam agora tarja a dizer
que o irmão existe (241 referências sem caminho contra 67 qualificadas).

**Onde ancorar, e a ordem importa:**

| âncora | dura? |
|---|---|
| número de linha · contagem que alguém mantém · nome sem caminho havendo homónimo | **não** |
| **nome de símbolo** (`DEFAULT_CANON`, `\finalbuildfirstpage`) | **sim** — sobreviveu a todas as conversões |
| nome de estudo · geração · data · texto de um marcador | **sim** |

**Prefira o símbolo à frase:** uma âncora em prosa move-se com cada errata; um símbolo só muda
quando o código muda, e aí quebra ruidosamente. **E verifique a âncora em cada ficheiro que ela
reclama**, não uma vez.

**Prova:** `ARMADILHAS_DE_MEDICAO.md` §13 · correcção em `d0523c2d` · tarjas em `8adc2215`.

---

## Como acrescentar um verdete

Um verdete entra aqui quando a questão está **fechada com prova**, não quando alguém tem uma
opinião forte. Copie a forma: pergunta, paráfrases, veredito, data, prova com caminho, e o
documento que induz ao contrário. Se não conseguir escrever a linha da prova, ainda não é verdete.
