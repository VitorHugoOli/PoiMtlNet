# PENDENCIAS.md — o que falta, e de quem depende

**Fila viva. Se um item nao espera nada de ninguem, ele nao mora aqui.**

## Como este arquivo funciona

**Cada item tem a mesma forma, e ela e curta:**

```
### N.M Titulo de uma linha
**O que e.** Uma a tres frases: o achado, com o numero medido.
> **DECISAO SUA:** o que falta, com as opcoes e o custo de cada uma.
*Forense: ponteiro para o relatorio de rodada.*
```

**Onde cada coisa vive.** O tracker carrega a **decisao**; a **forense** (como o defeito foi descoberto, qual
instrumento mentiu, o que cada commit mediu) vai para `_round8/`. Em 2026-07-30 seis itens carregavam 34 mil dos 55 mil
caracteres do arquivo, quase tudo forense: foi para
[`_round8/29_pendencias_detail.md`](_round8/29_pendencias_detail.md), **nada apagado**, e o arquivo caiu de 67 mil para
37 mil.

**Para ADICIONAR um ponto seu:** escreva embaixo do item, comecando a linha com `> DECISSAO:` (ou
`> DECISAO:`). Eu leio isso como sua palavra final e nao reinterpreto. Se voce nao tiver numero de item, escreva no fim
do §2 com um titulo qualquer — eu numero e coloco no lugar.

**Para FECHAR um item:** ele sai daqui e vai para `_archive/PENDENCIAS_RESOLVIDOS.md` **com o motivo de saida no topo do
bloco**. O gate `check_tracker_refs.py` falha se um item desaparecer sem chegar ao arquivo — tres foram perdidos assim,
e voce achou dois deles lendo o arquivo. **Nao renumere:**
comentarios no fonte citam estes numeros, e um buraco na numeracao e melhor que um ponteiro errado.

**Ordem das secoes:** §2 (voce) -> §5 (do `CODEX_AUDIT`) -> §6 (do `CONSIDERATIONS`) -> §3 (terceiros) -> §4 (o que
auditar primeiro). Deliberada: o que depende de voce vem antes. O §6 entrou em 2026-07-30 e substitui o §2.8, que agora
registra o que foi feito em vez de pedir uma decisao.

---

## §2 · Aberto e bloqueado em VOCE

> **LIMPO EM 2026-07-30, a seu pedido.** Cinco itens desta secao estavam **de fato fechados** e foram
> movidos para `_archive/PENDENCIAS_RESOLVIDOS.md` com o motivo de saida no topo de cada bloco:
> **2.2** (push publicado, verificado por hash contra o remoto — o resto virou 2.16), **2.3** (fechado
> pela sua frase *"podemos fechar esse ponto"*), **2.7** (orcamento de tuning nao-recuperavel,
> registrado em `LEFT_OUT.md`), **2.13** (o comando contava 4 a mais por ser cego a comentarios;
> corrigido) e **2.17** (afirmacao falsa minha, corrigida com nota de git em `a07e547b`).
>
> **Os buracos na numeracao — 2.2, 2.3, 2.7, 2.13, 2.17 — sao esses cinco, e nao perdas.** Nao
> renumerei os que ficaram: seis comentarios no fonte e o `_round6/VERIFY_LIST.md` citam estes numeros,
> e renumerar transformaria cada citacao num ponteiro para o item errado, que e pior que um buraco.
> O gate `check_tracker_refs.py` agora falha se um item sair daqui sem chegar ao arquivo.
>
> **O que sobrou aqui espera VOCE, nao a mim.** Onde a medicao esta completa, o bloco `(A)/(B)/(C)`
> diz exatamente o que falta e quanto custa cada saida.

### 2.1 Os 53 marcadores `[NEEDS SIGN-OFF]` no fonte

**O que e.** 53 pontos do fonte marcados como precisando do seu aval. Nenhum bloqueia build; o gate
`check_verify_list` conta e a contagem bate. A lista completa, com arquivo, linha e o que cada um afirma, sai de:
`grep -rn "NEEDS SIGN-OFF" src/ --include="*.tex" | grep -v ":\s*%"`.

**Tres tem prioridade** (afirmam algo sobre trabalho publicado ou co-autorado): o paragrafo corrigido do Apendice B
sobre o Cap. 3, o numero limitado do Cap. 4 na conclusao, e a frase de reprodutibilidade enfraquecida. Estao detalhados
em `_round6/VERIFY_LIST.md` A1, A2 e A3.

> **DECISAO SUA:** ler os 53 e me dizer quais aprova. Nao precisa ser de uma vez — se me der os tres
> prioritarios, eu removo os marcadores deles e mantenho os outros 50.

*Forense (a tentativa de push destrutiva, o worktree, os artefatos divergentes): agora e o item 2.16 e o corpo integral
esta em [`_round8/29_pendencias_detail.md`](_round8/29_pendencias_detail.md).*

### 2.5 O tamanho de tipo das duas figuras de arquitetura — autorizado, mas eu nao consigo executar

**Voce autorizou:** *"pode aumentar, mas mantenha o espaco ja ocupado pela imagem... mantendo a proporcao"*, e observou
que o contraste hoje ja deixa legivel.

**Nao consigo fazer daqui:** nao ha `drawio` nem `inkscape` neste ambiente, e so **1 dos 2** `.drawio`
esta no repositorio. A receita esta em `_round6/12_figures.md` (subir `fontSize` de 13 para ~20 e reexportar na mesma
largura em pixels).

> **Seu, quando quiser:** reexportar as duas no Draw.io e me passar os PNG — eu troco e remeco o tipo na
> pagina. **Opcional**, pela sua propria observacao sobre o contraste.

### 2.8 `CONSIDERATIONS.md` — EXECUTADO nesta rodada; a fila de decisao virou o §6

**O que era.** `src_utils/CONSIDERATIONS.md` (1.229 linhas, nao commitado) trazia o feedback verbal do Germano
transcrito por voce (l. 3-58), o feedback escrito do Fabrício (l. 59-309), a auditoria Codex do Cap. 2 de 2026-07-28 (l.
310-994) e o adendo dela (l. 995-1229), com uma lista de trabalho consolidada. Voce pediu as duas coisas do item:
commitar o arquivo, e executar a divisao.

**O que foi feito (round 9, commit `d4078c75`).** O arquivo foi commitado **sem uma alteracao**
antes de qualquer reescrita, e depois reescrito no esquema de um bloco por item: **43 blocos**, IDs estaveis `FAB-01`..
`FAB-31` (a sua propria numeracao), `GER-01`..`GER-11` e `AUT-01`. A prosa original esta preservada byte a byte em
[`_round9/30_considerations_prosa_original.md`](_round9/30_considerations_prosa_original.md), com o sha256 e o comando
que o reproduz no cabecalho.

**O passe de citacoes obsoletas, antes de qualquer juizo.** Das 41 ancoras localizaveis, **32 sao exatas e 9 estao
obsoletas** (4 alteradas, 5 desaparecidas). As nove obsoletas sao **todas** do Fabrício, **todas** citam `0_main.tex`
(arquivo que deixou de existir em 2026-07-29), e sao exatamente os itens **FAB-02 a FAB-10**. As **20** ancoras dele
fora daquele arquivo conferem todas.

*(Este paragrafo dizia "10 obsoletas, todas citam `0_main.tex`" e "21 ancoras". Estava errado por uma: a decima era o
**FAB-27**, que esta em `tables/frame/lineage.tex`, e cuja citacao so falhou porque ele a escreveu prefixada com "Na
tabela: " e terminada em "....". O texto que ele cita esta presente palavra por palavra. Os numeros certos sao os de
cima, e sao os mesmos que o §6 carrega — o §2.8 tinha ficado com a versao anterior a correcao, o que deixou este arquivo
contradizendo a si mesmo por dois commits.)* Comandos em
[`_round9/31_stale_quote_pass.md`](_round9/31_stale_quote_pass.md).

**A divisao.** 21 itens "eu aplico", 21 precisam de voce (viraram o **§6**), 1 bloqueado (FAB-28, verificacao falhou).
Tres medicoes da auditoria de 2026-07-28 **nao reproduzem** e foram retomadas: ela contou 27 paragrafos / media 161 /
cinco acima de 240 palavras, e hoje sao 33 / 132 / 4.

**Preservado da esteira paralela (commit `c94d1f19`, 02:01:51), porque a minha reescrita deste bloco apagou a nota dela
e a nota esta certa** — anotada como dela, e reconciliada com a numeracao deste arquivo:

> **ATUALIZACAO 2026-07-30:** voce acrescentou um ponto seu no fim do arquivo (l. 1228), sobre
> otimalidade de Pareto nos fundamentos de MTL. **Esse esta sendo executado agora** — e o item 2.12 pelo
> outro lado (o termo esta em prosa publicada e nao esta no `GLOSSARY`, que e fail-closed), e a esteira que
> trabalha nele tambem le os itens G8 (definicoes formais de MTL) e G10 (o argumento tecnico para o achado
> de nao-conflito) porque sao a mesma peca de texto. **O resto do arquivo continua nao executado.**

Duas reconciliacoes, para os ponteiros nao apontarem para o vazio. Os `G8` e `G10` daquela nota sao, nos IDs estaveis
deste arquivo, o **GER-09** e o **GER-11**; e o "ponto no fim do arquivo (l. 1228)" e o **AUT-01**. E a frase final
daquela nota deixou de valer entre 02:01 e agora: o resto do
`CONSIDERATIONS.md` **foi** executado nesta rodada, no sentido de estar medido e dividido, mas **nenhum item foi
aplicado a nenhum capitulo** — que e provavelmente o que ela queria dizer.

> **AUTHOR:** nada aqui. Este item esta fechado; o que espera voce esta no **§6**.

*Forense: `_round9/31_stale_quote_pass.md` (o passe de obsolescencia, incluindo os dois instrumentos meus que estavam
quebrados e como cada um foi pego).*

### 2.9 Os tres datasets que faltavam no Apendice F — RODADOS. O apendice agora tem SETE, e sobra uma decisao pequena

**FEITO EM 2026-07-30, e nada aqui espera GPU.** California, Texas e Istanbul foram medidos no `nespedgpu`, um dataset
por job, sequencialmente. Cada `rc=0` foi lido do `_status.json` do proprio job, e cada duracao vem dos **stamps do
proprio dataset**, nunca do total do job:

| dataset    | job        | duracao (stamps proprios)    | folds |
|------------|------------|------------------------------|-------|
| istanbul   | `9f3da11f` | 19,3 min (04:56:49→05:16:07) | 5/5   |
| texas      | `6faa6e22` | 55,1 min (05:17:52→06:12:59) | 5/5   |
| california | `67585dff` | 44,3 min (06:14:17→06:58:37) | 5/5   |

**Custou ~2h de GPU, nao as ~6h estimadas** — e o motivo importa: `--no-checkpoints`. Os pesos salvos eram o que enchia
o disco (`results/check2hgi/texas/checkpoints` sozinho tinha 7,1G) enquanto o diretorio que carrega o diagnostico tem
~6-11 MB. `df -h /home` ficou em **313G usados / 61G livres, 84%, sem mover**, lido direto antes de cada submit e depois
de cada run. E nao muda numerica nenhuma: **medido**, nao argumentado — o alabama, que ja estava no apendice, foi
re-rodado com a flag e reproduziu os cinco CSVs por fold **byte a byte identicos**.

**Nada foi apagado na sua maquina.** Os 61G que voce liberou continuam livres; os dois arquivos corrompidos da corrida
de harvest antiga seguem em `~/cosine_appendix/california_f2` e `_f3` (md5 `2afa6aebfb...` nos dois) e **nao entraram**
no parquet: o california veio do job `67585dff`, cujos cinco folds tem cinco md5 distintos.

**O apendice passou de quatro para SETE datasets** — as suas **seis** mais a Georgia, que nao e uma das seis. 4.650
observacoes, todos os testes no **fold** como unidade. Equivalencia por TOST dentro de ±0,05 vale em todos os sete.
Verificado no PDF renderizado (pp. 97-102), nao no fonte.

**Um achado que vale mais que os tres datasets:** o "limite de 35 minutos do host" era **nosso**. O `job.sh` embrulha a
carga em `timeout <N>` com o `timeout_seconds` que o agente passou, e o `job.sh` do job 805120f1 diz `timeout 2100` =
35,0 min exatos. Nao existe teto de 35 min nesse host. O texas rodou 55 min. **O dataset que foi registrado como
impossivel era impossivel apenas sob um limite que nos mesmos definimos.**

> **DECISAO SUA, e e pequena:** o Cap. 5 continua reportando a medida antiga desse mesmo cosseno (de
> desenvolvimento, quatro seeds, preparacao anterior) com media +0,001 e maior media por dataset
> +0,0032. O apendice agora da +0,00102 no conjunto e +0,0112 no alabama. **Nao sao contraditorios** —
> sao corridas diferentes, e o apendice ja diz isso em uma frase. Voce decide se quer (**a**) deixar
> como esta, ou (**b**) que eu proponha uma nota no Cap. 5 apontando para o Apendice F. **Eu nao toco
> no Cap. 5**: esta sob revisao.

*Forense completa (a corrida de harvest, o gate de distincao por md5 validado por sabotagem, o `c.download()` que
achatou quinze arquivos em cinco, e o cap que era nosso): [`_round9/30_cosine_six.md`](_round9/30_cosine_six.md).*

> **AUTHOR:** Vamos um pouco ainda mais longe, voce pode editar o Cap. 5 tanto o artigo original quanto o da
> dissertacao, como estamos um fase de revisão no mobiwac, conseguimos mandar uma revisão ainda.

### 2.11 A assimetria do resultado de regiao: o Cap. 5 ressalva, e o resto do documento nao

**Origem:** `_round6/VERIFY_LIST.md` itens 4 e 5 (achado L-5 do ledger), entregues em 2026-07-30.

**(A) O que e.** `chapters/5_mobiwac/05_setup.tex` diz que o plano de analise *"did not cover next-region superiority,
so the four next-region gains ... are secondary results outside it"*. O resto do documento afirma o mesmo resultado
**sem essa ressalva**. Medido com o varredor que remove comentarios, sobre os 54 `.tex`:

| onde                                                                                                            | forma                                                                  |
|-----------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| `1_introduction.tex:132`, `6_conclusion.tex:21` e `:93`                                                         | "four of six" / "four of the six"                                      |
| `2_fundamentals.tex:786`, `5_mobiwac/01_introduction.tex:39`, `5_mobiwac/08_conclusion.tex:14`, `5_mobiwac.tex` | idem                                                                   |
| `content.tex:166` (Resumo e Abstract)                                                                           | "quatro deles" / "four of them" — a mesma alegacao, em outras palavras |

No PDF de defesa (100 pp) a alegacao sem ressalva imprime nas **pp. 14, 58, 59, 76, 77 e 78**; a ressalva imprime **so
na p. 67**. Sao sete sitios em prosa mais as duas parafrases do pre-textual, contra uma ressalva.

**(B) Por que importa.** O registro estatistico de 2026-07-27 e inequivoco: o teste primario registrado para **toda**
celula de regiao e nao-inferioridade TOST. Uma leitura rapida do Resumo, da Introducao ou da Conclusao le "outperforms
em quatro de seis" como resultado primario; a p. 67 diz que nao e. Nenhuma track da rodada 6 assumiu isso (achado L-5 do
ledger).

**(C) O que eu preciso de voce.** Uma regra, e ela vale para os nove sitios de uma vez:

> **(a)** o texto de moldura acrescenta "as a secondary result" (ou equivalente) **uma vez**, no
> ponto que voce escolher — o candidato natural e a Conclusao, que ja e o lugar onde o escopo do
> plano de analise e discutido. Custo: uma frase, mais linha de errata se voce quiser rastrear.
> **(b)** a assimetria e deliberada — o Cap. 5 e o capitulo que carrega o metodo, entao e onde a
> ressalva pertence — e isso vai para `LEFT_OUT.md` com o motivo. Custo: zero no texto, mas o
> registro passa a existir.
>
> **Eu nao decido isto** porque muda o que o Resumo e a Conclusao afirmam sobre o resultado
> principal do Cap. 5, que e prosa sua sobre um resultado seu.

> **AUTHOR:**  Na verdade se esses resultados forem de grande importancia e algo importante para narrativa, vale deixar
> eles
> como primario é só alterar o mobiwac, ao invés do outros textos, o que acha ? Se não concorda vide a narrativa do
> texto vamos de A

### 2.12 `Pareto-stationary point` esta na prosa e nao esta no registro (o `GLOSSARY` e fail-closed)

**(A) O que e.** A regra de manutencao do `GLOSSARY.md` e explicita: *"a term not in this registry may not be used in
dissertation prose"*. Medido hoje:

```bash
cd /Users/vitor/Desktop/mestrado/ingred/articles/dissertacao
grep -c 'Pareto-stationary' GLOSSARY.md          # 0
```

e o termo esta em prosa em **cinco** sitios (o registro anterior dizia dois), em quatro arquivos:
`chapters/3_cbic/method.tex` (*"convergence to a Pareto-stationary point"*), duas vezes em
`chapters/3_cbic/basis.tex` (*"Pareto-optimal descent directions"* e *"Pareto efficiency"*),
`chapters/4_courb/methodology.tex` (a garantia do Nash-MTL, a mesma frase do item 3 do
`VERIFY_LIST`) e a linha de errata que registra essa correcao, em `tables/courb/errata.tex`, que ainda traz a forma sem
hifen, "Pareto stationary". As outras duas entradas que o item 4 daquele arquivo pedia — **bilinear discriminator** e
**logistic function** — **ja entraram**
(`GLOSSARY.md:71` e `:72`); so esta faltou.

**Duas correcoes ao proprio texto acima, medidas em 2026-07-30 e registradas aqui porque mudam o que um agente pode
fazer.** Primeira: **nao sao todos prosa publicada**. As frases do Cap. 3 sao (cada uma e substring literal de
`articles/CBIC___MTL/sections/*.tex`), mas a do Cap. 4 **nao e** — e a frase corrigida por errata deste proprio
documento, ja listada no Apendice B, e a fonte PT publicada nao tem nenhuma ocorrencia do termo
(`articles/CoUrb_2026/src`, nove arquivos `.tex`, zero). Segunda: as paginas. O registro anterior dizia pp. 36 e 48,
medidas num build de 100 paginas; o trecho novo da §2.3 acrescenta uma pagina, entao **medido contra o build de defesa
de 101 paginas deste commit**:
§2.3 nova na p. 23, `3_cbic/basis.tex` nas pp. 31 e 32, `3_cbic/method.tex` na p. 37,
`4_courb/methodology.tex` na p. 49, e a linha de errata na **p. 16 do volume suplementar**
(`make extra`, 20 paginas), que nao entra no build de defesa.

**(B) Por que importa.** O `make check` **nao pega isto**: existe um gate de "Pareto" mas ele e informativo e conta
ocorrencias, nao registro. E o termo nao pode ser simplesmente removido — as duas frases sao publicadas, entao tirar o
termo e editar uma frase publicada, com linha no Apendice B.

**(C) O que eu preciso de voce.** Uma decisao, tres saidas:

> **(a)** registrar o termo (uma linha na §4 do `GLOSSARY`, a definicao ja esta escrita na propria
> frase do Cap. 4: *"a point at which some convex combination of the task gradients is zero"*).
> Custo: uma linha, e o documento fica consistente com a propria regra.
> **(b)** trocar o termo nas duas frases publicadas. Custo: duas linhas de errata no Apendice B, e
> voce esta editando prosa publicada por uma questao de vocabulario.
> **(c)** registrar uma excecao explicita para termos que chegam em prosa reproduzida. Custo: uma
> nota no `GLOSSARY`, e a regra deixa de ser fail-closed para essa classe.

> DESICAO: A.

**EXECUTADO EM 2026-07-30, a sua decisao (a).** O termo entrou em `GLOSSARY.md:103` (§4) e `:148` (§6), e com ele tres
outros que a §2.3 passou a usar: **Pareto dominance**, **Pareto optimality** e **gradient conflict**. Cada definicao foi
copiada de um artigo aberto no PDF naquela sessao, com a pagina na propria linha do registro; o razao completo, com a
declaracao exata de convergencia de cada metodo, esta em [`_round9/31_pareto.md`](_round9/31_pareto.md). O trecho novo
da §2.3 renderiza na **p. 23** do build de defesa de 101 paginas, e o gate agora pega a regressao:
`check_audit_claims.py`
probes `R9-pareto`, `R9-conflict`, `R9-nocount` e `R9-glossary`, cada um validado por sabotagem (reverter a propriedade
e ler rc=1). A anotacao do `VERIFY_LIST` item 4 que esperava
`Pareto-stationary 0` foi corrigida para `2` no mesmo commit, porque a decisao (a) a tornou falsa.

**Uma coisa ficou faltando, e e sua.** Tres dos quatro termos PT nao existem em nenhum lugar deste repositorio, nem na
fonte PT publicada, nem em nenhuma superficie em portugues da dissertacao. Nao os inventei em silencio: entraram
marcados como propostos. `otimalidade de Pareto` **nao** e um deles, e sua propria palavra (l. 98 deste arquivo) e a do
Germano (`articles/[mobiwac]/REVIEW_GERMANO.md:778`).

> **DECISAO SUA:** confirmar ou trocar os tres nao atestados: **dominância de Pareto**, **ponto
> Pareto-estacionário**, **conflito de gradientes**. Custo: uma linha. Nenhuma superficie PT usa os
> quatro hoje (o Cap. 2 e em ingles e o Resumo nao os menciona), entao isto nao bloqueia build nem
> gate; fica registrado para que um tradutor futuro ache uma decisao e nao um vazio.

> **TAMBEM SUA, e e uma questao de errata, nao de vocabulario.** Verifiquei as cinco frases contra a
> garantia que o artigo-fonte de cada uma realmente enuncia. **Todas as cinco enunciam corretamente**,
> e nada foi editado. Uma imprecisao: `3_cbic/basis.tex` (p. 31) diz que o MGDA *"finds Pareto-optimal
> descent directions"*, e o que `sener2018mgda` enuncia (pp. 4 e 6) e uma dicotomia — ou o ponto e
> Pareto-estacionario, ou a direcao *"decreases all objectives"*; a otimalidade de Pareto que o artigo
> reivindica e para o limite superior, sob suposicoes (p. 1). E prosa publicada reproduzida e a
> compressao e a que a literatura de MTL costuma fazer, entao **deixei como esta**. Se quiser corrigir,
> e uma linha de errata no Apendice B, e a decisao e sua.

> **AUTHOR:** Otimo trabalho, pode adicionar essa linha no appendix B, para termos conhecimento desse detalhe menor e
> não deixar passar batido.

### 2.14 O intervalo de paginas do `nash`: nao da para verificar daqui

**Origem:** `_round6/VERIFY_LIST.md` item 14, entregue em 2026-07-30 (precedente `standley2020tasks`).

**(A) O que e.** `references.bib` traz `pages = {16428--16446}` para
`@inproceedings{nash}` (Navon et al., *Multi-Task Learning as a Bargaining Game*, ICML 2022). Tentado de novo nesta
sessao, contra as fontes de registro que o sandbox alcanca:

| fonte                            | resposta                                                                                                 |
|----------------------------------|----------------------------------------------------------------------------------------------------------|
| OpenAlex                         | um unico registro, `W4225981399`, tipo **preprint**, venue "arXiv", `first_page` e `last_page` **nulos** |
| Crossref (`query.bibliographic`) | cinco obras, **nenhuma delas este artigo** — nao ha DOI registrado da versao de anais                    |
| `proceedings.mlr.press`          | **fora da allowlist** do sandbox; nao acessado                                                           |

**(B) Por que importa.** Pelo §1 do `AGENT_GUARDRAILS`, um identificador que nao foi aberto na fonte de registro nao
pode ser apresentado como conferido. O campo esta no `.bib` e nao esta verificado.

**(C) O que eu preciso de voce.** Um clique fecha: `proceedings.mlr.press/v162/navon22a.html`.

> **(a)** confirmar o intervalo e ele fica; **(b)** apagar o campo `pages`, que e exatamente o
> precedente que esta bibliografia ja adotou para `standley2020tasks`.

> **AUTHOR:** I get from the website:
> @InProceedings{pmlr-v162-navon22a, title = {Multi-Task Learning as a Bargaining Game}, author = {Navon, Aviv and
> Shamsian, Aviv and Achituve, Idan and Maron, Haggai and Kawaguchi, Kenji and Chechik, Gal and Fetaya, Ethan},
> booktitle = {Proceedings of the 39th International Conference on Machine Learning}, pages = {16428--16446}, year =
> {2022}, editor = {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and
> Sabato,
> Sivan}, volume = {162}, series = {Proceedings of Machine Learning Research}, month = {17--23 Jul}, publisher = {PMLR},
> pdf = {https://proceedings.mlr.press/v162/navon22a/navon22a.pdf},
> url = {https://proceedings.mlr.press/v162/navon22a.html}, abstract = {In Multi-task learning (MTL), a joint model is
> trained to simultaneously make predictions for several tasks. Joint training reduces computation costs and improves
> data
> efficiency; however, since the gradients of these different tasks may conflict, training a joint model for MTL often
> yields lower performance than its corresponding single-task counterparts. A common method for alleviating this issue
> is
> to combine per-task gradients into a joint update direction using a particular heuristic. In this paper, we propose
> viewing the gradients combination step as a bargaining game, where tasks negotiate to reach an agreement on a joint
> direction of parameter update. Under certain assumptions, the bargaining problem has a unique solution, known as
> the <em>Nash Bargaining Solution</em>, which we propose to use as a principled approach to multi-task learning. We
> describe a new MTL optimization procedure, Nash-MTL, and derive theoretical guarantees for its convergence.
> Empirically,
> we show that Nash-MTL achieves state-of-the-art results on multiple MTL benchmarks in various domains.} }
> Intresting that in the google scholar use the citation:
> @article{navon2022multi, title={Multi-task learning as a bargaining game}, author={Navon, Aviv and Shamsian, Aviv and
> Achituve, Idan and Maron, Haggai and Kawaguchi, Kenji and Chechik, Gal and Fetaya, Ethan}, journal={arXiv preprint
> arXiv:2202.01017}, year={2022} }

### 2.15 Tres citacoes NOT-SUPPORTED e um termo banido, todos em prosa publicada reproduzida

**Origem:** `_round6/VERIFY_LIST.md` itens 15 e 16, entregues juntos em 2026-07-30 como uma decisao unica.

**(A) O que e.** Quatro pontos, um so tipo de decisao: **nenhum deles pode ser corrigido por um agente**, porque todos
estao em frases publicadas, e mexer nelas gera linha de errata.

| onde                              | o que                                                                       |
|-----------------------------------|-----------------------------------------------------------------------------|
| `3_cbic/method.tex`               | `ruder2017sluice` citado para regularizacao implicita do hard sharing       |
| `4_courb/methodology.tex:173`     | `sun2020go` citado para ciclos temporais revelarem a *funcao* do lugar      |
| `4_courb/methodology.tex:184`     | `belkin2003laplacian` citado para um regularizador hierarquico de embedding |
| `4_courb/methodology.tex:173,184` | **`fclass` em prosa renderizada**, 4 ocorrencias em 3 linhas                |

**(B) Sobre o `fclass`, que ninguem tinha levantado.** O `GLOSSARY.md:73` diz, com todas as letras:
*"In code this column is `spot`, renamed `fclass` at `hgi/preprocess.py:62`; **NEVER write `fclass`
in prose**"*. Varri os 54 `.tex` sem comentarios: `4_courb/methodology.tex` e o **unico** arquivo. E as frases identicas
estao publicadas no CoUrb (`src_en/sections/metodology.tex:109` e `:120`), entao e o mesmo tipo de decisao das tres
citacoes acima. Nenhum gate pega: o de codenomes casa
`B9|v1[1-7]|champion-G|H3-alt|dk_ovl|log_T|substrate`, e `fclass` nao esta na lista.

**(C) O que eu preciso de voce.** Uma decisao por linha, ou uma regra para as quatro:

> **(a)** trocar. As substituicoes sugeridas (`baxter2000model`, `Xu2023`) **ja estao** na
> bibliografia e ja sao citadas para essas alegacoes em outros pontos; para o `fclass`, o termo
> registrado e "fine class". Custo: uma linha de errata no Apendice B por sitio.
> **(b)** manter e registrar. As frases sao do artigo publicado e reproduzi-las fielmente e
> defensavel; a divergencia vai para `LEFT_OUT.md`. Custo: zero no texto.
>
> `ruder2017sluice` merece uma nota: a mesma chave carrega **tambem** uma decisao de metadados (o
> titulo no `.bib` e o do preprint superado, "Sluice Networks..."; o titulo de registro e "Latent
> Multi-task Architecture Learning" e a versao de registro e AAAI 2019,
> `10.1609/aaai.v33i01.33014822`, pp. 4822-4829). Decida as duas juntas para tocar a entrada **uma
> vez** so.

> **AUTHOR:** Vamos de troca em ambas, seguindo o caminho A, modificando os artigos originais e adicionando uma entrada
> no appendix B.

### 2.16 Quatro artefatos publicados **divergiram** das copias locais (o item 2.2 cobria dois)

**(A) O que e.** O Apendice A cita treze caminhos `\path{}`. A pergunta "quantos faltam no branch publico" ja teve
**quatro** respostas nesta base (9 de 13, depois 5, depois 4, agora esta). O motivo de todas as anteriores e o mesmo:
`git cat-file -e mobiwac:<caminho>` pergunta *"este CAMINHO esta no branch"*, e a alegacao e *"este ARQUIVO esta no
branch"* — e o branch `mobiwac` **nao tem arvore
`docs/`**, guarda esses artefatos em `analysis_protocol/`. Remedi por **hash**, comparando cada arquivo local com os
blobs do branch:

| classe                                   | n | quais                                                                                                                                                                                                                                                                               |
|------------------------------------------|--:|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| no branch, **byte a byte identicos**     | 8 | `folds.py`; `STATISTICAL_PROTOCOL.md`, `JOINT_BEST_RESULTS.md`, `m1_full_output.txt`, `m2_prereg_output.txt` (sob `analysis_protocol/`); `build_phase3_per_fold_transitions.sh`; `score_joint_best.py`; `autocorrelation_ceiling.py` (em `scripts/`, nao `scripts/embedding_eval/`) |
| no branch com **conteudo diferente**     | 4 | `superiority_wilcoxon.py`, `region_match_tost.py`, `m1_stats_n20.py`, `m2_prereg_perfold.py`                                                                                                                                                                                        |
| diretorio (o instrumento nao classifica) | 1 | `stats_n20/`                                                                                                                                                                                                                                                                        |

**Nada esta faltando.** Quatro artefatos publicados **divergiram**:

| arquivo                   | local | no branch | linhas diferentes |
|---------------------------|------:|----------:|------------------:|
| `superiority_wilcoxon.py` |   147 |       126 |                37 |
| `region_match_tost.py`    |    74 |        74 |                 2 |
| `m1_stats_n20.py`         |   411 |       335 |                84 |
| `m2_prereg_perfold.py`    |   214 |       222 |                36 |

**(B) Por que importa.** O seu item **2.2 ja reservou exatamente esta decisao a voce**, mas para **dois** arquivos
(`m1_stats_n20.py` e `m2_prereg_perfold.py`). Sao **quatro**. Substituir um artefato publicado por uma versao local
divergente e decisao de autor, nao faxina — nao toquei em nada.

**(C) O que eu preciso de voce.** Por arquivo, ou uma regra para os quatro:

> **(a)** publicar a versao local, se ela e a correta (um commit seu no branch `mobiwac`);
> **(b)** deixar como esta, se a versao publicada e a que gerou os numeros do artigo — que e o caso
> a favor de deixar quieto, e o mais provavel para `region_match_tost.py`, com 2 linhas de diferenca.
>
> A prosa do Apendice A **nao depende disto**: ela ja diz que os scripts estatisticos *"are part of
> the working repository and are supplied on request"*, o que nenhuma das leituras acima torna falso.

> **AUTHOR:** Vamso publicar as alterações na brnach do mobiwac.

### 2.18 Um `refs/notes/commits` foi para o `origin` sem eu ter pedido, e a decisao de remover e sua

**Medido em 2026-07-30, nada foi alterado no remoto.** `git ls-remote origin | grep notes` retorna
`refs/notes/commits` apontando para `99c0a34b1a`, identico ao ref local. Ele **esta publicado**.

**Como foi.** Nenhum comando meu pediu isso. O git tem uma configuracao (`notes.rewriteRef` /
`remote.origin.push` com refspec ampla, ou `push.default` com notes habilitados) que empurra
`refs/notes/*` junto de um push comum; o stderr de um dos meus pushes mostrou
`* [new reference] refs/notes/commits`. Um sub-agente reportou isso por conta propria, incluindo o fato de ter
subestimado o escopo na primeira vez que descreveu.

**O que ha nesses notes: 15 anotacoes, e todas sao correcoes de mensagens de commit minhas.** Cada uma diz que uma frase
de commit era falsa e qual e a medicao correta — a convencao deste repositorio para nao reescrever historia. Sao,
literalmente, o registro dos meus proprios erros.

**Por que provavelmente nao e grave — agora medido em TODOS, nao em seis.** A primeira versao deste item afirmava "os 14
commits anotados nao estao em nenhum branch do origin" a partir de uma sondagem que rodou com `head -6`: **oito nunca
foram checados**, e a recomendacao abaixo repousava nessa generalizacao. Re-medido sem o `head`, e a contagem tambem
estava errada — sao **15** notes, nao 14:

```
for h in $(git notes list | awk '{print $2}'); do
  git branch -r --contains "$h" | grep -c "origin/"; done
# checked=15  on_public_branch=0
```

**15 de 15 verificados, zero em qualquer branch do `origin`.** Sao objetos alcancaveis apenas pelo ref de notes, nao
historia visivel de nenhum branch publico. Quem clonar o repositorio **nao recebe notes por padrao** (precisa de
`git fetch origin refs/notes/*:refs/notes/*`).

> **DECISAO SUA, e eu nao vou tomar por voce.** Tres opcoes:
> 1. **Deixar.** Elas documentam correcoes honestas e nao aparecem em clone normal. Custo zero.
> 2. **Remover do remoto:** `git push origin :refs/notes/commits` — apaga o ref publicado e mantem os
>    notes locais. Uma linha, reversivel (basta empurrar de novo).
> 3. **Impedir que volte:** `git config --local notes.rewriteRef ""` e conferir
>    `git config --get-all remote.origin.push`.
>
> Eu recomendo a **2 + 3** se este repositorio for ficar publico com a defesa, e a **1** se ele
> permanecer privado. Nao executei nenhuma delas porque mexer em ref publicado e sua alcada.

> **AUTHOR:** 2+3

### 2.19 Quatro numeros do registro de fechados nao reproduzem; um tem tres respostas

**O que e.** O item 1.2 do `_archive/PENDENCIAS_RESOLVIDOS.md` tem nove linhas; cinco conferem. Das quatro restantes,
nenhuma reproduz da arvore viva: comentarios medem 3.614 linhas contra 1.269 afirmadas, o `preamble.tex` tem 14
placeholders contra 3, e `geometry`/`linespread` nao estao nesse arquivo. **Sao velhos, nao errados** — medidos na
rodada 6, contra uma arvore que desde entao ganhou um apendice e perdeu o `0_main.tex`. O defeito duravel: nenhum
registra **contra qual estado da arvore**
foi tomado, e medicao sem isso so pode ser re-tomada, nunca re-conferida.

> **DECISAO SUA, e e uma so:** qual convencao de contagem de palavras vale para o Resumo e o Abstract
> no deposito? Ha **tres** respostas (310/271 no relatorio, 312/277 de uma esteira, 345/307 do meu
> instrumento) porque ha tres convencoes. Um numero impresso no seu deposito e seu. Diga a convencao,
> eu fixo uma, aplico, e ponho o comando no arquivo.

> **AUTHOR:**  310/271 no relatorio, eu não entendi porque há 3, mas o quantidade de plavras no resumo hoje são essas.

### 2.20 O Cap. 4 italiciza ingles corriqueiro 153 vezes, e este item DESAPARECEU do tracker sem decisao

> **ESTE ITEM FOI PERDIDO, nao resolvido.** Ele existiu ate `1ef83867` (2026-07-28) e saiu do arquivo
> naquele commit **sem ir para o `_archive/PENDENCIAS_RESOLVIDOS.md` e sem uma decisao sua**. O titulo
> dizia explicitamente *"e uma decisao sua"*. Reencontrado em 2026-07-30 varrendo as 63 revisoes do
> tracker por titulo, nao por numero — porque os numeros foram reciclados em tres renumeracoes.

**Re-medido agora, na prosa viva (comentarios removidos), e os numeros continuam praticamente iguais aos de dois dias
atras:**

| capitulo   | `\emph`/`\textit`  |
|------------|--------------------|
| Cap. 1     | 6                  |
| Cap. 2     | 6                  |
| Cap. 3     | 23                 |
| **Cap. 4** | **153** (eram 155) |
| Cap. 5     | 10                 |
| Cap. 6     | 0                  |

Mais italicizados no Cap. 4: `embedding` 18, `baseline` 16, `encoders` 15, `encoder` 14,
`embeddings` 12, `check-ins` 7. **E inconsistente consigo mesmo** — a mesma palavra aparece nas duas formas: `encoder`
italico 14 / romano 8, `encoders` 15 / 7, `baseline` 16 / 4, `embedding` 18 / 1.

**A causa e legitima, a consequencia nao.** Isso vem do artigo em portugues, onde italicizar estrangeirismo e a pratica
correta. Num capitulo **em ingles** a mesma marcacao nao marca mais estrangeirismo: le-se como enfase numa palavra que
nao tem nenhuma, e o proprio capitulo se contradiz.

> **DECISAO SUA, e continua sendo. Tres caminhos:**
> 1. **Deixar como esta.** O Cap. 4 e capitulo de artigo publicado; a marcacao veio de la. Custo zero,
>    mas um leitor em ingles ve enfase onde nao ha.
> 2. **Remover o italico de vocabulario corrente** (embedding, baseline, encoder e plurais), mantendo
>    italico so em termo tecnico em primeiro uso. ~90 substituicoes, mecanicas, e eu registro como
>    partida de errata no Apendice B por ser capitulo publicado.
> 3. **Uniformizar sem remover:** escolher uma forma por palavra e aplicar. Resolve a contradicao
>    interna sem mudar a densidade de italico.
>
> Eu recomendaria a **2**, e nao aplico nada sem voce: e prosa de artigo publicado e o proprio item
> dizia que a decisao e sua. `WRITING_LAW` nao cobre italico de estrangeirismo em capitulo traduzido,
> entao nao ha regra para eu invocar.

> **AUTHOR:**  2.

### 2.21 O segundo ponto do seu orientador: como os termos entram

**O que e.** Ele escreveu *"soa um pouco estranho o jeito que alguns termos sao inseridos (marquei alguns la)"*. O
buraco concreto que o item registrava — o revisor de estilo nunca ter lido o
`articles/[mobiwac]/GLOSSARY.md`, que vence para o Cap. 5 — esta **fechado**: revisor re-rodado em 28/07, as tres
proibicoes da §3 ausentes, as 28 da §4 medidas (22 ausentes, 6 dentro da condicao, e **uma violacao real corrigida**:
`region head` -> `region output`).

**Nao medido, e nao vou dizer que esta:** §6, §7 e §8 do glossario sao julgamento de estilo, sem grep que decida.

> **DECISAO SUA, pequena:** ele marcou termos num PDF que **eu nao tenho**. Me passe as marcacoes e eu
> trato uma por uma.

*Medicao termo por termo: [`_round8/29_pendencias_detail.md`](_round8/29_pendencias_detail.md).*

> **AUTHOR:** `license the verbs` em fundamentação téorica. Foi o unico termo que ele marcou e comnetou para ter cuidado
> com o termos.

### 2.22 Apendice F: o revisor de excelencia aprova, mas a assinatura e sua

**(A) O que e.** O apendice F carrega `[NEEDS SIGN-OFF]` desde a rodada 7 porque faz duas afirmacoes novas que nao estao
na whitelist de nenhum artigo: a mecanistica (ortogonalidade explica por que nenhum balanceador melhorou o peso fixo) e
a de extensao (o diagnostico serve para outras arquiteturas). O revisor de excelencia leu o capitulo inteiro e o
apendice no PDF renderizado e respondeu a pergunta que eu fiz a ele: **aprovaria a assinatura** com duas correcoes,
ambas ja aplicadas nesta rodada (`e6ae1c0d` tirou a clausula do hard sharing, `3623dde8` corrigiu "replacing the sharing
scheme" para
"varying the gradient balancer"). Ele tambem re-derivou as 63 celulas da Tabela 11 a partir do parquet e todas
reproduziram.

**(B) O que ja foi feito.** As cinco revisoes de personas rodaram sobre os quatro arquivos alterados (2_fundamentals,
apx_f_cosine, 1_introduction, tables/frame/cosine). Nove itens REQUIRED aplicados, cada um verificado por mim na fonte
antes de aceitar. Relatorios em `_round9/38_style_r9b.md`,
`39_mtl_r9b.md`, `40_readability_r9b.md`, `41_ai_tells_r9b.md`, `42_excellence_r9b.md`.

**(C) O que eu preciso de voce.** A assinatura, ou a instrucao de tirar o apendice. Ele nao e um erro para consertar: e
material novo que so voce pode aprovar antes de ir ao orientador. Custo de tirar: um
`\input` comentado, mais o ponteiro novo no Cap. 6 (`3623dde8`) e a referencia em 2.3 (p. 23).

> **AUTHOR:** I approve the appedix F you can remove the `[NEEDS SIGN-OFF]`, but i have some considerations:
> 0. You must rename all the appendix, so the letters respect the correct order in the current version of the text. In
     this case this appendix would be the letter D.
> 1. On the `an end up worse at both than two dedicated models are at one each.` we shoud add a reference to it.
> 2. Don't say `stranger result` without cite some ref, but my main take is not to say this, is your jugdment and can
     cause questions in the reviewer
> 3. This phrase: "That is why varying the gradient balancer changed so little in the first study, and why changing the
     representation changed so much in the second and third.". Have a huge erros in the first sutdy we use 2 diffents
     tasks, than the last third study of mobiwac, also the arch of the MTLnet was different so we can relate the results
     of this appendix to the first and secon study.
> 4. On the "the cosine of the angle between the two resulting gradient vectors was recorded" we should cite some
     article/studie/document that show this apporach
> 5. Exclude this: "so one configuration on one dataset is five series of fifty values, and two of Florida’s
     carryapartialre-run ontopoftheirs.", is over detail.
> 6. This is a implementation detail let's exclude also: "That chapter reaches the same conclusion from a smaller
     development-time measurement,onanearlierdatapreparationandoverfourseedsratherthanper-epochseries,so the
     twosetsofnumbersarenotinterchangeableand this appendix supersedes nothingthere."
> 7. on the: "Every test below therefore runs on five fold... Whereacountofobservationsappearsitdescribesthedata,
     notatest’ssample size." this part explain the how the experiments was runned is importante, but we can simplify the
     details, don't need to do a lot of tech explanation, also about the florida I belive explain it in much details can
     cause confusion, lets try to be more straight.
> 8. The phrase: "feature needs saying plainly" this is britisher english and this boke on of the agents_guartrails. If
     this is not in the guardrails add this and eval in the rest of the text if we have similar stuctures that are
     britisher.
> 9. This phrase:  "Two departures from that flat picture appear" is pure A.I, we cna be more simple and direct.
> 10. This phrase: "both are worth reporting rather than smoothing" we don't need to say this, appears as we are try to
      hide somthing we just need to report.
> 11. On the: "A 𝑡-test does reject on both datasets and for both
      departures,butatfiveobservationsthatrestsentirelyonassumingnormality,andthisappendix
      willnotacceptforoneclaimabasisitrejectsforanother.", you don't say which datasets, and this phrase is confusing
      and hard to read for whom don't have a lot of knowhow. We can try to improve the rest of this paragraph
> 12. The phrase: "Both point away from trouble in any case. A positive cosine is mild cooperation, not conflict, and
      the decline stays inside the margin throughout while moving toward zero rather thanawayfromit." is well written,
      but is not natural for a non native writer in english, and force a non native read more than once to understand.
> 13. On the paragraph that starts with: "The second is about the arc of the three studies.", we need to take care cause
      the first two studies was diferrent tasks that these ones that we are testing int eh appendix F. Maybe remove
      this.
> 14. Somthing that worths to mention, don't need fither explanation, in the F.3 is that besides the gradients don't
      addup, this don't means that the tasks are not sharing their knowladge since exstie otehr mechanims like the gate
      in the arch and so on...

### 2.23 Cinco itens RECOMMENDED das revisoes que eu nao apliquei

**(A) O que e.** Ficaram por decisao de escopo, nao por esquecimento. Cada um tem quote e pagina no relatorio citado.

| id   | onde     | o que                                                 | por que nao apliquei                        |
|------|----------|-------------------------------------------------------|---------------------------------------------|
| R-3  | p.26-27  | §2.5 nao retoma a pergunta que §2.3 deixa aberta      | mexe em prosa que nao mudou nesta rodada    |
| R-5  | p.23     | uma frase de 66 palavras com tres pontuacoes fortes   | reescrita de estilo, sua chamada            |
| R-6  | p.23     | so uma referencia no corpo aponta para o Apendice F   | pode ser deliberado                         |
| EX-6 | p.101    | uma comparacao entre datasets que o texto nao precisa | `40_readability` e `42_excellence` divergem |
| EX-9 | p.23, 99 | densidade de figuras de linguagem                     | OPTIONAL nos dois relatorios                |

**(B) O que ja foi feito.** Triados e medidos; nada aplicado.

**(C) O que eu preciso de voce.** Diga quais valem e eu aplico. Nenhum e um erro de fato.

> **AUTHOR:** Aplique o R-3,5,6 e o EX-6, seguindo a recomendação do 42_excellence, não aplique o EX-9.

### 2.24 Um `towards` britanico em prosa publicada do CBIC, e a saida e uma linha de errata

**O que e.** Sua queixa 8 do item 2.22 (o `needs saying plainly`) gerou a lei de registro e o gate 25
(`check_register.py`). Varridos os 54 `.tex` vivos mais o `references.bib`, **doze linhas de achado em onze sitios**
(uma frase pode disparar duas regras): **5** grafias britanicas, **1** construcao britanica (a sua) e **6** formas de
fraseado. **Seis eram nossas e foram corrigidas**; **cinco** estao no Apendice F e a outra esteira ja as fechou.
**Sobrou uma, e ela e sua**, porque esta em prosa publicada. (As cinco do apendice do cosseno estao contadas aqui como "Apendice F", que era a letra quando esta varredura mediu; a outra esteira aplicou o seu ponto 0 e **reletrou para Apendice D** no commit `4eea637a`. O arquivo continua `chapters/apx_f_cosine.tex` e o gate e ancorado no caminho, nao na letra.)

*(Este bloco dizia **nove ocorrencias** e "quatro/quatro/uma". Estava errado, e o erro foi pego por revisao, nao por
mim: eu somei categorias de cabeca em vez de contar as linhas do instrumento. Medido agora rodando o gate sobre a arvore
do `06529ed6` com o `OPEN_REGISTER` vazio, para que nada fique retido e todo achado imprima: `rc=1`, "6 British
spelling/construction hit (s) and 6 hard-phrasing shape (s)", 12 linhas. A conferencia fecha nos dois sentidos: 6
corrigidas + 5 do Apendice F + 1 sua =

12. O detalhamento linha por linha esta na §1.3 do relatorio.)*

`chapters/3_cbic/conclusion.tex`, p. 43 do build de defesa:

> "The representation learned by the shared layers might have become biased **towards** the features
> required for the simpler, static classification task"

**Medido, nao suposto:** a frase e substring literal de
`articles/CBIC___MTL/sections/conclusion.tex:13`, e este `towards` e a **unica** forma britanica em toda a fonte
publicada do CBIC (zero `-our`, zero `-ise`, zero `whilst`); CoUrb-EN e MobiWac nao tem nenhuma.

**Por que nao apliquei.** Pela NORTH_STAR §5.7, mudar uma palavra de prosa publicada e decisao sua, e esta e questao de
vocabulario, nao de correcao. O gate mantem o achado **aberto por nome** e falha se a entrada ficar obsoleta, entao nao
se perde em nenhuma das duas saidas.

> **DECISAO SUA, e e uma linha:** (**a**) trocar por `toward` e acrescentar uma linha em
> `tables/cbic/errata_wording.tex`, que ja carrega **quatorze** linhas exatamente desta classe
> ("By leveraging shared information" -> "By using shared information"), todas sob a legenda *"claim
> strength unchanged or reduced, never raised"*; a troca nao muda alegacao nenhuma. (**b**) deixar
> como esta, e a entrada do registro aberto do gate passa a ser o registro permanente da decisao.

**Um segundo ponto, tambem seu, e eu deliberadamente NAO fiz gate dele.** O ponto final fica **fora**
da aspa de fechamento em **13 sitios** (`tables/cbic/errata.tex` 5, `tables/courb/errata.tex` 3,
`chapters/3_cbic/method.tex` 2, `tables/cbic/errata_wording.tex` 2, `chapters/apx_b_errata.tex` 1). O estilo americano
poe dentro. **Todos os 13 estao em tabelas de errata onde a string citada e a evidencia**, e mover um ponto para dentro
de uma citacao altera a citacao. E decisao sobre a convencao da errata, nao erro de grafia. Se quiser, e mecanico e faco
os 13 numa passagem.

*Forense: [`_round9/44_register_law.md`](_round9/44_register_law.md) (a varredura com contagem por arquivo, as regras, e
o transcrito de validacao do gate nos dois sentidos).*

> **DECISAO SUA:** ______

## §5 · Levantados do `CODEX_AUDIT.md` quando ele foi arquivado (2026-07-29)

> **NOVE DOS DEZ ESTAO FECHADOS e foram movidos para `_archive/PENDENCIAS_RESOLVIDOS.md` em
> 2026-07-30.** **Sete** deles estao cobertos por uma sonda do gate `check_audit_claims.py`, que
> **falha** se a correcao sair do documento — foi exatamente essa a licao desta rodada, em que oito de
> nove estavam marcados como aplicados sem estar. **Dois nao tem sonda**, e a primeira versao deste
> cabecalho dizia que todos tinham: **5.6** foi verificado direto no render (as duas datas do Gowalla
> imprimem na p. 79) e **5.10** e um registro de nao-pendencias, nao uma afirmacao do documento. A
> tabela por item esta no banner do arquivo. O decimo (COD-018, credito por papel no CoUrb) foi
> **retirado por voce**, e o gate carrega a sua frase para que ninguem o "termine" por engano.
>
> Sobrou **um**, abaixo, e ele espera uma escolha sua, nao trabalho meu.

Voce pediu: *"About the codex_audit if we finish with it archive it or delete, and if some point still pending my
approval or I need to be aware add in the pendencias."* Fiz a varredura dos **26 itens** (18 COD- mais 8 NUM-), das 16
caixas `DECISAO` que voce escreveu no arquivo, e da tabela de desfecho da rodada 6. O arquivo esta agora em
[`_archive/CODEX_AUDIT.md`](_archive/CODEX_AUDIT.md), inteiro.

**O resultado, e ele nao e agradavel.** A tabela de desfecho marca 22 dos 26 itens como aplicados. Conferi cada um **no
PDF renderizado e no fonte vivo**, nao na tabela, e **nove instrucoes suas nao estao no documento**. Nao e que estejam
mal aplicadas: as frases que voce mandou mudar continuam palavra por palavra como estavam. Cinco delas a tabela de
desfecho declara "APLICADO".

Nao sei se ninguem chegou nelas ou se cairam entre escopos de trilhas — o `CODEX_AUDIT.md` §6 as listava como
"corrigiveis sem o autor" e o §7 como "precisa do autor", e a rodada 6 tinha oito trilhas. O que eu sei e o que a
medicao mostra. **Cada item abaixo traz o comando que o mede**, para voce nao ter que acreditar em mim.

Para conferir os nove de uma vez, do diretorio da dissertacao:

```bash
cd /Users/vitor/Desktop/mestrado/ingred/articles/dissertacao
for p in 'leakage-guarded' 'equivalence is well powered' 'unbalanced result for the MTL and single' \
         'revise that verdict by changing the input representation' 'mean reciprocal rank'; do
  printf '%-58s %s\n' "$p" "$(grep -rl "$p" src/chapters/*.tex src/chapters/*/*.tex 2>/dev/null | tr '\n' ' ')"
done
# EXPECT: lines=5
```

Cinco linhas, cada uma nomeando o arquivo onde a frase ainda vive. Se uma linha vier vazia, aquele item foi resolvido
depois desta varredura.

### 5.6b A premissa da sua decisao 5.6 nao e o que os arquivos mostram — resolvi imprimindo AS DUAS datas

**Medido em 2026-07-30, nos cinco parquets que este trabalho consome.** Sua decisao no 5.6 foi *"Busque pelo que o
artigo original cita e vamos usar isso em ambos. Inclusive ambos usaram o mesmo recorte nao houve diferenca."* A
primeira metade foi cumprida: o `cho2011gowalla` foi aberto em primeira mao (PDF dos proprios autores, Secao 2, p.2) e
ele diz **Fev 2009 a Out 2010**.

**A segunda metade nao se sustenta.** Os cinco estados usados nao param em Out 2010:

| estado     | primeiro check-in | ultimo check-in | n         |
|------------|-------------------|-----------------|-----------|
| Alabama    | 2009-03-18        | 2011-07-27      | 113.846   |
| Arizona    | 2009-03-26        | 2011-07-04      | 236.450   |
| Florida    | 2009-03-13        | 2011-08-11      | 1.407.034 |
| Texas      | 2009-01-21        | **2011-08-16**  | 4.089.892 |
| California | 2009-01-24        | 2011-08-14      | 3.171.380 |

Uniao: **2009-01-21 a 2011-08-16** — dez meses depois da janela que o artigo declara.

**Por que isso importa e nao e frescura.** A frase esta no Cap. 6 sob a limitacao *"Data vintage"*. Ali o leitor le a
data como **a safra dos dados que voce usou**, nao como uma nota sobre o que outro artigo coletou. Imprimir so Fev
2009–Out 2010 subestimaria o proprio corpus em dez meses.

**O que eu fiz.** A frase agora carrega as duas datas: o que os autores relatam, e o que a extracao daqui abrange, com a
medicao completa e o comando no comentario de proveniencia do
`6_conclusion.tex`. Nao e uma correcao da sua decisao — voce estava decidindo **qual fonte citar**, e essa parte esta
cumprida.

> **DECISAO SUA.** Se voce preferir imprimir **so** a janela do artigo, a clausula depois da virgula e
> a que sai, e eu removo. Marcado com `[NEEDS SIGN-OFF: PENDENCIAS_RESOLVIDOS 5.6 (arquivado 2026-07-30), round8]` no
> fonte.

> **AUTHOR:** Vamos usar só as datas que encontramos no database de 2009 a 2011, pode omitir que no artigo eles comentam
> que é de 2009 a 20010. O need sing-off assim como os demais já resolvidos que estão no latex pode ser removidos não
> precisam fica lá. Se quider documentar isso tem que ser em algum lugar do src_util.

## §6 · As decisoes que sairam do `CONSIDERATIONS.md` (round 9)

> **De onde isto vem.** `CONSIDERATIONS.md` foi reescrito em 43 blocos com ID estavel: **20** itens eu
> aplico, **22** precisam de voce e estao aqui, **1** esta bloqueado numa verificacao que falhou.
> **7 edicoes ja estao aplicadas e conferidas no PDF renderizado** (nos dois sentidos: texto novo
> presente, texto antigo ausente), e o **FAB-01 ja estava satisfeito** — conferido, nao aplicado, porque
> nao havia o que editar. Os outros 12 esperam a outra esteira soltar o `2_fundamentals.tex` ou uma
> linha sua no `GLOSSARY` (`_round9/33_apply_plan.md`). O bloco
> completo de cada item (citacao, status no fonte vivo, meu raciocinio, onde renderiza, probe, commit da
> medicao) esta em [`CONSIDERATIONS.md`](CONSIDERATIONS.md).
>
> **Medido contra o commit `d4078c75`** (`make check` rc=0, 22 gates, lido direto e nao por pipe;
> `make selftest` rc=0). **Nao apliquei nada em capitulo nenhum nesta passagem** — a divisao e o produto.
>
> **O que voce nao pode saber sem a medicao:** das 41 passagens localizaveis citadas pelos dois revisores,
> **9 estao obsoletas** — todas do Fabrício, todas no `0_main.tex` (extinto em 2026-07-29), e sao exatamente
> FAB-02 a FAB-10. Tres delas (**FAB-04, FAB-05, FAB-08**) ja estao satisfeitas: o texto que ele pediu para
> tirar nao existe mais.
>
> **Substitui o §2.8**, que agora registra o que foi feito.

### 6.1 Onde eu discordo do revisor — tres itens, e a decisao e sua

#### FAB-17 — the fourth-task paragraph is confusing and could go

**Citacao:** exata — `chapters/1_introduction.tex:61-66`. **Renderiza em:** 1_introduction.tex:61-64

**Minha leitura:** DISAGREE, and this is the one I would push back on. The paragraph he would delete is the
AVAL-signed-off task-pair acknowledgment (a) and the three-legged task-choice defense (b) recorded in NORTH_STAR 6 Ch.1
beat 4. It exists because a prior review round required the document to state plainly that the task pair EVOLVED and to
defend the task choice. Deleting it reopens both. If it is confusing, the fix is to split it into two shorter sentences,
not to remove it.

1. **Remover o paragrafo, como ele pede** — REABRE dois pontos com aval registrado (NORTH_STAR §6 Ch.1 beat 4 (a) e
   (b)): o reconhecimento de que o par de tarefas MUDOU, e a defesa da escolha das tarefas. Uma rodada anterior exigiu
   os dois
2. **Dividir em duas frases curtas e manter o conteudo** — atende a queixa real (confusao) sem reabrir o aval; ~30 min
3. **Levar a ele com o registro do aval na mao** — ele decide sabendo o que o paragrafo protege

> **DECISAO SUA:** Eu também discordo, se compravarmos que esse pragrafo está correto. Ele fala que uma quarta tarefa
> aprece no artigo ? Isso está errado, CBIC e COURB: poi category classification and next-category, já no mobiwac:
> next-region and next-category, ou seja são 3 tarefas a um todo. Além disso eu acho que a frase:
> `with the static classification of a place's category`, para explicar a tarefa de classificação de categoria de um poi
> desconhecido, não está bem explicada é esta confusa. Isso é outro ponto a tarefa no 2 primeiros artigos era
> classificação de categoria de um poi deconhecido, avalise se o restante do texto está tratando isso dessa forma.
> Também temos que avaliar se esse error de 4 tarefas não aparece em outros lugares.

#### FAB-27 — the table's chapter references are wrong

**Citacao:** exata — `tables/frame/lineage.tex:5-8 (caption) -> Table 2.1`. **Renderiza em:**
tables/frame/lineage.tex -> Table 2.1

**Minha leitura:** disagree as stated; a real problem underneath. Checked directly: ch:courb -> 4 and ch:mobiwac -> 5 in
build/main-aux, both resolve, nothing dangling. What is wrong is that ONE column mixes bracketed citations and chapter
pointers, so the chapter numbers read as mis-rendered citations. Also ST-MTLNet HAS a published record
(paiva2026stmtlnet, already in the bib and cited in Ch.4/Ch.5) and the table shows it only as 'Chapter 4'. Two fixes are
possible and they differ in cost.

1. **So dividir a coluna em "Reference" e "Where in this dissertation"** — resolve a ambiguidade real (a coluna
   mistura [41] com "Chapter 4"); ~1h com rebuild
2. **Dividir a coluna E citar o `paiva2026stmtlnet` na linha do ST-MTLNet** — a entrada JA existe no .bib e ja e citada
   nos Cap. 4 e 5; a tabela hoje mostra so "Chapter 4" para um artigo publicado com DOI
3. **Perguntar a ele primeiro** — as `\ref` estao corretas (`ch:courb`->4, `ch:mobiwac`->5, lidas do build); se ele viu
   `??` num PDF, era .aux velho, nao erro de fonte

> **DECISAO SUA:** Vamos deixar em como está, voce já corrigiu o que precisava. Algo que temos que tomar cuidado e que a
> tabela está passando da margem.

#### GER-02 — DGI and HGI are presented as counterpoints, and they are not

**Citacao:** parafrase (ele descreveu, nao citou). **Renderiza em:** 2_fundamentals.tex:143-165 + Table 2.1

**Minha leitura:** partially disagree, and the disagreement is mine to state, not to settle. The prose already says HGI
'builds directly on' DGI and 'extends' the objective, which is inheritance language, not contrast. I do not think the
paragraph sets them against each other. His reading has a real cause though: the lineage table lists them as adjacent
rows under 'What it added'. Also his framing 'HGI e uma aplicacao' would UNDER-credit Huang et al., and would sit badly
beside our claim that Check2HGI extends the same hierarchy.

1. **Uma clausula explicitando os papeis (DGI = o objetivo; HGI = o objetivo instanciado na hierarquia
   POI-regiao-cidade)** — atende a causa real da leitura dele sem mexer numa descricao correta; ~20 min
2. **Reescrever como ele sugere, chamando o HGI de "aplicacao"** — EU DISCORDO: sub-credita Huang et al., que contribuem
   um objetivo novo, e fica mal ao lado da nossa propria frase de que o Check2HGI *extends* a mesma hierarquia
3. **Nao mexer** — a prosa ja diz "builds directly on" e "extends"; mas dois leitores tropecaram

> **DECISAO SUA:** De fato isso já tá bem explicado no texto, então não vamos mexer. Algo que podemos melhorar e
> explicação do hgi e como ele
> funciona, vide que essa é uma das aboardagens mais importantes para as contribuições da dissertação.

### 6.2 Onde o pedido colide com uma regra de honestidade do proprio documento

#### FAB-09 — simplify the results description; keep it higher level

**Citacao:** ALTERADA — `content.tex`. **Renderiza em:** content.tex:91-94

**Minha leitura:** partly satisfied. The joint-best gloss he wanted simplified is GONE from the Resumo; the numbers (5,3
a 9,4) and the TOST clause remain. Cutting them further trades honesty rules (every number carries its convention)
against his readability ask.

1. **Cortar a faixa 5,3-9,4 e o TOST do Resumo, deixando "supera nas seis" e "equipara-se nas outras duas"** — atende o
   pedido; MAS o WRITING_LAW §3 exige que todo numero carregue convencao, e que "equipara-se" venha com o teste
2. **Manter os numeros e cortar so a glosa de joint-best** — ja feito: a glosa saiu na reescrita de 28/07
3. **Manter como esta** — contraria o pedido dele

> **DECISAO SUA:** Ele revisou uma versão antiga do Resumo, essa nova versão já cumpre bem o intuito final. Vamos manter
> em como esta.

#### FAB-22 — the result detail does not belong in the introduction

**Citacao:** exata — `chapters/1_introduction.tex:130-133`. **Renderiza em:** 1_introduction.tex:130-132

**Minha leitura:** his call: it collides with an honesty rule. He wants the result detail out of the introduction.
WRITING_LAW 3 requires every number to carry its reference point and convention, and the region verbs to be bound to
their tests. Dropping 'four of six ... TOST' from the intro removes exactly that binding, so the sentence must either
keep the test or stop making the comparative claim.

1. **Tirar "quatro dos seis ... TOST" da introducao** — atende; MAS remove exatamente a amarra verbo-teste que o
   WRITING_LAW §3 exige, e a frase continua afirmando comparacao
2. **Tirar a comparacao TODA da introducao e deixar so a pergunta** — coerente: sem afirmacao, sem teste
3. **Manter** — contraria o pedido

> **DECISAO SUA:** Eu acho que não precisamos tirar tudo, o ponto e o excesso de inforamção. Podemos comentar algo
> somente sobre o fatoq ue os experimento foram realizados em 5 estados do estados unidos e um dataset non-U.S para
> metritos de generalização. Também, podemos comentar que usamos métodos estatisticos para comprovar essa difernça.
> Enfim a ideia e ser mais deireto. Não remover tudo.

#### FAB-03 — put the open question in the present tense

**Citacao:** ALTERADA — `content.tex`. **Renderiza em:** content.tex:76

**Minha leitura:** agree on substance; the sentence changed. He asked for the present tense. The live wording is
'permanecia em aberto' (imperfect), not the 'era' he quoted. Present tense would read 'permanece em aberto'. It is a
claim-time question: the document elsewhere time-indexes CBIC/CoUrb conclusions, and the abstract saying 'permanece'
asserts the question is open TODAY, after the dissertation answered it.

1. **Presente: "permanece em aberto"** — uma palavra, nos dois idiomas; MAS afirma que a pergunta esta aberta HOJE,
   depois de a dissertacao responder
2. **Manter o imperfeito "permanecia"** — preserva a regra de indexacao temporal do WRITING_LAW §3; contraria o pedido
   dele
3. **Reformular: "era uma questao aberta no inicio desta pesquisa"** — explicita o ponto de referencia, atende o sentido
   dele sem a leitura de presente

> **DECISAO SUA:** Manter o imperfeito "permanecia"

#### FAB-18 — put the open question in the present tense

**Citacao:** exata — `1_introduction.tex:92`, `content.tex:156` (EN) e a contraparte PT em `content.tex:76`. **Renderiza
em:** 1_introduction.tex:92 + content.tex:156 (EN) + content.tex:76 (PT)

**Minha leitura:** agree on substance; same decision as FAB-03, not a separate one. He asks for the present tense here
and at FAB-03; it is one claim in three places (1_introduction.tex:92, content.tex:156 EN, content.tex:76 PT). I first
bucketed this as 'apply' and FAB-03 as 'decide', which cannot both be right: the same sentence cannot be mine to edit in
English and his to rule on in Portuguese, and the Resumo/Abstract pair must stay claim-for-claim identical (WRITING_LAW
6). The substantive question is the one FAB-03 carries: 'is/permanece' asserts the question is open TODAY, after this
dissertation answered it, which is what the time-indexing rule exists to prevent. Caught while applying it, not while
sorting.

1. **Presente nos tres sitios: 'is an open question' / 'permanece em aberto'** — atende ao pedido literal dele nos dois
   idiomas; MAS afirma que a pergunta esta aberta hoje, depois de o documento responde-la, que e exatamente o que a
   regra de indexacao temporal existe para evitar
2. **Reformular os tres para explicitar o ponto de referencia: 'was an open question at the start of this research' / '
   era uma questao aberta no inicio desta pesquisa'** — e o que eu recomendo: da a ele o efeito que ele quer (o texto le
   como se escrito no inicio) sem afirmar que continua aberta; ~15 min, tres sitios, e o par Resumo/Abstract continua
   identico
3. **Manter o passado nos tres** — preserva a regra; contraria o pedido explicito dele, duas vezes

> **DECISAO SUA:** No item FAB-03 eu optei por continuar no imperfeito. Assim para manter a concistencia acho que temos
> que ir com a escolha 2.

#### GER-11 — the task non-conflict finding needs stronger evidence, and generalizes into future work

**Citacao:** parafrase (ele descreveu, nao citou). **Renderiza em:** 6_conclusion.tex + Appendix F

**Minha leitura:** agree on both halves. The non-conflict evidence is thinner than the claim: one mean (+0.001), four
seeds, four Gowalla states of which GA is not one of the six datasets, taken during development on an earlier data
preparation, with no spread and no Istanbul. A mean cannot distinguish 'consistently orthogonal' from 'strongly
conflicting in both directions and cancelling'. Either strengthen it or downgrade the sentence.

1. **Recomputar o cosseno na configuracao reportada, nos seis conjuntos incluindo Istambul, com dispersao por
   conjunto** — transforma anedota em resultado, e o Germano pediu exatamente isso; custo de GPU, e o item 2.9 ja tem
   tres conjuntos parados
2. **Rebaixar a frase do Cap. 6 para "uma observacao de tempo de desenvolvimento, oferecida como consistente com a
   interpretacao e nao como evidencia dela"** — zero custo de compute, honesto, e e o que eu faria primeiro; a frase
   atual sustenta "essencialmente ortogonais" com UMA media, quatro sementes, quatro estados do Gowalla dos quais GA nao
   e um dos seis, sem dispersao e sem Istambul
3. **As duas: rebaixar agora, fortalecer se a GPU liberar** — protege a defesa e mantem a porta aberta

> **DECISAO SUA:** Já corrigimos isso na versão mais atual do texto, o appendix F está bem mais maduro e com os
> resultados mais consolidados. Acho que não precisamos atuar quanto a esse ponto.

### 6.3 Onde a passagem citada nao existe mais, e o pedido precisa de nova redacao

#### FAB-02 — start a new sentence on the shared-history point

**Citacao:** DESAPARECIDA — `content.tex`. **Renderiza em:** content.tex:73-74

**Minha leitura:** his edit no longer applies as written. The clause he quoted is GONE: the Resumo was cut and rebuilt
on 2026-07-28. The live sentence is 'as duas tarefas leem o mesmo histórico, portanto um único modelo poderia
aprendê-las em conjunto'. His point (start a new sentence, foreground the data argument) can still be applied to the new
sentence, but it is a rewrite of text he has not read.

1. **Aplicar o espirito no texto NOVO: quebrar a frase do Resumo em duas, comecando a segunda em "Em termos de dados, as
   duas tarefas..."** — reescrita de uma frase que ele nao leu; o Resumo e o Abstract tem que mudar juntos ou se
   contradizem
2. **Nao aplicar e registrar que a passagem citada nao existe mais** — zero custo; ele pode repetir o ponto ao ler o
   build novo

> **DECISAO SUA:** Vamos descartar esse ponto, estamos achando um equilibro bom para o resumo.

#### FAB-07 — name the study instead of the stage of the research

**Citacao:** DESAPARECIDA — `content.tex`. **Renderiza em:** content.tex:82

**Minha leitura:** agree; his own note has a '??'. He wrote 'no primeiro estudo (??)', i.e. he was unsure. The quoted
phrase is GONE; the live text reads 'naquela configuração'. Whether to name the study or the configuration is a
claim-scope choice: 'no segundo estudo' is where the diagnosis was made, not the first.

1. **"no segundo estudo"** — nomeia o estudo, que e o que ele queria; e o SEGUNDO, nao o primeiro que ele chutou com "
   (??)"
2. **Manter "naquela configuracao"** — preserva o escopo exato (a conclusao vale para aquela configuracao, nao para o
   estudo todo)
3. **As duas: "no segundo estudo, naquela configuracao"** — mais longo, e o Resumo esta no limite de tamanho

> **DECISAO SUA:** Não precisa atuar

#### FAB-10 — the results sentence opens confusingly

**Citacao:** ALTERADA — `content.tex`. **Renderiza em:** content.tex:94

**Minha leitura:** agree on substance; sentence changed. 'e a condição é o achado' is gone; the live sentence is 'A
resposta é, portanto, condicional: se o aprendizado multitarefa ajuda depende...'. His proposed opener ('Como principais
resultados, identificamos que...') is a first-person results framing, a register choice for the PT Resumo.

1. **Adotar o abre dele: "Como principais resultados, identificamos que..."** — primeira pessoa do plural no Resumo; o
   resto do Resumo e impessoal
2. **Manter "A resposta e, portanto, condicional: ..."** — mantem o registro; contraria o pedido
3. **Meio: "Os resultados mostram que o ganho do aprendizado multitarefa depende de..."** — impessoal e direto

> **DECISAO SUA:** Vamos com a 3 opção algo como: "Os resultados mostram que o benefício do aprendizado multitarefa
> depende da representação de entrada e da topologia de compartilhamento entre as tarefas."

### 6.4 Ja satisfeitos — so falta voce confirmar

#### FAB-04 — do not mention the coletanea in the Resumo

**Citacao:** DESAPARECIDA — `content.tex`. **Renderiza em:** content.tex (absent)

**Minha leitura:** his edit is already satisfied. 'coletânea' occurs NOWHERE in the live Resumo or Abstract, and nowhere
in the live tree. Either an earlier round already applied this, or he read a build that predates the rebuild. Confirm he
is content.

1. **Confirmar como satisfeito** — zero; a palavra nao existe no fonte vivo
2. **Perguntar a ele se o build que leu era antigo** — um e-mail; evita ele repetir o ponto

> **DECISAO SUA:** Perfeito, mantemos assim!

#### FAB-05 — drop the negative-result / diagnosis / resolution clause

**Citacao:** DESAPARECIDA — `chapters/1_introduction.tex`. **Renderiza em:** content.tex (absent)

**Minha leitura:** already satisfied. Same as FAB-04: the 'na ordem em que aconteceram' clause is gone.

1. **Confirmar como satisfeito** — zero; a clausula nao existe mais

> **DECISAO SUA:** Perfeito, mantemos assim!

#### FAB-08 — drop the task-pair clause from the Resumo

**Citacao:** DESAPARECIDA — `content.tex`. **Renderiza em:** content.tex (absent)

**Minha leitura:** already satisfied. The task-pair clause is GONE from the Resumo. But note FAB-17 asks to delete the
SAME content from the introduction, where it is AVAL-signed-off. Deciding 08 and 17 together avoids removing it from
both places.

1. **Confirmar como satisfeito no Resumo** — zero
2. **Decidir junto com o FAB-17** — o mesmo conteudo esta na introducao COM aval registrado; decidir separado corre o
   risco de tirar dos dois lugares

> **DECISAO SUA:** Perfeito, mantemos assim! Mas vamos adicionar um comentario sobre o resumo, que uma informação está
> sendo omitida o fato de serem tarefas diferentes entre os dois primeiros e o ultimo artigo.

### 6.5 Detalhe de dados na introducao — ele quer fora, e ha um contraste em jogo

#### FAB-14 — the seven classes belong to the data, not the introduction

**Citacao:** exata — `chapters/1_introduction.tex:52-54`. **Renderiza em:** 1_introduction.tex:52-53

**Minha leitura:** disagree, and it is his call. He says the 7 classes are a property of the data, not the problem. He
is right in principle. But the count is what makes the next-category output space concrete against next region's
hundreds-to-thousands, which is the contrast 1.2 is built on, and Appendix B's static-scope note depends on the taxonomy
being introduced. Cost of removing: the reader meets 'seven' first in 2.4.

1. **Tirar as sete classes da introducao** — atende; o leitor so encontra o numero em §2.4, e o contraste 7 vs milhares
   que §1.2 constroi perde um lado
2. **Manter** — contraria o pedido de um membro da banca
3. **Manter so o contraste sem o numero: "um punhado de classes contra milhares de regioes"** — preserva o contraste,
   tira o dado

> **DECISAO SUA:** ______

#### FAB-15 — census tract and mahalle belong to the data, not the introduction

**Citacao:** exata — `chapters/1_introduction.tex:54-56`. **Renderiza em:** 1_introduction.tex:54

**Minha leitura:** disagree, same reason. Same class as FAB-14: census tract / mahalle is what makes 'region' concrete.
Removing it leaves 'the official geographic unit' undefined until 2.4.

1. **Tirar census tract / mahalle da introducao** — atende; "unidade geografica oficial" fica sem definicao ate §2.4
2. **Manter** — contraria o pedido
3. **Mover para uma nota de rode** — compromisso; o Viegas usa notas assim

> **DECISAO SUA:** ______

### 6.6 Os itens grandes do Germano — custo real, retorno real

#### GER-08 — several concepts have no formal definition, starting with a check-in

**Citacao:** parafrase (ele descreveu, nao citou). **Renderiza em:** new 2.1 subsection

**Minha leitura:** strongly agree; it is a cross-chapter edit. The chapter defines a check-in in prose only and then
writes L_c2p over check-in and place embeddings with no notation for a check-in, a user, a place, a category, or a
region. His reason is the right one: the later chapters' equations need symbols with an origin. But notation must be
checked against Chapters 3-5 AS COMMITTED and every new symbol registered in the fail-closed GLOSSARY first, so this is
not a Chapter 2 edit.

1. **Adicionar o bloco de notacao (check-in, usuario, lugar, categoria, regiao, historico, as tres tarefas) como
   subsecao no fim de §2.1** — o item de maior retorno das duas revisoes: da simbolos as equacoes dos Cap. 3-5. MAS e
   edicao TRANSVERSAL: a notacao tem que ser conferida contra os Cap. 3-5 COMO ESTAO, e cada simbolo novo tem que entrar
   no GLOSSARY (fail-closed) ANTES de aparecer. Estimo meio dia mais a sua aprovacao do registro
2. **Versao minima: definir formalmente so o check-in e as tres tarefas** — ~2h; atende o exemplo que ele deu e deixa o
   resto
3. **Nao fazer nesta rodada** — e o ponto mais forte da revisao do Germano e o precedente do proprio programa (a
   dissertacao dele) o cumpre

> **DECISAO SUA:** ______

#### GER-09 — 2.3 needs MTL formalism, the balancer lineage, and a definition of loss conflict

**Citacao:** exata — `chapters/2_fundamentals.tex:383`. **Renderiza em:** 2_fundamentals.tex:383-508 -> 2.3

**Minha leitura:** strongly agree, and it is the largest item here. 2.3 defines MTL in one prose clause and never writes
the total loss; names eight balancers in eight sentences of identical shape with no taxonomy; credits almost no lineage;
and never defines what conflict IS, which is what PCGrad, CAGrad and Aligned-MTL all act on. That last gap is why
Chapter 6's +0.001 cosine lands with no definition behind it.

1. **Fazer os quatro: a equacao da perda total, agrupar os balanceadores por mecanismo (peso vs cirurgia de gradiente),
   creditar a linhagem, e definir conflito pelo cosseno** — o maior item de conteudo; ~1 dia. Define a quantidade que o
   +0,001 do Cap. 6 mede, que hoje nao tem definicao em lugar nenhum
2. **So a equacao da perda e a definicao de conflito** — ~3h; e o minimo que torna o numero do Cap. 6 legivel
3. **Nao fazer** — a pergunta de pesquisa da dissertacao e se MTL ajuda, e o capitulo nunca escreve o objetivo de MTL

> **DECISAO SUA:** ______

#### GER-10 — the fundamentals need a logical narrative built on formal definition blocks

**Citacao:** parafrase (ele descreveu, nao citou). **Renderiza em:** whole chapter

**Minha leitura:** agree, as a drafting principle. This is the synthesis of his other points, and the comparative
evidence agrees: our Ch.2 has 5 sections and ZERO subsections; the approved same-advisor precedent (Viegas) has 5
sections and 19 subsections at similar length. Adding two heading levels is what gives GER-08 and GER-09 somewhere to
go.

1. **Adicionar dois niveis de titulo em todo o capitulo** — a prosa quase nao muda e o capitulo fica navegavel; e onde o
   GER-08 e o GER-09 vao morar. Precedente: Viegas tem 5 secoes e 19 subsecoes com o mesmo tamanho; o nosso tem 5 e ZERO
2. **Subsecoes so em §2.2 e §2.3** — meia medida, cobre onde estao as duas lacunas
3. **Nao fazer** — o capitulo e uma referencia que o leitor consulta vindo do Cap. 5; sem titulos ele nao acha nada

> **DECISAO SUA:** ______

#### GER-03 — the HGI tuning sweep is thrown into the text with no connection to it

**Citacao:** exata — `chapters/2_fundamentals.tex:167-170`. **Renderiza em:** 2_fundamentals.tex:167-174 -> 2.2

**Minha leitura:** agree it does not belong; the gate constrains HOW. Strongly agree the four-point sweep is a methods
result in a fundamentals chapter, reported to four decimals under an averaging convention the chapter has not yet fixed.
BUT the sentence is probe NUM-4 in check_audit_claims.py, which requires 0.8186 to be PRESENT with its spreads and
averaging convention. Relocating is compatible with the probe if the probe moves with it; deleting the numbers is not.

1. **Mover o sweep para o Cap. 5 (metodo ou apendice) e mover o probe NUM-4 com ele, no MESMO commit** — atende ao
   ponto; o probe exige que 0.8186 esteja PRESENTE com dispersoes e convencao, e mudar de arquivo sem repontar o probe
   deixa o gate vermelho
2. **Manter uma frase conceitual sem numeros em §2.2 e o sweep no Cap. 5** — o que eu recomendo; resolve tambem
   o [VERIFY] da convencao de media, que hoje reporta quatro decimais sob uma convencao que o capitulo ainda nao fixou
3. **Manter onde esta e so resolver o [VERIFY]** — mais barato; o defeito de fluxo que ele apontou continua

> **DECISAO SUA:** ______

#### GER-04 — the static-vector paragraph reads like introduction prose, and it matters

**Citacao:** exata — `chapters/2_fundamentals.tex:192-193`. **Renderiza em:** 2_fundamentals.tex:192-199

**Minha leitura:** agree it reads as introduction; he also says keep it. He called it well written and important, and
only observed it reads like introduction prose. There is no defect to fix here, only a placement question, and it is 108
chars from the NUM-4 probe string.

1. **Nao mexer** — ele mesmo disse que esta bem escrito e que e importante; nao ha defeito a corrigir
2. **Mover para o inicio de §2.2 como paragrafo de abertura** — atende a impressao de "texto de introducao"; mexe num
   paragrafo a 108 caracteres do probe NUM-4

> **DECISAO SUA:** ______

### 6.7 Bloqueado numa verificacao que falhou

#### FAB-28 — there is more MTL-for-POI work than the two papers cited

**Citacao:** exata — `chapters/2_fundamentals.tex:454`. **Renderiza em:** 2_fundamentals.tex:454 -> 2.3

**Minha leitura:** agree on substance; verification FAILED on the decisive paper. He is right that the coverage is thin,
and the real exposure is that 2.3 claims NO multi-task model predicts next region as a co-equal end target while
wang2025hamtl (hierarchy-aware MTL for user LOCATION prediction, J. Supercomputing 81 (11):1196, 2025) sits uncited in
the same bibliography. Whether the claim survives turns on whether that paper treats a region-like unit as an END
TARGET, and I could NOT establish it: OpenAlex has no abstract, Crossref has no abstract, the configured Springer key
returns 401 on meta/v2, metadata and meta/v1, the paper is closed access (Unpaywall oa_status=closed), and the landing
page 303-redirects to an authentication gate. Semantic Scholar offers only a MACHINE-GENERATED tldr, which
AGENT_GUARDRAILS R5 forbids as a source. Four of the five other candidates ARE verified and citable (see the ledger);
this one is not, and it is the one the novelty sentence depends on.

1. **Voce (ou a biblioteca da UFV) abre o `wang2025hamtl` e me passa o resumo** — desbloqueia; e o unico caminho
   admissivel sob o §1
2. **Citar os quatro verificados (`Zhang2020`, `Halder2022`, `Xu2023` como MTL, `Halder2021` por atributo) e NAO citar o
   `wang2025hamtl`** — amplia a cobertura sem apoiar alegacao em titulo; a frase de novidade continua exposta a quem
   achar o artigo
3. **Enfraquecer a frase de novidade para "entre os trabalhos revisados aqui"** — honesto e barato; abre mao de uma
   alegacao que pode muito bem sobreviver

> **DECISAO SUA:** ______

### 6.8 A sua propria pergunta

#### AUT-01 — does the MTL fundamentals need Pareto optimality

**Citacao:** -. **Renderiza em:** 2_fundamentals.tex 2.3

**Minha leitura:** agree it needs a brief treatment. The author's own added question: does the MTL fundamentals need
Pareto optimality. Since 2.3 names gradient-surgery balancers, and MGDA/CAGrad/Nash-MTL are all argued in terms of
Pareto-stationary points, the concept is already implicit. Note 'Pareto-stationary point' is ALREADY in the prose and is
PENDENCIAS 2.12 (unregistered in the fail-closed GLOSSARY), so this item and 2.12 are the same decision.

1. **Um paragrafo breve em §2.3: o problema multitarefa e multi-objetivo, os balanceadores de cirurgia de gradiente sao
   argumentados em termos de estacionariedade de Pareto, e por isso o MGDA/CAGrad/Nash-MTL existem** — atende sua
   intuicao e da espinha ao paragrafo dos balanceadores; ~1h. Note que `Pareto-stationary point` JA esta na prosa e e o
   item 2.12 (nao registrado no GLOSSARY), entao este item e o 2.12 sao a MESMA decisao
2. **So registrar `Pareto` no GLOSSARY e nao expandir** — fecha o 2.12 sem crescer o capitulo
3. **Nao tratar** — o capitulo nomeia balanceadores cuja justificativa e Pareto e nunca diz isso

> **DECISAO SUA:** ______

### 6.9 Edicao concorrente durante esta rodada — RESOLVIDA pela propria esteira, e registrada

**O que aconteceu.** Enquanto eu media, **outra esteira** alterou arquivos que eu nao toquei:
`GLOSSARY.md` (02:11 e 02:30, os quatro registros de Pareto, que fecham a **2.12** pela opcao (a)),
`src/chapters/2_fundamentals.tex` (02:23, +106 linhas em §2.3: a equacao da perda total mais o tratamento de Pareto),
`check_audit_claims.py` (02:38, cinco probes dela sobre a minha mudanca de segunda raiz) e `_round6/VERIFY_LIST.md` (03:
01).

**Duas falhas de gate que eu vi eram dela, e as duas ja estao fechadas — por ela, nao por mim.**
Registrado porque eu quase abri uma decisao sua sobre a primeira:

| o que falhou                                                | causa                                                                                                                                                                   | estado agora                                                          |
|-------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| `check_verify_list`: `EXPECT: contains=Pareto-stationary 0` | o registro do termo tornou a anotacao falsa                                                                                                                             | corrigida para `2` no mesmo commit do registro, que e o caminho da V6 |
| `check_verify_list`: `CD-FAIL` num bloco de build           | um bloco cercado em `VERIFY_LIST.md:326` sem sufixo `cd src` para o extrator remover, entao o gate **rodava um build de verdade** (108 s medidos) e nunca via `REACHED` | o bloco foi tirado da cerca em 03:01; `make check` volta a 2,4 s      |

**Consequencia metodologica, e essa vale mais que os dois defeitos.** Eu li os dois como vermelhos meus, e um deles eu
ja tinha escrito aqui como uma decisao sua com tres opcoes. Numa arvore com duas esteiras, **um gate vermelho nao e
evidencia de nada ate voce reconferir de quem e o arquivo e qual e o mtime dele** — o meu primeiro diagnostico foi de um
arquivo lido no meio da escrita de outra pessoa.

**Um item meu mudou de balde por causa disso, e nao foi por edicao concorrente:** o **FAB-18** virou decisao sua (§6.2),
porque ele pede o presente na mesma frase que o FAB-03, e eu tinha posto um em cada balde. Achei ao aplicar, nao ao
classificar.

**Tres coordenadas minhas andaram +106 linhas** por causa da edicao em §2.3, e foram reconferidas linha a linha, nao
deslocadas por aritmetica: FAB-28 (454 -> 560), FAB-29 (513 -> 619), FAB-30 (515-516 ->
621-622 e 670 -> 776).

**E um defeito que eu encontrei e corrigi, este nao concorrente:** o §4 item 5 deste arquivo mandava voce rodar
`make final` (alvo renomeado para `academico` em 2026-07-29) e prometia **108/105/109**
paginas. A arvore da **102/99/103**. O `sync_page_counts.py` varria so `CLAUDE.md`, `PLAN.md` e
`codex_reviewer.md`, e por isso o numero envelheceu sem gate — **este arquivo entrou na varredura em 2026-07-30**,
depois de a minha propria correcao aqui ficar obsoleta por uma pagina em menos de uma hora (a outra esteira somou uma
pagina ao §2.3 entre uma medicao e a seguinte).

> **DECISAO SUA:** nada aqui. Fica como registro.

### 6.10 Tres bloqueadores das personas de revisao — todos conferidos por mim no fonte

As quatro personas de fato (04 concordancia, 05 citacoes, 06 numeros, 07 alegacoes/honestidade)
rodaram contra o build `03b53d16`. Relatorios em `_round9/reviews/`, resumo em
`_round9/37_reviewer_gate_round9.md`. **Tres bloqueadores.** Reconferi os tres abrindo o fonte, porque auto-relato de
persona nao e evidencia (L6). Os tres sao decisao sua: dois sao alegacoes de conteudo e o terceiro tem duas saidas de
custo diferente.

#### BLQ-1 — o Apendice F diz que o compartilhamento duro "nao custa nada", e chama de duro a topologia errada

**Onde:** `chapters/apx_f_cosine.tex`:290, renderiza na p. 101 do build de defesa. **Texto:** "That is why hard sharing
costs nothing in this architecture, and why Chapter~\ref{ch:mobiwac} finds no balancer improving on a fixed loss
weighting".

**Dois defeitos numa clausula, os dois confirmados no fonte:**

1. A afirmacao de custo e **irrestrita**, e o proprio documento a desmente:
   `5_mobiwac/06_results.tex`:145-146 registra o ganho de regiao indo "from $-0.41$ at the smallest count to $+2.20$ at
   the largest". Existe celula negativa, e o Cap. 5 a chama de deficit pequeno mas estatisticamente significativo.
2. A arquitetura do Cap. 5 **nao e compartilhamento duro** pela definicao do proprio Cap. 2:
   `5_mobiwac/01_introduction.tex` a descreve como "a shared trunk (a cross-attention stack where the two tasks exchange
   semantic context) and a private spatial path for the region task".

1. **Trocar pela afirmacao licenciada, que o Cap. 5 ja usa: "sharing stopped hurting" (p. 80), nomeando a topologia como
   tronco de atencao cruzada** — corrige os dois defeitos numa frase, nao perde o argumento (o ponto do apendice e a
   ortogonalidade, nao o custo), ~10 min
2. **So restringir o custo ("costs nothing on the category task", ou "at four of the six datasets")**
   — mantem a palavra "custo", mas continua chamando a topologia de compartilhamento duro
3. **Deixar como esta** — a frase e uma generalizacao irrestrita contra uma celula negativa medida no mesmo documento; e
   o tipo de frase que uma banca localiza

> **DECISAO SUA:** ______

#### BLQ-2 — a resposta consolidada usa "everywhere" nu e colapsa a particao da regiao

**Onde:** `chapters/6_conclusion.tex`:106, renderiza na p. 79. E a frase mais citavel da dissertacao. **Texto:**
"outperforms the dedicated models on the category task everywhere and outperforms or matches them on the region task."

**Confirmado contra a lei, que aqui e explicita.** `WRITING_LAW.md`:83 diz, literalmente, que a afirmacao de escala fica
escopada aos cinco estados e **bare "everywhere" never**; a linha 75 proibe
"outperforms region everywhere" por nome. E "outperforms or matches" joga fora a particao quatro-de-seis / TOST que a
redacao protegida exige — particao que **este mesmo capitulo enuncia corretamente na pagina anterior**.

1. **Reescrever com a particao explicita: supera nas seis para categoria; para regiao, supera em quatro dos seis e nao e
   inferior (TOST, +/-2 pp) nos outros dois** — e a redacao protegida, ja usada duas vezes no mesmo capitulo, ~15 min
2. **Manter "everywhere" so para categoria (onde e verdade nas seis) e abrir a particao so na regiao**
   — mais curto, mas "everywhere" nu esta proibido pela linha 83 sem excecao por tarefa
3. **Deixar como esta** — contradiz a lei que voce mesmo escreveu, na frase que mais vai ser lida

> **DECISAO SUA:** ______

#### BLQ-3 — dois fatores de escala calculados na prosa, sem ledger e sem script

**Onde:** `chapters/apx_f_cosine.tex`:317-319, p. 101. **Texto:** "this axis spans a factor of thirty-six in volume and
one of sixteen in the size of the region label set".

**Os numeros estao CERTOS** — 4.089.892 / 113.846 = 35,92 e 8.501 / 520 = 16,35, e os quatro extremos sao rastreaveis em
`datasets.tex`. O achado nao e erro de conta: e que os dois fatores sao **derivados na prosa**, e N2/N3 mandam citar,
nunca calcular. Nada os regenera quando um dataset muda.

1. **Enunciar os quatro extremos (que ja sao rastreaveis) e cortar as razoes** — o leitor faz a divisao se quiser; some
   a alegacao nao rastreavel, ~5 min
2. **Manter as razoes e registra-las no ledger do apendice com o comando que as produz** — preserva a frase como esta,
   custa uma linha de ledger e um probe
3. **Deixar como esta** — dois numeros no volume construido sem origem registrada

> **DECISAO SUA:** ______

### 6.11 Segunda onda de personas — um bloqueador confirmado, um rebaixado por mim

Cinco personas que voce pediu (15 legibilidade, 16 credibilidade-IA, 17 excelencia, 11 POI/mobilidade, 10 MTL) contra o
build `901a0408`. Relatorios em `_round9/reviews/`, resumo em
`_round9/38_reviewer_wave2_round9.md`. Reconferi no fonte tudo que entrou aqui.

#### BLQ-4 — o Apendice F descreve um experimento que nunca foi feito

**Onde:** `chapters/apx_f_cosine.tex`:83, renderiza na p. 97 do build de defesa. **Texto:** "That is why replacing the
sharing scheme changed so little in the first study, and why changing the representation changed so much in the second
and third."

**Confirmado, e o proprio documento diz o contrario em tres lugares.** O Cap. 3 construiu **uma**
arquitetura (`3_cbic/method.tex`:69, "built upon a hard parameter-sharing scheme") e lista as alternativas como trabalho
futuro (`3_cbic/conclusion.tex`:23, "We plan to explore alternative parameter-sharing mechanisms, such as **soft
sharing (e.g., Cross-Stitch Networks) or Mixture-of-Experts (MoE) models**"). Os resultados dele comparam MTL contra os
modelos de tarefa unica, contra MHA+PE e contra o HMRM — nunca dois esquemas de compartilhamento. E o
`1_introduction.tex`:133-135 atribui a troca da topologia de compartilhamento ao **terceiro** estudo.

**O mesmo apendice acerta duas paginas depois**, em :294-299: o nulo do primeiro estudo *foi lido*
como evidencia sobre compartilhamento na epoca, e o limite estava em outro lugar. Ou seja, a frase certa ja existe no
mesmo arquivo.

1. **Reescrever a frase de :83 na forma que :294-299 ja usa — o nulo do primeiro estudo foi interpretado como limite do
   compartilhamento, e a medicao mostra que nao era** — corrige a afirmacao sem perder o argumento do apendice, ~10 min
2. **Cortar a clausula "replacing the sharing scheme" e ficar so com a metade da representacao** — mais curto; perde a
   ponte para o arco dos tres estudos
3. **Deixar como esta** — um leitor da p. 97 e informado de um experimento que nao aconteceu; e do tipo que a banca
   pergunta

> **DECISAO SUA:** ______

#### BLQ-5 — a persona de MTL abriu um bloqueador no PCGrad, e eu o REBAIXEI; a decisao final e sua

**Onde:** `chapters/2_fundamentals.tex`:442-445. **Texto:** "PCGrad guarantees that one projected update leaves the
multi-task loss no higher than the unmodified gradient would ... and it makes no Pareto claim at all
\cite{yu2020pcgrad}."

**O que ela alegou:** que a p. 5 do CAGrad atribui convergencia a um ponto de Pareto arbitrario a *propria analise* do
PCGrad, o que contradiria o "no Pareto claim at all".

**Por que eu rebaixei.** Abri os dois registros nesta sessao. O arXiv:2001.06782 (PCGrad, NeurIPS

2020) nao tem 'Pareto' nem frase de convergencia no resumo, e a extracao do proprio repo achou **zero**
      'Pareto' em 27 paginas com o instrumento validado no mesmo texto. O resumo do arXiv:2110.14048 (CAGrad)
      diz que os metodos anteriores "lack convergence guarantee and/or could converge to any Pareto-stationary point".
      **A clausula do capitulo esta citada a `yu2020pcgrad` e fala do que aquele artigo afirma**; a frase do CAGrad e um
      terceiro caracterizando a familia. As duas coisas convivem.

**O que sobra, e por isso o item existe:** se a p. 5 do CAGrad de fato atribui o resultado a analise do PCGrad, um
leitor que conhece o CAGrad vai achar "no Pareto claim at all" mais categorico do que a literatura em volta sustenta.
**Eu nao consegui ler a p. 5** — so o resumo era alcancavel sem o PDF — e por isso rebaixei em vez de descartar, e digo
qual metade nao conferi.

1. **Suavizar para "and makes no Pareto claim of its own"** — mantem a verdade sobre a fonte citada e remove o absoluto
   que incomoda quem conhece o CAGrad, ~5 min
2. **Abrir a p. 5 do CAGrad e decidir com ela na mao** — o certo se voce tem acesso; e um paragrafo
3. **Deixar como esta** — a frase e exata sobre a fonte que cita, e a persona nao a refutou

> **DECISAO SUA:** ______

#### Sem decisao sua, so registro: o que as duas personas de dominio confirmaram

A de POI/mobilidade nao achou bloqueador: a distincao proxima categoria / proxima regiao / proximo lugar se sustenta em
tudo que ela leu, o argumento nivel-de-check-in contra nivel-de-lugar esta correto como afirmacao de modelagem de
mobilidade e ancorado no CTLE e nao no seu proprio resultado, e as divulgacoes de protocolo do Cap. 5 sao **mais fortes
que a norma dessa literatura**. Os quatro should-fix dela sao todos a mesma classe: **lugar, nao verdade** — defesas que
o Cap. 5 carrega e o Cap. 2 nao (construcao de janela, transdutividade, justificativa da unidade regional, a intuicao de
revisita). A de MTL verificou **seis das sete clausulas de garantia** do bloco de Pareto de hoje contra cinco PDFs de
origem, e retirou duas conclusoes proprias depois de abrir as paginas que as refutavam. A de excelencia deu **VERY GOOD
forte** com caminho barato para outstanding, e registrou que o resultado nulo publicado e **inequivocamente um ativo**
como esta apresentado.

---

## §3 · Aberto e bloqueado em terceiros

| Item                                               | Bloqueado em                     | Estado                                                                                                                                |
|----------------------------------------------------|----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| Dois membros da banca e a data da defesa           | Orientador / PPGCC               | Placeholders honestos entre colchetes em `0_main.tex`; **os colchetes aparecem no PDF**, entao nada inventado e apresentado como fato |
| Folha de aprovacao assinada                        | A defesa                         | `make ppgc` gera o PDF com o placeholder; a versao assinada o substitui depois                                                        |
| Status do MobiWac                                  | Revisores                        | A redacao e sempre "submitted, under review", em todo o documento. **Nao mudar** ate haver decisao                                    |
| `\finalbuildfirstpage` conferido contra o RASCUNHO | Upload pos-defesa ao AcademicoPG | Agora **8**, derivado das 7 paginas pre-textuais do build de deposito e verificado no render. Confira contra o RASCUNHO quando subir  |

---

## §4 · Retirado

A lista priorizada de auditoria vivia aqui e apontava para [`_round6/VERIFY_LIST.md`](_round6/VERIFY_LIST.md). **Esse
registro esta fechado** (medido 2026-07-30): os sete itens A1-A7 carregam cada um a sua disposicao de round 8 ou 9, e
`check_verify_list.py` sai 0. O ultimo em aberto era o A3, a decisao de publicar os arquivos que faltavam — resolvida
por execucao, nao por decisao: os seis artefatos de analise estao no `origin/mobiwac`, conferidos contra o REMOTO com
`git ls-tree -r origin/mobiwac`.

O aviso que estava aqui — *nao confie no sucesso auto-reportado, incluindo o meu* — nao se perdeu: virou lei em
`AGENT_GUARDRAILS.md` §4b, hoje com dezessete regras numeradas, cada uma escrita a partir de um caso real deste projeto.
E o lugar certo, porque uma advertencia numa fila de tarefas some quando a fila e limpa.
