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

**Onde cada coisa vive.** O tracker carrega a **decisao**; a **forense** (como o defeito foi
descoberto, qual instrumento mentiu, o que cada commit mediu) vai para `_round8/`. Em 2026-07-30 seis
itens carregavam 34 mil dos 55 mil caracteres do arquivo, quase tudo forense: foi para
[`_round8/29_pendencias_detail.md`](_round8/29_pendencias_detail.md), **nada apagado**, e o arquivo
caiu de 67 mil para 37 mil.

**Para ADICIONAR um ponto seu:** escreva embaixo do item, comecando a linha com `> DECISSAO:` (ou
`> DECISAO:`). Eu leio isso como sua palavra final e nao reinterpreto. Se voce nao tiver numero de
item, escreva no fim do §2 com um titulo qualquer — eu numero e coloco no lugar.

**Para FECHAR um item:** ele sai daqui e vai para `_archive/PENDENCIAS_RESOLVIDOS.md` **com o motivo de
saida no topo do bloco**. O gate `check_tracker_refs.py` falha se um item desaparecer sem chegar ao
arquivo — tres foram perdidos assim, e voce achou dois deles lendo o arquivo. **Nao renumere:**
comentarios no fonte citam estes numeros, e um buraco na numeracao e melhor que um ponteiro errado.

**Ordem das secoes:** §2 (voce) -> §5 (do `CODEX_AUDIT`) -> §3 (terceiros) -> §4 (o que auditar
primeiro). Deliberada: o que depende de voce vem antes.

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
`check_verify_list` conta e a contagem bate. A lista completa, com arquivo, linha e o que cada um
afirma, sai de: `grep -rn "NEEDS SIGN-OFF" src/ --include="*.tex" | grep -v ":\s*%"`.

**Tres tem prioridade** (afirmam algo sobre trabalho publicado ou co-autorado): o paragrafo corrigido
do Apendice B sobre o Cap. 3, o numero limitado do Cap. 4 na conclusao, e a frase de reprodutibilidade
enfraquecida. Estao detalhados em `_round6/VERIFY_LIST.md` A1, A2 e A3.

> **DECISAO SUA:** ler os 53 e me dizer quais aprova. Nao precisa ser de uma vez — se me der os tres
> prioritarios, eu removo os marcadores deles e mantenho os outros 50.

*Forense (a tentativa de push destrutiva, o worktree, os artefatos divergentes): agora e o item 2.16 e
o corpo integral esta em [`_round8/29_pendencias_detail.md`](_round8/29_pendencias_detail.md).*

### 2.5 O tamanho de tipo das duas figuras de arquitetura — autorizado, mas eu nao consigo executar

**Voce autorizou:** *"pode aumentar, mas mantenha o espaco ja ocupado pela imagem... mantendo a
proporcao"*, e observou que o contraste hoje ja deixa legivel.

**Nao consigo fazer daqui:** nao ha `drawio` nem `inkscape` neste ambiente, e so **1 dos 2** `.drawio`
esta no repositorio. A receita esta em `_round6/12_figures.md` (subir `fontSize` de 13 para ~20 e
reexportar na mesma largura em pixels).

> **Seu, quando quiser:** reexportar as duas no Draw.io e me passar os PNG — eu troco e remeco o tipo na
> pagina. **Opcional**, pela sua propria observacao sobre o contraste.

### 2.8 `CONSIDERATIONS.md`: uma rodada NOVA que chegou durante esta, e que eu NAO executei

**(A) O que falta.** `src_utils/CONSIDERATIONS.md` apareceu na arvore de trabalho **durante** esta rodada (modificado
19:04, nao commitado, 1.229 linhas). Ele contem material que nao estava no escopo que voce me deu:

| Secao                                    | O que e                                                                                          |
|------------------------------------------|--------------------------------------------------------------------------------------------------|
| `## Germano` (l. 3-58)                   | Feedback **verbal** do Germano sobre o Cap. 2, transcrito por voce                               |
| `## Fabrício` (l. 59-309)                | Feedback do **orientador** sobre o Cap. 2                                                        |
| `# Codex Audit — Chapter 2` (l. 310-994) | Auditoria dos dois feedbacks, comparacao contra `exemples/`, e uma lista de trabalho consolidada |
| `# Addendum (2026-07-28)` (l. 995-1229)  | O ponto de fluxo do Germano e o item G10 (o achado de conflito de tarefas)                       |

**(B) Por que importa.** Isto e feedback do **orientador** e de um leitor externo sobre o capitulo de fundamentacao, com
uma lista de trabalho ja consolidada. E a proxima rodada, e e mais importante que a maior parte do que sobrou aqui. Nao
esta perdido: o arquivo esta no disco. Mas nao esta commitado, e nenhum item dele foi aplicado ao texto.

**(C) O que eu preciso de voce.** Duas coisas. Primeiro, **commitar o arquivo** se ele estiver pronto (eu
deliberadamente nao commitei prosa sua em andamento). Segundo, dizer se quer que eu execute a lista de trabalho
consolidada dele — ela e uma rodada propria, com pesquisa e verificacao, e nao a comecei porque nao foi o que voce pediu
nesta.

**Por que eu nao agi nisso.** O escopo desta rodada foi `CODEX_AUDIT.md` mais as suas decisoes em
`PENDENCIAS.md`. Aplicar 1.229 linhas de feedback novo no fim de uma rodada longa, sem voce ter pedido, seria exatamente
o tipo de improviso que o `AGENT_GUARDRAILS` manda parar e sinalizar.

### 2.9 O disco do `nespedgpu` — liberado por voce; sobrou decidir se roda o resto

**O que e.** O disco encheu (0 bytes livres) e matou tres datasets do Apendice F; voce liberou espaco.
O apendice hoje reporta **quatro** datasets (florida, alabama, arizona, georgia) e diz que california,
texas e istanbul foram bloqueados, o que e verdade e esta escrito.

> **DECISAO SUA:** rodar os tres que faltam (~6h de GPU, e o apendice passa a seis) ou publicar com
> quatro. **Eu nao apago nada na sua maquina** — os 61G sao seus checkpoints.

*Forense (o crash `basic_ios::clear` que era falha de escrita, a corrida de harvest que produziu dois
folds identicos, e por que descartei aqueles dados): [`_round8/29_pendencias_detail.md`](_round8/29_pendencias_detail.md).*

### 2.11 A assimetria do resultado de regiao: o Cap. 5 ressalva, e o resto do documento nao

**Origem:** `_round6/VERIFY_LIST.md` itens 4 e 5 (achado L-5 do ledger), entregues em 2026-07-30.

**(A) O que e.** `chapters/5_mobiwac/05_setup.tex` diz que o plano de analise *"did not cover
next-region superiority, so the four next-region gains ... are secondary results outside it"*. O
resto do documento afirma o mesmo resultado **sem essa ressalva**. Medido com o varredor que remove
comentarios, sobre os 54 `.tex`:

| onde | forma |
|---|---|
| `1_introduction.tex:132`, `6_conclusion.tex:21` e `:93` | "four of six" / "four of the six" |
| `2_fundamentals.tex:786`, `5_mobiwac/01_introduction.tex:39`, `5_mobiwac/08_conclusion.tex:14`, `5_mobiwac.tex` | idem |
| `content.tex:166` (Resumo e Abstract) | "quatro deles" / "four of them" — a mesma alegacao, em outras palavras |

No PDF de defesa (100 pp) a alegacao sem ressalva imprime nas **pp. 14, 58, 59, 76, 77 e 78**; a
ressalva imprime **so na p. 67**. Sao sete sitios em prosa mais as duas parafrases do pre-textual,
contra uma ressalva.

**(B) Por que importa.** O registro estatistico de 2026-07-27 e inequivoco: o teste primario
registrado para **toda** celula de regiao e nao-inferioridade TOST. Uma leitura rapida do Resumo, da
Introducao ou da Conclusao le "outperforms em quatro de seis" como resultado primario; a p. 67 diz
que nao e. Nenhuma track da rodada 6 assumiu isso (achado L-5 do ledger).

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

### 2.12 `Pareto-stationary point` esta na prosa e nao esta no registro (o `GLOSSARY` e fail-closed)

**(A) O que e.** A regra de manutencao do `GLOSSARY.md` e explicita: *"a term not in this registry
may not be used in dissertation prose"*. Medido hoje:

```bash
cd /Users/vitor/Desktop/mestrado/ingred/articles/dissertacao
grep -c 'Pareto-stationary' GLOSSARY.md          # 0
```

e o termo esta em prosa em **dois** sitios, os dois em texto publicado reproduzido:
`chapters/3_cbic/method.tex` (*"convergence to a Pareto-stationary point"*) e
`chapters/4_courb/methodology.tex` (a garantia do Nash-MTL, a mesma frase do item 3 do
`VERIFY_LIST`). Imprimem nas **pp. 36 e 48**. `tables/courb/errata.tex` ainda traz a forma sem
hifen, "Pareto stationary". As outras duas entradas que o item 4 daquele arquivo pedia — **bilinear
discriminator** e **logistic function** — **ja entraram** (`GLOSSARY.md:71` e `:72`); so esta faltou.

**(B) Por que importa.** O `make check` **nao pega isto**: existe um gate de "Pareto" mas ele e
informativo e conta ocorrencias, nao registro. E o termo nao pode ser simplesmente removido — as
duas frases sao publicadas, entao tirar o termo e editar uma frase publicada, com linha no
Apendice B.

**(C) O que eu preciso de voce.** Uma decisao, tres saidas:

> **(a)** registrar o termo (uma linha na §4 do `GLOSSARY`, a definicao ja esta escrita na propria
> frase do Cap. 4: *"a point at which some convex combination of the task gradients is zero"*).
> Custo: uma linha, e o documento fica consistente com a propria regra.
> **(b)** trocar o termo nas duas frases publicadas. Custo: duas linhas de errata no Apendice B, e
> voce esta editando prosa publicada por uma questao de vocabulario.
> **(c)** registrar uma excecao explicita para termos que chegam em prosa reproduzida. Custo: uma
> nota no `GLOSSARY`, e a regra deixa de ser fail-closed para essa classe.

### 2.14 O intervalo de paginas do `nash`: nao da para verificar daqui

**Origem:** `_round6/VERIFY_LIST.md` item 14, entregue em 2026-07-30 (precedente `standley2020tasks`).

**(A) O que e.** `references.bib` traz `pages = {16428--16446}` para
`@inproceedings{nash}` (Navon et al., *Multi-Task Learning as a Bargaining Game*, ICML 2022).
Tentado de novo nesta sessao, contra as fontes de registro que o sandbox alcanca:

| fonte | resposta |
|---|---|
| OpenAlex | um unico registro, `W4225981399`, tipo **preprint**, venue "arXiv", `first_page` e `last_page` **nulos** |
| Crossref (`query.bibliographic`) | cinco obras, **nenhuma delas este artigo** — nao ha DOI registrado da versao de anais |
| `proceedings.mlr.press` | **fora da allowlist** do sandbox; nao acessado |

**(B) Por que importa.** Pelo §1 do `AGENT_GUARDRAILS`, um identificador que nao foi aberto na fonte
de registro nao pode ser apresentado como conferido. O campo esta no `.bib` e nao esta verificado.

**(C) O que eu preciso de voce.** Um clique fecha: `proceedings.mlr.press/v162/navon22a.html`.

> **(a)** confirmar o intervalo e ele fica; **(b)** apagar o campo `pages`, que e exatamente o
> precedente que esta bibliografia ja adotou para `standley2020tasks`.

### 2.15 Tres citacoes NOT-SUPPORTED e um termo banido, todos em prosa publicada reproduzida

**Origem:** `_round6/VERIFY_LIST.md` itens 15 e 16, entregues juntos em 2026-07-30 como uma decisao unica.

**(A) O que e.** Quatro pontos, um so tipo de decisao: **nenhum deles pode ser corrigido por um
agente**, porque todos estao em frases publicadas, e mexer nelas gera linha de errata.

| onde | o que |
|---|---|
| `3_cbic/method.tex` | `ruder2017sluice` citado para regularizacao implicita do hard sharing |
| `4_courb/methodology.tex:173` | `sun2020go` citado para ciclos temporais revelarem a *funcao* do lugar |
| `4_courb/methodology.tex:184` | `belkin2003laplacian` citado para um regularizador hierarquico de embedding |
| `4_courb/methodology.tex:173,184` | **`fclass` em prosa renderizada**, 4 ocorrencias em 3 linhas |

**(B) Sobre o `fclass`, que ninguem tinha levantado.** O `GLOSSARY.md:73` diz, com todas as letras:
*"In code this column is `spot`, renamed `fclass` at `hgi/preprocess.py:62`; **NEVER write `fclass`
in prose**"*. Varri os 54 `.tex` sem comentarios: `4_courb/methodology.tex` e o **unico** arquivo. E
as frases identicas estao publicadas no CoUrb (`src_en/sections/metodology.tex:109` e `:120`), entao
e o mesmo tipo de decisao das tres citacoes acima. Nenhum gate pega: o de codenomes casa
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

### 2.16 Quatro artefatos publicados **divergiram** das copias locais (o item 2.2 cobria dois)

**(A) O que e.** O Apendice A cita treze caminhos `\path{}`. A pergunta "quantos faltam no branch
publico" ja teve **quatro** respostas nesta base (9 de 13, depois 5, depois 4, agora esta). O motivo
de todas as anteriores e o mesmo: `git cat-file -e mobiwac:<caminho>` pergunta *"este CAMINHO esta
no branch"*, e a alegacao e *"este ARQUIVO esta no branch"* — e o branch `mobiwac` **nao tem arvore
`docs/`**, guarda esses artefatos em `analysis_protocol/`. Remedi por **hash**, comparando cada
arquivo local com os blobs do branch:

| classe | n | quais |
|---|--:|---|
| no branch, **byte a byte identicos** | 8 | `folds.py`; `STATISTICAL_PROTOCOL.md`, `JOINT_BEST_RESULTS.md`, `m1_full_output.txt`, `m2_prereg_output.txt` (sob `analysis_protocol/`); `build_phase3_per_fold_transitions.sh`; `score_joint_best.py`; `autocorrelation_ceiling.py` (em `scripts/`, nao `scripts/embedding_eval/`) |
| no branch com **conteudo diferente** | 4 | `superiority_wilcoxon.py`, `region_match_tost.py`, `m1_stats_n20.py`, `m2_prereg_perfold.py` |
| diretorio (o instrumento nao classifica) | 1 | `stats_n20/` |

**Nada esta faltando.** Quatro artefatos publicados **divergiram**:

| arquivo | local | no branch | linhas diferentes |
|---|--:|--:|--:|
| `superiority_wilcoxon.py` | 147 | 126 | 37 |
| `region_match_tost.py` | 74 | 74 | 2 |
| `m1_stats_n20.py` | 411 | 335 | 84 |
| `m2_prereg_perfold.py` | 214 | 222 | 36 |

**(B) Por que importa.** O seu item **2.2 ja reservou exatamente esta decisao a voce**, mas para
**dois** arquivos (`m1_stats_n20.py` e `m2_prereg_perfold.py`). Sao **quatro**. Substituir um
artefato publicado por uma versao local divergente e decisao de autor, nao faxina — nao toquei em
nada.

**(C) O que eu preciso de voce.** Por arquivo, ou uma regra para os quatro:

> **(a)** publicar a versao local, se ela e a correta (um commit seu no branch `mobiwac`);
> **(b)** deixar como esta, se a versao publicada e a que gerou os numeros do artigo — que e o caso
> a favor de deixar quieto, e o mais provavel para `region_match_tost.py`, com 2 linhas de diferenca.
>
> A prosa do Apendice A **nao depende disto**: ela ja diz que os scripts estatisticos *"are part of
> the working repository and are supplied on request"*, o que nenhuma das leituras acima torna falso.

### 2.18 Um `refs/notes/commits` foi para o `origin` sem eu ter pedido, e a decisao de remover e sua

**Medido em 2026-07-30, nada foi alterado no remoto.** `git ls-remote origin | grep notes` retorna
`refs/notes/commits` apontando para `99c0a34b1a`, identico ao ref local. Ele **esta publicado**.

**Como foi.** Nenhum comando meu pediu isso. O git tem uma configuracao (`notes.rewriteRef` /
`remote.origin.push` com refspec ampla, ou `push.default` com notes habilitados) que empurra
`refs/notes/*` junto de um push comum; o stderr de um dos meus pushes mostrou
`* [new reference] refs/notes/commits`. Um sub-agente reportou isso por conta propria, incluindo o
fato de ter subestimado o escopo na primeira vez que descreveu.

**O que ha nesses notes: 15 anotacoes, e todas sao correcoes de mensagens de commit minhas.** Cada
uma diz que uma frase de commit era falsa e qual e a medicao correta — a convencao deste repositorio
para nao reescrever historia. Sao, literalmente, o registro dos meus proprios erros.

**Por que provavelmente nao e grave — agora medido em TODOS, nao em seis.** A primeira versao deste
item afirmava "os 14 commits anotados nao estao em nenhum branch do origin" a partir de uma sondagem
que rodou com `head -6`: **oito nunca foram checados**, e a recomendacao abaixo repousava nessa
generalizacao. Re-medido sem o `head`, e a contagem tambem estava errada — sao **15** notes, nao 14:

```
for h in $(git notes list | awk '{print $2}'); do
  git branch -r --contains "$h" | grep -c "origin/"; done
# checked=15  on_public_branch=0
```

**15 de 15 verificados, zero em qualquer branch do `origin`.** Sao objetos alcancaveis apenas
pelo ref de notes, nao historia visivel de nenhum branch publico. Quem clonar o repositorio **nao
recebe notes por padrao** (precisa de `git fetch origin refs/notes/*:refs/notes/*`).

> **DECISAO SUA, e eu nao vou tomar por voce.** Tres opcoes:
> 1. **Deixar.** Elas documentam correcoes honestas e nao aparecem em clone normal. Custo zero.
> 2. **Remover do remoto:** `git push origin :refs/notes/commits` — apaga o ref publicado e mantem os
>    notes locais. Uma linha, reversivel (basta empurrar de novo).
> 3. **Impedir que volte:** `git config --local notes.rewriteRef ""` e conferir
>    `git config --get-all remote.origin.push`.
>
> Eu recomendo a **2 + 3** se este repositorio for ficar publico com a defesa, e a **1** se ele
> permanecer privado. Nao executei nenhuma delas porque mexer em ref publicado e sua alcada.

### 2.19 Quatro numeros do registro de fechados nao reproduzem; um tem tres respostas

**O que e.** O item 1.2 do `_archive/PENDENCIAS_RESOLVIDOS.md` tem nove linhas; cinco conferem. Das
quatro restantes, nenhuma reproduz da arvore viva: comentarios medem 3.614 linhas contra 1.269
afirmadas, o `preamble.tex` tem 14 placeholders contra 3, e `geometry`/`linespread` nao estao nesse
arquivo. **Sao velhos, nao errados** — medidos na rodada 6, contra uma arvore que desde entao ganhou um
apendice e perdeu o `0_main.tex`. O defeito duravel: nenhum registra **contra qual estado da arvore**
foi tomado, e medicao sem isso so pode ser re-tomada, nunca re-conferida.

> **DECISAO SUA, e e uma so:** qual convencao de contagem de palavras vale para o Resumo e o Abstract
> no deposito? Ha **tres** respostas (310/271 no relatorio, 312/277 de uma esteira, 345/307 do meu
> instrumento) porque ha tres convencoes. Um numero impresso no seu deposito e seu. Diga a convencao,
> eu fixo uma, aplico, e ponho o comando no arquivo.

### 2.20 O Cap. 4 italiciza ingles corriqueiro 153 vezes, e este item DESAPARECEU do tracker sem decisao

> **ESTE ITEM FOI PERDIDO, nao resolvido.** Ele existiu ate `1ef83867` (2026-07-28) e saiu do arquivo
> naquele commit **sem ir para o `_archive/PENDENCIAS_RESOLVIDOS.md` e sem uma decisao sua**. O titulo
> dizia explicitamente *"e uma decisao sua"*. Reencontrado em 2026-07-30 varrendo as 63 revisoes do
> tracker por titulo, nao por numero — porque os numeros foram reciclados em tres renumeracoes.

**Re-medido agora, na prosa viva (comentarios removidos), e os numeros continuam praticamente iguais
aos de dois dias atras:**

| capitulo | `\emph`/`\textit` |
|---|---|
| Cap. 1 | 6 |
| Cap. 2 | 6 |
| Cap. 3 | 23 |
| **Cap. 4** | **153** (eram 155) |
| Cap. 5 | 10 |
| Cap. 6 | 0 |

Mais italicizados no Cap. 4: `embedding` 18, `baseline` 16, `encoders` 15, `encoder` 14,
`embeddings` 12, `check-ins` 7. **E inconsistente consigo mesmo** — a mesma palavra aparece nas duas
formas: `encoder` italico 14 / romano 8, `encoders` 15 / 7, `baseline` 16 / 4, `embedding` 18 / 1.

**A causa e legitima, a consequencia nao.** Isso vem do artigo em portugues, onde italicizar
estrangeirismo e a pratica correta. Num capitulo **em ingles** a mesma marcacao nao marca mais
estrangeirismo: le-se como enfase numa palavra que nao tem nenhuma, e o proprio capitulo se contradiz.

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

### 2.21 O segundo ponto do seu orientador: como os termos entram

**O que e.** Ele escreveu *"soa um pouco estranho o jeito que alguns termos sao inseridos (marquei
alguns la)"*. O buraco concreto que o item registrava — o revisor de estilo nunca ter lido o
`articles/[mobiwac]/GLOSSARY.md`, que vence para o Cap. 5 — esta **fechado**: revisor re-rodado em
28/07, as tres proibicoes da §3 ausentes, as 28 da §4 medidas (22 ausentes, 6 dentro da condicao, e
**uma violacao real corrigida**: `region head` -> `region output`).

**Nao medido, e nao vou dizer que esta:** §6, §7 e §8 do glossario sao julgamento de estilo, sem grep
que decida.

> **DECISAO SUA, pequena:** ele marcou termos num PDF que **eu nao tenho**. Me passe as marcacoes e eu
> trato uma por uma.

*Medicao termo por termo: [`_round8/29_pendencias_detail.md`](_round8/29_pendencias_detail.md).*

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

**Medido em 2026-07-30, nos cinco parquets que este trabalho consome.** Sua decisao no 5.6 foi
*"Busque pelo que o artigo original cita e vamos usar isso em ambos. Inclusive ambos usaram o mesmo
recorte nao houve diferenca."* A primeira metade foi cumprida: o `cho2011gowalla` foi aberto em
primeira mao (PDF dos proprios autores, Secao 2, p.2) e ele diz **Fev 2009 a Out 2010**.

**A segunda metade nao se sustenta.** Os cinco estados usados nao param em Out 2010:

| estado | primeiro check-in | ultimo check-in | n |
|---|---|---|---|
| Alabama | 2009-03-18 | 2011-07-27 | 113.846 |
| Arizona | 2009-03-26 | 2011-07-04 | 236.450 |
| Florida | 2009-03-13 | 2011-08-11 | 1.407.034 |
| Texas | 2009-01-21 | **2011-08-16** | 4.089.892 |
| California | 2009-01-24 | 2011-08-14 | 3.171.380 |

Uniao: **2009-01-21 a 2011-08-16** — dez meses depois da janela que o artigo declara.

**Por que isso importa e nao e frescura.** A frase esta no Cap. 6 sob a limitacao *"Data vintage"*.
Ali o leitor le a data como **a safra dos dados que voce usou**, nao como uma nota sobre o que outro
artigo coletou. Imprimir so Fev 2009–Out 2010 subestimaria o proprio corpus em dez meses.

**O que eu fiz.** A frase agora carrega as duas datas: o que os autores relatam, e o que a extracao
daqui abrange, com a medicao completa e o comando no comentario de proveniencia do
`6_conclusion.tex`. Nao e uma correcao da sua decisao — voce estava decidindo **qual fonte citar**, e
essa parte esta cumprida.

> **DECISAO SUA.** Se voce preferir imprimir **so** a janela do artigo, a clausula depois da virgula e
> a que sai, e eu removo. Marcado com `[NEEDS SIGN-OFF: PENDENCIAS_RESOLVIDOS 5.6 (arquivado 2026-07-30), round8]` no fonte.

## §3 · Aberto e bloqueado em terceiros

| Item                                               | Bloqueado em                     | Estado                                                                                                                                |
|----------------------------------------------------|----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| Dois membros da banca e a data da defesa           | Orientador / PPGCC               | Placeholders honestos entre colchetes em `0_main.tex`; **os colchetes aparecem no PDF**, entao nada inventado e apresentado como fato |
| Folha de aprovacao assinada                        | A defesa                         | `make ppgc` gera o PDF com o placeholder; a versao assinada o substitui depois                                                        |
| Status do MobiWac                                  | Revisores                        | A redacao e sempre "submitted, under review", em todo o documento. **Nao mudar** ate haver decisao                                    |
| `\finalbuildfirstpage` conferido contra o RASCUNHO | Upload pos-defesa ao AcademicoPG | Agora **8**, derivado das 7 paginas pre-textuais do build de deposito e verificado no render. Confira contra o RASCUNHO quando subir  |

---

## §4 · O que auditar primeiro, se voce tiver uma hora

A lista priorizada esta em [`_round6/VERIFY_LIST.md`](_round6/VERIFY_LIST.md), com o comando de verificacao de cada
item. Os cinco de maior consequencia:

1. **O paragrafo D-01 em `apx_b_static_scope.tex`** (p. 99 do build de defesa). E a unica prosa nova que faz uma
   afirmacao publica sobre um resultado co-autorado, e eu errei nele uma vez.
2. **O par Resumo/Abstract** (pp. 2-3). Mais lido que qualquer outra pagina.
3. **As duas sentencas D-02 em `6_conclusion.tex`** (p. 76). Elas mudam o que o numero mais citado do Ch.4 licencia.
4. **A frase de reprodutibilidade em `apx_a_contributions.tex`** (p. 88), contra 2.2 acima.
5. **`make check` e os tres builds.** `cd articles/dissertacao && source src_utils/texenv.sh &&
   (cd src && make defense && make final && make ppgc && make check)`. Deve sair 0 e dar 108/105/109.

> **Nao confie no sucesso auto-reportado, incluindo o meu.** Esta rodada corrigiu **oito** afirmacoes
> minhas que nao se sustentaram na medicao: um limite falso que eu carreguei ao corrigir um escopo,
> uma exculpacao do Ch.3 que nao segue da premissa, "all gates pass" com o gate saindo 2, "byte
> identical ... same SHA" quando o que e identico e a camada de texto, um instrumento de tamanho de
> fonte cego ao `\includegraphics`, uma linha de ancoragem que eu li errado, um flag levantado contra
> uma afirmacao correta lendo uma revisao superada, e um teste de gate invalido porque eu copiei o PDF
> corrigido para a arvore quebrada. Todas as oito foram achadas por outra passagem, nao por mim.
