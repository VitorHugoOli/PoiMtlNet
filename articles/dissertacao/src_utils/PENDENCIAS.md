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

### 2.1 Os marcadores `[NEEDS SIGN-OFF]` no fonte — **54** medidos em 2026-07-30

**O que e.** Pontos do fonte marcados como precisando do seu aval. Nenhum bloqueia build, e **nenhum aparece no PDF**:
todos vivem em comentario `%`. **O numero anda** — tracks paralelas removem marcadores conforme voce decide, e ele caiu
de 56 para 54 durante esta propria varredura. Confie no comando, nao no titulo:

```bash
cd /Users/vitor/Desktop/mestrado/ingred/articles/dissertacao
grep -rc "\[NEEDS SIGN-OFF" src --include="*.tex" --exclude-dir=build | grep -v ":0$" | sort -t: -k2 -rn
```

Medido assim em 2026-07-30 sobre `5c074a2a` mais a arvore de trabalho: **54 marcadores em 21 arquivos** (52 com corpo
`[NEEDS SIGN-OFF: ...]` e 2 retrovisores nus `[NEEDS SIGN-OFF]`); 58 se `src/build/` entrar, porque
`build/fmt/_body.tex`
e copia gerada — dai o `--exclude-dir=build`.

*(O comando que estava aqui — `grep -rn ... | grep -v ":\s*%"` — imprimia **zero linhas** e saia `rc=1`: o `-v` casa o
`%` do comentario em que cada marcador vive, entao removia justamente tudo o que devia contar. E nenhum gate conta estes
marcadores: `check_verify_list` executa blocos documentados, nao mede esta contagem, ao contrario do que este item
afirmava.)*

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

**Nao consigo fazer daqui:** nao ha `drawio` nem `inkscape` neste ambiente. **Os dois `.drawio` estao no repositorio** —
`figures/mtlnet_poi_new.drawio` (13.640 B, `fontSize=14`) e `figures/courb/arquitetura_modelo.drawio`
(14.588 B, `fontSize=13`), medidos em 2026-07-30 com `find . -name '*.drawio'` (quatro no repo inteiro). A receita esta
em `_round6/12_figures.md` (subir `fontSize` para ~20 e reexportar na mesma largura em pixels).

*(Este bloco dizia **"so 1 dos 2"**. Era falso, e o commit `b89a9876` ja tinha diagnosticado exatamente isso — o
instrumento era `ls src/figures/*.drawio`, glob nao-recursivo que nao ve `figures/courb/` — mas a correcao nao chegou ao
arquivo. Tamanhos de tipo medidos, no `LEFT_OUT.md` LO-6: **45,3%** do corpo no do Cap. 3 e **44,4%** no do Cap. 4,
contra corpo de 11,96 pt. O raster do Cap. 3 e byte-identico ao publicado do CBIC, conferido por sha256.)*

> **Seu, quando quiser:** reexportar as duas no Draw.io e me passar os PNG — eu troco e remeco o tipo na
> pagina. **Opcional**, pela sua propria observacao sobre o contraste.

### 2.27 A arvore revisada do autor entrou no `src`, e o que ficou aberto nela

**(A) O que e.** Em 2026-08-02 o autor entregou `src_clean`, lido e editado por ele. O merge esta em
`src_utils/_round9/49_clean_tree_merge.md`. A prosa dele entrou byte a byte nos 54 arquivos; a camada
de comentario do `src` (4.114 linhas, 275 blocos, 54 marcadores `[NEEDS SIGN-OFF]`) foi reancorada
por cima. 228 dos 275 blocos reancoraram exatamente.

**(B) O que fica aberto para voce.**

1. **28 blocos marcados `[ORPHANED 2026-08-02]`** — eram 47, e voce resolveu 19 no commit `45c75611`
   ("remove orphaned comments and clean up LaTeX files"). Medido em 2026-08-02:
   `grep -rho 'ORPHANED 2026-08-02' src --include='*.tex' --exclude-dir=build | wc -l` = **28**. Cada um
   anota uma frase que a sua revisao reescreveu ou cortou; nenhum foi apagado por mim. A tabela original
   dos 47 esta no relatorio 49. Sao seus para manter, reescrever ou deletar; um agente nao deve decidir isso.

2. **54 marcadores `[NEEDS SIGN-OFF]` continuam abertos**, distribuidos em 21 arquivos, com 7 em
   `2_fundamentals.tex`, 8 em `6_conclusion.tex` e 6 em `apx_a_contributions.tex`. Sao afirmacoes que
   nenhum artigo publicado sustenta e que dependem da sua assinatura.

3. **A grafia do termo central foi uniformizada em "multitask"**, como manda `GLOSSARY.md:130`. As 36
   ocorrencias hifenizadas que restam sao TITULOS CITADOS no `references.bib` e nao podem ser
   alteradas sem falsear as fontes.

4. **`apx_g_hgi_tuning.tex` e um apendice novo seu**, que recebeu a varredura do peso do HGI que saiu
   do capitulo 2. Renderiza na p. 106 da defesa. Ele nao esta no `main_extra`, so no volume principal.

**(C) Status.** Builds 106/103/107/22 pp, zero erros, zero referencias indefinidas; 25 gates e o
selftest em rc=0, lidos diretamente.

### 2.28 Varredura de auditoria de 2026-08-02: 14 itens fechados, 5 abertos, 2 surpresas

**(A) O que foi feito.** Voce pediu para auditar cada item do §2 e do §5, medindo o estado do documento em
vez de ler o cabecalho do proprio item. Os 19 itens em escopo foram medidos contra a arvore em `45c75611`
mais a arvore de trabalho. **14 fecharam e foram para
[`_archive/PENDENCIAS_RESOLVIDOS.md`](_archive/PENDENCIAS_RESOLVIDOS.md)** com a evidencia e a sua decisao
preservadas verbatim; 51 citacoes a esses itens foram reapontadas para o arquivo, mais 3 no `GLOSSARY.md`
e neste arquivo, e o gate `check_tracker_refs` voltou a rc=0.

**(B) As duas surpresas, e as duas vao nas duas direcoes.**

1. **`2.26` estava dado por resolvido e nao estava.** Voce escreveu "Aplique o R15-10 e o R15-09" e nenhum
   dos dois havia sido aplicado: `"Two patterns stand out in the data."` e `"Settling that needs"` continuavam
   na prosa viva do apendice do cosseno. **Aplicados agora** (2026-08-02): `"The figure shows two patterns."` e
   `"Answering that question needs the same diagnostic"`.

2. **`EX-9` dentro do `2.23`: a sua revisao desfez a sua propria decisao.** Voce escreveu "nao aplique o
   EX-9", cuja familia eram quatro frases (`deserves one statement`, `worth reporting`, `needs saying`,
   `worth stating`). Todas as quatro **sairam** da prosa viva; `git log -S` mostra duas saindo no seu proprio
   `src_clean` (`807183c1`). Voce foi consultado e decidiu que a sua leitura com o texto na mao superseda a
   decisao anterior. Registrado como SUPERSEDIDO, nao como aplicado.
   **E o meu probe nao pegou isso:** o `A23-EX9` vigiava `"Pareto front"`, que continua no texto, em vez das
   frases que a decisao protegia — passava enquanto a decisao era desfeita. Reapontado para a definicao de
   fronteira de Pareto que voce de fato manteve, e validado nos dois sentidos.

**(C) Um item que o tracker dava por aberto e estava aplicado.** O `2.20` (italico em ingles corriqueiro no
Cap. 4): a sua opcao 2 esta aplicada. `\textit` na prosa viva do Cap. 4 = **48**, contra 157 no fonte
em `5c074a2a`; os sobreviventes sao os 7 nomes de categoria, nomes de modelo e substantivos proprios. Duas
formas arguveis sobraram (`one-hot`, `skip-gram`) e nao mexi nelas.

**(D) Segunda passagem, 2026-08-02: o §5 retirado, e o 2.21 e o 2.24 fechados.** O §5 foi **re-medido** depois da
fusao e virou ponteiro: os onze itens estao no arquivo e as conclusoes sobreviveram (o comando do proprio
banner ainda reproduz o que ele afirmava). O **2.21** fechou — o termo que o seu orientador marcou,
`license the verbs`, ja tinha saido do Cap. 2 na sua revisao, e a metafora foi trocada por `supports` nos
tres sitios vivos restantes mais a glosa do `GLOSSARY`; os usos em `apx_e_ethics.tex` ficaram, porque ali
`license` e licenca de software de verdade. O **2.24** fechou nas duas metades: a norma ABNT NBR 10520:2023
esta na §1 do `WRITING_LAW` com gate e self-test, dois fragmentos foram corrigidos e a citacao de frase
completa ficou por sua isencao; e o `towards` fica como esta por sua decisao, com a entrada do
`OPEN_REGISTER` como registro permanente dela.

**Sobram tres itens seus:** `2.1`, `2.5` e `2.27`.

*Forense: [`_round9/50_pendencias_audit.md`](_round9/50_pendencias_audit.md), com a medicao de cada um dos 19.*

## §5 · Retirado

Os onze itens levantados do `CODEX_AUDIT.md` quando ele foi arquivado (5.1 a 5.10 mais o 5.6b) estao
**todos fechados** e vivem em [`_archive/PENDENCIAS_RESOLVIDOS.md`](_archive/PENDENCIAS_RESOLVIDOS.md),
cada um com a medicao que o fechou e a sua decisao verbatim. O `CODEX_AUDIT.md` inteiro esta em
[`_archive/CODEX_AUDIT.md`](_archive/CODEX_AUDIT.md).

**Re-medido em 2026-08-02, depois da fusao da sua arvore revisada, porque um arquivo fechado nao e um
arquivo que continua verdadeiro.** As conclusoes sobreviveram: o comando que o banner desta secao
documentava ainda reproduz o que ele afirmava — quatro das cinco frases retiradas vem vazias e a quinta
aparece so em `tables/cbic/errata_wording.tex`, que e a tabela de errata, onde a redacao antiga esta como
evidencia citada e nao como alegacao viva. Os treze probes `COD-`/`NUM-`/`R8-` do
`check_audit_claims.py` seguem em `rc=0`, e o 5.6 (as duas datas do Gowalla) confere no render.

Sete dos itens tem sonda no gate, que **falha** se a correcao sair do documento; **dois nao tem**, e a razao
esta no banner do arquivo: o 5.6 foi verificado direto no render e o 5.10 e um registro de nao-pendencias,
nao uma afirmacao do documento. O decimo (COD-018, credito por papel no CoUrb) foi **retirado por voce**, e o
gate carrega a sua frase para que ninguem o "termine" por engano.

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

> **DECISAO SUA:** Nesse caso eu concordo com ele, é podemos só remover a frase: ", one of seven top-level classes.",
> isso já resolveria.

#### FAB-15 — census tract and mahalle belong to the data, not the introduction

**Citacao:** exata — `chapters/1_introduction.tex:54-56`. **Renderiza em:** 1_introduction.tex:54

**Minha leitura:** disagree, same reason. Same class as FAB-14: census tract / mahalle is what makes 'region' concrete.
Removing it leaves 'the official geographic unit' undefined until 2.4.

1. **Tirar census tract / mahalle da introducao** — atende; "unidade geografica oficial" fica sem definicao ate §2.4
2. **Manter** — contraria o pedido
3. **Mover para uma nota de rode** — compromisso; o Viegas usa notas assim

> **DECISAO SUA:** Vamos com a opção 1.

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

> **DECISAO SUA:** Opção 1

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

> **DECISAO SUA:** Opção 1. Voce pode se aproveitar da explicação de cosseno do appendix F, apesar de estarmos fazendo
> essa nova parte não precisa remover nada do appendix F, caso fique repetitivo.

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

> **DECISAO SUA:** Opção 1.

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

> **DECISAO SUA:** Aqui temos alguns problemas, vamos aos fatos. A parte que está com problema hoje é a: "The baseline
> was also tuned rather than taken as published: the cross-region edge weight of their Equation 2, set to 0.4 for the
> dense Chinese cities they study, was raised to 0.7 for the sparser United States state datasets used here. The sweep
> that fixed that value ran on Alabama over four settings of the weight, 0.4, 0.5, 0.6, and 0.7, each measured over five
> folds with a budget of 50 epochs, and the category F1 rose monotonically across them, from $0.7388 \pm 0.0205$ at the
> published setting to $0.8186 \pm 0.0123$ at the adopted one, on a zero-to-one scale, with the spread taken across the
> five folds."
> O ponto e que ela parece sem nexo, para quem está lendo rapido ou desatendo, não entende que estamos discutindo sobre
> o fato que os hyperparmsteeros usado pelo hgi em nosso artigo foi retirado do artigo base que se baseava em uma cidade
> chinesa. Esse é o primeiro ponto essa frase está confusa sem nexo, sengundo ponto ela está "overwhelming", ou seja,
> ela está com execesso de informação, podemos simplificar bem isso e caso o leitor uqe se interesee acesso o artigo
> deles para saber mais. Por fim um outro ponto, é quanto ao fato disso não pertencer a fundamentação téorica, é sim a
> métodologia. Eu proponho melhorar esse texto e deixar isso em um appendice.

#### GER-04 — the static-vector paragraph reads like introduction prose, and it matters

**Citacao:** exata — `chapters/2_fundamentals.tex:192-193`. **Renderiza em:** 2_fundamentals.tex:192-199

**Minha leitura:** agree it reads as introduction; he also says keep it. He called it well written and important, and
only observed it reads like introduction prose. There is no defect to fix here, only a placement question, and it is 108
chars from the NUM-4 probe string.

1. **Nao mexer** — ele mesmo disse que esta bem escrito e que e importante; nao ha defeito a corrigir
2. **Mover para o inicio de §2.2 como paragrafo de abertura** — atende a impressao de "texto de introducao"; mexe num
   paragrafo a 108 caracteres do probe NUM-4

> **DECISAO SUA:** Eu acredito que tenhamos que deixar ele onde está, na introdução já comentamos sobre o ponto de a
> literatura usar em sua maioria poi embedding ao inves de checking embedding, caso não comentamos isso bem lá, porfavor
> melhore o texto da introdução. Mas, quanto aqui eu acredito que temos que sim, manter esse paragrafo onde ele está ele
> server de cola e ponte falarmos sobre outros tipos de embedding, estes que serão usados no checking embedding. E aqui
> abrimos outra ponta no paragrafo: "Beforethatstep,severalgeneralencoderssupplythecontextastaticplacevectoromits.",
> precisamos melhorar sua conexão com o paragrafo anterior falando sobre esse ponto que o checkin embdding precisa de
> embedding temporais e locacionais e onde entra o conteduo do resto do paragrafo; valide essa minha ideaia.

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

> **DECISAO SUA:** Vamos então avaliar o wang2025hamtl, adicione no caminho:
> articles/dissertacao/science/articles/wang2025hamtl.pdf. Apos ler ele avalie quais adicionar na dissertação se todos
> forme interessante, adicionamos todos.

### 6.8 A sua propria pergunta

#### AUT-01 — does the MTL fundamentals need Pareto optimality

**Citacao:** -. **Renderiza em:** 2_fundamentals.tex 2.3

**Minha leitura:** agree it needs a brief treatment. The author's own added question: does the MTL fundamentals need
Pareto optimality. Since 2.3 names gradient-surgery balancers, and MGDA/CAGrad/Nash-MTL are all argued in terms of
Pareto-stationary points, the concept is already implicit. Note 'Pareto-stationary point' is ALREADY in the prose and is
PENDENCIAS_RESOLVIDOS 2.12 (arquivado 2026-08-02) (unregistered in the fail-closed GLOSSARY), so this item and 2.12 are the same decision.

1. **Um paragrafo breve em §2.3: o problema multitarefa e multi-objetivo, os balanceadores de cirurgia de gradiente sao
   argumentados em termos de estacionariedade de Pareto, e por isso o MGDA/CAGrad/Nash-MTL existem** — atende sua
   intuicao e da espinha ao paragrafo dos balanceadores; ~1h. Note que `Pareto-stationary point` JA esta na prosa e e o
   item 2.12 (nao registrado no GLOSSARY), entao este item e o 2.12 sao a MESMA decisao
2. **So registrar `Pareto` no GLOSSARY e nao expandir** — fecha o 2.12 sem crescer o capitulo
3. **Nao tratar** — o capitulo nomeia balanceadores cuja justificativa e Pareto e nunca diz isso

> **DECISAO SUA:** Vamos com a opção 1. Aqui vale notar que em uma ultima interação, realizamos o item 2.12, então
> avalie como está e se julgar necesssario voce adiciona mais contexto embasado com referencias.

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

> **DECISAO SUA:** Já fizemos algumas correções no appendix F, mas se esse ainda continuar com esses erros logicos,
> podemos seguir com a opção 1.

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

> **DECISAO SUA:**  Eu acho que isso já foi mudado no texto, mas se caso se manter mantenha everywhere para o categoria
> e especifique onde for preciso para o next-region.

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

> **DECISAO SUA:** Pode remover, e pare ser sincero e uma leitura rapida eu nem entendi o que esses numeros siginificam,
> então fará até melhor para leitura.

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

> **DECISAO SUA:** Vamos de A.

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

> **DECISAO SUA:** Vamos de 1.

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


### 6.12 Rodada 10 — as suas 28 decisoes, auditadas contra o texto VIVO antes de qualquer acao

Baseline `dda8978e`; entregue em `984c70be`, `aaf4e7eb`, `5415d86d`, `d7e8c598`. Auditoria completa em
`_round10/30_r10_audit.md`; relatorios dos itens grandes em `_round10/28_hamtl.md` e
`_round10/29_ch2_definitions.md`.

**Voce avisou que o texto tinha mudado, e o aviso foi a coisa mais util desta rodada.** Entre os itens
serem escritos e as suas decisoes chegarem, **60 commits** entraram na arvore. **Nove das 28 decisoes
pedem texto que nao existe mais** e foram fechadas como ja-feitas, nao aplicadas: FAB-10, FAB-14,
FAB-17, GER-03, GER-11, BLQ-1, BLQ-2 (no Cap. 6), BLQ-4, BLQ-5. O GER-03 e o caso mais claro: a sua
decisao apontava tres defeitos (sem nexo, "overwhelming", pertence a metodologia) e os tres estao
resolvidos por sua propria mao, com a varredura agora em `apx_g_hgi_tuning.tex` fechando num aviso de
escopo de que ela **nao** e evidencia sobre HGI contra Check2HGI.

**Quatro edicoes aplicadas, cada uma conferida no PDF renderizado nos dois sentidos:**

| item | o que faltava de verdade |
|---|---|
| BLQ-2 | UM site sobrevivia, em `1_introduction.tex`:289-292, e um grep por linha nao o via porque a frase quebra em tres linhas. Agora espelha o Cap. 6: categoria nos seis, regiao em quatro dos seis, TOST nos outros dois |
| BLQ-3 | os dois fatores derivados sairam do Apendice F; as quatro contagens de ponta ficam, cada uma rastreavel a tabela do §D.1 |
| FAB-08 | comentario registrando o que o Resumo omite: que **o par de tarefas muda** entre os estudos, que o corte foi deliberado na rodada 6, e onde o leitor recebe a informacao |
| FAB-22 | faltava a clausula que carrega o seu argumento: **Istanbul esta ali porque nao e dataset dos Estados Unidos**. Sem isso, Istanbul le como um sexto dataset e nao como evidencia de generalizacao |

**FAB-28 esta resolvido, e era o bloqueador da rodada 9.** Voce colocou o PDF em disco e ele foi lido
inteiro. A frase de ausencia do Cap. 2 **sobrevive sem mudanca**, por tres motivos independentes na
fonte: o HAMTL chama a predicao de localizacao de tarefa **principal** e a de categoria de **auxiliar**
(p. 2), logo as duas cabecas nao sao co-iguais; ele nao nomeia nenhuma unidade tipo regiao em 28 paginas
(uma varredura completa por region/grid/district/zone/administrative devolve **um** acerto, dentro do
titulo da referencia [40], Tobler 1970, p. 27); e ele nao reporta nenhuma metrica do lado da categoria.
**O que estava errado era a NOSSA descricao dele**, que nomeava o componente errado e escondia a
assimetria principal/auxiliar. Corrigida para as palavras dos proprios autores. Uma referencia entrou
(`wang2024iemtlf`, o trabalho anterior do mesmo grupo) com `[VERIFY]` no bib, porque so o titulo foi
verificavel.

**Uma constatacao honesta que o item nao previa:** o HAMTL **nao** abre para uma literatura grande de
MTL-para-POI nao citada. A propria secao de MTL dele quase nao cita mobilidade, e as dez referencias de
MTL em cascata sao imagem medica, PLN e recomendacao multi-comportamento. Nao foram abertas nem
propostas, e estao nomeadas como nao-alcancadas. Se voce quiser amplitude de MTL geral em vez de
MTL-para-POI, esse e o conjunto, e vale um item proprio.

**GER-04: voce pediu para eu validar a sua ideia, e ela esta certa e ja implementada** por voce. A frase
que voce citou nao existe mais. O §2.2 agora le: limitacao (um vetor fixo por lugar, logo uma manha de
quarta e um sabado a noite tem entradas identicas) -> "A per-visit representation needs temporal and
spatial context in addition to the identity of the visited POI" -> o inventario de encoders -> o nivel
de check-in que os consome. E exatamente a sua proposta. A metade da introducao tambem nao precisa de
nada: `1_introduction.tex`:125-131 ja carrega o argumento.

**GER-08 / GER-09 / GER-10 / AUT-01 sairam como um unico trabalho**, porque os quatro reestruturam o
mesmo arquivo. Onze definicoes numeradas e referenciaveis (2.1 check-in ate 2.11 conflito de gradiente,
esta com a formula do cosseno), com nove referencias cruzadas alem dos onze blocos, que e o que faz da
coisa a narrativa que o GER-10 pede em vez de um deposito de definicoes. **Zero crescimento de pagina**,
e o modo importa: as definicoes custaram uma pagina, o `make check` ficou vermelho com quatro contagens
obsoletas, e em vez de rodar `sync_page_counts --write` para o gate concordar com um documento mais
longo, o texto foi compactado ate a contagem voltar a 106 e o gate ficar verde sozinho. Nenhuma probe
foi enfraquecida. O AUT-01 foi avaliado como **ja suficiente** e nada foi acrescentado, o que a sua
decisao autorizava ("se julgar necessario").

#### O que espera VOCE, e nao e trabalho que eu poderia ter feito

1. **Duas linhas de registro no GLOSSARY**, propostas e nao usadas como termo novo: `soft parameter
   sharing` (compartilhamento flexivel de parametros) e `negative transfer` (transferencia negativa).
   As duas expressoes **ja estavam na prosa viva** antes desta sessao e o §6 registra so `hard parameter
   sharing`, entao converter prosa em blocos de definicao nao ampliou a regra fail-closed. Mas as linhas
   deviam existir, e linhas de registro sao suas.
2. **A Definicao 2.7 junta dois conceitos** (representacao em nivel de lugar e em nivel de check-in) sob
   um mesmo cabecalho, para segurar as 106 paginas. Separar custa um numero de definicao e cerca de
   quatro linhas, que teriam de ser recuperadas em outro ponto do §2.2.
3. **Nove atribuicoes de linhagem, das quais eu reconferi cinco** nos registros de origem (Shikun Liu /
   DWA, Bo Liu / CAGrad, Bo Liu / FAMO, Zhao Chen / GradNorm, Aviv Navon / Nash-MTL, todas confirmadas).
   As outras quatro repousam na verificacao do agente, sem passe de critico. Digo em vez de omitir.
4. **FAB-27, a margem: medida e nao reproduz.** Zero `Overfull \hbox` nos quatro builds; extracao de
   tinta pagina por pagina nas 106 paginas contra o bloco de texto real (455,24pt, perguntado ao TeX)
   poe **toda** pagina 0,5 mm **dentro** do bloco, a mais apertada em 1,99 cm contra a regra de 2 cm. A
   sua observacao era verdadeira e ja foi corrigida (o commit `6d780b58` se chama "fix the p.96
   overflow"). Nao ha o que mexer, e se voce ainda ve a tabela passando, me diga em qual pagina do PDF.

### 6.13 Rodada 12 — dois defeitos que VOCE achou nas definicoes, e duas linhas de registro que eu nao posso escrever

**Voce encontrou o que o item 7 (passe critico nas definicoes) existia para encontrar**, antes do passe
rodar. Os dois sao lacunas de formalizacao, nao afirmacoes erradas, e os dois ficam exatamente nos pontos
onde o capitulo carrega o argumento.

**Defeito 1 — a Definicao 2.8 nao tinha simbolo.** A 2.7 da o vetor de nivel de lugar como
$\mathbf{e}_p$ e a 2.8 dizia apenas "um vetor para cada check-in". O contraste lugar-versus-check-in e o
pivo da dissertacao inteira, e so um dos lados podia ser escrito. Corrigido para $\mathbf{e}_{x_i}$.
Medido antes de cunhar: `\mathbf{e}_{x_i}`, `\mathbf{e}_{x}`, `\mathbf{e}_i`, `\mathbf{z}_i` e
`\mathbf{z}_{x_i}` aparecem ZERO vezes na arvore viva, entao nada foi sobrecarregado; os unicos
subscritos de $\mathbf{e}$ em uso eram $p$ e os genericos $1$/$2$/$+$/$-$ das equacoes do
discriminador. O simbolo espelha o $\mathbf{e}_p$ e muda so o que o subscrito indexa, que E a distincao
sendo definida. Probe `R12-eqxi`.

**Defeito 2 — a Definicao 2.6 era a unica tarefa em prosa pura.** A 2.3, a 2.4 e a 2.5 dao uma funcao
($g_{\mathrm{cat}}(\mathbf{e}_p) \to c_p$, $f_{\mathrm{cat}}(H_i) \to c_i$,
$f_{\mathrm{reg}}(H_i) \to r_i$) e o next place nao dava. Isso fazia a tarefa excluida parecer um tipo
DIFERENTE de objeto justamente onde a funcao do capitulo e manter as tres formalmente distintas.
Corrigido para $f_{\mathrm{place}}(H_i) \to p_i$, e nomear a funcao AFIA a declaracao de escopo em vez
de enfraquece-la: a exclusao passa a incidir sobre um objeto definido. A frase agora diz que nenhum
capitulo reporta resultado para $f_{\mathrm{place}}$. Probes `R12-fplace` e `R12-fplace2`.

> **DECISAO SUA — duas linhas do GLOSSARY §1.1, que e sua e nao minha.** Eu usei os dois simbolos na
> prosa do Capitulo 2 porque eles instanciam objetos ja registrados ($x_i$, $H_i$, $p_i$, $\mathcal{P}$),
> mas a TABELA DE NOTACAO e um registro, e linha de registro e sua. As duas propostas:
>
> | Simbolo | Definicao proposta | Nota |
> |---|---|---|
> | $\mathbf{e}_{x_i}$ | A representacao aprendida do check-in $x_i$. | Entrada de nivel de check-in; e o "per-visit Check2HGI vector" que a linha do $\mathbf{e}_p$ ja menciona sem nomear. |
> | $f_{\mathrm{place}}(H_i)$ | Preditor sequencial cujo alvo e o proximo POI $p_i$. | Nomeado apenas para delimitar escopo; nenhum capitulo reporta resultado para ele. |
>
> 1. **Registrar as duas** — fecha a lacuna que a propria linha do $\mathbf{e}_p$ aponta ("distinct from
>    a per-visit Check2HGI vector", sem simbolo).
> 2. **Registrar so o $\mathbf{e}_{x_i}$** — o $f_{\mathrm{place}}$ existe para ser excluido, e voce pode
>    preferir nao registrar notacao de uma tarefa que a dissertacao nao executa.
> 3. **Nenhuma das duas, e eu retiro os simbolos da prosa** — volta ao estado anterior, com os dois
>    defeitos que voce achou de volta junto.
>
> **DECISAO SUA:** ______

### 6.14 Rodada 12 — as suas quatro descobertas nas definicoes, resolvidas em projeto, e duas decisoes que sobraram

**Voce achou um defeito estrutural, e ele mede como real.** O grafo de dependencias das doze definicoes
tem **exatamente uma** violacao da ordem de leitura: a **2.3** consome $\mathbf{e}_p$, que a **2.7**
introduz 318 linhas depois. Todas as outras consumacoes sao para tras e legais. Medido, nao estimado
(`_round12/46`).

**A sua pergunta sobre a literatura tem resposta, e voce estava certo.** Quatro fontes abertas nesta
sessao, cada uma citada do proprio texto (`_round12/48`): CSLSL (Def 1 registro = tupla, Def 2 trajetoria =
sequencia de REGISTROS), CTLE ("a trajectory consisting of sequential visiting records", com o embedding
como uma FUNCAO $z(l)$), HAMTL (Def 1 ponto, Def 2 trajetoria de pontos) e **o seu proprio MobiWac**
(`03_problem.tex`: "Given a user's time-ordered check-in history"). **Quatro de quatro definem a sequencia
sobre observacoes cruas e introduzem a representacao depois, como um mapa.** O caso decisivo e o CTLE:
um artigo cuja contribuicao E o embedding e que ainda assim separa as duas camadas.

**O QUE ISSO MUDOU NO PROJETO.** A correcao da sua descoberta 2 NAO e redefinir $H_i$ sobre embeddings. O
$H_i$ continua sequencia de check-ins e entra um **mapa de representacao nomeado** $\rho(x_i)\in\mathbb{R}^d$,
estendido a historias, antes das tarefas. O argumento que sustenta isso nao e "a area faz assim": e que a
sua tese central e que a REPRESENTACAO domina, e isso so e enunciavel se a tarefa for o MESMO OBJETO nos
tres estudos enquanto a representacao varia. Definir a historia sobre embeddings faria a definicao da
tarefa mudar de capitulo para capitulo e destruiria o ponto de referencia fixo contra o qual a tese e
medida. As tres alternativas rejeitadas estao argumentadas em `_round12/47`.

**Sobre o $\mathbf{e}_{p_i}$: voce estava meio certo, e a metade importa.** O $\mathbf{e}_p$ **fica** na
tarefa estatica, porque ela e quantificada sobre POIs e nao existe indice de check-in ali; renomear
faria uma tarefa estatica parecer sequencial. O que uma POSICAO de historia carrega e $\mathbf{e}_{p_j}$,
a composicao de $j\mapsto p_j$ com $p\mapsto\mathbf{e}_p$, e o projeto tira essa composicao da prosa e a
poe numa equacao. A sua descoberta 4 se dissolve nessa fatoracao: a equacao
$f_{\mathrm{place}}(H_i)\longrightarrow p_i$ esta correta como esta.

**Um defeito que nenhum de nos dois listou:** $c_p$ aparece LIVRE na Definicao 2.3. Medido: ocorre uma
unica vez na prosa viva, dentro daquela definicao, sem introducao previa; e a linha do
$g_{\mathrm{cat}}$ no GLOSSARY §1.1 ja o usa, ou seja, o registro antecipava a ligacao que faltava.

**Nenhum probe quebra.** Os 21 probes que pinam este capitulo continuam verdes e as tres cadeias pinadas
(`R12-eqxi`, `R12-fplace`, `R12-fplace2`) foram carregadas caractere por caractere no projeto. Verificado
por mim, nao aceito do relatorio.

> **DECISAO SUA 1 — doze definicoes ou treze.** O mapa $\rho$ pode entrar como uma **definicao numerada**
> (fica 2.1-2.13, e o mapa ganha o mesmo peso visual das outras) ou como uma **equacao em display** na
> prosa que antecede as tarefas (fica em doze, e o mapa aparece como notacao). As duas formas estao
> especificadas em `_round12/47` §3. Nao ha diferenca matematica; a diferenca e o quanto voce quer que a
> fatoracao "mesma tarefa, representacao variavel" salte aos olhos de quem le, dado que ela e a forma
> logica da sua tese.
>
> **DECISAO SUA:** ______

> **DECISAO SUA 2 — como chamar a entrada do Capitulo 4.** O projeto descreve a entrada dos Capitulos 3 e
> 4 como de nivel de lugar, e para o Capitulo 4 isso e impreciso. Medido em
> `4_courb/methodology.tex:153`: o componente temporal "represents the timestamp of each check-in", e
> `:144` diz que ele mapeia "the temporal values of each check-in (hour of day and day of week)". Ou seja
> a concatenacao do Capitulo 4 tem um canal que varia POR VISITA, mesmo com o canal categorico
> (HGI) sendo por lugar. Tres saidas:
> 1. **Chamar o Capitulo 4 de hibrido** — dois canais por lugar mais um por visita. Mais preciso, e
>    enfraquece um pouco a linha narrativa "lugar -> check-in" que o Capitulo 5 fecha.
> 2. **Manter "nivel de lugar" com uma ressalva de uma frase** nomeando o canal temporal como a excecao.
>    Preserva a narrativa e nao esconde o fato.
> 3. **Manter como esta** — o vetor ainda e indexado pelo lugar visitado no Capitulo 4, e o canal temporal
>    e uma feature dele. Defensavel, mas um leitor atento do `methodology.tex` pode perguntar.
> O Capitulo 4 e versao de registro e nao muda; a decisao e sobre como o Capitulo 2 o descreve.
>
> **DECISAO SUA:** ______

---

## §3 · Aberto e bloqueado em terceiros

| Item                                               | Bloqueado em                     | Estado                                                                                                                                                                                                                                        |
|----------------------------------------------------|----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Dois membros da banca e a data da defesa           | Orientador / PPGCC               | Placeholders entre colchetes em `preamble.tex:217-219`. **Nao imprimem em nenhum dos tres builds** (`\folhadeaprovacao` esta comentada em `abntex2-UFV.sty:166-170`), entao nao ha nada inventado no PDF — nem os nomes reais quando chegarem |
| Folha de aprovacao assinada                        | A defesa                         | `make ppgc` gera o PDF com o placeholder; a versao assinada o substitui depois                                                                                                                                                                |
| Status do MobiWac                                  | Revisores                        | A redacao e sempre "submitted, under review", em todo o documento. **Nao mudar** ate haver decisao                                                                                                                                            |
| `\finalbuildfirstpage` conferido contra o RASCUNHO | Upload pos-defesa ao AcademicoPG | Agora **9** (`main.tex:95`), das 8 paginas pre-textuais do build de deposito; a primeira pagina de corpo do `main_academico.pdf` e a fisica 9 e imprime 9. Confira contra o RASCUNHO quando subir                                             |

---

## §4 · Pensamentos e considerações do Autor

1. No resumo e na introdução, valha fazermos uma analise sobre o uso de especificidades, e por especificidades me refiro
   a menção como: "O modelo final foi avaliado em cinco estados dos Estados Uni- dos, extraídos do Gowalla, e em
   Istambul, extraída do Massive-STEPS"(resumo), "vinte modelos ajustados por configuração, quatro inicializações
   aleatórias sobre cinco partições fixas e testes pareados sobre as médias de cada inicialização"(resumo) ou "The
   category space contains Community, Entertainment, Food, Nightlife, Outdoors, Shopping, and Travel."(2.1.1.2). Vide
   que o nosso trabalho ele é generico, podemos usar qualquer dataset e quaquer N numeros de categorias. Assim, nos
   texto que não explicativos de métodologia ou sobre os dados em si, podemos ser mais genericos; no resumo que dei
   exemplo podemos usar algo como: "O modelo final foi avaliado em 6 datasets, sendo 4 localizados nos U.S e afins de
   generalização um não U.S"(Temos que refinar bem essa frase é só um exemplo). Ainda sobre esse tópico, no resumo tabém
   falamos
   "considerando uma margem de dois pontos de Acc@10 pelo procedimento TOST", outra especificação que não vejo
   necessidade.
2. Padronização das palavras tecnicas. Palavras que já estão no "List of abbreviations and acronyms" estão sendo
   escritas de forma distintas pelo texto. Um exemplo é Point of Interest que em muitos locais aparece como
   Point-of-Interest. Outro detalhe é o uso correto dessas palavras como no caso de Multi-Task Learning, essa palavra no
   artigo original é escrita como: Multitask Learning, sem o hifén sem contar que o APA Style, também define a
   preferencia por uso de plavras sem hifén; enfim faça uma pesquisa para validar essa é outras palavras técnicas e
   vamos substituir onde necessario.
3. Na introdução a frase: ",while place categories provide the semantic information used by location-based
   services [4].", em uma leitura rápida parece não ter nexo com o que está sendo dito anteriormente ou parece ser uma
   frase só jogada sem muito contexto.
4. Na frase: "Some methods used in this dissertation also come from neighboring geospatial tasks. In particular, the
   spatial encoders examined in the second study were first validated for applications such as
   speciesrecognitionandremote-sensingclassification [5, 6].", será que já fala de métodologia no primiero paragrafo é
   interessante, estamso contrunido a problmeatica e as bases do problema para o leitor, é o que seria "neighboring
   geospatial tasks", me parece bem solto.
5. Na frease: "Because next category and next region are predicted from the same visit history, one model may serve both
   tasks.", temos que na frase anterior defimos 3 tarefas que treinamos nos modelos conjuntos. Será que essa frase não
   serai melhor escrita como algo: "Because the previous tasks cited consumes the same visit history, one unify model
   ,such as Multitask learning (MTL),may server for them. MTL ..."(Temos que melhorar)
6. We are using the term: "static place categories" to refear to the poi classification of a unkown POI category. But in
   the §1.1 we are define this task as "category classification, a static task that predicts..."; my point is for a
   reader that are not familirazed with the tasks and are nobie about this point, this will create confusion. FOr the
   other tasks we define them and use the same jarguan throught the text.
7. Na frase: "Under a check-in level representation, static category classification is a less natural companion task
   than a second sequential target, so the final task pair becomes next category and next region.", acho que a
   explicação do porque da mudança ainda falta conteudo, principalmente, pq outro motivo é diria até que mais forte apra
   mudança foiq eu na literatura essas outras duas tarefas possuem mais forças que a classifiçação de poi; sem contar
   que temos outro problema que não sei se temos que explicitar, mas sobre regime de checking level e nosso motor de
   embedding a o embedding pode vazar e deve vazar qual a categoria do checking atual, até para o modelos conseguir
   prever com mais exatidão a proxima categoria.
8. A frase: "The dissertation does not treat these as one unchanged experiment. It presents a sequence in which a result
   valid for one configuration leads to a diagnosis and then to a different, explicitly bounded solution."; Eu acredito
   que podemos melhorar bem, na clareza ainda mais quando se comparado com a que estava antes: ", and this dissertation
   names that evolution plainly rather than narrating one fixed experiment. The arc it reports is a correction trail:
   each study revised what the previous one concluded, and the later chapters state precisely which earlier conclusions
   they supersede."
9. We use hard share paramter through all the introduction and never explain what is: An common MTL apporach on the
   paramter and the data flow are shared. Eval this.
10. No paragrafo: "Evaluate whether a joint model with hard parameter sharing benefits static category classification
    and next-category prediction when compared with dedicated single-task models (Chapter3).", Acho que podmes incluir:
    "Evaluate on how an MTL model can be build and work on the POI field and whether a joint model using hard sharing
    benefits poi classification and next-category...."
11. No paragrtafo: "Consolidatetheevidenceundertheuser-disjointcross-validation,significance-testing,and
    non-inferiorityprotocolusedinthefinalstudy (Chapter 6)." isso tá errado é no Chapter 5.
12. No paragrafo: "The joint setting imposes a single-model constraint: one trained artifact must produce both outputs
    in one forward pass.". Que joint settings ?
13. Na parte de contribuições não estamos a destacar achados importantes, e ela não está bem escrita faça uma avaliação
    mais profunda. Vou dar alguns exemplos: - O check2hgi é um avanço quanto ao uso de um embedding de mobilidade que se
    utiliza checkin ao inves de poi, e este pode ser usado em varios trabalhos futuros com difererntes propositos; -
    Nosso MTLnet final ou o joint model final ele é um modelo que pode ser usado para treinamento conjunto das tarefas
    ou ainda pode ser expandido para outras tarefas dado sua modularidade; - O achado que as tarefas parecem não serem
    conflitantes em um modelo MTL. (Esse tem que tomar bastante cuidado);; - Nossos artigos são pioneiros na utilização
    e MTL para essas duas tarefas, tarefas essas que podemos ter um escopo mais abrangente que o de next-poi Por favor,
    avalie eses pontos, avalie os que já estão e faca uma analise pelo texto e trabalho para ver se não estamos
    esquecendo nada.
14. A frase: "It indicates that mobility is learnable, but it is not a reference point for the category and region
    metrics defined in Section~\ref{sec:fund:eval}." Para mim não faz sentido dizer "It indicates that mobility is
    learnable, but it is not a reference point for the category and region metrics defined in Section~\ref{sec:fund:
    eval}.", justamente o contrario podemos sim ter os estudo de mobility e next-location como referencia para o
    category e regio. A frase original era: "This bound is specific to next-location prediction at coarse spatial
    resolution; it shows that mobility is far from random and is learnable at all, and Section~\ref{sec:fund:eval}
    states the reference points that actually bound the category and region tasks studied here."
15. Eu tô com um medo, eu posso estar deixando passar batido por já ter lido varias vez, mas se referir as tarefas como,
    sequenciais e estaticas, como no exemplo: "three experimental tasks, two sequential and one static.", para mim faz
    muinto sentido, só que será que estamos explicando isso bem no texto ? E estamos explicando antes de usarmos no
    texto, as vezes deixa a explicação na lista de abreviassões e accronimos ?. Faca uma analise.
16. O §2.1.1.1 foi uma introdução bem legal, porém está precisa ser revisada, é precismos ser mais precisos na
    explicação, algumas lacunas ainda estão presentes como o fato de não falarmos o que é 𝑥𝑖 ou H𝑖. Da onde saiu 𝑐𝑝, 𝑐i
    e ri, precisamos explicar o que é isso, estamos tacando simbolos sem explicar.
17. No §2.1.1.1 na frase: "The category space contains Community, Entertainment, Food, Nightlife, Outdoors, Shopping,
    and Travel. The region target is a census tract in the United States datasets and a mahalle in Istanbul. ", listar
    as categorias em um definição técnica é um erro, até pq nosso modelo poderia usar N categoria e não só essas 7, além
    disso citar os datasets não é algo para agora mas para a sessão §2.4
18. No §2.2.2 se graph-infomax é tão importante para o trabalho, temos que explicar em linhas gerais o que ele é e como
    funciona.
19. No §2.2.2 a frase: "The representations used in this work are trained without category or region labels.", tem que
    ser dita com bastante cudidado, por que no hgi vamso sim usar o categoryu como target, não usamos nos dois primeiros
    para não gerar vazamento de dados para tarefa estatica.
20. Não sei se no §2.2.2, mas no check2hgi, como descrito
    em:  [check2hgi_v17_complete_picture.md](../science/check2hgi_v17_complete_picture.md) tabém usamos POI2vec, não
    teriamos que citar ele ?
21. A frase: "MTLnet uses FiLM to condition its shared layers on task identity. Chapter 4 keeps this architecture but
    replaces its single place embedding withspatial, temporal,and categoricalencoders.The controlled changeisolates the
    effectoftheinput representation.". não deveria estar no §2.2.3.1, deveria estar na sessão sobre MTL
22. On the §2.2.3.2 get more context in the
    file: [check2hgi_v17_complete_picture.md](../science/check2hgi_v17_complete_picture.md) if necessary, and evla if
    what is in there is correct.
23. A frase: "The representation changes are paired with a controlled progression in model architecture." não ficou
    clara para mim, é não pareceu uma boa frase de transição.
24. A frase: "The models therefore differ in their sharing topology and in the private input available to the region
    output." Temos que explicar isso melhor, no MTLnet ele já recebia duas entradas, a diferença e que as duas entradas
    lá era de um mesmo embedding, aqui os embeedings são saidas diferetned do check2hgi, apesar de serem diferentes elas
    ainda possuem correlação.[VALIDE E PESQUISE MAIS NA CODEBASE]
25. Será que nomear o joint model, como MTLChkNet agora seria muito tarde ? Alguma outra sugestão de nome melhor ?
    Podemos, até atualizar no mobiwac, vide que ainda está em revisão ?
26. No §2.3.3 não explicamos o que é o `L𝑘`, also the explanation make in the §2.3.3 and §2.3.2.1 is working but i
    belive that we can improve make it more easy to read the concepts better and the constructioin of the logic flow
    easy to follow and undertand.
27. Reading more about pareto make me think, shouldn't we have some claim about the pareto property that we enconunter
    in the chapter 5 ? Even if this claim came in the appendix F ?
28. The §2.3.2 and §2.3.3 are very poor organized and repetitive, the concepts are out of order and the paragraphs
    requeries read more the once and go and back on other paragraphs to have a complet explanation. My take would be
    start wiht the §2.3.2.2 that define the problem, from the problem we formal define it with the §2.3.2, then we
    discuss the current options of the literature witht the §2.3.3, then we closes with §2.3.2.1 and with the part B of
    the §2.3.2.2 where discuss about the chapter 5 finds and the appendix D. what do you think ?
29. Na frase: "Equivalently, the reported OOD-discounted Acc@10 is the in-distribution Acc@10 multiplied by one minus
    the out-of-distributionfraction." eu não a entendi, sem contar que não estamos explicando o que é OOD.
30. No §2.4 precisamos reorganizar a ordem das sub seções e melhora a escrita de algumas, entre a §2.4.1 e a §2.4.2,
    temo que criar um nova chamada `preparation and data split` onde vamos pegar o que já temos no segundo paragrafo do
    §2.4.3 e descorrer mais sobre como os dados estão sendo preparados é a metodologia de split e separação antes dos
    dados entrarem no modelo. Enfim, seguimos para o `Metrics andreferencepoints` e o §2.4.3 vira
    `Comparision and statistical decisions` onde descorremos sobre o problem de comprar diferetnes resultados e como
    criamos uma métodologia estruturada para isso.
31. No primeiro paragrafo do chap 6, é falado: "This dissertation examined whether multitask learning helps
    next-category and next-region prediction and what determines the answer." Mas, esquecemos de falar sobre o
    poi-classification.
32. No paragrafo 2 do §6.2, onde temos: "so the gain does not come from the region task teaching the category task;"
    Isso está bastante errado já analisamos isso e validamos que na verdade esse não e o big picture e que essas
    analises, só comprovam que o loss não estava contribuindo para o ganho, mas a métodologia do cross-switch e outros
    artefatos ainda continuam auxiliando no ganho. (Pesquise e se aprodunde sobre isso); Além disso avali se esse mesmo
    erros está acontecendo em outras partes do texto.
33. On the "Contributions by chapter", we should focus less on the results and numbers and show more the conecptual
    contributions and finds, use numbers and results only if very necessary. We should reserve the results for the
    `The consolidated answer`, where we should show the results not exensivally, but we an show it more here.
34. Sobre as limitações, em §6.3 tenho alguns pensamentos sobre eles: - The data vintage is a problem, but we use the
    Massive steps from 2025; - The `Transductiverepresentation` desirves a huge warn that this is a problem of several
    apporachs in the literature; - The `The task-pair confound.` I am against it, the problem of isolates the previus
    MTLnet wiht the check2hgi, is that the check2hgi is a checking embeeding so the poi-classification would recive a
    data-leack.
35. Sobre o future works tenho outros pontos que considero essenciais de serem detacados e discutidos: - Melhor
    integração do check2hgi, hoje ele possui um arch que varias partes são acopladas como o Poi2vec, poderiamos tentar
    fazer algo mais integrado; - O testar o uso de outras abordagens modernas de MTL, usando soft-sharing; - Testar com
    mais categorias além de 7; - no check2hgi testar hypergraphs, assunto envolga no contexto the mobility; - Executar
    para mais datasets não U.S; - Testar cascate no junto ao MTL; - O embedding já serve para tentar trainar para o
    next-poi, as vezes podemos analisar alguma ou outra feature que podemos adicionar, mas do jeito que está hoje já
    conseguimos usar, basta modificarmos a pipeline de inputs e criar uma cabeça para o next-poi e acopla-la no nosso
    joint-model; (Eu vejo esse sendo o mais promissor de todos.)
36. Eu gostaria de uma avalaição critaca da conclusão eu tenho a impressão que ela está em um bom caminho, mas ainda
    falta algo para ela ficar melhor. Compare com o que os articles/dissertacao/exemples fazem nas dissertações deles.
    Ainda sobre a conclusão meu maior problema está sendo com o `The consolidated answer` esse tem um conteudo bastante
    interessante, mas parece se focar muito em numeros que já foram mostrados nos artigos, assim acho que aqui seria um
    lugar para mostra os numero mais de forma geral ficando na narrativa da resposta final achada. Essa parte ela bem
    importante, pq ela fecha o arco do artigo ela tem que ser prazerosa e facil de sere lida. O fluxo em alto nivel: ```
    Question and thesis-> Chain of cause and effect (explain the chain of discovers that leaves to the resutls) ->
    Show what we got wrong in your initial thesis ->
    Connect to the real lesson and results through the lens of the discovers.
    ```
37. 
    
