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

### 2.1 Os marcadores `[NEEDS SIGN-OFF]` no fonte — **56** medidos em 2026-08-02, agora com mapa item por item

**O que e.** Pontos do fonte marcados como precisando do seu aval. Nenhum bloqueia build, e **nenhum aparece no PDF**:
todos vivem em comentario `%`. **O numero anda** — tracks paralelas removem marcadores conforme voce decide.

**Novo em 2026-08-02: [`src_utils/NEEDS_SIGN_OFF.md`](NEEDS_SIGN_OFF.md)** traduz os 56 marcadores para PT-BR, um
por um, com contexto, a pergunta exata e um espaco `> **SUA DECISAO:**` para voce responder — o mesmo padrao
deste arquivo. Cada item foi conferido contra o fonte vivo (`grep` na linha exata) antes de entrar no mapa.
Quando um item for resolvido la, ele sai daquele arquivo e o `[NEEDS SIGN-OFF]` correspondente sai do `.tex`. Confie no comando, nao no titulo:

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

**PASSE DE VALIDACAO ADVERSARIAL — cinco defeitos, todos de redacao, nenhum estrutural.** Um segundo
agente fable-5 reconstruiu o grafo de dependencias por conta propria e enumerou os 21 probes contra o
texto PROPOSTO. **Estrutura aprovada; zero probes quebram.** Mas achou tres coisas que teriam entrado
erradas, e eu conferi as tres no fonte:

- **F-1, e e a que importa.** O projeto dizia que a instanciacao do Capitulo 5 e $\rho(x_i)=\mathbf{e}_{x_i}$.
  `5_mobiwac/04_method.tex:27` diz outra coisa: "the category task reads the window of per-visit vectors
  (the semantic stream); the region task reads the same window of visits, **each visit now represented by
  the trained vector of its region node** from the same graph (the spatial stream)". Sao DUAS correntes,
  nao uma. Aplicado como estava, o Capitulo 2 descreveria errado justamente o estudo em que o arco se
  resolve. A conclusao estrutural nao muda: $H_i$ cru mais um $\rho$ nomeado continua valendo, e o $\rho$
  simplesmente tem duas instanciacoes ali.
- **F-2.** "every predictive model in this dissertation reads $\rho(H_i)$" e falso para a tarefa estatica:
  `2_fundamentals.tex:116-124` mostra $g_{\mathrm{cat}}(\mathbf{e}_p)\to c_p$, sem historia nenhuma. E a
  tarefa estatica e exatamente a que o `apx_b_static_scope` existe para manter separada.
- **F-4, que fecha um item aberto do proprio projeto.** Ele anotava que "os estudos" preenchem janelas
  curtas com zeros. `5_mobiwac/05_setup.tex:28`: "we keep only the full-length window ending there and
  **drop these padded duplicates**". O Capitulo 5 descarta, nao preenche.
- **F-3** (o PAR de tarefas muda no Capitulo 5, entao a frase correta e "hold the task definitions fixed")
  e **F-5** ($d$ ja esta em uso vivo nos capitulos publicados como dimensao e como distancia $d_{ij}$,
  entao a linha de registro precisa de nota de escopo em vez de alegar notacao livre).

**Plano de edicao em 8 passos** escrito em `_round12/49`, com os passos 3-5 obrigatoriamente no mesmo
commit, os comandos que encontram cada referencia cruzada, e o aviso de validar cada probe por sabotagem
lendo o resultado ANTES de restaurar.

> **DECISAO SUA 3 — como o Capitulo 2 enuncia a entrada de duas correntes do Capitulo 5** (nova, vem do
> F-1). Duas saidas:
> 1. **Nomear as duas correntes na propria observacao** — precisa, e antecipa no Capitulo 2 um detalhe de
>    arquitetura que o Capitulo 5 explica.
> 2. **Dizer so que o Capitulo 5 instancia no nivel de check-in e remeter as correntes ao Capitulo 5** —
>    mais leve, e deixa a observacao incompleta para quem le so o Capitulo 2.
>
> **DECISAO SUA:** ______

### 6.15 Rodada 12 — as suas tres respostas, e a decisao 2 volta com a pergunta consertada

**Decisao 1 — TREZE. FECHADA.** "Vamos de treze." O $\rho$ entra como definicao numerada e o capitulo
passa a 2.1-2.13. Consequencia mecanica conferida, nao assumida: a renumeracao desloca tudo depois do
$\rho$, entao cada `\ref{def:fund:*}` sera reconferido na aplicacao; e o probe `R11-def27` pina
`\label{def:fund:checkinlevel}`, que e um LABEL e nao um numero, logo a renumeracao nao pode quebra-lo
(verificado contra a tupla de probes carregada, nao contra a minha memoria dela).

**Decisao 3 — NOMEAR AS DUAS CORRENTES. FECHADA.** Voce confirmou o defeito F-1 e deu o mecanismo: "o
HGI produz dois embeddings finais um de regiao e outro de checking, e usamos essas duas entradas,
respectivamentte, next-region e next-category". O "respectivamente" e o que decide a favor de nomear as
duas em vez de remeter ao Capitulo 5, e a observacao de fatoracao no `fundamentals/DEFINITIONS.md` ja diz
exatamente isso, com a citacao de `5_mobiwac/04_method.tex:27`.

**Decisao 2 — REABERTA, porque a minha pergunta estava mal posta e o codigo mostra isso.** Voce pediu o
estudo no codigo e em `docs/`, e ele **inverte a premissa**. Nao ha transformacao de nivel de check-in
para nivel de POI no pipeline do Time2Vec: nao existe `groupby`, media, nem reducao por POI em lugar
nenhum. O que existe e uma **recusa**:

- `src/data/inputs/builders.py:40` poe `TIME2VEC` em `_CHECKIN_LEVEL_ENGINES`;
- `builders.py:191-192`, o construtor da entrada da tarefa de categoria, **levanta erro** nesses engines:
  "Rejects check-in-level engines (Time2Vec, Check2HGI) -- category task requires one embedding per POI";
- `research/embeddings/time2vec/README.md:66-69` diz o mesmo e aponta aquele arquivo como o lugar da
  recusa, e declara a saida como `N_checkins x (6 metadata + 64 dims)`;
- `docs/context/EMBEDDINGS.md:83` "Type: Check-in-level (one embedding per visit event)", e :103-104 diz
  que a janela do next vira "a true spatio-temporal trajectory, **not just a POI sequence**".

**E a tensao esta DENTRO do capitulo publicado, nao so entre a prosa e o codigo.** O `methodology.tex:93`
diz que "each POI is represented by the embedding resulting from the concatenation of the three
components, $\mathbf{E}_{cat}=[\mathbf{E}_{HGI}\|\mathbf{E}_{loc}\|\mathbf{E}_{time}]$ ... forming pairs
$(\mathbf{E}_{cat},c)$ where $c$ is the real category of the POI" — o que exige um valor de
$\mathbf{E}_{time}$ por POI. O `:153` diz que $\mathbf{E}_{time}$ "represents the timestamp of each
check-in" — um valor por CHECK-IN. **As duas frases nao podem ser ambas verdadeiras como estao escritas.**

**O que eu NAO posso resolver, e por que.** Nao consigo estabelecer o que a RODADA PUBLICADA do CoUrb
fez. O capitulo e versao de registro de experimentos anteriores; o codigo em disco e o de hoje. Tres
possibilidades ficam abertas e so voce fecha, porque distinguir precisa dos artefatos da rodada e nao da
arvore de fontes:
1. a rodada publicada agregava o $\mathbf{E}_{time}$ para nivel de POI, e esse passo foi removido depois;
2. a tarefa de categoria da rodada publicada rodou SEM o canal temporal, e o `:93` superdeclara a entrada;
3. a rodada publicada antecede a guarda e alimentou a tarefa de categoria com vetores de nivel de
   check-in, o que faria os numeros publicados de categoria descreverem uma entrada diferente da que o
   `:93` afirma.

Se a resposta for (2) ou (3), e materia de errata sob o `NORTH_STAR` §5.7, e o apendice de errata **nao**
carrega isso hoje (conferido: ele trata do lead da tabela de categoria e de um re-typeset tipografico
desta mesma passagem em `:311-318`, e nada sobre o nivel do canal temporal). **O Capitulo 4 nao foi
editado**, e nem sera por mim: e versao de registro.

**Consequencia para o Capitulo 2, que era o motivo da decisao.** A minha opcao 3 estava ERRADA, nao certa:
`EMBEDDINGS.md:103-104` diz que a janela do next explicitamente nao e uma sequencia de POIs. E eu **nao
posso** escrever que o canal temporal e agregado ao lugar, que e o texto que a sua premissa produziria —
e a unica afirmacao que o codigo refuta. O que o Capitulo 2 pode dizer, e continuar verdadeiro sob as
tres possibilidades, e que o Capitulo 4 troca o vetor monolitico por uma representacao DECOMPOSTA cujos
componentes sao aprendidos por encoders separados, sem afirmar um nivel unico para o resultado.

> **DECISAO SUA — qual das tres possibilidades vale, e o que fazer com o `:93`.** As opcoes nao sao
> simetricas em custo:
> 1. **Voce confirma (1), que houve agregacao na rodada publicada.** Entao o `:93` esta certo, o `:153`
>    esta incompleto, e o Capitulo 2 pode dizer "nivel de lugar" nomeando a agregacao. Precisa do
>    artefato ou do commit que a fazia.
> 2. **Voce confirma (2), a categoria rodou sem o canal temporal.** Entao o `:93` superdeclara e vira
>    linha de errata; os numeros publicados continuam validos, so a descricao da entrada muda.
> 3. **Voce confirma (3).** Custo maior: os numeros de categoria descrevem uma entrada diferente da
>    declarada, e a errata tem de dizer isso.
> 4. **Deixar em aberto com um `[VERIFY]` no Capitulo 4** e o Capitulo 2 usando a redacao neutra
>    ("representacao decomposta", sem nivel). Nao resolve, mas nao arrisca uma afirmacao falsa, e mantem
>    o Capitulo 2 destravado enquanto voce busca o artefato.
>
> **DECISAO SUA:** ______

### 6.16 Rodada 12 — AD-5 e AD-6 fechadas, e o AD-4 virou uma questao maior do que a que eu fiz

**AD-5 — FECHADA.** As duas linhas de registro do §6.13 estao autorizadas: $\mathbf{e}_{x_i}$ e
$f_{\mathrm{place}}(H_i)$. E com treze definicoes a linha do $\rho$ deixa de ser opcional e passa a ser
**necessaria**, porque o $\rho$ agora e um objeto numerado do capitulo e nao mais notacao de prosa.

**AD-6 — FECHADA: afiar.** A frase e `2_fundamentals.tex:85`, dentro da D2: "A target label is withheld
from it when one of the sequential tasks is trained." Antes de reescrever, medi o que ela precisa dizer, e
ha um ponto que muda o objetivo do conserto:

- **Qual rotulo depende da tarefa**, e isso e a razao de ela estar vaga. A Definicao 2.1 da
  $x_i=(u,p_i,t_i,c_i,r_i)$, entao o alvo e $c_i$ para next-category e $r_i$ para next-region. Uma frase
  so nao pode fixar um campo, e afiar sem partir em duas exige nomear o alvo **como funcao da tarefa**.
- **O que e retido nao e um campo da HISTORIA.** O $H_i=(x_{i-\ell},\ldots,x_{i-1})$ ja **exclui** o
  $x_i$ por construcao, e e no $x_i$ que o alvo vive. Os elementos da historia mantem categoria e regiao.
- **E aqui esta o defeito real da frase atual:** como esta escrita, ela convida a leitura oposta, de que
  as categorias sao apagadas da historia, o que seria uma descricao FALSA dos dados. O proprio MobiWac diz
  o contrario em `5_mobiwac/05_setup.tex:76`: "a per-visit vector legitimately carries more than the
  previous category, including the place, its neighborhood, and the hour of the visit". Ou seja, afiar
  nao e cosmetico: e trocar uma frase que pode ser lida errado por uma que diz de onde o alvo sai e o que
  **permanece** disponivel.

---

**AD-4 — a sua resposta e maior do que a minha pergunta, e o seu argumento e melhor do que o do projeto.**

Eu perguntei so o **titulo** de uma subsubsecao dentro da §2.1. Voce respondeu propondo **inverter a
ordem**: representacao primeiro, com a definicao formal ali, e depois as tarefas que se pode treinar com
essas representacoes. E voce mesmo pos a ressalva: "a narrativa tem que esta acima desse problema, temos
que ter muito cuidade".

**O argumento e melhor.** O projeto move duas definicoes para DENTRO da secao de tarefas para tirar a
dependencia para frente. Voce a elimina na RAIZ: se a representacao vem primeiro, nada precisa se mover,
porque as tarefas passam a consumir objetos que o leitor ja tem. E a ordem de leitura passa a espelhar a
propria tese, que e a de que a representacao e o fator dominante.

Como voce pediu cuidado, o custo, medido:

| custo | o que e | tamanho |
|---|---|---|
| referencias cruzadas | so **tres** referencias vivas aos dois labels, **todas dentro do proprio `2_fundamentals.tex`** (`sec:fund:tasks` x1, `sec:fund:repr` x2). Nada fora do capitulo aponta para eles. | pequeno |
| prosa que fica FALSA | `:14-20`, a abertura do capitulo, enuncia a ordem em prosa com os refs ("It **first** distinguishes the POI prediction tasks... It **then** traces mobility representations"); e `:27`, a abertura da §2.1, e a **justificativa explicita** da ordem atual ("This section defines the prediction targets **before** reviewing the methods used for them"). Essa segunda nao fica so obsoleta: fica uma afirmacao falsa sobre a estrutura do proprio capitulo. | duas reescritas |
| **estrutura do capitulo** | a ordem 2.1-2.5 esta **fixada no mapa do capitulo**, `NORTH_STAR.md:73-80`, e o escopo do projeto diz para nao inventar secoes. Inverter 2.1 e 2.2 **edita o mapa**, e e outra ordem de magnitude em relacao a nomear uma subsubsecao. | **e a razao de a decisao ser sua** |
| gates | **nenhum.** Medi todos os `check_*.py` e o `check.sh`: os quatro arquivos que mencionam "2.1" falam de itens do PENDENCIAS ou de fixtures de teste, nao do capitulo; e o `R11-fab15` pina uma frase da INTRODUCAO, nao a ordem. | zero |
| efeito colateral | `fundamentals/` tem diretorios `2.1_poi_prediction_tasks` e `2.2_representations_for_mobility`. A pasta e provenance congelada e os nomes registram como o capitulo **foi** construido, entao nao e erro, mas passa a descrever outra ordem. | anotar |

> **DECISAO SUA — e eu separei em duas porque sao decisoes diferentes de tamanhos diferentes.**
>
> **(a) O AD-4 como eu perguntei.** Uma subsubsecao dentro da §2.1 com o titulo que voce der, a ordem das
> secoes intacta, e o plano de 8 passos vale como esta. Resolve a dependencia para frente e nao toca o
> mapa. Custo: as duas definicoes de representacao passam a viver na secao de tarefas, que e exatamente o
> arranjo que o seu argumento diz ser menos natural.
>
> **(b) A inversao 2.1 <-> 2.2 que voce propos.** Item NOVO, maior, e uma mudanca de estrutura de
> capitulo. Ganha a ordem de leitura que espelha a tese e nada precisa se mover para dentro de secao
> alheia. Custo: reescrever `:14-20`, reescrever `:27`, e uma decisao sua sobre alterar o mapa do
> `NORTH_STAR`. As tres referencias cruzadas sao triviais.
>
> Se escolher (b), eu **nao** executo junto com o resto do round sem essa linha explicita: e mudanca de
> estrutura, e o plano de 8 passos foi escrito para (a).
>
> **DECISAO SUA:** ______

**Fora de escopo, e eu tinha listado indevidamente:** o `src_utils/NEEDS_SIGN_OFF.md` (56 marcadores). Voce
foi explicito de que este round e sobre as definitions. Retirado daqui; segue em aberto no lugar dele.

**AD-2 continua pre-requisito e nao paralela.** Voce pediu o estudo antes da resposta concreta, e ele esta
feito: `_round12/50_courb_temporal_level_investigation.md`. O resultado inverte a premissa (nao existe
agregacao; o construtor da tarefa de categoria **recusa** o canal temporal) e deixa tres possibilidades
que so os artefatos da rodada publicada distinguem. O §6.15 as lista com o custo de cada uma.

### 6.17 Rodada 12 — AD-4 fechada sob condicao, e um erro meu de relato que voce pegou

**PRIMEIRO, O MEU ERRO, porque voce o pegou e eu nao.** Foi relatado a voce que o
`fundamentals/DEFINITIONS.md` nunca chegou a ser escrito, porque o kernel reiniciou duas vezes durante a
aplicacao das seis correcoes. Voce respondeu: "Voce esta equivocado: articles/dissertacao/fundamentals/
DEFINITIONS.md, acredito ser esse arquivo." **Voce estava certo.** Conferido em disco e no git: 43.509
bytes, 560 linhas, commitado e limpo (`6e43663b`, depois `ae843c78`), com as seis correcoes presentes na
copia commitada — `with TWO` (F-1) x1, `the trained vector of its region node` x2, `SEQUENTIAL tasks` x2,
`hold the task DEFINITIONS fixed` x3, `Chapter 5 DROPS the padded windows` x1, `the letter $d$ is NOT
unclaimed` x1. O relato descrevia o estado de ANTES da escrita e nao foi reconferido depois dela. A licao e
a mesma que o `_round9/34` ja carrega em outras formas: **um estado que eu observei uma vez nao e um estado
que eu observo agora**, e "o kernel reiniciou" descreve o meu processo, nao o disco.

**AD-4 — FECHADA SOB CONDICAO, e a condicao pode cancelar a subsecao.** Suas palavras: "vamos de `Cheking
and place representation`, but as I said we need to create a better plan for it, maybe with this
inveration we even need this new section."

O titulo fica **"Check-in and place representation"**. Voce escreveu "Cheking"; o `GLOSSARY` registra
**check-in** (14 ocorrencias, "never 'event'") e `checking` aparece **zero** vezes, entao a grafia canonica
e essa. **Um ponto de registro que eu prefiro te mostrar a substituir em silencio:** `place embedding`
esta registrado e `place representation` **nao** esta (zero ocorrencias). A segunda metade do seu titulo e
portanto uma variante nao registrada de um termo que existe. Tres saidas:
1. **Manter "Check-in and place representation"** e registrar `place representation` no §1.1 — a linha e
   sua, e o termo passa a ser canonico. Le melhor e e o que voce pediu.
2. **"Place embedding and check-in-level representation"** — as duas metades ja registradas, e sao
   literalmente os titulos das duas definicoes que a subsecao hospedaria. Mais longo, e e exatamente a
   cabeca dupla que a rodada 11 **separou** por juntar dois conceitos sob um nome.
3. **Manter o titulo e nao registrar nada** — funciona na pratica, mas deixa a regra fail-closed esticada,
   que e o mesmo estado que produziu a divida do `soft parameter sharing` e do `negative transfer`.

> **DECISAO SUA:** ______

**Mas a sua propria condicao e a parte que manda:** "maybe with this inversion we even need this new
section". Se a §2.2 vier primeiro, as definicoes de representacao **ja estao na secao certa** e um
compartimento dentro da secao de tarefas fica sem funcao. Entao o titulo e uma resposta **condicional** e
nao uma autorizacao para criar a subsecao. Nao criei nada.

**AD-5 e AD-6 — fechadas, registradas no §6.16.** Com treze definicoes a linha do $\rho$ passa de opcional
a **necessaria**.

**AD-2 — continua aberta, e o estudo que voce pediu esta feito** (`_round12/50`). Ele **inverte a
premissa**: nao existe agregacao de check-in para POI no pipeline, e o construtor da entrada da tarefa de
categoria **recusa** o canal temporal (`builders.py:191-192`). O `§6.15` lista as tres possibilidades que
sobram, com o custo de cada, e todas as tres precisam dos artefatos da rodada publicada — nao da arvore de
fontes — para serem distinguidas.

**O ITEM MAIOR CONTINUA SEM DECISAO: a inversao §2.1 <-> §2.2.** Nao e o AD-4 e nao deve ser fechada como
se fosse. O custo esta medido no `§6.16` e agora tambem no `fundamentals/DEFINITIONS.md` §11: tres
referencias cruzadas (todas dentro do capitulo), duas passagens de prosa que ficam **falsas** (`:14-20` e
`:27`), o mapa fixado do capitulo em `NORTH_STAR.md:73-80`, e **zero** gates. E a consequencia que importa
para o cronograma: **o plano de 8 passos foi escrito para a outra forma e nao sobrevive a inversao.** Ele
tem de ser refeito ANTES de qualquer edicao, nao ajustado depois.

Sua ordem de trabalho esta registrada e sera respeitada: "depois dessas definicoes podemos aplicado."
Nenhuma edicao no `.tex` foi feita.

### 6.18 Rodada 12 — a AD-2 esta RESPONDIDA pelo codigo original, e a resposta e uma quarta possibilidade

Voce disse: "Calma analise o codigo original do chap. 4: /Users/vitor/Desktop/mestrado/temp/tarik-new,
antes de decidirmos." **Era exatamente a fonte que faltava**, e ela fecha o que o `_round12/50` dizia que
so os artefatos da rodada publicada poderiam fechar. **Nenhuma das tres possibilidades que eu listei era a
certa.**

**1. O embedding temporal e por CHECK-IN, e a prova esta nos outputs guardados do proprio notebook, nao na
leitura da intencao do codigo.** `Time_Encoder.ipynb`, California:

| celula | output guardado | o que significa |
|---|---|---|
| 2 | `N checkins (antes de filtrar): 2535573` | a entrada e a tabela de check-ins |
| 3 | `(2535573, 2)` | duas features por check-in (`t_hour = hora/24`, `t_dow = dia/7`) |
| 13 | `time_embeds_sin shape: (2535573, 64)` | **uma linha de 64 dimensoes por check-in** |

Como as features sao hora e dia da semana **por check-in**, duas visitas ao mesmo POI em horarios
diferentes produzem vetores **diferentes**. A celula 14 monta o frame indexado por `placeid` e grava num
caminho cujo nome ja diz o nivel, `time_encoder_embeddings_sin_CHECKIN_{estado}.csv`; a 15 copia para
`data/output/{estado}/time_embedding_novo.csv`, que e o que o ETL le.

**2. E a entrada da tarefa de categoria reduz de check-in para POI DESCARTANDO LINHAS, nao agregando.**
`PoiMtlNet_Novo/src/etl/create_inputs_hgi.py:437`, literal:

    time_df = time_emb[["placeid"] + num_cols_time].drop_duplicates("placeid")

Com uma linha por check-in indexada por `placeid`, o `drop_duplicates("placeid")` **mantem a primeira
ocorrencia de cada POI e joga as outras visitas fora.** Os tres componentes sao then unidos por `placeid`
(:441-443) e a categoria anexada por `placeid` (:448), o que produz os pares $(\mathbf{E}_{cat}, c)$ do
`methodology.tex:93`. O default do `process_state` e `cat_embeddings=("poi","loc","time")`, entao o canal
temporal **esta** na entrada da categoria.

**A RESPONSTA, e por que ela nao estava entre as tres.** Existe uma reducao de check-in para POI, e ela e
`drop_duplicates`, **nao uma agregacao**. O seu instinto de que algo converte o nivel estava certo; a
operacao nao e media nem pooling, ela **seleciona uma visita arbitraria por POI**. Ou seja: o canal
temporal que chega a tarefa de categoria carrega o timestamp de **uma** visita, nao um resumo das visitas
daquele POI, e a variacao que o encoder foi construido para capturar e descartada nessa tarefa.

| possibilidade que eu havia listado | veredito |
|---|---|
| (1) agregacao removida depois | **a mais proxima, e errada na palavra que importa.** A reducao existe e **continua** no codigo original. Nao e agregacao. |
| (2) a categoria rodou sem o canal temporal | **refutada.** O `cat_embeddings` inclui `time` por default. |
| (3) vetores de nivel check-in na tarefa de categoria | **refutada para a categoria.** Sao reduzidos a um por `placeid` antes. |

**O QUE ISSO DIZ SOBRE O `:93` E O `:153`, e a conclusao e mais branda do que eu temia.** As duas frases
sao **individualmente corretas**: o `:153` esta certo de que o encoder produz um vetor por check-in, e o
`:93` esta certo de que um vetor de 192 dimensoes em nivel de POI e pareado com a categoria do POI. **O que
o texto publicado nunca enuncia e o passo entre as duas**, e esse passo perde informacao de um jeito que um
leitor gostaria de saber: uma visita por POI sobrevive e as demais sao descartadas.

Isso e **lacuna de descricao, nao numero errado** — os pares com que a tarefa de categoria treinou sao
exatamente os que o `:93` descreve. Se o capitulo deve registrar o passo de selecao e materia de **errata**
sob o `NORTH_STAR` §5.7, e o `apx_b_errata.tex` nao a carrega hoje. **O Capitulo 4 nao foi editado.**

> **DECISAO SUA — o que fazer com a lacuna:**
> 1. **Linha de errata** registrando que o canal temporal e reduzido a uma visita por POI por selecao da
>    primeira ocorrencia. Custo: uma linha na tabela de errata do Cap. 4. Ganho: a descricao passa a
>    corresponder ao que rodou, e ninguem reproduzindo o trabalho se surpreende.
> 2. **Nao registrar**, tratando como detalhe de implementacao. Nenhum numero muda e o `:93` continua
>    verdadeiro. Risco: um leitor atento que rode o codigo encontra o `drop_duplicates` e conclui que o
>    texto o omitiu.
> 3. **Registrar so no Capitulo 2**, na frase que descreve a entrada do Cap. 4, sem tocar a errata.
>
> **DECISAO SUA:** ______

**O que o Capitulo 2 pode dizer agora, e ja esta corrigido no `fundamentals/DEFINITIONS.md` §3:** a
instanciacao do Cap. 4 dizia "de uma funcao do POI visitado e, no Cap. 4, do timestamp da visita". Isso
agora esta provado **impreciso**: o timestamp e de **uma visita selecionada**, nao da visita naquela
posicao da janela. As duas redacoes erradas estao nomeadas no arquivo: "do timestamp da visita" e
"agregado".
### 6.19 Um item que nunca chegou a voce: a sobrecarga de indices na D13 (AD-7)

Estava no §9 do `DEFINITIONS.md` como item 4 e nunca virou decisao. Na definicao de conflito de gradiente,
o $\mathbf{g}_i$, o $\mathbf{g}_j$ e o $\varphi_{ij}$ indexam **tarefas**; em todo o resto do capitulo o
indice $i$ indexa **check-ins** ($x_i$, $H_i$, $\mathbf{e}_{x_i}$). Conferido no bloco vivo.

Um leitor que acompanha os indices ao longo do capitulo tropeca nisso. **Escopo do probe, dito com
precisao em vez de superdimensionado:** o `R10-cosine` pina a string `def:fund:conflict`, que e o **label**,
entao uma renomeacao de indices nao o quebraria; o que uma renomeacao arrisca e a prosa ao redor e o
apendice que aponta de volta para essa definicao.

> **DECISAO SUA:**
> 1. **Renomear os indices da D13** para algo como $a$ e $b$ (tarefas), deixando o $i$ livre para
>    check-ins em todo o capitulo. Custo: a equacao `eq:fund:cosine`, a prosa ao redor, e conferir o
>    apendice que referencia a definicao.
> 2. **Deixar como esta e adicionar meia frase** dizendo que ali os indices sao de tarefa. Custo minimo,
>    resolve para o leitor atento, mantem a notacao da fonte (`yu2020pcgrad` usa $i$ e $j$ para tarefas).
> 3. **Nao fazer nada.** E convencao da literatura e ninguem reclamou.
>
> **DECISAO SUA:** ______

### 6.20 Rodada 12 — as suas quatro decisoes, e o que cada uma NAO autoriza

**ITEM 1 — A INVERSAO §2.1 <-> §2.2: voce autorizou o ESTUDO, nao a mudanca.** "Use um fable 5 para
explorar e validar a inversao, vale ler o articles/dissertacao/fundamentals, onde planejamos o fundamentos
no principio." Um agente `claude-fable-5` esta rodando com esse escopo e com escrita limitada a **um**
arquivo novo (`_round12/52_inversion_study.md`); ele nao pode tocar `.tex`, `GLOSSARY`, `NORTH_STAR` nem
este arquivo.

**A sua escolha de fonte foi boa e vale dizer por que.** A pasta `fundamentals/` e o registro **congelado**
de como o Cap. 2 foi planejado, e os diretorios dela sao literalmente `2.1_poi_prediction_tasks`,
`2.2_representations_for_mobility`, e assim por diante — ou seja, ela carrega o raciocinio original da ordem
que a inversao quer trocar. **Uma medicao previa, que o agente vai confirmar ou refutar:** varri a pasta por
palavras de ordem ("order", "before review", "first distinguish", "sequence") e **nenhum arquivo argumenta a
ordem**. O `fundamentals.tex` apenas monta 2.1 a 2.5 na sequencia. Se isso se confirmar, o achado e que a
ordem foi **herdada do escopo do projeto e nunca defendida**, o que enfraquece o argumento de tradicao
contra a inversao. Nao vou tratar isso como fechado antes do relatorio.

**ITEM 2 — AD-7, OPCAO 1: RENOMEAR OS INDICES DA D13.** Feito **no desenho**, nao no capitulo. O
$\mathbf{g}_i,\mathbf{g}_j,\varphi_{ij}$ passam a $\mathbf{g}_a,\mathbf{g}_b,\varphi_{ab}$, liberando o $i$
para check-ins em todo o capitulo.

Duas medicoes que corrigem o que eu havia dito antes:
- **O escopo e menor do que a minha nota de custo sugeria.** Os simbolos aparecem em **seis linhas vivas**,
  todas dentro do bloco da D13 (`2_fundamentals.tex:888-895`), e **em nenhum outro lugar da arvore**. O
  apendice do cosseno menciona o conceito num comentario e nunca o label nem os simbolos, entao a minha
  frase de que a renomeacao "exige conferir o apendice" era mais larga do que a evidencia.
- **Os dois probes sobrevivem, testado e nao raciocinado.** Apliquei a renomeacao ao texto real do capitulo
  e rodei os padroes: o `R9-conflict` casa `cosine between their gradients` (prosa, sem simbolos) e o
  `R10-cosine` casa o label `def:fund:conflict`. Ambos valem antes e depois.

**Por que no desenho e nao no capitulo agora:** o §5 do `DEFINITIONS.md` especifica a D13 e o redesenho
ainda esta pendente. Editar o capitulo hoje poria o capitulo e o desenho fora de sincronia justamente na
janela em que o redesenho espera decisao. Entra junto com o redesenho.

**ITEM 3 — ERRATA DO AD-2, OPCAO 2: NAO REGISTRAR.** A reducao check-in para POI por
`drop_duplicates("placeid")` fica como detalhe de implementacao. Nenhum numero muda, o
`methodology.tex:93` continua verdadeiro, o `apx_b_errata.tex` nao recebe linha, e o Capitulo 4 segue
intocado como versao de registro.

> **UM PONTO QUE A SUA DECISAO NAO COBRE, e que eu registrei no desenho para quem for aplicar:**
> "nao registrar" e **silencio**, nao licenca para escrever a coisa errada. A instanciacao do Cap. 4 no
> `DEFINITIONS.md` §3 dizia "uma funcao do POI visitado e, no Cap. 4, do timestamp da visita", e isso esta
> **provado impreciso**: o timestamp e de **uma visita selecionada** (a primeira por `placeid`), nao da
> visita naquela posicao da janela. Sob a opcao 2 o Cap. 2 escreve a forma **neutra** — "uma funcao do POI
> visitado", sem qualificacao temporal — e as duas redacoes erradas continuam **proibidas**: "do timestamp
> da visita" (falsa) e "agregado" (a operacao seleciona, nao combina). Os probes `R12-dropdup`,
> `R12-shape`, `R12-notwrong` e `R12-notagg2` guardam isso.

**ITEM 4 — `place representation` ENTRA NO REGISTRO.** Linha adicionada ao `GLOSSARY.md` §6 ao lado de
`place embedding`. Com isso o titulo do AD-4 fica **"Check-in and place representation"** com as duas
metades registradas (a grafia canonica e `check-in`; `checking` nao existe no registro).

**Mas registrei com uma nota de escopo, porque duas linhas quase sinonimas colidem com o §5 do
`WRITING_LAW`** ("One name per concept for the whole document; synonym-cycling is both imprecise and an AI
tell"). A nota admite `place representation` em **titulos e cabecalhos**, onde faz paralelo com
`check-in-level representation`, e mantem `place embedding` como o nome canonico **na prosa corrida**. Sem
isso, a linha nova licenciaria exatamente a rotacao que a lei proibe. Se voce preferir sem a nota, e uma
linha sua.

**E isto nao cria a subsubsecao.** O AD-4 segue **condicional**: "maybe with this inversion we even need
this new section". O item 1 acima e o que decide se ela existe.

**AINDA DEVIDO A VOCE, e eu nao pedi antes:** a linha de registro do $\rho$ (que virou **necessaria** quando
voce escolheu treze definicoes) e a nota de escopo do $d$ (o $d$ nao esta livre: $d_{ij}$ e distancia
geodesica no Cap. 3 e $d_{\mathrm{shared}}$ e largura de tronco no Cap. 5). Ambas sao linhas do `GLOSSARY`
§1.1, e a tabela e sua.

**Nenhum `.tex` foi editado.** A sua ordem segue valendo: "depois dessas definicoes podemos aplicado", e o
plano de 8 passos do `_round12/49` continua sem valer enquanto a inversao nao for decidida.

### 6.21 Rodada 12 — RETRATACAO: a AD-2 nao esta respondida, e eu a fechei sobre um elo que nao verifiquei

**Retiro a resposta que dei ao §6.18.** Um revisor achou o defeito e ele e real. A conclusao dependia de dois
arquivos serem o mesmo arquivo, e eu nunca conferi isso.

| o que | onde | nome e formato |
|---|---|---|
| o que o ETL LE | `PoiMtlNet_Novo/src/etl/create_inputs_hgi.py:415` | `{OUTPUT_DIR}/{state}/time_embedding.**parquet**` |
| o que o notebook ESCREVE | `Time_Encoder.ipynb`, celula 15 | `.../{estado}/time_embedding_**novo**.**csv**` |
| o que outras duas linhas do notebook citam | `Time_Encoder.ipynb:1714,:1741` | `.../alabama/time_embedding.**csv**` |

Eu escrevi "which is what the ETL reads" e segui em frente. **Sao tres nomes e dois formatos.**

**E a lacuna e maior do que o nome do arquivo.** Remedido depois do achado: **nada naquele repositorio
escreve o `time_embedding.parquet`** (procurei em todo `.py` sob `PoiMtlNet_Novo/`; a unica ocorrencia e a
leitura na :415), **nao existe conversao csv para parquet** em `src/etl/` nem em `pipelines/`, o arquivo
**nao esta em disco**, e o proprio `CLAUDE.md:91` daquele repositorio descreve este ETL lendo um `.csv`,
discordando do proprio codigo. **Quem produz a tabela que o ETL consome esta fora daquele repositorio, e a
granularidade dela e desconhecida para mim.**

**O que isso derruba, dito por inteiro e nao minimizado.** Se o `time_embedding.parquet` ja for de nivel de
POI, o `drop_duplicates("placeid")` e um dedup inofensivo e **nao existe passo de selecao nenhum**. Cai com
o elo: a moldura da "quarta possibilidade", o "mantem a primeira visita e descarta as outras", e toda a
analise de que o `methodology.tex:93` e o `:153` sao individualmente corretos com um passo perdido entre
eles, que era a base do argumento de lacuna de descricao e de errata.

**O que continua valendo, medido e intacto:** o encoder emite **uma linha por check-in** (celula 13, output
guardado `(2535573, 64)`, contra 2.535.573 check-ins da celula 2, com features hora e dia por check-in na
celula 3); o dedup por `placeid` esta no caminho da tarefa de categoria (`:437`); e o canal temporal esta
naquela entrada (`cat_embeddings` inclui `time`). O ETL portanto **espera** uma tabela que precisa reduzir
por `placeid` — o que e **sugestivo** e nao prova: um dedup defensivo contra uma tabela de POI com linhas
repetidas e igualmente compativel com o codigo.

**O QUE FECHA A AD-2 E UM ARTEFATO SO:** o `data/output/{state}/time_embedding.parquet` da rodada do CoUrb,
ou o que o produziu. Um `len(df)` contra a contagem de POIs e de check-ins daquele estado decide:
`N_checkins` linhas e o passo de selecao e real; `N_pois` linhas e o dedup nao faz nada e a descricao
publicada nao tem lacuna. Marcado como `[VERIFY]` no `_round12/50`.

> **DECISAO SUA — nada precisa mudar no texto, mas voce pode fechar isto se tiver o artefato:**
> 1. **Voce localiza o `time_embedding.parquet` daquela rodada** (ou o notebook/script que o gerou) e eu
>    meco a granularidade. Fecha a AD-2 de vez.
> 2. **Deixar em `[VERIFY]`.** Custo zero para o texto: a sua propria decisao de nao registrar mantem o
>    Cap. 2 na forma neutra, e agora com razao mais forte — nao "existe um passo que escolhemos nao
>    mencionar", e sim "o nivel nao esta estabelecido, entao o capitulo nao afirma nada sobre ele".
>
> **DECISAO SUA:** ______

**O erro, nomeado, porque e o segundo da mesma familia neste projeto.** Os dois extremos eram reais e
verificados por mim (a forma `(2535573, 64)` e o dedup na `:437`). Tendo verificado os dois, tratei o
caminho entre eles como verificado tambem. **Uma corrente nao esta verificada quando os elos dela estao.** E
a mesma forma do postmortem fabricado que o `_round9/34` registra: la eu inventei tres mecanismos para
explicar uma falha real, aqui inventei uma ligacao para juntar dois fatos reais. Nas duas vezes as
observacoes eram boas e a **relacao** foi suprida pela expectativa. O teste que faltava era um `grep` pelo
escritor daquele nome de arquivo.

**Nenhum texto do capitulo esta errado**, e um probe que pinava a conclusao retratada (`R12-notwrong`) ficou
vermelho quando eu corrigi o registro — um probe sobre uma conclusao retratada e pior que probe nenhum,
porque briga com a correcao. Substituido por probes sobre a **retratacao** (`R12-retract`, `R12-retract2`,
`R12-verify`).

### 6.22 Rodada 12 — AD-2 encerrada como `[VERIFY]`, e o `place representation` sai do registro

**ITEM 1 — AD-2, OPCAO B: FICA EM `[VERIFY]` E O ASSUNTO ESTA ENCERRADO.** "Vamos de B, e matamos esse
assunto... eu nao tenho o `time_embedding.parquet`." Registrado como **LO-12** no `LEFT_OUT.md`, e o Cap. 2
segue na forma neutra ("a vector that is a function of the visited POI"), com "do timestamp da visita" e
"agregado" ainda proibidas. A retratacao do §6.21 e o **estado final** desta decisao, nao uma pendencia.

**E EU NAO RODEI A REGENERACAO, embora voce tenha autorizado** ("se quiser rodar para saber o resultado e
documentar, pode"). A razao e exatamente o defeito que acabei de retratar: regenerar o embedding a partir dos
check-ins mediria o **codigo de hoje**, nao a rodada publicada, e produziria um numero com cara de resposta
para uma pergunta sobre outro objeto. Seria fabricar o mesmo tipo de evidencia que eu retirei, um passo mais
adiante. Alem disso nada naquele repositorio escreve o parquet, entao nao existe pipeline para re-rodar: eu
inventaria um e reportaria o comportamento dele como historia.

**O que eu MEDI, e esta no LO-12 como condicional explicito:** quantas visitas por POI o dedup descartaria
**se** a tabela fosse de nivel de check-in. Dos `checkins_by_state`: Alabama 113.846 check-ins / 11.848 POIs
= 9,61x; Arizona 11,44x; Georgia 13,57x; Florida 18,38x; Texas 4.089.892 / 160.938 = 25,41x. **Isso
quantifica uma hipotese, nao um fato** — e a entrada diz isso com essas palavras. As contagens de check-in
conferem com as que a dissertacao ja publica para Alabama e Texas, o que serve de conferencia cruzada.

**Uma nota de admissibilidade que vale registrar.** A regra do proprio `LEFT_OUT.md` (l. 9-13) exige que o
achado esteja **estabelecido**, e a minha primeira formulacao ("o canal temporal e reduzido a uma visita por
POI") **nao esta** — e a alegacao que retratei. Uma entrada assim importaria a alegacao retratada para o
registro, que e o pior lugar possivel para ela. Entao o **assunto** da LO-12 e a **tensao nao resolvida**
entre o `:93` e o `:153`, que **esta** estabelecida, e nao um passo de selecao. A distincao e o que faz a
entrada passar a propria regra do arquivo.

**Uma pista adjacente que NAO decide, registrada para ninguem promove-la a prova.** O
`apx_b_errata.tex:190-191` diz, recuperado do codigo liberado, "that the sample unit of the category task is
the place, so no place spans two folds". Conferido: e a errata do **Artigo 1** (CBIC), cujo estudo usa o
embedding de grafo e nao o encoder temporal, e "uma linha por place na amostragem" e compativel **tanto** com
o dedup produzindo isso **quanto** com a tabela ja vir POI-level. E a mesma ambiguidade, nao a resolucao
dela.

---

**ITEM 2 — `place representation` SAI DO REGISTRO. SO `place embedding`.** "vamos usar so place embedding
para evitar conflitos e interpretacoes dubias." Feito: a linha e a nota de escopo sairam do `GLOSSARY.md` §6,
e a tabela PT voltou a ser contigua (a linha nova a partia em duas).

**E o probe saiu no MESMO commit.** O `R12-placerep` pinava justamente aquela nota de escopo, entao ficaria
vermelho **por uma decisao sua** — que e o defeito do `R12-notwrong`: um gate brigando com a correcao em vez
de protege-la. Removido, com o motivo no lugar dele. **Nao pus substituto, e vale dizer por que:** banir o
termo da arvore falharia sozinho, porque este arquivo, o `check_audit_claims.py` e os registros da rodada 12
discutem o termo pelo nome ao registrar a decisao, e um probe de ausencia dispararia no **registro da
decisao**. E ao escrever isso eu afirmei que existia um `check_glossary_terms` cuidando disso: **nao existe**.
Os checkers sao os catorze `check_*.py` do `src_utils/`, e so este e o `check_verify_list.py` leem o
`GLOSSARY`. Entao a verdade e que revogar a linha deixa o termo **sem gate**, e isso esta escrito no arquivo
como lacuna conhecida em vez de coberta em outro lugar.

**O TITULO DO AD-4 VOLTOU A FICAR ABERTO, e eu nao vou gastar uma decisao sua nele agora.** Voce deu
"Check-in and place representation"; com o termo fora do registro, a segunda metade deixa de ser admissivel
pela regra fail-closed. Mas o AD-4 e **condicional** de qualquer modo: se a inversao §2.1 <-> §2.2 avancar, a
subsubsecao pode nao existir e o titulo fica irrelevante. Os candidatos, para quando forem necessarios, estao
na linha do AD-4 no `DEFINITIONS.md` §10: (a) "Check-in and place embedding", as duas metades registradas mas
nao a sua redacao; (b) "Place embedding and check-in-level representation", literalmente os titulos das duas
definicoes, mas e a cabeca dupla que a rodada 11 separou de proposito. **Nada foi criado.**

**ITEM 3 — a inversao espera voce.** O estudo do agente esta rodando com escrita limitada a
`_round12/52_inversion_study.md`, apontado para o `fundamentals/` como voce pediu. Nao ha nada seu pendente
nisso ate o relatorio existir.

**CONTINUAM DEVIDAS, e voce nao as tratou nesta rodada:** as duas linhas do `GLOSSARY` §1.1 — a do $\rho$
(que virou **necessaria** quando voce escolheu treze definicoes, porque o $\rho$ passou a ser objeto numerado
do capitulo) e a nota de escopo do $d$ (o $d$ nao esta livre: $d_{ij}$ e distancia geodesica no Cap. 3 e
$d_{\mathrm{shared}}$ e largura de tronco no Cap. 5). Estao no §6.20.

### 6.23 `make extra` esta VERMELHO, e nao e o documento — e o `sed` do proprio script de build

**Eu reportei "all four builds rc=0" nos commits `d36da8c5` e `e1fd3619` e isso estava ERRADO.** O
`extra` retorna 2. Descobri porque a ferramenta de shell comecou a devolver saida vazia com rc nao-zero,
me obrigou a rodar tudo por outro caminho, e ai o `extra` apareceu vermelho. **As duas ferramentas
estavam degradadas ao mesmo tempo**, que e exatamente quando um auto-relato vale menos.

**NAO E CULPA DAS MINHAS EDICOES — mas a primeira evidencia que eu dei disso estava ERRADA e um revisor
pegou.** Eu escrevi que o `git diff 1117b3f9..HEAD -- src/` listava "nenhum arquivo". Aquele comando rodou com
o `cwd` na pasta `dissertacao/` e um pathspec comecando por `articles/dissertacao/`, entao ele procurou em
`articles/dissertacao/articles/dissertacao/`, que nao existe. **Um resultado vazio de um caminho inexistente
nao mede nada**, e eu apresentei isso como "medido e nao suposto".

**REMEDIDO da raiz do repositorio, e o numero muda:** os dois commits **tocam sim** um arquivo sob `src/` —
o `src/dissertacao.pdf`, em ambos. **A conclusao sobrevive, por evidencia diferente:** o `dissertacao.pdf` e
uma **saida** de build, copiada pelo `Makefile:36` (`cp build/main.pdf dissertacao.pdf`), e nao entra em
compilacao nenhuma; e **nenhum `.tex` foi alterado pelos dois commits** — a lista completa e
`DEFINITIONS.md`, `dissertacao.pdf`, `LEFT_OUT.md`, `PENDENCIAS.md`, `_round12/50`, `_round9/34` e
`check_audit_claims.py`. Nada que o `main_extra` inclua.

**A CAUSA, passo a passo:**
1. `make extra` -> rc=2, **mas o PDF sai**: 26 paginas, 0 erros de TeX.
2. `pdflatex` sozinho -> rc=0. `bibtex` sozinho -> rc=0.
3. `bash -x` no `latexbuild.sh`: o script chega em `PAGES=$(sed ...)` e o **`PAGES` sai vazio**, entao
   entra no ramo "NO PAGE COUNT ... the build did not finish" e sai com erro.
4. Mas o log **tem** a linha: `Output written on ... (26 pages, 200259 bytes).`, uma ocorrencia.
5. Rodando o mesmo `sed` na mao: **`sed: RE error: illegal byte sequence`**. Com `LC_ALL=C` ele imprime
   `26` normalmente.
6. O log **nao e UTF-8 valido**: o byte 61294 e um `\xea` Latin-1 dentro da hifenizacao de
   "In-te-li-g\xean-cia Com-pu-ta-ci-o-nal", de uma entrada da bibliografia do CBIC que so aparece no
   apendice do `extra`.

Ou seja: o TeX escreve no log uma palavra portuguesa hifenizada com um byte Latin-1, e o **`sed` do BSD
(macOS) recusa o arquivo inteiro num locale UTF-8** — o GNU `sed` nao faria isso. O build **funcionou**;
o que falhou foi a extracao da contagem de paginas, e o script honestamente reporta a propria falha.

**O `extra` e o unico alvo cujo log carrega esse byte**, que e por que os outros tres passam.

> **DECISAO SUA — a correcao e de uma linha, mas o arquivo e compartilhado:**
> 1. **`LC_ALL=C` na linha do `PAGES`** (`src_utils/latexbuild.sh:49`). E o minimo, resolve a causa, e o
>    `LC_ALL=C` e a forma padrao de dizer "trate como bytes". Risco: nenhum que eu veja; a regex e ASCII.
> 2. **`LC_ALL=C` na linha do `PAGES` e tambem no `grep` do `ERRS`** (linhas 49-50), por simetria. **Eu
>    levantei a suspeita de que o `grep` estivesse silenciosamente devolvendo 0 e FUI CONFERIR: nao esta.**
>    O `grep -c '^! '` devolve `0` tanto no locale UTF-8 quanto sob `LC_ALL=C` naquele mesmo log, entao a
>    contagem `tex_errors=0` do `extra` e verdadeira. Esta opcao e higiene, nao correcao de defeito.
> 3. **Nao mexer** e aceitar o `extra` vermelho ate a proxima rodada.
>
> **DECISAO SUA:** ______

**Por que eu nao apliquei sozinho:** o `latexbuild.sh` e compartilhado com o agente paralelo, o cabecalho
dele documenta tres propriedades que foram aprendidas quebrando o arquivo, e a opcao 2 muda o
comportamento de uma checagem de erros. Isso e chamada sua, nao limpeza de fim de rodada minha.

**Uma suspeita que eu levantei e depois DERRUBEI com um comando, em vez de deixar no ar:** achei que o
`grep -c '^! '` pudesse estar falhando pelo mesmo byte e devolvendo `0` sem significado. Testei nos dois
locales contra aquele log e ele devolve `0` nos dois. Entao o `tex_errors=0` do `extra` vale, e o unico
defeito real e a contagem de paginas.

### 6.24 A comparacao das duas opcoes — o estudo novo RECOMENDA MANTER A ORDEM, ao contrario do `_round12/52`

O relatorio esta em `_round12/53_order_comparison.md`. **Nada foi decidido nem editado**, e a sua suspensao da
inversao continua valendo.

**A RECOMENDACAO: OPCAO (a) — MANTER `2.1` tarefas, `2.2` representacoes**, e aplicar o redesenho pelo plano do
`_round12/49`. E o oposto do que o `52` recomendou, com praticamente a mesma base de evidencia; a divergencia
esta no criterio que **voce** definiu.

**COMO ELE RESOLVEU O ARGUMENTO DO ESPELHO, que era a sua preocupacao.** Os dois espelhos sao fieis a tese: o
tasks-first espelha o **DESENHO** do argumento (fixe a referencia, varie a alavanca) e o representations-first
espelha a **RESPOSTA** (a alavanca e que importava). O desempate dele nao e a tese, e o **leitor**, com dois
argumentos:
1. **O espelho-da-resposta so paga a quem ja tem a resposta.** A elegancia de "representacoes primeiro, porque
   as representacoes dominaram" e legivel para quem relê ou para voce, que conhece o arco. Um membro de banca
   em primeira leitura ainda nao tem a tese como conviccao; para esse leitor as tarefas sao o enquadramento que
   torna cada definicao posterior inteligivel. **O espelho-do-desenho e andaime para a primeira leitura; o
   espelho-da-resposta e recompensa para a segunda** — e a regra G3 do proprio projeto faz da primeira leitura
   a que governa.
2. **A invariante e o que os tres artigos compartilham, e o Cap. 2 e onde o leitor adquire o quadro
   compartilhado.** O que e constante nos tres e a definicao das tarefas; o que varia e o $\rho$. Um capitulo de
   fundamentos ordenado invariante-primeiro entrega o quadro fixo uma vez e deixa cada capitulo de artigo variar
   a alavanca contra ele.

**E ELE DESMONTOU O ROTULO "ja esta validada" — que era meu, e estava frouxo.** Eu conferi separadamente e
confirmo: **o rotulo e do `52`, nao do `49`.** Um `grep` por "validated" no `49` da **zero** ocorrencias dessa
alegacao. O que o `49` tem e uma **validacao adversarial (Parte A)** das propriedades internas do desenho —
ordem de dependencia, boa formacao, colisoes de notacao, sobrevivencia de probes — e isso e diferente de dizer
que a **opcao** esta validada. O estudo aponta tres qualificadores, e o primeiro e o que importa: **o AD-4, que
e o sign-off na forma da §2.1 e no titulo da subsubsecao, esta ABERTO.** Nas palavras dele, a opcao esta
"mechanically checked and narratively unsigned" — o **AD-4 e o custo narrativo da opcao (a) em forma de
decisao**.

**CINCO PONTOS EM QUE ELE DISCORDA DO `52`**, e eu destaco os dois que me parecem mais fortes:
- **O `52` resolveu o espelho por deferencia, nao por argumento.** Ele concedeu que a escolha era autoral e
  entao a resolveu com "o julgamento do autor ja se inclina para o espelho-da-resposta". **A sua suspensao
  provou essa leitura errada**, e tratar um entusiasmo inicial como veredicto e exatamente o padrao de
  sycophancy que o `AGENT_GUARDRAILS` §7 nomeia.
- **O `52` precificou so um lado.** Mediu a (b) meticulosamente e cotou a (a) como zero, "o fallback". Os
  custos narrativos da (a) nao aparecem lá, e nem as economias dela (nenhuma edicao do mapa, nenhuma ponte
  nova, nenhum rastreador reancorado, nenhum toque na introducao). **Uma comparacao nao se ganha medindo um
  competidor.**
- E o achado historico negativo do `52` **dissolve so o argumento de tradicao**, que ninguem precisava fazer.
  As razoes de presente para tasks-first (a forma dos artigos, o trabalho do leitor, o gate de primeira
  leitura) nao dependem de intencao registrada.

**AS CONDICOES QUE ELE POE NA OPCAO (a)**, e a primeira e desenhada para o seu proprio criterio: **KC-1 — o
AD-4 e resolvido PRIMEIRO**, com voce vendo a forma redigida da §2.1 e nomeando a cabeca. **Se voce rejeitar a
forma ao ler, essa rejeicao e o sinal para reabrir a (b)**, porque o AD-4 e o custo narrativo da (a) tornado
visivel. Depois: uma frase de abertura ligando as representacoes aos alvos; o ajuste da frase de escopo em
`:27-28` no mesmo commit; e as coordenadas rederivadas contra a arvore viva na aplicacao.

**O QUE EU CONFERI, porque auto-relato nao e evidencia:**
- **Coordenadas:** o `GLOSSARY.md:45` (linha do $\rho$), o `DEFINITIONS.md:590` (linha do AD-4), o `:576`, o
  `PENDENCIAS.md:1484`, e o `2_fundamentals.tex:27`, `:68`, `:76` — **todas leem como ele diz**.
- **A contagem de definicoes:** rodei o `grep` dele e as **doze** definicoes estao exatamente nas linhas que
  ele lista, o `wc -l` da **1527** como ele diz (o meu 1528 anterior era erro meu, de contar uma quebra final),
  e as tres ocorrencias de "rho" sao mesmo substrings de "neighborhood"/"node2vec".
- **Ele releu a arvore DEPOIS do meu commit do §6.25:** cota as linhas do `GLOSSARY` como "already paid", o que
  esta certo — elas servem as duas opcoes.
- **Duas derivas pequenas, e as duas sao MINHAS, nao erros dele:** o "ja esta validada" que ele cita no
  `PENDENCIAS:1633` esta hoje na `:1663` (o meu §6.25 empurrou), e o `wc` mudou pelo mesmo motivo.
- **Um numero dele esta desatualizado por minha causa:** ele fala de **21** probes no capitulo; hoje sao **22**,
  porque eu adicionei o `R12-placeterm` nesta rodada. **A alegacao de fundo continua verdadeira, e eu a testei
  em vez de aceitar:** nenhum dos 22 casa com prosa que enuncia a ordem, entao **nenhum probe restringe a
  ordem das secoes**.
- **Wall-clock — CORRIGIDO, e eu tinha escrito o oposto do verdadeiro.** Eu publiquei "2.236 s, 7 por cento
  DENTRO, o primeiro estudo a fechar no prazo". **Os 2.236 s nao sao medicao do estudo: e o instante em que a
  MINHA janela de coleta fechou**, com o relatorio dele ainda em `running` e todos os campos vazios. Li um
  timeout como um termino. **O que esta medido: ele ainda estava processando aos 3.201 s, ou seja 33 por cento
  ACIMA do checkpoint de 2.400 s**, e eu o encerrei nesse ponto. **Ele estourou, como todas as ondas deste
  projeto.** Duas coisas atenuam e nenhuma apaga: o arquivo entregue ja estava **completo em disco** quando eu
  o li (34.525 bytes, todas as secoes), entao o excesso foi depois da escrita e nao trabalho perdido; e o
  auto-relato dele (~1.787 s) esta **subestimado em pelo menos 1.414 s** contra o momento em que eu o parei —
  que e exatamente por que ele mesmo disse que a minha medicao governa, e por que auto-relato de prazo nao
  vale neste projeto.

> **DECISAO SUA — e as duas opcoes estao agora precificadas dos dois lados:**
> 1. **Opcao (a), manter a ordem**, com o KC-1 primeiro: voce le a forma redigida da §2.1 e nomeia a cabeca,
>    e uma rejeicao sua ali reabre a (b). E a recomendacao do estudo novo.
> 2. **Opcao (b), inverter**, com as cinco condicoes do `52`. Se voce escolher esta, **duas perguntas suas
>    voltam**: a ordem da sintese da §2.5 e a edicao do `NORTH_STAR.md:73-80`.
> 3. **Ler os dois relatorios antes de decidir** — o `52` defende a (b), o `53` defende a (a), e eles
>    discordam por escrito e nao por omissao.
>
> **DECISAO SUA:** ______

**Nenhum `.tex` foi tocado, o `NORTH_STAR.md:73-80` esta intacto, e o plano do `49` continua parado.**

### 6.25 O `extra` esta VERDE, e as duas linhas do `GLOSSARY` §1.1 entraram

**ITEM 1 — `LC_ALL=C` nas duas linhas, opcao 2. FEITO, e o `extra` passou a rc=0.** Os quatro alvos agora
saem `104/101/105/26` paginas com `tex_errors=0` em todos, e o `make check` e o `make selftest` em rc=0 —
**seis comandos verdes, lidos um por um**. Era a primeira vez nesta rodada que o `extra` fechava.

**O comentario diz exatamente o que voce pediu, e diz a diferenca entre as duas linhas:**
- a `:PAGES` **e correcao de defeito real** — o `sed` do BSD abortava o arquivo inteiro por causa do byte
  Latin-1 na hifenizacao de "In-te-li-g^en-cia Com-pu-ta-ci-o-nal", entao o `PAGES` saia vazio e o script
  reportava "the build did not finish" sobre um build que tinha produzido um PDF correto de 26 paginas;
- a `:ERRS` **e higiene e simetria, NAO conserto de bug**, e o comentario registra a medicao que estabelece
  isso: o `grep -c '^! '` devolve `0` **nos dois locales** naquele log, entao o `tex_errors=0` do `extra`
  **e verdadeiro**. Esta escrito no arquivo com essas palavras para ninguem depois ler a `:ERRS` como reparo
  de um defeito que nunca existiu.

Dois probes guardam a distincao (`R12-locale`, `R12-locale2`), porque confundir as duas linhas e exatamente
o que voce pediu para evitar. Reli o arquivo do disco imediatamente antes de editar e usei caminhos
explicitos no commit.

---

**ITEM 2 — as duas linhas do `GLOSSARY` §1.1: ENTRARAM.**

A do **$\rho$** registra o mapa de representacao e, no campo de notas, a razao pela qual ele importa: os
modelos das tarefas **sequenciais** leem $\rho(H_i)$ e nao o $H_i$, a tarefa estatica le $\mathbf{e}_p$
direto, e **os tres estudos mantem as definicoes de tarefa fixas e variam o $\rho$** — que e o que torna a
alegacao central do trabalho enunciavel.

A do **$d$** entrou **com a nota de escopo**, e aqui **corrigi um detalhe da sua propria nota**: voce escreveu
que o $d_{\mathrm{shared}}$ e largura de tronco **no Cap. 5**. **Medido: ele nao aparece no Cap. 5.** As tres
ocorrencias vivas estao nos **Caps. 3 e 4** (`3_cbic/method.tex:78`, `4_courb/methodology.tex:25` e `:258`,
com $256$ onde o valor e dado). O $d_{ij}$ (distancia geodesica no peso da aresta de Delaunay,
`3_cbic/method.tex:23`) tem quatro ocorrencias, tambem nos Caps. 3 e 4. A linha registra os capitulos
**medidos**, nao os da nota, e diz que nenhum dos dois subscritos e instancia da linha nova.

---

**ITEM 3 — a inversao: o estudo comparativo esta rodando, e eu NAO toquei em nenhum `.tex` de ordem.** A sua
suspensao esta respeitada: nada de inverter, nada de editar o `NORTH_STAR.md:73-80`, e o plano de oito passos
do `_round12/49` continua parado. O resultado vai para o **§6.24**, como voce pediu.

**E uma coisa que eu conferi antes de comecar, porque a sua nota dizia o contrario:** os tres itens que voce
listou como "no disco, sem commit" (a correcao do primeiro `[VERIFY]` do `_round12/52`, a nota de wall-clock, e
o probe `R12-studyfix`) **ja estavam commitados**, no `5446ffb3`. A arvore estava limpa. **Mas o seu ponto sobre
o `R12-studyfix` estava certo e importava:** aquele probe nunca havia disparado, porque a frase-alvo quebra
linha no fonte e a minha perna procurou a forma sem quebra. **Validei agora contra a forma envolvida**, testando
o padrao atraves do `strip_text` como voce disse: ele dispara quando a frase de correcao sai, e fica silencioso
sob uma edicao vizinha nao relacionada. Agora carrega informacao.

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
    
