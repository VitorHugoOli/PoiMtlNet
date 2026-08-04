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

**Ordem das secoes:** §2 (voce) -> §3 (terceiros) -> §4 (o que auditar primeiro). Deliberada: o
que depende de voce vem antes. **§5 removida em 2026-08-03** (retirado; os onze itens que apontava
continuam em `_archive/PENDENCIAS_RESOLVIDOS.md`, re-medidos e intactos).

**O §6 saiu deste arquivo em 2026-08-03, a seu pedido, e nao foi perdido.** Ele entrou em 2026-07-30
substituindo o §2.8 e carregou vinte e seis itens vindos do `CONSIDERATIONS.md`; **os vinte e seis foram
respondidos por voce** e estao em `_archive/PENDENCIAS_RESOLVIDOS.md`, cada um com o motivo da saida no topo do
bloco e sob o cabecalho que registra o encerramento da secao inteira. A numeracao 6.1 a 6.26 **nao** foi
reaproveitada, e as dezenove citacoes que apontavam para ca foram repontadas para o arquivo na forma historica
que o `check_tracker_refs.py` reconhece. O §6 seguiu a mesma trajetoria do §2.8: deixou de pedir decisao e
virou registro.

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

### 2.29 Rodada 12, 2026-08-03 — o §6 fechou inteiro, as duas linhas do `GLOSSARY` entraram, e voce mesmo escreveu a D2

**Registro, nao pedido.** Nada aqui espera voce; esta secao existe para que nada disto seja reaberto.

**AS DUAS LINHAS DA §1.1, aplicadas por mim sob a sua autorizacao explicita.** Voce disse "eu autorizo voce a
colar elas no glossary", que e a **opcao 3** do antigo §6.26 e nao a 1 — a diferenca importa, porque a regra
da casa e que a tabela de notacao e sua e um agente **propoe** linhas. As duas fecham uma lacuna fail-closed
medida: o $\mathbf{e}_{x_i}$ estava em uso vivo na Definicao 2.4 e o $f_{\mathrm{place}}(H_i)$ na 2.9, e
nenhum dos dois estava registrado. Isso completa a **AD-5**. **E a consequencia que a propria opcao 3
previa foi cumprida no mesmo commit:** o comentario do `2_fundamentals.tex` afirmava, verbatim, que a linha
"is PROPOSED to the author and is not written by an agent; the notation table is his" — verdadeiro quando
escrito e **tornado falso pelo ato de escrever a linha**. Ele agora cita a frase antiga como superada, diz
que voce autorizou a excecao, e mantem a regra geral de pe.

**O §6 SAIU INTEIRO, a seu pedido.** Os vinte e seis itens foram respondidos por voce, os dez `______` que
restavam eram **residuo de formatacao** (cada um ja respondido em outra secao, conferido um por um), e a
`h3` e o cabecalho `## §6` foram removidos. **Duas coisas que eu conferi porque este arquivo manda:**
1. **Chegada antes de apagar.** Para cada bloco eu confeti que o cabecalho **e** uma linha interior do
   corpo estavam no `_archive/PENDENCIAS_RESOLVIDOS.md` antes de remover. Tres itens desta lista se
   perderam no passado exatamente por apagar antes de conferir.
2. **Ponteiros.** Remover a secao orfanou **dezenove** citacoes no fonte e **quatro probes**. As citacoes
   foram repontadas para a forma historica que o `check_tracker_refs.py` reconhece
   (`PENDENCIAS_RESOLVIDOS <n>.<m> (arquivado 2026-08-03)`), e os quatro probes (`R9-pend6`, `R9-blq4`,
   `R9-blq5`, `R12-extra`) passaram a ler o arquivo — cada string **verificada presente lá** antes do
   repoint, nao suposta. O `R9-pend6` deixou de pinar um cabecalho que voce mandou remover e passa a pinar
   o registro do encerramento, com o `R9-pend6b` guardando o cabecalho citado verbatim para quem encontrar
   um comentario antigo dizendo "§6". A numeracao 6.1 a 6.26 **nao** foi reaproveitada.

**A SUA D2, afiada em cima e nao reescrita (AD-6).** Voce substituiu a frase vaga de retencao pela sua, que
nomeia o alvo por tarefa e enuncia a posse do rotulo. Tres afiamentos, nenhum tocando o seu conteudo:
- **Referencia para frente.** A sua frase era a **primeira** ocorrencia viva de "next-category prediction" e
  "next-region prediction" no capitulo, e as duas sao definidas ~130 linhas adiante. Medido, nao suposto. Os
  simbolos estavam bem (a D1 vincula o $c_i$ e o $r_i$), entao e mais leve que um simbolo-antes-da-definicao,
  mas e a propriedade que a ordem dos passos do redesenho existia para proteger. Resolvido apontando para
  frente **explicitamente**, em vez de tirar os nomes das tarefas que voce escolheu.
- **A metade positiva.** Excluir o rotulo do $x_i$ afasta o vazamento; faltava dizer que as categorias e as
  regioes das visitas **passadas** sao entrada legitima — que e exatamente a duvida que gerou a sua edicao.
  Agora esta dito numa oracao.
- Largura de linha de volta as ~85 colunas do arquivo.

**Nada mudou na 2.5, e isso e deliberado.** A sua leitura estava certa: um "historico de regioes" e uma
**projecao** de $H_i$ e nao uma entrada diferente, porque a regiao ja esta dentro do check-in pela D1, e o
que o modelo le e $\rho(H_i)$. Se a definicao da tarefa dissesse "recebe um historico de regioes", ela
passaria a descrever a escolha de representacao do Cap. 5 e o Cap. 3 nao caberia mais nela.

**Um defeito meu, apanhado por um revisor:** eu publiquei "os oito probes novos validados por sabotagem"
quando eram **sete**. O oitavo era justamente o probe de **ausencia** — o unico cuja falha e o silencio.
Corrigido, validado nos dois ramos, e a regra que evita a repeticao esta no `_round9/34`: reconciliar os
nomes dos probes validados contra os adicionados **como conjuntos**, nao pela contagem de linhas.

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
    
