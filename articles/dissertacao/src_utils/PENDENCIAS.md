# PENDENCIAS.md — o que depende de voce

> **Registro de pendencias da dissertacao (v2, 2026-07-26).** Cada item aqui esta bloqueado em um
> fato externo, uma decisao sua, ou uma aprovacao do orientador/Comissao. Nada aqui pode ser
> resolvido por um agente, e nenhum foi resolvido sozinho.
>
> A rodada de correcoes de 2026-07-26 fechou 26 dos 39 achados da revisao. O que sobrou esta
> abaixo. Auditoria completa: [`_archive/reviews_v1/dissertation_review_v2.md`](_archive/reviews_v1/dissertation_review_v2.md).
>
> Formato de cada item: **(A) o que falta**, **(B) por que importa**, **(C) o que eu preciso de
> voce**. Onde ja existe texto pronto ou pesquisa feita, o caminho esta indicado.
>
> **Estado do build agora:** defesa 96 pp, final 91 pp, 0 caixas estouradas, 0 citacoes indefinidas,
> 0 referencias indefinidas, 0 erros de BibTeX, lint exit 0.

---

## Estado apos a rodada de 2026-07-27

**Build:** defesa 103 pp, final 99 pp, 0 caixas estouradas, 0 citacoes/referencias indefinidas, 0 erros de BibTeX, lint
0, 10/10 fixtures do detector de prosa presa.

**O que rodou nesta rodada, e o que cada um achou:**

| Revisor                    | Estado antes                        | Veredicto                                                                                    |
|----------------------------|-------------------------------------|----------------------------------------------------------------------------------------------|
| Persona 15 (leitura)       | 1 dia desatualizado                 | Confirma que o Apendice D reescrito **se sustenta sozinho**; 3 costuras, todas aplicadas     |
| Persona 16 (credibilidade) | **nunca tinha rodado nesta versao** | Risco humano BAIXO; 2 BLOCKERs nos meus proprios textos, os dois corrigidos                  |
| Fact gate (G2)             | v2, build antigo                    | **GATE FAIL** em um numero (o 22,4 por cento); corrigido. 4 MAJOR + 3 MINOR, todos aplicados |
| Banca (persona 12)         | v2, build antigo                    | **APROVADO COM CORRECOES MENORES**, com 2 obrigatorias, as duas fechadas                     |

**Sobre a sua pergunta dos guardrails:** as regras mecanicas foram respeitadas nesta rodada. Auditei 1.725 palavras de
prosa nova contra o `WRITING_LAW §4`: zero termos banidos, zero travessoes, zero contracoes, zero termos fora do
registro. Mas isso nao pegava a sua reclamacao real. O Apendice D tinha as **frases mais curtas do documento**; o
problema era colisao de conceitos e dependencia externa, nao tamanho de frase. Foi por isso que medi antes de
reescrever.

**O defeito que mais se repetiu, e o que fiz sobre ele:** prosa engolida por comentario LaTeX, agora **dez** ocorrencias
no historico do documento. Tres foram encontradas pelo codex nesta rodada, duas pela persona 16 e pela banca, e **duas
foram pegas pela ferramenta** enquanto eu editava, que e a primeira vez que a maquina pega antes do revisor. O detector
foi reescrito em volta do teste que importa (o texto aparece no PDF ou nao), tem 10 fixtures no repositorio, e o
`check.sh` roda os fixtures **antes** de confiar no detector.

---

> **Itens ja encerrados** sairam deste arquivo em 2026-07-27 e vivem em
> [`_archive/PENDENCIAS_RESOLVIDOS.md`](_archive/PENDENCIAS_RESOLVIDOS.md), com decisao e commit de cada um.

## BLOCO 0 — a revisao do codex: auditada, e o que sobrou para voce

Voce pediu para auditar antes de agir. Feito, achado por achado, contra a fonte. O documento
`codex_reviewer.md` agora carrega um **AUDIT VERDICT** em cada achado, e as evidencias estao em
[`CODEX_AUDIT.md`](CODEX_AUDIT.md) e [`CODEX_VS_PERSONAS.md`](CODEX_VS_PERSONAS.md).

**Contagem:** 5 RESOLVIDOS, 1 REFUTADO como enunciado, 1 CONFIRMADO e corrigido, o resto PARCIAL com o residuo vivo
nomeado caso a caso. **Quatro alegacoes nao se sustentam** na evidencia, entre elas uma exigencia de citacao que
contraria uma decisao sua ja registrada, e uma atribuicao de capitulo que le uma frase do apendice do MobiWac como se
fosse do Cap. 4.

**Um fato estrutural que governa a leitura:** o codex leu um par de 97/92 paginas. O que esta em disco e **103/99**.
Todos os `file:line` dele deslizaram; a auditoria re-ancorou cada um pelo conteudo. Os achados sobrevivem, as
coordenadas nao.

**Os quatro achados de maior valor nao foram pegos por nenhum outro revisor**, e os quatro estao corrigidos: a contagem
errada do controle de capacidade (tres bracos de vinte, sessenta ajustes, e 56,16 e uma media e nao um maximo), o artigo
errado do Mikolov para negative sampling, uma frase do Cap. 5 que reafirmava uma atribuicao que o proprio capitulo
recusa, e uma frase do Cap. 3 que descreve uma feature de no que o codigo liberado nao constroi. Nove personas, o fact
gate e o simulador de banca passaram por todos os quatro sem ver.

### 0.2 Dois itens das personas 15 e 16 que dependem de voce

Os dois BLOCKERs delas foram corrigidos. Sobram dois pontos que sao decisao sua:

1. **A linha de divulgacao de IA no front matter** (persona 16, o achado de maior valor dela, aberto desde a v1). O
   Apendice C esta na p. 97 do build de defesa, e a persona argumenta que uma linha curta no front matter, antes do
   corpo, muda como um examinador desconfiado le o documento inteiro:
   ele encontra a divulgacao antes de formar suspeita, e nao depois. Nao escrevi nada: e uma frase em seu nome sobre o
   seu proprio processo.

2. **O `\label{apx:ethics}` orfao** (persona 15). O Apendice E tem label e nada o referencia. E inofensivo hoje, mas se
   voce quiser que algum capitulo aponte para a etica (o Cap. 1 seria o lugar natural, junto da descricao dos dados), me
   diga onde e eu insiro a referencia.

> DECISAO: 1. No `Contents` ele já está presente, mas de qualquer forma estamos seguindo as diretrizes do manual da UFV,
> acredito que manter neese formato sejá o melhor, se algum revisor achar pertinente ele mandara eu mudar.; 2. Gostei da
> ideia de referenciarmos de forma rapida no Cap. 1.

---

### 0.1 O que do codex ainda depende de voce

| Item    | O que falta                                                                                                                                                                    |
|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| COD-002 | A tarefa estatica do Cap. 4: o determinismo esta medido e confirmado, mas a **divulgacao** ainda nao existe no texto, e escrever isso exige aviso ao co-autor (ja no item 2.2) |
| COD-003 | O teste nao-linear de aresta futura na linhagem exata entregue. O texto **nao** exagera (isso foi refutado), mas o teste em si e uma decisao de compute sua (item 3.4)         |
| COD-005 | Nash-MTL: voce ja decidiu sobre o PCGrad. Sobra a alegacao de custo, ja declarada no Apendice B como preservada deliberadamente (item 1b.3)                                    |
| COD-006 | "before any result was read" e "well powered": os dois excedem o desenho. Correcao de uma frase cada, mas em texto de moldura que voce assina                                  |
| COD-007 | Os registros de protocolo que faltam no Cap. 3. Sao registros historicos que so voce sabe se existem                                                                           |
| COD-013 | O Apendice C diz que voce leu e aprovou cada palavra. 32 marcadores abertos (item 3.1)                                                                                         |
| COD-016 | O passe de linguagem: a frase de 114 palavras do resumo e o bloco de 546 palavras da integridade. Voce ja pediu para adiar (item 3.3)                                          |
| COD-017 | O float grande do Apendice B e os rotulos de 6,97 pt nos diagramas: bloqueado na arte-fonte (item 3.2)                                                                         |
| NUM-4   | O `[VERIFY]` do 0,74 -> 0,82 do HGI: a fonte da 0,7388 +- 0,0205 -> 0,8186 +- 0,0123 em 5 folds x 50 epocas, e a prosa nao diz a convencao                                     |

---

## BLOCO 0b — o que o orientador levantou (2026-07-27)

### ~~Notacao das citacoes: (N) -> [N]~~ — APLICADO 2026-07-27

Palavras dele: *"sugiro mudar a notacao das citacoes tambem. Esta com (NUMERO) e na listagem esta com o
numero sem []. Acho melhor usar o [NUMERO] e quando necessario, colocar o nome dos autores com et al."*

**Medido antes de mexer:** 236 citacoes no formato `(N)` no texto, zero em `[N]`, e a listagem com o
numero pelado (`1 SILVA, V. H. O. et al. ...`). Ele estava certo nas duas metades, e as duas metades
tinham causas diferentes:

| Onde | Mecanismo | Correcao |
|---|---|---|
| no texto | `abntex2cite` chama `\setcitebrackets`, que no estilo `num` usa `()` | `\citebrackets{[}{]}` depois do pacote |
| na listagem | `\@biblabel` monta o rotulo via `\citenumstyle`, que imprime o numero pelado | `\@biblabel` redefinido dentro de `\AtBeginDocument` |

**Resultado:** 334 citacoes `[N]` no texto e a listagem em `[1] SILVA, V. H. O. et al. ...`, as duas na
mesma forma. Paginas inalteradas (103/99), zero referencias indefinidas. As duas ocorrencias residuais
de `(N)` sao legitimas: um numero de volume de revista e uma referencia a "capitulo (4)".

**Precedente, porque isso e escolha de estilo e nao norma:** o Viegas — o exemplar cujo padrao este
documento segue — usa `[N]` (193 ocorrencias contra 15 de `(N)`); Germano, canesche, passe e lapsusvgi
usam `(N)`. Ou seja, os dois formatos passaram no programa. A decisao e dele e esta aplicada.

**A segunda metade do pedido dele ainda e sua:** *"quando necessario, colocar o nome dos autores com et
al."* Isso e caso a caso — trocar `\cite{}` por `\citeonline{}` onde o autor deve aparecer no corpo da
frase ("Silva et al. mostram que...") em vez de so o numero. Nao fiz em lote porque **muda o sujeito
gramatical de cada frase afetada**, e escolher onde o autor merece destaque e julgamento seu. Diga
quais passagens e eu aplico.

> DECISAO (quais passagens levam `\citeonline`?): _______________________________________

### 3.4 O jeito como os termos entram (o segundo ponto dele) — **revisao rodando**

Palavras dele: *"so tome cuidado com o uso de IA e os termos menos comuns que sao usados... soa um pouco
estranho o jeito que alguns termos sao inseridos (marquei alguns la)"*.

**Voce perguntou se rodamos o revisor disso. A resposta honesta e: rodei agora, e havia um buraco
real.** A persona **03 (style auditor)** e o gate G3, obrigatorio antes de cada entrega ao orientador.
O relatorio v2 dela e de **26/07 contra um build de 94 paginas** — o documento tem 103 hoje e levou
umas trinta commits desde entao.

**Pior, e este e o ponto que voce levantou sobre o glossario do MobiWac:** o brief dela, na linha 27,
manda ler `articles/[mobiwac]/GLOSSARY.md` e diz que **ele vence para o Cap. 5**. O relatorio v2 tem
**zero** referencias a esse arquivo. Aquele glossario tem 393 linhas, com uma tabela de 26 linhas de
substituicao de jargao e uma secao de palavras a evitar. **Nao foi aplicado.**

**Uma violacao eu ja achei e corrigi:** a palavra **"arm"** esta na lista never-use dele ("clinical-trial
word, foreign to this audience") e **eu mesmo a inseri** na frase de limitacoes do Cap. 5 nesta rodada.
Corrigida para "both models" / "the dedicated model" nos **dois** textos, porque o meu port tinha levado
a violacao para o artigo tambem.

A persona 03 esta rodando agora com o glossario do MobiWac como carga explicita, mais o pedido dele
operacionalizado em cinco testes mediveis (termo usado antes de ser definido; definido duas vezes; glosa
em registro que briga com a frase; termo usado uma unica vez no documento; empilhamento de glosas
apositivas). Resultado em `_review_v3/03_style_auditor_report.md`.

> **Falta voce:** ele disse *"marquei alguns la"*. Onde estao as marcacoes? Num PDF comentado, num
> e-mail, no Word? Com a lista dele em maos eu cruzo com o que a persona achou e trato os dois.

> DECISAO / ONDE ESTAO AS MARCACOES: ____________________________________________

---

## BLOCO 1 — bloqueiam a entrega, nao a ciencia

### 1.1 Banca, data, capa e folha de aprovacao (REV-023)

**(A)** `0_main.tex:122-124` tem tres placeholders entre colchetes (membros da banca e data). O build de defesa comeca
na folha de rosto: nao ha capa (`\imprimircapa` existe no `.sty` mas nunca e chamado), `\campus{}` nunca e setado, e nao
ha ficha catalografica. A folha de aprovacao e um placeholder literal.

**(B)** Um documento cientificamente correto nao pode ser depositado com front matter incompleto. Isso independe de tudo
o mais nesta lista.

**(C)** Preciso de: nomes e afiliacoes dos membros da banca, a data marcada da defesa, e a decisao sobre a capa. Sobre a
folha de aprovacao, a decisao 3.9 do doc anterior continua valendo e a minha recomendacao nao mudou: manter o
placeholder honesto, que e o que o precedente do Germano de fato faz (ele deixou o `\includepdf` do modelo COMENTADO).

**Ja feito nesta rodada:** a macro `\imprimirfolhadeaprovacao` no `abntex2-UFV.sty` tinha o nome de **outro aluno**
(`Gabriel Vita Silva Franco`) hardcoded. Estava inofensivo porque a macro nao e chamada, mas quem trocasse o placeholder
pela macro imprimiria o autor errado na folha de assinaturas. Corrigido para usar `\imprimirautor`.

> DECISAO / DADOS: A) o campus é o Florestal, sobre a folha de rosto como e feito nos outras dissertacoes de exemplos,
> quanto ao restante dos itens fica em aberto até meu orientador me retornar.
> B) Vamos preenchendo de acordo com o que formos completando
> Se possivel vamos tentar remover referencias aos exemplos que usamos como o do germano e do Gabriel
> Algo que gostaria de discutir com voce ainda sobre o topico de organização do latext e sobre como está nosso main. Eu
> acredito que poderiamos ter um main.tex, esse serai um arquivo limpo como 0_main.tex no qual terimos o confteudo da
> dissertacão sem a folha de aprovacao, e para a folhar teriamos outro main_ppgc.tex com a folha de aprovacao. Com isso
> mudariamos o makefile para ser mais simplificado hoje ele está bem complexo. Além desse ponto algo que está me
> incomodando bastante e o execcso de comments, e algo bom e necessario para mantermos o track de varis inforamcoes
> criticas, mas sera que não teria como cortar alguns comentarios ou ser mais direto. Outro ponto é esse e mais critico,
> nos chapters os textos estão corridos, principalmente para os artigos courb, cbic e mobiwac, no latex original desses
> o texto era divido e as tabelas separadas, assim dando mais facilidade de manutenção, até existem pasta mas elas estão
> vazias. Outro ponto é sobre margem,
> padding e outras formataçoes, estamos aplicando as melhores praticas ? (olhe no exemplo do germano e nos gits de
> exemplos de tese e dissertacao que tinhamos pego), pergunto isso pq eu posso estar com um falso precentimento que
> estmos aplicadnod algumas formataçoes de forma locais en quanto elas deveriam ser globais. Enfim, pf avalie esses
> pontos com cuidado, sinta-se avontade para negar ou contra argumentar e vamos tomar as decissões que fazem mais
> sentido para o texto e organizacão.

### 1.2 Pacote de aprovacoes do orientador (uma conversa so)

**(A)** Quatro decisoes que so o orientador (e possivelmente a Comissao) fecha, e que e melhor levar juntas: (i) o
**frame em ingles**; (ii) a **inclusao do capitulo CoUrb** traduzido, em que voce e segundo autor; (iii) o **titulo
final** (a opcao 1 esta ativa como titulo de trabalho, as alternativas estao comentadas no `0_main.tex`); (iv) a
**politica de errata** adotada.

**(B)** A politica de errata e a que mais trava trabalho: quase toda correcao em texto publicado desta rodada entrou
pelo mecanismo do Apendice B, e ele so fica legitimo com o aval dele.

**(C)** Uma conversa, quatro respostas.

> DECISAO: __________________________________________________

### 2.2 Escopo da tarefa estatica do Cap. 4 (REV-002) — **medido nesta rodada, e o resultado nao ajuda**

**(A)** Voce escreveu: *"se nao me engano usou o fclass e nao a categoria ... vamos avaliar o tamanho do problema,
porque os numeros ficaram bem proximos do DGI."* A premissa esta certa. **Eu medi, e ela nao ajuda.**

Em `data/checkins_by_state/Alabama.parquet` (113.846 linhas): o corpus tem **275 valores distintos de `fclass`** (a
categoria fina: Airport, Coffee Shop, Seafood) e **7 categorias de topo**, que sao o alvo. **Cada um dos 275 mapeia para
exatamente uma categoria. Zero mapeiam para mais de uma.**

A cadeia, cada elo verificado em codigo: `poi2vec.py:486-487` faz
`poi_embeddings[valid] = fclass_embeddings[fclass_values[valid]]`, entao o vetor do lugar e funcao pura do `fclass`; o
`fclass` determina a categoria deterministicamente; por composicao, **o embedding de lugar determina exatamente o rotulo
alvo**. Usar `fclass` em vez de `category` deixa a entrada *mais* informativa sobre o alvo, nao menos. E os numeros
"proximos do DGI" sao consistentes com isso, e nao um alivio: a entrada do DGI e a media one-hot dos vizinhos da mesma
taxonomia.

**Importante, e a favor do documento:** isso vale para a tarefa **estatica**. A tarefa **sequencial** dos dois capitulos
e limpa (`3_cbic.tex:161-167`, `4_courb.tex:125`) e nao e afetada. A revisao v1 nao fez essa distincao, e ela e a
diferenca entre uma frase de escopo e uma retratacao.

**(B)** `apx_a_contributions.tex:91-93` conta a banca que uma submissao anterior foi atacada por vazamento de rotulo e
que o Cap. 5 responde com uma auditoria dedicada. Quem ler isso e depois ler o Cap. 4 vai perguntar por que o Cap. 4 nao
recebeu tratamento equivalente.

**(C)** Duas coisas: (i) o Cap. 4 e artigo **publicado e co-autorado**, com o Tarik como primeiro autor, entao a frase
de escopo precisa de **aviso de cortesia** a ele antes de entrar; (ii) voce sugeriu um apendice para o tema, e concordo,
e o lugar certo para a medicao acima. Autoriza?

> DECISAO: Ok, ótima revisão, mas vamos em partes. Eu entendi que vc audito e validamos o que tinhamos medo o poi2vec
> usado junto ao hgi no courb possue vazamento de dados. Isso é um ponto, mas esse não se aplica ao DGI que usamos no
> cbic, então a tarega estatica ela só possui problema no courb. E sendo bem honesto eu já suspeitava até por isso no
> mobiwac eu tomei a decissão de alterar. Dito tudo isso, é levando em consideração todos os pontos, eu acredito que
> valha um appendix para isso ou inserimos essa discução em um dos appendix, e no prefacio do courb apontamos para esse
> apendix. O que acha ?

**MINHA RESPOSTA A SUA PERGUNTA ("o que acha?"), 2026-07-27.**

**Concordo com o apendice, e a sua leitura tecnica esta correta em todos os tres pontos.** Confirmei cada um contra o
codigo antes de responder:

1. **O DGI do CBIC nao tem o problema.** `research/embeddings/dgi/preprocess.py:112-133` monta a feature de no como a
   **media dos one-hots dos vizinhos, excluindo o proprio no**. O rotulo do no nao entra na sua propria representacao.
   Voce estava certo em separar os dois casos.
2. **O HGI do CoUrb tem.** O `poi2vec` consome `fclass`, e eu medi: **275 valores distintos de
   `fclass`, cada um mapeando para exatamente uma das 7 categorias, zero ambiguidade**. O `fclass` e um refinamento
   estrito do rotulo-alvo, logo determina o alvo. O revisor de mobilidade re-derivou isso independentemente **nos cinco
   estados**, o que e mais forte do que a minha medicao num.
3. **O MobiWac ja resolve.** Voce trocou a representacao, e por isso o Cap. 5 nao herda o problema.

**Onde eu discordaria da forma, nao do conteudo:** um apendice novo so para isto seria o **sexto**, e o documento ja tem
cinco. A minha recomendacao e **inserir a discussao no Apendice B**, que e exatamente o instrumento para "o que o texto
publicado diz x o que sabemos hoje", e apontar do prefacio do Cap. 4 para lá — a estrutura que voce propos, num apendice
que ja existe. Se preferir um apendice proprio eu faco; e a sua chamada de forma.

**Por que eu nao escrevi ainda:** isto e uma afirmacao publica de que um resultado publicado do CoUrb, com o **Tarik
como primeiro autor**, tem um vazamento de rotulo. Isso pede um aviso de cortesia a ele antes de entrar no documento, e
esse aviso e seu para dar. Diga quando ele estiver ciente e eu escrevo — a medicao, o codigo e os numeros estao prontos
e verificados.

> DECISAO (aviso ao co-autor dado? forma: Apendice B ou apendice proprio?): Isso já está de acrodo com ele, podemos
> adicionar isso no appendix B, mas deixe isso facil de ser comentado, vide que eu ainda vou discutir com meu orientador
> sobre se argumentamos ou não quanto a isso.

### 1b.4 O determinismo da categoria agora esta medido nos CINCO estados (persona 11)

Nao e decisao, e reforco: a persona 11 refez a medicao do item 2.4 em todos os estados, nao so no Alabama. **284 a 365
valores `fclass` distintos por estado, nenhum mapeando para mais de uma das 7 classes-alvo, em Alabama, Arizona,
Florida, California e Texas.** Ou seja: nao e artefato de um dataset. A frase de escopo do Cap. 4 continua sendo a
pendencia (aviso ao co-autor primeiro).

---

## BLOCO 2b — decisoes herdadas do `_archive/reviews_v1/DECISOES_PENDENTES_ptBR.md` que continuam abertas

> Auditei os 12 itens daquele documento contra o fonte de hoje. **Tres continuam abertos** e estao
> abaixo. **Seis foram resolvidos** nas rodadas seguintes e estao registrados no fim desta secao,
> para voce nao reabrir a esmo. Os outros tres ja aparecem nos Blocos 1 e 3 deste documento
> (titulo, Resumo/Abstract, folha de aprovacao, figuras).

### 2b.3 Movimentos opcionais de excelencia (era 3.10)

**Estado:** nenhum dos tres existe. Verifiquei: nao ha tabela contribuicoes→alegacoes no §1.6, nao ha tabela consolidada
de resultados no Cap. 6, nao ha apendice de artefatos.

Sao adicoes de **frame** (nao tocam resultado), na lente de premio SBC-CTD: (a) tabela contribuicao→alegacao no §1.6;
(b) tabela consolidada cross-chapter no Cap. 6; (c) apendice de reprodutibilidade (codigo, seeds, configs).
**Observacao:** o (c) ficou mais facil agora, porque o Apendice D novo ja estabelece o padrao de citar script + arquivo
de saida para cada numero.

**DECISAO:** quer algum dos tres? (cada um e ~1 pagina)
> DECISAO: Eu gosto te todas as opções. Meu receio e ser muito vide a quantidade de pagina e as varias mudanças no texto
> que estamos fazendo. Eu acho que o A e o com menor ganho, o B e o C, são opcionais interessantes. Como isso está sendo
> feito nas dissertações de exemplos de excleencias que captamos ?

**RESPOSTA MEDIDA A SUA PERGUNTA, 2026-07-27.** Voce perguntou como isso e feito nas dissertacoes de excelencia que
captamos. Fui olhar nos cinco exemplares em `exemples/`, procurando cada um dos tres padroes no texto extraido:

| Exemplar                             | Paginas | (a) contribuicao→alegacao | (b) resultados consolidados | (c) reprodutibilidade                               |
|--------------------------------------|---------|---------------------------|-----------------------------|-----------------------------------------------------|
| `lapsusvgi.pdf`                      | 77      | nao                       | nao                         | **sim**                                             |
| `canesche_2021.pdf`                  | 108     | nao                       | nao                         | **sim**                                             |
| `dissertacao_viegas_2026-02-09.pdf`  | 100     | nao                       | nao                         | **sim**                                             |
| `passe.pdf`                          | 68      | nao                       | nao                         | nao                                                 |
| `Dissertação_Mestrado___Germano.pdf` | 96      | nao                       | **sim**                     | **sim**                                             |
| **nosso**                            | **103** | nao                       | nao                         | parcial (Apendice D cita script + saida por numero) |

**A medicao muda a minha recomendacao, e ela coincide com o seu instinto.**

- **(a) tabela contribuicao→alegacao: nao facamos.** **Zero de cinco** exemplares tem. Voce disse que o A e o de menor
  ganho e a evidencia concorda: nao e convencao do programa, e num documento em formato de coletanea o mapeamento
  contribuicao→capitulo ja e explicito por construcao.
- **(c) reprodutibilidade: e o unico que tem apoio empirico forte** — **quatro de cinco**, incluindo o precedente do
  Germano com o mesmo orientador. E o Apendice D novo ja estabeleceu o padrao (cada numero cita script e arquivo de
  saida), entao seria uma consolidacao de ~1 pagina, nao um texto novo. **Se voce fizer um dos tres, faca este.**
- **(b) resultados consolidados: um de cinco.** O Germano tem. E defensavel mas nao e convencao, e no nosso caso a
  Tabela do Cap. 5 ja e o resultado consolidado do trabalho.

**Sobre o seu receio de paginas:** estamos em **103**, contra 68 a 108 nos exemplares. Nao estamos longos; estamos no
meio da faixa, e o Canesche com 108 e mais longo. Uma pagina de reprodutibilidade nao muda esse quadro. **A minha
recomendacao: so o (c), e so quando o texto assentar** — ele inventaria scripts e seeds, entao escrever antes das
ultimas edicoes garante retrabalho.

> DECISAO: Concordo, vamos com a opc: C). E acredito que possamos usar o Appendix A para isso, o que acha ? Lá já
> citamos a questão das contribuições no codigo. Se não um appendix novo mas bem enxuto e direto apontando para um
> arquivo de entrada na codebase que faça explicações mais elaboradas.

### Resolvidos desde aquele documento (registrado para nao reabrir)

| Item de la                                            | Estado hoje                                                                                                                                                                                                                                               |
|-------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 3.1 Wilcoxon x t pareado                              | **RESOLVIDO.** Cap. 2 (`:497-503`) e Cap. 5 (`:412`) agora concordam: t pareado nas medias por repeticao **mais** Wilcoxon nos folds individuais, ambos reportados, com o piso do p exato do Wilcoxon explicado. O desvio do pre-registro esta declarado. |
| 3.2 CV usuario-disjunta: documento todo ou so Cap. 5? | **RESOLVIDO** (REV-006). O Cap. 2 agora escopa explicitamente: os testes "license verbs in Chapter 5 alone" (`:495-496`).                                                                                                                                 |
| 3.3 Pre-registro da nao-inferioridade explicito       | **RESOLVIDO e reforcado.** `5_mobiwac.tex:412` declara o plano escrito, fixado antes de ler resultado, a margem de dois pontos, **e** que ele nao cobria superioridade de regiao (os 4 ganhos sao secundarios). Mais honesto que o pedido original.       |
| 3.4 Vintage 2009-2011                                 | **APLICADO** na rodada 3.                                                                                                                                                                                                                                 |
| 3.5 Ponte "next-POI"                                  | **APLICADO** na rodada 3, e o Cap. 3 recebeu nota de rodape adicional nesta rodada (REV-010).                                                                                                                                                             |
| 3.6 Contradicao class-weighted CE                     | **RESOLVIDO** nesta rodada. `2_fundamentals.tex:456` agora diz "plain unweighted cross-entropy; class weighting, tested there on both outputs, lowered..." — concorda com o Cap. 5.                                                                       |

---

## BLOCO 3 — assinaturas e itens adiados

### 3.1 Os 32 marcadores `[NEEDS SIGN-OFF]`

Voce pediu a lista. Os 32 marcadores em 10 arquivos, todos comentarios LaTeX (**nenhum renderiza**, entao nao ha
sujeira no PDF). O risco nao e visual: e que o **Apendice C afirma** que o autor leu e aprovou cada palavra, enquanto o
proprio apendice esta marcado como nao aprovado. Voce ja decidiu manter o Apendice C como esta, o que torna esta lista o
caminho para tornar a afirmacao verdadeira.

| Arquivo | Marcadores | Nota |
|---|---|---|
| `0_main.tex` | 6 | Resumo e Abstract: **par de paridade** |
| `chapters/5_mobiwac.tex` | 6 | Prefacio, recap, figura, atribuicao, clausula do F50, o piso de Markov |
| `chapters/6_conclusion.tex` | 6 | Escopos de alegacao, clausula do F50, contagem do controle de capacidade |
| `chapters/apx_a_contributions.tex` | 4 | Apendice inteiro; a §A.2 foi removida nesta rodada |
| `chapters/apx_b_errata.tex` | 3 | Apendice inteiro, errata de grafia, errata do Mikolov, errata do DGI |
| `chapters/1_introduction.tex` | 2 | Correcao de gate L3, unidade inferencial |
| `chapters/2_fundamentals.tex` | 2 | Escopo dos 93% do Song, de-duplicacao L3, descricao do CAGrad |
| `chapters/apx_c_ai_disclosure.tex` | 1 | Apendice inteiro (COD-013: ele afirma que voce leu cada palavra) |
| `chapters/apx_d_ceiling.tex` | 1 | Apendice reescrito (label-history benchmark) |
| `chapters/apx_e_ethics.tex` | 1 | **Apendice novo**: afirmacoes institucionais em seu nome |
| **TOTAL** | **32** | medido em 2026-07-27 apos a extracao das tabelas |

**Regra que nao da para contornar:** os 6 do `0_main.tex` sao **um par**. Resumo e Abstract carregam as mesmas
alegacoes, e aprovar um sem o outro quebra a paridade. Leia os dois lado a lado.

**Um termo novo precisa entrar no GLOSSARY antes de virar canonico:** usei **"modelos ajustados"**
como equivalente PT de "fitted models" no Resumo. O GLOSSARY §6 nao tem essa entrada, e a regra e fail-closed (o termo
entra no registro **antes** de entrar no texto). Confirma o termo?

> DECISAO: Eu ainda vou ler o texto como um todo e passar por varios deles e tmb dependo da decisão do meu professor. No
> momento, só aponte via esse documento os mais criticos a serem resolvidos.

### 3.2 Figura 2 do Cap. 4: rotulos em portugues (REV-022)

**(A)** A figura da arquitetura na p. 48 tem `Encoder Espacial`, `Encoder Temporal`, `Encoder
Categorico`, `Coordenadas (lat, lon)`, `Timestamps (hora, dia)`, `Categorias (POI graph)` dentro de um capitulo em
ingles, sob legenda em ingles.

**(B)** Duas personas classificaram como bloqueador visual.

**(C)** **Bloqueado por falta do fonte.** As Figuras 1, 2 e 3 existem so como PNG achatado; nao ha
`.drawio`, `.svg` nem `.py` em lugar nenhum sob `articles/dissertacao/`. Preciso de uma de duas coisas: o arquivo fonte
(com os autores do CoUrb, provavelmente com o Tarik), ou autorizacao para **recriar** a figura do zero. Recriar levanta
questao de fidelidade, porque a figura pertence a um artigo publicado co-autorado, entao nao faco sozinho.

**Ja feito:** o rotulo do eixo da Figura 6 dizia "Score (0-1)" para uma silhueta definida em
[-1, 1]; corrigido e a figura foi regerada (o resto do PDF e byte-identico).

> DECISAO: Eu adicione os drawio: articles/dissertacao/src/figures/mtlnet_poi_new.drawio e o
> articles/dissertacao/src/figures/courb/arquitetura_modelo.drawio, respecitvamente par ao cbic e o courb. Quanto a
> imagem de distribuicao_estados.png, essa também tem palavras em portgues, e para gerar ela temos que investigar o
> /Users/vitor/Desktop/mestrado/temp/tarik-new.

