# PENDENCIAS.md — o que depende de voce

> **Registro de pendencias da dissertacao (v2, 2026-07-26).** Cada item aqui esta bloqueado em um
> fato externo, uma decisao sua, ou uma aprovacao do orientador/Comissao. Nada aqui pode ser
> resolvido por um agente, e nenhum foi resolvido sozinho.
>
> A rodada de correcoes de 2026-07-26 fechou 26 dos 39 achados da revisao. O que sobrou esta
> abaixo. Auditoria completa: [`dissertation_review_v2.md`](dissertation_review_v2.md).
>
> Formato de cada item: **(A) o que falta**, **(B) por que importa**, **(C) o que eu preciso de
> voce**. Onde ja existe texto pronto ou pesquisa feita, o caminho esta indicado.
>
> **Estado do build agora:** defesa 96 pp, final 91 pp, 0 caixas estouradas, 0 citacoes indefinidas,
> 0 referencias indefinidas, 0 erros de BibTeX, lint exit 0.

---

## BLOCO 1 — bloqueiam a entrega, nao a ciencia

### 1.1 Banca, data, capa e folha de aprovacao (REV-023)

**(A)** `0_main.tex:122-124` tem tres placeholders entre colchetes (membros da banca e data). O
build de defesa comeca na folha de rosto: nao ha capa (`\imprimircapa` existe no `.sty` mas nunca e
chamado), `\campus{}` nunca e setado, e nao ha ficha catalografica. A folha de aprovacao e um
placeholder literal.

**(B)** Um documento cientificamente correto nao pode ser depositado com front matter incompleto.
Isso independe de tudo o mais nesta lista.

**(C)** Preciso de: nomes e afiliacoes dos membros da banca, a data marcada da defesa, e a decisao
sobre a capa. Sobre a folha de aprovacao, a decisao 3.9 do doc anterior continua valendo e a minha
recomendacao nao mudou: manter o placeholder honesto, que e o que o precedente do Germano de fato
faz (ele deixou o `\includepdf` do modelo COMENTADO).

**Ja feito nesta rodada:** a macro `\imprimirfolhadeaprovacao` no `abntex2-UFV.sty` tinha o nome de
**outro aluno** (`Gabriel Vita Silva Franco`) hardcoded. Estava inofensivo porque a macro nao e
chamada, mas quem trocasse o placeholder pela macro imprimiria o autor errado na folha de
assinaturas. Corrigido para usar `\imprimirautor`.

> DECISAO / DADOS: __________________________________________________

### 1.2 Pacote de aprovacoes do orientador (uma conversa so)

**(A)** Quatro decisoes que so o orientador (e possivelmente a Comissao) fecha, e que e melhor
levar juntas: (i) o **frame em ingles**; (ii) a **inclusao do capitulo CoUrb** traduzido, em que
voce e segundo autor; (iii) o **titulo final** (a opcao 1 esta ativa como titulo de trabalho, as
alternativas estao comentadas no `0_main.tex`); (iv) a **politica de errata** adotada.

**(B)** A politica de errata e a que mais trava trabalho: quase toda correcao em texto publicado
desta rodada entrou pelo mecanismo do Apendice B, e ele so fica legitimo com o aval dele.

**(C)** Uma conversa, quatro respostas.

> DECISAO: __________________________________________________

### 1.3 Fonte da bibliografia: 12 pt ou `\footnotesize`? (REV-024)

**(A)** `0_main.tex:369-370` envolve a bibliografia em `{\footnotesize ...}`. As paginas de
referencia medem **9,96 pt** contra **11,96 pt** do corpo. A regra (`UFV_COMPLIANCE.md:32`, Manual
§8) diz 12, sem excecao para bibliografia.

**(B)** O que muda o quadro: o construto foi herdado do esqueleto do **Germano**, que defendeu em
2024 com o mesmo orientador e cujas paginas de referencia medem os mesmos 9,96 pt. O Viegas, ao
contrario, usa 11,96. Ou seja: um exemplar fez e passou, outro nao fez, e a regra escrita diz 12. O
seu proprio doc de compliance antecipa esse dilema e responde **"comply, don't gamble"**.

**(C)** A edicao e uma delecao de uma linha. O motivo de precisar de voce e a consequencia: a
secao de referencias cresce cerca de duas paginas, e isso interage com a paginacao do AcademicoPG,
que so fecha depois do rascunho no portal. Vale levar junto com o item 1.2.

> DECISAO: __________________________________________________

---

## BLOCO 2 — exposicao cientifica real

### 2.1 Etica e governanca de dados — **o item mais exposto da lista** (REV-026)

**(A)** Uma varredura dos nove arquivos de capitulo por
`ethic|privacy|re-identif|anonym|consent|GDPR|LGPD|IRB|licen[cs]e|terms of use` retorna quatro
ocorrencias, e **todas as quatro sao o verbo "license"** em "the test that licenses the verb
outperforms". Nao existe **uma unica frase renderizada** sobre etica, privacidade,
re-identificacao, consentimento ou licenciamento, numa dissertacao cujo objeto sao trajetorias de
movimento por usuario.

**(B)** O simulador de banca desta rodada perguntou isso diretamente e classificou como
*obrigatoria*. Disponibilidade publica nao elimina risco de re-identificacao, e a banca vai
perguntar por que processar trajetorias individuais foi eticamente aceitavel. E muito mais provavel
perguntarem pela ausencia do que contestarem o paragrafo depois de escrito.

**(C)** A pesquisa de licenca **ja esta feita** e esta em
[`DATASET_LICENSING_FINDINGS.md`](DATASET_LICENSING_FINDINGS.md). O que ficou verificado:

- **Gowalla (Cap. 5):** o ETL consome o dump do Figshare (registro 22126586, DOI
  `10.6084/m9.figshare.22126586.v2`), rotulado **CC0**, e os tres arquivos batem com o que o
  `src/etl/gowalla/main.py:22-24` le. **Ressalva que voce precisa saber:** o CC0 foi aplicado por um
  **depositante terceiro**, e a origem que o registro cita (`yongliu.org/datasets/`) nao abriu. Nada
  prova que o depositante tinha o direito de aplicar CC0.
- **Discrepancia real:** `docs/context/DATASETS.md:187-199` documenta o **SNAP** como fonte do
  Gowalla e nao tem linha de licenca nenhuma. Mas o SNAP e um artefato diferente: 6.442.890
  check-ins sem categoria, contra 36.001.959 com a anotacao de sete categorias no dump do Figshare.
  O registro documenta uma fonte que o pipeline nao le.
- **Massive-STEPS (Istanbul):** o "Open-source; academic research" do repo pode ser substituido por
  uma licenca identificada, **Apache-2.0**, no card do Hugging Face e no `LICENSE` do GitHub.

**Preciso de voce, tres fatos, e nenhum eu posso inventar:**

1. A UFV/PPGCC exige determinacao de **CEP** para analise secundaria de dados publicos
   de-identificados? (Se exige, o numero/parecer. Se dispensa, sob qual regra.) **Nao vou fabricar
   aprovacao nem dispensa de CEP.**
2. Voce confirma o dump do Figshare como a fonte de registro do Gowalla, ciente da ressalva do
   depositante terceiro?
3. Quer que eu corrija `docs/context/DATASETS.md` para registrar a fonte que o pipeline realmente
   le, com a linha de licenca?

Com esses tres, o paragrafo de governanca se escreve sozinho, curto e factual.

> DECISAO: __________________________________________________

### 2.2 Escopo da tarefa estatica do Cap. 4 (REV-002) — **medido nesta rodada, e o resultado nao ajuda**

**(A)** Voce escreveu: *"se nao me engano usou o fclass e nao a categoria ... vamos avaliar o
tamanho do problema, porque os numeros ficaram bem proximos do DGI."* A premissa esta certa. **Eu
medi, e ela nao ajuda.**

Em `data/checkins_by_state/Alabama.parquet` (113.846 linhas): o corpus tem **275 valores distintos
de `fclass`** (a categoria fina: Airport, Coffee Shop, Seafood) e **7 categorias de topo**, que sao
o alvo. **Cada um dos 275 mapeia para exatamente uma categoria. Zero mapeiam para mais de uma.**

A cadeia, cada elo verificado em codigo: `poi2vec.py:486-487` faz
`poi_embeddings[valid] = fclass_embeddings[fclass_values[valid]]`, entao o vetor do lugar e funcao
pura do `fclass`; o `fclass` determina a categoria deterministicamente; por composicao, **o
embedding de lugar determina exatamente o rotulo alvo**. Usar `fclass` em vez de `category` deixa a
entrada *mais* informativa sobre o alvo, nao menos. E os numeros "proximos do DGI" sao consistentes
com isso, e nao um alivio: a entrada do DGI e a media one-hot dos vizinhos da mesma taxonomia.

**Importante, e a favor do documento:** isso vale para a tarefa **estatica**. A tarefa
**sequencial** dos dois capitulos e limpa (`3_cbic.tex:161-167`, `4_courb.tex:125`) e nao e afetada.
A revisao v1 nao fez essa distincao, e ela e a diferenca entre uma frase de escopo e uma retratacao.

**(B)** `apx_a_contributions.tex:91-93` conta a banca que uma submissao anterior foi atacada por
vazamento de rotulo e que o Cap. 5 responde com uma auditoria dedicada. Quem ler isso e depois ler
o Cap. 4 vai perguntar por que o Cap. 4 nao recebeu tratamento equivalente.

**(C)** Duas coisas: (i) o Cap. 4 e artigo **publicado e co-autorado**, com o Tarik como primeiro
autor, entao a frase de escopo precisa de **aviso de cortesia** a ele antes de entrar; (ii) voce
sugeriu um apendice para o tema, e concordo, e o lugar certo para a medicao acima. Autoriza?

> DECISAO: __________________________________________________

### 2.3 Conflito: sua decisao sobre Nash x instrucao do NORTH_STAR (REV-005)

**(A)** Voce decidiu: *"vamos ignorar esse erro ... para o cap. 3 nao adicionamos caveat nem
errata."* Aplicado exatamente. Mas o `NORTH_STAR.md:146` lista **"Nash-MTL caveat as in Ch.3"** como
item de honestidade **do Cap. 4**, escrito e nunca executado. Sua decisao (posterior) esta contra
uma instrucao escrita (anterior).

**(B)** Escopo do que esta em jogo, para a decisao ser barata: **so a alegacao de preferencia de
otimizador** morre. O resultado principal do Cap. 3 (paridade MTL x single-task) nao depende de qual
balanceador estava ativo, e o Cap. 5 nao usa Nash.

**(C)** Ou (a) mantenho como esta e **corrijo o NORTH_STAR** para registrar que a instrucao foi
revogada por decisao sua, ou (b) adiciono a frase de caveat no prefacio do Cap. 4. Nao resolvi
sozinho porque e uma contradicao entre duas ordens suas.

> DECISAO: __________________________________________________

### 2.4 PCGrad continua nomeado (REV-011)

**(A)** Apliquei sua redacao, "at their default configurations", em `5_mobiwac.tex:185`. Mas o
qualificador **nao cobre o PCGrad**: pelo audit (`T4_audit_and_verdict.md:26-31`), a exclusao dele e
um resultado de **fiacao**, nao de configuracao (sob a torre dupla a torre privada treina em peso
unitario de qualquer jeito, entao o metodo colapsa para peso igual). Um resultado de fiacao e
invariante a configuracao. O Nash-MTL, o outro metodo nomeado, estava corretamente ligado
(`T4:37-39`).

**(C)** Minha recomendacao, **nao aplicada**: remover `PCGrad \cite{yu2020pcgrad}, ` da frase que
cita em `:183` e deixar o Nash-MTL carregar a evidencia nomeada. Sua instrucao prevalece; o nome
fica ate voce decidir.

> DECISAO: __________________________________________________

---

## BLOCO 1b — decisoes NOVAS, abertas pelas tres revisoes de dominio

> Rodei as tres personas que faltavam (08 fidelidade de traducao, 10 MTL, 11 POI/mobilidade). Elas
> acharam **quatro blockers**, todos defeitos reais no PDF entregue, e todos ja corrigidos. Tres das
> correcoes precisam de uma decisao sua, porque mexem em texto publicado ou em uma alegacao central.

### 1b.1 A atribuicao do ganho: "shared trunk" nao se sustenta (persona 10, BLOCKER)

**O achado.** O Cap. 5 dizia que o ganho de categoria vem de "a stronger shared trunk" e fechava com
"We report this attribution as a finding, not a hypothesis". O controle de congelamento
(`W6_ENCODER_ISOLATION.md:20-24`) remove o **treino** da regiao, entao ele elimina a hipotese
"regiao ensina categoria", mas **nao localiza** o componente: nao remove o encoder proprio da
categoria, os FFNs por fluxo, nem a profundidade extra. E o repo tem o braco que testa o trunk
direto, com resposta oposta: `F50_T1_5_CROSSATTN_ABSORPTION.md:19-20` mede, na Florida, cat F1
68,36 ± 0,74 com cross-attention LIGADA contra 68,32 ± 0,67 DESLIGADA, delta **-0,04 ± 0,13**,
estatisticamente indistinguivel (:37); a :80 chama o rotulo "shared" de impropio na FL, 95% cat-only
por massa de gradiente. Verifiquei as duas citacoes nos arquivos.

**O que fiz.** Mantive o resultado **negativo** como achado (o ganho nao e transferencia entre
tarefas: isso o controle estabelece). Rebaixei a atribuicao **positiva** de "o trunk" para "a
arquitetura conjunta", declarei que o controle nao localiza o componente, e **divulguei a ablacao
discordante** no proprio capitulo, com o escopo de um dataset dito. Sincronizei os dois pontos do
Cap. 6.

**(C) O que preciso de voce.** Duas coisas: (i) o Cap. 5 esta sob o regime de errata e isto reescreve
uma frase de **interpretacao**, entao precisa de linha no Apendice B, que **nao escrevi** esperando
sua decisao de redacao; (ii) se preferir citar o empate do cascade (`CSLSL_CASCADE.md:19`, que corta
o canal simetrico e empata dentro de 0,02 pp) em vez da ablacao F50, ou rodar a ablacao nos outros
datasets antes de divulgar, eu reestruturo. O que **nao** deixei foi a alegacao de pe como
estabelecida com o repo guardando um teste nulo daquele exato mecanismo.
> DECISAO: __________________________________________________

### 1b.2 O piso de Markov esta acima dos baselines externos (persona 11, BLOCKER)

**O achado.** O Cap. 5 chama os sistemas externos de "the per-task state of the art, on our data" e,
vinte paginas depois, imprime um piso de Markov de primeira ordem **acima** do HMT-GRN nos seis
datasets, acima do ReHDM em tres e do STAN em quatro. Os dois conjuntos de numeros estavam certos e
rastreaveis; nada os ligava. Um parecerista le o par como sinal de que as reimplementacoes estao
mal treinadas e **desconta a comparacao externa inteira**, inclusive as partes que favorecem voce.

**O que fiz.** Adicionei um paragrafo que declara a comparacao e a explica como propriedade do
**protocolo**, nao veredito sobre aqueles sistemas: nossas janelas andam uma visita por vez, entao a
persistencia de regiao e forte e uma tabela de transicao a consome direto, enquanto os externos
preveem **lugar** e chegam a regiao pelo mapa lugar-regiao, descartando esse sinal. O capitulo agora
trata o **piso** como a referencia a bater. Contagens 6/3/4 recalculadas por mim da tabela do proprio
capitulo e dos JSONs de piso.

**(C) Preciso de voce:** so a leitura. Se discordar do enquadramento (protocolo, nao qualidade dos
sistemas), me diga e eu reescrevo. **E um item herdado para corrigir a parte:**
`docs/results/closing_data/MACS_BOARD_RESULTS.md:47` ainda afirma "HMT reg clears the Markov floor",
verdadeiro contra o piso antigo nao-sobreposto (AL 0,4701) e nunca revisitado quando o piso foi
recalculado sob a janela stride-1 (AL 0,6226). O capitulo herdou a inconsistencia; o registro interno
deveria ser corrigido.
> DECISAO: __________________________________________________

### 1b.3 O custo do Nash-MTL: corrigir ou so declarar? (persona 10, MAJOR)

**O achado.** A frase publicada "requires only two matrix-vector products per iteration" nao tem
apoio no artigo (a expressao nao ocorre nele) e **subestima** o custo: as duas implementacoes rodam
um procedimento concavo-convexo iterativo, 20 passos por padrao, cada um uma resolucao convexa, alem
de um backward por tarefa. Voce ja tinha decidido REPORTED, NOT CORRECTED, e **mantive** sua decisao.
O ponto novo da persona e a **assimetria**: o Apendice B corrige a clausula **vizinha da mesma
frase** (sinais de gradiente), entao o silencio sobre esta metade convida a pergunta.

**O que fiz.** A secao "deliberadamente preservados" do Apendice B agora **nomeia** a preservacao,
passando de dois para tres elementos, e diz que a correcao correria **a seu favor**.

**(C)** Manter assim (recomendo), ou mover para a tabela de errata e corrigir de fato?
> DECISAO: __________________________________________________

### 1b.4 O determinismo da categoria agora esta medido nos CINCO estados (persona 11)

Nao e decisao, e reforco: a persona 11 refez a medicao do item 2.4 em todos os estados, nao so no
Alabama. **284 a 365 valores `fclass` distintos por estado, nenhum mapeando para mais de uma das 7
classes-alvo, em Alabama, Arizona, Florida, California e Texas.** Ou seja: nao e artefato de um
dataset. A frase de escopo do Cap. 4 continua sendo a pendencia (aviso ao co-autor primeiro).

---

## BLOCO 2b — decisoes herdadas do `DECISOES_PENDENTES_ptBR.md` que continuam abertas

> Auditei os 12 itens daquele documento contra o fonte de hoje. **Tres continuam abertos** e estao
> abaixo. **Seis foram resolvidos** nas rodadas seguintes e estao registrados no fim desta secao,
> para voce nao reabrir a esmo. Os outros tres ja aparecem nos Blocos 1 e 3 deste documento
> (titulo, Resumo/Abstract, folha de aprovacao, figuras).

### 2b.1 Apendice A — manter ou remover a secao do BRACIS (era 2.3, e o item mais consequente)

**Estado:** aberto, e nao mexi. `apx_a_contributions.tex:111` ainda tem
`\section{An earlier unpublished iteration}`.

**O trade-off, que continua valendo.** A §A.1 (tooling) ja lidera, que e o que voce pediu. Mas a
§A.2 e o **dispositivo de contencao** (AGENT_GUARDRAILS C4, NORTH_STAR §3): o documento revela que
houve uma iteracao anterior cuja alegacao central estava errada. Se um membro da banca descobrir a
submissao rejeitada e o documento nao a tratar, **le como ocultacao**. Remover e uma decisao de
risco, nao de estilo, e por isso nao removi.

**Uma coisa nova desde entao, que reforca manter.** O simulador de banca desta rodada faz exatamente
a pergunta que a §A.2 antecipa. Com ela, a resposta ja esta no documento.

**DECISAO:** (a) manter as duas secoes (recomendo), ou (b) remover a §A.2 e eu ajusto o
NORTH_STAR §3 registrando a mudanca de politica.
> DECISAO: __________________________________________________

### 2b.2 O nome "MTLnet" e a grafia (era 3.7)

**Estado:** parcialmente resolvido, com um resto real. O Cap. 3 hoje tem **1** ocorrencia de
"MTLnet" (o prefacio ja nomeia o modelo, entao a costura que voce apontou esta fechada). O que
**nao** esta resolvido e a grafia: o Cap. 4 escreve **MTLNet 46 vezes** contra 4 de "MTLnet",
enquanto o frame e o Cap. 5 usam "MTLnet".

**Por que nao padronizei sozinho.** O Cap. 4 e **texto publicado**. Trocar 46 ocorrencias e uma
alteracao de texto publicado que exige linha de errata, e o proprio artigo CoUrb usa "MTLNet". A
regra do repo (`4_courb.tex:84`) ja declara que o capitulo preserva a grafia do publicado.

**DECISAO:** (a) deixar como esta — o Cap. 4 preserva a grafia publicada, e a nota em `:84` ja
explica ao leitor (recomendo); ou (b) padronizar para "MTLnet" no Cap. 4 e eu abro a linha de
errata no Apendice B. Qual e a canonica para o GLOSSARY?
> DECISAO: __________________________________________________

### 2b.3 Movimentos opcionais de excelencia (era 3.10)

**Estado:** nenhum dos tres existe. Verifiquei: nao ha tabela contribuicoes→alegacoes no §1.6, nao
ha tabela consolidada de resultados no Cap. 6, nao ha apendice de artefatos.

Sao adicoes de **frame** (nao tocam resultado), na lente de premio SBC-CTD: (a) tabela
contribuicao→alegacao no §1.6; (b) tabela consolidada cross-chapter no Cap. 6; (c) apendice de
reprodutibilidade (codigo, seeds, configs). **Observacao:** o (c) ficou mais facil agora, porque o
Apendice D novo ja estabelece o padrao de citar script + arquivo de saida para cada numero.

**DECISAO:** quer algum dos tres? (cada um e ~1 pagina)
> DECISAO: __________________________________________________

### Resolvidos desde aquele documento (registrado para nao reabrir)

| Item de la | Estado hoje |
|---|---|
| 3.1 Wilcoxon x t pareado | **RESOLVIDO.** Cap. 2 (`:497-503`) e Cap. 5 (`:412`) agora concordam: t pareado nas medias por repeticao **mais** Wilcoxon nos folds individuais, ambos reportados, com o piso do p exato do Wilcoxon explicado. O desvio do pre-registro esta declarado. |
| 3.2 CV usuario-disjunta: documento todo ou so Cap. 5? | **RESOLVIDO** (REV-006). O Cap. 2 agora escopa explicitamente: os testes "license verbs in Chapter 5 alone" (`:495-496`). |
| 3.3 Pre-registro da nao-inferioridade explicito | **RESOLVIDO e reforcado.** `5_mobiwac.tex:412` declara o plano escrito, fixado antes de ler resultado, a margem de dois pontos, **e** que ele nao cobria superioridade de regiao (os 4 ganhos sao secundarios). Mais honesto que o pedido original. |
| 3.4 Vintage 2009-2011 | **APLICADO** na rodada 3. |
| 3.5 Ponte "next-POI" | **APLICADO** na rodada 3, e o Cap. 3 recebeu nota de rodape adicional nesta rodada (REV-010). |
| 3.6 Contradicao class-weighted CE | **RESOLVIDO** nesta rodada. `2_fundamentals.tex:456` agora diz "plain unweighted cross-entropy; class weighting, tested there on both outputs, lowered..." — concorda com o Cap. 5. |

---

## BLOCO 3 — assinaturas e itens adiados

### 3.1 Os 27 marcadores `[NEEDS SIGN-OFF]`

Voce pediu a lista. Sao 27 marcadores em 9 arquivos, todos comentarios LaTeX (**nenhum renderiza**,
entao nao ha sujeira no PDF). O risco nao e visual: e que o **Apendice C afirma** que o autor leu e
aprovou cada palavra, enquanto o proprio apendice esta marcado como nao aprovado. Voce ja decidiu
manter o Apendice C como esta, o que torna esta lista o caminho para tornar a afirmacao verdadeira.

| Arquivo | Qtd | O que e |
|---|---|---|
| `0_main.tex` | 6 | Resumo e Abstract: **par de paridade**, incluindo as mudancas de unidade inferencial desta rodada |
| `chapters/6_conclusion.tex` | 5 | Escopos de alegacao: joint model qualificado, largura 64→192, California completa, parametro escopado, convencao 64,51 |
| `chapters/5_mobiwac.tex` | 4 | Prefacio, recap, figura restaurada, **mais a atribuicao do trunk rebaixada (item 1b.1)** |
| `chapters/apx_a_contributions.tex` | 4 | Apendice inteiro, mais as tres correcoes desta rodada |
| `chapters/1_introduction.tex` | 2 | Correcao de gate L3, unidade inferencial |
| `chapters/2_fundamentals.tex` | 2 | Escopo dos 93% do Song, de-duplicacao L3 |
| `chapters/apx_b_errata.tex` | 2 | Apendice inteiro, **mais a preservacao do custo do Nash declarada (item 1b.3)** |
| `chapters/apx_c_ai_disclosure.tex` | 1 | Apendice inteiro |
| `chapters/apx_d_ceiling.tex` | 1 | Apendice novo (o teto de autocorrelacao, item 3.4) |
| **TOTAL** | **27** | contagem medida em 9 arquivos, 2026-07-26 |

**Regra que nao da para contornar:** os 6 do `0_main.tex` sao **um par**. Resumo e Abstract carregam
as mesmas alegacoes, e aprovar um sem o outro quebra a paridade. Leia os dois lado a lado.

**Um termo novo precisa entrar no GLOSSARY antes de virar canonico:** usei **"modelos ajustados"**
como equivalente PT de "fitted models" no Resumo. O GLOSSARY §6 nao tem essa entrada, e a regra e
fail-closed (o termo entra no registro **antes** de entrar no texto). Confirma o termo?

> DECISAO: __________________________________________________

### 3.2 Figura 2 do Cap. 4: rotulos em portugues (REV-022)

**(A)** A figura da arquitetura na p. 48 tem `Encoder Espacial`, `Encoder Temporal`, `Encoder
Categorico`, `Coordenadas (lat, lon)`, `Timestamps (hora, dia)`, `Categorias (POI graph)` dentro de
um capitulo em ingles, sob legenda em ingles.

**(B)** Duas personas classificaram como bloqueador visual.

**(C)** **Bloqueado por falta do fonte.** As Figuras 1, 2 e 3 existem so como PNG achatado; nao ha
`.drawio`, `.svg` nem `.py` em lugar nenhum sob `articles/dissertacao/`. Preciso de uma de duas
coisas: o arquivo fonte (com os autores do CoUrb, provavelmente com o Tarik), ou autorizacao para
**recriar** a figura do zero. Recriar levanta questao de fidelidade, porque a figura pertence a um
artigo publicado co-autorado, entao nao faco sozinho.

**Ja feito:** o rotulo do eixo da Figura 6 dizia "Score (0-1)" para uma silhueta definida em
[-1, 1]; corrigido e a figura foi regerada (o resto do PDF e byte-identico).

> DECISAO: __________________________________________________

### 3.3 Resumo e Abstract: tamanho (REV-018)

**(A)** Abstract 429 palavras, Resumo 505. No build de defesa o Resumo enche a p. 3 e deixa **duas
palavras-chave sozinhas** numa p. 4 praticamente em branco (61 caracteres na pagina inteira).

**(B)** **Nenhuma norma esta sendo violada** — verifiquei: nem o `UFV_COMPLIANCE.md` nem o Manual
04/2026 impoem limite de palavras ou paginas; a unica regra de palavra e "uma palavra por linha" nas
palavras-chave. E polimento, nao compliance.

**(C)** Sua instrucao foi deixar por ultimo, depois que o texto assentar, e concordo. Quando quiser,
ha duas rotas: (i) comprimir os dois em paridade, o que mexe em alegacao e portanto e seu; ou (ii)
uma alternativa **puramente mecanica**, um ajuste de `\clearpage`/espacamento que tira a pagina
quase-branca sem tocar em uma palavra. A (ii) da para fazer agora se quiser resolver o efeito
visual e adiar o resto.

> DECISAO: __________________________________________________

### 3.4 O teto de autocorrelacao — **RECONSTRUIDO nesta rodada** (REV-001)

**FEITO.** Voce pediu para reconstruir o teto, e ele esta reconstruido. O resultado mudou o texto do
Cap. 5, e para pior no sentido honesto: a alegacao anterior era mais forte do que a evidencia.

**O que estava errado.** O registro interno usa "teto de autocorrelacao" para duas coisas
diferentes, e o Cap. 5 herdou a confusao:

- **(a) o encoder de referencia limpo** — o que o `leak_sniff.py` de fato compara em codigo
  (`:63,:87`: sinaliza se `perstep > controle + margem`, margem 0,03). Na Florida: **0,4090**.
- **(b) o teto de autocorrelacao propriamente** — o que a categoria da ultima visita permite,
  sozinha. E propriedade da **sequencia de rotulos**, nao de encoder nenhum. Nunca foi medido.

O `RESCREEN.md:57` chama ~0,45 de "the autocorrelation ceiling"; o `:87` chama o controle limpo de
"the ceiling (~0,41)". Sao quantidades distintas, e o Cap. 5 dizia que um encoder limpo "define" o
teto em 0,41.

**O que eu medi.** Script novo: `scripts/embedding_eval/autocorrelation_ceiling.py`. Nao le nenhum
embedding. Le so a historia de categorias da janela de 9 visitas, com a mesma regra de derivacao do
input de treino (`src/data/inputs/next_region.py:132-146`) e o mesmo protocolo do probe
(GroupKFold(5) por usuario, macro-F1, media dos folds). Quatro preditores so-de-rotulo: persistencia,
one-hot da ultima categoria, contagens da janela, one-hots posicionais. O teto e o melhor deles.

| Dataset | Teto (rotulo so) | Piso (classe majoritaria) |
|---|---|---|
| Alabama | 0.2800 | 0,0727 |
| Arizona | 0.3232 | 0,0725 |
| Florida | 0.3617 | 0,0566 |
| California | 0.3242 | 0,0704 |
| Istanbul | 0.3016 | 0,0715 |

**As duas leituras, e elas apontam para lados diferentes:**

1. **A triagem em si nao muda.** Os vereditos dela sao relativos: desqualifica quem passa o
   encoder de referencia por mais que a margem de 3 pontos. O encoder de atencao desqualificado na
   Florida passa por **8,9 pontos**. A decisao nao depende de onde o teto esta.
2. **A leitura absoluta fica mais fraca do que o texto dizia.** Todos os encoders triados na
   Florida, **inclusive os limpos**, ficam acima do teto so-de-rotulo: 0,4090 e 0,4197 contra
   0,3617, uma folga de 4 a 6 pontos. **Isso nao e prova de vazamento** — um vetor por visita
   carrega legitimamente mais que a categoria anterior (o lugar, a vizinhanca no grafo, a hora), e
   qualquer um desses preve a proxima categoria sem informacao andando para tras no tempo. Mas
   significa que a triagem limita encoders **entre si**, nao contra um padrao absoluto.

**Aplicado:** o paragrafo dos quatro fundamentos do Cap. 5 agora faz a alegacao mais fraca e
correta (a triagem e relativa), cita o teto medido, e o **Apendice D** novo carrega a tabela e as
duas leituras. Nenhum numero de encoder mudou; todos continuam vindo do `leak_sniff_fl.csv` e do
`leak_sniff_resln_fl.csv`. Dois termos novos entraram no GLOSSARY (**label-only ceiling** e **clean
reference encoder**) exatamente para que nao voltem a ser trocados um pelo outro.

**Limites de cobertura, declarados no apendice:** (i) **Texas nao entra** — o
`output/check2hgi/texas/temp/` nao tem `checkin_graph.pt` nem `sequences_next.parquet` (so as
saidas de embedding do engine de design), entao o teto la exigiria re-rodar o pre-processamento;
(ii) **Istanbul** tem 196 de 29.816 lugares com mais de uma categoria (venues recategorizados ao
longo do tempo); usei a categoria modal, e descartar os ambiguos move o teto em menos de um
milesimo (0,3009 contra 0,3016).

**O que ainda NAO fecha, e por que eu nao rodei a sonda nos outros datasets.** Os outros dois
limites declarados no Cap. 5 continuam de pe e a sonda nao os resolve: ela e **linear** (e o
`RESCREEN.md:94` documenta um encoder que passou no gate linear e vazou sob modelo sequencial), e
mediria os **mesmos builds ancestrais**, nao a linhagem entregue. Rodar a sonda em AL/AZ/CA/IST
agora produziria numeros comparaveis ao teto — isso ficou possivel — mas a comparacao interessante
(a linhagem que gerou os resultados do Cap. 5) exige re-exportar aquela linhagem.
**DECISAO:** quer que eu rode a sonda nos quatro datasets que agora tem teto, aceitando que ela
mede os builds ancestrais? (o ambiente `leakprobe` esta pronto; e barato)
> DECISAO: __________________________________________________

### 3.5 Higiene do repositorio, herdada (REV-007)

Dois itens claim-neutros que a auditoria estatistica anterior deixou abertos e que nao toquei
porque estao dentro de registros de resultado:

1. **Falta um gerador.** A entrada §8 de 2026-07-18 imprime os ICs exatos que os capitulos carregam,
   e todos reproduzem a partir dos arrays commitados, entao os valores sao solidos. Mas nenhum
   script commitado emite aquela entrada: o gerador da rodada joint-best nao esta na arvore.
2. **Docstrings.** `superiority_wilcoxon.py` e `m1_stats_n20.py` ainda afirmam um registro que o
   protocolo nao contem, e o `stats_n20/RESULTS.md` §1b repete. Higiene, sem efeito em alegacao.

> DECISAO: __________________________________________________

---

## Notas de rodape uteis

- **Ordem de aplicacao**, se voce for reabrir algo: registro fail-closed (GLOSSARY) → governanca →
  texto de moldura → capitulos publicados sob errata → layout → build → revisao. A ordem importa: o
  layout depende das quebras de pagina que o texto move, e o Apendice B so fica correto se escrito
  **depois** das correcoes que ele declara.
- **Defeitos meus, declarados.** Quatro falhas na minha propria verificacao, todas encontradas pela
  revisao e nao por mim, todas corrigidas: (i) o teste de citacao indefinida reportou "0" enquanto
  quatro citacoes renderizavam como `(??)` nos dois PDFs, porque o grep era ancorado por linha e o
  LaTeX quebra avisos em varias linhas; (ii) o teste de pagina so-com-floats dependia de uma linha
  de log que o LaTeX nem sempre emite, e reportou "nenhuma" para um build cuja p. 71 era so floats;
  (iii) o commit `2f1cd5b3` registrou "no floats-only page" como verificado quando nao estava (a
  condicao foi de fato corrigida no commit seguinte, `e84b37c0`; deixei o commit como esta,
  porque reescrever historico esconderia o erro); (iv) o script imprimia uma linha para o build
  final mesmo quando so o de defesa tinha rodado, entao duas mensagens de commit citaram numeros do
  final que ainda nao tinham sido medidos. O verificador agora achata o log, le o `.blg`, mede
  paginas so-com-floats a partir do PDF e reporta so as variantes que aquela execucao construiu.
  A causa raiz das citacoes `(??)` eram arquivos `.aux` velhos **commitados** na raiz do `src/`, que
  o BibTeX le antes do `build/`; foram removidos e entraram no `.gitignore` com o motivo registrado.
- **O que nao esta pendente, embora pareca:** a discrepancia 87/83 x 89/84 de paginas era real e
  esta corrigida (agora 94/89 apos as correcoes, medido). As duas violacoes de margem foram
  eliminadas. As tabelas do Cap. 5 que estavam em 8 pt agora renderizam em 11,96 pt, tamanho de
  corpo.
