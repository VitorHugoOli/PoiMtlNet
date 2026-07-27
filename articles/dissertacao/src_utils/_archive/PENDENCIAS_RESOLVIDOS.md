# Itens RESOLVIDOS do PENDENCIAS.md — log de encerramento

Movidos para fora do `PENDENCIAS.md` em 2026-07-27, a pedido do autor, para que aquele arquivo carregue
**apenas o que ainda depende dele**. Nada foi apagado: cada item abaixo mantem a decisao, o que foi
aplicado e o commit. Serve para nao reabrir o que ja foi fechado.

| Onde vive o que ainda esta aberto |
|---|
| `src_utils/PENDENCIAS.md` — o registro vivo |
| `src_utils/codex_reviewer.md` — a revisao externa, com veredito por achado |

---

### ~~2.1 Etica e governanca de dados — **o item mais exposto da lista** (REV-026)~~ — RESOLVIDO 2026-07-27

**Feito, mas precisa da sua leitura.** O Apendice E ("Data ethics and governance", ~790 palavras)
esta escrito e no build. Todas as licencas foram reabertas na fonte nesta sessao, nao herdadas da nota: Figshare devolve
**CC0** para o DOI `10.6084/m9.figshare.22126586.v2`, o Hugging Face devolve **apache-2.0**, e a API do GitHub devolve
SPDX **Apache-2.0**.

Duas coisas que o apendice diz porque o codigo diz, e que voce deve querer conferir:

1. Uma busca no repositorio inteiro por `jitter|perturb|laplace|anonym|deidentif|mask|obfusc` nao retorna **nenhum** hit
   em nenhum caminho de ETL. O apendice afirma que **nao ha de-identificacao aplicada**, em vez de sugerir uma protecao
   que nao existe.
2. O endereco original do Gowalla nao da mais 403: hoje ele **redireciona (301) para um dominio comercial sem relacao**.
   Os termos originais nao estao apenas nao lidos, sumiram.

Sobre o comite de etica, respondendo a sua pergunta 1: a dissertacao de 2024 do mesmo orientador foi extraida inteira
(96 paginas) e buscada por `comite|CAAE|Plataforma Brasil|CEP|IRB|ethics
approval|Resolucao N` — **zero hits**, contra dois hits em "Ethical Statement" que provam que a busca estava lendo texto
real. Ela tem um §2.6 sobre privacidade de localizacao que diz que latitude e longitude ficaram sem mascara. O apendice
registra isso como precedente, registra a sua posicao de que a revisao nao era exigida, e **nao afirma aprovacao nem
isencao**.

**O que preciso de voce:** ler o Apendice E e assinar. Ele faz afirmacoes institucionais em seu nome. Esta marcado
`[NEEDS SIGN-OFF: AUTHOR]`. Commit `9e2b5157`.

> DECISAO: Ficou bem legal, é eu aprovo esse appendix com algumas alterações. Primeramente é mais critical: O gowalla
> orignal, salvo engano é o: https://snap.stanford.edu/data/loc-gowalla.html, o link do figshare que não está
> funcionando
> é acesseivvel por: https://web.archive.org/web/20220709062539/http://www.yongliu.org/datasets.html. Assim, vamos
> refazer o texto do gowalla e ser mais susinto e direto, não precisamos falar de coisas que são efemeras como o fato do
> site não está funcionando. O segundo paragrafo desse appendix que tmb está relacionado ao gowalla, tmb está um pouco
> confuso. Voce pode remover por completo o E.3, não precisamos dessa justificativa por hora.

---

### ~~2.3 Conflito: Nash x NORTH_STAR (REV-005)~~ — RESOLVIDO 2026-07-27

Sua opcao A aplicada. O `NORTH_STAR.md` marca a instrucao "Nash-MTL caveat as in Ch.3" como **REVOGADA**
com a sua decisao, a data e o escopo que ela abre mao (so a alegacao de preferencia de otimizador).
Verificado que nenhum capitulo de moldura amplifica a preferencia. Commit `e8974e81`.

---

### ~~2.4 PCGrad e a largura do sweep (REV-011)~~ — RESOLVIDO 2026-07-27

Seu apontamento para `docs/studies/` rendeu: o `T4_full_screen.json` tem **19 balanceadores** em dois
estados, nao dois. O Cap. 5 agora declara o numero e nomeia as **duas** excecoes (nash_mtl +0,68 e
scale_norm +0,19 em categoria no AL) com o que cada uma abre mao, em vez de "nenhum dos que testamos".
PCGrad fica nomeado, como voce pediu. Commits `e8974e81`, `bff15ed3`.

---

### ~~1b.1 A atribuicao do ganho: "shared trunk" nao se sustenta (persona 10, BLOCKER)~~ — RESOLVIDO 2026-07-27

**Aplicado nos dois capitulos, e a sua desconfianca sobre o experimento estava certa.**

Voce pediu para auditar o F50 antes de cita-lo. Auditei, e ele nao sustenta o que a tese dizia, por dois motivos
independentes que estao no proprio registro:

1. O `F50_T1_5_CROSSATTN_ABSORPTION.md:229` chama o **proprio** nulo de "misleading" e de "hidden compensation effect,
   not a true null contribution". O F49 companheiro mede a contribuicao arquitetural em **-16,16 pp**: o nulo aparece
   porque o encoder de categoria absorve o deficit.
2. A configuracao ablacionada nao e a que foi entregue. O F50 rodou `check2hgi/florida` em bs2048 (abril); o board
   entregue roda `check2hgi_dk_ovl` em bs8192 (`catx_v17_n20/*.json`, campo
   `rundir`). E o mecanismo depende do prior `alpha*log_T`, que o proprio Cap. 5 diz que **nossos modelos nao usam**.

Os dois capitulos agora mantem o valor e **estreitam a inferencia**: a ablacao nao e oferecida como prova de que o trunk
nao contribui. A atribuicao continua retida, e continua apoiada no controle de congelamento e no controle de capacidade,
que este achado nao toca. Commit `06b64cab`.

**Nao rodei de novo, de proposito.** Uma versao citavel exige o engine entregue, o batch entregue, sem o prior, e seeds
suficientes: um treino por seed por fold por braco na GPU, nao uma sonda. Isso e uma pergunta de pesquisa, nao uma
clausula de hedge.

**O que preciso de voce:** (a) o texto reescrito entra como esta, ou (b) voce prefere remover a clausula inteira, ja que
a atribuicao ja fica retida pelos outros dois controles? E: isso merece linha de errata no Apendice B? E texto de
moldura, nao publicado, entao a minha leitura e que nao precisa — mas reescreve uma frase de interpretacao, e a decisao
e sua.

> DECISAO: Vamos lá eu acredito que valha a pena remover essa clausula, para não gerar confusão no leitor, e em
> sequencia documentar esse problema para que em um estudo externo façamos essa analise até agosto, para que caso seja
> objeto de pergunta já tenhamos resposta, esse estudo por hora pode ser documetno em um arquivo md na pasta
> articles/[mobiwac]/science. Mas, como eu disse por hora eu acredito que valha remover isso do cap. 5 e do texto
> original: articles/[mobiwac]. Eu reiterio mais uma vez que ainda podemos alterar no original e evitar erratas, pq o
> mobiwac, aidna esta em etapa de revisão.

---

### ~~1b.2 O piso de Markov esta acima dos baselines externos (persona 11, BLOCKER)~~ — RESOLVIDO 2026-07-27

**Aplicado.** O paragrafo agora declara a assimetria de protocolo sistema por sistema em vez de oferecer uma explicacao
causal unica, e diz explicitamente que nenhum dos fatos estabelece por que o piso fica acima. As contagens verificadas
(piso acima do HMT-GRN em 6/6, do ReHDM em 3, do STAN em 4) e os 22,4 por cento de revisita no Alabama ficam como estao.

Corrigi tambem um erro **meu**, de ontem: eu tinha escrito que os tres sistemas externos preveem um lugar e sao lidos no
nivel de regiao, enquanto o mesmo capitulo chama o HMT-GRN de *region-native*
em `:418`, `:622` e `:755`. Commit `ff96dcaf`. O registro interno estava desatualizado no mesmo ponto e foi corrigido em
`f978b16b`.

---

### ~~2b.1 Apendice A — manter ou remover a secao do BRACIS (era 2.3, e o item mais

consequente)~~ — RESOLVIDO 2026-07-27

Aplicado: a secao A.2 foi removida, e o sweep que voce pediu encontrou e reconciliou as alegacoes dependentes. O
`NORTH_STAR §3`, que mandava manter o dispositivo de contencao, ficou marcado como superado pela sua decisao com a data,
em vez de apagado. A sigla BRACIS saiu da lista de abreviaturas, porque A.2 era o unico texto que a usava. Commit
`21124a8c`.

---

### ~~2b.2 O nome "MTLnet" e a grafia (era 3.7)~~ — RESOLVIDO 2026-07-27

Aplicado: 26 sites normalizados para `MTLnet` no Cap. 4, com a errata no Apendice B. A autoridade e o `GLOSSARY.md:41`,
ja que o artigo do CBIC nunca nomeia o modelo em prosa (so um nome de arquivo de figura e a URL do repositorio). O
`ST-MTLNet` mantem o N maiusculo: e um nome registrado a parte, e a expansao publicada *Spatial-Temporal MTLNet* tambem
fica. A frase do `:84` que dizia que o capitulo preservava a grafia publicada virou falsa com a mudanca e foi removida.
Commit `ff96dcaf`.

---

### ~~3.4 O teto de autocorrelacao — **RECONSTRUIDO nesta rodada** (REV-001)~~ — RESOLVIDO 2026-07-27

**Aplicado, e o nome mudou.** A revisao do codex tem razao: chamar aquilo de "teto" afirma mais do que a analise mostra.
Nao e um limite superior, e o **melhor de quatro preditores especificados**
sobre o historico de rotulos, e um modelo diferente pode supera-lo com o mesmo historico.

O termo agora e **label-history benchmark** no Cap. 5, no Apendice D e no `GLOSSARY.md`, com as formulacoes antigas
banidas no registro. A palavra "teto" continua correta para o *dedicated single-task ceiling*, que e o escore de um
modelo treinado de verdade.

O Apendice D foi reescrito por causa da sua objecao de leitura. Medi antes de reescrever: o texto tinha as **frases mais
curtas do documento** (media 21,2 palavras contra 30,0 na conclusao), entao o tamanho nao era o defeito. Os defeitos
medidos eram (i) colisao de conceitos, "ceiling" 8 vezes contra "reference" 5 em 508 palavras, para um apendice cuja
funcao e justamente separar as duas coisas, e (ii) dependencia externa, "screen" 9 vezes mas definido so no Cap. 5.
Commits `ff96dcaf`
e `21124a8c`.

**O que preciso de voce:** ler o Apendice D novo e dizer se ele agora se sustenta sozinho. Se continuar confuso, a
alternativa e dobra-lo em um paragrafo do Cap. 5, e eu faco.

> DECISAO: __________________________________________________

---

### ~~3.5 Higiene do repositorio, herdada (REV-007)~~ — RESOLVIDO 2026-07-27

Aplicado. (1) Os docstrings do `superiority_wilcoxon.py` e do `m1_stats_n20.py` afirmavam um registro que o
`STATISTICAL_PROTOCOL.md` nao contem: o protocolo fixa a tarefa de regiao em **nao-inferioridade** (`:44`, `:213-215`),
nunca em superioridade. Cada um agora separa o que e registrado (superioridade de categoria) do que e **post-hoc**
(superioridade de regiao), citando as linhas do protocolo. As alegacoes continuam reportadas; so o footing foi
corrigido. Isso ja estava no log como D-4 e nunca tinha sido feito. (2) O `MACS_BOARD_RESULTS.md:47` dizia que o HMT-GRN
supera o piso de Markov. Verifiquei contra os dois pisos: no piso atual o piso esta acima do HMT-GRN em 5 de 5 estados
americanos, e **no piso antigo ja estava acima em 3 de 5** (CA, TX, FL). Ou seja, a alegacao sem qualificacao nunca foi
verdadeira. Os dois pisos agora aparecem tabelados com a janela e a fonte de cada um. Commit `f978b16b`.

## Notas de rodape uteis

- **Ordem de aplicacao**, se voce for reabrir algo: registro fail-closed (GLOSSARY) → governanca → texto de moldura →
  capitulos publicados sob errata → layout → build → revisao. A ordem importa: o layout depende das quebras de pagina
  que o texto move, e o Apendice B so fica correto se escrito **depois** das correcoes que ele declara.
- **Defeitos meus, declarados.** Quatro falhas na minha propria verificacao, todas encontradas pela revisao e nao por
  mim, todas corrigidas: (i) o teste de citacao indefinida reportou "0" enquanto quatro citacoes renderizavam como
  `(??)` nos dois PDFs, porque o grep era ancorado por linha e o LaTeX quebra avisos em varias linhas; (ii) o teste de
  pagina so-com-floats dependia de uma linha de log que o LaTeX nem sempre emite, e reportou "nenhuma" para um build
  cuja p. 71 era so floats; (iii) o commit `2f1cd5b3` registrou "no floats-only page" como verificado quando nao estava
  (a condicao foi de fato corrigida no commit seguinte, `e84b37c0`; deixei o commit como esta, porque reescrever
  historico esconderia o erro); (iv) o script imprimia uma linha para o build final mesmo quando so o de defesa tinha
  rodado, entao duas mensagens de commit citaram numeros do final que ainda nao tinham sido medidos. O verificador agora
  achata o log, le o `.blg`, mede paginas so-com-floats a partir do PDF e reporta so as variantes que aquela execucao
  construiu. A causa raiz das citacoes `(??)` eram arquivos `.aux` velhos **commitados** na raiz do `src/`, que o BibTeX
  le antes do `build/`; foram removidos e entraram no `.gitignore` com o motivo registrado.
- **O que nao esta pendente, embora pareca:** a discrepancia 87/83 x 89/84 de paginas era real e esta corrigida (agora
  94/89 apos as correcoes, medido). As duas violacoes de margem foram eliminadas. As tabelas do Cap. 5 que estavam em 8
  pt agora renderizam em 11,96 pt, tamanho de corpo.

---

## Encerrados na rodada de 2026-07-27 (segunda leva)

### ~~Outros pontos: erratas do MobiWac durante a revisao~~ — APLICADO 2026-07-27

Sua instrucao foi seguida: o artigo esta em revisao, entao correcao menor **nao vira errata** — aplico
no texto original tambem e os dois ficam identicos. Classifiquei as quatro linhas da tabela de errata do
Cap. 5 contra o `[mobiwac]/src/`:

| Correcao | Destino | Por que |
|---|---|---|
| frase dos balanceadores | **os dois textos** (`02_related.tex`) | menor: troca uma alegacao vaga pelo numero real |
| terceira limitacao | **os dois textos** (`07_discussion.tex`) | menor: declara uma consequencia que o artigo ja divulga |
| paragrafo de integridade | **fica errata** | o quarto fundamento cita o Apendice D, que nao cabe no artigo |
| controle de congelamento | **fica errata** | cita a tabela de resultados da dissertacao |

Detalhes que exigiram cuidado: o rotulo `sec:mobiwac:setup-windows` da dissertacao **nao existe** no
artigo, foi remapeado para o `sec:setup-windows` dele, e "this chapter's claims" virou "the paper's
claims". Artigo reconstruido: **9 pp, 0 referencias indefinidas, 0 citacoes indefinidas, 0 erros**.
Registrado no `[mobiwac]/ERRATA.md`. A frase de abertura do Apendice B foi corrigida de quatro para
duas e agora **explica a politica ao leitor**. Seu trabalho nao commitado de 25/07 em `[mobiwac]/` foi
preservado (conferi os mtimes antes de editar).

---

## BLOCO 2 — exposicao cientifica real

---

### ~~1b.3 O custo do Nash-MTL~~ — CORRIGIDO 2026-07-27, um so padrao

Sua razao foi aceita e aplicada: um padrao unico para erro factual. A clausula publicada "requires only
two matrix-vector products per iteration" saiu da frase reproduzida no Cap. 3, com nota de rodape
explicando, e entrou na **tabela de errata do CBIC**, ao lado da correcao de escala de gradiente que fica
na mesma frase publicada. A secao "deliberadamente preservados" caiu de tres para dois elementos.
Isto **supera** a decisao anterior de so declarar. A correcao **aumenta** o custo de um metodo que o
capitulo usou, ou seja, corre contra o interesse da propria dissertacao — e portanto conservadora alem de
consistente. Commit `d1911c0a`.

---

### ~~3.3 Resumo e Abstract: tamanho (REV-018)~~ — **A PAGINA FECHOU** 2026-07-27

Voce escolheu a rota (i), cortar 60 a 80 palavras. Reportei **36 em paridade** (Resumo 565 -> 529,
Abstract 485 -> 452): sairam as duas glosas parenteticas da selecao \emph{joint-best} e a frase de
motivacao foi comprimida.

> **[CORRECAO 2026-07-27]** Aquele "36" nao era so compressao. A edicao **apagou por acidente a
> clausula de abertura de quatro frases**, e a pagina fechou em parte por isso. Restauradas
> (`1bf9a227`), a contagem sobe para 542 / 466, o que separa as duas coisas: **compressao genuina de
> glosa 23 (PT) / 19 (EN)**, **clausulas apagadas 13 (PT) / 14 (EN)**. O documento voltou a 104 pp e a
> p. 4 quase em branco voltou com ele. Ver `PENDENCIAS.md` BLOCO 0c. Cada numero, nome de teste e token de alegacao foi verificado presente nos dois idiomas.

**E foi suficiente: a pagina quase em branco acabou.** 104 -> **103 pp**; o Resumo e suas palavras-chave
dividem a p. 3, o Abstract fica com a p. 4, e a p. 4 saiu inteiramente da lista de paginas com pouco
texto. Nao precisou dos 60 a 80 completos.

**Um defeito meu, e o diagnostico que eu escrevi sobre ele tambem estava errado.** Trocar o
`\needspace` por um `minipage` foi certo, mas o motivo que eu registrei era falso e uma auditoria pegou.
Re-medido nos dois mecanismos, com o Resumo em 529 palavras naquele momento (hoje 544, apos a
restauracao das clausulas) e a macro **corretamente escapada**:

| Mecanismo | Paginas | Onde fica o bloco |
|---|---|---|
| `\needspace{7\onelineskip}` | 104 | **inteiro** na p. 4 (21 palavras, rotulo incluido) |
| `minipage` | **103** | na p. 3, junto do Resumo — a pagina fecha |

Ou seja: o `needspace` **funciona** para manter rotulo e palavras-chave juntos. O que ele nao faz e
**puxar** o bloco para a pagina anterior, porque reserva espaco para as *linhas* seguintes; quando o
bloco nao cabe, ele move o bloco em vez de encaixa-lo. Um `minipage` e uma caixa, entao encaixa.

A afirmacao anterior de que o `needspace` "nunca funcionou em nenhum valor" veio de **um bug meu de
escape**: um regex escreveu `\needspace{N\\onelineskip}` com barra dupla, e as tentativas seguintes
(8, 9, 10) usaram um padrao de barra simples que **nao casava** com a forma corrompida — entao aqueles
tres testes nunca foram aplicados e devolveram resultados identicos byte a byte que eu li como
evidencia. O `minipage` fica (mede melhor), mas por este motivo, nao pelo que eu havia escrito.

**O que ainda e seu:** voce pediu revisores sobre o Resumo e o Abstract para avaliar qualidade e
excelencia contra os exemplares. Vale rodar a persona 15 com esse escopo especifico na proxima rodada,
com os textos ja no tamanho final.

> DECISAO (rodar a persona 15 sobre o par Resumo/Abstract?): __________________________________
