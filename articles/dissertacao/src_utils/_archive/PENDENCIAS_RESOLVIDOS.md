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

---

# Round 6 (2026-07-28) — o §1 inteiro do PENDENCIAS, movido com os commits

**Movido em 2026-07-29 a pedido do autor** (*"parse what has been complete to the
`_archive/PENDENCIAS_RESOLVIDOS.md`, so this file be clean as possible with only the decisions and
important points that I need to take"*). O que esta abaixo saiu de `src_utils/PENDENCIAS.md` §1
palavra por palavra, **com os 19 hashes de commit intactos**, para que a trilha de auditoria
sobreviva a mudanca. Nada foi reescrito, resumido ou reclassificado.

**Como conferir que os hashes ainda resolvem**, do repo root:

```bash
cd /Users/vitor/Desktop/mestrado/ingred
for h in 18b817d9 28097d93 29c7629c 2d117c7a 40ed8e7b 4b609643 4e84cf7a 519de348 6d780b58 \
         6ee23ca7 7a91b720 a7ab2eaa a880632b a8865214 ba90aa6d c6e62c62 e9370222 ec1cea0d fecc7fb1; do
  git cat-file -e "$h^{commit}" 2>/dev/null || echo "MISSING $h"
done; echo "checked 19"
# EXPECT: contains=checked 19
```

Rodado em 2026-07-29: os 19 resolvem, nenhum `MISSING`.

**Uma correcao ao §1 que ele carregava.** O §1 abria com uma tabela de estado do build medida em
`29c7629c` (108/105/109 paginas, `make check` exit 0). Essa tabela ficou no `PENDENCIAS.md`, que e
onde um numero vivo pertence, e foi remedida na rodada 7; o texto abaixo e o registro do que fechou,
nao do estado atual.

## Encerrados na rodada de 2026-07-28 (round 6)

### 1.1 O bloqueador que ninguem viu: o fonte nao compilava

**Fechado em `ba90aa6d`.** De `6d780b58` a `a880632b` a chave de abertura do grupo `{\small ...}` em
`tables/frame/bib_errata.tex` estava ausente e a de fechamento sobreviveu. Todo build morria com
`! Extra }, or forgotten \endgroup`. Seis mensagens de commit reportaram "104/99 pp, 0 overfull, 0
undefined" nesse periodo.

**Por que passou:** os dois caminhos de build discordavam e o que foi acreditado era o cego. O
`Makefile` usa `-halt-on-error` e nao produzia nada; `src_utils/build.sh` roda em
`-interaction=nonstopmode`, onde o pdflatex se recupera do erro e **escreve um PDF completo**, que o
script media e certificava limpo — porque nunca procurava por erros. Agora procura (`tex_errors`), e
o gate foi validado nas duas direcoes antes de ser aceito.

**Registrado em:** `science/AGENT_HANDOFF.md` §2.3b e `AGENT_GUARDRAILS.md` §7 (duas linhas novas de
vies: desconfie da ferramenta que reporta sucesso quando duas discordam; valide todo gate novo contra
uma arvore onde o defeito esta presente).

### 1.2 As decisoes suas que foram aplicadas

| Sua decisao | O que foi feito | Commit |
|---|---|---|
| COD-007: recuperar protocolo de Ch.3/Ch.4 | Eixo de split, seeds e regra de checkpoint recuperados do codigo e dos artefatos de execucao, e adicionados como **adicoes declaradas** com trilha em Apendice B | `519de348`, `a7ab2eaa` |
| COD-002: registrar o que fica fora do texto | [`LEFT_OUT.md`](LEFT_OUT.md) criado, 8 entradas, cada uma com achado, o que o texto diz em vez disso, por que esta fora, e **quem decidiu com data** | `e9370222` |
| 2.2: a tarefa estatica do Ch.4 em Apendice B, "facil de ser comentado" | Secao nova em `chapters/apx_b_static_scope.tex`, incluida por **um** `\input`. Caminho de supressao **testado** em arvore copiada: compila limpo, sem referencia pendente | `28097d93` |
| Split do `main.tex` | `main.tex` = build da defesa; `main_ppgc.tex` = o mesmo PDF **mais** a folha de aceite, em duas linhas de conteudo, para os dois nao divergirem. Terceiro alvo `make ppgc` | `7a91b720` |
| Chapters "corridos": dividir como nos artigos originais | Os tres capitulos de artigo divididos em 18 arquivos por secao, espelhando os nomes de arquivo de cada artigo. Verificado mecanico: camada de texto dos tres builds **identica byte a byte** antes e depois | `4e84cf7a` |
| Resumo/Abstract: cortar e refazer | 500 -> 310 palavras e 423 -> 271, refeitos como par de paridade de 11 sentencas, 19 claims em ambas as linguas | `40ed8e7b` |
| Margens e formatacao local | **Medido, nada a mudar.** Sondei a geometria real compilando uma pagina com o preambulo do documento: 3/2/3/2 cm e entrelinha exatamente 1,500x, todos exatos ao manual §7 | `2d117c7a` |
| Volume de comentarios | **Medido, recomendo nao comprimir.** 1.217 de 1.269 linhas de comentario (95%) carregam um fato rastreavel; as 52 restantes sao banners estruturais e a sua propria fila de sign-off | `2d117c7a` |
| Nomes de exemplo no front matter | **Nao existem.** Todos os campos reais, exceto tres placeholders honestos entre colchetes (dois membros da banca e a data), que e o estado correto | `18b817d9` |

### 1.3 Os defeitos que as revisoes acharam, e que foram corrigidos

Oito trilhas de revisao rodaram sobre o texto desta rodada. Nenhuma tinha visto o que a outra fez.
O que elas acharam:

| ID | Gravidade | O defeito | Commit |
|---|---|---|---|
| N-1 | BLOCKER | O limite "dentro de ±0,003" do cosseno de gradientes e **falso para Alabama** (+0,0032). Criado nesta rodada: o **escopo** da frase foi corrigido de tres para quatro datasets e o limite foi carregado sem reverificar a grandeza que depende do escopo. Corrigido na dissertacao **e** no manuscrito | `fecc7fb1` |
| N-2 | MAJOR | Ch.2 afirmava que Ch.3 "nao identifica o eixo de split", o que a adicao COD-007 desta rodada tornou falso no mesmo dia. O reparo foi **previsto por escrito** e redigido pela trilha de protocolo, e caiu entre dois escopos | `fecc7fb1` |
| D-01 | BLOCKER | Minha propria secao de Apendice B concluia que "o rotulo de um lugar nunca entra na sua propria representacao" no Ch.3. **A premissa e verdadeira e a conclusao nao segue:** o grafo e nao dirigido e a convolucao agrega o no com a vizinhanca, entao o rotulo volta no primeiro salto. Reproduzido em grafo de 4 nos: h_0[Food] = 0,667 contra x_0[Food] = 0,000 | `4b609643` |
| D-02 | BLOCKER | Ch.6 citava o ganho de 20,2 a 22,0 pontos do Ch.4 **sem rotular a tarefa**, como o diagnostico do arco inteiro. O ganho e da tarefa **estatica**, que o Apendice B desta rodada desqualifica. O numero fica (e a figura auditada do capitulo publicado), agora com tarefa e qualificacao, e o diagnostico repousa na tarefa sequencial | `4b609643` |
| F-01 | BLOCKER | **5 de 13** caminhos de reprodutibilidade do Apendice A **nao estao** no branch publico (auditado por CONTEUDO 2026-07-29; a contagem anterior de 9 comparava caminhos, e tres arquivos ja estao publicos sob `analysis_protocol/`) que o Ch.5 aponta em nota de rodape. Todos os 13 existem nesta maquina: a promessa estava errada, nao o codigo. (A primeira contagem dizia 8 de 12; ver `c6e62c62` -- um `grep` por linha perdeu `m1_full_output.txt`, que divide a linha com outro caminho) | `ec1cea0d`, `c6e62c62` |
| F-02 | MAJOR | A pagina 77, secao 6.2, tinha uma sentenca **sem sujeito**: "California run, completed since, repeats the pattern". O artigo "The" terminava a linha anterior e foi absorvido por um bloco de comentario inserido depois. Recuperado do commit original | `ec1cea0d` |
| C-1 | MAJOR | O build de **deposito** (AcademicoPG) imprimia 11 na pagina fisica 8. `\finalbuildfirstpage` estava fixo no offset do build de defesa, e o deposito tem tres paginas pre-textuais menos. Nao conformidade de numeracao no unico build que e depositado | `29c7629c` |
| E-5 | MAJOR | **Dez marcas de nota de rodape eram hyperlinks vivos para a pagina 1** em todos os tres builds. A persona mediu **onde os links caem**, nao apenas se os destinos resolvem. Corrigido com `hyperfootnotes=false` passado em tempo de carga (em `\hypersetup` **nao** funciona: o abntex2 ja carregou o hyperref) | `29c7629c` |
| E-2 | MAJOR | Seis arquivos sem diretiva `% !TeX root`, e depois do split esses seis incluiam os tres masters de capitulo. Segunda instancia na semana. Gate novo `check_tex_root.py` achou **18 outros** | `29c7629c` |
| STY-01 | MAJOR | Sete termos em uso que o registro fail-closed nao tinha, **dois deles em portugues no Resumo** que a minha propria passagem de registro seis horas antes nao cobriu | `a8865214` |
| AIC-01 | MAJOR | A densidade de paralelismo negativo foi **congelada** por uma revisao anterior e esta rodada a levantou de 67 para 79. O diagnostico da persona e o que importa: *"um guard que vive so no relatorio de uma rodada anterior e um guard que ninguem esta checando."* Movido para `check_negative_parallelism.py` | `a8865214` |
| C-6 | MAJOR | **`make check` saiu com codigo 2 durante toda a rodada** enquanto seis commits diziam "all gates pass". Dois falsos positivos ("this article" no apendice de errata, que e correto; "Pareto", que e o termo tecnico). Ambos isentos com a justificativa no lugar | `6ee23ca7` |
| L-9 | MINOR | O Apendice B imprimia "todos os 25 lugares" com uma decomposicao que soma 25 apenas contando um cabecalho de subsecao onde ha dois. Reenumerado: 28 | `6ee23ca7` |
| M-1, M-2 | MAJOR | O Resumo/Abstract perdeu o indice temporal do diagnostico do CoUrb e usava um universal sem escopo ("em todos os estados") cujo antecedente mais proximo e o numero errado de estados. Corrigidos **em paridade** nas duas linguas | `6ee23ca7` |

### 1.4 Gates novos, todos validados nas duas direcoes antes de serem aceitos

| Gate | A classe silenciosa que ele pega | Por que nenhum outro gate a via |
|---|---|---|
| `build.sh` `tex_errors` | O fonte nao compila | `nonstopmode` se recupera e escreve um PDF completo, que o script media |
| `check_doubled_macro.py` | `\\ref{...}` com barra dobrada, que imprime o rotulo cru | O pdflatex nao avisa (as duas metades sao legais) e `undef_ref` fica em 0, corretamente |
| `check_negative_parallelism.py` | Densidade de paralelismo negativo acima do teto | Vivia so num relatorio de revisao |
| `check_tex_root.py` | Diretiva `% !TeX root` ausente ou apontando para arquivo inexistente | Invisivel para o `make`, que le o `main.tex` e nunca olha um comentario magico |

---


---

## Recuperados na varredura de 2026-07-30 (rodada 8)

Itens que sairam do `PENDENCIAS.md` **fechados de forma legitima** — com commit e com marca de
resolucao no titulo — mas que nunca foram copiados para este arquivo. Encontrados varrendo as 63
revisoes do tracker **por titulo**, porque os numeros de item foram reciclados em tres renumeracoes e
uma busca por numero nao os acha. Cada um foi re-verificado contra a arvore viva antes de ser
registrado aqui; o que NAO se sustentou voltou para o `PENDENCIAS.md` em vez de vir para ca.

### ~~Fonte da bibliografia: 12 pt ou `\footnotesize`? (REV-024)~~ — RESOLVIDO 2026-07-27

> **Nao estava aqui, e deveria.** Fechado com commit em `63b6ad33` (2026-07-27) e removido do tracker
> sem ser copiado para este arquivo. Reencontrado na varredura de 2026-07-30 e **re-verificado**: o
> wrapper `{\footnotesize ...}` nao existe em nenhum arquivo raiz vivo (`main.tex`, `preamble.tex`,
> `content.tex` -> 0 ocorrencias), entao o fechamento se sustenta.

Aplicado: o wrapper `{\footnotesize ...}` saiu do `0_main.tex` e a bibliografia compoe em 12 pt,
conforme a decisao do autor e o `UFV_COMPLIANCE.md:32`. O `\campus{Campus Florestal}` foi setado e
**nao renderiza** hoje: a macro so e lida dentro de `\imprimircapa`, que nenhum build chama. Passa a
aparecer quando a capa for decidida. Commit `9e2b5157`.

