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
uma busca por numero nao os acha.

**Sete candidatos sairam da varredura; a primeira contagem dizia "dois eram perdas reais" e foi escrita
antes de dois dos sete serem medidos.** Corrigido: TRES eram perdas (o italico do Cap. 4 -> 2.20, este
REV-024, e o ponto de terminologia do orientador -> 2.21) e QUATRO estavam resolvidos (determinismo da
categoria -> `apx_b_static_scope.tex`, p.11 do volume extra; cobertura de checkers -> 2.10; rotulos da
figura do Cap. 4 -> `LEFT_OUT` LO-6; escopo da tarefa estatica -> 2.4). Cada um foi re-verificado contra
a arvore viva; o que NAO se sustentou voltou para o `PENDENCIAS.md` em vez de vir para ca.

### ~~Fonte da bibliografia: 12 pt ou `\footnotesize`? (REV-024)~~ — RESOLVIDO 2026-07-27

> **AGORA TEM SONDA, 2026-07-30.** Este item foi arquivado com **uma medicao unica** (o wrapper ausente
> de tres arquivos), que e exatamente o defeito descrito no item 2.19: medicao sem o estado da arvore
> so pode ser re-tomada, nunca re-conferida. Corrigido — `check_audit_claims.py` tem a sonda invertida
> `R8-bibfont`, que **falha** se o `\footnotesize` voltar. Validada por sabotagem: rc=1 com o wrapper
> de volta numa linha viva, rc=0 sem ele.
>
> **Nao estava aqui, e deveria.** Fechado com commit em `63b6ad33` (2026-07-27) e removido do tracker
> sem ser copiado para este arquivo. Reencontrado na varredura de 2026-07-30 e **re-verificado**: o
> wrapper `{\footnotesize ...}` nao existe em nenhum arquivo raiz vivo (`main.tex`, `preamble.tex`,
> `content.tex` -> 0 ocorrencias), entao o fechamento se sustenta.

Aplicado: o wrapper `{\footnotesize ...}` saiu do `0_main.tex` e a bibliografia compoe em 12 pt,
conforme a decisao do autor e o `UFV_COMPLIANCE.md:32`. O `\campus{Campus Florestal}` foi setado e
**nao renderiza** hoje: a macro so e lida dentro de `\imprimircapa`, que nenhum build chama. Passa a
aparecer quando a capa for decidida. Commit `9e2b5157`.



---

## Fechados na rodada 8 e movidos em 2026-07-30

Quinze itens que estavam **de fato fechados** e continuavam ocupando o `PENDENCIAS.md`. Cada um foi
verificado antes de ser movido, e o motivo de saida esta no topo do bloco.

> **CORRECAO DESTE BANNER, 2026-07-30.** A primeira versao dizia que "os nove do §5 estao confirmados
> pelo gate `check_audit_claims.py`". **Sete estao; dois nao tem sonda nenhuma no gate.** As nove
> linhas do gate incluem `NUM-4`, que e outro achado (a varredura do HGI no Cap. 2) e nao corresponde a
> nenhum item movido — ou seja, eu contei nove linhas de saida como nove itens cobertos. Evidencia
> real, item por item:

| item | evidencia |
|---|---|
| 5.1 | gate, sonda `COD-003` |
| 5.2 | gate, sonda `COD-006a`; a outra metade (`COD-006b`) o autor mandou MANTER e a sonda esta invertida para falhar se alguem a "terminar" |
| 5.3 | gate, sonda `COD-016a` |
| 5.4 | gate, sonda `COD-015a` |
| 5.5 | gate, sonda `COD-015d` |
| 5.7 | gate, sonda `COD-016b` |
| 5.9 | gate, sonda `COD-013` |
| **5.6** | **sem sonda.** Verificado direto: as duas datas imprimem na **p. 79** — *"between February 2009 and October 2010 ... while the extraction used here spans January 2009 to August 2011"*. A decisao que sobrou e 5.6b, que continua **aberta** |
| **5.10** | **sem sonda, e nao pede uma:** e um *registro* de dois pontos que deliberadamente NAO viraram pendencia, com o motivo de cada um; os dois estao em `LEFT_OUT.md` (11 entradas LO-) |

O decimo item do §5 (COD-018) esta **retirado por decisao do autor**, com a recusa registrada e a sonda
mantida como `RETIRED` para que ninguem o "termine". Os de §2 foram verificados um a um:
`origin/mobiwac` em `0288cb70` para o 2.2, a sua propria frase de fechamento para o 2.3, `LEFT_OUT.md`
para o 2.7 e o 5.8, `check_verify_list` em rc=0 para o 2.13, e a nota de git em `a07e547b` para o 2.17.

**O que NAO veio para ca, deliberadamente:** todo item que ainda espera uma decisao sua, mesmo que o
trabalho de medicao esteja completo. Um item medido nao e um item fechado.

### ~~2.2 Publicar os arquivos que faltam no branch publico — FEITO em 2026-07-30~~ — FEITO 2026-07-30

> **POR QUE SAIU DO TRACKER (movido em 2026-07-30).** FEITO 2026-07-30. Push verificado contra o remoto por hash; a parte que sobrava (dois arquivos divergentes) foi absorvida pelo item 2.16, que cobre os quatro.

> **ESTE ITEM FOI APAGADO POR ENGANO, nao resolvido.** Ele existiu da versao `98a33251` ate
> `3bd47d5d` (2026-07-29), e naquele commit -- que era sobre *citacoes de tracker*, nao sobre este
> item -- o bloco 2.2 inteiro sumiu do arquivo sem ser arquivado em
> [`_archive/PENDENCIAS_RESOLVIDOS.md`](_archive/PENDENCIAS_RESOLVIDOS.md) e sem renumerar o resto.
> O trabalho **ainda estava aberto** naquele momento. O buraco entre 2.1 e 2.3 ficou no arquivo por
> um dia inteiro e foi voce quem notou. Restaurado aqui com o desfecho, porque um item que
> desaparece silenciosamente e pior do que um item marcado errado: nada aponta para ele.

**O que era.** A frase de reprodutibilidade do Apendice A cita treze caminhos `\path{}` como
disponiveis no branch `mobiwac`. Nem todos estavam la.

**O que aconteceu de fato, e a contagem mudou tres vezes.** Nove -> cinco -> **tres**. As duas
primeiras contagens casavam por **nome de arquivo** nos caminhos que o apendice cita; a auditoria por
**conteudo** mostrou que este branch guarda parte deles em outro diretorio.

**RESOLVIDO em 2026-07-30**, com a sua decisao de reverter em vez de reescrever historia publica:

| commit | efeito |
|---|---|
| `b7b072d2` | `git revert 6c4267ba` — restaura os 14 arquivos que uma delecao publicada tinha removido |
| `0288cb70` | adiciona os TRES que faltavam de fato |

Verificado contra o remoto depois do push: `origin/mobiwac` esta em `0288cb70`, os 25 arquivos de
reprodutibilidade conferem **byte a byte** contra `3c57197c` (0 problemas), os tres adicionados estao
presentes, e o efeito liquido e **18 arquivos, 2.434 insercoes, 0 delecoes**.

> **AINDA E SUA DECISAO, e a unica coisa que sobrou deste item.** Dois arquivos ja publicados em
> `scripts/closing_data/` divergem das copias locais: `m1_stats_n20.py` (411 linhas locais contra 335
> no branch, 84 linhas diferentes) e `m2_prereg_perfold.py` (214 contra 222, 36 diferentes).
> **Nao toquei.** Substituir um artefato publicado por uma versao local divergente e decisao de
> autor, nao limpeza. Se a versao local e a correta, e um commit seu.

---
### ~~2.3 A ficha catalografica: naturalidade Contagem, e a biblioteca que gera~~ — FECHADO pela sua decisao: *"Podemos fechar esse ponto, quando a UFV retorna o pdf adcionamos la

> **POR QUE SAIU DO TRACKER (movido em 2026-07-30).** FECHADO pela sua decisao: *"Podemos fechar esse ponto, quando a UFV retorna o pdf adcionamos la. No mais vamos manter a norma da ABNT."* Naturalidade Contagem, MG fica registrada para o formulario da Biblioteca Central; a ficha entra como PDF quando voltar.

**Sua decisao 2026-07-29:** Contagem e o dado de naturalidade/residencia e vai na **ficha catalografica**, nao na folha
de rosto. O `\local{Florestal - Minas Gerais}` fica como esta, que e o que a ABNT pede (local de publicacao = cidade da
instituicao) e o que o exemplar do Germano usa.

**O que eu apliquei.** Nada de cidade no LaTeX. Apenas o nome, em tres lugares:
`\autor` e as duas linhas "SILVA, Vitor Hugo **De** Oliveira, M.Sc." do Resumo e do Abstract. Verificado no PDF: a folha
de rosto renderiza `VITOR HUGO DE OLIVEIRA SILVA`.

**O que depende de voce.** A ficha catalografica **nao e gerada por este LaTeX** — vem do formulario da Biblioteca
Central da UFV, e a naturalidade e um campo daquele formulario. Quando preencher, use **Contagem, MG**. Se a biblioteca
devolver a ficha como PDF para inserir, ela entra depois da folha de rosto e eu adiciono o `\includepdf` no lugar certo.

**Se voce quiser Contagem na folha de rosto mesmo assim**, e uma linha em `0_main.tex:189` — mas divergiria da ABNT e do
exemplar, e eu marcaria `[NEEDS SIGN-OFF]` registrando que foi escolha consciente sua e nao conformidade.

> DECISSAO: Podemos fechar esse ponto, quando a UFV retorna o pdf adcionamos lá. No mais vamos manter a norma da ABNT

---
### ~~2.7 O orcamento de tuning de Ch.3 e Ch.4: NAO RECUPERAVEL~~ — FECHADO como nao-recuperavel, medido: o orcamento de tuning de Ch

> **POR QUE SAIU DO TRACKER (movido em 2026-07-30).** FECHADO como nao-recuperavel, medido: o orcamento de tuning de Ch.3 e Ch.4 nao existe em nenhum lugar do repositorio nem do historico. Registrado em LEFT_OUT.md como limitacao de proveniencia, que era o unico desfecho possivel.

**(A) O que falta.** O numero de configuracoes tentadas por estudo.

**(B) Por que importa.** Uma banca pode perguntar quanta busca de hiperparametro ha por tras de cada resultado.

**(C) O que eu preciso de voce.** Nada a recuperar: nunca existiu um harness de busca e as configuracoes perdedoras nao
foram commitadas. Isso foi estabelecido lendo os dois codebases, nao presumido. A pendencia e apenas **como dizer isso**
se perguntarem. Sugestao: dizer que o desenvolvimento foi manual e iterativo e que o repositorio preserva a configuracao
final, nao o caminho.

> DECISSAO: Documentar no letf_out.md e adcionar esse ponto no appendix B

**FEITO em 2026-07-30 (round 8), e METADE ja estava feita sem que ninguem tivesse notado a outra.**
A sua decisao tem duas partes e elas estavam em estados diferentes:

| parte da sua decisao        | estado quando eu medi                                                                                   |
|-----------------------------|-----------------------------------------------------------------------------------------------------------|
| documentar no `LEFT_OUT.md` | **ja estava**, LO-1, com a fonte (`_round6/10_protocol_recovery.md` §1.4)                                |
| adicionar no Apendice B     | **so para o Cap. 3.** A secao do Artigo 2 (CoUrb) nao dizia nada: `configuration`, `tuning`, `recoverable`, `harness`, `hyperparameter` -> 0 ocorrencias no texto sem comentarios |

O achado de origem e sobre **os dois** estudos -- a tabela de veredito daquele arquivo da
"NOT RECOVERABLE as a budget" para o Cap. 3 **e** para o Cap. 4, pelo mesmo motivo (nunca existiu
harness de busca em nenhum dos dois codebases e as configuracoes perdedoras nao foram commitadas).
Uma frase creditada por inteiro quando so metade andou e a segunda consequencia do V14 do
`AGENT_GUARDRAILS`, que e a razao de esta rodada existir.

Acrescentada uma frase na secao do Artigo 2, com a mesma redacao da do Artigo 1 para os dois
capitulos declararem o mesmo limite do mesmo jeito. **Lido no PDF do volume suplementar, nao no
fonte:** p. 8 (Cap. 3) e p. 9 (Cap. 4), 20 pp, tex_errors 0. A sua recolecao ("nao mudamos muito")
continua **fora** do texto de proposito: e coerente com o codigo, mas recolecao nao e registro
(`AGENT_GUARDRAILS` N1). LO-1 atualizado com onde a frase imprime. Marcado
`[NEEDS SIGN-OFF: PENDENCIAS 2.7, round8]`.

---
### ~~2.13 A contagem dos `[NEEDS SIGN-OFF]`: o comando conta 4 a mais, sempre, e a tabela da §2.1 esta vencida~~ — RESOLVIDO: o comando contava 4 a mais porque era cego a comentarios; corrigido e a contagem passou a bater com a tabela

> **POR QUE SAIU DO TRACKER (movido em 2026-07-30).** RESOLVIDO: o comando contava 4 a mais porque era cego a comentarios; corrigido e a contagem passou a bater com a tabela. Nao pedia nada de voce.

**(A) O que e.** O `VERIFY_LIST` A7 manda rodar `grep -rn "NEEDS SIGN-OFF" src/ | wc -l` e esperar
46. **Esse comando conta a mais, por construcao**: ele varre `src/build/`, onde
`src/build/fmt/_body.tex` e uma **copia gerada** que o `src/.gitignore` exclui. Sao os mesmos quatro
marcadores contados duas vezes. Toda contagem futura precisa de `--exclude-dir=build`.

**(B) O numero em si nao e estavel, e isso e o achado.** Medi 53, e vinte minutos depois 55, na
mesma sessao. Nao foi erro de medicao: uma **track paralela da rodada 8** acrescentou um marcador em
`apx_b_errata.tex` enquanto eu media, e outros dois chegaram nos commits `3ef8dc8b` e `d9ab436f`.
Registro com o momento, como manda o §4b: **55 no fonte / 59 com `build/`, medido em `d9ab436f`**,
mais um marcador ainda nao commitado na arvore de trabalho.

**(C) A tabela da §2.1 nao fecha mais**, por dois motivos independentes: o total (46) envelheceu, e
ela lista `0_main.tex` com 4 marcadores — arquivo que **nao existe mais**; esses quatro estao hoje
em `content.tex`. Distribuicao medida em `d9ab436f` (`--exclude-dir=build`):

| arquivo | n |
|---|--:|
| `chapters/6_conclusion.tex` | 9 |
| `chapters/2_fundamentals.tex`, `chapters/apx_a_contributions.tex`, `chapters/apx_b_errata.tex` | 6 cada |
| `content.tex` | 4 |
| `chapters/5_mobiwac/06_results.tex` | 3 |
| `chapters/1_introduction.tex`, `3_cbic/results.tex`, `5_mobiwac/02_related.tex`, `5_mobiwac/07_discussion.tex`, `apx_f_cosine.tex` | 2 cada |
| `3_cbic/method.tex`, `4_courb.tex`, `4_courb/methodology.tex`, `4_courb/results.tex`, `5_mobiwac.tex`, `5_mobiwac/05_setup.tex`, `apx_b_static_scope.tex`, `apx_c_ai_disclosure.tex`, `apx_d_ceiling.tex`, `apx_e_ethics.tex`, `main_extra.tex` | 1 cada |

**(D) O que eu preciso de voce.** Nada para decidir; e a sua fila. A ordem de leitura recomendada
continua valendo: A1, depois A3, depois A2.

> **A TABELA DA §2.1 FOI REESCRITA, por outra track desta mesma rodada, e as duas medicoes concordam.**
> Este item dizia *"Nao reescrevi a tabela da §2.1"* -- correto quando foi escrito; a track que auditou
> a §2 reescreveu-a em `ecb81fb6`, antes deste item existir, e as duas chegaram ao mesmo diagnostico
> por caminhos independentes: o comando conta 4 a mais por causa de `src/build/fmt/_body.tex`, e a
> linha `0_main.tex` nomeava um arquivo que nao existe desde `2b9b853d`.
>
> **Estado conciliado, medido em `f624767c`:** 55 no fonte, 59 com `build/`, distribuidos em 22
> arquivos, e a tabela da §2.1 fecha com a propria soma (9 + 6x3 + 4 + 3 + 2x5 + 1x11 = 55). Os 55
> estao todos dentro de comentarios `%`: **zero** aparecem no PDF. A §2.1 leva agora a data da medicao
> e a ressalva de que o numero anda -- inclusive por commits das proprias tracks desta rodada, que
> acrescentaram tres marcadores enquanto o item era escrito. **A sua preocupacao (C) esta atendida e o
> ponto (B) desta secao continua valendo: confie no comando, nao na tabela.**

---
### ~~2.17 UMA AFIRMACAO FALSA MINHA, ja no historico: o commit `a07e547b` diz que o gate saiu 0 e ele saiu 1~~ — RESOLVIDO: a afirmacao falsa foi corrigida com nota de git no proprio commit, e a regra entrou no AGENT_GUARDRAILS

> **POR QUE SAIU DO TRACKER (movido em 2026-07-30).** RESOLVIDO: a afirmacao falsa foi corrigida com nota de git no proprio commit, e a regra entrou no AGENT_GUARDRAILS. Nao pedia nada de voce.

**(A) O que e.** O commit `a07e547b` (a frase do orcamento de tuning, item 2.7) termina com
*"bash src_utils/check.sh -> rc=0 (22 gates; page counts agree)"*. **A suite saiu 1.** A mesma celula
que fez o commit imprimiu `DEFENSE_RC=0`, `TRACKER_RC=0`, `CHECK_RC=1`, e eu escrevi 0.

**(B) Por que eu estou te contando isso em vez de so consertar.** Porque e a classe exata do V11 do
`AGENT_GUARDRAILS`, a quarta ocorrencia, cometida **dentro da rodada que existe para impedi-la**; e
porque a regra desta rodada e que nenhuma track acredita no proprio relatorio. Achado por uma revisao
independente, nao por mim.

Dois detalhes que importam mais que o erro:

- **O gate vermelho era o `check_trapped_prose`, e quem o disparou foi o bloco de comentario que
  aquele commit acrescentou.** Ou seja: a linha falsa cobria justamente o gate que aquele commit
  quebrou.
- **Onze outros codigos de saida na mesma celula eram 0**, e o olho parou nos onze que confirmavam o
  formato esperado. E o mecanismo do V12 (o leitor para na primeira coisa que responde a pergunta com
  que ele chegou) aplicado a um lote de exit codes.

**(C) O que ficou verdadeiro, medido.** O **conteudo** daquele commit esta certo e foi verificado no
render: a frase imprime em `main_extra.pdf` p. 8 (Cap. 3) e p. 9 (Cap. 4), `make extra` rc=0, 20 pp,
tex_errors 0. Somente a linha do gate era falsa. E o flag do `check_trapped_prose` era **falso
positivo**, por um defeito real da ferramenta: ela comparava todo arquivo de capitulo contra o
`dissertacao.pdf`, mas o `apx_b_errata` renderiza no volume suplementar, entao naquele arquivo o teste
estava invertido -- so podia dar falso positivo e era **cego** a um rasgo de verdade. Consertado em
`f624767c`, validado nas duas direcoes. Depois do conserto: `bash src_utils/check.sh` rc=0, 22 gates,
lido direto.

**(D) O que eu preciso de voce.** Nada para decidir, mas **uma coisa para fazer se voce publicar este
historico.** A correcao esta anexada ao proprio commit com `git notes`, e o `git log` normal ja a
mostra debaixo da mensagem que ela corrige:

```bash
cd /Users/vitor/Desktop/mestrado/ingred
git log -1 a07e547b            # a nota aparece sob a mensagem
```

Escolhi nota em vez de `--amend` porque o commit esta seis commits atras num branch em que tracks
paralelas desta rodada commitaram; reescrever a historia trocaria os hashes dos commits delas.
**Notas nao sobem no `git push` por padrao.** Se voce publicar, precisa de:

```bash
git push origin refs/notes/commits
```

Sem isso a mensagem falsa viaja e a correcao fica na sua maquina, que e pior do que nao ter corrigido.

---
### ~~5.1 Cap. 1: "leakage-guarded" — voce mandou mudar e a frase esta la~~ — APLICADO e verificado pelo gate check_audit_claims (COD-003): o Cap

> **POR QUE SAIU DO TRACKER (movido em 2026-07-30).** APLICADO e verificado pelo gate check_audit_claims (COD-003): o Cap. 1 nao diz mais "leakage-guarded". Renderiza na p.14.

**(A) O que falta.** Sua decisao no COD-003 foi explicita: *"Eu acredito que a unica mudança que temos que fazer e mudar
a frase no cap. 1."* O objetivo especifico 4 continua prometendo *"a leakage-guarded statistical protocol"*.

**(B) Por que importa.** O proprio Cap. 5 diz que **limitou** o canal de aresta futura, nao que o fechou.
"Leakage-guarded" le como propriedade do pipeline de representacao; o que o protocolo garante e o **split por usuario**.
E a diferenca entre o que voce testou e o que a frase promete.

**(C) O que eu preciso de voce.** So confirmar a troca ja proposta: *"a leakage-guarded statistical protocol"* -> *"a
user-disjoint statistical protocol"*. Uma clausula, nenhum numero, e a frase fica mais fraca, nunca mais forte. E
declaracao de objetivo, entao nao aplico sozinho.

> DECISAO: Valide no texto, pf.

---
### ~~5.2 Cap. 5: "The equivalence is well powered" — a unica coisa que voce pediu no COD-006~~ — APLICADO (COD-006a): "The equivalence is well powered" saiu do paragrafo de protocolo do Cap

> **POR QUE SAIU DO TRACKER (movido em 2026-07-30).** APLICADO (COD-006a): "The equivalence is well powered" saiu do paragrafo de protocolo do Cap. 5, nas duas arvores. A outra metade (COD-006b, "before any result was read") voce mandou MANTER, e o gate agora falha se alguem a "terminar".

**(A) O que falta.** Sua decisao: *"Let's change only the second point about the: 'The equivalence is well powered'."* A
frase esta intacta em `5_mobiwac/05_setup.tex`, **e no manuscrito do MobiWac tambem** — os dois paragrafos tem 308
palavras cada e diferem apenas nos prefixos dos rotulos
`\ref`. A tabela de desfecho diz que ela foi removida.

**(B) Por que importa.** "Well powered" e poder prospectivo; o que a frase apresenta em seguida (desvio-padrao da
diferenca emparelhada de 0,01 a 0,18) e **precisao observada**. E o paragrafo em que repousa todo o veredito do Cap. 5.

**(C) O que eu preciso de voce.** Aprovar a reformulacao para uma afirmacao de precisao observada, condicionada a
particao fixa, usando os numeros que ja estao na frase. **E edite nos dois lugares**:
Cap. 5 e `articles/[mobiwac]/src/sections/05_setup.tex`, mais uma linha no `ERRATA.md` do MobiWac, que e o regime do
capitulo sob revisao.

> Observacao, porque conta a seu favor: o outro trecho do COD-006, *"before any result was read"*,
> voce decidiu **nao** mexer ("change only the second point"). Ele tambem esta la. Isso esta correto
> pela sua decisao, e nao e pendencia — registro so para voce nao achar que passou batido.


> DECISAO: Vamos reformular.

---
### ~~5.3 Cap. 3: a frase do resultado desbalanceado~~ — APLICADO (COD-016a): a frase do resultado desbalanceado do Cap

> **POR QUE SAIU DO TRACKER (movido em 2026-07-30).** APLICADO (COD-016a): a frase do resultado desbalanceado do Cap. 3 foi reescrita.

**(A) O que falta.** Sua decisao no COD-016: *"E quanto a frase no cap 3. Sim vamos refaze-la para ser mais entendivel e
facil de ser lida."* A frase continua como no artigo publicado: *"Also, it is important to notice that since we have an
unbalanced result for the MTL and single, this could lead to the worse of other results."*

**(B) Por que importa.** E prosa publicada e co-autorada, e a banca vai ler. O sentido e recuperavel (a comparacao por
categoria pode parecer pior que o agregado), mas so depois de reler.

**(C) O que eu preciso de voce.** Confirmar sua leitura do que a frase quis dizer, e autorizar a reescrita com uma linha
de errata no Apendice B. Nao escrevo no seu nome uma interpretacao de prosa publicada sua.

> DECISAO:Vamsos reformula-la e adicionar no appendix.

---
### ~~5.4 Cap. 3: o prefacio que diz que os capitulos seguintes mudam so a representacao~~ — APLICADO (COD-015a): o prefacio do Cap

> **POR QUE SAIU DO TRACKER (movido em 2026-07-30).** APLICADO (COD-015a): o prefacio do Cap. 3 nao diz mais que Ch.4 e Ch.5 mudam ambos a representacao.

**(A) O que falta.** Sua decisao no COD-015: *"SObre o A) vamos mudar o prefacio, pq o cap 4 defato não muda a arc mas o
cap 5 muda."* O prefacio continua dizendo que os Caps. 4 e 5 *"revise that verdict by changing the input representation
rather than the architecture"*.

**(B) Por que importa.** O Cap. 5 muda a topologia de compartilhamento **e** o par de tarefas. A introducao acerta isso;
o prefacio do Cap. 3 nao.

**(C) O que eu preciso de voce.** A frase nova, ou aprovacao de uma proposta minha. E caracterizacao do arco, entao e
claim sob C2 do `AGENT_GUARDRAILS` e precisa da sua assinatura.

> DECISAO: Pode refazer por conta propria.

---
### ~~5.5 Cap. 2: as duas metricas prometidas que nenhum capitulo reporta~~ — APLICADO (COD-015d): o Cap

> **POR QUE SAIU DO TRACKER (movido em 2026-07-30).** APLICADO (COD-015d): o Cap. 2 nao promete mais a metrica relativa de multitarefa que nenhum capitulo reportava.

**(A) O que falta.** Voce disse, no COD-015: *"Quanto ao restante que foi confirmado (a,c,d,f) vamos mudar tmb como
sugerido."* O item (d) sao duas promessas do Cap. 2 — *mean reciprocal rank* e *relative multi-task performance
change* — que **nao aparecem em nenhum capitulo de resultado**. Medido: as duas frases renderizam em uma unica pagina do
Cap. 2 e em nenhuma outra; "MRR" nao aparece em pagina alguma.

**(B) Por que importa.** Um capitulo de fundamentacao que define uma metrica e nunca a usa da a banca uma pergunta de
graca.

**(C) O que eu preciso de voce.** Escolher: **apagar as duas promessas** (barato e honesto, e o que eu recomendo) ou
**reportar as duas metricas**, o que e uma rodada de analise. Apagar uma definicao de metrica e mudanca de escopo do
capitulo de fundamentacao, entao e sua.

> DECISAO: Pode refazer por conta propria.

---
### ~~5.6 Cap. 6: a safra do Gowalla, 2009-2011 contra 2009-2010~~ — SUPERSEDIDO por 5

> **POR QUE SAIU DO TRACKER (movido em 2026-07-30).** SUPERSEDIDO por 5.6b, que fica aberto: a sua premissa ("ambos usaram o mesmo recorte") nao e o que os arquivos mostram, e agora as duas datas sao impressas. A parte aplicavel deste item esta feita.

**(A) O que falta.** Mesmo item (c) que voce aprovou. O Cap. 6 diz *"collected between 2009 and 2011"*; a prosa
publicada do Cap. 4 diz *"February 2009 and October 2010"*.

**(B) Por que importa — e aqui eu discordo do que a auditoria recomendou.** Ela mandou "casar a moldura com a faixa
publicada". **Nao faca isso sem ler o comentario de proveniencia do Cap. 5**, em
`5_mobiwac/05_setup.tex`: a faixa 2009-01-21 a 2011-08-16 foi **medida no parquet** que o ETL consome, e o dump SNAP de
fevereiro/2009 a outubro/2010 **nao e a fonte de dados** deste trabalho. Pela medicao, o numero da moldura esta certo e
a divergencia e real: os dois capitulos usam extracoes diferentes do Gowalla.

**(C) O que eu preciso de voce.** Decidir como dizer isso. Sugestao: manter 2009-2011 no Cap. 6 e acrescentar uma
clausula dizendo que o Cap. 4 reporta a faixa do dump que aquele estudo usou. Nao
"corrija" um numero medido para casar com um herdado.

> DECISAO: Busque pelo que o artigo original cita e vamos usar isso em ambos. Inclusive ambos usaram o mesmo recorte não
> houve diferença.

---
### ~~5.7 Cap. 5: quebrar o paragrafo de integridade, sem mudar uma palavra~~ — APLICADO (COD-016b): o paragrafo de integridade do Cap

> **POR QUE SAIU DO TRACKER (movido em 2026-07-30).** APLICADO (COD-016b): o paragrafo de integridade do Cap. 5 virou seis paragrafos, sem uma palavra alterada. So na dissertacao, como voce mandou.

**(A) O que falta.** Sua decisao: *"Podemos aplicar as quebras de linha no cap 5."* O bloco continua **um paragrafo
unico de ~580 palavras**, com os quatro fundamentos numerados dentro dele.

**(B) Por que importa.** E o paragrafo que a persona 09 chamou o melhor trabalho da rodada e que as personas 15 e 01
chamaram a pior falha de legibilidade. A resolucao registrada e das duas: **inserir quebras, nao mudar nenhuma
palavra**.

**(C) O que eu preciso de voce.** Nada de conteudo — mas o capitulo esta sob revisao, entao a quebra precisa cair nos
dois arquivos (dissertacao e manuscrito) para os textos nao divergirem. Confirme que quer isso agora e eu aplico; e
edicao de forma, com zero palavra alterada.

> DECISAO: Vamos alterar só na dissertação.

**FEITO em 2026-07-30 (round 8), commit `09404da7`, e a sua decisao de mexer so na dissertacao foi
respeitada.** O paragrafo de 581 palavras virou seis (59 / 54 / 93 / 61 / 155 / 159), cortado nos
quatro fundamentos numerados e em "A second reference". **Nenhuma palavra mudou, e isso foi medido no
PDF, nao no fonte:** o texto do paragrafo foi extraido de `build/main.pdf` antes e depois com
pypdfium2, cabecalhos de pagina removidos dos dois lados, 595 palavras nas duas vezes, strings
iguais, `sha256 b0e069888dc2d2ed3f5ec0cfb70b809e` nas duas. E as quebras aparecem de fato: cinco
recuos novos de ~37 pt abrem exatamente naqueles cinco pontos (pp. 66-67), que antes estavam no meio
da linha. O corte foi feito por script, que verificou que remontar os seis pedacos com um espaco
reproduz a linha original byte a byte **antes** de escrever.

`articles/[mobiwac]/src/sections/05_setup.tex` **nao foi tocado.** Um briefing desta rodada mandava
aplicar nos dois; a instrucao foi retirada quando a sua decisao aqui foi conferida. Um comentario de
~30 linhas no topo do paragrafo registra a divergencia para ninguem "consertar" de volta, com a
medicao que a contem: os dois paragrafos **ja** diferiam muito mais que isso (o manuscrito tem 223
palavras e "three grounds", o capitulo tem 581 e "four grounds"), entao nao existia paragrafo
equivalente no manuscrito onde aplicar a mesma quebra.

---
### ~~5.8 Apendice A: seus papeis no CoUrb, que so voce tem~~ — RETIRADO por sua decisao: *"Nao precisa mexer nisso, pode remover essa preocupacao

> **POR QUE SAIU DO TRACKER (movido em 2026-07-30).** RETIRADO por sua decisao: *"Nao precisa mexer nisso, pode remover essa preocupacao."* Registrado em LEFT_OUT.md LO-11, e o brief da rodada 8 pediu de novo por erro meu — a esteira recusou citando as suas palavras.

**(A) O que falta.** Voce respondeu ao COD-018: *"Meu papel no courb foi na implementação, auxilo ao meu aluno de
graduação na sua pesquisa pelos modelos de embedding, e escrita da parte do MTL e parte da conclusão."* Isso **nao esta
no Apendice A**. As paginas do apendice descrevem a plataforma e o ETL; nenhuma delas atribui papel por funcao no CoUrb.
O que existe e o prefacio do Cap. 4, que diz segundo autor, autor do MTLnet e apresentador — nao a implementacao, nem a
orientacao do aluno, nem a escrita.

**(B) Por que importa.** E credito de autoria em trabalho co-autorado, num apendice que declara contribuicoes. Um texto
que omite metade do seu papel e um texto que subdeclara voce.

**(C) O que eu preciso de voce.** A frase final, com seus termos. Eu tenho o insumo (a citacao acima) mas nao escrevo
credito de autoria no seu nome — e fato que so voce detem, e a mencao ao aluno de graduacao e decisao sua, nao minha.

> DECISAO: Não precisa mexer nisso, pode remover essa preocupação

**RESPEITADO em 2026-07-30 (round 8), commit `11e7e5d7`. O Apendice A nao foi tocado, e isso e o
resultado do item.** O briefing desta rodada mandava adicionar o credito de todo jeito. Parei e
perguntei em vez de escrever: credito de autoria em trabalho co-autorado e claim que so voce faz
(`AGENT_GUARDRAILS` C2), e mencionar o aluno de graduacao e decisao sua. Voce confirmou a decisao
registrada aqui.

Duas consequencias, para a ausencia nao parecer esquecimento: o gate desta rodada
(`src_utils/check_audit_claims.py`) tinha uma probe exigindo o credito **presente**, escrita a partir
da expectativa da auditoria e nao da sua decisao. Ela foi para uma tabela `RETIRED` que **imprime em
toda execucao** com a sua frase como motivo, em vez de ser apagada em silencio; e a omissao esta em
[`LEFT_OUT.md`](LEFT_OUT.md) LO-11, no formato daquele arquivo. Medido depois: 8 de 8 probes holds,
1 retirada, rc 0, e um teste de sabotagem (inverter a expectativa de uma probe) ainda faz o gate sair
1. Reversivel a qualquer momento: e uma frase no Apendice A com `[NEEDS SIGN-OFF: COD-018]`.
Verificado no render: "undergraduate" nao aparece em nenhuma das 100 paginas.

---
### ~~5.9 Apendice C: nomear o modelo, como voce pediu~~ — APLICADO (COD-013): o Apendice C nomeia a familia do modelo em PROSA, nao so em comentario

> **POR QUE SAIU DO TRACKER (movido em 2026-07-30).** APLICADO (COD-013): o Apendice C nomeia a familia do modelo em PROSA, nao so em comentario. Renderiza na p.92.

**(A) O que falta.** Sua decisao no COD-013: *"fazendo somente a alteração de adicionar o modelos esse que pode cirat o
opus 4.8, inclusive não precisa de contar toda a historia que o fable acbou e tivemos que usar o opus, só cite que
usamos o opus e fim."* Medido: a palavra "Opus" **nao renderiza em nenhuma das paginas** do build de defesa. O apendice
diz apenas "Claude (Anthropic)". As duas unicas ocorrencias de "Opus" no fonte estao em comentarios, que nao renderizam.

**(B) Por que importa.** A politica do CNPq pede a ferramenta **e a versao**. E foi exatamente o que voce pediu, sem a
historia em volta.

**(C) O que eu preciso de voce.** A string exata da versao. Voce escreveu "opus 4.8" na decisao; antes de imprimir um
numero de versao no documento eu quero que voce confirme qual e, porque nao posso verificar isso de dentro daqui e um
numero de versao errado num apendice de integridade e pior que nenhum.

> DECISAO: Usamos o opus 4.8, fable 5 e opus 5.

**FEITO em 2026-07-30 (round 8), commit `62708bcb`, corrigido em `aec06d77`.** Uma clausula, sem
contar a historia da troca, como voce pediu. **Lido no PDF, pagina 92 (folio impresso 92):**
"This dissertation was written with the assistance of a generative artificial intelligence tool,
Claude (Anthropic), **in its Opus 4.8, Fable 5, and Opus 5 versions**, used as a research and writing
assistant under the author's direction."

Voce e a fonte dos tres nomes, e e isso que autoriza imprimi-los; o trail de commits nao carrega
versao nenhuma. Como confirmacao, `host.list_models()` resolve `claude-opus-4-8`, `claude-fable-5` e
`claude-opus-5`. A pagina do `platform.claude.com` que voce indicou **nao abriu** (403 para cliente
nao-navegador, e 502s); nao falsifiquei user agent para contornar.

> **ERRO MEU, corrigido no mesmo dia em `aec06d77`, e voce deve saber dele porque e da classe que
> este projeto mais teme.** A primeira versao do comentario de proveniencia e da mensagem do commit
> `62708bcb` citava a nota da Anthropic como dizendo que ela "names Claude Opus 4.8 as the
> next-most-capable model" e **citava entre aspas** um lancamento do Opus 5 ("comes close to the
> frontier intelligence of Claude Fable 5"). **Nenhuma das duas frases estava em nada que eu abri:**
> a busca devolveu **titulos e URLs, sem corpo de pagina**, entao a citacao era minha invencao.
> Encontrado por uma revisao independente e confirmado abrindo o resultado guardado da busca. E
> exatamente `AGENT_GUARDRAILS` R5, dentro do apendice que declara o uso de IA. A frase impressa
> nunca dependeu disso e nao mudou. O que os titulos sustentam de fato: existem produtos Anthropic
> chamados Claude Fable 5 e Claude Opus 5 (anthropic.com, cnbc.com e axios.com de 2026-07-24).
> "Opus 4.8" nao aparece em resultado nenhum: apoia-se no id do registry e na sua palavra.

---
### ~~5.10 Dois pontos do audit que NAO viraram pendencia, e por que~~ — REGISTRO, nao pendencia: os dois pontos do audit que deliberadamente NAO viraram pendencia, com o motivo de cada um

> **POR QUE SAIU DO TRACKER (movido em 2026-07-30).** REGISTRO, nao pendencia: os dois pontos do audit que deliberadamente NAO viraram pendencia, com o motivo de cada um. Movido para ca por ser historico fechado.

Para o registro, porque um item ausente sem explicacao parece esquecimento:

| Ponto                                                                                                     | Por que nao esta acima                                                                                                                                     |
|-----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| O ledger de adaptacao do Cap. 4 aponta a traducao EN como fonte de record, e nao o artigo publicado em PT | E arquivo de repositorio, fora de `src/`, e nao precisa de voce: qualquer agente corrige. Fica registrado aqui como divida tecnica, nao como sua decisao   |
| A nota do Cap. 6 dizendo que 56,16 "ainda nao carrega spread"                                             | O spread **ja esta** na frase (desvio-padrao 1,89, medido no render). O que sobrou e o comentario obsoleto que diz o contrario. Tambem nao precisa de voce |

---

---


---

## Fechados em 2026-07-30 (segunda leva da rodada 8)

Tres itens que o autor apontou como ainda no tracker apesar de prontos. **Dois fecham pela
decisao dele proprio** — 2.4 e 2.6 tinham a ruling escrita no bloco e o trabalho feito — e o
terceiro fecha porque o proprio bloco `(C)` dele dizia "Nada": e registro, nao pendencia.
Cada um re-verificado contra a arvore viva antes de sair, nao pelo que eu lembrava.

### ~~2.10 Cobertura de auto-teste dos catorze checkers — registro, nao pendencia~~

> *Titulo anterior, mantido para rastreabilidade (o item foi encurtado antes de ser arquivado, e uma busca por titulo antigo tem de acha-lo): "Dos catorze checkers: sete se auto-verificam de verdade, um tem auto-teste que nao morde, dois tem fixtures, quatro nao tem nada".*

> **POR QUE SAIU DO TRACKER (movido em 2026-07-30).** FECHADO como registro: o proprio bloco `(C)` do item dizia **"Nada"** de voce. E uma medicao concluida (sete dos catorze checkers morrem quando sabotados, um tem auto-teste que nao cobre o detector, dois tem fixtures, quatro nao tem nada) mais a regra V13. `make selftest` nomeia os nao-provados em cada execucao, entao o gap fica visivel sem o tracker. rc=0.

**O que e.** Medido por sabotagem (quebrar o detector e ler o codigo de saida), nao por procurar um
`def self_test`: **sete** morrem quando quebrados, **um** tem auto-teste que nao cobre o proprio
detector (`check_negative_parallelism`), **dois** tem fixtures, **quatro** nao tem nada.
`make selftest` nomeia os nao-provados em cada execucao em vez de omiti-los.

**(C) Nada de voce.** A tabela completa e a regra que saiu disso (`AGENT_GUARDRAILS` §4b V13) estao em
[`_round8/29_pendencias_detail.md`](_round8/29_pendencias_detail.md). Fica aqui so como registro do
que ainda nao tem prova.

---

### ~~2.4 A secao de escopo da tarefa estatica~~

> **POR QUE SAIU DO TRACKER (movido em 2026-07-30).** FECHADO pela sua decisao, as duas partes feitas. O Apendice B saiu do corpo principal (rodada 7, esta no volume suplementar) e a prosa do B.5 foi reescrita em 2026-07-30. Verificado na prosa viva: a frase que voce citou tem **0** ocorrencias, o texto diz que a entrada e o nome do proprio local (4 ocorrencias), e a faixa 284-365 esta re-confirmada contra o pipeline. Renderiza na p. 11 do volume extra.

Sua decisao tinha duas partes e **as duas estao feitas**: o Apendice B saiu do corpo principal (rodada
7, esta no volume suplementar) e a prosa do B.5 foi reescrita em 2026-07-30. Verificado: a frase que
voce citou (*"built from a fine-grained class label attached to each place"*) nao existe mais, o texto
agora diz que a entrada e o **nome do proprio local**, e a faixa 284-365 esta re-confirmada contra o
pipeline. Renderiza na p. 11 do volume extra.

*Este bloco fica aqui uma rodada para voce conferir e depois vai para o arquivo.*

---

### ~~2.6 A coluna do CBIC que nao reproduz~~

> **POR QUE SAIU DO TRACKER (movido em 2026-07-30).** FECHADO pela sua decisao *"documentar no left_out.md"*. `LEFT_OUT.md` LO-2 carrega o achado (tres das quatro colunas publicadas do CBIC reproduzem exatamente, 21 de 21 celulas cada; a quarta nao casa com nenhum artefato) e a restricao que ele impoe. A linha "Open for the author" dentro do LO-2 era anterior a sua decisao e foi removida.

Sua decisao foi *"documentar no left_out.md"*, e esta feito: `LEFT_OUT.md` LO-2 carrega o achado
completo (tres das quatro colunas publicadas do CBIC reproduzem exatamente, 21 de 21 celulas cada; a
quarta nao casa com nenhum artefato deste repositorio) e a restricao que ele impoe — nenhuma frase
futura pode dizer que os numeros do CBIC sao reproduziveis sem excluir essa coluna.


**A sua palavra, verbatim:** *"DECISSAO: Documentar no letf_out.md"*.

**O que o item media, restaurado 2026-07-30:** a coluna que nao reproduz e a de **proxima-categoria do
modelo conjunto**; as outras tres conferem **21/21 celulas** cada. Nao ha erro conhecido no numero
publicado — o que falta e a **execucao (rundir) que o gerou**, e a pergunta que eu fiz era se existe um
fora deste repositorio.

*A linha "Open for the author" dentro do LO-2 e anterior a sua decisao e esta obsoleta.*

---

### ~~2.22 Apendice F: o revisor de excelencia aprova, mas a assinatura e sua~~ (arquivado 2026-07-30)

**FECHADO PELO AUTOR.** Ele aprovou: *"I approve the appedix F you can remove the `[NEEDS SIGN-OFF]`,
but i have some considerations"*, seguido de quinze pontos numerados (0-14). Todos os quinze foram
aplicados na rodada 9c e o marcador saiu do fonte.

**O que mudou, medido no PDF renderizado e nao no relatorio de nenhuma esteira:**
- **Ponto 0, a renumeracao.** O volume principal imprimia A, C, E, F porque os antigos B e D foram para
  o volume suplementar. Agora imprime **APPENDIX A (p.90), B (p.93), C (p.94), D (p.97)**, em sequencia.
  O apendice do cosseno e o **D**. O volume suplementar mantem as letras historicas B e D, e o paragrafo
  "About this volume" explica a ambiguidade em vez de escondê-la.
- **Pontos 1 e 4, as duas citacoes.** `standley2020tasks` para "can end up worse at both than two
  dedicated models are at one each" e `yu2020pcgrad` para o cosseno como a quantidade que a literatura
  de gradient surgery usa para definir conflito. Nenhuma entrada nova foi inventada: as duas ja estavam
  no `references.bib`, que e a resposta correta mais barata.
- **Pontos 3 e 13, o arco dos tres estudos.** Ele estava certo, e a medicao e mais forte do que ele
  colocou: o Cap. 3 e o Cap. 4 preveem classificacao de categoria e proximo POI com hard sharing e FiLM;
  o Cap. 5 e este apendice preveem proxima categoria e proxima regiao sobre um tronco de cross-attention
  sem camadas ocultas em comum. **Nem o par de tarefas nem a arquitetura sao compartilhados.** As duas
  sentencas foram REMOVIDAS, nao reescopadas: qualquer versao reescopada seria especulacao que os dados
  nao sustentam.
- **Pontos 2, 5, 6, 7, 9, 10, 11, 12:** cortes e simplificacao. As frases que ele citou estao todas em
  zero ocorrencias na prosa viva. O ponto 11 agora nomeia os datasets (Alabama e Georgia), conferidos
  contra a saida de `cosine_stats6.py`.
- **Ponto 8, o ingles britanico.** "feature needs saying plainly" -> "must be stated plainly". A regra e
  o gate nasceram deste ponto: `check_register.py` (gate 25) cobre grafias E construcoes, porque a
  instancia dele era uma construcao needs+gerundio e uma lista de grafias teria perdido justamente o
  achado dele.
- **Ponto 14, a qualificacao.** Uma sentenca em D.3: gradientes ortogonais nao significam que as tarefas
  nao compartilham conhecimento, porque os dois fluxos ainda trocam informacao pelo tronco de
  cross-attention.

**Onde ler a apuracao ponto a ponto:** `_round9/43_apxf_author.md`, com as palavras dele, o que foi feito
e a evidencia renderizada de cada um.

**A DECISAO DELE, VERBATIM**, preservada aqui porque os quinze pontos numerados sao o registro do
que ele pediu e nenhum resumo substitui as palavras dele:

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



