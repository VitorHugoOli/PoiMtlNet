# HANDOFF_PPT.md — para quem assumir a implementação dos slides

> **Escrito 2026-08-26, noite, pela sessão `ppt`. Defesa: sexta, 28/08, 10:00, remota (Google Meet).**
> Banca: Fabrício A. Silva (orientador/presidente), Clayson S. F. de Sousa Celes (ITA, externo),
> Alex Borges.
>
> **Quem aprova é o autor.** Não existe portão com o orientador para a apresentação.
>
> **Este documento é sobre o que a minha sessão errou e como não repetir.** O estado e as armadilhas
> de build estão no [`HANDOFF.md`](HANDOFF.md); o método, no [`HANDOFF_SLIDES.md`](HANDOFF_SLIDES.md);
> o conteúdo, no [`HANDOFF_GATE.md`](HANDOFF_GATE.md). **A lei da forma é
> [`BOAS_PRATICAS_SLIDES.md`](BOAS_PRATICAS_SLIDES.md)** — onde este arquivo divergir dela, ela vence.

---

## 0 · A divisão de responsabilidades, que é o que torna isto trabalhável

| | quem | o que decide |
|---|---|---|
| **conteúdo** | `gate` | o que é dito, se é verdade, se a literatura sustenta |
| **forma** | `ppt` (esta sessão) | como o conteúdo validado vira slide: hierarquia, composição, densidade, legibilidade, consistência |
| **figuras** | `tikz` | os diagramas, em tamanho final |
| **aprovação** | **o autor** | tudo que decide *o que não vai ser dito* |

**A regra que fecha a fronteira, e ela custou um erro para ser descoberta:**

> **O `gate` diz o conteúdo. O `ppt` diz quantas linhas ele ocupa.**

O `gate` especificou "figura + duas linhas" para os slides do DGI e do HGI. **Eram duas *frases*, e elas
quebravam em cinco linhas físicas.** Os dois orçamentos batiam no papel e nada cabia na tela. Quando
receber uma especificação em "N linhas", **converta antes de aceitar**: a 140 mm de `\textwidth`,
`\footnotesize` cabe ~15 palavras por linha.

⚠ **Não redija ciência.** Quando o `gate` apontar um defeito sem dar redação — aconteceu com o
`:1400`, que apontava uma coluna de desvio inexistente —, **aplique o que ele redigiu e devolva o
resto**. Inventar a frase certa ali é assumir o papel dele.

---

## 1 · O build. Esta é a regra que eu violei o dia inteiro

```bash
cd articles/dissertacao/presentation/slides
make all          # xelatex · bibtex · xelatex · xelatex -- TRÊS passes, em build/
```

**NUNCA** `xelatex main.tex` na mão, e **nunca** dois passes.

⚠ **`make all` leva 2–4 min. O tempo padrão do Bash é 120 s.** Rodar sem `timeout` maior faz o
build ser interrompido no meio e **deixa `build/main.log` pela metade** — e um log parcial não tem
`Overfull` para achar, então **a contagem dá zero e parece um deck limpo.** Isso me pegou três
vezes numa noite.

**Antes de ler o log, confirme as duas coisas:**
```bash
grep -c 'Output written' build/main.log     # 1 = completo, 0 = parcial
[ main.pdf -nt main.tex ] && echo ok        # PDF mais novo que a fonte
```
**Log com mtime MAIOR que o do PDF significa build em curso.** Qualquer número dali é lixo.

**O que dois passes produzem:** o divisor da Seção 5 renderizou **sem o fundo em degradê** — texto
branco sobre branco, os seis títulos do sumário invisíveis. O `remember picture, overlay` do
`nesped.sty` precisa das posições do `.aux` **convergidas**, e enquanto o conteúdo ainda se move dois
passes não convergem.

**NUNCA `pdflatex`**: compila sem erro e a capa, o sumário e todo `\specialframe` saem sem fundo.

---

## 2 · O catálogo de erros desta sessão, e o padrão que os une

**Todos têm a mesma forma: o instrumento funcionou e respondeu outra pergunta.**

| # | o que eu fiz | o que o instrumento disse | o que era |
|---|---|---|---|
| 1 | reportei ✔ 0,906 para o slide 44 | verdade — **da página 52** | o slide era a **51**, com `Overfull` de 24,4 pt e uma condição inteira cortada fora da página |
| 2 | `grep 'fine class'` | 0 ocorrências | a fonte quebra a linha **entre as duas palavras** |
| 3 | "zero `Overfull` novo" na nota do 27 | verde no `\hbox` | o defeito era `\vbox` — **10,5 pt**, e a tinta subiu para 0,973 |
| 4 | "o total declarado está superestimado" | verdade **para o S45** | no agregado **subestimava em 9:22** |
| 5 | "os overlays são +4:00 do excesso" | — | **minha própria tabela somava +2:36** |
| 6 | medi 444 palavras de deriva na fala | verdade | varri o bloco `% FALA` inteiro, que **também guarda notas datadas de projeto** |
| 7 | contei a fala do S34 em 8 s | verdade sob a minha regra | a regra de parada listava `Próxima` como marcador de nota, e **a fala começa com "Próxima região é…"** |
| 8 | atribuí falas a frames por ordem | verdade | as falas de **divisor de seção** não têm frame próprio e contaminavam o frame seguinte |

**Três checagens que teriam pego todos:**

1. **ancore a verificação no número impresso lido do PDF na mesma execução**, nunca num índice de
   página lembrado;
2. **quando afirmar um agregado, some-o**. Não infira a direção de um caso;
3. **pergunte o que o instrumento NÃO responde** antes de reportar o verde dele.

### 2.1 · A quarta classe, que nenhum instrumento pegava

O divisor sem fundo **passou pelos três**: o `pdftotext` extraía os seis títulos, nenhum `Overfull`
disparou, e a varredura de tinta **classificava a página como fundo cheio e a excluía** — a regra que
existe porque `\specialframe` e `\tocframe` são legitimamente escuros era a que escondia o defeito.

**Quem achou foi o autor, olhando.**

**Corrigido no instrumento** (`ink_sweep.py`, 26/08): numa página que deveria ter fundo, a regra
inverte e **exige tinta em vez de ignorá-la**. Validado contra o render da página quebrada:

| | tinta na área útil | |
|---|---:|---|
| divisor quebrado | **0,02 %** | dispara |
| divisor saudável | 99,0 % | excluído como fundo cheio |

### 2.4 · Contra reescrita, o `diff` é estruturalmente cego

**A pergunta certa não é "os dois textos batem?", é "o que entrou no original DEPOIS que a
reescrita nasceu?"**

O `diff_fala.py` compara fala↔fala. **Entre uma fala e a reescrita dela a similaridade é 0,20–0,37
por desenho** — a reescrita é 50% menor. A ferramenta reportaria divergência máxima, estando certa
e sendo inútil. ⚠ **É a única classe para a qual ela é cega por construção.**

**Caso real (27/08):** cinco falas curtas escritas em 26/08. Comparar não achou nada. **Procurar o
que entrou depois achou:** o `gate` acrescentara um "asterisco honesto" à fala do protocolo 4 no
dia seguinte, e a reescrita não o tinha. **A tela carrega um `*` cujo referente estaria só na
versão antiga** — rodapé órfão, que é a divergência tela↔fala ao contrário e a pior das duas.

### 2.5 · Imitar o vizinho propaga o defeito do vizinho

**Eu medi que `\textcolor{gray}` (= `black!50`) dá 3,95:1 e reprova a WCAG AA. Escrevi no
`BOAS_PRATICAS`. E no dia seguinte usei `\textcolor{gray}` num rodapé que escrevi.**

**Não foi esquecimento — foi imitação.** Copiei o rodapé de um frame vizinho para acertar o idioma
da seção, e vieram junto a cor errada e o formato. **Levei o meu próprio achado para dentro do erro
que ele descrevia.** Os 49 casos iguais na Série B vinham todos da mesma imitação.

⚠ **E o conserto criou uma superfície de erro que não existia.** O rodapé velho dizia *"migrado em
27/08"* — **não tinha número nenhum para estar errado**. Ao escrever proveniência de verdade,
escrevi `Figura 4` onde é `Figura 6, p. 80`. **Fazer certo custa uma verificação que fazer
parecido não custava.**

### 2.3 · Um número derivado escrito à mão não sabe que ficou velho

Formulação do `gate`, 26/08, depois de encontrar **37 dos 49 campos `Slide impresso:` do
`SLIDES.md` deslocados em +1** — um bloco removido tinha ficado na lista ativa e empurrou toda a
numeração a partir dele.

⚠ **O perigo é o silêncio.** Se a instrução *"inverta os impressos 13 e 14"* viesse dessa tabela,
os slides invertidos seriam os errados — **e nada reclamaria**: os dois números existem, os dois
frames existem, compila, o `Overfull` não muda, a tinta não muda, o `pdftotext` extrai tudo.
**Nenhum instrumento de página vê.**

**A defesa, nas duas superfícies:**

| onde | como |
|---|---|
| **na tela** (LaTeX resolve) | `\label` no frame alvo, `\ref` no que cita. Testado sob o tema nesped: devolve o número impresso correto |
| **em documento** (não resolve) | **re-derivar do artefato e verificar por título**, nunca ajustar à mão |

**Nunca ajuste à mão** — ajustar à mão é como os 37 sobreviveram.

⚠ **E marque o tempo quando um comentário citar número.** Um comentário que diz *"o **antigo**
impresso 37"* é seguro; um que diz *"o impresso 37"* é indistinguível de um que envelheceu.

### 2.2 · A quinta classe: a fala e a tela divergem

Quando o corpo de um slide muda, **não envelhece só o título — envelhecem o título, a fala e a
proveniência.** Nenhum dos três dispara `Overfull`, e nenhum instrumento de página os alcança,
**porque o defeito não está na tela: está na diferença entre a tela e a boca.**

Casos reais: o slide 3 perdeu *"point-of-interest prediction"* da tela e a fala continuou dizendo-o;
o `M29` tirou *"acima do HMRM em todas as categorias"* da tela e a fala manteve; o `M23` tirou
*"o Capítulo 5 não se apoia nisso"* e a fala manteve.

**A varredura** — cinco linhas de regex sobre os blocos `% FALA` da trilha principal, com o frame
dono resolvido por posição — procura **os termos que se sabe terem saído da tela**:

```python
ALVOS = {'pontos de interesse': 'saiu da ancora', '95 por cento': 'o CI virou 90%', ...}
# para cada linha de comentario que casar, resolve o \begin{frame} seguinte e reporta os dois
```

⚠ **Confira cada achado lendo o contexto antes de reportar.** Na primeira rodada, **2 das 4
ocorrências eram falsas** — uma era a ressalva do MobiWac, que diz "submetido" de propósito, e a
outra dizia "computação" no sentido certo do par `operational / computational`.

⚠ **O limite, que precisa ser declarado junto:** ela só acha termo que alguém **sabe** ter tirado.
**Uma fala que envelheceu por mudança de sentido sem mudança de palavra passa direto** — essa é a
triagem semântica do `gate`, e esta varredura não a substitui.

---

## 3 · Padrões de forma que valeram altura, sem cortar palavra

- **`\begin{center}` é uma `trivlist` e cobra `\topsep` dos DOIS lados.** Trocar por
  `\noindent\makebox[\textwidth]{…}` devolve **~8 pt por figura**, de graça;
- **a moldura de um `block` custa ~28 pt de cromo sem texto.** Primeira coisa a examinar quando um
  slide estoura por poucos pontos;
- **fórmula em `\[ \]` custa duas linhas a mais que a mesma fórmula inline**;
- **rebalancear colunas** (42/55 em vez de 48/48) tira duas linhas sem cortar uma palavra;
- **conteúdo centrado cresce para os dois lados**: a altura absorvível é `2 × min(folga acima, folga
  abaixo)`, medida a partir da base real da faixa de título **daquela** página.

### 3.0 · Quem recusa fica devendo a revisita

A `tikz` recusou separar a discriminação do fluxo na chapa do HGI, **por uma razão medida e
verdadeira na hora**: as entradas do `POI–Region` teriam de atravessar uma faixa inteira e o
resultado era um novelo. Depois ela introduziu os conectores A/B **para outro problema** — e eles
derrubavam a recusa anterior como consequência direta. **Ninguém voltou a testar.** Só reabriu
porque o autor insistiu, e a chapa nova é melhor **e mais baixa**.

> **Quem recusa fica devendo a revisita quando a própria caixa de ferramentas muda — e o dono da
> recusa é quem tem menos chance de notar, porque já resolveu aquilo na cabeça.**

*(Formulação da `tikz`, corrigindo a minha, que era mais leniente: eu tinha escrito "sem ninguém
perceber", como se fosse acidente sem dono.)*

### 3.1 · A regra que decide corpo quando dois elementos competem

> **O que a tela precisa carregar sozinha tem prioridade de corpo sobre o que a fala carrega junto.**

Os números de uma tabela nunca são lidos em voz alta; as ressalvas obrigadas pela §8.6, sim. Por isso
a ressalva metodológica do HGI ficou na tela em `\footnotesize` e a explicação causal foi para a fala.

### 3.2 · ⚠ A armadilha do subtítulo — três reincidências

`\begin{frame}{título}{grupo}` é sintaxe válida. **Se o corpo abrir com `{`, o beamer lê o grupo como
subtítulo** e o conteúdo renderiza dentro da faixa colorida — com log limpo e `pdftotext` extraindo
tudo. **Uma linha em branco depois do título também encerra a varredura do argumento opcional.**

O guarda é uma linha que se declara:

```latex
\begin{frame}{Título}
    \vspace{0pt}% ⚠ GUARDA -- NAO REMOVER. Sem algo que nao seja "{" abrindo o corpo, o
    % beamer le o grupo seguinte como SUBTITULO. Custa zero altura.
    {\small ...
```

> **Um guarda que parece folga vai ser removido por quem estiver otimizando folga — inclusive por
> quem o pôs.** Eu removi o meu numa varredura de espaçamento e reintroduzi o defeito.

---

## 4 · O pipeline do SPEECH

```
SLIDES.md  →  build_speech_1_extract.py  →  build_speech_2_emit.py  →  make -f Makefile.speech
```

**O extrator lê só quatro campos:** `Tempo:`, `Fala (PT):`, `Nunca dizer:`, `LEDGER:`.
⚠ **`Na tela:` e `Proveniência:` NUNCA chegam ao SPEECH.** Regenerá-los não muda o roteiro — o valor
deles é servir de referência a quem escreve a fala.

**Ordem obrigatória:** `SLIDES.md` → reconstruir o deck → regerar o SPEECH. Gerar antes produz um
roteiro de um deck morto, **que é pior que um roteiro velho porque parece novo**.

**Dois defeitos corrigidos em 26/08:**
- o regex era `### (S\d+) · ` e **não casava com sufixo de letra**. `S5b` (o slide das tarefas) e
  `S26b` eram **invisíveis** — duas telas iriam para a defesa sem fala nenhuma, sem nenhum aviso.
  Agora `### (S\d+[a-z]?) · `;
- consertar só o regex **quebrava o emissor**, que fazia `int(code[1:])` em cinco pontos. Chave
  numérica `_n()` que ordena `S5b` logo depois de `S5`.

⚠ **Valide o emissor num sandbox, nunca no lugar.** A minha primeira correção deixou escapar uma
quinta ocorrência e o emissor explodiu — se eu tivesse rodado na pasta real, teria destruído o SPEECH.

### 4.2 · ⚠ A fala é o que está ENTRE ASPAS

Descoberto 26/08. **Doze dos 56 blocos carregavam anotação editorial dentro do campo
`Fala (PT)`** — um parentético de versão antes da aspa de abertura, uma nota de revisão depois da
de fechamento. O extrator lia o campo inteiro, então o roteiro impresso traria
`*(v3, 26/08 — a primeira frase foi refeita…)*` **no meio do que o autor lê em voz alta.**

**Não era visível de lado nenhum:** no `SLIDES.md` a anotação está formatada e é o `gate`
escrevendo para si mesmo; no `.tex` ela não existe. **Só aparece no que o extrator produz.**

**A regra, medida antes de ser adotada:** 55 dos 56 blocos põem a fala entre aspas e a anotação
sempre fora delas. `so_a_fala()` faz, **nesta ordem**:

1. remove `*(v… )*` e tudo a partir do primeiro `⚠`;
2. toma o que está entre a primeira e a última aspa;
3. cai para o campo inteiro se não houver aspas.

⚠ **A ordem não é arbitrária.** Limpar as aspas primeiro falha: uma nota de revisão pode **citar
outra nota entre aspas**, e o `rfind` cai dentro da citação. **Editorial primeiro, aspas depois.**

**E o passo 3 é silencioso por desenho** — daí o guarda `_guarda_editorial()`, que falha o build
com o código do bloco se sobrar marca editorial. Cobertura medida com casos sintéticos: um bloco
sem aspas com `*(v…)*` ou `⚠` **já está coberto pela pré-limpeza**; o guarda pega **marca
editorial em prosa**, que é a que ninguém antecipa.

⚠ **O guarda NÃO enumera marcas, e a razão é medida.** O `gate` usa **19 formas** de anotação e
**inventou 4 delas num único dia** — uma lista persegue um alvo que ele move sozinho. A invariante
substitui as 19:

> **Nenhuma fala legítima tem crase.** O autor não diz caminho de arquivo, nome de variável nem
> número de seção em voz alta. Mas **toda** marca editorial que sobreviva ao `so_a_fala` carrega
> uma — `` `AUT-19` ``, `` `main.tex:759` ``, `` `§8.11` ``.

Medido nos dois lados antes de adotar: **0/56 falas extraídas têm crase**, e **3 campos brutos
têm** — isto é, zero falso positivo e a limpeza está funcionando. O glifo de status
(`✅🔴🛑⟵⚠`) entra junto: é a marca usada quando não se põe crase. **Testado: dispara inclusive
numa forma que o `gate` ainda não inventou.**

### 4.1 · Contar a fala

Regra: **tokens separados por espaço, sem filtro** (qualquer filtro subestima 15–20 % num idioma
latino). Divida por **140 ppm**.

- **o bloco de fala termina** no primeiro comentário que seja nota de projeto (`% NOTA`, data ISO,
  `% [BLOCO`, `⚠`). **Não liste palavras comuns como marcador** — foi assim que `Próxima` decepou a
  fala do S34;
- **pareie fala↔frame por posição no arquivo**, nunca por título: quatro overlays compartilham
  `\frametitle` e uma chave por título colapsa três deles;
- **uma fala que não precede imediatamente um frame é divisor de seção** e não pertence a nenhum.

---

## 5 · Estado do deck em 26/08, 19h

- **105 páginas · 48 slides impressos** — estado de 27/08, 02h
- **os 6 divisores e a capa desenharam o fundo** (verificado por luminância, não por fé)
- **0 colisões** com o número do frame · tinta máxima **0,977** · mediana **0,914**
- **17 páginas acima de 0,93**, duas delas deliberadas e registradas: **41 · The verdict** e
  **24 · The diagnostic result** — ambas completas, nada cortado
- **as chapas do DGI e do HGI integradas** (slides 14 e 22), sem `width=` — vêm em tamanho final,
  **`Overfull` zero nas duas** (DGI 50,4 mm, HGI 123,7 × 57,2 mm com a discriminação em faixa própria)
- **fala escrita: 7.607 palavras = 54:20 a 140 ppm.** Com as cinco reescritas do `gate` pendentes de
  aprovação do autor: **6.900 = 49:17**

⚠ **"49:17" e "cabe" não são a mesma frase.** É leitura corrida, sem pausa e sem uma frase a mais
para responder um olhar da banca. O Art. 23 é **teto**, não alvo.

- **slide 3** reconstruído na moldura do impresso 6 do template: frame comum, título + parágrafo
  em `\Large`, fora do `\specialframe` (ver §5.7 do `BOAS_PRATICAS`). A pergunta vem do objetivo
  geral do volume entregue, sem parênteses
- **slide 30** (`AUT-18`): só a imagem. ⚠ **A `fig1_dataflow` estava a `0.50\textwidth` sendo que
  o tamanho natural dela é 136,6 × 62,4 mm** — exibida pela metade, com o texto interno abaixo do
  piso de legibilidade. Os quatro marcadores é que a espremiam; a fala carrega os quatro
- **varredura visual** das 9 páginas mudadas, renderizadas a 1230 px (tamanho do Meet): todas OK

### 5.0b · A bijeção dos links não é frame↔alvo

**A regra de aceitação é `\hyperlink` ↔ `\hypertarget`: todo código resolve, nenhum alvo fica sem
link.** ⚠ **NÃO é "um alvo por frame".** Depois da reescrita da Série B, códigos fundidos mantêm o
próprio `\hypertarget` no frame de destino — é assim que um botão antigo continua resolvendo.
**Uma conferência que espere `N frames == N alvos` reprova um deck correto, com autoridade.**

Hoje: **50 frames, 50 alvos.** Previsto após a reescrita: **50 frames, 55 alvos.**

### 5.1 · Em aberto — tudo é decisão do autor

| | |
|---|---|
| **o corte de ~4 min** | 49:17 como está · −180 palavras para 48:00 · −320 para 47:00 |
| **as cinco falas curtas** | medidas (707 palavras cortadas), **não aplicadas** |
| **`S46`** | o `gate` segura a sincronização até o corte ser decidido |
| **SPEECH não regerado** | ordem obrigatória: `SLIDES.md` → reconstruir → regerar |
| **gênero da figura do 30** | `c2h_flow.pdf` pronto; **não bloqueia nada** — trocar é uma palavra |

---

## 6 · O protocolo, em ordem

```bash
cd articles/dissertacao/presentation

# 1 · SLIDES.md primeiro, depois propague para slides/main.tex
# 2 · TRÊS passes, sempre
cd slides && make all

# 3 · canária -- o último elemento visível de cada frame tocado aparece?
pdftotext main.pdf -            # SEM -layout

# 4 · tinta, e agora também a página pálida (§2.1)
cd .. && python3 ink_sweep.py slides/main.pdf

# 5 · OLHE, no tamanho real do Meet
pdftoppm -f <p> -l <p> -png -scale-to-x 1230 slides/main.pdf /tmp/s

# 6 · SPEECH só DEPOIS de reconstruir o deck
python3 build_speech_1_extract.py && python3 build_speech_2_emit.py
```

**O que cada passo não responde:**

> **Extrair o PDF pega termo errado; só renderizar pega conteúdo ausente.**
> **E só olhar pega o que foi desenhado em branco sobre branco.**

**Antes de escrever qualquer edição em lote: valide TODAS as âncoras primeiro, escreva só depois.**
Um `sys.exit` antes da primeira escrita já me impediu duas vezes de corromper arquivo alheio — uma
delas porque `MobiWac 2026, submitted` tinha duas ocorrências e a segunda era nome de artefato.

---

# 7 · A rodada de 27–28/08 — o que ela ensinou que as anteriores não tinham

Um dia inteiro de execução com o `gate` e o `extra`, na véspera da defesa. Sete classes novas, e a
maioria é sobre **instrumentos que funcionam e respondem outra pergunta** — a família que este
documento já registrava, agora com casos que a nomeiam melhor.

## 7.1 · 🛑 `columns` que transborda NÃO gera `Overfull`

**O defeito mais caro do dia, e ele quase foi para a defesa.** O slide 32 em `\small` perdia a
palavra `MERGE` — a última da frase que a própria fala designa como *"a que eu quero que fique da
tela"*. Log limpo, `Overfull` zero, metade da frase-chave fora da página.

> **Uma coluna de `columns` estoura em silêncio.** A varredura de caixa do deck é contagem de
> `Overfull`, então **todo slide de duas colunas esteve fora do alcance do instrumento principal**
> — e são ~11 frames, incluindo o índice B0.

**O instrumento:** `canaria_coluna.py`. Extrai a última palavra visível de cada `column` do `.tex` e
procura-a — **palavra única, nunca frase** — no `pdftotext` **daquela página**. Roda também o teste
dos botões do índice.

⚠ **Ela própria teve dois defeitos, e ambos são a lição:** colhia nome de cor de dentro de
`\textcolor{secondary}{...}`, e quando não localizava a página **caía para procurar em todas** — o
que transforma qualquer palavra comum num verde sem valor. **Agora responde `?  (página não
localizada)`.** Uma canária que não sabe dizer *não sei* não é canária.

## 7.2 · 🛑 O rótulo de um botão de índice é indistinguível do código do slide

**Três vezes no mesmo dia.** Qualquer busca por `\textbf{B6-6}` ou `\textbf{B-P1}` encontra **o botão
do índice B0**, porque o índice vem antes no ficheiro. Consequências reais:

- o detector de `Overfull` mediu o índice e respondeu **`✅ CABE`** num frame **45 pt fora**;
- uma medição de folga inteira foi feita no frame errado e reportada como boa;
- eu quase reportei ao `gate` que um slide não continha um termo que continha.

> **A âncora certa é o `\framesubtitle` do próprio frame**, resolvido dentro do intervalo
> `\begin{frame}…\end{frame}`. Nunca a string do código solta. Está em `/tmp/_ov.py` e devia ser
> promovido a ferramenta versionada.

## 7.3 · 🛑 Substituir um intervalo de frames leva as falas que estão entre eles

A fala vive no `% FALA:` **fora** do frame. Ao trocar dois frames por dois novos, **apaguei 57
palavras de fala** sem perceber — e nenhuma verificação de PDF a apanha, porque comentário não
renderiza.

> **A regra: contar os `% FALA:` antes e depois de qualquer substituição de intervalo.** Usei-a nas
> três migrações seguintes: `96 → 96 → 96`.

## 7.4 · Palavras não preveem altura — nos dois sentidos

| o que se pediu | o que custou |
|---|---|
| `+8 palavras` (par `hard`/`soft`) | **+27,64 pt** — o `\\` força duas linhas novas |
| *"corta as caixas 1 e 2 antes da 3"* | **zero** — a altura de uma linha de `minipage` é o **máximo** das três |
| tirar uma coluna 100% derivável | **zero** — o estouro era contagem de linhas, não largura |
| rebalancear larguras de tabela | **piorou** (11,56 contra 2,06) — estreitar uma coluna empurra linhas |

> **A fronteira, formulada com o `gate`:** *quando a minha alavanca muda **quais palavras estão na
> tela**, ela é dele; quando a spec dele contém uma instrução de composição (`\\`, `\vspace`,
> `minipage`, `tabular`), o custo é meu de medir.* **Nenhum dos dois prevê o lado do outro.**

## 7.5 · ⚠ Remover metade de um par simétrico não remove metade do efeito

Tirei só o `\vfill` de baixo de um frame: **de +6,1 para +22,6 pt.** Sem par, o de cima empurra tudo
para o fundo. **Forma pura, erro inteiro meu.**

## 7.6 · Sondas que fazem no-op silencioso

`str.replace(a, b)` devolve a string **intacta** quando `a` não existe. Duas sondas minhas mediram
ficheiros que não continham o que eu pensava estar a medir, e **reportaram `✅ CABE`**.

> **Toda substituição — em sonda também, não só em aplicação — leva asserção de contagem antes.**
> A partir daí as sondas passaram a recusar (`🛑 4 ocorrências, esperava 1`) em vez de mentir.

## 7.7 · Réguas que medem o próprio cromo

A varredura de tinta reportava **50 páginas** acima de 0,96. **Era o número do frame**, que vive na
faixa direita e desce a ~0,987 em toda página. Excluindo os 7% à direita: **16 páginas**, das quais
**uma** na trilha principal.

> **Quarta régua do dia a incluir no alvo alguma coisa que está lá por desenho** — junto com o
> `Overfull` ancorado no `\begin`, o detector que achava o índice, e a canária que caía para "todas
> as páginas". **O padrão é sempre o mesmo.**

## 7.8 · Duas coisas que a forma resolveu melhor que o corte

- **O `B7-3` pagou a dívida declarada com CONTEÚDO, não com cromo.** Eu tinha escrito *"as minhas
  alavancas de cromo estão esgotadas neste frame"* — e estavam. Dois cortes de prosa do `gate`
  compraram um **degrau inteiro de corpo** (`\footnotesize` → `\small`).
- **Dividir venceu caber.** O `B6-6` não comportava duas cláusulas novas mais o rodapé (medido:
  +2,06 · +17,28 · +36,28, com o rodapé a cair fora em silêncio). **Partido por eixo, tudo entra** —
  e cada ressalva fica ao lado da tabela a que pertence, que era o que o conteúdo já pedia.

## 7.9 · Coordenação: o ficheiro tem um dono de cada vez

Em 28/08 havia **duas sessões `ppt`** e a mesma spec podia chegar às duas.

> **Antes de escrever depois de uma pausa: `stat` no `main.tex` e conferir marcas conhecidas das
> próprias edições.** Uma resposta do outro agente é mais lenta e mais fraca que a evidência do
> ficheiro. E uma mensagem de coordenação pode ficar **retida à espera de aprovação** — não a
> esperes.

---

# 8 · A rodada de citações (28/08) — quatro classes que só aparecem ao inserir em massa

Quarenta e uma citações mapeadas, vinte e oito aplicadas, uma decisão devolvida ao autor.
As lições são sobre **inserir texto curto em muitos sítios de uma vez** — um regime que nenhuma
das rodadas anteriores tinha exercitado.

## 8.1 · 🛑 A unidade de custo é a LINHA, e linhas são quantizadas

Quatro medições no mesmo dia, todas contra a intuição de caracteres:

| pedido | previsto | medido |
|---|---|---|
| `+8 palavras` (par hard/soft) | pequeno | **+27,6 pt** — o `\\` força duas linhas |
| *"corta as caixas 1 e 2 antes da 3"* | libertaria espaço | **zero** — `minipage`: altura é o **máximo** |
| ordem de sacrifício do slide 21 | libertaria uma linha | **zero** — cortava a coluna **mais curta** |
| citação só com o ano (`251 → 84` caracteres) | `40,8 → 13,6 pt` | **+42,39 pt, idêntico** |

> **Caracteres não compram nada até removerem uma quebra.** E numa `columns`/`minipage`,
> **só a coluna mais alta paga** — cortar da outra é trabalho perdido por mais conteúdo que saia.

**O fluxo que funciona**, acordado com o `gate`: ele manda o conteúdo, **eu meço e digo onde o corte
tem efeito**, ele escolhe o quê. No slide 21 isso produziu um corte que era **redundância criada por
nós dois**, não sacrifício.

## 8.2 · 🛑 Um inseridor precisa de fronteira de palavra E de olhar à volta

Três defeitos distintos numa única passagem de 28 inserções:

- **`Sphere2Vec (Mai et al., 2023)-M`** — `str.find` casa `Sphere2Vec` dentro de `Sphere2Vec-M`.
  **A correção é `(?<![\w-])nome(?![\w-])`.** Também morde `STAN` dentro de `Standley` e `HGI`
  dentro de `Check2HGI`;
- **`(MTL (Caruana, 1997))`** — a citação entrou num parêntese que já existia. **Fronteira de
  palavra não resolve isto**: é preciso olhar o contexto. A saída melhor não é desaninhar (custou
  6,24 pt ao empurrar a quebra) — é **fundir**: `(DRRGNN, Zhu et al., 2022)`;
- **`\\scriptsize`** — o meu script de conserto escreveu barra dupla dentro de uma f-string. **Em
  LaTeX `\\` é quebra de linha e o build PASSA**, projetando uma quebra mais o texto literal.
  **Um conserto silencioso pior que o defeito.** Apanhado ao reler a linha, por nenhuma varredura.

## 8.3 · ⚠ Sucesso PARCIAL é mais perigoso que falha total

O aplicador resolvia o frame pelo número impresso. **Três versões, três resultados:**

    v1  exigia que o titulo casasse um padrao      ->  39 aplicadas, 6 falhas
    v2  contava TODOS os \begin{frame}             ->   2 aplicadas, 43 falhas
    v3  conta todos, exclui os comentados          ->  45 aplicadas, 0 falhas

> **A v1 é a perigosa.** Ela não deu erro — deu **39 sucessos**, e as 39 estavam em frames
> **deslocados**. As seis falhas fazem olhar para o sítio errado. Se eu não tivesse conferido as
> seis contra o PDF, teria medido outro deck e o `gate` teria decidido cortes com base nele.

⚠ E a causa da v2 é a mesma família do `% FALA:`: **eu próprio escrevi `% \begin{frame}{...}` em
comentários a explicar sintaxe**, e cada um contava como frame. **O comentário é invisível para o
leitor e visível para a ferramenta.**

## 8.4 · Um instrumento que acerta por acidente encerra a investigação

O `gate` foi confirmar a margem de 2 pp e encontrou `tost=0.02` em `finalize_phase3.py:267`.
**O número batia. A chamada era de outra etapa** (Check2HGI × HGI, não o teste do veredito).

> **Um instrumento que erra chama atenção; um que acerta pela razão errada fecha a pergunta.**
> O que o salvou foi o autor ter escrito *"confira se os 2 pp correspondem a esse teste e não a
> outra etapa"* — a pergunta certa veio de fora.

## 8.5 · Onde o deck ficou

    104 paginas · 6 Overfull, todos < 6,3 pt · 55/55 links · zero orfaos · 96 falas
    canaria de coluna: 21 colunas · canaria de indice: 54 botoes · SPEECH 48:23

**Aberto, e todo com outros:** as doze citações do slide 8 (não cabem em nenhuma das quatro formas
medidas — decisão do autor), o `tower` na Série B (`extra`), e a chapa `c2h_deep` a dizer
`pretrained` (`tikz`, com fallback armado).
