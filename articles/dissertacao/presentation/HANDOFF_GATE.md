# HANDOFF_GATE.md — para quem assumir o conteúdo da defesa

> **Escrito 2026-08-26, noite, pela sessão `gate`. Defesa: sexta, 28/08, 10:00, remota.**
> Banca: Fabrício A. Silva (orientador/presidente), Clayson S. F. de Sousa Celes (ITA, externo),
> Alex Borges.
>
> **Quem aprova é o autor.** Não existe portão com o orientador para a apresentação — decisão dele,
> registrada. Não reintroduza.
>
> **Este documento é o que a minha sessão sabe e os arquivos não dizem.** Os fatos estão no
> `considerations.md`; aqui está o que custou caro descobrir.

---

## 0 · Leia nesta ordem, e pare de ler quando puder agir

| # | arquivo | por quê |
|---|---|---|
| 1 | **este documento, §3** | os erros já cometidos. **É a seção que economiza o seu dia** |
| 2 | [`considerations.md`](considerations.md) **§✔ Decisões do autor** | 18+ decisões dele, com o efeito de cada uma. **Nada ali se reabre** |
| 3 | [`considerations.md`](considerations.md) **§🗺 As três numerações** | o deck tem três numerações em circulação. **Ancore por TÍTULO, sempre** |
| 4 | [`BOAS_PRATICAS_SLIDES.md`](BOAS_PRATICAS_SLIDES.md) | a referência de forma, escrita pela `ppt`. Medida, não lembrada |
| 5 | [`HANDOFF.md`](HANDOFF.md) **§3** | os sete casos de *"o instrumento passou porque mediu outra coisa"*. Hoje viraram onze |

**A lei da palavra** continua sendo `../WRITING_LAW.md`, `../GLOSSARY.md` (**fail-closed**) e
`../AGENT_GUARDRAILS.md`. **A lei da estrutura** é o `PLANO_FLUXO_DEFESA.md` §8 — **com duas regras
revogadas pelo autor hoje**: a §8.13 (contribuição duas vezes) e a §8.5 (carimbo em arte reproduzida).

---

## 1 · Quem faz o quê

| sessão | escopo | fronteira |
|---|---|---|
| **`gate`** *(esta)* | **conteúdo e estrutura.** O que entra em cada slide, em que ordem, com que palavras. Dono do `considerations.md` e do `SLIDES.md` | **não toca no `slides/main.tex`** |
| **`ppt`** | **implementação.** Única mão no `.tex`. Mede caixa, renderiza, verifica | **não decide conteúdo nem redige ciência** |
| **`tikz`** | **os diagramas.** Entrega `.tex` standalone; a `ppt` integra | **não toca no `main.tex`** |
| *(a definir)* | **a série B (extras)** — 49 páginas. **O autor reservou para outro agente** | ver §6 |
| ~~`presentation`~~ | **desligada.** O histórico dela está absorvido no `considerations.md` §4E | — |

> **A divisão funcionou porque cada um recusou o trabalho do outro.** A `ppt` parou várias vezes
> dizendo *"aqui inventar seria eu escrevendo ciência"*; a `tikz` levou decisões de glossário ao
> `gate` em vez de escolher. **Mantenha isso.**

---

## 2 · Estado, 27/08 ~12:00 — o dia inteiro depois

**Deck: 102 páginas · 48 impressos · 5 `Overfull` na trilha principal (máx 2,3 mm) · Série B em ZERO ·
51 links / 51 alvos · zero órfãos.** Fala: **48:10**.

> ⚠ **Remeça antes de citar.** O deck andou ~15 vezes em 27/08. **E use padrão CURTO ao grepar o
> PDF** — a extração quebra linha no meio de frases, e eu dei um falso alarme hoje procurando
> `"absolute scores are optimistic"` inteiro num texto que a quebra tinha partido.

### O que foi decidido e executado em 27/08

| | |
|---|---|
| **três inversões de ordem** | Fundamentos `5↔6` · Seção 3 `13↔14` · Seção 4 (a arte vai para o fim) — `AUT-21/22/23` |
| **o `Result 1` saiu da trilha** | migrou para a Série B; `Result 2/3` renumerados — `AUT-29` |
| **o último terço mudou de GÊNERO** | 42 · 43 · 45 · 46 · 47 — *"a plateia já não lê, só escuta"* — `AUT-24` |
| **os cinco cortes de fala** | aplicados · **684 palavras · 4:53** — `AUT-30` |
| **a Série B refeita** | 44 faixas de título, todas auditadas por mim · zero `Overfull` |
| **o `1,1 M` de parâmetros** | **resolvido**: era `644.359 + 417.117`, e o primeiro termo é o build defeituoso de 2 camadas. **AL 4,2 vs 1,85 (2,3×) · CA 5,2 vs 2,80 (1,8×)** |
| **`stream`/`tower`** | **revogados pelo autor** depois de os ter autorizado. Nome padronizado = **o nome da tarefa** |

### O que está PRONTO e esperando só a palavra do autor

**As cinco falas curtas** (`S41`–`S44`, os quatro passos do protocolo, e `S46`, o Result 3).
Escritas, entregues à `ppt`, **medidas, NÃO aplicadas**. **As cinco vieram abaixo do alvo** — cortam
**707 palavras**, contra as 598 previstas.

| | antes | depois |
|---|---:|---:|
| os quatro overlays do protocolo | 6:51 | **3:23** |
| o `S46` · Result 3 | 195 s | **99 s** |
| **deck inteiro** | 7.607 palavras · **54:20** | 6.900 palavras · **49:17** |

> 🔴 **A decisão que espera o autor.** Ele **já não escolhe entre "no teto" e "com folga"** — escolhe
> entre **49:17** e algo menor: **−180 palavras** leva a 48:00, **−320** a 47:00.
>
> **49:17 dá 43 segundos de margem**, e continua sendo margem de **leitura corrida** — sem pausa, sem
> hesitação, sem uma frase a mais para responder um olhar. O Art. 23 é **teto, não alvo**.
>
> ⚠ **E há um dado novo que ele não tinha:** as cinco reescritas couberam com **13% de sobra sobre o
> alvo**, o que sugere **gordura da mesma classe** (a fala recitando a tela) em outros blocos.
> ⚠ **Mas só um candidato sobreviveu à remedição: o `S48`** (119 s). O `S30` mede **75 s** contra 100
> declarados e o `S52` mede **78 s** contra 96 — **os campos `Tempo:` deles é que estavam altos, não a
> fala.** Eu os listei como gordura sem remedir.

---

## 3 · 🔴 OS ERROS QUE EU JÁ COMETI

**Leia isto antes de confiar em qualquer coisa que eu escrevi.** Não são deslizes isolados — são
**nove classes**, e cada uma vai voltar. *(Nove classes + seis avulsos = quinze.)*

### Classe 1 · Eu li o meu relatório em vez do artefato

**Três itens sumiram em silêncio.** A frase do Florida no slide dos datasets, a inversão dos itens do
slide 1 e a menção redundante no slide 2 estavam **especificados na §9 e nunca entraram em lote
nenhum**. Pior: **a minha reconciliação da §14 os listou como "na fila"** — eu conferi a spec contra a
minha própria fila em vez de contra o `.tex`. **O autor reclamou duas vezes, e as duas tinha razão.**

> **A regra que decorre:** *"na fila" não é um estado verificável.* Ou está no `.tex`, ou não está.
> **Todo lote que você mandar deve vir com a lista do que ele cobre, e você deve conferir a spec
> inteira contra os lotes despachados antes de dizer que uma seção está coberta.**

### Classe 2 · Eu usei o instrumento errado e reportei o verde dele

**Eu afirmei que `fine class` tinha zero ocorrências no deck.** Rodei `grep 'fine class'` — e **a
fonte quebra a linha entre as duas palavras**. O termo estava vivo em tela. A `ppt` o achou
**renderizando**.

> **Os dois padrões que ficaram, e nenhum substitui o outro:**
> - **auditoria de termo sobre o PDF EXTRAÍDO**, não sobre a fonte — a fonte mente por quebra de linha
>   e por macro;
> - ⚠ **e ela não pega conteúdo ausente**: o PDF só mostra **o que foi desenhado**. Conteúdo empurrado
>   para fora da caixa **não aparece na extração**. **Extrair pega termo errado; renderizar pega
>   conteúdo ausente.**

### Classe 3 · Eu escrevi contra uma sensação em vez de um alvo medido

**Especifiquei "figura + duas linhas" para os slides de diagrama.** Contei "duas linhas" como duas
linhas **físicas**; elas quebravam em **cinco**. **Conteúdo obrigatório caiu da caixa nos dois slides,
sem um aviso no log** — inclusive a ressalva metodológica do HGI, que a §8.6 obriga.

> **O contrato que ficou: eu digo o CONTEÚDO, a `ppt` diz QUANTAS LINHAS ele ocupa.** Nunca especifique
> em linhas. Peça o orçamento em palavras e escreva contra ele.
> **Referência: a 140 ppm, 1 s = 2,33 palavras.**

### Classe 4 · Eu parafraseei texto entregue e introduzi defeito

**Duas vezes no mesmo dia:**
- escrevi a pergunta de pesquisa do slide 3 **com dois travessões dentro de frase completa** —
  violando a regra que eu mesmo estava aplicando. **O texto entregue usa parênteses**
  (`1_introduction.tex:133-134`). A paráfrase não ganhava nada e criava a violação;
- recomendei **substituir a frase do critério do balanceador** por uma descritiva de literatura. O
  defeito real era outro: **o deck tinha cortado o prefixo de escopo `For this dissertation`** do
  texto entregue (`2_fundamentals.tex:1391-1392`). **Restaurar três palavras resolvia** — e elas
  **já estão no deck** (`main.tex:321`).

> **A regra:** quando algo na tela soa errado, **vá ao texto entregue antes de reescrever.** Nove vezes
> em dez o defeito é um corte, não a redação.

### Classe 5 · Eu inventei estrutura sem verificar quem a consome

**Criei os identificadores `S5b` e `S26b`** no `SLIDES.md`. O extrator do roteiro lê `### (S\d+) · ` —
**o sufixo de letra não casa. Duas telas iriam para a defesa sem fala nenhuma**, e uma delas é o slide
das tarefas, que é conteúdo novo. **Zero avisos: a ferramenta não conta o que não viu.**

> 🔴 **E aconteceu de novo no mesmo dia, no mesmo arquivo, com o mesmo consumidor.**
>
> Ao sincronizar as falas eu marquei cada bloco corrigido com um parentético — *"(v3, 26/08 —
> sincronizada com o deck)"* — **dentro do campo `Fala (PT)`**. O extrator lê o campo **inteiro**.
> **Doze blocos** iam para o roteiro impresso com a minha nota de bastidor no meio do que o autor
> lê em voz alta.
>
> **E era invisível dos dois lados que alguém olha:** no `SLIDES.md` a anotação está formatada e
> parece obviamente editorial; no `.tex` ela não existe. **Só aparece no que o extrator produz** — e
> o `SPEECH` estava desatualizado desde a véspera, então ninguém tinha visto.
>
> A `ppt` corrigiu no extrator (tira o editorial, **depois** procura as aspas — nessa ordem, porque
> a minha nota do `S51` tem aspas internas). **Verifiquei por fora: 12 → 0, nenhuma fala vazia.**
> E registrei a convenção no cabeçalho do `SLIDES.md`, porque **o conserto depende dela**: sem
> aspas, o extrator devolve o campo inteiro e o vazamento volta calado.
>
> **A lição é a mesma da Classe 5 e por isso ela está aqui e não nos avulsos:** *toda vez que você
> escreve num campo, pergunte quem mais lê esse campo.* Duas vezes em um dia, no mesmo arquivo.

> ⚠ **E só metade está resolvida.** O regex já aceita sufixo de letra (`build_speech_1_extract.py:30`)
> e o `S5b` resolve. **O `S26b` continua sem número impresso**: o título do bloco traz `(a arte)`, que
> não existe em slide nenhum, e ele não está no `ALIAS` (`:60-66`). **Fica em aberto** — custa pouco
> porque o bloco **não tem fala própria**, mas não está fechado.

### Classe 6 · Eu escrevi um aviso que desligou a verificação certa

**E esta é a mais cara do dia, porque o defeito não estava em nenhum dos dois arquivos: estava na
diferença entre eles.**

No `considerations.md` §4F eu escrevi *"o `% FALA` do `.tex` NÃO é espelho da fala — para medir
sincronia, meça texto de TELA"*. **Sobre contagem de palavras eu estava certo.** Sobre conteúdo, não —
e o aviso autorizava a inferência *"não compare os dois"*, que é exatamente a comparação que faltava.

**Resultado, medido em 26/08: 17 das 50 falas divergiam.** Não por erro de um dos dois lados — por
**edição de um lado só**. E o `SLIDES.md` é a **fonte do `SPEECH`**: nas 17, é a versão dele que sai
pela boca do autor.

**Cinco decisões registradas estavam vivas no roteiro depois de terem morrido no deck** — a "sexta
limitação" que a `AUT-11` matou, o carimbo que a `AUT-5` matou, as janelas que a `C30`/`G5` matou,
o *"e concorda"* do Wilcoxon que não tem proveniência.

🔴 **E na direção contrária, que é pior: duas divulgações de honestidade existiam SÓ no deck** — a
ressalva de vazamento da tarefa estática e a de que o controle de dimensão equalizada **nunca foi
executado**. **Uma regeração do `SPEECH` apagava as duas da fala.**

> **A regra:** antes de regerar o `SPEECH`, rode **`presentation/diff_fala.py`**. Custa segundos.
> ⚠ **Ele pareia por POSIÇÃO, não por título** — dois frames com o mesmo `\frametitle` são dois
> frames. É a terceira vez que esse erro aparece com roupa diferente: **título não é identidade.**
>
> **E a lição maior, que é sobre mim e não sobre a ferramenta:** eu escrevi um aviso verdadeiro
> (*"não conte palavras por ali"*) e ele foi lido como um aviso mais largo (*"não olhe por ali"*).
> **Um aviso que desliga uma verificação precisa dizer o que ele NÃO desliga.**

---

### Classe 11 · Eu pus uma instrução de LAYOUT dentro de uma spec de CONTEÚDO, e o orçamento deixou de prever

**A Classe 3 estabeleceu o contrato: *eu digo o CONTEÚDO, a `ppt` diz QUANTAS LINHAS ele ocupa*, e
o orçamento vai em PALAVRAS. Cumpri o contrato na letra e quebrei-o no espírito.**

No `ITEM B` da rodada de 28/08 especifiquei o par `hard`/`soft` do slide 8 com o cálculo *"a metade
`hard` encolhe de 16 para 10 palavras, o custo líquido é **+8 palavras**"*. **O frame passou de
+0,44 para +27,64 pt.**

**A causa está na minha própria spec:** eu escrevi um `\\` entre as duas metades. Esse `\\` é uma
**instrução de composição**, não conteúdo — força a metade `hard` a passar de 2 linhas para 1 e dá
2 linhas à `soft`. **Duas linhas novas, não oito palavras.**

> **A regra:** o orçamento em palavras só prevê altura enquanto o texto **flui**. No instante em que
> a minha spec contém `\\`, `\vspace`, `minipage`, `columns` ou `tabular`, **eu deixei de
> especificar conteúdo e passei a especificar layout — e o número de palavras deixa de significar
> qualquer coisa.** Nesses casos não estime: mande o conteúdo **sem** a instrução de quebra e deixe
> a decisão de composição com quem mede.

**E a `ppt` devolveu o lado simétrico, que fecha a regra:** ela atravessa a mesma fronteira ao
contrário. Encurtou `Texas category` para `Texas` por parecer redundância de forma — *"isso era
conteúdo com roupa de forma, e passou porque cortes de uma palavra são os mais fáceis de
justificar"*. 🔴 **E eu tentei absolvê-la desse corte, medindo o slide errado.** Procurei `Texas` perto de
valores de `p` no PDF, caí no frame `Capacity control` (faixa `B-P1`, eixo REGIÃO, números
63.446/64.931/64.503) e respondi-lhe *"o teu corte estava certo, não te penalizes"*. **A linha que
ela editou é outra**: está no `V19-2` (`Next category, sealed test`), diz `Texas category is the one
disagreement: −0,13 excluding zero, −0,007 including it`, e os números que eu citei **não existem
nela** — aparecem uma vez cada, no outro frame. **Ela repôs o `category` e tinha razão**, com o meu
próprio argumento: no deck, `Texas` é célebre como ganho de **região** (+1,21), então numa tela de
categoria a palavra nua convida a importar o resultado errado.

> ⚠ **É a quarta aparição de *"título não é identidade"*, e desta vez a minha.** As três anteriores
> estão no §3 e nas regras de medição (o `diff_fala` que pareia por posição; o detector de
> `Overfull` ancorado no `\begin{frame}`; o `grep` que encontrou o **botão do índice** `B-P1` antes
> do frame). **Um padrão que ocorre em vários sítios devolve o PRIMEIRO, não o CERTO** — e o índice
> vem sempre antes. Eu cometi-o **na mensagem seguinte** àquela em que registei a lição dela.
> A âncora correta é o `\framesubtitle` do próprio frame, nunca o rótulo do botão.

> **A formulação dela, que é a melhor que temos:** *"quando a minha alavanca muda **quais palavras
> estão na tela**, ela é tua; quando a tua spec contém uma instrução de composição, o custo é meu
> de medir. **Nenhum dos dois consegue prever o lado do outro.**"*
>
> Os quatro casos de um só dia: `+8 palavras` → **27,6 pt**; a ordem *"corte as caixas 1 e 2"* →
> **zero**; `Texas category` → `Texas`; e o `\vfill` sem par → **o dobro** do estouro.

> 🔴 **A REGRA, medida quatro vezes em 27/08 e sempre a dar zero ou o dobro:** neste deck a unidade
> de custo é a **LINHA**, e linhas são **quantizadas**. **Caracteres não compram nada até removerem
> uma quebra** — a `ppt` mediu que doze citações custam o mesmo em `(2018)` ou em
> `(Kendall et al., 2018)`, porque o que custa é a quebra, não o que a provocou. E em `columns` ou
> `minipage`, **a altura é o MÁXIMO entre as colunas: só a coluna mais alta paga.** Cortar da mais
> curta é trabalho perdido, por mais conteúdo que se sacrifique.
>
> | o que eu previ | o que mediu |
> |---|---|
> | `+8 palavras` | **+27,6 pt** — o `\\` que eu escrevi forçava a quebra |
> | *"corta as caixas 1 e 2 antes da 3"* | **zero** — minipage, altura é o máximo das três |
> | ordem de sacrifício do slide 21 | **zero** — eu ordenei na coluna curta; paga a comprida |
> | `V1, só o ano` no slide 8 | **zero** — idêntico ao autor-ano completo |
>
> **O protocolo que decorre:** eu ordeno por **valor de conteúdo**; a `ppt` diz **qual coluna paga**
> antes de eu ordenar. Nenhum dos dois consegue fazer o do outro.

⚠ **É a segunda vez no mesmo dia.** No `ITEM 6` eu mandei *"corte as caixas 1 e 2 antes da 3"* para
poupar altura numa faixa de minipages — e a `ppt` mediu que **comprou zero**: a altura de uma linha
de minipages é o **máximo** das três, e a caixa 3 tem duas linhas. **A regra que eu apliquei valia
para prosa corrida, e eu apliquei-a a caixas lado a lado.** Nos dois casos o erro é o mesmo:
raciocinar sobre texto quando o objeto na tela já não é texto.

---

### Classe 9 · Eu aceitei a caracterização de um registro em vez de abrir o registro

**A `ppt` descreveu a `V3` como *"os dois termos que a `V3` trocou por Next Category e Next Region"*.
Eu repeti isso em três mensagens, para ela, para a `tikz` e para o autor — e quase mandei editar a
Figura 2 do volume depositado por causa disso.**

**Fui ao registro e a `V3` diz o oposto:** que a família `stream`/`tower` **não estava no
`GLOSSARY`** — *"é buraco do registro, não defeito do deck"* — e a **`AUT-14` do autor AUTORIZOU os
termos**. **Nunca houve troca. Nunca houve defeito.**

> **A caracterização de um registro não é o registro.** Eu passei o dia cobrando isso dos outros —
> *"não conclua de um grep"*, *"vá à fonte"* — e aceitei de segunda mão a descrição de uma decisão
> que estava a três linhas de distância.

⚠ **O que salvou não fui eu:** a `tikz` parou por **outra** razão (a legenda da Figura 2 diz
`(semantic)`/`(spatial)`), e foi essa parada que me obrigou a verificar.

---

### Classe 10 · Eu busquei num escopo que não podia conter a resposta, e li o vazio como ausência

**A `extra` citou `EVALUATION_PROTOCOL.md §10` e §1 para corrigir a linha `Partition` do `V19-1`.
Eu procurei o arquivo, não achei, procurei as strings citadas, não achei — e escrevi ao autor que
"a fonte que ele citou não existe neste repositório".**

**Ela existe.** Vive em `/Users/vitor/Desktop/mestrado/mtlcheck/`, que é **outro repositório**. As
três citações conferem verbatim (`:323-324`, `:331`, `:54`). E o **rodapé do próprio slide já
dizia onde** — `mtlcheck a6639c0a` — enquanto eu procurava em `ingred`.

> **Um `grep` vazio prova que a string não está no escopo, e nada mais.** A Classe 9 diz para abrir
> o registro em vez de aceitar a caracterização; esta diz o inverso e é mais fácil de cometer:
> **antes de declarar que um registro não existe, verificar se você o procurou onde ele viveria.**
> O sinal estava na tela que eu estava a corrigir.

⚠ **O agravante:** eu quase transformei um erro meu de escopo numa acusação de fabricação contra um
par que estava certo. **Reportei a dúvida como dúvida** — e foi só isso que a manteve reparável.
Se eu tivesse reportado como achado, teria queimado o trabalho dela e a correção não teria entrado.

---

### Classe 8 · Eu apliquei o contrato certo a uma cláusula que ele não cobre

**O contrato do dia era:** *nada sai da tela sem estar na fala.* Verifiquei, estava, cortei.
**A cláusula era `"absolute scores are optimistic"`, e o `SLIDES.md:830` registrava desde 24/08 que
ela devia FICAR na tela — com a razão escrita: *"mandá-la para a fala pareceria escondê-la"*.**

**Cobertura pela fala era exatamente o que aquela decisão tinha rejeitado.** Eu não li o bloco antes
de reescrever o slide; li só a fala.

> 🛑 **A exceção, e ela precisa viajar colada ao contrato** *(formulação da `ppt`)*: **para uma
> cláusula AUTOINCRIMINATÓRIA, mandar para a fala não a preserva — muda o que ela significa.**
> Uma ressalva contra si mesmo **dita e não projetada** lê-se como concessão arrancada; **projetada**,
> lê-se como honestidade oferecida. **A `R6` diz o mesmo por outro caminho: a ressalva acompanha o
> resultado.**

**A linha que separa as duas, e ela é o que decide cada caso:** vale para a cláusula que **qualifica
um número que está na tela** — essa desce com o número. Não vale para a que qualifica uma **escolha
de método** sem número associado: essa pode viver na fala e na reserva. *(Foi por isso que o
`"a convenção alternativa é mais favorável"` pôde sair da tela do 36 e o `optimistic` não podia.)*

✅ **Varri a classe inteira no deck depois disso.** A trilha principal mantém **quatro** cláusulas
contra si mesma na tela: `absolute scores are optimistic` · `Not width-matched` · `the evidence does
not separate the contributions of the shared trunk and the private spatial path` · `this chapter does
not isolate each encoder`. **Só uma tinha saído, e voltou.**

### E o protocolo que decorre, que é de mão dupla

- **Meu:** ao especificar reescrita, **leio o bloco do `SLIDES.md`, não só a fala** — é lá que moram
  as decisões sobre *onde* uma coisa tem de viver, e elas não aparecem em varredura de texto;
- **Da `ppt`:** ao devolver *"não cabe, corte N palavras"*, ela não sabe quais linhas estão
  protegidas. **Mando a marca `[INTOCÁVEL]` junto com o texto**, e ela mede contra o resto.

---

### Classe 12 · Eu especifiquei o conserto de uma ALEGAÇÃO para uma superfície só

**Encontrei que o slide 6 afirmava falsamente que todos os modelos nomeados predizem *"the exact
establishment"* — o `HST-LSTM` prediz AOI. Especifiquei tirá-lo da lista. Da TELA.**

**A alegação vivia em três superfícies**, e a `ppt` encontrou as outras duas: a **fala** (*"…ST-RNN,
DeepMove, HST-LSTM, Flashback… **Todos eles predizem o lugar exato**"* — **mais categórica que a
tela, sem hedge nenhum**) e o campo `Na tela:` do `SLIDES.md`. Tirar só da tela **deixaria o autor a
dizer em voz alta exatamente o que a tela acabara de parar de afirmar.**

> **A regra:** remover uma PALAVRA é edição de uma superfície. Remover uma **ALEGAÇÃO** é edição de
> todas as superfícies onde ela é feita. **Quando a spec corrige uma alegação, ela tem de enumerar
> onde a alegação vive** — tela, `% FALA:`, `SLIDES.md`, Série B — e o executor confere as três.

⚠ **É a `Classe 6` outra vez, do outro lado.** Lá eu escrevi um aviso que desligou a comparação
tela↔fala; aqui eu simplesmente não a fiz. **Eu cobro esta varredura da `ppt` desde ontem e não a
apliquei a mim.**

---

### Classe 7 · Eu deixei entrar conteúdo do autor sem validá-lo contra o que a tese entrega

**Apontado por ele, em 27/08, e a formulação é dele:** *"você que é o agente que deveria averiguar e
validar o conteúdo, não averiguou e nem validou o que estava entrando."*

O bloco de trabalho futuro do slide 47 propunha **"a shared trunk with MMoE, or with
cross-attention"**. O `5_mobiwac/04_method.tex:28` descreve o modelo entregue como *"the shared
trunk, **a cross-attention stack** of two blocks"*. **O slide final propunha, como trabalho futuro, a
arquitetura do Capítulo 5.** E "drop the embedding layer" também já estava feito.

**A causa não é desatenção, é PERDA DE ESCOPO na migração.** O item veio da lista crua dele
(`wrapup/Questions_author.md:118`), escrita sobre o **MTLnet do Capítulo 3** — ali "camada embedding"
é o FiLM. Ao entrar na **Conclusão**, virou trabalho futuro da dissertação inteira, e nesse escopo é
falso.

> **A regra:** quando um item do autor migra de um contexto para outro, **o escopo não migra junto —
> ele tem de ser reconstruído.** E a validação certa não é "isto faz sentido?", é **"isto já está
> entregue em algum capítulo?"**
>
> ⚠ **E o que torna isto uma classe e não um deslize:** eu auditei a lista de CONTRIBUIÇÕES dele na
> §13, com cuidado, e **não auditei a de TRABALHOS FUTUROS — que está no mesmo arquivo, quarenta
> linhas abaixo.** Auditar metade de uma fonte e tratar a fonte como auditada.

---

### E mais seis, avulsos, que valem pelo padrão

| erro | lição |
|---|---|
| Incluí o **slide 41 (veredito)** na regra de negrito/sublinhado | **é tabela de VEREDITO, não de placar.** Marcar "melhor" nas células dentro da margem **reintroduz o veredito de vencedor que a lei proíbe**. ⚠ **E eu escrevi depois que "ele não tem negrito" — tem:** `main.tex:1481-1490` marca **as três células que superam com Holm**, e só essas. **Sublinhado: zero. A lição está respeitada; a minha frase estava errada** |
| O meu **mapa de renumeração** estava deslocado, e **o deslocamento não era uniforme** — quatro faixas | **ancore por título.** O número é derivado e muda a cada remoção |
| Descrevi a **`fig1_dataflow` original** como se fosse a chapa nova, criando um dilema inexistente | **verifique o artefato de que está falando**, não a memória dele |
| Marquei a remoção da linha de `Controls` como **"risco de esconder"** | os números eram **pré-vazamento** e a frase presa a eles é **aritmeticamente falsa** contra a Tabela 9. **Remover era corrigir** |
| Repeti o **`+4:00`** da `ppt` sem somar a tabela dela (era `+2:36`) | **confira o número do par também.** Ela confere os meus o dia inteiro |
| Não vi que **o desenho do HGI do autor estava no `considerations.md`**, destruído por um formatador | procurei "HGI", vi texto colapsado e **descartei sem ler**. Ele teve de me dizer duas vezes |

---

### O padrão por trás das seis classes, formulado pela `ppt` melhor do que eu formulei

> *"Você escreveu num campo sem perguntar quem mais o lê; eu li um campo sem perguntar quem mais
> escreve nele."*

**Três defeitos do dia moram na FRONTEIRA entre dois artefatos, e nos três cada lado estava certo
sozinho:**

| o defeito | lado A | lado B | onde estava o erro |
|---|---|---|---|
| a ressalva de vazamento | correta no deck | ausente no `SLIDES.md` | **existia só em um** |
| a minha anotação de versão | legítima no `SLIDES.md` | inexistente no `.tex` | **no que o extrator fazia com ela** |
| `S5b` / `S26b` | códigos válidos | regex do extrator | **ninguém perguntou ao consumidor** |

> **A varredura que decorre, e ela é a que faltava o dia inteiro:** para cada artefato que você
> escreve, liste **quem mais o lê** e **quem mais escreve nele**. O defeito raramente está no
> arquivo; está no que o vizinho supõe sobre ele.

### E duas regras sobre COMO medir, que custaram um defeito cada

**1 · Um teste que injeta um defeito tem de provar que injetou.** A `ppt` testou um guarda copiando
o `SLIDES.md` para um sandbox, injetando a marca e rodando o extrator. **Passou.** Não disparou
porque o `re.sub` **não casou** — o arquivo saiu byte-idêntico ao original. Só o md5 denunciou.
**Ela estava medindo um guarda contra um alvo intacto e lendo o verde como resultado.**

**2 · Uma invariante medida em zero precisa de um segundo número que prove que o caso ocorre.** Eu
propus um guarda apoiado em *"nenhuma das 56 falas extraídas tem crase — 0/56"*. **Ela mediu o
campo BRUTO também: 3/56.** É esse 3 que transforma o zero em evidência — sem ele, o `0` não
distingue *"a limpeza funciona"* de *"nunca houve o que limpar"*, e o guarda podia estar guardando
uma porta que ninguém usa.

**3 · Uma âncora errada não erra às vezes — erra sempre, e sempre para verde.** A `ppt` verificava
`Overfull` com `grep "detected at line $n"`, com `$n` = a linha do **`\begin{frame}`**. **O LaTeX
reporta o estouro na linha do `\end{frame}`.** O grep nunca achava nada, para frame nenhum, nunca —
e cada "✔ sem `Overfull`" do dia consultava uma linha onde nada seria reportado. **Um instrumento que
erra às vezes te avisa; um que responde "limpo" a tudo parece funcionar para sempre.** A correção é a
mesma de sempre: **atribuir por INTERVALO `[begin, end]`, não por linha de abertura.**

**3b · `grep -o` destrói adjacência, e a saída parece uma frase.** A `extra` mandou-me, como
verbatim do Apêndice E, *"The region head deliberately keeps two routes. The **private route** reads
the raw 9×64 region history."* **A linha diz `private tower`.** Ela tinha rodado
`grep -o 'two routes[^.]*\.\|private route[^.]*\.'`, e o `-o` devolveu **duas linhas de saída
separadas** — de sítios diferentes do ficheiro, uma delas de uma legenda de figura 44 linhas antes.
**Ela costurou-as numa frase e atribuiu a uma linha.**

> **`-o` remove o contexto E o número de linha. Adjacência na SAÍDA não é adjacência no TEXTO.**
> Para extrair citação, use `grep -n` com a linha inteira, ou `sed -n 'N,Mp'`, e leia o que está à
> volta. **Nunca componha uma citação a partir de dois fragmentos de saída.**

⚠ **Esta é de espécie pior do que as outras quatro do dia.** Truncagem, substring, escopo de árvore
e âncora ambígua fazem **falhar em ver** ou **ver de mais**. Esta **produz evidência que não existe**
— num projeto cuja lei é *"copiado da célula, nunca re-derivado"*. E chegava à conclusão certa, o que
é exatamente o que faz ninguém voltar a verificar. **A regra que fica, e vale para os dois lados:
uma citação errada que chega à conclusão certa é o caso mais perigoso.**

**4 · Não leia um log que ainda está sendo escrito.** Duas leituras de `build/main.log` durante o
`make` devolveram **`0 Overfull`** — num log pela metade não há o que achar, então o zero não diz
nada. **Sintoma diagnosticável: `main.log` com mtime MAIS NOVO que `main.pdf` significa build em
curso.** Espere `pgrep xetex` esvaziar e o mtime parar.

**5 · Um controle que você escolheu depois da conclusão não é controle.** A `ppt` mediu que os slides
que o autor chamou de "textuais demais" tinham 6–23 palavras por unidade contra **3–4** dos que ele
não citou — linha limpa, medições individuais todas corretas. **Fui medir os 49 e a linha não
existe:** três dos quatro citados estão **na mediana ou abaixo** (o `43` é dos mais magros do deck), e
os seis mais pesados **não foram citados**. O controle dela eram quatro slides que ela escolheu — os
quatro de **tabela de números**. Contra 3 palavras por célula, qualquer slide de prosa parece fora da
curva. Nas palavras dela: *"eu não escolhi o controle, eu escolhi a conclusão e fui buscar o controle
que a produzia."*

> ⚠ **A defesa desta é diferente das quatro acima, e é por isso que ela é a mais perigosa.** Contra
> falso verde a pergunta é *o vermelho era possível?*. **Contra amostra enviesada não há pergunta que
> salve** — só **medir a população inteira antes de traçar a linha**. Ela traçou com nove de
> quarenta e nove.

> **As cinco dizem a mesma coisa: um verde só vale se você provar que o vermelho tinha por onde
> aparecer — e que você não escolheu onde olhar depois de saber o que queria ver.**

**6 · Um lote parcialmente liberado cria estados que ninguém desenhou.** O `B6-6` ficou, no PDF de
27/08, com a **faixa nova** (lote 2, liberado) e a **linha velha do corpo** (lote 3+, bloqueado) na
**mesma página** — e a faixa existia justamente para corrigir a linha. **O slide passou a afirmar e
refutar na mesma respiração, o que é pior que o defeito original**, que ao menos era coerente
consigo mesmo.

> **Ao cortar trabalho em lotes, pergunte o que cada fronteira deixa VISÍVEL ao mesmo tempo.**
> O teste: *a afirmação da faixa contradiz alguma linha do corpo?*

### E um padrão de caixa que vale antes de qualquer corte de texto

🔧 **Quando faltam poucos pontos, procure o AMBIENTE antes de procurar a palavra.** Três casos medidos
em 26/08: a moldura do `block` (**~28 pt**), o `\begin{center}` (**8 pt**, e ele cobra `\topsep` dos
dois lados), o `itemize` de **um** item (**~6 pt** de `topsep`+`partopsep`, sem ganhar nada). **Somados,
42 pt — mais que o pior estouro do deck inteiro.**

> 🛑 **E aqui eu escrevi uma consequência ERRADA, corrigida pela `ppt` na mesma hora. Deixo o erro à
> vista porque ele é fácil de repetir.**
>
> Eu escrevi: *"antes de eu especificar corte de conteúdo num slide apertado, varra o ambiente —
> talvez não precise cortar nada."* **Isso confunde três orçamentos que não se convertem:**
>
> | orçamento | unidade | o cromo devolve |
> |---|---|---|
> | **o relógio** (49:17 contra 50 min) | **palavras de FALA** | **nada.** O autor lê o mesmo texto |
> | **a tela cheia demais** (o que o autor pediu) | **palavras na TELA** | **nada.** 8 pt de padding a menos não fazem cinco linhas de prosa virarem âncora |
> | **o `Overfull`** | **pontos de altura** | **isto sim** |
>
> **O cromo compra pontos de página, e só.** As 707 palavras das cinco reescritas continuam
> necessárias inteiras para levar 54:20 a 49:17, e um slide que o autor chama de "excessivamente
> textual" continua textual depois de trocar `center` por `makebox`.
>
> **O padrão do ambiente vale — mas só contra `Overfull`.** Não o ofereça como alternativa a cortar
> texto.

---

## 4 · O que eu estava fazendo quando parei

**A `ppt` executa; eu escrevo e verifico.** Enquanto ela trabalha no `.tex`, o meu trabalho é:

1. **🔴 As cinco falas curtas** — escritas, entregues, **seguradas até o autor decidir o corte**. Ver §2;
2. **As falas de categoria B** — nove blocos em que a fala descreve outra estrutura (itens que viraram
   tabela, blocos que sumiram) **mas nada de errado seria DITO**. Custa fluência, não credibilidade.
   **É a única fila minha que sobrou, e é opcional**;
3. ✅ **Re-auditoria feita às 20:30, com o deck estável** (105 pp · as duas chapas sem `Overfull`).
   A de 18:07 tinha rodado sobre um arquivo que mudou seis vezes durante a corrida. **A nova está
   limpa**, e o que ela conferiu está logo abaixo;
4. ✅ **A sincronia `.tex` ↔ `SLIDES.md`** — **16 das 17 fechadas em 26/08.** Sobra o **`S46`**, que eu
   segurei de propósito: é uma das cinco falas reescritas, e sincronizar antes da decisão do autor
   seria escrever por cima da reescrita. **Ver a Classe 6.**

### A re-auditoria de conteúdo, 26/08 20:30 — o que foi conferido e o que passou

Sobre o texto de tela extraído do `main.pdf` estável (13.419 palavras).

| o que | resultado |
|---|---|
| **`outperforms`** | **3 ocorrências, as três legítimas**: a meta-frase que declara a própria lei, o cabeçalho da tabela de veredito marcando **só as três células que sobrevivem a Holm**, e uma da reserva |
| **`match` / `ties`** (21 no total) | **nenhuma é verbo de veredito.** São `matched folds`, `capacity-matched`, `width-matched`, a frase anti-match do capítulo (*"A claimed gain and a claimed match require different tests"*) e **duas que ENFORÇAM a lei** (*"deficits, not ties"*, *"None of them is a tie"*) |
| **`venue`** | 2, as duas `venue-type feature`, **verbatim de `4_courb.tex:42`** — ver §7B item 2 |
| **`novelty`** | 1, **verbatim de `5_mobiwac/02_related.tex:71`**, e a `WRITING_LAW.md:277` diz explicitamente *"do not over-ban: robust, novel, framework…"* |
| **`next place`** | 7, e **todas necessárias**: o deck tem de nomear o que ele NÃO prediz |
| `AUT-5` carimbos | **0 na tela** ✓ |
| `AUT-11` task-pair confound | **0** ✓ |
| `AUT-12` `Contrastive infomax` | **1** ✓ |
| `AUT-13` linha `Controls` | **0** ✓ |
| `AUT-2` `Contributions` como título | **1 cópia** na trilha principal ✓ |
| `AUT-6` `POI-RGNN` · `ReHDM` | presentes ✓ |
| `AUT-17` `Category Classification` | presente ✓ |
| ✅ `AUT-18` **slide 30** | **FEITO às 20:40** — só a imagem, `Overfull` zero. ⚠ **e eu errei ao dizer que dependia do gênero da figura**: a `fig1_dataflow` estava sendo exibida **pela metade** (`width=0.50\textwidth`), e tirados os marcadores ela entra no tamanho natural e fica legível. **O gênero segue aberto e não bloqueia nada** |

**Nenhum achado novo, e a `AUT-18` fechou logo depois. Nenhuma decisão registrada continua pendente na tela.**

---

**Nada mais está esperando em mim.** A especificação inteira foi despachada em seis lotes e executada.

---

## 5 · Os padrões que valem mais que os itens

**Anotados porque cada um custou pelo menos um defeito hoje.**

- **O contêiner custa mais que o conteúdo.** A moldura de um `exampleblock` custa **~28 pt** sem
  carregar texto nenhum; um `\begin{center}` é uma `trivlist` e cobra `\topsep` **dos dois lados**,
  **8 pt de graça**. **Quando um slide estoura por poucos pontos, examine a moldura antes do conteúdo.**
- **Um guarda que parece folga vai ser removido por quem estiver otimizando folga — inclusive por quem
  o pôs.** O guarda tem de **se declarar**. *(`\begin{frame}{título}{grupo}` é sintaxe válida: abrir o
  corpo com uma chave joga o conteúdo para dentro da faixa do título, com log limpo. Aconteceu três
  vezes hoje.)*
- **Validar o preview não é validar a entrega.** A `tikz` revisou onze rodadas no render do beamer
  enquanto o PDF que ia entregar saía em fonte serifada. **É a mesma classe do `src_fix/`**, que rodou
  nove dias de portões verdes sobre a árvore errada.
- **Um número certo com uma tabela errada é o pior artefato possível** — o total confere e ninguém
  desconfia da linha. Aconteceu na medição do relógio: **7.607 palavras estáveis nas três versões, e a
  atribuição por bloco só ficou certa na terceira.**
- **Meça, não estime.** Três sessões cometeram o mesmo erro hoje, de ângulos diferentes: largura de
  célula, largura de palavra, e a minha fila. **Uma estimativa bem fundamentada é mais difícil de
  contestar que um erro de execução.**
- **O que a tela precisa carregar sozinho tem prioridade de corpo sobre o que a fala carrega junto.**
  ⚠ **Ela decide corpo entre dois elementos obrigatórios; NÃO decide o que sai da tela.** Para isso
  vale o protocolo do `HANDOFF` §4c: **localizar a cláusula no destino ANTES de removê-la.**
- **Quando um termo está numa lista fail-closed e a defesa dele depende de uma distinção sutil, o termo
  sai** — explicá-la ao vivo custa mais do que ela vale.

---

## 6 · O que está aberto, e de quem é

| item | dono | nota |
|---|---|---|
| **O corte da fala** | **autor** | as cinco falas estão prontas com a `ppt`, medidas e **não aplicadas**. Escolha: **49:17** como está · **48:00** (−180 palavras) · **47:00** (−320) |
| ~~O slide 30 · Check2HGI~~ ✅ **FECHADO 26/08** — só a imagem, no tamanho natural | — | escolher o **gênero** da figura: *fluxo de dados* (a chapa existe, mostra os quatro andares, o check-in marcado e as duas saídas) ou *fluxo de treino* (irmã da chapa do HGI). **Nos dois casos o slide fica sem marcador.** A `ppt` deixou `graphicspath` e `.gitignore` prontos: **é edição de dois minutos** |
| **As duas divergências dos diagramas** | **autor** | a `tikz` desenhou **contra o pedido literal dele**, com a evidência no cabeçalho de cada `.tex` (`figures/src/dgi_flow.tex:18-24`). ⚠ **as duas estão CERTAS** — verifiquei no código e na descrição escrita por ele. 🛑 **NÃO as chame de `Q7`/`Q8`**: esses são rótulos da `tikz`, e colidem com as perguntas `Q7`/`Q8` do `considerations.md` §8, que são **outras** e já estão fechadas pela `AUT-3` |
| ~~A chapa do HGI~~ | — | ✅ **INTEGRADA** (`main.tex:759`), renderiza no impresso **22**. ⚠ eu a dei como pendente e não estava. Sobra só corrigir `figures/src/hgi_flow.tex:2`, que diz "impresso 23" |
| **A série B (49 páginas)** | *(outro agente)* | ver abaixo |
| **Regerar o `SPEECH`** | `ppt` | ⚠ **só depois** que a fila de falas fechar. Ordem: `SLIDES.md` → reconstruir o deck → regerar. ⚠⚠ **e o `SLIDES.md` carregava dois defeitos JÁ corrigidos no deck** — a linha *"Florida appears twice"* e o termo `fine class`. **Corrigidos em 26/08.** Se aparecer outro, **corrija no `SLIDES.md` ANTES de regerar**, senão o roteiro ressuscita o que o deck já enterrou |

### Para quem assumir a série B

> 🛑 **PROCURE PELO TEXTO, NÃO PELA LINHA.** Eu tinha escrito 2362 / 2067 / 2310 aqui. Quando fui
> reconferir, **na mesma noite**, os três estavam em **2289 / 1994 / 2244** — o deck andou 60 linhas
> debaixo da citação. **Toda linha da série B neste handoff é chute.** Use `grep -n` pela frase.

**Dois defeitos registrados e não executados**, porque o autor a reservou:
- **`inferential unit`** — `grep -n 'inferential unit' slides/main.tex`. **Termo que não existe no
  documento entregue.** §8.11 é fail-closed e **não distingue trilha principal de reserva**;
- **`Reported alongside, and it agrees`** — `grep -n 'and it agrees' slides/main.tex`, o slide
  `B1-4`. **Afirmação sem proveniência.** O Cap. 5 registra o Wilcoxon como sensibilidade no setup e
  **nunca reporta o resultado**. Grep em `06_results.tex` e `07_discussion.tex`: **zero ocorrências.**

🛑 **E uma coisa que NÃO pode ser "atualizada por consistência":** o slide que diz
*"**Submitted paper**, p. 9, limit 4 of 5"*. **Ali `submitted paper` não é status — é o NOME DE UM
ARTEFATO**, a versão submetida do manuscrito, cuja lista de limites **difere da da dissertação**. É
essa distinção que o slide existe para fazer. **O artigo foi aceito; esta linha continua `submitted`.**

---

## 7 · Cinco fatos que você vai precisar e que não estão em lugar óbvio

1. **O artigo do Cap. 5 foi ACEITO no MobiWac**, em 26/08. Está em três lugares do deck (slide 4, a
   nota do 27 e a fala do divisor). ⚠ **O volume entregue diz *submitted / under review* em CINCO
   lugares** — `tables/frame/lineage.tex:8`, `1_introduction.tex:168`, **`:342`**, `:376`,
   `5_mobiwac.tex:29`. ⚠ **eu tinha escrito quatro e o `:342` ficaria para trás numa errata** — por isso a nota carrega *"after the dissertation was deposited"*. **Sem essa ressalva,
   um arguidor lê `accepted` na tela e `under review` no volume e pergunta qual vale.**
   ⚠ **Provável errata para o depósito final** — não é escopo do `gate`, mas ninguém vai lembrar
   depois de sexta.
2. **A dissertação usa intervalo de confiança de 90%**, não 95% (`05_setup.tex:115`,
   `06_results.tex:208`). O slide do veredito rotulava **95%** — **em duas metades da mesma linha**,
   região e categoria. Corrigido. **Se voltar, é erro factual no slide que decide a defesa.**
3. **O piso de Markov de região está ACIMA dos três sistemas externos publicados** na maioria dos
   conjuntos — **acima do HMT-GRN nos seis**. **É o achado mais forte do Cap. 5 e ninguém o tinha
   dito assim.** Substituiu *"superamos a literatura"*, que superafirmava duas vezes.
4. **O termo `venue` é banido** como sinônimo de lugar (`GLOSSARY.md:84`). **Exceção única:
   `venue-type feature`**, que é verbatim do Cap. 4 e nomeia uma **coluna de dados**.
5. **`category classification` é o termo da Def. 2.6** — **não** *"categorical classification"*. A
   §8.11 é fail-closed.

---

## 7B · ⚠ O que uma verificação independente NÃO conseguiu confirmar

**Este handoff foi verificado por três agentes contra os artefatos, e eles acharam 17 afirmações
erradas minhas — todas corrigidas acima.** Mas cinco coisas ficaram **não verificáveis**, e quem
assumir precisa saber quais:

1. 🔴 **O total de palavras da fala.** Eu contei **7.898**; a `ppt` contou **7.607**. **Fui atrás da
   diferença: 347 tokens são NOTAS DE PROJETO dentro dos blocos de comentário** — avisos, datas, `⚠`
   — que a minha contagem ingênua somou como fala. **A da `ppt` está certa.** ⚠ **Mas o número não
   reproduz de nenhuma fonte sozinha**: o `SLIDES.md` bruto dá 8.691 e o `main.tex` dá 12.968 (que
   inclui a série B). **Todo o cálculo do relógio repousa nesse total. Se for remedir, declare o
   método.**
2. ✅ **FECHADO em 26/08 — a exceção `venue-type feature` tem fonte, e ela é forte.** Eu tinha
   registrado isto como "afirmação minha, não regra escrita". **Fui ao volume: é verbatim.**
   `src/chapters/4_courb.tex:42` traz a frase inteira que o deck reproduz — *"the venue-type feature
   maps one-to-one onto the seven top-level categories"* — dentro da própria ressalva de vazamento do
   capítulo. **A regra fail-closed governa termo que NÓS introduzimos, não texto reproduzido.**
   As duas ocorrências do deck (impresso 29 e a reserva) são a mesma frase do capítulo.
3. **A quebra de linha do `fine class`** não é mais verificável: o defeito foi corrigido, e o grep dá
   zero em toda a história commitada. **A evidência é de segunda mão.**
4. **As medições de render** (`exampleblock` ~28 pt, `center` 8 pt dos dois lados) vêm dos relatórios
   da `ppt`, não de medição minha.
5. **Os auto-relatos de sessão** — quantas rodadas, quem parou quando. **Trate como narrativa.**

---

## 8 · A armadilha final, e é a mais barata de evitar

**Quando o corpo de um slide muda, TRÊS textos envelhecem em silêncio junto com ele: o título, a
fala e a proveniência.** Nenhum dos três dispara `Overfull`, nenhum aparece no render, e o único que
alguém relê é o corpo.

O slide 3 passou o dia inteiro chamado **`The question, and the answer in one line`** — **depois que a
resposta foi removida dele.** Nem eu, nem a `ppt`, nem uma auditoria de quatro agentes pegou. **Todos
verificaram o corpo.**

**E aconteceu de novo no MESMO slide, horas depois.** A pergunta de pesquisa perdeu o parêntese
(`AUT-19`) e entrou na tela corretamente. **O bloco `% FALA:` logo acima continuou recitando a versão
velha** — a tela dizia *next-category and next-region prediction* e a boca ia dizer *point-of-interest
prediction*, o termo exato que a edição existiu para tirar. **Peguei por acaso, olhando o frame por
outro motivo.**

> **Acrescente à sua varredura, para cada slide que você editar:**
> 1. **o título** ainda descreve o que está na tela?
> 2. **o `% FALA:`** ainda diz o que a tela diz? *(a fala é o que sai da boca na frente da banca)*
> 3. **a proveniência** ainda aponta para a linha de onde o texto novo veio?
>
> E o mesmo trio no `SLIDES.md`, que é a fonte do `SPEECH` — corrigir só o `.tex` deixa o roteiro
> ressuscitar a versão velha na próxima regeração.

⚠ **E uma correção ao que eu tinha escrito aqui.** Eu afirmei que os `\specialframe` **não têm
`\frametitle`**. **Falso em três dos cinco:** `{The question}`, `{Closing}` e `{Obrigado}` têm título
e **são alcançados pelo extrator**. **Só dois não têm** — as duas transições internas, nas linhas 683
e 942.

**A lição sobrevive, e é sobre esses dois:** um frame sem `\frametitle` **não é alcançável pelo
extrator**, e **o texto dele só se confere lendo o corpo**. Foi assim que um cartão do roteiro ficou
com o título velho por horas.
