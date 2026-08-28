# HANDOFF_SLIDES.md — para o próximo agente que mexer no deck

> **Escrito 2026-08-24.** Defesa **sexta, 28/08/2026, 10:00, remota (Google Meet)**. Banca: Fabrício
> A. Silva (orientador/presidente), Clayson S. F. de Sousa Celes (ITA, externo), Alex Borges.
>
> **Este documento é sobre COMO trabalhar no deck.** O estado e as armadilhas de build estão no
> [`HANDOFF.md`](HANDOFF.md) ao lado, que continua valendo inteiro — **leia os dois**. Onde eles
> divergirem, o `HANDOFF.md` vence nos fatos e este vence no método.
>
> **Quem aprova é o autor.** Ele decide sobre conteúdo, ênfase e forma. Você mede, propõe com
> opções, e espera.

---

## 1 · As cinco coisas que você precisa saber antes de tocar em qualquer arquivo

### 1.1 · A brevidade é o alvo, e ele tem uma referência medida

O deck foi reescrito em 2026-08-24 contra um alvo **empírico**, não uma opinião: a defesa de
**Henrique de Souza Santana**, mesmo programa, mesmo template NESPeD, gravada em
`/Volumes/linux/VIDEO/Screen Recording 2026-07-08 at 10.02.56.mov` (61 min, 4K).
⚠ **Não é um PDF.** Um agente anterior procurou por PDF, não achou, e concluiu que a referência não
existia — estando o caminho num doc que ele mesmo escrevera.

**Medido em 10 slides amostrados da gravação: mediana 21 palavras de tela, média 28, máximo ~71.**

⚠ **Duas ressalvas de método, e as duas puxam para cortar MENOS:**
1. A contagem do Henrique é **OCR sobre a gravação** e enxerga texto dentro de figura; a nossa é
   **corpo LaTeX** e não enxerga. Não são a mesma régua.
2. **A regra dele não é "sem frases".** Ele escreve frase completa exatamente onde a frase **É** o
   conteúdo (a pergunta de pesquisa, o que o estudo faz), e nunca para explicar ou ressalvar. O
   slide 3 dele tem 71 palavras, quase todas numa pergunta de pesquisa emoldurada.

**A regra aprovada pelo autor, seis pontos:**

| # | |
|---|---|
| 1 | No máximo **uma frase completa por slide**, e só quando a frase é o conteúdo. Bloco próprio, corpo grande |
| 2 | Todo o resto vira **fragmento** sem verbo conjugado, terminado em `;` |
| 3 | **Ressalva, justificativa e explicação saem da tela** e vão para a fala, ditas no mesmo fôlego |
| 4 | **Slide de resultado é tabela ou figura + legenda.** Nunca prosa ao lado |
| 5 | Guarda-corpo: mediana **25–30**, teto **~70** — e o teto só vale quando é *um* bloco lido de uma vez |
| 6 | Sair do `\scriptsize` como padrão. Não por legibilidade, mas porque **corpo maior força brevidade** |

**Estado atual, medido:** 53 slides de deck, **mediana 89**, máximo 143, total 4.6k palavras
(era 119/233/6.5k na manhã de 24/08). ⚠ **A mediana de 89 é o piso desta abordagem.** O que resta
acima de 70 **não é verbosidade**: são tabelas, figuras com legenda, carimbos de convenção e blocos
de redação mandatada. Chegar a 25–30 exigiria **tirar conteúdo da tela**, que é decisão do autor
sobre o que a banca vê, não reescrita.

### 1.2 · Onde consumir cada informação

| o que você precisa | onde está | ⚠ |
|---|---|---|
| **de onde vem cada número** | [`../CLAUDE.md` §0](../CLAUDE.md) | **leia ANTES de escrever qualquer número.** É o erro mais repetido do projeto |
| a lei da estrutura do deck | [`PLANO_FLUXO_DEFESA.md`](PLANO_FLUXO_DEFESA.md), **§8 tem as 16 regras** | |
| o slide-a-slide | [`SLIDES.md`](SLIDES.md) | **é a FONTE da fala**; o `main.tex` é o espelho |
| estado, build, armadilhas | [`HANDOFF.md`](HANDOFF.md) | **o §3 tem 7 casos de "o instrumento passou porque mediu outra coisa"** |
| a lei da palavra | [`../WRITING_LAW.md`](../WRITING_LAW.md) | registro, lei dos verbos, construções banidas |
| o registro de termos | [`../GLOSSARY.md`](../GLOSSARY.md) | **fail-closed**: termo fora do registro não pode ser usado |
| protocolo de número e afirmação | [`../AGENT_GUARDRAILS.md`](../AGENT_GUARDRAILS.md) | |
| perguntas de banca com resposta pronta | [`../wrapup/open_points/ARGUICAO.md`](../wrapup/open_points/ARGUICAO.md) | |
| a comparação com a literatura | [`../wrapup/open_points/BASELINES_EXTERNOS.md`](../wrapup/open_points/BASELINES_EXTERNOS.md) | o que se pode afirmar, em qual eixo, e o que não |
| o roteiro de fala | [`SPEECH.md`](SPEECH.md) / `SPEECH.pdf` | **gerado**, não editado à mão — ver §4 |

### 1.3 · As regras de escrita, e a exceção registrada

O `PLANO §8 regra 15` estende as três leis aos slides, **mas com uma ressalva que importa**:

> *"**Um deck não é prosa**, mas as três leis governam **palavra e número** igual."*

Consequência prática, decidida pelo autor em 2026-08-24 e registrada em [`HANDOFF.md` §4f](HANDOFF.md):
**o travessão fica quando é separador de rótulo, título de bloco ou subtítulo de frame; sai quando
faz trabalho de prosa dentro de uma frase completa.** ⚠ **Não "conserte" isso** — a
`WRITING_LAW.md:131` diz *"No em-dash anywhere"* e uma varredura ingênua vai querer eliminar os 93
que ficam.

⚠ **Lição de método, e vale para toda varredura de estilo:** a primeira contagem dizia 28 para
remover; a verificação caso a caso deu **11**. Uma varredura por *presença* teria tocado 105 lugares
onde 11 precisavam mudar. **Classifique por função antes de substituir.**

O que **está limpo** e você não precisa reverificar (medido 24/08): as listas banidas
(`delve`, `showcase`, `leverage`, `moreover`, `comprehensive`/`robust` decorativos) dão **zero** na
tela do deck principal.

### 1.4 · Use um agente Fable como revisor para o que é importante

**Decisão do autor.** Para qualquer coisa que envolva **conteúdo, ênfase, estrutura ou uma
afirmação que a banca possa atacar**, rode uma revisão crítica por um agente Fable **antes de
aplicar**. Isso não é cerimônia: numa sessão ele achou três defeitos que ninguém tinha visto.

| o que o Fable achou | o que teria acontecido sem ele |
|---|---|
| o slide de contribuições era, **inteiro**, tela de outros três slides — e um deles era o **imediatamente anterior** no fechamento | a banca veria dois terços de um slide repetidos um clique depois |
| o risco de reivindicar "superamos a literatura" no eixo de região, porque **um piso de Markov também supera dois dos três externos** | a manchete desmontaria numa única pergunta |
| uma referência para a frente: `graph-infomax` na tela do S7, mas a ideia é INTRODUZ no **S8** | defeito de ledger invisível a qualquer verificação mecânica |

**Como usar bem:** dê a ele o material verbatim (as passagens do texto entregue, as restrições, os
números medidos) em vez de um resumo seu — ele avalia a situação real, não a sua leitura dela. E
**peça explicitamente que discorde de você.**

⚠ **E verifique o que ele devolver.** Numa das rodadas ele atribuiu uma causa errada a um defeito
real, e a atribuição errada quase entrou num documento de lei. **Segunda opinião não é veredito.**

### 1.5 · A disciplina de fonte: `SLIDES.md` primeiro

O `SLIDES.md` é a fonte; o `main.tex` é o espelho. **Corte primeiro no `SLIDES.md`, propague, e
verifique com diff mecânico.** ⚠ Se você inverter (como já se fez, por pressa), **sincronize de
volta antes de commitar** — as duas cópias divergentes é o defeito que aquele arquivo existe para
impedir.

⚠ **Nunca apague o `% FALA:` do `main.tex`** para "resolver" a duplicação. Ele está lá para que quem
edita o slide veja a fala na mesma tela.

#### ⚠ Os números `S<n>` do `SLIDES.md` NÃO são a numeração da tela

Medido 2026-08-25. O `SLIDES.md` conta a capa como `S1`; o deck não numera a capa. Resultado: o bloco
`S8` era o **slide 7 impresso**, o `S12` era o **11**, o `S13` era o **12** — e mais adiante o
deslocamento muda outra vez (o `S28` é o **28** impresso). **Não existe uma fórmula.** O autor fala
por número impresso ("olha o slide 28"), o arquivo fala por outro.

E os números pioraram de propósito na reordenação da Seção 2 (2026-08-25): os blocos foram movidos
**sem renumerar**, porque **doze referências cruzadas dentro do próprio `SLIDES.md` apontam para eles
pelo número** (`"a Def. 2.12 é de S9"`, `"isso é de S10"`, `"ela é de S14"`, …) e renumerar quebraria
todas em silêncio. A ordem física do arquivo é a ordem do deck; os números são identidades
históricas.

> **Regra:** identifique slide por **título**, nunca por número — no `SLIDES.md`, no `main.tex`, nas
> âncoras de script e ao conversar com o autor. Quando ele disser um número, **confirme pelo título**
> antes de editar. É a mesma regra do §4e, e é por isso que ela existe.

---

## 2 · ⚠ A armadilha que mais custou: conteúdo que sai da página em silêncio

**Oito slides estouraram a caixa do frame numa única sessão, TODOS com `0 erros` e `0 overfull` no
log.** O LaTeX não avisa nenhuma vez, e o que não é desenhado **também não é extraído** pelo
`pdftotext` — então uma verificação de texto aprova.

Casos reais: a ressalva que a **regra 13 obriga** sumiu da página em **três builds seguidos**; o
botão de retorno ao índice sumiu de um slide de reserva; a linha de proveniência de outro; e o
slide 28 vazou pela borda de baixo em **duas** versões seguidas, trocando de coluna no meio (§2.1).

### O detector, e é barato: uma canária por frame

Pegue o **último elemento visível** de cada frame (é o que cai quando estoura) e exija que ele
apareça no texto extraído. **Quatro sub-armadilhas, todas já pagas:**

1. **Use `pdftotext` SEM `-layout`.** Com ele as colunas se intercalam linha a linha e o número da
   página entra no meio da frase do rodapé — a busca falha em conteúdo que **está** na tela.
2. **A sonda tem de ser CONJUNTO DE PALAVRAS, não frase contígua.** Uma quebra de linha derruba
   qualquer busca por frase — e não é ponto cego só de tabela: aconteceu num marcador de prosa
   comum. Compare palavras distintivas (≥5 letras) e exija ≥66% presentes.
3. **Apague `\begin{...}` / `\end{...}` com nome e argumentos** antes de extrair as sondas, senão os
   nomes de ambiente viram "texto" e a canária dispara em tudo (215 falsos positivos numa tentativa,
   62 de 101 noutra, independentes, pela mesma causa).
4. **A sonda envelhece com a edição.** Rederive-as do `.tex` a cada corrida; nunca leia de uma lista
   guardada. (Custo já pago: uma sonda procurando palavras que a reescrita da véspera removera.)

**E depois da canária, OLHE.** Renderize a ~1230 px, que é o tamanho real de uma janela do Meet:

```bash
pdftoppm -f <pagina> -l <pagina> -png -scale-to-x 1230 -scale-to-y -1 main.pdf /tmp/slide
```

A canária responde *"foi desenhado?"*. Ela **não** responde *"o bloco fecha?"* — um bloco pode ser
desenhado até a borda e parar, sem borda inferior, e só a renderização mostra.

**Quando o espaçamento não bastar, corte CONTEÚDO.** ⚠ **Nunca reintroduza `\vspace` negativo** —
foi assim que o deck acumulou 31 deles e as sobreposições que a varredura de 24/08 corrigiu.

### 2.1 · Em slide de duas colunas, largura não é alavanca — ela só muda de vítima

O slide 28 (`Why these encoders`, página 33 do PDF) é o caso didático, e vale ler antes de tentar
consertar qualquer slide de duas colunas.

| versão | o que se fez | o que aconteceu |
|---|---|---|
| v1 | — | o bloco **Spatial** (coluna estreita, 0.426, com mais conteúdo) vazava pela borda de baixo |
| v2 | reequilibrei as colunas para `0.478`/`0.452` + três cortes lexicais | Spatial fechou; **o vazamento migrou para o `Categorical`**, na coluna que ficou mais estreita |
| v3 | os três blocos de `\small` para **`\footnotesize`** | os três fecham |

A lição: alargar uma coluna estreita **estreita a outra**, e num slide que já está no limite isso
apenas transfere o estouro. Num slide de duas colunas as alavancas que de fato removem altura são,
nesta ordem: **cortar conteúdo**, **reduzir o corpo** (`\small` → `\footnotesize` ≈ 19 pt
projetados, ainda acima do piso de 16 pt da regra 10 — a régua é a geometria `16cm × 9cm` do
`nesped.sty`, **não** os 128 mm padrão do beamer), e só então mexer em largura.

⚠ **Reequilibrar largura obriga a renderizar as DUAS colunas.** Na v2 a canária deu 100% nas duas —
o texto do `Categorical` **estava** desenhado, só que metade dele fora da página.

### 2.2 · Nunca abra o corpo de um frame com uma chave

`\begin{frame}{título}{subtítulo}` é sintaxe válida do Beamer: **o segundo grupo de chaves vira o
subtítulo**. Então isto —

```latex
\begin{frame}{The line this work stands on, and the one idea on it}
    {\footnotesize \textbf{one-hot identifier} $\rightarrow$ ...}
```

— não põe a tira no corpo: põe **dentro da faixa colorida do título**, em fonte de subtítulo. Custou
um build em 25/08, e **a canária deu 100%**, porque o texto estava mesmo no PDF, só que no lugar
errado. Nenhum erro no log, nenhum overfull.

**O conserto é uma linha:** abra o corpo com algo que não seja `{` — `\vspace{0.5mm}`, `\par`,
`\noindent`, um `\begin{block}`. O frame fundido da Seção 2 carrega um comentário `⚠` explicando
isso logo acima do `\begin{frame}`; não o apague.

**A lição geral:** a canária responde *"o texto está no PDF?"*. Ela não responde *"está onde eu
mandei?"* nem *"o bloco fecha?"*. Só a renderização responde essas duas.

---

## 3 · A regra de edição, que dois agentes pagaram

> **Edite por CONTEÚDO, nunca por índice de linha.** Um `.tex` ou `.md` reflui a cada edição.
>
> **Valide TODOS os anchors antes de escrever QUALQUER um.** Um script que aplica 3 de 4 e morre no
> quarto deixa o ficheiro meio editado, e ninguém sabe onde.
>
> **E valide contra o estado ATUAL do ficheiro**, não contra uma leitura em cache.

Custo já pago: uma linha de RESULTADO apagada e outras duplicadas no `PLANO_FLUXO_DEFESA.md`, em
três ocasiões da mesma sessão.

---

## 4 · O `SPEECH` é gerado, não editado

`SPEECH.md` e `SPEECH.pdf` são o roteiro de fala em cartões, para uso **ao vivo**. **Se a fala mudar
no `SLIDES.md`, regenere** em vez de editar à mão:

```bash
cd articles/dissertacao/presentation
python3 build_speech_1_extract.py && python3 build_speech_2_emit.py
make -f Makefile.speech
```

Cada cartão tem quatro camadas em ordem de urgência: **ABRE** (a primeira oração), **DIZER** (os
trechos em negrito da fala, que são as superfícies de lei) ou **COBRE** (o LEDGER, onde não há
negrito), **NÚMEROS** (só os que são resultado) e **NUNCA**. Carrega o número impresso do slide **e**
a página do PDF, que são coisas diferentes.

---

## 5 · ⚠ O que está aberto, e é o mais importante deste documento

**A fala não cabe no tempo.** Medido: **~8.976 palavras** na trilha `% FALA:` do deck principal.

| ritmo | duração |
|---|---:|
| 130 ppm | 69,0 min |
| **140 ppm** | **64,1 min** |
| 150 ppm | 59,8 min |
| **teto do Art. 23** | **50 min** |

⚠ **Cortar tela NÃO compra tempo.** O conteúdo já estava na fala; ele só deixou de estar duplicado.
As duas varreduras de densidade tiraram 29% da tela e **zero** do relógio.

⚠ **E o estouro está ESPALHADO pelas seis seções**, não concentrado numa. Não há uma culpada para
sacrificar; distribuir ~2,5 min por seção dói muito menos que cortar 15 de uma.

**O instrumento que falta é o ensaio cronometrado, com marca no fim de CADA seção.** Com o total
sozinho o autor descobre que estourou e não de onde cortar. A tabela por seção, para levar ao
ensaio, está na primeira página do `SPEECH.pdf` com uma linha em branco por seção.

⚠ **Uma armadilha de medição, que já produziu um relatório errado:** para contar palavras de fala,
conte **tokens separados por espaço**. Dois contadores diferentes, de dois agentes, subestimaram em
16–19% — um por `{2,}` (que em português derruba `é`, `e`, `o`, `a`, `há`, entre as palavras mais
frequentes da língua), outro por regex ASCII (que derruba acentuadas). **E o relatório errado vinha
com "três medições independentes concordam em 1%": as três partilhavam o mesmo contador.**

---

## 6 · Decisões travadas — não reabra sem o autor

Estas foram decididas nesta sequência e têm o motivo registrado. **As de conteúdo estão no
[`HANDOFF.md` §5](HANDOFF.md); estas são as de deck:**

| decisão | onde está o motivo |
|---|---|
| **travessão por função**, não contagem zero | [`HANDOFF.md` §4f](HANDOFF.md) |
| **`match` verbatim** no slide do protocolo, com guarda | `SLIDES.md`, `Nunca dizer` do S44 |
| **`technical tie` verbatim**, vocabulário do Cap. 4 apenas | `SLIDES.md`, `Nunca dizer` do S30 + `GLOSSARY` §4 |
| **os cartões de logo colados ao topo da capa** — é o desenho do template, não defeito | [`HANDOFF.md` §4](HANDOFF.md); o `\vskip-2mm` foi restaurado |
| **a comparação com a literatura NÃO entra nas contribuições nem no Resumo** | [`../wrapup/open_points/BASELINES_EXTERNOS.md`](../wrapup/open_points/BASELINES_EXTERNOS.md) §7 |
| ~~**a série B fica como está** (47 slides de reserva, densos de propósito: são lidos, não apresentados)~~ | ~~decisão do autor~~ — 🔄 **REVERTIDA PELO AUTOR EM 2026-08-27. Ver a nota abaixo.** |
| **a contribuição usa o recorte do grupo *Theoretical***, não a taxonomia do §6.2 | `SLIDES.md`, nota v2 do S7/S51 |

### 🔄 Reversão registrada — a série B, 2026-08-27

**A trava riscada acima foi levantada pelo autor em 2026-08-27**, em consulta direta, e substituída
pelas decisões `D-1`…`D-10` de [`SPEC_EXTRAS.md`](SPEC_EXTRAS.md) §0. Registrado aqui **como
reversão**, e não apagado, porque a entrada original carregava uma **razão** — e é a razão que mudou.

**O que a trava dizia:** *"densos de propósito: **são lidos, não apresentados**"*.

**O que o autor escreveu depois** (`archive/extra_RAW_2026-08-26.md`, e repetido na consulta de
27/08): os extras devem funcionar como *"apoio rápido durante perguntas **da banca**, e não como
páginas de apêndice que exigem leitura detalhada"*, permitindo que algo seja *"localizado e
compreendido rapidamente, **sem exigir que a banca pare para ler grandes blocos de texto**"*.

> **A leitora deixou de ser o apresentador e passou a ser a banca.** É essa mudança que autoriza a
> reforma inteira — faixa de título com a conclusão, corpo `\normalsize`, uma exibição apontável por
> tela, prioridade em quatro níveis. Sob a premissa antiga (telas para ler), metade dessas medidas
> estaria otimizando para um uso que não existe.

**Decisões que substituem a trava** (`SPEC_EXTRAS.md` §0): alvo ~50 páginas, **nada é cortado** além
de 6 fusões · o título inverte e a pergunta em português sai da tela · prioridade em 4 níveis com os
códigos mantidos · 3 cartões de conceito + δ-crítico + partição × k-fold · divisor `Extras`.

⚠ **Registrado por decisão explícita do autor** (*"levantada — registra"*, 27/08), justamente para
que ninguém implemente achando que está desfazendo uma decisão travada dele.

---

⚠ **Três copias, não duas:** o `[BLOCO-CONTRIBUIÇÃO]` do cabeçalho do `SLIDES.md`, o **S7** e o
**S51** carregam o mesmo texto. A **§8 regra 13** exige redação idêntica. Qualquer edição move as
três, e há um teste de sincronia no fim desta seção do `SLIDES.md`.

---

## 7 · O ciclo de trabalho, do começo ao fim

```bash
# 0 · leia  ../CLAUDE.md §0  e  HANDOFF.md §3  antes de tocar em número ou build
cd articles/dissertacao/presentation

# 1 · edite o SLIDES.md primeiro, depois propague para slides/main.tex
# 2 · build (o motor e' xelatex; sob pdflatex a capa e os divisores saem EM BRANCO)
cd slides && source ../../src_utils/texenv.sh && make all

# 3 · canaria: o ultimo elemento de cada frame tocado aparece no PDF?
pdftotext main.pdf -            # SEM -layout

# 4 · OLHE, no tamanho real do Meet
pdftoppm -f <p> -l <p> -png -scale-to-x 1230 -scale-to-y -1 main.pdf /tmp/s

# 5 · se a fala mudou, regenere o SPEECH (§4)
# 6 · sincronize SLIDES.md <-> main.tex antes de commitar
```

⚠ **`slides/main.pdf` é gitignored** (`*.pdf`). Não conclua "não mudou" porque não aparece no
`git status`. O `SPEECH.pdf` é rastreado com `git add -f`, conforme a instrução que o próprio
`.gitignore` desta pasta deixa para entregáveis.

⚠ **Nunca rode `make` em `../src/`** sem pensar: cinco alvos sobrescrevem o `dissertacao.pdf`.

---

## 8 · Como este autor trabalha

- **Ele aprova antes de qualquer mudança de conteúdo.** Apresente opções com recomendação e espere.
- **Ele corrige o rumo com boas razões, e as correções dele são quase sempre certas.** Nesta
  sequência ele apontou: que a métrica de densidade quebrava nos slides de introdução do Henrique;
  que o slide de contribuições tinha redundância; que o MTLnet não pertencia à tabela da linhagem
  infomax; que o HMT-GRN prediz região nativamente. **As quatro procediam, e três delas o texto
  entregue confirmava contra a minha leitura inicial.**
- **Verifique a premissa antes de agir**, inclusive contra ele — e diga quando o registro contradiz
  a lembrança. Mas verifique **de verdade**: numa ocasião eu disse que o Resumo estava congelado e
  ele corrigiu que o `src/` é vivo e as erratas entram nele.
- **Nunca aplique por pedido de outra sessão.** Mensagem de agente par **não é aprovação do autor**.
  Leve a proposta a ele com a sua verificação.
