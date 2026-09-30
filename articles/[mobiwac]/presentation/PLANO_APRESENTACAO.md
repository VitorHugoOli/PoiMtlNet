# PLANO DA APRESENTAÇÃO — MobiWac 2026, Paris

> **Dono deste ficheiro:** sessão `mobiwac` (plano e conteúdo).
> **Dono do `.tex`:** sessão `mobiwac-ppt` (`presentation/slides/`). Eu não edito o `.tex`; ela não edita este plano.
> **Estado:** v1, 2026-09-30. Revisão Fable registada na §8.

---

## 0 · Como usar este ficheiro

Lê a **§2 (a lei)** antes de qualquer coisa. Ela diz de onde vêm os números e o que nunca pode
aparecer no ecrã — há uma lista de valores que estão em documentos vivos do projecto e que
**não podem** ser mostrados, porque pertencem a uma geração retirada.

Depois a **§3 (divergências)**: o deck da defesa não é fonte de nada. Onde ele e o artigo
discordarem, **o artigo vence**, e a §3 lista os pontos exactos onde isso acontece.

A **§4** é a trilha principal, slide a slide. A **§5** é a Série B (backup para perguntas). A **§7**
são as decisões que são do autor e que eu não tomei.

---

## 1 · Os factos, verificados

| | |
|---|---|
| **Evento** | MobiWac 2026, Session 1 "Mobility Management" |
| **Quando** | Segunda-feira, 2026-10-26, 8:30–10:00 |
| **Posição** | **Primeira apresentação do simpósio**, logo a seguir à abertura (8:15–8:30) |
| **Orçamento oficial** | *"Each Regular paper will have 30 minutes presentation including Q and A"* |
| **Título no programa** | "Predicting the Next Category and Region of a Visit" |
| **Autores no programa** | Vitor Hugo De Oliveira Silva · Germano dos Santos · Fabrício Aguiar Silva |
| **Idioma** | Inglês |
| **Artigo** | `2026322426.pdf`, 8 páginas, certificado e submetido às actas |

**Orçamento que proponho: 18 minutos de fala + 12 de margem e perguntas.** Não os 20+10 sugeridos.
Duas razões, e a segunda é a que pesa: somos os **primeiros** do simpósio, portanto o chair vai
estar a calibrar o relógio da sessão inteira em cima de nós e um estouro nosso arrasta os dois
seguintes; e 8:30 de segunda é a pior janela de atenção da conferência — 18 minutos bem ditos valem
mais do que 22 apressados. A §4 soma exactamente 18:00.

**A plateia.** MobiWac/MSWiM é de redes sem fio e gestão de mobilidade, **não de aprendizagem
automática**. Pelo programa, os vizinhos de sessão são TSCH para enxames de robôs e um sistema
multi-agente com LLM; à volta, 5G/6G, SDN, UAV, WiFi. Consequências directas, e são a razão pela
qual esta trilha não é o capítulo da dissertação encurtado:

- **o fio de motivação de gestão de mobilidade pesa mais aqui do que na dissertação** — e o artigo
  já o tem escrito (caching, provisionamento de capacidade, predição de handover);
- **não se pode assumir** GNN, macro-F1, MTL, Holm, TOST. O que não for explicado em meia frase
  sai do ecrã;
- **o resultado mais vendável a esta plateia não é estatístico, é operacional**: um modelo, uma
  passagem para a frente, duas respostas.

---

## 2 · A lei

### 2.1 De onde vêm os números

**Fonte única:** `CAMERA_READY.md §3` (v18, convenção *joint-best*). Não usar o `RESULTS_BOARD.md`
(está morto), nem números lidos do deck sem os conferir contra o §3.

- Categoria (macro-F1), conjunto vs dedicado: **supera em 1 de 6** — Florida **+0,19** (Holm p = 0,011).
- Região (Acc@10), conjunto vs dedicado: **supera em 2 de 6** — Texas **+1,21**, California **+1,06**.
- As outras quatro células de região ficam **dentro da margem de 2 pp** pré-registada (TOST).
- Representação (Tab. 2): **+0,23 a +6,29** macro-F1, 6/6 datasets, 5/5 folds cada.
- Margens sobre as externas: categoria ≥ **+3,06**; região ≥ **+3,55**.
- Piso de Markov excedido por **+4,1 a +10,0**.

### 2.2 ⛔ NUNCA MOSTRAR

Estes valores estão em documentos do projecto e em versões antigas. São da geração **v17**, que
tinha um vazamento de rótulo. **Nenhum pode aparecer num slide.** Lista completa em
`CAMERA_READY.md §4`; os que mais provavelmente se infiltram num deck:

| ⛔ | Substituto correcto |
|---|---|
| Categoria conjunto 63,32 / 64,51 / 65,79 / **79,84** / 77,24 / 77,05 | 35,42 / 30,59 / 34,57 / 37,55 / 36,19 / 35,63 |
| "supera o dedicado em todos os datasets" (categoria) | supera **só em Florida** |
| "supera em quatro de seis" (região) | supera em **dois**: TX e CA |
| Representação "+27,63 … +39,62" / "cerca de 28 a 40 pontos" | **+0,23 … +6,29** |
| "pelo menos 4 Acc@10" e "pelo menos 33 macro-F1" | **3,55** e **3,06** |
| Markov excedido por "4,9 a 10,3" | **4,1 a 10,0** |
| "a lei de escala com o número de regiões" | **morta** — CA tem mais regiões do que TX e ganho menor |
| Deltas do CTLE +37,8 / +37,0 / +28,7 | **não existem** em v18; só a ordenação sobrevive |

> ### ⛔ A armadilha mais tentadora, e é específica desta plateia
>
> Existe no projecto uma **métrica de erro geográfico** (*near-miss*): a mediana da distância entre a
> região prevista e a verdadeira, 3,16–8,13 km, com um piso de par aleatório de 20–241 km — ou seja,
> os erros caem ~6 a 34× mais perto do que o acaso. **É o número mais sedutor que temos para uma
> plateia de redes, e não pode ir para o ecrã.** Duas razões independentes, qualquer uma bastava:
>
> 1. **foi medida sobre o substrato v17** (`dk_ovl`, semente 0, PR #59) — é da geração retirada;
> 2. **o artigo publicado diz explicitamente que não foi medida**: *"measuring it requires per-visit
>    predictions that the evaluation path does not retain, so it is left to future work. This remains
>    motivation, not a measured service result."* (`src/sections/07_discussion.tex`).
>
> Mostrá-la contradiz o texto publicado **e** cita uma geração morta. Não é decisão do autor: é
> proibição. Re-medir em v18 também não é saída — o `nespedgpu` parou permanentemente para trabalho
> de GPU (2026-09-30).
>
> **O que se pode dizer, e é publicado e serve o mesmo propósito:** a leitura de Acc@10 como taxa de
> acerto de uma *shortlist* — *"em California, dez regiões em 8 501 contêm a região verdadeira 64,54 %
> das vezes; no Texas, dez em 6 553 contêm-na 66,15 %"*. Está na §7 do artigo, é v18, e é exactamente
> a formulação que uma plateia de serviços entende. **Usar esta, nunca a dos quilómetros.**

### 2.3 Uma decisão do autor que está fechada e não se reabre

**D1 — avisar os *chairs* da mudança v17→v18: DECIDIDO PELO AUTOR a 2026-09-06, rota (a):** corrigir
o texto, deixá-lo o melhor possível, reenviar, **sem nota aos chairs**. O `CAMERA_READY.md §6 D1`
regista a decisão e diz *"está encerrada e não se reabre"*.

**Consequência para o palco:** **não existe slide de divulgação v17→v18**, nem na trilha principal
nem na Série B. O que se apresenta são os números publicados, que são os do artigo. O plano não
propõe nada em contrário.

**Mas há um efeito no Q&A que o autor deve conhecer**, e por isso existe a **B4**: o artigo publicado
*já descreve* a propriedade *forward-only* no terceiro limite da §7 — *"the graph does not pass
information from a later visit back to an earlier one, in training or at readout, which is what keeps
a node from carrying a feature of the target it is used to predict"*. Se alguém perguntar sobre
vazamento em grafos de visitas, **a resposta está no texto publicado** e aponta-se para lá. Isso é
responder a uma pergunta técnica, não reabrir a D1.

---

## 3 · Divergências deck ↔ camera-ready

Método: peguei nos commits que alteraram conteúdo no artigo e procurei no deck as frases que saíram.
O resultado é **melhor do que eu esperava** — o deck foi construído depois da regeneração v18, para a
defesa de 2026-08-28, e enuncia vereditos em vez de mecanismos. **Não carrega nenhuma** das frases que
o camera-ready corrigiu nas últimas semanas.

Verificado, ocorrências no deck: alegação de primazia **0**; linguagem de atribuição à partilha **0**;
controlo de capacidade **0**; a frase da cobertura da procura **0**; a divulgação do prior de transição
**0**. Os números das tabelas do deck batem **célula a célula** com o `CAMERA_READY §3`.

Sobram **três** divergências reais. Duas são para corrigir, uma é a armadilha da §2.2.

### D-1 · Arredondamento na contagem de parâmetros — CORRIGIR

- **deck:** `AL 4.2 M vs 1.85 M --- 2.3×`
- **artigo:** *"about 4.2 million parameters at Alabama against **1.9 million** for the two combined
  (5.2 against 2.8 at California)"* (`src/sections/04_method.tex`)

O `1.85` do deck e o `1.9` do artigo são o mesmo número com arredondamentos diferentes. Um ouvinte
não nota; um leitor das actas que confira, nota. **Usar a redacção do artigo.** Também vale notar que
o artigo diz *"larger than either dedicated model **and than the two combined**"* — a comparação é
contra a soma, e o slide deve deixar isso explícito para não parecer que 4,2 M se compara com um só.

### D-2 · Os quatro limites não são os mesmos — CORRIGIR, e importa

| # | Deck (defesa) | Artigo publicado (§7) |
|---|---|---|
| 1 | Representação treinada uma vez, sobre todos os lugares | ✅ igual (+ o *rebuild* por fold move ≤ 0,33 Acc@10 / 0,29 macro-F1) |
| 2 | Sem terceira partição — scores absolutos optimistas | ✅ igual (+ o dedicado de categoria recebe a procura mais larga, o que torna a diferença de categoria conservadora) |
| 3 | **Nenhum serviço de mobilidade construído ou avaliado** | ❌ **não é um dos quatro limites do artigo** |
| 4 | Cada visita lê só o seu próprio passado | ✅ igual (é o limite *forward-only*) |
| — | *(ausente do deck)* | ⚠ **"o modelo conjunto tem mais parâmetros do que os dois dedicados somados, portanto o resultado de região em TX e CA pode vir do tamanho e não da partilha. O desenho não separa os dois."** |

**O deck trocou o limite que guarda a alegação principal por outro.** O "nenhum serviço construído"
está no artigo — mas **dentro do esboço de serviço**, como a frase *"This remains motivation, not a
measured service result"*, não como limite numerado. O limite numerado que falta ao deck é o de
**tamanho vs partilha**, e é precisamente o que impede a leitura de que as duas vitórias de região
provam que a partilha funciona.

**Recomendação: o slide dos limites usa os quatro do artigo.** A frase do serviço entra noutro sítio
— no slide da leitura honesta (S11) ou no do esboço de serviço —, onde é uma ressalva de âmbito e não
um dos quatro limites formais.

### D-3 · A métrica geográfica — ver §2.2. **Não entra.**

---

## 4 · A trilha principal — 12 slides, ~15 min

> **v2, depois da revisão Fable.** A v1 tinha 15 slides e 18:00. Três mudanças estruturais, todas
> aceites e todas verificadas contra o artigo antes de aceitar — ver §8.

Convenção de cada entrada: **mensagem única**, **mostra**, **fonte**, **segundos**, **do deck**.

---

### S1 · Título — 20 s
- **Mostra:** título completo, os três autores, NESPeD-LAB / Universidade Federal de Viçosa, MobiWac 2026.
- **Do deck:** **novo.**

---

### S2 · O dia de uma pessoa, e a pergunta — 130 s
- **Mensagem:** ninguém precisa do próximo *lugar*; duas perguntas mais grosseiras chegam — que tipo de sítio, e que bairro.
- **Mostra:** uma figura **esquemática** (TikZ, desenhada no `.tex`): uma malha de *tracts*, nove pontos
  ligados por setas rotulados *coffee · gym · office · lunch*, e um `?` num *tract* realçado; por baixo,
  duas caixas — *what kind of place?* · *which neighbourhood?*
- **⚠ Não é um mapa real nem um utilizador real**, e não deve ser apresentada como tal. É um esquema.
  Se alguém perguntar, é uma ilustração do formato da entrada, não um caso do conjunto de dados.
- **Fonte:** `src/sections/01_introduction.tex` ¶1–2 · `src/tables/tbl1_datasets.tex` (contagens) · `04_method.tex` §IV-A (o âmbito) · `05_setup.tex` (o horizonte) · `07_discussion.tex` (a shortlist).
- **Do deck:** **novo.**

**Três coisas que este slide tem de fazer, e a v1 falhava nas três:**

1. **Definir "região" aqui, não no backup.** A palavra usa-se do S2 ao S11 e na v1 só estava definida
   num slide de Série B. Ninguém naquela sala sabe que significa um *census tract* (ou uma *mahalle*
   em Istanbul), 520 a 8 501 por conjunto. Sem isso não conseguem julgar se "dez em 8 501" é bom.
2. **Não abrir a reivindicar handover.** O artigo é explícito (`04_method.tex:38-41`): *"A census
   tract is a neighborhood, not a radio cell… Cell association and handover are radio-level decisions
   and remain out of scope."* A sessão chama-se "Mobility Management" e a sala é de handover. Abrir a
   reivindicá-lo convida à pergunta *"que ganho ao nível da célula mediram?"*, cuja resposta é
   "nenhum, está fora de âmbito". **Usar handover como analogia de escala, não como alegação** — é
   exactamente o que o artigo faz.
3. **Dar a resposta ao minuto 2, não ao minuto 11.** Às 8:30 de segunda, quem não souber ao fim de
   dois minutos se vale a pena ouvir, desliga.

**Guião proposto (~120 s falados, inglês, cada facto do artigo):**

> *[mapa, nove pontos, interrogação]*
> "Here is one person's day in Texas: nine check-ins. Coffee, gym, office, lunch. The tenth is the
> question. But nobody needs the tenth *place* — Texas has a hundred and sixty thousand of them. A
> service needs two coarser things. What **kind** of place comes next — food, shopping, nightlife.
> And which **neighbourhood** — a census tract. A thousand of them in Alabama, eight and a half
> thousand in California.
> If you know the neighbourhood before the person gets there, you can push content to the cache that
> serves it, and plan capacity where demand is about to land. You already do this one level down, at
> the radio cell, with handover prediction. We are at the neighbourhood. **A tract is not a cell, and
> we claim nothing about cells.**
> How far ahead? In our data the median gap from the last visit to the next runs from **under half an
> hour** in Florida to **five and a half hours** in Istanbul.
> The obvious engineering answer is two models, one per question — two things to train, version and
> serve. The obvious shortcut, one model for both, is known to fail: parameters shared between two
> jobs can settle on a compromise that is worse at both. So: **can one model answer both, and what
> does it cost?**
> The answer, so you can decide whether to keep listening: **yes.** One model, one forward pass. In
> California, ten neighbourhoods out of 8 501 contain the right one 64 percent of the time. On the
> neighbourhood question it beats a dedicated model at the two biggest datasets and is at worst
> **0.9 points** worse at the others. And on the type-of-place question, **how we represent a visit**
> moved the score more than anything about the model. Here is how."

---

### S3 · Check2HGI: um quarto nível abaixo do lugar — 100 s
- **Mensagem:** em vez de um vector fixo por lugar, cada check-in ganha o seu próprio vector, com o seu contexto.
- **Mostra:** a chapa `c2h_deep.pdf` (cidade → região → lugar → **check-in**), e uma linha para o que o vector carrega.
- **Fonte:** `src/sections/04_method.tex`; chapa em `articles/dissertacao/presentation/figures/plates/c2h_deep.pdf`.
- **Do deck:** **fundir dois frames** — "Check2HGI: a fourth level below the place" + "What each visit contributes" passam a um.
- **Frase que faz o trabalho:** *"o mesmo café às terças de manhã e aos sábados à noite não é o mesmo sinal."*
- **⚠ Manter, é o terceiro limite do artigo:** cada visita lê **só o seu próprio passado**.

---

### S4 · Resultado 1: a representação — 115 s
- **Mensagem:** mudando **só a entrada**, a categoria melhora nos seis conjuntos e em todos os folds.
- **Mostra:** a tabela de 6 linhas, Δ de **+0,23 a +6,29**.
- **Fonte:** `CAMERA_READY §3.3`; `src/tables/tbl2_substrate.tex`; o controlo em `06_results.tex:63-72`.
- **Do deck:** **reaproveitar** a tabela.

**⚠ Duas coisas ditas em voz alta, porque a tabela não as diz e ambas saem na primeira pergunta:**

- **O ganho é grande onde os dados são pequenos** (Istanbul +6,29) e **abaixo de um ponto nos três
  maiores**. Em Florida (+0,23) o desvio entre folds (0,42) é **maior do que o gap**, e o teste
  pareado **não separa** as duas representações (p = 0,067).
- **E o atalho barato faz quase o mesmo — dito por nós, não por quem pergunta.** O artigo mede um
  controlo de concatenação: pegar no vector de lugar e juntar-lhe as mesmas *features* cruas por
  visita (categoria em one-hot, hora, dia da semana). Ganha **+2,0 / +1,7 / +0,8** em AL/AZ/FL,
  contra os nossos **+1,62 / +2,58 / +0,23** nos mesmos estados. **Em Alabama e Florida o atalho
  ganha mais do que o grafo.** O artigo escreve-o assim: *"Most of the category difference is
  therefore already available in the raw per-visit features, and this control does not separate what
  the check-in-level representation adds beyond them on this axis."*
  **Meia frase no palco:** *"and to be clear — most of that gain is already there if you just hand
  the place vector the hour and the category; what the graph adds beyond that, this control does not
  separate."* Dito por nós é honestidade; dito por um ouvinte é uma ferida. Ver **B9**.

---

### S5 · A arquitectura — 50 s
- **Mensagem:** duas entradas, um tronco partilhado onde cada tarefa lê o contexto da outra, e uma via espacial privada para a região.
- **Mostra:** `fig2_model_slides.pdf`, uma frase.
- **Do deck:** **fundir e cortar** — os dois frames "sharing by exchange (1/2)" e "(2/2)" passam a um, de 170 s para 50. É o minuto 8; é aqui que a sala se vai embora se houver detalhe a mais. Ninguém nesta sala precisa de saber quantos blocos tem o tronco.

---

### S6 · Como foi medido — 90 s
- **Mensagem:** mesmos utilizadores, mesmas janelas, e a margem foi escrita antes de se ver qualquer resultado.
- **Mostra:** quatro linhas —
  - **as regiões:** 520 (Istanbul) a 8 501 (California), *census tracts*;
  - **a partição é por utilizador:** *"as visitas de um utilizador de teste nunca aparecem no treino"* (textual do artigo). Todos os números reportados são sobre utilizadores que o modelo nunca viu;
  - **vinte modelos treinados por configuração** (4 sementes × 5 folds);
  - **escrevemos antes de ver qualquer resultado o que contaria como "não pior": dois pontos**;
  - **o que se mede:** macro-F1 na categoria · **Acc@10 na região — *a região verdadeira está nas dez
    primeiras?*** ⬅ acrescentado a pedido do `mobiwac-ppt`, e bem: o S7 lê Acc@10 e nenhum slide a
    definia. Numa sala que não usa a métrica, uma tabela de Acc@10 sem esta linha é ruído.
- **Fonte:** `src/sections/05_setup.tex` (Windows/Splitting) · `CAMERA_READY §3` (a convenção).
- **Do deck:** **editar agressivamente** — os **quatro** frames "The protocol, in four steps" colapsam num.
- **⚠ A frase que não se corta:** a da margem pré-registada. É a única coisa aqui que distingue este trabalho de um afinado até dar bem.

---

### S7 · Um modelo, duas respostas: o que custa — 100 s
- **Mensagem:** o preço de um modelo em vez de dois, dito de uma vez.
- **Mostra:** **só** as colunas Dedicado vs Conjunto da Tabela 3 — 6 linhas × 2 tarefas, deltas a cor (verde em TX/CA/FL, cinzento no resto). **Sem IC no ecrã** (ilegível do fundo da sala; os intervalos ficam na Série B).
- **Fonte:** `CAMERA_READY §3.1 e §3.2`.
- **Do deck:** **editar** — a tabela do frame "The verdict" simplificada.

> ⚠ **O Δ não é a subtracção das colunas, e alguém vai subtrair.** O Δ é a média pareada sobre as
> quatro sementes; as colunas são arredondadas independentemente, e isso faz **cinco** células
> discordarem na segunda decimal: AL cat (−0,18 vs −0,19), FL cat (+0,20 vs +0,19), TX cat (−0,14 vs
> −0,13), AL reg (−0,88 vs −0,87), FL reg (−0,15 vs −0,16). O `CAMERA_READY §7`, item 6, diz que a
> deltas de um quinto de ponto **subtrair as colunas é a primeira coisa que um revisor faz**.
> **Nota no ecrã** (a que o `mobiwac-ppt` já pôs, e aprovo): *"Δ is the paired mean over the four
> seeds, so it can differ from the rounded columns in the second decimal."* Manter as três colunas —
> tirar Dedicado e Conjunto e deixar só o Δ pouparia a nota, mas custa à plateia a noção de escala
> (um Δ de +1,21 sobre 64,94 lê-se de outra maneira do que +1,21 sozinho).

**Veredito falado, textual:**

> *"On region, the one model is better at the two largest datasets, by about one point, and worse
> everywhere else, by at most 0.87 points — which is inside the two-point margin we fixed before we
> saw any result. On category the two are the same within a fifth of a point. Every one of the four
> region deficits is a real deficit, not a tie. **That is the price of one model instead of two.**"*

E a frase do artigo que é a tese, e que a v1 não tinha em lado nenhum (`07_discussion.tex:26`):
**"the trade is a measured one and not a free substitution."**

**⚠ Também aqui, faladas e não em slide próprio:** o conjunto **é também o modelo maior**, e o
desenho **não separa** tamanho de partilha; e o emparelhamento das duas vitórias com os dois maiores
vocabulários de região é **observação, não lei** — CA tem mais regiões do que TX e ganho menor.

---

### S8 · E contra os sistemas publicados — 45 s
- **Mensagem:** estamos acima de tudo o que voltámos a correr — e essa margem não é prova sobre partilha.
- **Mostra:** a tabela das externas (POI-RGNN · HMT-GRN · ReHDM · STAN), uma frase.
- **Fonte:** `CAMERA_READY §3.4`.
- **Do deck:** **reaproveitar**, mas **reduzido e despromovido**.

> **A mudança estrutural mais importante da v2.** Na v1 este era o primeiro e mais longo slide do
> Resultado 2 (105 s) e a mensagem era *"o conjunto está acima de todas as externas"*. O meu próprio
> aviso no mesmo slide dizia que isso **não é propriedade do conjunto** — o dedicado também lá está,
> e a margem **inclui** a vantagem da representação. O `CAMERA_READY §3.4` diz que **não pode liderar
> as contribuições nem o resumo**. Liderar com ela e retratá-la 70 s depois é vender, tabelar e
> desdizer. Agora vem **depois** do veredito e em 45 s.

**Frase única, corrigida (ver §8, F11):** *"Both our models — the dedicated and the joint — are above
every system we re-ran. The joint model by at least **3.06** points on category and **3.55** on region.
That margin belongs mostly to the representation from Result 1, so it is not evidence for sharing."*
E segue.

> ⚠ **Não dizer "both our models ... by at least three points".** Os 3,06 e 3,55 são margens **do
> conjunto**. As do **dedicado** são **+2,86** (categoria, mínimo em Florida sobre POI-RGNN) e **+3,27**
> (região) — `CAMERA_READY §3.4`. Dizer "ambos, pelo menos três pontos" é falso por 0,14 numa célula,
> e é o tipo de número que alguém confere.

---

### S9 · O que um serviço receberia — 60 s
- **Mensagem:** o que isto entrega a quem consome as predições — e o que não foi construído.
- **Mostra:** três linhas e uma ressalva.
  - **Horizonte:** mediana de **0,4 h (Florida) a 5,5 h (Istanbul)** da última visita ao alvo; **5 a 27 %** dos alvos ocorrem mais de **3 dias** depois.
  - **Utilizadores nunca vistos:** a partição é por utilizador; as visitas de um utilizador de teste nunca aparecem no treino. É uma propriedade de implantação, não um detalhe metodológico.
  - **Lista curta:** dez regiões em **8 501** contêm a verdadeira **64,54 %** das vezes (California); dez em 6 553, **66,15 %** (Texas).
  - **Ressalva, textual:** *"We built and evaluated no such service; this is motivation, not a measured service result."*
- **Fonte:** `05_setup.tex` (Windows/Splitting) · `07_discussion.tex` (a shortlist e a ressalva).
- **Do deck:** **novo.**

> **Porque é que isto passou a existir.** A v1 recusava um slide de serviço com o argumento de que
> *"um slide sobre um serviço não construído convida à pergunta 'então construíram?'"*. Essa pergunta
> é **boa** e a resposta é uma frase que está no artigo. O que não é bom é uma plateia de gestão de
> mobilidade sair sem nenhuma ideia do que um consumidor destas predições recebe. Os três factos
> acima estão todos no artigo e nenhum estava em slide nenhum. É também aqui que a frase do serviço
> não construído ganha casa — ela não é um dos quatro limites (ver D-2).

---

### S10 · O que custa, e quatro limites — 90 s
- **Mensagem:** o modelo conjunto é maior do que os dois somados, e há quatro coisas que qualificam tudo o que foi dito.
- **Mostra:** `AL 4,2 M vs 1,9 M` · `CA 5,2 M vs 2,8 M` (**contra os dois dedicados somados**); depois os quatro limites do artigo.
- **Fonte:** `04_method.tex` (com o arredondamento do artigo — ver D-1) · `07_discussion.tex`, "Four limits qualify these results".
- **Do deck:** **fundir e corrigir** — "Limitations and trade-offs" com o terceiro item trocado (ver D-2).
- **Porque é que os dois se fundem:** o custo **é** o quarto limite do artigo. Separá-los repetia a mesma ideia em dois slides.
- **⚠ Preparar a pergunta que este título provoca:** para esta sala, "custo" são milissegundos e watts, não parâmetros. Ver a tabela de perguntas sem slide, na §5.

---

### S11 · Conclusão — 60 s
- **Mensagem:** quando a representação preserva o contexto de cada visita e a arquitectura mantém uma via espacial privada onde as tarefas diferem, um modelo prevê o quê e o onde numa só passagem.
- **Mostra:** três linhas, **com as palavras da conclusão do artigo**: a evidência *"does not attribute the outcome to sharing alone"* · um modelo lê as duas respostas de um só ficheiro guardado · *"These results do not mean that multi-task learning helps automatically."*
- **Fonte:** `src/sections/08_conclusion.tex`, quase textual.
- **Do deck:** **novo.**
- **Nota sobre "a cost bounded on both axes":** a v1 punha esta frase aqui como se fosse da conclusão.
  É do **resumo** (`src/main.tex:115`), não da conclusão. É legítima — é do artigo — mas quem conferir
  contra o PDF vai procurá-la na §VIII e não a encontra. Usar as palavras da conclusão.

---

### S12 · Reprodutibilidade e agradecimentos — 30 s
- **Mostra:** `github.com/VitorHugoOli/PoiMtlNet/tree/mobiwac` · as duas fontes de dados públicas · **FAPEMIG, CNPq, CAPES**.
- **Fonte:** `src/main.tex` (nota de rodapé do código) e `\section*{Acknowledgment}`.
- **Do deck:** **novo.** O deck não tinha nem o URL nem os financiadores. É o slide que fica no ecrã durante as perguntas e o que é fotografado.

---

**Soma:** 20 + 130 + 100 + 115 + 50 + 90 + 100 + 45 + 60 + 90 + 60 + 30 = **890 s ≈ 14:50.**

**Isto é de propósito.** Sobram ~15 minutos de margem e perguntas num slot de 30. Às 8:30 de segunda,
como primeiro orador, **ar vale mais do que slides** — e um orador que acaba antes do tempo e responde
bem deixa melhor impressão do que um que estoura. Se o autor quiser aproximar-se dos 18 minutos, os
três sítios onde o tempo rende, por esta ordem: **+60 s no S4** (o controlo de concatenação merece ser
explicado devagar), **+40 s no S7** (o veredito é o núcleo), **+30 s no S2**. Não repor o S5 nem o
antigo slide de trabalhos relacionados.

**O que saiu da v1 e porquê:** o slide de trabalhos relacionados (40 s) — esta plateia não tem mapa
onde pôr MCMG e HMT-GRN, e a única frase de que precisa cabe no fim do S2; e o slide autónomo da
"leitura honesta" (70 s) — as suas três balas passaram a ser ditas **junto dos números**, no S7, que
é onde o artigo as põe. Isolar as ressalvas num slide só delas fá-las soar a confissão.

---

## 5 · Série B — backup para as perguntas

Um slide, uma resposta, sem construção. Chamam-se pelo número.

| # | Pergunta previsível | O que o slide mostra | Fonte |
|---|---|---|---|
| **B1** | *"Porquê *tracts* e não uma grelha?"* | A definição e a contagem por conjunto (520 → 8 501). **Reduzido**: a definição de região passou para o S2. | `CAMERA_READY §3.4`; `tbl1_datasets.tex` |
| **B2** | *"De onde vêm os 2 pontos? Escolheram depois de ver?"* | Pré-registada, **só para a região**; a categoria **não tem** margem registada — por isso as cinco células dizem "não resolvido" e nunca "equivalente" | `CAMERA_READY §3` |
| **B3** | *"Um Markov não faz isto?"* | Piso de Markov-1 (51,23–72,47), margem **+4,1 a +10,0**; e HMT-GRN abaixo desse piso nos seis, STAN em quatro, ReHDM em três. **⚠ Dizer a margem mais estreita antes que a calculem: Florida, +4,07.** | `CAMERA_READY §3.4` |
| **B4** | *"Como sabem que o vector não vê o alvo?"* | O grafo é **forward-only**: um nó lê só as visitas que o precedem, no treino e na leitura. **Publicado, terceiro limite da §7.** | `07_discussion.tex` |
| **B5** | *"Quanto custa um modelo em vez de dois?"* | 4,2 M vs 1,9 M (AL), 5,2 M vs 2,8 M (CA), contra **os dois somados** | `04_method.tex` |
| **B6** | *"Se a representação faz quase tudo, para que serve o MTL?"* | A resposta honesta do artigo: na categoria a representação move mais do que a escolha entre um modelo e dois; se a troca acrescenta algo **não é separado pela evidência aqui** | `08_conclusion.tex` |
| **B7** | *"Porquê estas referências externas?"* | A tabela completa com as duas notas honestas: STAN com folds parciais (TX 4/5, CA 2/5), ReHDM com uma só semente em TX e CA | `CAMERA_READY §3.4` |
| **B8** | *"O ganho cresce com o número de regiões?"* | **Não.** Observação, não lei: CA tem mais regiões do que TX e ganho **menor** (+1,06 vs +1,21) | `CAMERA_READY §5 C4` |
| **B9** | *"Isto não é só juntar o timestamp ao vector de lugar?"* | **NOVO, e é a pergunta mais perigosa da sala.** O controlo de concatenação: +2,0 / +1,7 / +0,8 (AL/AZ/FL) contra os nossos +1,62 / +2,58 / +0,23. **Em AL e FL o atalho ganha mais.** A frase do artigo, textual. | `06_results.tex:63-72` |
| **B10** | *"E os intervalos de confiança?"* | A tabela de 12 células com IC a 90 % — os que saíram do S7 para ser legível do fundo da sala | `CAMERA_READY §3.1, §3.2` |

**Perguntas sem slide, com a frase preparada.** Esta tabela é tão importante como os slides: são as
perguntas desta sala em concreto, e nenhuma tinha resposta na v1.

| Pergunta | A frase |
|---|---|
| **"Latência? Energia? Corre na borda, no telemóvel?"** | *"About five million parameters and a window of nine visits per query. **We did not measure latency or energy.**"* ⚠ O título do S10 ("o que custa") torna esta pergunta inevitável: para esta sala, custo são **milissegundos e watts**, não parâmetros. |
| **"Partiram por utilizador e não por tempo. E deriva? E re-treino?"** | Textual do artigo: *"The split is by user and not by time, so the evaluation does not measure drift or the effect of sporadic events."* E depois o lado bom: por ser por utilizador, **todos os números são sobre utilizadores nunca vistos**. |
| **"Gowalla é de 2009–2011. Ainda representa alguma coisa?"** | Istanbul vem do Massive-STEPS, recolha recente; o Gowalla é o *benchmark* padrão desta linha. O que interessa é o padrão que **se repete** entre continentes e épocas. |
| **"Privacidade: o operador precisa do histórico completo de cada utilizador, e a representação foi treinada sobre toda a gente."** | Do lado do servidor, sobre os registos do próprio operador; **nenhum mecanismo de privacidade é reivindicado**. E o primeiro limite mede o efeito: reconstruir a representação só com utilizadores de treino move os resultados ≤ 0,33 Acc@10 e ≤ 0,29 macro-F1. |
| **"E em quilómetros, os erros ficam a que distância?"** | ⚠ **Ensaiar esta recusa em voz alta**, porque vem como pergunta simpática de alguém que ia gostar da resposta. *"Not measured — it needs per-visit predictions the evaluation path does not keep. It is the first thing we would measure for a service."* **Nunca dar os números de km** (§2.2). |
| **"Só quatro corridas?"** | Cada semente é uma validação cruzada completa de 5 folds — **vinte modelos treinados** por configuração. O desvio entre sementes da diferença pareada é 0,02–0,16. |
| **"Quem consome isto — o operador, a aplicação, a cidade?"** | Antecipação de carga e procura ao nível do bairro: *caching*, planeamento de capacidade. **Nomear o consumidor**, não deixar em abstracto. |

## 6 · O que o deck da defesa não tinha e esta plateia precisa

Quatro coisas. As três primeiras são as que mais rendem por segundo investido.

1. **A motivação de rede, à frente (S2).** O deck abre pelo diagnóstico do capítulo anterior da
   dissertação — uma continuidade que aqui não existe. Para esta plateia a porta de entrada é
   *caching* e provisionamento de capacidade, e o artigo já a tem escrita.
2. **A leitura de Acc@10 como taxa de acerto de uma shortlist (S9/S11).** *"Dez regiões em 8 501
   contêm a verdadeira 64,54 % das vezes"* é a frase que traduz uma métrica de ML para uma decisão de
   serviço. Está publicada. É o substituto legítimo da métrica de quilómetros proibida.
3. **Reprodutibilidade e financiadores (S15).** O deck não tinha nem o URL do código nem
   FAPEMIG/CNPq/CAPES; o artigo tem os dois, e o agradecimento entrou a pedido do orientador.
   Numa conferência, o URL no último slide é o que fica fotografado.
4. **O posicionamento vs trabalhos relacionados, em 40 segundos (S3).** A plateia não conhece esta
   literatura. Não precisa do mapa; precisa de saber que a pergunta não estava respondida.

E uma coisa que o deck tinha e que aqui **sai**: a escada dos três estudos da dissertação
(CBIC → CoUrb → MobiWac). Não há contexto para ela numa sessão de 30 minutos e ela não ajuda a
responder à pergunta do artigo.

---

## 7 · DECISÕES DO AUTOR

Não decidi nenhuma destas. Cada uma com opções e a minha recomendação.

### DA-1 · Duração alvo
> ⚠ **Actualizado para a v2.** A v1 dizia "18 min" e "a §4 soma 18:00". Depois dos cortes da revisão
> **a §4 soma ~14:50**, e eu deixei a recomendação para trás — um ponteiro podre dentro do meu próprio
> documento, do mesmo tipo que ando a converter no resto do repositório.

- **(a) ~15 min de fala, ~15 de margem e perguntas** — ⭐ **recomendo**, e é o que a §4 entrega hoje.
  Somos os primeiros do simpósio: o chair calibra o relógio da sessão em cima de nós, e um estouro
  arrasta os dois seguintes. Acabar cedo e responder bem deixa melhor impressão do que estourar.
- (b) ~18 min — gastar os 3 minutos onde a §4 diz: +60 s no S4 (o controlo de concatenação merece ser
  explicado devagar), +40 s no S7, +30 s no S2. **Não** repor a arquitectura nem os trabalhos relacionados.
- (c) 20+ min — não recomendo em nenhuma circunstância neste slot.

### DA-2 · Quanta estatística vai ao ecrã
- **(a) Vereditos e intervalos, sem nomear os testes** — ⭐ **recomendo.** "Supera", "fica dentro da
  margem registada", com o IC a 90 % visível. Holm e TOST ficam para a Série B (B2).
- (b) Nomear Holm e TOST na trilha principal — é mais rigoroso e é o que a defesa fez, mas custa
  ~25 s a explicar a uma plateia que não os usa.

### DA-3 · Se perguntarem directamente porque é que os números diferem de uma versão anterior
Isto **não reabre a D1** — D1 é sobre notificar os chairs, e está fechada. Isto é sobre o que se
responde de pé, se alguém que tenha visto o manuscrito aceite perguntar.
- **(a) Responder pelo mecanismo, sem história de processo** — ⭐ **recomendo.** A resposta é a B4:
  a representação é construída sobre um grafo *forward-only*, e está declarada no artigo publicado.
  É verdadeira, é curta e é o que está no texto.
- (b) Descrever a mudança de geração explicitamente — honesto, mas abre um tema de processo à frente
  de uma sala, e o autor já decidiu que o lugar disso era o texto.
- **É decisão sua porque é a sua voz no palco, não a minha.** O plano não põe nada disto num slide.

### DA-4 · O esboço de serviço entra ou sai — ⚠ **mudei de recomendação na v2**
O artigo tem um esboço de serviço de ~90 palavras na §7 (a leitura de shortlist).
- **(a) Slide próprio de 60 s — o S9** — ⭐ **recomendo agora.**
- (b) Meia frase noutro slide, sem slide próprio — era a minha recomendação na v1.

**Porque mudei.** Na v1 recusei o slide próprio com o argumento de que *"um slide inteiro sobre um
serviço não construído convida à pergunta 'então construíram?'"*. A revisão apontou, e concordo, que
**essa pergunta é boa e a resposta é uma frase que está no artigo**. O que é mau é uma plateia de
gestão de mobilidade sair sem nenhuma ideia do que um consumidor destas predições recebe. Ao montar
o S9 apareceram **três factos operacionais que estão no artigo e não estavam em slide nenhum** — o
horizonte (mediana 0,4 h a 5,5 h; 5–27 % dos alvos além de 3 dias), a partição por utilizador (*"as
visitas de um utilizador de teste nunca aparecem no treino"*), e a leitura de shortlist. O primeiro
é literalmente *a* pergunta desta sala: **com quanta antecedência me posso preparar.**

### DA-5 · Quem apresenta, e se há ensaio cronometrado
Não é decisão técnica, mas condiciona o plano. A v2 deixa a trilha em **~14:50**, com margem
confortável — já não é preciso um ensaio para caber. **Mas continua a ser preciso um ensaio para o
S2**, que é o único slide onde o guião é quase palavra a palavra e onde 130 s mal ditos custam a
sala inteira. Se só houver tempo para ensaiar uma coisa, ensaiar o S2 e a recusa dos quilómetros.

---

## 8 · Revisão Fable — o que apontou, o que aceitei, o que recusei

Corrida sobre a v1. **Reproduzi cada afirmação verificável contra o artigo antes de aceitar**, porque
um relatório de um par é uma afirmação de segunda mão e já me entrou uma correcção errada assinada
por dois. Resultado da reprodução: **todas as afirmações que verifiquei estavam certas**, e numa
delas **o errado era eu**.

### Aceite integralmente

**F1 · O Resultado 2 liderava com a alegação que o artigo desautoriza.** Na v1, o primeiro e mais
longo slide do Resultado 2 (105 s) tinha como mensagem *"o conjunto está acima de todas as
externas"* — e o meu próprio aviso, no mesmo slide, dizia que isso não é propriedade do conjunto e
não é prova sobre partilha. Vender, tabelar e desdizer, em 270 s e três slides, sem nunca dizer a
frase verdadeira de uma só vez. **Aceite:** S7 passa a ser o veredito, o S8 as externas em 45 s e
despromovidas, e o slide autónomo da "leitura honesta" desapareceu — as suas balas passaram a ser
ditas junto dos números, que é onde o artigo as põe. E entrou a frase que é a tese e que a v1 não
tinha em lado nenhum: **`07_discussion.tex:26` — "the trade is a measured one and not a free
substitution."** *(Reproduzido: a frase existe, textual.)*

**F2 · O S2 abria a reivindicar handover, que o artigo põe fora de âmbito.** *(Reproduzido:
`04_method.tex:38-41` — "A census tract is a neighborhood, not a radio cell… Cell association and
handover are radio-level decisions and remain out of scope."* Numa sessão chamada "Mobility
Management", abrir a reivindicá-lo convida à pergunta cuja resposta é "nada, está fora de âmbito".
**Aceite:** handover passa a analogia de escala, como o artigo faz.

**F3 · "Região" usava-se do princípio ao fim e só estava definida num slide de backup.** Erro meu e
óbvio depois de apontado. **Aceite:** definida no S2, com as contagens (520 → 8 501) repetidas no S6.

**F4 · O S2 não era bom o suficiente para as 8:30 de segunda** — 125 s de prosa sem imagem, sem
número, a fechar em vocabulário de MTL, e com a resposta guardada para o minuto 11. **Aceite,
incluindo o guião proposto**, que reescrevi ligeiramente e verifiquei facto a facto.

**F5 · A pergunta mais perigosa da sala não tinha resposta.** Esta é a melhor apanha da revisão. O
artigo mede um **controlo de concatenação** — vector de lugar + as mesmas features cruas por visita —
que ganha **+2,0 / +1,7 / +0,8** em AL/AZ/FL contra os nossos **+1,62 / +2,58 / +0,23**. **Em Alabama
e Florida o atalho barato ganha mais do que o grafo**, e o artigo escreve-o: *"Most of the category
difference is therefore already available in the raw per-visit features."* Isto não estava na trilha
nem em backup nenhum. **Aceite:** entra como caveat falado no S4 e como **B9**. *(Reproduzido em
`06_results.tex:63-72`.)*

**F6 · Faltava o que um serviço receberia.** Ver DA-4 — mudei de recomendação. **Aceite:** novo S9.

**F7 · Oito perguntas desta sala sem resposta preparada** — latência/energia, deriva e re-treino,
época dos dados, privacidade, os quilómetros, a margem mais estreita sobre o Markov, "só quatro
corridas?", e quem consome isto. **Aceite:** entraram todas na tabela de perguntas sem slide, na §5.

**F8 · Cortes de arco** — trabalhos relacionados fora, arquitectura de 170 s para 50, protocolo de
quatro frames para um, custo e limites fundidos, IC fora do ecrã. **Aceite.**

### Aceite com emenda

**F9 · "A cost bounded on both axes" não é da conclusão.** Correcto — mas também não é inexistente:
está no **resumo**, `src/main.tex:115`. A revisão diz que a conclusão não a tem, e tem razão; eu
tinha-a posto no slide de conclusão como se fosse de lá. **Emenda:** o S11 usa as palavras da
conclusão, e o plano regista de onde a frase é, para quem a for procurar na §VIII não concluir que
inventámos.

### Onde a revisão estava certa e eu estava errado

**F10.** Ao verificar, afirmei que os números do horizonte (0,4 h a 5,5 h; 5–27 % além de 3 dias)
**não estavam no artigo** — o meu `grep` devolvia três linhas e nenhuma os trazia. **Estavam.** A
frase atravessa mudanças de linha num parágrafo LaTeX e os meus padrões não a apanharam; só apareceu
quando **li** o parágrafo em vez de o procurar. É a mesma família do truncamento lido como ausência,
e é a segunda vez nesta linhagem de sessões. Se eu tivesse escrito a recusa em vez de ler, teria
mandado a revisão embora com um facto verdadeiro marcado como falso.

### Não aceite

Nada. Não houve ponto da revisão que eu tenha recusado.

---

## 8b · Segunda passagem — as cinco apanhas do `mobiwac-ppt` (v2.1)

Ao montar o deck a partir do mapa v2, a sessão `mobiwac-ppt` devolveu cinco defeitos. **Reproduzi
os cinco. Os cinco estavam certos**, e o primeiro é um erro factual que eu próprio introduzi.

**F11 · O guião do S8 era falso para o modelo dedicado. — CORRIGIDO**
Escrevi *"Both our models are above every system we re-ran, **by at least three points**."* Para o
conjunto é verdade (≥ 3,06 categoria, ≥ 3,55 região). Para o **dedicado** não: em Florida, na
categoria, 37,35 − 34,49 = **2,86**. O `CAMERA_READY §3.4` diz exactamente isto, e **o aviso estava
no meu próprio plano, no mesmo slide** — eu tinha-o escrito na v1 e mantive-o na v2.

**A causa é a que interessa, e é minha.** Esta frase veio da revisão Fable (F1). Verifiquei as
*outras* afirmações do relatório contra o artigo, uma a uma, e esta aceitei-a pela aritmética,
porque "pelo menos três pontos" soa a arredondamento seguro de 3,06. Não é: aplica-se a margem de um
modelo ao outro. **Reproduzir um relatório de um par não é reproduzir as frases que dele se copiam
para o produto final** — e foi precisamente o que eu escrevi na §8 que tinha feito.

**F12 · O Δ da tabela do S7 não sobrevive à subtracção das colunas. — ACEITE, nota no ecrã**
Cinco células discordam na segunda decimal, porque o Δ é a média pareada e as colunas arredondam
independentemente. Está no `CAMERA_READY §7` item 6, que diz que subtrair as colunas é a primeira
coisa que um revisor faz. Aprovei a nota que o par pôs no slide e registei as cinco células no S7.
*(Nota de rigor ao par: a referência é o §7 item 6; não existe um "§7.6" numerado. O conteúdo que
citaste está certo e as cinco células que nomeaste batem com a lista do documento.)*

**F13 · Nenhum slide definia Acc@10, e o S7 lê Acc@10. — ACEITE**
Buraco meu. Numa sala que não usa a métrica, uma tabela de Acc@10 sem a definição é ruído. Entra
como quinta linha do S6: *"a região verdadeira está nas dez primeiras?"*

**F14 · O guião do S2 dizia "half an hour" e "a few thousand per state". — CORRIGIDO**
0,4 h são **24 minutos**, menos de meia hora; e "alguns milhares por estado" não cobre o Alabama, que
tem **1 109** regiões. Dois números meus, ditos por aproximação em vez de por leitura. Corrigidos
para *"under half an hour"* e *"a thousand in Alabama, eight and a half thousand in California"*.

**F15 · A DA-1 continuava a recomendar 18 min e a dizer que "a §4 soma 18:00". — CORRIGIDO**
A §4 da v2 soma **14:50**. Reescrevi a trilha e deixei a decisão para trás a apontar para um número
que já não existia — **um ponteiro podre dentro do meu próprio documento**, da mesma família que ando
a converter no resto do repositório, e escrito no mesmo dia. A DA-1 passa a recomendar ~15 min e diz
onde gastar se o autor quiser 18.

---

**O padrão das duas passagens, que vale mais do que qualquer um dos itens.** A revisão Fable apanhou
o que estava **errado na estrutura**; a montagem apanhou o que estava **errado nos detalhes**, e só
apareceu porque alguém teve de pôr cada frase num slide e cada número numa tabela. Nenhuma das duas
teria encontrado o que a outra encontrou. **Um plano só se prova quando alguém o tenta executar** —
e das cinco apanhas desta segunda passagem, três (F11, F14, F15) são coisas que eu escrevi e que
nenhuma releitura minha teria apanhado, porque eu leria o que quis dizer.

### O que a revisão confirmou sem alterações

A §2 (a lei, incluindo a proibição da métrica de quilómetros), a §3 inteira (as três divergências
deck ↔ artigo), a escolha da chapa no slide da representação, o caveat falado sobre Florida
(p = 0,067; desvio entre folds 0,42 > gap 0,23), os quatro limites, o slide de reprodutibilidade, e
os backups B2/B4/B6/B8. E verificou, célula a célula, que **todos os números que o plano cita estão
certos**.

---

## 9 · Proveniência

Tudo o que está neste plano foi verificado nesta sessão contra as fontes, não citado de memória:

- os números contra `CAMERA_READY.md §3` e a lista de proibidos contra o `§4`;
- a decisão D1 contra `CAMERA_READY.md §6 D1` (decidida 2026-09-06, encerrada);
- as divergências deck ↔ artigo por busca das frases corrigidas dentro do
  `articles/dissertacao/presentation/slides/main.tex` (primazia 0 · atribuição 0 · controlo de
  capacidade 0 · cobertura da procura 0 · prior de transição 0);
- os quatro limites contra o parágrafo *"Four limits qualify these results"* de
  `src/sections/07_discussion.tex`, comparados item a item com o frame "Limitations and trade-offs";
- a contagem de parâmetros contra `src/sections/04_method.tex`;
- o estado da métrica geográfica contra `src/sections/07_discussion.tex` (*"left to future work"*) e
  contra a proveniência v17 do cálculo.

Acrescentado na v2, tudo reproduzido contra o artigo antes de entrar:

- o âmbito de handover contra `04_method.tex:38-41`;
- o controlo de concatenação e os seis números contra `06_results.tex:63-72`;
- o horizonte (0,4–5,5 h; 5–27 % além de 3 dias) e *"the visits of a test user never appear in
  training"* contra `05_setup.tex`, **lidos e não procurados** — ver F10 na §8;
- *"the trade is a measured one and not a free substitution"* contra `07_discussion.tex:26`;
- *"a cost bounded on both axes"* localizado em `src/main.tex:115` (o resumo), não na conclusão.

**O que não verifiquei:** os factos do programa (dia, hora, ordem da sessão, regra dos 30 minutos)
vêm da leitura que a sessão `mobiwac-ppt` fez do site a 2026-09-30. Não os reproduzi. Se algum deles
mudar, muda o orçamento da §4 e mais nada.
