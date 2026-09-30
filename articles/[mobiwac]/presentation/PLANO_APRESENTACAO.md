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

## 4 · A trilha principal — 15 slides, 18:00

Convenção de cada entrada: **mensagem única** (a única coisa que o ouvinte leva), **mostra**,
**fonte**, **segundos**, **do deck**.

---

### S1 · Título — 20 s
- **Mensagem:** quem somos e o que vamos responder.
- **Mostra:** título completo do artigo, os três autores, NESPeD-LAB / Universidade Federal de Viçosa, MobiWac 2026.
- **Fonte:** `src/main.tex`.
- **Do deck:** **novo** (o deck da defesa tem capa de dissertação).

---

### S2 · Porque é que isto interessa a quem gere mobilidade — 125 s
- **Mensagem:** se um serviço souber para onde uma pessoa vai a seguir, pode preparar-se antes — e prever o POI exacto é difícil e mais do que um serviço precisa.
- **Mostra:** três frases, sem tabela. (a) check-ins registam como as pessoas se movem pela cidade; a mobilidade individual é altamente previsível em princípio; (b) um serviço que antecipa pode **pôr conteúdo em cache onde o utilizador vai** e **provisionar capacidade no destino antes da procura chegar** — a predição de handover já deixa serviços celulares adaptarem-se em antecipação, ao nível da rede; (c) duas perguntas grosseiras chegam: **que tipo de sítio** e **que região**.
- **Fonte:** `src/sections/01_introduction.tex`, parágrafos 1–2 (Gowalla, Song, Bastug, Vielhaus, Moura).
- **Segundos:** é o slide mais longo da trilha de propósito. É aqui que uma plateia de redes decide se vai ouvir os 16 minutos seguintes.
- **Do deck:** **novo.** O deck abre pelo diagnóstico do capítulo anterior da dissertação, que aqui não existe e não interessa.
- **Nota de entrega:** acaba com a pergunta que dá o resto da palestra — *"can one model answer both?"*, e a razão pela qual isso não é óbvio: em MTL os parâmetros partilhados podem convergir para um compromisso óptimo para nenhuma das tarefas, ajudando uma e prejudicando a outra. É o custo que vamos medir.

---

### S3 · Onde é que isto se encaixa — 40 s
- **Mensagem:** a literatura modela várias granularidades ao mesmo tempo, mas categoria e região aparecem como sinais auxiliares de uma tarefa principal de próximo-lugar; aqui são o alvo.
- **Mostra:** duas linhas. "Auxiliar de next-place: MCMG, HMT-GRN." / "Alvo de igual estatuto: pouco explorado."
- **Fonte:** `src/sections/02_related.tex` — usar a redacção publicada, *"fine-grained region as an end target of equal standing, rather than an auxiliary coarse grid cell, is underexplored"*. **Não dizer "first".**
- **Do deck:** **editar** — condensar os dois frames "Related work of this study, part 1/2 e 2/2" num só. A plateia não conhece esta literatura e não precisa do mapa completo; precisa de saber que a pergunta não está respondida.

---

### S4 · Check2HGI: um quarto nível abaixo do lugar — 85 s
- **Mensagem:** em vez de um vector fixo por lugar, cada check-in ganha o seu próprio vector.
- **Mostra:** a chapa `c2h_deep.pdf` (cidade → região → lugar → **check-in**).
- **Fonte:** `src/sections/04_method.tex` + `src/figs/fig1_dataflow_new.tex`; chapa em `articles/dissertacao/presentation/figures/plates/c2h_deep.pdf`.
- **Do deck:** **reaproveitar como está.** É o melhor slide do deck e a chapa é excelente.
- **Nota de entrega:** a frase que faz o trabalho é *"o mesmo café às terças de manhã e aos sábados à noite não é o mesmo sinal"*. Meia frase, concreta, e a plateia percebe sem GNN.

---

### S5 · O que cada visita contribui — 60 s
- **Mensagem:** o vector de cada visita carrega o seu contexto — a hora, os lugares à volta, as visitas recentes.
- **Mostra:** o frame do deck, encurtado.
- **Fonte:** `src/sections/04_method.tex`.
- **Do deck:** **editar** — cortar ~40 % do texto. Este frame na defesa respondia a uma banca; aqui é uma ponte de 60 segundos.
- **⚠ Manter:** a cláusula de que cada visita lê **só o seu próprio passado**. Não é detalhe — é o terceiro limite do artigo e é a âncora da resposta B4.

---

### S6 · Resultado 1: a representação — 105 s
- **Mensagem:** mudando **só a entrada**, a categoria melhora em todos os seis conjuntos e em todos os folds.
- **Mostra:** a tabela de 6 linhas: check-in vs lugar, Δ de **+0,23 a +6,29**.
- **Fonte:** `CAMERA_READY §3.3`; tabela do artigo `src/tables/tbl2_substrate.tex`.
- **Do deck:** **reaproveitar** (os números já batem célula a célula).
- **⚠ Dizer em voz alta, porque a tabela não o diz:** o ganho é **grande onde os dados são pequenos** (Istanbul +6,29) e **abaixo de um ponto nos três maiores**. Em Florida (+0,23) o desvio-padrão entre folds (0,42) é **maior do que o próprio gap**, e o teste pareado **não separa** as duas representações (p = 0,067). Dito isto em 15 segundos, ganha-se a plateia; escondido, perde-se na primeira pergunta.

---

### S7 · A arquitectura: partilha por troca — 85 s
- **Mensagem:** um tronco partilhado onde as duas tarefas trocam contexto semântico, e uma via espacial privada para a região.
- **Mostra:** `fig2_model_slides.pdf`.
- **Fonte:** `src/figs/fig2_model.tex`; `src/sections/04_method.tex`.
- **Do deck:** **editar** — fundir os dois frames "sharing by exchange (1/2)" e "(2/2)" num só. Na defesa eram dois porque a banca ia perguntar; aqui a plateia quer a ideia, não a implementação.

---

### S8 · O protocolo, num slide — 70 s
- **Mensagem:** o que foi comparado, com o quê, e como se decidiu.
- **Mostra:** quatro linhas — unidade de dados (janelas de 9 visitas + alvo) · o que se mede (macro-F1 na categoria, Acc@10 na região) · o que se compara (conjunto vs dois dedicados, mesmos folds, mesmas janelas) · como se decide (4 sementes × 5 folds, teste pareado, correcção de Holm, margem de 2 pp **registada antes de ler qualquer resultado**).
- **Fonte:** `src/sections/05_setup.tex`; `CAMERA_READY §3` (cabeçalho da convenção).
- **Do deck:** **editar agressivamente** — os **quatro** frames "The protocol, in four steps" colapsam num. Na defesa o protocolo era o produto; aqui é infra-estrutura da credibilidade e vale 70 segundos.
- **⚠ A frase que não se corta:** *"a margem foi registada antes de qualquer resultado ser lido"*. É a única coisa neste slide que distingue este trabalho de um que afinou até dar bem.

---

### S9 · Resultado 2: um modelo, duas tarefas — 105 s
- **Mensagem:** o modelo conjunto fica acima de todas as referências externas nas duas tarefas e em todos os conjuntos.
- **Mostra:** a tabela dupla do deck — categoria (POI-RGNN · dedicado · conjunto) e região (HMT-GRN · ReHDM · STAN · dedicado · conjunto).
- **Fonte:** `CAMERA_READY §3.1, §3.2, §3.4`; tabela do artigo `src/tables/tbl3_results.tex`.
- **Do deck:** **reaproveitar.**
- **⚠ A ressalva obrigatória, e está no `CAMERA_READY §3.4`:** *"o modelo **dedicado** também está acima de todas as externas em todos os conjuntos"*. "Acima de todas as externas" **não é** uma propriedade do modelo conjunto — e a margem externa **inclui** a vantagem da representação, portanto **não é prova sobre MTL**. Se este slide for apresentado como vitória do conjunto, é um erro factual que o artigo não comete.

---

### S10 · O veredito, conjunto a conjunto — 95 s
- **Mensagem:** três células em doze são vitórias; as outras estão dentro de margens declaradas.
- **Mostra:** a tabela de deltas com IC a 90 % — região e categoria lado a lado, os três ▲ marcados (TX +1,21 · CA +1,06 · FL +0,19).
- **Fonte:** `CAMERA_READY §3.1 e §3.2`.
- **Do deck:** **reaproveitar** (bate célula a célula).

---

### S11 · A leitura honesta — 70 s
- **Mensagem:** o que estes números **não** dizem.
- **Mostra:** três balas.
  1. As quatro células "match" da região são **défices dentro da margem**, nunca empates — todos os intervalos ficam inteiramente abaixo de zero. **Nunca dizer "no difference".**
  2. O modelo conjunto **é também o modelo maior**. O desenho **não separa** tamanho de partilha, portanto as vitórias de região em TX e CA podem vir do tamanho.
  3. O emparelhamento das duas vitórias de região com os dois maiores vocabulários de região é uma **observação, não uma lei** — CA tem mais regiões do que TX e ganho menor.
- **Fonte:** `src/sections/07_discussion.tex`; `CAMERA_READY §3` (nota das três de doze) e §5 C4.
- **Do deck:** **novo.** O deck dispersa isto por rodapés; aqui merece um slide, e é o slide que protege o autor nas perguntas.
- **Nota:** é também aqui que entra, em meia frase, o *"nenhum serviço de mobilidade foi construído ou avaliado — isto é motivação, não um resultado de serviço medido"*, que é onde o artigo o põe (ver D-2).

---

### S12 · O que custa — 65 s
- **Mensagem:** o modelo conjunto é maior do que os dois dedicados somados; o que se ganha é operacional, não aritmético.
- **Mostra:** `AL 4.2 M vs 1.9 M` · `CA 5.2 M vs 2.8 M`; e a frase — um artefacto para treinar, versionar e implantar; uma passagem para a frente em vez de duas.
- **Fonte:** `src/sections/04_method.tex` — **usar o arredondamento do artigo** (ver D-1).
- **Do deck:** **editar** (corrigir `1.85 M` → `1.9 M`, e explicitar "vs os dois somados").

---

### S13 · Quatro limites — 65 s
- **Mensagem:** o que qualifica estes resultados.
- **Mostra:** os **quatro do artigo** (ver D-2): representação treinada uma vez sobre todos os lugares · a selecção de época consulta o fold onde o score é lido, portanto os absolutos são optimistas · cada nó de visita lê só as visitas que o precedem · o conjunto tem mais parâmetros do que os dois somados, portanto a região pode vir do tamanho.
- **Fonte:** `src/sections/07_discussion.tex`, o parágrafo "Four limits qualify these results".
- **Do deck:** **editar** — trocar o terceiro item do deck pelo limite de tamanho vs partilha.

---

### S14 · Conclusão — 60 s
- **Mensagem:** quando a representação preserva o contexto de cada visita e a arquitectura mantém uma via espacial privada onde as tarefas diferem, um modelo prevê o que e o onde numa só passagem, a um custo limitado nos dois eixos.
- **Mostra:** três linhas. A representação move mais o resultado do que a escolha entre um modelo e dois · um modelo lê as duas respostas de um só ficheiro guardado · isto **não** significa que MTL ajude automaticamente.
- **Fonte:** `src/sections/08_conclusion.tex` — quase textual.
- **Do deck:** **novo** (a conclusão do deck é da dissertação inteira, três estudos).

---

### S15 · Reprodutibilidade e agradecimentos — 30 s
- **Mensagem:** está tudo lá; e quem pagou.
- **Mostra:** `github.com/VitorHugoOli/PoiMtlNet/tree/mobiwac` (modelo, representação, baselines, testes estatísticos) · as duas fontes de dados públicas (dump do Gowalla anotado por categoria; Massive-STEPS) · **FAPEMIG, CNPq, CAPES**.
- **Fonte:** `src/main.tex` (nota de rodapé do código) e a secção `\section*{Acknowledgment}`.
- **Do deck:** **novo.** O deck da defesa não tinha nem o URL nem os financiadores — e o artigo tem os dois. Ver §6.

---

**Soma:** 20 + 125 + 40 + 85 + 60 + 105 + 85 + 70 + 105 + 95 + 70 + 65 + 65 + 60 + 30 = **1 080 s = 18:00**.

---

## 5 · Série B — backup para as perguntas

Slides que **não** entram na trilha. Ficam depois do último e chamam-se pelo número se a pergunta
vier. Regra: um slide, uma resposta, sem construção.

| # | Pergunta previsível | O que o slide mostra | Fonte |
|---|---|---|---|
| **B1** | *"Porquê região a este nível? Como é definida?"* | A definição de região usada e a contagem por conjunto: Istanbul 520 · AL 1 109 · AZ 1 547 · FL 4 703 · TX 6 553 · CA 8 501 | `CAMERA_READY §3.4`; `src/tables/tbl1_datasets.tex` |
| **B2** | *"De onde vêm os 2 pontos de margem? Escolheram depois de ver?"* | A margem foi **pré-registada, só para o eixo da região**, antes de qualquer leitura; a categoria **não tem** margem de equivalência registada — por isso as cinco células de categoria dizem "não resolvido" e nunca "equivalente" | `CAMERA_READY §3` (cabeçalho) |
| **B3** | *"Um Markov não faz isto?"* | O piso de Markov de primeira ordem (51,23–72,47) e a margem sobre ele (**+4,1 a +10,0**); e o facto incómodo de que **HMT-GRN fica abaixo desse piso nos seis**, STAN em quatro, ReHDM em três | `CAMERA_READY §3.4` |
| **B4** | *"Como é que sabem que o vector de uma visita não vê o alvo?"* | O grafo é **forward-only**: um nó de visita lê só as visitas que o precedem, no treino e na leitura, e o grafo não passa informação de uma visita posterior para uma anterior — é isso que impede um nó de carregar uma *feature* do alvo que ele prevê. **Está publicado, terceiro limite da §7.** | `src/sections/07_discussion.tex` |
| **B5** | *"Quanto custa um modelo em vez de dois?"* | 4,2 M vs 1,9 M (AL) e 5,2 M vs 2,8 M (CA) contra **os dois somados**; o ganho é operacional — um artefacto, uma passagem | `src/sections/04_method.tex` |
| **B6** | *"Se a representação faz quase tudo, para que serve o MTL?"* | A resposta honesta do artigo: na categoria a representação move mais o resultado do que a escolha entre um modelo e dois; se a troca entre as tarefas acrescenta algo **não é separado pela evidência aqui** | `src/sections/08_conclusion.tex` |
| **B7** | *"Porquê estas cinco referências externas?"* | A tabela completa das externas, com as duas notas de rodapé honestas: STAN com folds parciais (TX 4/5, CA 2/5) e ReHDM com uma só semente em TX e CA | `CAMERA_READY §3.4` |
| **B8** | *"O ganho de região cresce com o número de regiões?"* | **Não.** É uma observação, não uma lei: CA tem mais regiões do que TX e ganho **menor** (+1,06 vs +1,21) | `CAMERA_READY §5 C4` |

**Perguntas sem slide, só com resposta preparada** (não valem um slide, mas o autor deve ter a frase):

- *"Os números absolutos parecem baixos"* — são macro-F1 sobre um problema de muitas classes com
  desequilíbrio forte; e a selecção de época consulta o fold onde o score é lido, portanto **os
  absolutos são optimistas e está declarado**. A comparação conjunto vs dedicado é afectada muito
  menos: a regra de selecção é a mesma para os dois, nos mesmos folds.
- *"Isto funciona noutra cidade?"* — Istanbul foi escolhido precisamente por diferir dos cinco
  estados dos EUA, e é onde o ganho de representação é **maior** (+6,29). Generalização para além
  destes seis não foi testada.
- *"Porque é que o dedicado de categoria recebeu uma procura mais larga?"* — recebeu, está declarado
  no segundo limite, e isso torna a diferença de categoria **conservadora**, não o contrário.

---

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
- **(a) 18 min de fala, 12 de margem e perguntas** — ⭐ **recomendo.** Somos os primeiros do
  simpósio: o chair calibra o relógio da sessão em cima de nós, e um estouro arrasta os dois
  seguintes. A §4 soma exactamente 18:00.
- (b) 20 + 10, como o `mobiwac-ppt` sugeriu — cabe, mas sem folga para uma pergunta longa no fim.
- (c) 22 + 8 — só se o chair confirmar no local que a sessão está adiantada.

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

### DA-4 · O esboço de serviço entra ou sai
O artigo tem um esboço de serviço de ~90 palavras na §7 (a leitura de shortlist).
- **(a) Entra, como meia frase no S9 e uma bala no S11** — ⭐ **recomendo.** É o que traduz o
  resultado para esta plateia, e a ressalva *"motivação, não um resultado de serviço medido"* vai
  colada.
- (b) Slide próprio de 60 s — rouba tempo à S2, que é mais importante, e um slide inteiro sobre um
  serviço não construído convida à pergunta "então construíram?".

### DA-5 · Quem apresenta, e se há ensaio cronometrado
Não é decisão técnica, mas condiciona o plano: 18 minutos só funcionam com **um** ensaio cronometrado.
Se não houver ensaio, recomendo cortar a S3 e a S5 (100 s) e apresentar 13 slides.

---

## 8 · Revisão Fable

> _A preencher quando a revisão correr. Registo: o que o Fable apontou, o que aceitei, o que recusei
> e porquê._

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

**O que não verifiquei:** os factos do programa (dia, hora, ordem da sessão, regra dos 30 minutos)
vêm da leitura que a sessão `mobiwac-ppt` fez do site a 2026-09-30. Não os reproduzi. Se algum deles
mudar, muda o orçamento da §4 e mais nada.
