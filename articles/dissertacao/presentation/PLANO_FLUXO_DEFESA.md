# PLANO_FLUXO_DEFESA.md — a arquitetura da apresentação

> **O que este documento é.** O **fluxo** da apresentação de defesa: o arco, as seções e
> subseções, a ordem, o orçamento de tempo, e as regras que o documento seguinte — o
> slide-a-slide — terá de obedecer. Escrito 2026-08-21, sete dias antes da defesa.
>
> **O que este documento NÃO é.** Não é o conteúdo dos slides. Nenhuma linha aqui diz o que
> vai escrito em qual slide. Isso é o `SLIDES.md`, que só pode ser escrito depois que este
> plano for validado com o orientador (§8).
>
> **Base.** [`APRESENTACAO_DEFESA_GUIDE.md`](APRESENTACAO_DEFESA_GUIDE.md) (logística, Art. 23,
> o exemplo real da defesa do Henrique, o orçamento-rascunho §4.1) + o texto entregue em
> [`../src/`](../src/) + o dossiê de arguição em [`../wrapup/`](../wrapup/).
> Onde este plano diverge do guia, a divergência está declarada em §7.
>
> **Como foi produzido.** Cinco leituras independentes do texto entregue, duas do dossiê de
> defesa, um desenho de arco, e três passagens adversariais que derrubaram quatro frases do
> primeiro rascunho. O registro dessas quatro correções está em §7 — elas são a razão pela
> qual este plano diz *menos* que o rascunho e é mais forte por isso.

---

## 0 · O que já está decidido

| Decisão                           | Ruling                                                                                                                                                                                                          | Quando            |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| **Veredito cedo**                 | O resultado aparece na abertura, em linguagem de lei, nos primeiros três minutos. Cada slide seguinte é resposta a uma pergunta já feita.                                                                       | Autor, 2026-08-21 |
| **Idioma**                        | **Fala em português, slides em inglês.** Os slides espelham o texto entregue, então nenhum número ou notação diverge entre tela e documento.                                                                    | Autor, 2026-08-21 |
| **Pós-submissão**                 | P1 (capacidade) e mtlcheck ficam **só na trilha de reserva**. A trilha principal é redigida para *sobreviver* a eles.                                                                                           | Autor, 2026-08-21 |
| **A posição sobre o tronco**      | O autor sustenta que **não há dados suficientes para provar que o tronco compartilhado não contribuiu**. Ver §5.3 — é defensável e já está no texto entregue.                                                   | Autor, 2026-08-21 |
| **Defeito do Resumo**             | Corrigido no fonte para a versão final + errata registrada + **nunca dito em voz alta**, exceto se perguntado. Ver §6, família 6.                                                                               | Autor, 2026-08-21 |
| **Template e barra de navegação** | **`slides/`** (Beamer, do NESPeD — autoria de Henrique S. Santana, a mesma defesa que o guia §4.0 analisa). A barra de navegação **é nativa do template**, não precisa ser construída. Ver §11. | Autor, 2026-08-21 |
| **Regra de ordenação** | §2 só introduz o que **≥2 dos 3 artigos usam na mesma forma**; o resto entra na seção do artigo que usa. Resolve o ponto 8. Ver §4. | Autor, 2026-08-21 |
| **Nomes das seções 3–5** | O **título do artigo**, não o veículo. Barra: a **linhagem de modelos** — MTLnet · ST-MTLNet · Check2HGI. | Autor, 2026-08-21 |
| **Protocolo estatístico** | Introduzido em **5.4**, não na Seção 2 — só o Cap. 5 o usa. Idem Acc@10 e joint-best. | Autor, 2026-08-21 |
| **Karpathy** | **Não na conclusão.** Série B, como contexto que abre a oferta proativa do P1. Ver §6. | Autor, 2026-08-21 |
| Logística                         | 28/08/2026, 10:00–12:30, **remota** (Google Meet). Banca: Fabrício A. Silva (presidente), Clayson S. F. de Sousa Celes (ITA, externo), Alex Borges.                                                             | Guia §0           |
| Teto                              | **50 minutos** (Art. 23). Este plano orça **45 min**, folga de 5.                                                                                                                                               | Regimento         |

---

## 1 · A espinha

> **Uma frase, a que a banca deve conseguir repetir de volta:**
>
> *MTL não ajuda predição de POI automaticamente. Um nulo previsto, sob embedding por lugar e
> compartilhamento rígido, foi diagnosticado — com a arquitetura congelada — como um problema de
> representação, e resolvido por uma representação em nível de check-in com atenção cruzada, sob
> um protocolo sem vazamento e disjunto por usuário. A contribuição são as **condições
> identificadas**, não um "sim" universal.*

Essa espinha **não foi inventada para a apresentação**. É a do próprio documento (`1_introduction.tex` §1.2):

> *"O primeiro reporta um resultado negativo, o segundo identifica seu gargalo principal, e o
> terceiro testa a solução resultante. Essa progressão é ela própria parte da contribuição,
> porque cada estudo estreita a explicação sustentada pela evidência."*

Isso importa porque é exatamente o que a pesquisa de banca (`../docs/research/banca_evaluation_research_2026-07-20.md`)
identifica como a diferença entre uma coletânea que funciona e uma que falha: §6 diz que "capítulos de moldura que
apenas resumem os artigos" é o principal modo de errar; §7 diz que "uma narrativa de evolução honesta transforma a
fraqueza do formato em evidência de processo científico".

---

## 2 · Os quatro atos, e as transições

As transições são o entregável desta seção — não os atos. Elas são o fio condutor tornado audível, e **passagens de
corte de tempo não podem removê-las**.

### ATO I — "A pergunta e o chão comum" (min 0–9, seções 1–2)

Abre na regularidade da mobilidade e nas aplicações, enuncia a pergunta de pesquisa literalmente, fixa o escopo
(Definições 2.7/2.8 dentro, 2.9 fora), dá o veredito em uma linha, e então **paga a dívida de duplicação de uma só
vez**.

> **Transição de saída (dita, em slide próprio):**
> *"Com o vocabulário, os dados e a métrica fixados uma única vez, cada estudo agora só precisa
> dizer o que mudou — e cada um nomeia a sua própria convenção de avaliação quando chegar a hora.
> O primeiro usou o que a literatura oferecia: um vetor por lugar e um tronco compartilhado."*
>
> ⚠ **Reescrita 2026-08-21.** A versão anterior dizia *"as regras de decisão fixadas uma única
> vez"* — o que deixou de ser verdade quando o protocolo estatístico migrou para 5.4 sob a regra
> de §4. A frase prometia um fechamento que a Seção 2 não entrega mais.

### ATO II — "O nulo e o seu diagnóstico" (min 9–20, seções 3–4)

O Cap. 3 entrega o nulo e, o que importa mais, a **bifurcação de três suspeitos**. O Cap. 4 condena um deles por
experimento controlado.

> **Transição interna (Cap. 3 → Cap. 4):**
> *"Um nulo com três suspeitos não encerra a investigação — desenha o próximo experimento:
> congelar a arquitetura e mover apenas a entrada."*
>
> **Transição de saída (Ato II → III):**
> *"Com a arquitetura fixa, a entrada moveu o resultado: a representação é o gargalo. Mas o
> diagnóstico ainda é em nível de lugar, sob um protocolo que deixa o mesmo usuário dos dois
> lados da divisão. O terceiro estudo reconstrói as três camadas — representação, topologia e
> protocolo — e, na verificação final, encontramos e fechamos um vazamento no próprio grafo."*

**O vazamento é o pivô, e é dito em voz alta.** Enquadramento: *a verificação funcionando*, não uma confissão. Ele é
mencionado aqui e retomado uma vez dentro de 5.2 — nunca escondido, nunca repetido uma terceira vez.

### ATO III — "A resolução" (min 20–40, seção 5)

Representação → topologia → protocolo → veredito → o trade medido, **nessa ordem**, para que todo resultado caia sobre
regras já enunciadas — as **superfícies** no Ato I, a **maquinaria** em 5.4, minutos antes do
veredito que a usa.

> **Transição de saída:**
> *"Um veredito condicional, medido sob o protocolo mais estrito dos três. O que os três estudos,
> juntos, estabelecem — e o que não estabelecem?"*

### ATO IV — "A resposta condicional" (min 40–45, seção 6)

A resposta condicional, a contribuição una (não uma por artigo), as limitações oferecidas antes de perguntadas, e o
retorno às aplicações como fecho. **A fala termina onde começou, um nível de entendimento acima** — que é a coletânea
funcionando.

---

## 3 · O plano de seções

Seis seções nomeadas. **Os rótulos da barra de navegação são a linhagem de modelos** — a barra
narra a própria evolução que é o arco, e casa com a Tabela 1 da dissertação:

> **Introdução · Fundamentos · MTLnet · ST-MTLNet · Check2HGI · Conclusão**

As seções 3–5 levam o **título do artigo** como título de seção; o veículo aparece só como
proveniência no divisor.

**Minutos são alvos pontuais:** 4 + 5 + 5 + 6 + 20 + 5 = **45 min**, cinco abaixo do teto do
Art. 23. *(Revisão de 2026-08-21: eram 43, e uma auditoria de orçamento mostrou que os quatro cards
de mecanismo novos — DGI, NashMTL, HGI, encoders — custam 40–75 s cada e não cabiam nos +2 min que
as seções 3 e 4 tinham ganhado. Os 2 min entram onde os cards estão, mais 1 min na Seção 5, que
absorveu o protocolo estatístico e o related-work próprio.)*

### Seção 1 · Abertura — a pergunta e o escopo — **4 min**

| # | Subseção |
|---|---|
| 1.1 | Motivação em um fôlego: mobilidade é altamente regular (Song *et al.*, ~93%) — **com a ressalva de não-teto na mesma frase** — e as aplicações que dependem de categoria e região |
| 1.2 | A pergunta de pesquisa, literal (§1.2), **e o veredito em uma linha nas superfícies registradas** (§5.1) |
| 1.3 | Escopo positivo: Defs. 2.7/2.8 dentro; Def. 2.9 (próximo lugar) definida **para ser excluída**; sete categorias; região = setor censitário / *mahalle* |
| 1.4 | O arco em uma linha + roteiro: os três artigos pelos **títulos de capítulo**, com veículo/ano/autoria |

**Propósito.** Fixa pergunta, fronteira e destino em menos de cinco minutos.
**Se cortada:** a banca encontra o nulo do Cap. 3 sem saber que foi *previsto*.

> **Corolário do gloss.** O veredito do minuto 3 usa as **superfícies registradas** — *"permanece
> dentro da margem de dois pontos, registrada antes de qualquer resultado ser lido"* e *"equivalente
> a zero dentro de meio ponto"* — que o GLOSSARY já autoriza em linguagem legível. **Um gloss não é
> um `INTRODUZ`** e não licencia reuso: a maquinaria (TOST, Holm, *t* pareado) só entra em 5.4.

**Nunca dizer.** "Pioneiro"/"o primeiro". Nenhum número além do veredito. Nunca "prediz o próximo POI".

### Seção 2 · Fundamentos compartilhados — dito uma vez — **5 min**

| # | Subseção |
|---|---|
| 2.1 | A linhagem em **uma** tabela (Tabela 1) + **a ideia infomax numa frase**: *distinguir um pareamento verdadeiro de um corrompido* — cobre DGI, HGI e Check2HGI de uma vez. Mais o diagrama de níveis (lugar → visita) |
| 2.2 | Vocabulário de MTL: compartilhamento rígido (Def. 2.10), transferência negativa (Def. 2.12) como o **risco nomeado**, e o critério **declarado no Cap. 2** (*um balanceador só é útil se superar uma ponderação fixa ajustada*) |
| 2.3 | A base de evidência: **uma** tabela de seis conjuntos, dizendo qual capítulo usou quais. **Florida é dita aqui, uma vez**: 990.518 check-ins nos Caps. 3/4 e 1.407.034 no Cap. 5 — *duas extrações*, nunca "superconjunto" (não há evidência de contenção) |
| 2.4 | **A métrica que os três compartilham**: macro-F1 e por quê (Food ≈ um terço; a perda **não** é reponderada) + o piso de classe majoritária |
| 2.5 | **O protocolo dos dois primeiros estudos** + as regras de leitura: estratificado por amostra (o mesmo usuário dos dois lados), leitura *diagnostic-best*, 5 folds; **a lei dos verbos**; e o anúncio de que *cada estudo nomeia a sua convenção* |
| 2.6 | Trabalhos relacionados em **DOIS slides**: (a) o eixo que diferencia — categoria/região como **meio** para o próximo lugar × como **fim**; (b) o **mapa de onde saem os baselines** (cada bloco apresenta os seus) |

**Propósito.** O motor de de-duplicação. **Se cortada:** cada bloco repaga preâmbulo e a fala estoura.

> **Encolheu de 6 para 5 min** sob a regra de §4. **Saíram para a Seção 5**: Acc@10 + desconto OOD,
> o escore de seleção conjunta, sementes/20 modelos/*t* pareado/TOST/Holm, e a profundidade do split
> disjunto. **Entrou**: a frase infomax e o diagrama de níveis em 2.1, e o segundo slide de
> trabalhos relacionados. O tempo liberado financia os cards das seções 3 e 4.

**Nunca dizer.** Nenhum resultado, nenhum número de capítulo. Nenhuma afirmação de Pareto. Nunca
chamar as médias por categoria dos Caps. 3/4 de "macro-F1". **Nunca Space2Vec nem POI2Vec como
componentes** — não estão no GLOSSARY e não são deste trabalho (POI2Vec é arte prévia do Cap. 4).

### Seção 3 · *Multitask Learning for POI Category and Next-POI Prediction* — **5 min**
*(barra: **MTLnet** · proveniência no divisor: CBIC 2025, DOI 10.21528/CBIC2025-1191324)*

| # | Subseção |
|---|---|
| 3.1 | O par: classificação estática (Def. 2.6) + próxima categoria — um estático, um sequencial |
| 3.2 | **MTLnet** (Figura 1, **redesenhada em Beamer** — o raster publicado imprime pequeno demais): encoders por tarefa → FiLM (uma cláusula) → tronco residual → duas cabeças |
| **3.2A** | **Card DGI — "Como funciona \| Por quê"** (ponto 4): retoma o infomax de 2.1; **grafo de Delaunay sobre os POIs, arestas com decaimento logarítmico da distância**; um vetor por lugar. ⚠ **Não dizer "one-hot da própria categoria":** a nota de rodapé do capítulo entregue registra que a implementação liberada alimenta **a média dos one-hots dos vizinhos, com o vetor do próprio POI excluído**, e essa distinção é a defesa de que a tarefa estática lê **homofilia espacial**, não o próprio rótulo. Ou o card diz isso, ou omite o atributo de nó e deixa para o B4. **Sem "coocorrência"** — o Cap. 3 não tem esse canal |
| 3.3 | Setup em três linhas: Florida, sete categorias, 5 folds — **e a autodeclaração de protocolo dita aqui** (retoma 2.5) |
| **3.3A** | **Card NashMTL** (ponto 4): retoma o critério de 2.2; intuição de barganha — evitar que uma tarefa domine; e que a adoção é **conclusão do seu tempo**, não defendida hoje |
| 3.4 | O resultado nas palavras do capítulo: *"largamente comparáveis, sem vantagem clara ou consistente"* + **as lideranças divididas**, mostradas como o próprio deck do CBIC as mostrou |
| 3.5 | A bifurcação de três hipóteses: dissimilaridade / insuficiência de representação / rigidez de topologia |

**Propósito.** Estabelece que o nulo foi **previsto, publicado e produtivo** — ele fabrica os três
suspeitos. **Se cortada:** o Cap. 4 não tem pergunta a responder.

> **FLOPs/convergência (sua dúvida no ponto 4): fora da trilha principal, vai para a série B.**
> A tabela reproduz (80,88 s contra 34,97 s cumulativos = **2,3×**), mas a *prosa publicada* do CBIC
> diz "quase quatro vezes" e "cerca do dobro" em MFLOPs — e a `ERRATA.md` do próprio artigo registra
> as duas como defeituosas. Mostrar a tabela custa ~40 s e convida a pergunta de custo antes de a
> história ter chegado ao ponto. **Slide B pronto; nunca a prosa.**

**Nunca dizer.** Nunca um número do Cap. 3 ao lado de um do Cap. 5. Transferência negativa foi
**hipotetizada, nunca observada**. E **nunca "ambas as baselines externas batidas em absoluto"** —
vale só na tarefa estática; na sequencial MHA+PE lidera **Community, Food e Shopping**.

### Seção 4 · *ST-MTLNet: Spatio-Temporal POI Representations* — **6 min**
*(barra: **ST-MTLNet** · proveniência: CoUrb 2026/SBRC, DOI 10.5753/courb.2026.22960)*

| # | Subseção |
|---|---|
| 4.1 | A pergunta herdada — *"arquitetura ou representação?"* — e o desenho: **MTLnet congelado**, só a entrada muda (retoma 3.2) |
| **4.1A** | **Card HGI** (ponto 5, o conceito que você chamou de importância extrema): o mecanismo em palavras simples — encoder de categoria pré-treinado → uma camada de convolução sobre o grafo de POIs → atenção por região → embedding de cidade ponderado por área; discriminador bilinear, **sem rótulos da tarefa final**. Retoma o infomax de 2.1 |
| **4.1B** | **Card "por que estes encoders"** (ponto 5): a decomposição em **três canais de 64 dims** — espacial (**SIREN × Sphere2Vec-M**, comparados), temporal (**Time2Vec**), categórico (**duas fases: um POI Encoder com caminhadas aleatórias + o HGI**). Cada um com a razão da escolha |
| 4.2 | **Ressalva primeiro, número depois**: a tarefa estática lê o próprio rótulo (tipo de local ↔ 1:1 nas sete categorias) → o ganho **não diz nada sobre a tarefa sequencial** |
| 4.3 | O resultado diagnóstico: a tarefa **sequencial**, cujo alvo nunca está na entrada |
| 4.4 | Bordas honestas, com **rótulo de tarefa explícito**: *"Travel (categoria) ✓ × Travel (próxima categoria) ✗"* — com a razão do próprio capítulo (topologia de grafo ainda vence em movimento esparso de longa distância). E a comparação **não pareada em largura** (192 × 64 dims) |
| 4.5 | A frase de entrega, literal: arquitetura fixa, entrada mudou, resultado moveu → **a representação é o gargalo** |

**Propósito.** Remove um suspeito por experimento controlado — o meio causal do arco.

> **O juiz do CoUrb já atacou este material.** Duas instruções dele valem como lei aqui: (a) rotular
> Travel por tarefa, porque ele **ganha em categoria e perde na sequencial** e a sala se confunde;
> (b) o intervalo de ganho é **melhor-de-dois por linha** e isso tem de ser declarado — SIREN
> sozinho no Texas dá +17,89, fora da faixa anunciada. Nunca apresentar a faixa sem dizer que é
> melhor-de-dois.

**Nunca dizer.** Número antes da ressalva. "macro-F1 subiu 20–22" (são médias por categoria).
"Pareado em largura". Deixar o ganho estático falar pela sequencial. Ampliar crédito de autoria.

### Seção 5 · *A Check-in-Level Multitask Study of Next Category and Region* — **20 min**
*(barra: **Check2HGI** · proveniência: MobiWac 2026, submetido)*

| # | Subseção | min |
|---|---|---:|
| 5.1 | O que muda, **como consequências do diagnóstico**: representação (lugar → check-in), topologia (rígida → atenção cruzada), protocolo (por amostra → disjunto por usuário) | 1 |
| 5.2 | **Check2HGI** (ponto 6): Figura 4; o diagrama de níveis de 2.1 **reusado** — *"um quarto nível abaixo do lugar"*; **como ele se apoia no Cap. 4** (o HGI de 4.1A estendido um nível); a aresta **só para frente**; Figura 6 (separabilidade); Tabela 9 | 4,5 |
| **5.2A** | **Trabalhos relacionados deste estudo** (ponto 6): por que embedding em nível de check-in é **novo na linha**; CTLE como a arte prévia mais próxima | 0,75 |
| 5.3 | A arquitetura (ponto 6 — **o que mudou no MTL e por quê**): Figura 5; compartilhamento **por troca**, não por camadas possuídas; o caminho espacial privado da região | 3 |
| **5.4** | **Protocolo e metodologia estatística, introduzidos AQUI** (pontos 6 e 8): CV 5-fold **disjunta por usuário**; **Acc@10 + desconto OOD + piso Markov-1**; janelas **sobrepostas, stride 1**; **joint-best**; sementes {0,1,7,100} = **20 modelos ajustados, unidade inferencial n = 4**; *t* pareado sobre as médias por semente; **TOST na margem registrada**; Holm. **E o desvio declarado** do plano registrado (Wilcoxon sobre folds → *t* pareado sobre médias por semente) | 4,25 |
| 5.5 | **O veredito**: Tabela 10 + Figura 7, em linguagem de lei (§5.1) | 5 |
| 5.6 | O trade medido e os quatro limites declarados | 1,5 |

> ### A escada de 5.4 — quatro degraus, um slide cada
>
> Seu ponto 7 diz que o método estatístico do Cap. 5 está "confuso e jogado". **A correção não é
> falar dele em outro lugar; é dar-lhe ordem.** Uma lista de nove itens separados por vírgula em
> 4,25 min reproduziria dentro da fala exatamente a queixa que você fez do capítulo. Os degraus:
>
> | # | Degrau | O que fixa |
> |---|---|---|
> | 1 | **A unidade de dados** | CV 5-fold **disjunta por usuário** (a reparação da falha declarada em 3.3) + janelas **sobrepostas, stride 1** |
> | 2 | **O que se mede** | **Acc@10** + desconto OOD + os pisos (Markov-1) |
> | 3 | **O que se compara** | **joint-best** como convenção de checkpoint; **20 modelos ajustados**; **unidade inferencial n = 4** (as médias por semente) |
> | 4 | **Como se decide** | *t* pareado sobre as quatro médias; **TOST na margem registrada**; **Holm** dentro da família de tarefa; **e o desvio declarado** (o plano registrava Wilcoxon sobre folds) |
>
> Cada degrau responde uma pergunta que o anterior deixa em aberto. É essa ordem que o `SLIDES.md`
> herda — **não a lista**.

**Propósito.** O resultado que a dissertação defende, sobre regras que a banca aceitou minutos antes.
**Se comprimida abaixo de ~18 min:** o veredito chega sem proveniência e vira alegação.

**Nunca dizer.** "Empata"/"matches"/"ties"/"em todos". Aplicar a margem de dois pontos à categoria.
Chamar as quatro células dentro da margem de empates. Creditar TX/CA a transferência entre tarefas
(§5.3). *"Sob um décimo do salto place→check-in"* (**aritmeticamente falso** contra a Tabela 9 na
mesma página). A frase retratada sobre representação hierárquica × injeção de features.

### Seção 6 · Conclusão Geral — a resposta condicional — **5 min**

| # | Subseção |
|---|---|
| 6.1 | A resposta condicional (§6.2): MTL ajudou **sob este desenho e este protocolo** — e o que isso não autoriza |
| 6.2 | A contribuição una (§6.5), **redação idêntica à do slide de abertura**: metade prática + metade científica |
| 6.3 | **O centro de gravidade da seção** (seu ponto 7): as seis limitações do §6.3 amarradas **1:1** aos seis trabalhos futuros do §6.4 — cada uma falada como um próximo passo concreto e condicional |
| 6.4 | Fecho: retomada das aplicações + agradecimentos + a linha do repositório. **O takeaway na tela; o "obrigado" pela voz** (instrução do orientador no deck do CoUrb) |

**Nunca dizer.** "MTL funciona" sem condição. Re-caminhar a cadeia dos três estudos. Nenhum número novo.

> **A outra metade do seu ponto 7** — o método estatístico "confuso e jogado" no Cap. 5 — **não se
> resolve aqui**. Resolve-se estruturalmente em 5.4, que passa a apresentá-lo como uma escada única
> em vez de espalhado; e a *melhoria* de método vive na série B, no slide de nomenclatura (§6, B3),
> onde ela é o que é: trabalho posterior que **nomeou um confundimento**, não uma correção dos
> números entregues.

---

## 4 · O ledger de de-duplicação

**Sua prioridade declarada**, agora governada por uma regra em vez de caso a caso.

> ### A regra
>
> **1 · Posse.** Um elemento é introduzido no seu **ponto de primeiro uso na LINHAGEM** — porque a
> linhagem *é* o arco, e a herança é o que a fala narra. A **Seção 2** recebe apenas o que é
> **transversal E não faz parte da herança que o próprio arco conta**. Usado por dois artigos **em
> formas diferentes** → cada forma entra onde é usada, e a Seção 2 fica com a **frase de contraste**.
>
> **Exceção declarada:** um dataset usado por um só estudo entra com esse estudo, **salvo quando a
> tabela de evidência é o contraste** — a dos seis conjuntos é mostrada uma vez, inteira, em 2.3.
>
> **2 · Profundidade.** Onde a ideia é compartilhada mas o **mecanismo** pertence a um artigo, a
> Seção 2 introduz só a profundidade compartilhada (uma frase, uma linha de tabela) e o artigo dono
> introduz o mecanismo.
>
> **3 · Gloss.** O veredito da abertura pode usar as **superfícies registradas** antes de a
> maquinaria existir. **Um gloss não é um `INTRODUZ`** e não licencia reuso.

**Por que "linhagem" e não "contagem de artigos".** A primeira redação desta regra dizia *"§2 só
introduz o que ≥2 dos 3 artigos usam"*, e eu afirmei que ela reproduzia sozinha os seus julgamentos.
**Ela não reproduz.** A revisão de fecho testou-a contra o próprio ledger e ela falha em quatro
linhas, todas na mesma direção: mandaria para a Seção 2 exatamente o que o arco precisa narrar como
**herança**.

| elemento | usado por | a contagem mandaria | o certo é | por quê |
|---|---|---|---|---|
| MTLnet + FiLM | Caps. 3 e 4, forma idêntica | Seção 2 | **3.2** | *"a mesma arquitetura, sem alterar uma linha"* **é o argumento de controle do Cap. 4** — dizer antes destrói a herança |
| Nash-MTL | Caps. 3 e 4 | Seção 2 | **3.3A** | idem; e o Cap. 4 o usa (`4_courb/methodology.tex:96`) |
| HGI | Caps. 4 e 5, forma *place-level* idêntica (a coluna place-level da Tabela 9 **é** o HGI) | Seção 2 | **4.1A** | o Check2HGI **estende** o HGI um nível abaixo; introduzi-lo cedo apaga a extensão |
| Istambul | só Cap. 5 | Seção 5 | **2.3** | a tabela de evidência é o contraste, e mostrá-la partida é pior |

A regra de linhagem acerta as quatro **e** continua acertando os seus três julgamentos.

| Elemento | INTRODUZ em | Retoma em | Forma da retomada |
|---|---|---|---|
| Gowalla + Istanbul, estatísticas | **2.3** | 3, 4, 5 | "nos três estados que este estudo usou" |
| As sete categorias | **2.3** | 3, 4, 5 | "as mesmas sete categorias" |
| macro-F1 + piso de classe majoritária | **2.4** | 3, 4, 5 | "a mesma macro-F1" — **a definição**; os valores impressos dos Caps. 3/4 seguem "F1 por categoria", com carimbo |
| MTL, compartilhamento rígido, transferência negativa | **2.2** | 3, 5 | "o risco nomeado na Seção 2" |
| **A ideia infomax** (o objetivo comum a DGI/HGI/Check2HGI) | **2.1** | 3.2A, 4.1A, 5.2 | "o mesmo objetivo da Seção 2, um nível abaixo" |
| Linhagem de modelos (Tabela 1) + diagrama de níveis | **2.1** | 3, 4, 5.2 | apontar para a Tabela 1 |
| Protocolo dos dois primeiros estudos + lei dos verbos | **2.5** | 3.3, 4 | "o protocolo que a Seção 2 chamou de estratificado por amostra" |
| Eixo que diferencia + mapa de baselines | **2.6** | 3.4, 5.2A | "o eixo da Seção 2" |
| Restrição de modelo único | **1.3** | 5, 6 | "a restrição de um artefato" |
| **DGI (mecanismo)** | **3.2A** | 4.1 | "a representação monolítica que o Cap. 4 decompõe" |
| **MTLnet + FiLM** | **3.2** | 4.1 | *"a mesma arquitetura, sem alterar uma linha"* |
| **NashMTL** | **3.3A** | 4.1, 5.4 | "o balanceador dos **dois primeiros** estudos" — o Cap. 4 também treina com ele |
| **HGI (mecanismo)** | **4.1A** | 5.2 | "a hierarquia que o Check2HGI estende um nível abaixo" |
| **SIREN / Sphere2Vec-M / Time2Vec / POI Encoder** | **4.1B** | — | não retomados |
| **Check2HGI** | **5.2** | 6 | "a representação por visita" |
| **Acc@10 + desconto OOD + piso Markov-1** | **5.4** | 5.5, 6 | — |
| **Joint-best, sementes, n=4, TOST, Holm** | **5.4** | 5.5 | "o protocolo que acabamos de fixar" |
| **Split disjunto por usuário** | **5.4** | 6 | "o protocolo mais estrito dos três" |
| A aresta só para frente / o vazamento | **transição ATO II→III** | 5.2 (uma vez) | nunca uma terceira vez |
| O ladder de veredito | **5.5** | 6.1 | "o veredito que acabamos de ver" |

> ⚠ **Três armadilhas de nomenclatura que a fala desarma na Seção 1, uma frase cada:**
>
> 1. **"Next-POI Prediction" nos Caps. 3 e 4 significa próxima *categoria*.** Toda arte reproduzida
>    leva a anotação `Next-POI Prediction = next category (Def. 2.7)`.
> 2. **O par de tarefas muda entre os capítulos** — Caps. 3–4: estática + próxima categoria;
>    Cap. 5: próxima categoria + próxima região.
> 3. **A convenção métrica muda**: Caps. 3/4 reportam **F1 média por categoria**; o Cap. 5 reporta
>    **macro-F1**. Os valores 42–65 dos decks anteriores são legítimos — e caem dentro da faixa que
>    o Cap. 5 usa como tripwire. Toda arte reproduzida leva carimbo de convenção.
>
> ⚠ **Uma retomada que estava ERRADA no ledger anterior e foi removida:** *"as mesmas janelas"*.
> O `GLOSSARY:90` registra que o Cap. 5 usa janelas **sobrepostas (stride 1)** e os Caps. 3/4 usaram
> **não sobrepostas**. Não são as mesmas, e a equivalência falsa teria subido à tela.

---

## 5 · Três posições que precisam de redação exata

### 5.1 · O veredito (linguagem de lei — `../WRITING_LAW.md` §3)

> **Próxima categoria:** supera o modelo dedicado **em Florida** (+0,19 macro-F1, Holm *p* 0,011).
> As outras cinco diferenças são **não resolvidas**, cada uma limitada a meio ponto de zero.
>
> **Próxima região:** **não-inferior nos seis** conjuntos (TOST, margem de dois pontos registrada
> antes de qualquer resultado ser lido), com **Texas +1,21** e **Califórnia +1,06** superando.
> As **quatro** células dentro da margem são **déficits, não empates**, e **os quatro intervalos
> ficam inteiramente abaixo de zero** — o texto diz que a direção "é declarada, não arredondada":
>
> | dataset | Δ | intervalo |
> |---|---:|---|
> | Alabama | −0,87 | −1,00 a −0,75 |
> | Arizona | −0,44 | −0,62 a −0,25 |
> | Florida | −0,16 | −0,19 a −0,13 |
> | Istambul | −0,08 | −0,16 a −0,002 |
>
> **Enumere as quatro.** Citar três e omitir Istambul é o erro que a validação pegou no rascunho.

> **As cinco diferenças de categoria não resolvidas — com sinal, porque não apontam para o mesmo
> lado.** Três excluem zero antes da correção, e **em direções opostas**:
>
> | dataset | Δ | intervalo | a favor de |
> |---|---:|---|---|
> | Istambul | **+0,08** | +0,01 a +0,15 | **modelo conjunto** |
> | Arizona | −0,00 | −0,04 a +0,03 | — |
> | Califórnia | −0,00 | −0,03 a +0,02 | — |
> | Texas | **−0,13** | −0,19 a −0,08 | **dedicado** |
> | Alabama | **−0,19** | −0,33 a −0,04 | **dedicado** |
>
> Todas **não resolvidas após Holm**, e todas limitadas pelo meio ponto derivado. Nunca dizer
> "excluem zero" sem dizer **para que lado**.

"Supera" só nessas três células. Em nenhum outro lugar, em nenhum dos dois eixos.

### 5.2 · A Tabela 9, como a própria tabela a enuncia

O rascunho dizia *"check-in bate place-level nos seis"*. A nota de rodapé da tabela entregue diz outra coisa, e a fala
segue a tabela:

> *"O nível de check-in está à frente nos seis e é unânime nas cinco dobras em todos os conjuntos;
> um teste pareado separa as colunas em cinco dos seis — Florida é a exceção, a p = 0,07, e é o
> menor salto da tabela (+0,23)."*

### 5.3 · O tronco compartilhado — a posição do autor, redigida para se sustentar

O autor sustenta que **não há evidência suficiente para afirmar que o tronco compartilhado não contribuiu**. Isso é
defensável, e já está no texto entregue (`5_mobiwac/07_discussion.tex`):

> *"A evidência aqui não separa as contribuições [...] a representação compartilhada parece capaz
> de ajudar, e quanto ela ajuda é uma questão para trabalho futuro em vez de uma que este estudo
> resolve."*

**A formulação que se sustenta é simétrica**, e a simetria é o que a protege:

> *"A evidência não separa as contribuições do tronco compartilhado e do caminho espacial privado.
> Ela não estabelece que o compartilhamento ajuda, e não o descarta. A afirmação que faço é sobre
> o **desenho** — esta combinação produz uma saída de região acima de dois modelos dedicados nos
> dois conjuntos com os maiores vocabulários de região — não sobre transferência entre tarefas."*

⚠ **O risco a evitar é escorregar para o outro lado.** "Não podemos provar que não contribuiu" é verdade; "portanto
provavelmente contribuiu" não é. O P1 mostra que a vantagem **não sobrevive ao pareamento de capacidade** — isso não
prova ausência de contribuição, mas também não pode ser apresentado como se a apoiasse. Cláusula obrigatória em 5.5,
para que a fala sobreviva à pergunta:
os ganhos de região em TX/CA são **resultados secundários, fora do plano registrado**.

---

## 6 · A trilha de reserva (série B)

Vive **depois** do slide de agradecimentos, fora da contagem principal. O template resolve o "fora
da contagem": envolver a série em `\miniframesoff` … `\miniframeson` faz os slides **não
registrarem ponto na barra** (§11). ⚠ Mas o `\insertframenumber` **congela em vez de sumir** — os
rótulos `B-n` têm de estar **no conteúdo do slide**, nunca no rodapé.

**Contrato de cada slide B:** uma pergunta = um slide · **o título é a pergunta, em português, como
a banca a faria** · rodapé de proveniência (página do volume / caminho / `pós-submissão — não consta
em nenhum dos dois volumes`) · números copiados de célula impressa ou do `ladder_recompute.json`,
nunca re-derivados · onde a resposta honesta é "não foi medido", **o limite é a manchete** ·
**carimbo de convenção métrica** em toda arte reproduzida dos Caps. 3/4.

`B0` é um **índice clicável** (pergunta → slide, com `\hyperlink`), para chegar a qualquer slide em
dois cliques. Os dois volumes ficam abertos em janelas separadas para o *"vá à página X"*.

**Cobertura obrigatória (o guia §4.3 exige 1:1):** cada `[ABERTO]` e cada `U1`–`U8` do
[`../wrapup/open_points/ARGUICAO.md`](../wrapup/open_points/ARGUICAO.md) recebe **um** slide,
identificado pelo seu código, para que a conformidade seja verificável mecanicamente.
**Os dois `[ABERTO]` são Q5** (o tamanho da dependência entre as duas entradas do modelo conjunto)
**e Q8** (a vantagem de região é transferência ou é arquitetura e parâmetros?) — ambos com resposta
de limite, e o Q8 é o que o B-P1 e o slide Karpathy servem juntos.

| Família | Cobre |
|---|---|
| **B1 · Veredito e estatística** | as células com intervalos (§5.1, as quatro de região e as cinco de categoria **com sinal**); as duas convenções de checkpoint; a justificativa da margem de 2 pontos; n=4 e o piso do Wilcoxon em 0,0625; o piso de significância prática. *(Q1, Q2, Q4, §7.1)* |
| **B2 · Protocolo e vazamento** | transdutividade + o controle de reconstrução por dobra (com a ressalva de cobertura 67–87%); a aresta só-para-frente e quanto ela vale; a matriz de cobertura de busca (com ERR-6/ERR-7 oferecidas); **"o senhor escolheu a época no mesmo conjunto em que reporta?"**; a auditoria de proveniência pré-vazamento. *(Q3, Q12, Q19)* |
| **B3 · Pós-submissão** | **B-P1** *(= a resposta de execução a **Q14** e **U2**)*, **B-Q13** e **B-Q14** *(a divergência entre documentos: o artigo lista o confundimento de capacidade entre os seus limites e a dissertação não)*, os três com distintivo `OFERECER PROATIVAMENTE`; **B-Q15**; o **slide de nomenclatura** (abaixo); o **slide Karpathy** (abaixo); as contagens corrigidas do Apêndice G; e um **bloco reservado para o mtlcheck** (abaixo) |
| **B4 · Capítulos 3–4** | vazamento de rótulo da tarefa estática + confundimento de largura; **Travel rotulado por tarefa**; a acusação de *selection-overfitting* do juiz do CoUrb; validade do protocolo dos dois primeiros estudos; a divergência do corpus de Florida; **o custo do Cap. 3** (80,88 s × 34,97 s cumulativos = 2,3× — **a tabela, nunca a prosa publicada**, que a `ERRATA.md` do próprio artigo registra como defeituosa) |
| **B5 · Não foi medido** | U1–U8, cada um com o limite como manchete |
| **B6 · Documento e escopo** | o defeito do Resumo (§10) com Resumo × Cap. 5 lado a lado; a colisão de letras de apêndice; contagens de usuários (corpus bruto × pós-filtro); as sete perguntas `[FECHADO]` hoje sem família — **prioridade em Q22** (o piso de Markov de região acima de três sistemas publicados) |
| **B7 · "Como o Check2HGI e o modelo conjunto funcionam"** | **NOVA** (seu ponto 10): 5–7 slides sobre o **Apêndice E do volume principal**, reusando as figuras que já existem. A fala menciona as duas arquiteturas sem tempo para detalhe; esta família é onde o detalhe mora. Inclui a pergunta mais afiada do apêndice: *o cross-attention atende histórico de região de um usuário com histórico de categoria de outro?* |

### Três slides que este plano acrescenta por nome

**B-NOM · *"Vocês chamaram de seed o que a literatura chama de repetition. Isso muda os resultados?"***
Fonte: `mtlcheck/docs/NOMENCLATURE.md`.
→ **Não muda a inferência.** O teste reportado sempre foi o *t* pareado sobre as **quatro médias por
semente** — n = 4, 3 g.l. O `GLOSSARY:111` já registra `n = 20 (modelos ajustados)` e `n = 4
(unidade inferencial)` como coisas distintas, e **proíbe** escrever "n = 20 paired repetitions".
→ **Muda o nome, e nomeia um limite.** Cada "semente" é uma **repetição** de validação cruzada; e um
único inteiro dirige **duas coisas**: a partição e a inicialização. Logo "variância entre sementes"
é **partição + inicialização**, e **nenhum resultado já produzido as separa**.
→ **Magnitudes:** entre folds ~1,2 pp · entre repetições 0,02–0,07 pp · banda pareada 0,05–0,15 pp.
O termo de fold é 20–50× o de repetição **e é comum aos dois braços** — é por isso que o pareamento
detecta o que a análise não pareada não detecta.
→ **Enquadramento:** a reescrita não corrigiu um erro; **nomeou um confundimento que ninguém tinha
nomeado.** É a metade estrutural do seu ponto 7.

**B-KARPATHY · o contexto de capacidade** *(abre a oferta proativa do B-P1)*
Karpathy (2019) sobre desenhar MTL: *"how much feature sharing is there"*, *"tasks fight for the
same shared capacity"*, *"there's finite capacity to go around... I don't really have language to
describe how to correctly allocate capacity to tasks"*. Mais **Standley *et al.* (ICML 2020)**, a
matriz de afinidade entre tarefas — citável. PCGrad e GradNorm, que ele discute, **já estão na
dissertação**, então a ponte do documento para o slide é contínua, não importada.
→ **Função:** converter o P1 de ferida em posição — *"este é o problema aberto reconhecido da área;
o controle que rodei é exatamente a medição que essa literatura diz que ninguém sabe desenhar."*
> ⚠ **Nunca na trilha principal.** *"Tasks fight for capacity"* é um exagero na **direção oposta**
> à que o §5.3 protege; no fecho, puxaria a fala para conceder a posição do tronco. A formulação
> simétrica de §5.3 governa este slide, e ele **nunca "explica"** os resultados entregues.

**B-MTLCHECK · bloco reservado para os resultados novos** *(seu ponto 10)*
O sistema experimental foi reescrito (`../wrapup/NEW_VERSION.md`) e os resultados estão sendo
reexecutados. Entram **como material extra, nunca como correção**. O que já existe: a paridade de
oito células (delta médio −0,001 pp). ⚠ **Nunca misturar um número do mtlcheck com um da
dissertação na mesma frase** — protocolos diferentes (splits aninhados 70/10/20, métricas agrupadas
fora-de-dobra, margem derivada de 0,4 pp contra os 2 pp registrados). Sob aquele protocolo,
Alabama/região **vira inferior** — e isso vem de outro livro de regras, não corrige o Cap. 5.

## 7 · O que a validação adversarial mudou

Três passagens adversariais leram o rascunho contra o texto entregue. Quatro frases da trilha principal reivindicavam
mais do que a evidência sustenta, e **uma era simplesmente falsa**. Todas já estão corrigidas acima; ficam registradas
porque a versão corrigida diz *menos*, e é isso que a torna defensável.

| # | O rascunho dizia                                           | O que o texto entregue diz                                                             | Onde foi corrigido     |
|---|------------------------------------------------------------|----------------------------------------------------------------------------------------|------------------------|
| 1 | "ambas as baselines externas batidas em absoluto" (Cap. 3) | **Falso.** Só na tarefa estática. Na sequencial as lideranças se dividem               | §3, *nunca dizer*      |
| 2 | "check-in bate place-level nos seis"                       | O teste pareado separa em **cinco**; Florida a p=0,07                                  | §5.2                   |
| 3 | ganho de região em TX/CA como resultado de MTL             | O texto não credita a transferência; o P1 mostra capacidade                            | §5.3 + cláusula em 5.5 |
| 4 | o plano registrado como ativo de integridade sem ressalva  | Houve **desvio declarado**: Wilcoxon sobre dobras → t pareado sobre médias por semente | §3, seção 5.4          |

**Divergências deliberadas em relação ao guia** (declaradas, não acidentais):

1. **Um bloco de fundamentos que o §4.1 não tem.** O orçamento do guia vai de "Abertura" a
   "Cap. 3" sem parada, o que obriga cada capítulo a repagar preâmbulo — colidindo frontalmente com a prioridade de
   de-duplicação. Inserido, financiado pela folga aritmética abaixo.
2. **Alvos pontuais em vez de faixas.** As linhas do §4.1 somam 36–47 min, e o próprio guia declara "~42–47": o piso
   está subestimado em 6 minutos. Uma faixa de 11 minutos não é orçamento. Aqui: 4 + 5 + 5 + 6 + 20 + 5 = **45 min**, folga **5 min** contra o teto de 50
   *(revisto 2026-08-21 — ver §3)*.
3. **Numeração de figuras do documento, não do artigo.** A Figura 1 do artigo MobiWac é a **Figura 4** da dissertação. A
   banca lê o documento. Mapeamento fixo: dataflow = **Fig. 4**; modelo = **Fig. 5**; separabilidade = **Fig. 6**;
   deltas = **Fig. 7**; cosseno = **Fig. 8**; datasets = **Tab. 8**; representação = **Tab. 9**; conjunto × dedicado =
   **Tab. 10**.

---

## 8 · Regras que o `SLIDES.md` terá de obedecer

1. **Navegação.** Nativa do template (§11): seis `\section[curto]{longo}`. O **longo** é o título do artigo (seções
   3–5); o **curto** vai para a barra e é a **linhagem de modelos**: *Introdução · Fundamentos · **MTLnet** ·
   **ST-MTLNet** · **Check2HGI** · Conclusão*. A barra passa a narrar a própria evolução que é o arco, e casa com a
   Tabela 1 da dissertação. Nomes nunca mudam no meio do deck. **Nenhum `\section` fora dessas seis**, porque cada um
   vira um rótulo na barra; divisões internas usam `\subsection`.
2. **A linguagem do veredito é lei.** "Supera" só nas três células de §5.1.
3. **Proveniência de todo número.** Copiado de célula de tabela entregue (lendo o comentário de proveniência ao lado),
   da tabela publicada do Cap. 3/4, ou do `ladder_recompute.json`. **Nunca re-derivado.** Qualquer macro-F1 de categoria
   **de PRÓXIMA CATEGORIA** entre **54 e 80** é número vazado pré-v18: pare.
   ⚠ **A faixa é escopada de propósito.** Sem o escopo ela dispara nos Acc@10 de região legítimos (Alabama 69,24;
   Arizona 59,04), que estão corretos e impressos na Tabela 10.
4. **Etiqueta de ledger em cada slide.** `INTRODUZ <elemento>` ou `RETOMA <elemento>`. Um elemento carrega `INTRODUZ` em
   exatamente um slide, no bloco que §4 designa. Isso torna a de-duplicação **verificável**, não uma intenção.
5. **Arte reproduzida.** Toda figura/tabela dos Caps. 3/4 mantém a citação de origem e recebe a anotação
   `Next-POI Prediction = next category (Def. 2.7)`.
6. **Ordem da verdade: ressalva antes da manchete, sempre.** A tarefa estática do Cap. 4 é o caso paradigmático.
   Limitações acompanham o resultado que limitam.
7. **Proibição entre gerações.** Nenhum número do Cap. 3 e do Cap. 5 no mesmo eixo, tabela ou frase. Nenhum "fomos de X
   para Y" através de protocolos.
8. **Material pós-submissão só na série B**, cada slide com rodapé `pós-submissão — não consta em
   nenhum dos dois volumes`.
9. **Proibições literais nas notas do apresentador**: `100,2%` / `101,9%` (Apêndice G do suplemento — as contagens reais
   são 230% / 234%); `+2,12 / +2,05` do P1 (substrato superado); as células v17 `AL 63,56 / FL 79,85 / CA 77,05`;
   `1,1 milhão para os dois dedicados` (medido:
   1.850.980); `sob um décimo do salto`; e a frase do Resumo entregue.
10. **Densidade.** Uma ideia por slide. Tabelas 9 e 10 ganham slide próprio e 1,5–2 min cada. Slides de
    figura/definição/transição: 20–40 s. Mínimo 16 pt. Marcadores por palavra-chave, nunca parágrafos. Slides numerados
    (útil no remoto: *"volte ao slide 14"*).
11. **Terminologia fail-closed.** Só termos do [`../GLOSSARY.md`](../GLOSSARY.md); notação idêntica, caractere a
    caractere, à do documento.
12. **Slides de transição são estruturais.** Cada seção termina na sua frase de transição fixa de §2. Passagens de corte
    de tempo **não podem removê-las**. O divisor visual de cada seção é automático via `\autotocframe` (§11); a frase de
    transição é falada sobre ele, ou vai num
    `\specialframe` próprio quando merecer a tela inteira.
13. **A contribuição aparece duas vezes**, com redação idêntica: um slide cedo e o slide de fechamento — sempre com a
    ressalva de que o ganho é operacional, **não computacional**.
14. **Profundidade de apêndice fica fora do deck principal.** E "Apêndice B" é **sempre** nomeado com o volume: no
    principal é a Declaração de Uso de IA; no suplemento é a Errata.
15. **O texto dos slides obedece às mesmas três leis do documento** (seu ponto 1):
    [`../WRITING_LAW.md`](../WRITING_LAW.md) (registro, lei dos verbos, construções banidas),
    [`../GLOSSARY.md`](../GLOSSARY.md) (**fail-closed**: termo fora do registro não pode ser usado) e
    [`../AGENT_GUARDRAILS.md`](../AGENT_GUARDRAILS.md) (protocolo de número e de afirmação). Um deck não é prosa, mas
    as três leis governam palavra e número igual. As duas passagens de portão — **G2 (fato)** e **G3 (estilo)**, por
    agente que não escreveu o slide — rodam sobre o `SLIDES.md` antes do deck construído.
16. **O que nunca vai num slide.** A tabela de 21 linhas do Cap. 3; os formalismos do zoo de balanceadores; o
    levantamento completo de literatura; MFLOPs sem o enquadramento de tempo de parede; especulação sobre erro
    geográfico ou desempenho de serviço.

---

## 9 · Portões e prazos (hoje 2026-08-21, defesa 28/08)

| Quando          | O quê                                                                                                                                                                                                          |
|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **22/08**       | **Este plano para o Fabrício** — o arco, o orçamento e a barra de navegação, em uma página. O guia §2 é explícito: *a estrutura da apresentação é decisão sua e do orientador*. Nada abaixo começa antes disso |
| 24/08           | `SLIDES.md` — o slide-a-slide, obedecendo §8                                                                                                                                                                   |
| 25/08           | Deck construído; ensaio cronometrado nº 1                                                                                                                                                                      |
| 26/08           | Passagem do orientador; série B completa (cobertura 1:1 com ARGUICAO)                                                                                                                                          |
| 27/08           | Ensaio nº 2 no Meet, com compartilhamento de tela; PDF de reserva local; os dois volumes abertos em janelas separadas                                                                                          |
| **28/08 10:00** | Defesa                                                                                                                                                                                                         |

### Pendências que só o autor fecha

1. ~~**Barra de navegação**~~ — **FECHADA 2026-08-21**: o autor confirmou o template do NESPeD, que já traz a barra
   nativa. Ver §11.
2. ~~**Q13 / Q14 / Q15**~~ — **FECHADA 2026-08-21**: o autor decidiu **slides prontos para resposta oral**. B-P1 e
   B-Q13 mantêm o distintivo `OFERECER PROATIVAMENTE`; **B-Q15 ganha slide próprio** (estava diluído em B6).
3. **A frase do veredito em português**, ensaiada literalmente. Ela vem de §5.1 deste plano — **nunca do Resumo**.
   *(Confirmada pelo autor como pendência viva, 2026-08-21.)*
4. ~~**LO-11 (crédito de autoria no CoUrb)**~~ — **RETIRADA 2026-08-21** a pedido do autor: não precisa ser levantada.
5. **NSO-46** — o único marcador de aval ainda aberto (`wrapup/erratas/material_apx_static_scope.tex`). Decidir antes
   de 26/08, porque o slide B4 do autovazamento do Cap. 3 depende dele.
6. **A errata do `GLOSSARY:161`** (§10) — aplicada nesta revisão; confirmar que a redação serve.

---

## 10 · Registro de correção ao documento entregue

**2026-08-21 — o Resumo em português.** A frase *"superou os modelos dedicados na previsão da próxima categoria em todos
os conjuntos"* generalizava um resultado que o documento delimita a Florida. O defeito era **isolado**: o Abstract em
inglês, §2.5, o Cap. 5 e o Cap. 6 já diziam
"em um conjunto". Corrigido no fonte por decisão do autor, para valer na versão final; errata em
[`../wrapup/erratas/errata_resumo_escopo_categoria.tex`](../wrapup/erratas/errata_resumo_escopo_categoria.tex), com a
resposta oral pronta caso um arguidor cite o Resumo entregue.

**2026-08-21 — o `GLOSSARY.md` oferecia tradução para um verbo que ele próprio bane.** A tabela de
equivalentes em português (§6) trazia `outperforms / matches | supera / **equipara-se**`. Mas o
mesmo arquivo diz, na linha 107, *"Verdict verb 'matches' is banned"*, e na 109 proíbe *"matches,
empata, semelhante, a par"*; o `WRITING_LAW.md:226` bane a família inteira. **"Equipara-se" é
semanticamente a família banida.** O defeito era acionável: um redator de slides em português
consultando a tabela de equivalentes encontraria o termo proibido e o usaria de boa-fé — e as duas
superfícies corretas já estavam registradas três linhas acima. Corrigido: a linha agora aponta para
elas em vez de competir com elas, com o registro do defeito num comentário ao lado.

> **O `src/banca.pdf` não foi reconstruído.** Ele continua sendo o registro exato do que a
> banca recebeu (md5 `5be69d1b…`, 119 pp) e **diverge do fonte nesta única frase**. A correção foi
> verificada num build real do volume de defesa: a frase antiga tem zero ocorrências, a nova
> renderiza, `tex_errors=0`.

---

## 11 · O template — `slides/`

Template Beamer do NESPeD, de **Henrique S. Santana** — o autor da defesa que o guia §4.0 analisa quadro a quadro. A
barra de navegação que o guia recomenda **não precisa ser construída: ela é o comportamento padrão deste template**.
Lido e testado em 2026-08-21.

### 11.1 · O que é nativo, e o que isso resolve do plano

| Recurso do template                                           | Resolve                                                                                                                                        |
|---------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| `\insertnavigation` no *headline*, com `\beamer@compresstrue` | **A barra de navegação**: rótulos das seções + um ponto por slide, com o ponto atual destacado. Regra 1 de §8                                  |
| `\section{...}`                                               | Cada seção declarada vira **um rótulo na barra**. As seis seções de §3 são exatamente seis `\section`                                          |
| `\autotocframe{opts}` (`\AtBeginSection`)                     | **O divisor automático de seção** — insere um "Sumário" no início de cada seção, sem precisar lembrar. É o suporte visual das transições de §2 |
| `\tocframe[sectionstyle=…,subsectionstyle=…]`                 | Recapitulação sob demanda, com fundo em gradiente e a barra esmaecida                                                                          |
| `\miniframesoff` … `\miniframeson`                            | **Tira frames da contagem de pontinhos.** É o mecanismo exato de que a série B precisa (§6) para ficar fora do deck principal                  |
| `\insertframenumber` no *footline*                            | **Slides numerados** — o guia §4.4 pede isso, e no remoto é o que permite *"volte ao slide 14"*                                                |
| `\specialframe` (envolver o frame em chaves)                  | Frame de tela cheia com texto branco sobre gradiente — bom para as frases de transição de §2                                                   |
| `\titleframe{\titlelogo{…}}`                                  | Slide de título com logos                                                                                                                      |
| `\block` / `\alertblock` / `\exampleblock`                    | Destaques. `alertblock` é o lugar natural das ressalvas que §8 regra 6 manda vir **antes** do número                                           |

Opções do pacote: decoração `net` \| `accel` \| `data`; cor `green` \| `blue` \| `red`. O template vem com
`[net,green]`. ⚠ **A opção `red` está declarada vazia** no `.sty` — não define nenhuma cor. Não usar.

### 11.2 · Duas coisas operacionais que mudam o build

> ⚠ **1. O motor é XeLaTeX ou LuaLaTeX, NÃO `pdflatex`.** O `nesped.sty:15` carrega `fontspec`, que
> aborta sob `pdflatex` com *"requires either XeTeX or LuaTeX"* e **não produz PDF nenhum**. Isso
> difere do build da dissertação, que é `pdflatex` via `src_utils/latexbuild.sh` — então o deck
> **não** pode reusar aquele harness sem trocar o motor.
>
> ⚠ **2. O template tem um bug: `\pagewidth` não existe.** Em `nesped.sty:245`, dentro do
> *headline*, a faixa de gradiente sob a barra de navegação usa `\pagewidth` onde LaTeX/Beamer
> definem `\paperwidth` (o próprio bloco usa `\paperwidth` corretamente nas linhas acima).
> Resultado: `Undefined control sequence` **uma vez por frame** — 27 erros no build do template de
> exemplo — e a faixa decorativa não desenha. O PDF ainda sai, porque o erro não é fatal em
> `nonstopmode`, e é exatamente por isso que passa despercebido.
>
> Correção de uma linha, **validada numa cópia isolada** antes de qualquer coisa ser alterada:
> `(\pagewidth, 1mm)` → `(\paperwidth, 1mm)`. Medido: o template de exemplo sai de **27 erros para
> 0**, e a faixa passa a desenhar. (O original não foi tocado.)
> **Status 2026-08-21:** o autor delegou o conserto ao agente **`presentation-guide`** (a sessão que
> escreveu o guia). O escopo passado a ele: template compilando com 0 erros, motor e comando de
> build declarados, varredura própria por defeitos além dos dois acima (presumindo que
> `nonstopmode` esconde mais), verificação de que `\miniframesoff`/`\miniframeson` e os
> hyperlinks internos funcionam de fato, e um `main.tex` esqueleto já com as seis seções de §11.3.
> Ele também decide, com o autor, se a correção fica só na nossa cópia — o template é de terceiros
> e pode estar em uso por outras pessoas do NESPeD.
>
> **CORREÇÃO 2026-08-21 (mesmo dia):** a entrada abaixo, marcada FECHADO, deu a decoração de fundo
> (`\decorationnet`, os pontos com linhas conectando em malha triangulada) como defeito benigno de
> renderização do poppler. **Estava errado.** O Vitor comparou com uma captura de tela do template
> como deveria ficar (malha densa, todo ponto conectado aos vizinhos) contra o PDF que baixou do
> Overleaf — o mesmo problema aparece lá também: só pontos soltos, nenhuma linha. A causa real: em
> `\decorationnet`, cada um dos 12 blocos `\draw[primarysuper] \foreach ... { \ifnum \i>0 -- \fi
> (\node.center) }` insere um `--` **antes do primeiro ponto também** (deveria ser `\ifnum \i>1`,
> não `\i>0`), o que deixa o caminho do TikZ sem coordenada inicial e aborta a linha inteira —
> exatamente o "No current point in lineto" que o poppler reportava aos montes, e que eu MEDI
> existir também no PDF oficial do Henrique sem concluir que era a causa da malha faltando.
> Corrigido (`sed` nos 12 pontos, `nesped.sty`) e **validado contra a captura de tela do Vitor**:
> antes, 0 páginas sem o aviso do poppler; depois, 0 páginas COM o aviso, e a malha triangulada
> aparece igual à referência. `main.tex` e `template_showcase.tex` recompilados com a correção.
> Lição registrada: uma comparação pixel-a-pixel contra o PDF de referência não bastou porque a
> referência **também estava com o mesmo defeito** — só a captura de tela (de fora do PDF gerado
> por este `.sty`) revelou o "como deveria ser". Relato de correção enviado à sessão `ingred-14`.

> **FECHADO 2026-08-21 pelo `presentation-guide`.** `\pagewidth` corrigido; **um segundo defeito
> real foi encontrado**, mais sério que o primeiro porque atinge exatamente o uso que este plano já
> assume em §11.3: `\autotocframe` repassava o argumento com chaves (`\tocframe{#1}`) para um
> comando que só aceita colchetes (`\tocframe[#1]`) — com argumento vazio isso é inofensivo, mas
> com um argumento real como `sectionstyle=show/shaded, subsectionstyle=show/show/shaded` (o
> exemplo do próprio §11.3) o texto do argumento **vaza como conteúdo visível na tela** e ainda
> rouba um número de slide. Corrigido com `\ifstrempty` (etoolbox, já carregado). Achado adicional
> não coberto pelos dois de cima: com `subsectionstyle=show/show/shaded` e a profundidade de
> subseções deste plano (até 6 por seção), o recap automático de Sumário estoura o slide — 6
> `Overfull \vbox`, um por seção. **O `main.tex` usa `subsectionstyle=hide` por padrão** (0
> overfull, validado); reverter para `show/…` é possível mas então cada recap de seção precisa de
> um layout diferente (várias colunas, fonte menor), não testado. `\miniframesoff`/`\miniframeson`
> e os hyperlinks internos (`\hyperlink`/`\hypertarget`) confirmados funcionando via build de
> teste isolado. Build validado linha a linha contra `nesped_slides_template.pdf` (o PDF de
> referência que o autor baixou, produzido por LuaTeX) — texto idêntico, únicas diferenças são
> kerning entre XeLaTeX/LuaTeX e o estilo de maiúsculas dos autores na página de Referências
> (cosmético, específico da fonte, e fora do escopo do deck de defesa). Motor: `xelatex`. Comando:
> `make check` (compila e reporta erro/sucesso sem deixar PDF pela metade) ou `make all` (build
> completo com bibtex). `main.tex` (o antigo showcase de demonstração) foi preservado como
> `template_showcase.tex` — `make showcase` recompila-o — e o novo `main.tex` é o esqueleto real
> das seis seções, com um stub por subseção de §3, os pontos de transição de §2 comentados, e a
> série B (§6) demonstrada com um índice clicável (B0) + um exemplo completo (B1) para replicar.
> Decisão sobre o original: a correção fica **só nesta cópia** — reportar ao Henrique fica a
> critério do autor. Relato completo enviado à sessão `ingred-14`.

### 11.3 · Como as seis seções de §3 mapeiam

> ⚠ **CORRIGIDO 2026-08-21.** A versão anterior desta seção mandava usar
> `\autotocframe{sectionstyle=show/shaded, subsectionstyle=show/show/shaded}`. **Aquilo estava
> errado por dois motivos independentes**, os dois achados pelo agente `presentation-guide` e
> reproduzidos aqui antes de aceitos:
>
> 1. O `\autotocframe` original repassava o argumento **entre chaves** para o `\tocframe`, que só
>    aceita **colchetes**. Com um argumento real — exatamente o de cima — o texto das opções
>    **vira um slide visível**, um por seção, e rouba o número de slide. Medido: num teste com duas
>    seções, as páginas 2 e 5 imprimem literalmente `sectionstyle=show/shaded,
>    subsectionstyle=show/show/shaded`. No deck real, com seis seções, seriam **seis slides de
     > lixo**. E compila com **0 erros** — inteiramente silencioso.
>    Corrigido no `nesped.sty` com `\ifstrempty` (etoolbox, já carregado).
> 2. `subsectionstyle=show/show/shaded` **estoura o slide de recapitulação** com a profundidade de
>    subseções deste plano (até 6 por seção, em Fundamentos e MobiWac): 6 `Overfull \vbox`, um por
>    seção. Com `hide`, zero.

```latex
\usepackage[net,green]{nesped}
\autotocframe{sectionstyle=show/shaded, subsectionstyle=hide}

\titleframe{ \titlelogo{img/logo-nesped.png} \titlelogo{img/logo-ufv.png} }

\section[Introdução]{Abertura --- a pergunta e o escopo}                          % 4 min
\section[Fundamentos]{Fundamentos compartilhados}                                 % 5 min
\section[MTLnet]{Multitask Learning for POI Category and Next-POI Prediction}      % 5 min
\section[ST-MTLNet]{ST-MTLNet: Spatio-Temporal POI Representations}               % 6 min
\section[Check2HGI]{A Check-in-Level Multitask Study of Next Category and Region}  % 20 min
\section[Conclusão]{Conclusão Geral --- a resposta condicional}                    % 5 min

\miniframesoff                % a série B começa aqui: fora da barra e da contagem
```

**Regra que decorre disso:** nenhum `\section` além desses seis, porque cada um vira um rótulo na barra e a barra é o
fio condutor. Divisões internas usam `\subsection`, que aparece como pontos agrupados, não como rótulo novo.

### 11.4 · Estado entregue (2026-08-21, última atualização: separação template/slides + build/)

Consertado e verificado pelo agente `presentation-guide`, e **re-verificado nesta sessão**:

| item | estado |
|---|---|
| **Estrutura de pastas** | `nesped_slides_template/` = o template **puro** (só as fontes do NESPeD + os 3 bugs corrigidos, `main.tex` é a demonstração original do Henrique, intocada estruturalmente) · `slides/` = os slides reais da defesa, que usam uma cópia do template já corrigida |
| `nesped.sty` (nas duas pastas, cópias idênticas) | **três** bugs corrigidos: `\pagewidth`→`\paperwidth`; `\autotocframe` repassando `{#1}` em vez de `[#1]` (`\ifstrempty`); os 12 `\ifnum \i>0 -- \fi` de `\decorationnet` (deveria ser `\i>1`) que quebravam a malha triangulada de fundo — achado depois de comparar com uma captura de tela do Vitor, não só com o PDF de referência (que tinha o mesmo defeito). Só nas nossas cópias — o original de terceiros não foi tocado |
| `slides/main.tex` | **esqueleto real**: as seis seções, um frame-stub por subseção do §3 (`TODO n.n`), os quatro pontos de transição do §2 como comentário, e a série B demonstrada (B0 índice clicável + B1 completo replicável) |
| `slides/template_showcase.tex` | o `main.tex` demonstrativo original do Henrique, preservado por `git mv` como referência de sintaxe |
| **Build** | **um `Makefile` dentro de cada pasta** (`nesped_slides_template/Makefile`, `slides/Makefile` — não mais um na raiz, revertido a pedido do autor 2026-08-21). Cada um compila para `<pasta>/build/` (gitignored) e copia o `main.pdf` final para a raiz da própria pasta. Alvos, iguais nos dois: `make all` · `make check` (valida 0 erros sem gerar PDF pela metade) · `make clean` |
| **Pastas limpas (2026-08-21)** | removidos `.DS_Store`, `slides/template_showcase.tex`/`.pdf` (duplicata do `nesped_slides_template/main.tex`, que já é a demo original), e `build/` de ambas. O PDF de referência do Henrique mudou de `slides/` para `nesped_slides_template/nesped_slides_template.pdf` (mais coerente — ele valida o template, não o deck) |
| **motor** | **`xelatex`** — ver ⚠ abaixo sobre por que não é `pdflatex` |
| build | `make all` em cada pasta → **OK, `nesped_slides_template/main.pdf` 23 páginas + `slides/main.pdf` 42 páginas, 0 erros nos dois** (rodado nesta sessão) |

> ⚠ **`pdflatex` foi testado 2026-08-21 e NÃO é seguro para este template, apesar de compilar sem
> erro.** `fontspec` (a causa original de exigir xelatex/lualatex) na verdade nunca é usada em
> lugar nenhum do `.sty` além do `\RequirePackage` — é vestigial, e `cabin` já suporta os três
> motores sozinho. Guardando o require (`\ifPDFTeX\else\RequirePackage{fontspec}\fi`), o
> `pdflatex` compila com 0 erros e os slides de **conteúdo normal saem pixel-idênticos** ao
> xelatex (testado: página "Blocos"). **Mas os três mecanismos que pintam fundo em página
> inteira via `\tikz[remember picture,overlay]{\backgradient ...}` saem em branco** — sem
> gradiente, sem texto: `\titleframe` (a capa), `\tocframe`/`\autotocframe` (o recap de
> Sumário — **usado em toda seção deste deck**) e `\specialframe`. A causa provável é uma
> diferença de driver PGF/TikZ entre pdftex e xetex para `shading=axis` com ângulo, não
> investigada a fundo. **Não trocar o motor** sem resolver isso primeiro — quebraria a capa e
> todo divisor de seção do deck real, silenciosamente (compila, só não aparece nada).

> **Confirmado de forma independente nesta sessão, e o instrumento importou.** Compilei
> `slides/main.tex` com `pdflatex` (só desabilitando o `fontspec`): **42 páginas, 0 erros**, e a
> camada de TEXTO está completa — `pdftotext` acha os nomes das seções e os títulos normalmente.
> **Um teste de texto teria dado tudo certo.** Só a renderização revela o defeito: o slide de
> sumário sai com o fundo em branco, os nomes das seções invisíveis (texto branco sobre branco,
> porque o gradiente que deveria estar atrás não desenha) e só os numerais 1–6 fantasmas visíveis.
> Medida objetiva do mesmo slide, renderizado a 80 dpi: **2,2 KB sob `pdflatex` contra 37 KB sob
> `xelatex`** — dezesseis vezes menos tinta na página. No deck real isso atingiria a capa **e os
> seis divisores de seção**, sem um único erro de compilação.
>
> **Regra que decorre:** para este template, "compilou sem erro" e até "o texto está lá" não são
> evidência de que a página aparece. Qualquer mudança de motor, de classe ou de pacote gráfico
> precisa ser validada por **renderização**, não por log nem por extração de texto.

> ⚠ **A contagem de páginas do `make check` não é a final.** `check` é passe único, sem bibtex e
> sem o TOC resolvido, então reporta **24** páginas para o template onde o `make all` completo
> reporta **23** (medido nesta sessão, nos dois alvos). Use `check` para saber se compila; use
> `all` para qualquer número que vá ser citado.
>
> ⚠ **O PDF de referência oficial (`slides/nesped_slides_template.pdf`, LuaTeX, 586.883 B) está
> RASTREADO por exceção explícita no `.gitignore`.** Ele já foi apagado uma vez, em 2026-08-21,
> justamente por estar invisível ao `git status` sob a regra `*.pdf`. Se a pasta mudar de nome de
> novo, **mova a exceção junto** — a regra é por caminho, e a migração para `slides/` já a quebrou
> uma segunda vez.

**Três coisas a saber antes de escrever conteúdo:**

1. **O número de slide da série B congela, não some.** Sob `\miniframesoff` o frame não registra ponto na barra, mas
   `\insertframenumber` para de incrementar em vez de ficar em branco. Então os rótulos "B1, B2…" têm de estar **no
   conteúdo do slide**, não no rodapé.
2. **Nenhum campo do título pode ficar vazio.** `\subtitle`, `\institute` e `\date` vazios quebram o template com *"
   There's no line here to end"*. O esqueleto já preenche os quatro.
3. **Confirmar a grafia do nome e do título** exatamente como na folha de rosto entregue — o esqueleto usa `Vitor Hugo`
   e um `TODO` como placeholders.