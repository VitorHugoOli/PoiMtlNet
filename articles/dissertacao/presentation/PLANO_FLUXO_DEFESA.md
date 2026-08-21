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

| Decisão | Ruling | Quando |
|---|---|---|
| **Veredito cedo** | O resultado aparece na abertura, em linguagem de lei, nos primeiros três minutos. Cada slide seguinte é resposta a uma pergunta já feita. | Autor, 2026-08-21 |
| **Idioma** | **Fala em português, slides em inglês.** Os slides espelham o texto entregue, então nenhum número ou notação diverge entre tela e documento. | Autor, 2026-08-21 |
| **Pós-submissão** | P1 (capacidade) e mtlcheck ficam **só na trilha de reserva**. A trilha principal é redigida para *sobreviver* a eles. | Autor, 2026-08-21 |
| **A posição sobre o tronco** | O autor sustenta que **não há dados suficientes para provar que o tronco compartilhado não contribuiu**. Ver §5.3 — é defensável e já está no texto entregue. | Autor, 2026-08-21 |
| **Defeito do Resumo** | Corrigido no fonte para a versão final + errata registrada + **nunca dito em voz alta**, exceto se perguntado. Ver §6, família 6. | Autor, 2026-08-21 |
| **Template e barra de navegação** | **`nesped_slides_template/`** (Beamer, do NESPeD — autoria de Henrique S. Santana, a mesma defesa que o guia §4.0 analisa). A barra de navegação **é nativa do template**, não precisa ser construída. Ver §11. | Autor, 2026-08-21 |
| Logística | 28/08/2026, 10:00–12:30, **remota** (Google Meet). Banca: Fabrício A. Silva (presidente), Clayson S. F. de Sousa Celes (ITA, externo), Alex Borges. | Guia §0 |
| Teto | **50 minutos** (Art. 23). Este plano orça **43 min**, folga de 7. | Regimento |

---

## 1 · A espinha

> **Uma frase, a que a banca deve conseguir repetir de volta:**
>
> *MTL não ajuda predição de POI automaticamente. Um nulo previsto, sob embedding por lugar e
> compartilhamento rígido, foi diagnosticado — com a arquitetura congelada — como um problema de
> representação, e resolvido por uma representação em nível de check-in com atenção cruzada, sob
> um protocolo sem vazamento e disjunto por usuário. A contribuição são as **condições
> identificadas**, não um "sim" universal.*

Essa espinha **não foi inventada para a apresentação**. É a do próprio documento
(`1_introduction.tex` §1.2):

> *"O primeiro reporta um resultado negativo, o segundo identifica seu gargalo principal, e o
> terceiro testa a solução resultante. Essa progressão é ela própria parte da contribuição,
> porque cada estudo estreita a explicação sustentada pela evidência."*

Isso importa porque é exatamente o que a pesquisa de banca
(`../docs/research/banca_evaluation_research_2026-07-20.md`) identifica como a diferença entre
uma coletânea que funciona e uma que falha: §6 diz que "capítulos de moldura que apenas resumem
os artigos" é o principal modo de errar; §7 diz que "uma narrativa de evolução honesta transforma
a fraqueza do formato em evidência de processo científico".

---

## 2 · Os quatro atos, e as transições

As transições são o entregável desta seção — não os atos. Elas são o fio condutor tornado
audível, e **passagens de corte de tempo não podem removê-las**.

### ATO I — "A pergunta e o chão comum" (min 0–10, seções 1–2)

Abre na regularidade da mobilidade e nas aplicações, enuncia a pergunta de pesquisa
literalmente, fixa o escopo (Definições 2.7/2.8 dentro, 2.9 fora), dá o veredito em uma linha, e
então **paga a dívida de duplicação de uma só vez**.

> **Transição de saída (dita, em slide próprio):**
> *"Com o vocabulário, os dados e as regras de decisão fixados uma única vez, cada estudo agora
> só precisa dizer o que mudou. O primeiro usou o que a literatura oferecia — um vetor por lugar
> e um tronco compartilhado."*

### ATO II — "O nulo e o seu diagnóstico" (min 10–18, seções 3–4)

O Cap. 3 entrega o nulo e, o que importa mais, a **bifurcação de três suspeitos**. O Cap. 4
condena um deles por experimento controlado.

> **Transição interna (Cap. 3 → Cap. 4):**
> *"Um nulo com três suspeitos não encerra a investigação — desenha o próximo experimento:
> congelar a arquitetura e mover apenas a entrada."*
>
> **Transição de saída (Ato II → III):**
> *"Com a arquitetura fixa, a entrada moveu o resultado: a representação é o gargalo. Mas o
> diagnóstico ainda é em nível de lugar, sob um protocolo que deixa o mesmo usuário dos dois
> lados da divisão. O terceiro estudo reconstrói as três camadas — representação, topologia e
> protocolo — e, na verificação final, encontramos e fechamos um vazamento no próprio grafo."*

**O vazamento é o pivô, e é dito em voz alta.** Enquadramento: *a verificação funcionando*, não
uma confissão. Ele é mencionado aqui e retomado uma vez dentro de 5.2 — nunca escondido, nunca
repetido uma terceira vez.

### ATO III — "A resolução" (min 18–38, seção 5)

Representação → topologia → protocolo → veredito → o trade medido, **nessa ordem**, para que todo
resultado caia sobre regras já enunciadas no Ato I.

> **Transição de saída:**
> *"Um veredito condicional, medido sob o protocolo mais estrito dos três. O que os três estudos,
> juntos, estabelecem — e o que não estabelecem?"*

### ATO IV — "A resposta condicional" (min 38–43, seção 6)

A resposta condicional, a contribuição una (não uma por artigo), as limitações oferecidas antes
de perguntadas, e o retorno às aplicações como fecho. **A fala termina onde começou, um nível de
entendimento acima** — que é a coletânea funcionando.

---

## 3 · O plano de seções

Seis seções nomeadas, que são também os seis rótulos da barra de navegação (§4, regra 1).
**Minutos são alvos pontuais, não faixas** — uma faixa não é orçamento, não dá para ensaiar
contra ela.

### Seção 1 · Abertura — a pergunta e o escopo — **4 min**

| # | Subseção |
|---|---|
| 1.1 | Motivação em um fôlego: a mobilidade humana é altamente regular (Song *et al.*, ~93%) — **com a ressalva de não-teto anexada na mesma frase** — e as aplicações que dependem de categoria e região |
| 1.2 | A pergunta de pesquisa, literal (§1.2), **e o veredito em uma linha em linguagem de lei** |
| 1.3 | Escopo, enunciado positivamente: Definições 2.7/2.8 dentro; Definição 2.9 (próximo lugar) definida **para ser excluída**; sete categorias; região = setor censitário / *mahalle* |
| 1.4 | O arco em uma linha + o roteiro: três artigos, três veículos, a autoria de cada um |

**Propósito.** Fixa pergunta, fronteira e destino em menos de cinco minutos. **Se cortada:** a
banca encontra o nulo do Cap. 3 sem saber que ele foi *previsto*, e a coletânea vira três resumos.

**Nunca dizer.** "Pioneiro" ou "o primeiro". Nenhum número além do veredito. Nunca "prediz o
próximo POI".

### Seção 2 · Fundamentos compartilhados — dito uma vez — **6 min**

| # | Subseção |
|---|---|
| 2.1 | A linhagem em **uma** tabela (Tabela 1: DGI → HGI → Check2HGI; MTLnet → ST-MTLNet → modelo conjunto) + o contraste do título: **um vetor por lugar × um vetor por visita** (Def. 2.3 × 2.4) |
| 2.2 | Vocabulário de MTL: compartilhamento rígido (Def. 2.10), transferência negativa (Def. 2.12) como o **risco nomeado**, e o critério pré-registrado: *um balanceador só é útil se superar uma ponderação fixa ajustada* |
| 2.3 | A base de evidência: **uma** tabela de seis conjuntos (AL/AZ/FL/TX/CA + Istanbul, ordenada por contagem de regiões, 520 → 8.501), dizendo qual capítulo usou quais |
| 2.4 | Métricas: macro-F1 e por quê (Food ≈ um terço; a perda **não** é reponderada), Acc@10 com a regra de desconto OOD, e o escore de seleção conjunta |
| 2.5 | Protocolo e regras de decisão, **no mesmo slide**: estratificado por amostra (Caps. 3/4 — o mesmo usuário dos dois lados) × disjunto por usuário (Cap. 5), e a lei dos verbos (§2.4.4) |
| 2.6 | Trabalhos relacionados em **um** slide, organizado pelo eixo que importa: categoria/região como *meio* para o próximo lugar × como *fim* |

**Propósito.** É o **motor de de-duplicação do deck inteiro**. Tudo o que os três artigos repetem
é dito aqui uma vez, para que os blocos 3–5 digam só o que mudou. **Se cortada:** cada bloco
repaga 1–2 min de preâmbulo e a fala estoura o teto.

> ⚠ **Esta seção não existe no orçamento §4.1 do guia.** Ali o orçamento pula de "Abertura" direto
> para "Cap. 3", e é precisamente essa lacuna que força cada capítulo a reintroduzir datasets e
> métricas. Ver §7, divergência 1.

**Nunca dizer.** Nenhum resultado e nenhum número de capítulo aqui. Nenhuma afirmação de Pareto.
Nunca chamar as médias por categoria do Cap. 4 de "macro-F1". CTLE nomeado uma única vez.

### Seção 3 · Capítulo 3 — CBIC 2025: o nulo previsto — **4 min**

| # | Subseção |
|---|---|
| 3.1 | O par de tarefas: classificação estática (Def. 2.6) + próxima categoria — **um estático, um sequencial** |
| 3.2 | MTLnet (**Figura 1** — o único desenho completo de arquitetura do deck): encoders por tarefa → FiLM (uma cláusula) → tronco residual compartilhado → duas cabeças |
| 3.3 | Setup em três linhas: Florida, sete categorias, 5 folds — **e a autodeclaração de protocolo dita aqui**, não escondida |
| 3.4 | O resultado nas palavras do próprio capítulo: *"largamente comparáveis, sem vantagem clara ou consistente"*, mais o custo de treino |
| 3.5 | A bifurcação de três hipóteses: dissimilaridade de tarefas / insuficiência de representação / rigidez de topologia |

**Propósito.** Estabelece que o nulo foi **previsto, publicado e produtivo** — ele fabrica os três
suspeitos que estruturam tudo depois. **Se cortada:** o Cap. 4 não tem pergunta a responder.

**Nunca dizer.** Nunca um número do Cap. 3 ao lado de um do Cap. 5 (protocolo, sementes, modelo,
par de tarefas e geração diferentes — **não há delta a reivindicar**). Transferência negativa foi
**hipotetizada, nunca observada**. E — corrigido por dois validadores — **nunca "ambas as
baselines externas batidas em absoluto"**: isso vale só na tarefa estática; na sequencial as
lideranças se dividem (MHA+PE lidera Food e Shopping).

### Seção 4 · Capítulo 4 — CoUrb 2026: o diagnóstico controlado — **4 min**

| # | Subseção |
|---|---|
| 4.1 | A pergunta herdada — *"arquitetura ou representação?"* — e o desenho: MTLnet **congelado**, só a entrada muda |
| 4.2 | **Ressalva primeiro, número depois**: a tarefa estática lê o próprio rótulo (o tipo de local mapeia 1:1 nas sete categorias) → o ganho de 20–22 pontos **não diz nada sobre a tarefa sequencial** |
| 4.3 | O resultado diagnóstico: a tarefa **sequencial**, cujo alvo nunca está na entrada |
| 4.4 | As bordas honestas, oferecidas sem serem pedidas: Travel (a topologia de grafo ainda vence em movimento esparso de longa distância) e a comparação **não pareada em largura** (192 × 64 dims) |
| 4.5 | A frase de entrega, literal do Cap. 6: arquitetura fixa, entrada mudou, resultado moveu → **a representação é o gargalo** |

**Propósito.** Remove um suspeito da lista por experimento controlado — o meio causal do arco.
**Se cortada:** as escolhas do Cap. 5 parecem gosto em vez de consequência de um diagnóstico.

**Nunca dizer.** Nunca número antes da ressalva. Nunca "macro-F1 subiu 20–22" (são médias por
categoria, e não são diagnósticas). Nunca "pareado em largura". Nunca deixar o ganho estático
falar pela tarefa sequencial. Nunca ampliar o crédito de autoria além do registro público.

### Seção 5 · Capítulo 5 — MobiWac 2026: a resolução — **20 min**

| # | Subseção | min |
|---|---|---:|
| 5.1 | O que muda e por quê, **como consequências do diagnóstico**: representação (lugar → check-in), topologia (rígida → atenção cruzada), protocolo (por amostra → disjunto por usuário) | 1 |
| 5.2 | A representação: **Figura 4** (do sequência de check-ins às duas predições); o que um vetor por visita carrega; **Figura 6** (separabilidade); **Tabela 9**; e a aresta só-para-frente | 5 |
| 5.3 | A arquitetura: **Figura 5** — compartilhamento **por troca**, não por camadas possuídas; o caminho espacial privado da região | 3 |
| 5.4 | O protocolo: CV 5-fold disjunta por usuário, sementes {0,1,7,100}, 20 modelos ajustados, teste pareado sobre as **quatro médias por semente** — **e o desvio declarado do plano registrado** | 3 |
| 5.5 | O veredito: **Tabela 10** e **Figura 7**, falados em linguagem de lei | 5,5 |
| 5.6 | O trade medido e os quatro limites declarados | 1,5 |

**Propósito.** O resultado que a dissertação defende, entregue sobre regras que a banca já aceitou
no Ato I — quase metade da fala, coerente com o padrão *curto-curto-longo*. **Se comprimida
abaixo de ~18 min:** o veredito chega sem sua proveniência e vira alegação.

**Nunca dizer.** "Empata", "matches", "ties", "em todos". Nunca aplicar a margem de dois pontos à
categoria (registrada só para região; o meio ponto é um limite **derivado**). Nunca chamar as
quatro células dentro da margem de empates — são **déficits**, o maior AL −0,87. Nunca creditar
TX/CA a transferência entre tarefas (§5.3 abaixo). Nunca "sob um décimo do salto place→check-in"
(**aritmeticamente falso** contra a Tabela 9 na mesma página). Nunca a frase retratada de que o
ganho vem da representação hierárquica e não da injeção de features.

### Seção 6 · Conclusão Geral — a resposta condicional — **5 min**

| # | Subseção |
|---|---|
| 6.1 | A resposta condicional (§6.2): MTL ajudou **sob este desenho e este protocolo** — e o que isso não autoriza |
| 6.2 | A contribuição una (§6.5), **com a redação idêntica à do slide de abertura**: metade prática (um modelo, uma passagem) + metade científica (as condições) |
| 6.3 | **Um** slide de duas colunas: as seis limitações do §6.3 amarradas 1:1 aos seis trabalhos futuros do §6.4 — oferecidas antes de serem pedidas |
| 6.4 | Fecho: retomada das aplicações + agradecimentos + a linha do repositório |

**Propósito.** Converte três resultados em uma tese e fecha a moldura aberta no minuto 0.

**Nunca dizer.** Nunca "MTL funciona" sem condição. Nunca re-caminhar a cadeia dos três estudos
(a barra de navegação e uma cláusula bastam). Nenhum número novo aqui.

---

## 4 · O ledger de de-duplicação

**Esta é a prioridade declarada do autor**, e a regra é mecânica: cada elemento recebe
`INTRODUZ` em **exatamente um** bloco; todos os outros só podem `RETOMAR`, em uma cláusula, sem
re-explicar.

| Elemento | INTRODUZ em | Retoma em | Forma da retomada |
|---|---|---|---|
| Gowalla + Istanbul, estatísticas por dataset | **2.3** | 3, 4, 5 | "nos três estados que este estudo usou" |
| As sete categorias | **2.3** | 3, 4, 5 | "as mesmas sete categorias" |
| macro-F1 e o piso de classe majoritária | **2.4** | 3, 4, 5 | "a mesma macro-F1" |
| Acc@10 e o desconto OOD | **2.4** | 5 | — (só o Cap. 5 usa) |
| Os dois regimes de divisão | **2.5** | 3 (nomeia a falha), 5 (paga) | "o protocolo que a Seção 2 chamou de estratificado por amostra" |
| Sementes, folds, 20 modelos, teste pareado | **2.5** | 5 | "o protocolo já enunciado" |
| MTL, compartilhamento rígido, transferência negativa | **2.2** | 3, 5 | "o risco nomeado na Seção 2" |
| Linhagem DGI → HGI → Check2HGI | **2.1** | 3, 4, 5 | apontar para a Tabela 1 |
| Restrição de modelo único | **1.3** | 5, 6 | "a restrição de um artefato" |
| **MTLnet + a equação FiLM** | **3.2** | 4 | *"a mesma arquitetura, sem alterar uma linha"* |
| Janelas deslizantes 9+1 | **3.3** | 4, 5 | "as mesmas janelas" |
| Nash-MTL / família de balanceadores | **2.2** (família) / **3** (adoção) | 5 | "o critério da Seção 2" |
| Trabalhos relacionados | **2.6** | — | nunca reaberto |
| Check2HGI (o mecanismo) | **5.2** | — | a *ideia* já foi nomeada em 2.1 |
| A aresta só-para-frente / o vazamento | **Ato II→III** (transição) | 5.2 (uma vez) | nunca uma terceira vez |
| O ladder de veredito | **5.5** | 6.1 | "o veredito que acabamos de ver" |

> ⚠ **Duas armadilhas de nomenclatura que a fala precisa desarmar na Seção 1, cada uma em uma
> frase, antes que a banca as encontre sozinha:**
>
> 1. **"Next-POI Prediction" nos Caps. 3 e 4 significa próxima *categoria*, não próximo lugar.**
>    As figuras e tabelas reproduzidas desses capítulos carregam o rótulo antigo. Toda arte
>    reproduzida leva a anotação `Next-POI Prediction = next category (Def. 2.7)`.
> 2. **O par de tarefas muda entre os capítulos.** Caps. 3–4: estática + próxima categoria.
>    Cap. 5: próxima categoria + próxima região. O texto declara isso (§1.2); a fala também deve.
>
> E uma de dados: **Florida aparece com 990.518 check-ins no Cap. 3 e 1.407.034 no Cap. 5.** Não é
> conflito — são duas extrações da mesma fonte pública, em relação de superconjunto. Dito uma vez,
> na primeira menção a Florida.

---

## 5 · Três posições que precisam de redação exata

### 5.1 · O veredito (linguagem de lei — `../WRITING_LAW.md` §3)

> **Próxima categoria:** supera o modelo dedicado **em Florida** (+0,19 macro-F1, Holm *p* 0,011).
> As outras cinco diferenças são **não resolvidas**, cada uma limitada a meio ponto de zero.
>
> **Próxima região:** **não-inferior nos seis** conjuntos (TOST, margem de dois pontos registrada
> antes de qualquer resultado ser lido), com **Texas +1,21** e **Califórnia +1,06** superando.
> As quatro células dentro da margem são **déficits pequenos, não empates** — o maior, Alabama
> −0,87.

"Supera" só nessas três células. Em nenhum outro lugar, em nenhum dos dois eixos.

### 5.2 · A Tabela 9, como a própria tabela a enuncia

O rascunho dizia *"check-in bate place-level nos seis"*. A nota de rodapé da tabela entregue diz
outra coisa, e a fala segue a tabela:

> *"O nível de check-in está à frente nos seis e é unânime nas cinco dobras em todos os conjuntos;
> um teste pareado separa as colunas em cinco dos seis — Florida é a exceção, a p = 0,07, e é o
> menor salto da tabela (+0,23)."*

### 5.3 · O tronco compartilhado — a posição do autor, redigida para se sustentar

O autor sustenta que **não há evidência suficiente para afirmar que o tronco compartilhado não
contribuiu**. Isso é defensável, e já está no texto entregue
(`5_mobiwac/07_discussion.tex`):

> *"A evidência aqui não separa as contribuições [...] a representação compartilhada parece capaz
> de ajudar, e quanto ela ajuda é uma questão para trabalho futuro em vez de uma que este estudo
> resolve."*

**A formulação que se sustenta é simétrica**, e a simetria é o que a protege:

> *"A evidência não separa as contribuições do tronco compartilhado e do caminho espacial privado.
> Ela não estabelece que o compartilhamento ajuda, e não o descarta. A afirmação que faço é sobre
> o **desenho** — esta combinação produz uma saída de região acima de dois modelos dedicados nos
> dois conjuntos com os maiores vocabulários de região — não sobre transferência entre tarefas."*

⚠ **O risco a evitar é escorregar para o outro lado.** "Não podemos provar que não contribuiu" é
verdade; "portanto provavelmente contribuiu" não é. O P1 mostra que a vantagem **não sobrevive ao
pareamento de capacidade** — isso não prova ausência de contribuição, mas também não pode ser
apresentado como se a apoiasse. Cláusula obrigatória em 5.5, para que a fala sobreviva à pergunta:
os ganhos de região em TX/CA são **resultados secundários, fora do plano registrado**.

---

## 6 · A trilha de reserva (série B)

Vive **depois** do slide de agradecimentos, fora da contagem de 35–50. **O template já resolve o
"fora da contagem": envolver a série B em `\miniframesoff` … `\miniframeson` faz os slides não
registrarem ponto na barra de navegação** (§11), então o deck principal continua parecendo ter o
tamanho que tem. `B0` é um **índice
clicável**: família de pergunta → número do slide, com hyperlinks internos do PDF.

**Contrato de cada slide B:** uma pergunta = um slide · **o título é a pergunta, em português,
como a banca a faria** (o autor navega lendo títulos no B0) · rodapé de proveniência (página do
volume principal / do suplemento / caminho de arquivo / `pós-submissão — não consta em nenhum dos
dois volumes`) · números copiados de célula impressa ou do `ladder_recompute.json`, nunca
re-derivados · onde a resposta honesta é "não foi medido", **o limite é a manchete**.

**Cobertura obrigatória (o guia §4.3 exige 1:1):** cada `[ABERTO]` e cada `U1`–`U8` do
[`../wrapup/open_points/ARGUICAO.md`](../wrapup/open_points/ARGUICAO.md) recebe **um** slide,
identificado pelo seu código, para que a conformidade seja verificável mecanicamente.

| Família | Cobre | Fonte |
|---|---|---|
| **B1 · Veredito e estatística** | células com intervalos; as duas convenções de checkpoint; a justificativa da margem de 2 pontos; n=4 e o piso do Wilcoxon; o piso de significância prática | Q1, Q2, Q4, §7.1 |
| **B2 · Protocolo e vazamento** | transdutividade + o controle de reconstrução por dobra (com a ressalva de cobertura 67–87%); a aresta só-para-frente e o que ela vale; a matriz de cobertura de busca (com ERR-6/ERR-7 oferecidas) | Q3, Q12, Q19 |
| **B3 · Pós-submissão e retratações** | **B-P1** (dedicado com capacidade pareada supera o conjunto em CA) e **B-Q13** (o controle de concatenação reverte a frase de mecanismo depositada); mtlcheck; as contagens corrigidas do Apêndice G | P1, Q13, Q14, NEW_VERSION |
| **B4 · Capítulos 3–4** | vazamento de rótulo da tarefa estática + confundimento de largura (um slide); Travel; validade do protocolo dos dois primeiros estudos; a divergência do corpus de Florida | Q16, Q17, Q18, Q19, Q20 |
| **B5 · Não foi medido** | U1–U8, cada um com o limite como manchete e o que existe logo abaixo | U1–U8 |
| **B6 · Documento e escopo** | o defeito do Resumo (§0) com Resumo × Cap. 5 lado a lado; a colisão de letras de apêndice; o volume suplementar | errata_resumo_escopo_categoria, Q15 |

> **Dois slides levam distintivo `OFERECER PROATIVAMENTE`: B-P1 e B-Q13.** São abertos pelo autor
> no momento em que o tema encosta na arguição, **nunca defendidos de um canto**, e sempre abertos
> com a frase de proveniência primeiro: *"isto é trabalho posterior ao envio, com errata redigida
> para o depósito final"*.

---

## 7 · O que a validação adversarial mudou

Três passagens adversariais leram o rascunho contra o texto entregue. Quatro frases da trilha
principal reivindicavam mais do que a evidência sustenta, e **uma era simplesmente falsa**. Todas
já estão corrigidas acima; ficam registradas porque a versão corrigida diz *menos*, e é isso que a
torna defensável.

| # | O rascunho dizia | O que o texto entregue diz | Onde foi corrigido |
|---|---|---|---|
| 1 | "ambas as baselines externas batidas em absoluto" (Cap. 3) | **Falso.** Só na tarefa estática. Na sequencial as lideranças se dividem | §3, *nunca dizer* |
| 2 | "check-in bate place-level nos seis" | O teste pareado separa em **cinco**; Florida a p=0,07 | §5.2 |
| 3 | ganho de região em TX/CA como resultado de MTL | O texto não credita a transferência; o P1 mostra capacidade | §5.3 + cláusula em 5.5 |
| 4 | o plano registrado como ativo de integridade sem ressalva | Houve **desvio declarado**: Wilcoxon sobre dobras → t pareado sobre médias por semente | §3, seção 5.4 |

**Divergências deliberadas em relação ao guia** (declaradas, não acidentais):

1. **Um bloco de fundamentos que o §4.1 não tem.** O orçamento do guia vai de "Abertura" a
   "Cap. 3" sem parada, o que obriga cada capítulo a repagar preâmbulo — colidindo frontalmente
   com a prioridade de de-duplicação. Inserido, financiado pela folga aritmética abaixo.
2. **Alvos pontuais em vez de faixas.** As linhas do §4.1 somam 36–47 min, e o próprio guia
   declara "~42–47": o piso está subestimado em 6 minutos. Uma faixa de 11 minutos não é orçamento.
   Aqui: 4 + 6 + 4 + 4 + 20 + 5 = **43 min**, folga **7 min** contra o teto de 50.
3. **Numeração de figuras do documento, não do artigo.** A Figura 1 do artigo MobiWac é a
   **Figura 4** da dissertação. A banca lê o documento. Mapeamento fixo: dataflow = **Fig. 4**;
   modelo = **Fig. 5**; separabilidade = **Fig. 6**; deltas = **Fig. 7**; cosseno = **Fig. 8**;
   datasets = **Tab. 8**; representação = **Tab. 9**; conjunto × dedicado = **Tab. 10**.

---

## 8 · Regras que o `SLIDES.md` terá de obedecer

1. **Navegação.** Nativa do template (§11): declarar as seis seções com `\section{...}` — *Introdução ·
   Fundamentos · CBIC 2025 · CoUrb 2026 · MobiWac 2026 · Conclusão* — e a barra com os pontos por
   slide aparece sozinha. Nomes de seção nunca mudam no meio do deck. **Nenhum `\section` fora
   dessas seis**, porque cada um vira um rótulo na barra.
2. **A linguagem do veredito é lei.** "Supera" só nas três células de §5.1.
3. **Proveniência de todo número.** Copiado de célula de tabela entregue (lendo o comentário de
   proveniência ao lado), da tabela publicada do Cap. 3/4, ou do `ladder_recompute.json`. **Nunca
   re-derivado.** Qualquer macro-F1 de categoria fora de **30–38** é número vazado pré-v18: pare.
4. **Etiqueta de ledger em cada slide.** `INTRODUZ <elemento>` ou `RETOMA <elemento>`. Um elemento
   carrega `INTRODUZ` em exatamente um slide, no bloco que §4 designa. Isso torna a
   de-duplicação **verificável**, não uma intenção.
5. **Arte reproduzida.** Toda figura/tabela dos Caps. 3/4 mantém a citação de origem e recebe a
   anotação `Next-POI Prediction = next category (Def. 2.7)`.
6. **Ordem da verdade: ressalva antes da manchete, sempre.** A tarefa estática do Cap. 4 é o caso
   paradigmático. Limitações acompanham o resultado que limitam.
7. **Proibição entre gerações.** Nenhum número do Cap. 3 e do Cap. 5 no mesmo eixo, tabela ou
   frase. Nenhum "fomos de X para Y" através de protocolos.
8. **Material pós-submissão só na série B**, cada slide com rodapé `pós-submissão — não consta em
   nenhum dos dois volumes`.
9. **Proibições literais nas notas do apresentador**: `100,2%` / `101,9%` (Apêndice G do
   suplemento — as contagens reais são 230% / 234%); `+2,12 / +2,05` do P1 (substrato superado);
   as células v17 `AL 63,56 / FL 79,85 / CA 77,05`; `1,1 milhão para os dois dedicados` (medido:
   1.850.980); `sob um décimo do salto`; e a frase do Resumo entregue.
10. **Densidade.** Uma ideia por slide. Tabelas 9 e 10 ganham slide próprio e 1,5–2 min cada.
    Slides de figura/definição/transição: 20–40 s. Mínimo 16 pt. Marcadores por palavra-chave,
    nunca parágrafos. Slides numerados (útil no remoto: *"volte ao slide 14"*).
11. **Terminologia fail-closed.** Só termos do [`../GLOSSARY.md`](../GLOSSARY.md); notação
    idêntica, caractere a caractere, à do documento.
12. **Slides de transição são estruturais.** Cada seção termina na sua frase de transição fixa de
    §2. Passagens de corte de tempo **não podem removê-las**. O divisor visual de cada seção é
    automático via `\autotocframe` (§11); a frase de transição é falada sobre ele, ou vai num
    `\specialframe` próprio quando merecer a tela inteira.
13. **A contribuição aparece duas vezes**, com redação idêntica: um slide cedo e o slide de
    fechamento — sempre com a ressalva de que o ganho é operacional, **não computacional**.
14. **Profundidade de apêndice fica fora do deck principal.** E "Apêndice B" é **sempre** nomeado
    com o volume: no principal é a Declaração de Uso de IA; no suplemento é a Errata.
15. **O que nunca vai num slide.** A tabela de 21 linhas do Cap. 3; os formalismos do zoo de
    balanceadores; o levantamento completo de literatura; MFLOPs sem o enquadramento de tempo de
    parede; especulação sobre erro geográfico ou desempenho de serviço.

---

## 9 · Portões e prazos (hoje 2026-08-21, defesa 28/08)

| Quando | O quê |
|---|---|
| **22/08** | **Este plano para o Fabrício** — o arco, o orçamento e a barra de navegação, em uma página. O guia §2 é explícito: *a estrutura da apresentação é decisão sua e do orientador*. Nada abaixo começa antes disso |
| 24/08 | `SLIDES.md` — o slide-a-slide, obedecendo §8 |
| 25/08 | Deck construído; ensaio cronometrado nº 1 |
| 26/08 | Passagem do orientador; série B completa (cobertura 1:1 com ARGUICAO) |
| 27/08 | Ensaio nº 2 no Meet, com compartilhamento de tela; PDF de reserva local; os dois volumes abertos em janelas separadas |
| **28/08 10:00** | Defesa |

### Pendências que só o autor fecha

1. ~~**Barra de navegação**~~ — **FECHADA 2026-08-21**: o autor confirmou o template do NESPeD, que
   já traz a barra nativa. Ver §11.
2. **Q13 / Q14 / Q15** — errata *ou* resposta oral com escopo. Decide a redação e os distintivos
   dos dois slides de retratação (B-P1, B-Q13), e se a linha de errata do Q15 é retirada.
3. **A frase do veredito em português**, ensaiada literalmente. Ela vem de §5.1 deste plano —
   **nunca do Resumo**.
4. **LO-11 (crédito de autoria no CoUrb)** — este plano põe a frase registrada do §1.5 no slide de
   roteiro (1.4), uma linha, sem nomear o estudante. Confirmar essa colocação.

---

## 10 · Registro de correção ao documento entregue

**2026-08-21 — o Resumo em português.** A frase *"superou os modelos dedicados na previsão da
próxima categoria em todos os conjuntos"* generalizava um resultado que o documento delimita a
Florida. O defeito era **isolado**: o Abstract em inglês, §2.5, o Cap. 5 e o Cap. 6 já diziam
"em um conjunto". Corrigido no fonte por decisão do autor, para valer na versão final; errata em
[`../wrapup/erratas/errata_resumo_escopo_categoria.tex`](../wrapup/erratas/errata_resumo_escopo_categoria.tex),
com a resposta oral pronta caso um arguidor cite o Resumo entregue.

> **O `src/dissertacao.pdf` não foi reconstruído.** Ele continua sendo o registro exato do que a
> banca recebeu (md5 `5be69d1b…`, 119 pp) e **diverge do fonte nesta única frase**. A correção foi
> verificada num build real do volume de defesa: a frase antiga tem zero ocorrências, a nova
> renderiza, `tex_errors=0`.

---

## 11 · O template — `nesped_slides_template/`

Template Beamer do NESPeD, de **Henrique S. Santana** — o autor da defesa que o guia §4.0 analisa
quadro a quadro. A barra de navegação que o guia recomenda **não precisa ser construída: ela é o
comportamento padrão deste template**. Lido e testado em 2026-08-21.

### 11.1 · O que é nativo, e o que isso resolve do plano

| Recurso do template | Resolve |
|---|---|
| `\insertnavigation` no *headline*, com `\beamer@compresstrue` | **A barra de navegação**: rótulos das seções + um ponto por slide, com o ponto atual destacado. Regra 1 de §8 |
| `\section{...}` | Cada seção declarada vira **um rótulo na barra**. As seis seções de §3 são exatamente seis `\section` |
| `\autotocframe{opts}` (`\AtBeginSection`) | **O divisor automático de seção** — insere um "Sumário" no início de cada seção, sem precisar lembrar. É o suporte visual das transições de §2 |
| `\tocframe[sectionstyle=…,subsectionstyle=…]` | Recapitulação sob demanda, com fundo em gradiente e a barra esmaecida |
| `\miniframesoff` … `\miniframeson` | **Tira frames da contagem de pontinhos.** É o mecanismo exato de que a série B precisa (§6) para ficar fora do deck principal |
| `\insertframenumber` no *footline* | **Slides numerados** — o guia §4.4 pede isso, e no remoto é o que permite *"volte ao slide 14"* |
| `\specialframe` (envolver o frame em chaves) | Frame de tela cheia com texto branco sobre gradiente — bom para as frases de transição de §2 |
| `\titleframe{\titlelogo{…}}` | Slide de título com logos |
| `\block` / `\alertblock` / `\exampleblock` | Destaques. `alertblock` é o lugar natural das ressalvas que §8 regra 6 manda vir **antes** do número |

Opções do pacote: decoração `net` \| `accel` \| `data`; cor `green` \| `blue` \| `red`.
O template vem com `[net,green]`. ⚠ **A opção `red` está declarada vazia** no `.sty` — não define
nenhuma cor. Não usar.

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
>    lixo**. E compila com **0 erros** — inteiramente silencioso.
>    Corrigido no `nesped.sty` com `\ifstrempty` (etoolbox, já carregado).
> 2. `subsectionstyle=show/show/shaded` **estoura o slide de recapitulação** com a profundidade de
>    subseções deste plano (até 6 por seção, em Fundamentos e MobiWac): 6 `Overfull \vbox`, um por
>    seção. Com `hide`, zero.

```latex
\usepackage[net,green]{nesped}
\autotocframe{sectionstyle=show/shaded, subsectionstyle=hide}

\titleframe{ \titlelogo{img/logo-nesped.png} \titlelogo{img/logo-ufv.png} }

\section{Introdução}          % §3 seção 1  — 4 min
\section{Fundamentos}         % §3 seção 2  — 6 min   (o motor de de-duplicação)
\section{CBIC 2025}           % §3 seção 3  — 4 min
\section{CoUrb 2026}          % §3 seção 4  — 4 min
\section{MobiWac 2026}        % §3 seção 5  — 20 min
\section{Conclusão}           % §3 seção 6  — 5 min

\miniframesoff                % a série B começa aqui: fora da barra e da contagem
```

**Regra que decorre disso:** nenhum `\section` além desses seis, porque cada um vira um rótulo na
barra e a barra é o fio condutor. Divisões internas usam `\subsection`, que aparece como pontos
agrupados, não como rótulo novo.

### 11.4 · Estado entregue (2026-08-21)

Consertado e verificado pelo agente `presentation-guide`, e **re-verificado nesta sessão**:

| item | estado |
|---|---|
| `nesped.sty` | dois bugs corrigidos: `\pagewidth`→`\paperwidth` e o `\autotocframe` de §11.3. Só na nossa cópia — o original de terceiros não foi tocado |
| `main.tex` | **esqueleto real**: as seis seções, um frame-stub por subseção do §3 (`TODO n.n`), os quatro pontos de transição do §2 como comentário, e a série B demonstrada (B0 índice clicável + B1 completo replicável) |
| `template_showcase.tex` | o `main.tex` demonstrativo original do Henrique, preservado por `git mv` como referência de sintaxe |
| `Makefile` | `make check` (valida 0 erros, para iterar) · `make all` (build completo com bibtex) · `make fast` · `make showcase` · `make clean` |
| **motor** | **`xelatex`** |
| build | `make check` → **OK, 42 páginas, 0 erros** (rodado nesta sessão) |

**Três coisas a saber antes de escrever conteúdo:**

1. **O número de slide da série B congela, não some.** Sob `\miniframesoff` o frame não registra
   ponto na barra, mas `\insertframenumber` para de incrementar em vez de ficar em branco. Então
   os rótulos "B1, B2…" têm de estar **no conteúdo do slide**, não no rodapé.
2. **Nenhum campo do título pode ficar vazio.** `\subtitle`, `\institute` e `\date` vazios
   quebram o template com *"There's no line here to end"*. O esqueleto já preenche os quatro.
3. **Confirmar a grafia do nome e do título** exatamente como na folha de rosto entregue — o
   esqueleto usa `Vitor Hugo` e um `TODO` como placeholders.

