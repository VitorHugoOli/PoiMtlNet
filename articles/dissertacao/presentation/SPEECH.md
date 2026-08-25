# SPEECH.md — o que dizer, slide a slide

> **Defesa · sexta, 28/08/2026, 10:00 · remota (Google Meet).** Gerado do `SLIDES.md`, que é a fonte da fala. **Nada aqui é novo**: é o mesmo texto, reorganizado para ser lido de relance em vez de lido inteiro.

> **Como usar.** Cada cartão tem quatro camadas, em ordem de urgência: **ABRE** (a primeira oração, para pegar o fio sem ler), **DIZER** (as superfícies de lei e os números que não podem sair errado), **NUNCA** (o que anula o slide se escapar), e a fala completa embaixo, para consulta.

> ⚠ **O relógio.** Os tempos abaixo são os do plano e somam **48:30**. A fala escrita tem 8.840 palavras, que a 140 palavras/minuto dão **~63 min** — contra o teto de **50 min** do Art. 23. Os dois números não fecham, e o ensaio é que decide qual vale. **Cronometre o fim de cada seção.**


---

## Marcas de tempo — leve estas seis

| seção | slides | fim previsto | **seu tempo real** |
|---|---|---:|---|
| **ABERTURA — a pergunta e o escopo** | S1–S7 | 5:00 | ____________ |
| **FUNDAMENTOS — dito uma vez** | S8–S16 | 11:00 | ____________ |
| **MTLnet — Cap. 3 (CBIC)** | S17–S25 | 16:30 | ____________ |
| **ST-MTLNet — Cap. 4 (CoUrb)** | S26–S32 | 22:30 | ____________ |
| **Check2HGI — Cap. 5 (MobiWac)** | S33–S49 | 43:00 | ____________ |
| **CONCLUSÃO — a resposta condicional** | S50–S55 | 48:30 | ____________ |


---



# ABERTURA — a pergunta e o escopo


## S1 · Multitask Learning for POI Classification and Prediction Tasks
`PDF p.1` · **20 s** · fim previsto **0:20**

> ### ▶ Bom dia. Meu nome é Vitor Hugo, e vou apresentar minha dissertação de mestrado, orientada pelo professor Fabrício Silva, no PPGCC da Universidade Federal de Viçosa.

**● COBRE** — INTRODUZ nada (capa)

**✕ NUNCA** — nenhum resultado na capa. · Nunca usar o título de um dos artigos como título da dissertação.

<sub>Bom dia. Meu nome é Vitor Hugo, e vou apresentar minha dissertação de mestrado, orientada pelo professor Fabrício Silva, no PPGCC da Universidade Federal de Viçosa. Um aviso de forma antes de começar: os slides estão em inglês e a fala é em português. Os números na tela são os do documento que a banca recebeu, sem conversão.</sub>

---

## S2 · Movement is regular, and services depend on that
`slide **1** · PDF p.3` · **45 s** · fim previsto **1:05**

> ### ▶ O ponto de partida é uma regularidade.

**● COBRE** — INTRODUZ o gancho: a regularidade da mobilidade e as aplicações

**# NÚMEROS** — pessoa em cerca de 93 por cento. Faço

**✕ NUNCA** — "pioneiro", "o primeiro". · Nunca apresentar os 93 por cento como teto de nada. · Nenhum particular do corpus. · Nenhum número nosso.

<sub>O ponto de partida é uma regularidade. Os rastros de mobilidade são ruidosos, mas o movimento humano é bastante regular: uma análise de entropia sobre rastros em larga escala estimou a previsibilidade potencial da próxima localização de uma pessoa em cerca de 93 por cento. Faço a ressalva na mesma frase, porque ela importa. Essa estimativa é sobre a próxima localização em resolução grossa, e ela não é teto para nenhuma métrica que eu vou reportar aqui. O que ela mostra é que existe regularidade aprendível. As pessoas voltam a um conjunto pequeno de lugares, e fazem viagens longas com menos frequência. E antecipar o que e o onde da próxima visita é o que sustenta recomendação, navegação, planejamento de transporte e alocação de recursos por área.</sub>

---

## S3 · The ground: check-ins, mobility, and what joint training promises
`slide **2** · PDF p.4` · **55 s** · fim previsto **2:00**

> ### ▶ Duas palavras e um risco, antes da pergunta. Uma rede social baseada em localização é uma plataforma em que as pessoas registram os lugares por onde passam.

**● COBRE** — INTRODUZ LBSN e check-in (chão didático) · INTRODUZ a promessa operacional do aprendizado multitarefa · gloss de transferência negativa (a definição, Def. 2.12, é INTRODUZ em S9)

**✕ NUNCA** — nenhum particular do corpus. · Não definir formalmente transferência negativa aqui; · a Def. · 2.12 é de S9.

<sub>Duas palavras e um risco, antes da pergunta. Uma rede social baseada em localização é uma plataforma em que as pessoas registram os lugares por onde passam. O registro é o check-in, e ele liga um usuário, um ponto de interesse e um instante. É esse detalhe geográfico e temporal que sustenta o estudo de cidades a partir de dados, e a área que estuda como as pessoas se movem pela cidade é mobilidade humana. Agora o aprendizado multitarefa: treinar tarefas relacionadas juntas, para que compartilhem informação. Aqui o apelo é operacional. Um modelo para manter, uma passagem, as duas predições, em vez de dois modelos dedicados. Mas treino conjunto não garante predição melhor. Parâmetros compartilhados podem prejudicar uma tarefa, e essa falha tem nome: transferência negativa. A definição formal dela fica para a próxima seção. Por ora basta o nome, porque é ele que a pergunta seguinte carrega.</sub>

---

## S4 · The question, and the answer in one line
`slide **3** · PDF p.5` · **60 s** · fim previsto **3:00**

> ### ▶ A pergunta da dissertação, literalmente: o aprendizado multitarefa ajuda a predição de pontos de interesse, próxima categoria e próxima região, e de que depende a resposta?

**● DIZER EXATO**
- próxima região
- permanece dentro da margem de dois pontos
- próxima categoria
- supera na Flórida
- equivalentes a zero dentro de meio ponto

**✕ NUNCA** — "empata", "matches", "ties", "semelhante", "a par", "em todos". · Nunca aplicar a margem de dois pontos ao eixo de categoria, nem o meio ponto ao eixo de região. · Nunca chamar as quatro células dentro da margem de empates. · Nunca ler a frase do Resumo entregue.

<sub>A pergunta da dissertação, literalmente: o aprendizado multitarefa ajuda a predição de pontos de interesse, próxima categoria e próxima região, e de que depende a resposta? Ela vem com uma restrição que vale para tudo o que vem depois: um artefato treinado tem de produzir as duas saídas numa passagem só. E a resposta eu dou agora, no minuto três, e não no fim, porque daqui em diante cada slide é resposta a uma pergunta que eu já fiz. Na previsão da **próxima região**, o modelo conjunto **supera** os dedicados no **Texas** e na **Califórnia**, e nos outros quatro conjuntos **permanece dentro da margem de dois pontos**, registrada antes de qualquer resultado ser lido: quatro déficits pequenos, com a direção declarada, nenhum empate. Na **próxima categoria**, **supera na Flórida**, e as cinco diferenças restantes são **equivalentes a zero dentro de meio ponto**. As quatro células dentro da margem eu enumero uma a uma na Seção 5, com intervalo e com teste.</sub>

---

## S5 · What is predicted, and what is not
`slide **4** · PDF p.6` · **40 s** · fim previsto **3:40**

> ### ▶ O escopo, no positivo. Eu predigo duas propriedades da próxima visita: a categoria, que é o tipo do lugar, e a região, que é a unidade administrativa em escala de bairro onde a visita acontece.

**● COBRE** — INTRODUZ as três tarefas (Defs. 2.6, 2.7, 2.8) e a exclusão do próximo lugar (Def. 2.9) · RETOMA a restrição de modelo único

**✕ NUNCA** — "prediz o próximo POI". · Nenhum particular do corpus: nem o número de classes, nem setor censitário, nem mahalle, nem nome de estado (isso é de S10).

<sub>O escopo, no positivo. Eu predigo duas propriedades da próxima visita: a categoria, que é o tipo do lugar, e a região, que é a unidade administrativa em escala de bairro onde a visita acontece. O próximo lugar exato eu não predigo. Ele está definido no Capítulo 2 justamente para ser excluído, e nenhum capítulo reporta resultado para ele. Existe ainda uma terceira tarefa no trabalho, a classificação de categoria, que é estática: lê a representação de um lugar, e não um histórico. Ela é metade do par nos dois primeiros estudos.</sub>

---

## S6 · Three studies, in sequence
`slide **5** · PDF p.7` · **40 s** · fim previsto **4:20**

> ### ▶ A dissertação é uma coletânea de três artigos, e a ordem deles é o argumento.

**● COBRE** — INTRODUZ o arco e os três capítulos (título de capítulo, veículo, ano, autoria) · INTRODUZ a armadilha de nomenclatura *"Next-POI Prediction" = próxima categoria*

**✕ NUNCA** — ampliar o crédito de autoria no Cap. · 4 além do que o texto entregue declara. · Nenhum resultado dos três estudos aqui.

<sub>A dissertação é uma coletânea de três artigos, e a ordem deles é o argumento. O texto diz assim: o primeiro reporta um resultado negativo, o segundo identifica o gargalo principal, e o terceiro testa a solução que sai daí. Cada estudo estreita a explicação que a evidência sustenta, e é por isso que a progressão é ela própria parte da contribuição. Veículo, ano e autoria estão na tela. E um aviso de nome antes de seguir, porque ele evita confusão nas duas seções seguintes: o título do Capítulo 3 diz *Next-POI Prediction*, e nos Capítulos 3 e 4 essa expressão quer dizer próxima categoria, não próximo lugar.</sub>

---

## S7 · The contribution, in one block
`slide **6** · PDF p.8` · **40 s** · fim previsto **5:00**

> ### ▶ A contribuição, em duas metades, e eu volto a esta tela no fim com as mesmas palavras.

**● COBRE** — INTRODUZ a contribuição una (a segunda aparição, com redação idêntica, é o slide de fechamento da Seção 6)

**✕ NUNCA** — "MTL funciona" sem condição. · Nenhum número aqui, e em particular nenhuma contagem de parâmetros: a razão entre o modelo conjunto e os dois dedicados não foi re-medida (PLANO §8 regra 9).

<sub>A contribuição, em duas metades, e eu volto a esta tela no fim com as mesmas palavras. A metade prática: um modelo, uma passagem, duas predições. O ganho é operacional, não computacional. O modelo conjunto é o artefato maior, e uma passagem por ele custa mais do que rodar os dois dedicados; o que diminui é o número de modelos para treinar e manter. A metade científica: o que eu entrego são condições, não um sim universal. A representação de entrada e a topologia de compartilhamento decidem se o multitarefa ajuda nestas tarefas. É por isso que um resultado nulo com embedding por lugar e compartilhamento rígido não contradiz um resultado positivo com representação em nível de check-in e outra forma de compartilhar.</sub>

---


# FUNDAMENTOS — dito uma vez


## S8 · One lineage, one idea
`slide **7** · PDF p.10` · **55 s** · fim previsto **5:55**

> ### ▶ Uma ideia só cobre três métodos desta dissertação, então é melhor dizê-la agora do que três vezes.

**● COBRE** — INTRODUZ a ideia infomax · INTRODUZ a linhagem de modelos (Tab. 1) e o diagrama de níveis · *(o nome Check2HGI aparece na Tab. 1 como linha do mapa; o artefato é INTRODUZ em 5.2, e MTLnet em 3.2)*

**✕ NUNCA** — nenhum resultado, nenhum número de capítulo. · Não explicar FiLM aqui. · Nunca Space2Vec nem POI2Vec como componentes deste trabalho.

<sub>Uma ideia só cobre três métodos desta dissertação, então é melhor dizê-la agora do que três vezes. A ideia infomax, nas palavras do próprio capítulo: o modelo aprende vetores úteis sendo obrigado a distinguir um pareamento verdadeiro de um pareamento corrompido, e não precisa de rótulo nenhum para isso, porque os próprios dados dizem qual é o verdadeiro. O DGI faz essa comparação entre um nó e um resumo do grafo. O HGI estende o mesmo objetivo por uma hierarquia de lugar, região e cidade. E o Check2HGI, do Capítulo 5, acrescenta um quarto nível abaixo do lugar, que é o check-in. O mecanismo de cada um fica com o capítulo dono dele. A tabela na tela é a Tabela 1 da dissertação, e ela é o mapa da fala inteira: a cada seção eu volto a ela e digo em que linha eu estou.</sub>

---

## S9 · How two tasks share a model, and how that fails
`slide **8** · PDF p.11` · **40 s** · fim previsto **6:35**

> ### ▶ Duas definições e um critério. Compartilhamento rígido é a topologia em que todas as tarefas atravessam um mesmo tronco e só se separam na saída de cada uma.

**● COBRE** — INTRODUZ compartilhamento rígido (Def. 2.10) · INTRODUZ transferência negativa (Def. 2.12) · INTRODUZ o critério declarado para um balanceador · RETOMA a promessa operacional de 1.2

**✕ NUNCA** — nenhuma afirmação de otimalidade de Pareto sobre os nossos modelos: o Cap. · 2 recusa a afirmação explicitamente. · Nenhum resultado, nenhum número.

<sub>Duas definições e um critério. Compartilhamento rígido é a topologia em que todas as tarefas atravessam um mesmo tronco e só se separam na saída de cada uma. Transferência negativa é o desfecho que se teme: o treino conjunto deixa uma tarefa pior do que o modelo dedicado dela deixaria. O critério está declarado no Capítulo 2, e eu vou cobrá-lo mais adiante: um método de balanceamento só é útil se superar uma ponderação fixa bem ajustada. Guardem essa frase. É ela que decide o que eu posso e o que eu não posso afirmar sobre o balanceador na Seção 3.</sub>

---

## S10 · The evidence base: six datasets, said once
`slide **9** · PDF p.12` · **55 s** · fim previsto **7:30**

> ### ▶ Esta é a base de evidência inteira, dita uma vez só.

**● COBRE** — INTRODUZ a base de evidência (Tab. 8): Gowalla e Istanbul · INTRODUZ as sete categorias · INTRODUZ a região como unidade nomeada (census tract, *mahalle*)

**✕ NUNCA** — "superconjunto" para a Flórida. · Não há evidência de contenção entre as duas extrações. · Nenhum resultado de nenhum capítulo.

<sub>Esta é a base de evidência inteira, dita uma vez só. Cinco estados do Gowalla e Istambul, do Massive-STEPS, e a ordem da tabela é a do documento, por número de regiões. As sete categorias são as mesmas nos três estudos: Community, Entertainment, Food, Nightlife, Outdoors, Shopping e Travel. Região é o setor censitário nos cinco conjuntos americanos e o *mahalle* em Istambul. Os dois particionam a cidade em escala de bairro, e não são o mesmo tipo de objeto: um é unidade de medida, o outro é unidade de governo. A última coluna diz qual capítulo usou qual conjunto. E um aviso que evita uma pergunta depois: a Flórida aparece duas vezes nesta dissertação, e são duas extrações. Novecentos e noventa mil, quinhentos e dezoito check-ins nos Capítulos 3 e 4; um milhão, quatrocentos e sete mil e trinta e quatro no Capítulo 5. Eu não afirmo contenção entre as duas.</sub>

---

## S11 · The metric all three studies share
`slide **10** · PDF p.13` · **35 s** · fim previsto **8:05**

> ### ▶ A métrica de categoria dos três estudos é a macro-F1: a média das F1 por categoria, com cada categoria pesando igual.

**● COBRE** — INTRODUZ macro-F1 · INTRODUZ o piso de classe majoritária

**✕ NUNCA** — chamar de "macro-F1" os valores impressos dos Caps. · 3 e 4 (são uma F1 por categoria). · Nunca ler a coluna Majority da Tab. · 8 como se fosse a macro-F1 do preditor de classe majoritária: uma é a fração de rótulos na classe mais comum, a outra é o resultado de um preditor.

<sub>A métrica de categoria dos três estudos é a macro-F1: a média das F1 por categoria, com cada categoria pesando igual. A razão é a distribuição. Food é cerca de um terço dos check-ins num estado representativo, e uma acurácia simples esconderia o desempenho nas classes menores. Ela também tem um custo, e eu digo qual: a macro-F1 não mostra que classe melhorou, e pode ficar baixa mesmo com acurácia alta. Duas coisas que costumam ser perguntadas, e eu já respondo. A perda não é reponderada, é entropia cruzada sem peso. E toda macro-F1 que eu disser vem com o ponto de referência dela, que é o piso de classe majoritária.</sub>

---

## S12 · The protocol of the first two studies, and two names that change
`slide **11** · PDF p.14` · **50 s** · fim previsto **8:55**

> ### ▶ O protocolo dos dois primeiros estudos, e ele é diferente do terceiro.

**● COBRE** — INTRODUZ o protocolo dos dois primeiros estudos · INTRODUZ a lei dos verbos · INTRODUZ a armadilha do par de tarefas e a armadilha da convenção métrica · RETOMA a armadilha de nomenclatura de 1.5

**✕ NUNCA** — "as mesmas janelas". · O Cap. · 5 usa janelas deslizantes sobrepostas, com passo 1, e os Caps. · 3 e 4 usaram janelas não sobrepostas.

<sub>O protocolo dos dois primeiros estudos, e ele é diferente do terceiro. Validação cruzada de cinco partições, estratificada por amostra: os check-ins de um mesmo usuário podem cair dos dois lados da divisão. Orçamento cheio de épocas, sem parada antecipada, e cada tarefa lida na época de melhor validação dela. Médias e desvios entre as cinco partições, sem teste de significância. Daí sai a lei dos verbos que eu obedeço a fala inteira: *supera* fica reservado para teste pareado de superioridade, e os Capítulos 3 e 4 não têm teste, então eles reportam diferenças, não veredito. Faltam duas armadilhas de nome. A primeira: o par de tarefas muda. Nos dois primeiros é estática mais próxima categoria; no terceiro é próxima categoria mais próxima região. A segunda: a convenção métrica muda. Os Capítulos 3 e 4 imprimem uma F1 por categoria, e o Capítulo 5 reporta macro-F1, um número só. Não são a mesma escala, e toda tabela que eu reproduzir vai levar esse carimbo.</sub>

---

## S13 · Related work: POI prediction and multitask learning
`slide **12** · PDF p.15` · **35 s** · fim previsto **9:30**

> ### ▶ Duas telas de trabalho relacionado, e elas são o chão comum dos dois primeiros estudos.

**● COBRE** — INTRODUZ o contexto de POI e MTL, o chão comum dos Caps. 3 e 4

**✕ NUNCA** — nenhum resultado, nenhum número. · Nenhuma afirmação de ineditismo aqui: ela é de S14, e vem escopada.

<sub>Duas telas de trabalho relacionado, e elas são o chão comum dos dois primeiros estudos. A tarefa dominante da área é o próximo lugar exato. A linha vai dos recorrentes, ST-RNN, DeepMove, HST-LSTM, Flashback, para os de atenção, STAN, GeoSAN, GETNext. Todos eles predizem o lugar exato, e por isso nenhum é linha de base direta para os alvos que eu estudo. O par que os dois primeiros estudos atacam é outro: classificação de categoria e previsão da próxima categoria. E, em mobilidade, o multitarefa foi usado quase inteiramente a serviço do próximo lugar. O MCARNN prevê atividade e lugar juntos; o CSLSL prevê em cascata quando, o quê e onde; o iMTL e o HAMTL seguem a mesma direção. O TME é a exceção que puxa para o outro lado, com anotação semântica estática de ponto de interesse.</sub>

---

## S14 · The axis that separates this work
`slide **13** · PDF p.16` · **30 s** · fim previsto **10:00**

> ### ▶ O eixo que separa este trabalho da literatura não é a tarefa, é o papel dela.

**● COBRE** — INTRODUZ o eixo meio × fim · INTRODUZ o mapa de onde saem os métodos externos

**✕ NUNCA** — afirmação de ineditismo mais forte do que a do texto. · A redação entregue é escopada a "entre os trabalhos revisados nesta dissertação", e a fala mantém o escopo. · Nenhum resultado, nenhum número.

<sub>O eixo que separa este trabalho da literatura não é a tarefa, é o papel dela. De um lado, categoria e região como **meio** para chegar ao próximo lugar: o HMT-GRN usa a região prevista para estreitar a busca pelo lugar, o CatDM usa a categoria prevista para reduzir o conjunto de candidatos. Do outro lado, como **fim**: o DRRGNN prevê região de atividade, o POI-RGNN prevê a próxima categoria. E a frase do texto, escopada como ela está escrita: entre os trabalhos revisados nesta dissertação, nenhum trata próxima categoria e próxima região como alvos finais de igual estatuto num modelo conjunto que não prediz também o próximo lugar. O rodapé é o mapa de onde saem os métodos externos que vão aparecer nas tabelas.</sub>

---

## S15 · Related work in representation: the line this work stands on
`slide **14** · PDF p.17` · **40 s** · fim previsto **10:40**

> ### ▶ Esta é a base mais importante da dissertação, e por isso ela tem tela própria.

**● COBRE** — INTRODUZ a linhagem de representações, no nível de o que cada degrau resolve

**✕ NUNCA** — o mecanismo do DGI ou do HGI aqui: eles pertencem a 3.2A e 4.1A. · Nunca Space2Vec nem POI2Vec como componentes deste trabalho. · Nenhum resultado, nenhum número.

<sub>Esta é a base mais importante da dissertação, e por isso ela tem tela própria. É uma escada. Começa no identificador one-hot, que marca um lugar por posição e não codifica relação nenhuma. Sobe para as representações distribuídas, skip-gram, DeepWalk, node2vec, em que a geometria do vetor reflete a relação que está nos dados. Sobe de novo para as redes de grafo, GCN, GAT, GraphSAGE, que aprendem essa relação por agregação de vizinhança. Depois vem o DGI, que aplica o objetivo infomax entre um nó e um resumo do grafo, e o HGI, que estende o mesmo objetivo pela hierarquia de lugar, região e cidade. O mecanismo de cada um fica com o capítulo dono: o DGI no Capítulo 3, o HGI no Capítulo 4. Aqui é só o mapa, para a linhagem ficar legível quando eu chegar ao Check2HGI.</sub>

---

## S16 · With the vocabulary fixed, each study says only what it changed
`slide **15** · PDF p.18` · **20 s** · fim previsto **11:00**

> ### ▶ Com o vocabulário, os dados e a métrica fixados uma única vez, cada estudo agora só precisa dizer o que mudou, e cada um nomeia a sua própria convenção quando chegar a hora.

**● COBRE** — RETOMA vocabulário, dados e métrica

**✕ NUNCA** — que as "regras de decisão" ficaram fixadas aqui. · O protocolo estatístico entra só em 5.4, e prometer o fechamento que a Seção 2 não entrega é o defeito que a reescrita de 2026-08-21 corrigiu.

<sub>Com o vocabulário, os dados e a métrica fixados uma única vez, cada estudo agora só precisa dizer o que mudou, e cada um nomeia a sua própria convenção quando chegar a hora. E o par de tarefas dos dois primeiros estudos não é o do terceiro; quando ele mudar, eu aviso. O primeiro usou o que a literatura oferecia: um vetor por lugar e um tronco compartilhado.</sub>

---


# MTLnet — Cap. 3 (CBIC)


## S17 · One static task, one sequential task
`slide **16** · PDF p.20` · **30 s** · fim previsto **11:30**

> ### ▶ O par do primeiro estudo. Uma tarefa estática: ler a representação de um lugar e prever a categoria dele.

**● COBRE** — INTRODUZ a dicotomia estática × sequencial que o Cap. 3 põe à prova · RETOMA as Defs. 2.6 e 2.7 e a armadilha de nome de S12

**✕ NUNCA** — "prediz o próximo POI" ou "próximo lugar". · Neste capítulo Next-POI Prediction é a próxima categoria, e o carimbo está na tela.

<sub>O par do primeiro estudo. Uma tarefa estática: ler a representação de um lugar e prever a categoria dele. E uma tarefa sequencial: ler um histórico de nove visitas e prever a categoria da próxima. Na superfície elas são relacionadas, porque saem dos mesmos dados e do mesmo espaço de sete categorias. Na natureza, não: uma depende das características intrínsecas de um lugar, a outra depende de ordem temporal e de transição. O capítulo entra no experimento com uma hipótese declarada, e ela é negativa: essa diferença é grande o bastante para limitar o que um tronco compartilhado consegue fazer pelas duas.</sub>

---

## S18 · MTLnet
`slide **17** · PDF p.21` · **45 s** · fim previsto **12:15**

> ### ▶ Esta é a arquitetura, o MTLnet. Vale guardar a figura, porque o Capítulo 4 não vai alterar uma linha dela.

**● COBRE** — INTRODUZ MTLnet · INTRODUZ FiLM

**# NÚMEROS** — compartilhamento rígido da Definição 2.10. E no fim

**✕ NUNCA** — nenhum número do Cap. · 3 ao lado de um do Cap. · Nunca "backbone": o nome é tronco compartilhado.

<sub>Esta é a arquitetura, o MTLnet. Vale guardar a figura, porque o Capítulo 4 não vai alterar uma linha dela. Cada tarefa entra por um encoder próprio, um MLP. Vem então a modulação FiLM, e ela cabe numa cláusula: um vetor de identidade de tarefa gera uma escala e um deslocamento, aplicados às features antes da parte compartilhada, de modo que as duas tarefas leiam os mesmos parâmetros sob escalas diferentes. Depois vem o tronco de blocos residuais, que é o compartilhamento rígido da Definição 2.10. E no fim duas saídas, uma por tarefa. O rodapé é o detalhe que vai importar daqui a dois slides: o capítulo declara os parâmetros em dois conjuntos disjuntos, os compartilhados e os específicos de tarefa. É sobre o primeiro conjunto que um balanceador de gradientes age.</sub>

---

## S19 · DGI: how it works | why it was used
`slide **18** · PDF p.22` · **45 s** · fim previsto **13:00**

> ### ▶ O primeiro mecanismo, e ele responde uma pergunta que costuma vir.

**● COBRE** — INTRODUZ o mecanismo do DGI · RETOMA a ideia infomax de 2.1 e o degrau do DGI em 2.7

**✕ NUNCA** — "one-hot da própria categoria" como atributo de nó. · Nunca "coocorrência": este canal não existe no Cap. · Nunca "o DGI não vaza" (a formulação correta está no slide B4-LEAK).

<sub>O primeiro mecanismo, e ele responde uma pergunta que costuma vir. O DGI roda aqui sobre um grafo de Delaunay dos lugares da área, com pesos de aresta que decaem com a distância geodésica entre dois lugares, por uma função logarítmica dela. Uma camada de atenção de grafo produz um vetor de 64 dimensões por lugar. O objetivo de treino é o infomax da Seção 2: distinguir o grafo real de uma versão com as features dos nós embaralhadas, contra um resumo global do grafo. Agora o atributo de nó, que é onde eu quero ser exato, porque a nota de rodapé do capítulo entregue registra isso. A implementação liberada alimenta a rede com a média dos one-hots dos vizinhos do lugar, com o vetor do próprio lugar excluído. A distinção muda como o embedding deve ser lido: a entrada descreve a vizinhança, então a tarefa estática que ele sustenta é homofilia espacial, e não recuperação do rótulo do próprio lugar. O que sai daí é um vetor por lugar. Toda visita àquele lugar entra no modelo com o mesmo vetor, e essa frase é a que o Capítulo 5 vai atacar.</sub>

---

## S20 · Setup, and the protocol declared
`slide **19** · PDF p.23` · **30 s** · fim previsto **13:30**

> ### ▶ O setup em três linhas, e a terceira é a autodeclaração de protocolo que eu prometi na Seção 2.

**● COBRE** — RETOMA os dados de 2.3, as sete categorias de 2.3 e o protocolo de 2.5

**✕ NUNCA** — "as mesmas janelas" do Cap. · 5 (lá são sobrepostas, passo 1). · Nunca chamar a média por categoria deste capítulo de macro-F1.

<sub>O setup em três linhas, e a terceira é a autodeclaração de protocolo que eu prometi na Seção 2. Os dados são a Flórida do Gowalla, vinte mil trezentos e um usuários, sessenta e cinco mil e nove lugares, novecentos e noventa mil quinhentos e dezoito check-ins, nas mesmas sete categorias. As sequências são janelas não sobrepostas de nove visitas, e quem tem menos de cinco visitas sai. O protocolo é o estratificado por amostra: cinco partições, uma semente, orçamento cheio de épocas, cada tarefa lida na melhor época de validação dela, e média com desvio entre as cinco partições. Sem teste de significância. É por isso que este capítulo reporta diferenças, e não veredito, e é por isso que eu não vou usar o verbo supera em nenhum slide desta seção.</sub>

---

## S21 · Two losses, one set of parameters
`slide **20** · PDF p.24` · **40 s** · fim previsto **14:10**

> ### ▶ Antes de eu nomear o otimizador, o problema que ele existe para resolver.

**● COBRE** — INTRODUZ o problema multiobjetivo, a dominância e a fronteira de Pareto, e as duas classes de método de balanceamento

**✕ NUNCA** — nenhuma afirmação de Pareto sobre os nossos modelos. · Nenhum formalismo do zoo de balanceadores na tela (§8 regra 16): os nomes entram como lista, sem equação.

<sub>Antes de eu nomear o otimizador, o problema que ele existe para resolver. São duas perdas e um único conjunto de parâmetros compartilhados, e entre duas soluções não há ordem total: uma pode ser melhor numa tarefa e pior na outra, e as duas ficam incomparáveis. Escrever a soma ponderada das perdas não remove essa natureza multiobjetivo. Daí vem o vocabulário: uma configuração domina outra no sentido de Pareto quando não é pior em nenhuma perda e é melhor em pelo menos uma; ela é Pareto-ótima quando nenhuma outra a domina; e o conjunto dos vetores de perda dessas configurações é a fronteira de Pareto. A área respondeu a isso com uma família inteira de métodos, e a família se divide em duas classes: os que fixam os pesos das perdas e os que mudam a direção da atualização. Uma ressalva que é do Capítulo 2 e que eu repito de propósito: esta dissertação não reivindica nenhuma propriedade de Pareto para os modelos dela.</sub>

---

## S22 · Nash-MTL, and what the chapter may claim about it
`slide **21** · PDF p.25` · **40 s** · fim previsto **14:50**

> ### ▶ O Nash-MTL cai na segunda classe.

**● COBRE** — INTRODUZ Nash-MTL · RETOMA o critério de 2.2 e as duas classes de S21

**✕ NUNCA** — nenhuma afirmação de Pareto sobre os nossos modelos. · Nunca apresentar a adoção do Nash como posição atual da dissertação.

<sub>O Nash-MTL cai na segunda classe. Ele muda a direção da atualização, tratando a combinação dos gradientes como uma barganha cooperativa entre as tarefas: cada tarefa tem uma utilidade, que é a redução da perda dela, e a direção escolhida é a que maximiza o produto dessas utilidades, o que evita que uma domine a outra. A garantia é convergência para um ponto Pareto-estacionário, que é condição necessária e não suficiente para otimalidade de Pareto; a otimalidade exigiria uma hipótese de convexidade que uma rede profunda não satisfaz. Agora a parte que eu preciso dizer com cuidado. O Capítulo 3 adotou o Nash porque, na comparação dele, contra o PCGrad e contra não usar balanceador nenhum, ele deu a menor perda multitarefa combinada. Isso é conclusão do tempo dele, enfraquecida depois por um achado sobre a implementação do otimizador, e o Capítulo 5 não se apoia nisso. O critério da Seção 2 continua de pé: um balanceador só é útil se melhorar sobre uma ponderação fixa bem ajustada.</sub>

---

## S23 · The null result, shown rather than asserted
`slide **22** · PDF p.26` · **55 s** · fim previsto **15:45**

> ### ▶ O resultado. Eu prefiro mostrá-lo a afirmá-lo, então são as duas tabelas do capítulo, reduzidas ao bloco de F1.

**● COBRE** — INTRODUZ o resultado nulo do Cap. 3 · RETOMA o mapa de métodos externos de 2.6

**✕ NUNCA** — "ambas as baselines externas batidas em absoluto" (vale só na tarefa estática; · na sequencial o MHA+PE lidera Community, Food e Shopping). · Nunca "supera": não há teste pareado neste capítulo. · Nenhum número do Cap.

<sub>O resultado. Eu prefiro mostrá-lo a afirmá-lo, então são as duas tabelas do capítulo, reduzidas ao bloco de F1. À esquerda, a tarefa estática: os nossos dois modelos ficam acima da HMRM em todas as categorias. À direita, a tarefa sequencial, e é aqui que está o ponto: as lideranças se dividem. O MHA+PE fica com a melhor F1 em Community, Food e Shopping; o nosso multitarefa, em Nightlife e Travel; o de tarefa única, em Entertainment e Outdoors. E a comparação que interessa é entre as nossas duas colunas, que é a comparação entre multitarefa e dedicado. A conclusão é a do próprio capítulo, e está na tela: largamente comparáveis, sem vantagem clara ou consistente para o arranjo multitarefa nestes experimentos. Boa parte dessas diferenças cai dentro do desvio padrão entre partições. Repito o carimbo, porque ele evita a comparação errada mais tarde: isto é F1 por categoria, a convenção dos Capítulos 3 e 4, e não é a macro-F1 do Capítulo 5.</sub>

---

## S24 · A null with three suspects
`slide **23** · PDF p.27` · **25 s** · fim previsto **16:10**

> ### ▶ É aqui que o capítulo deixa de ser um resultado negativo e vira um programa de trabalho, porque ele nomeia três suspeitos, e não um.

**● COBRE** — INTRODUZ a bifurcação de três hipóteses

**✕ NUNCA** — transferência negativa como algo observado. · Nunca dar a um dos três suspeitos precedência que o capítulo não dá.

<sub>É aqui que o capítulo deixa de ser um resultado negativo e vira um programa de trabalho, porque ele nomeia três suspeitos, e não um. Dissimilaridade das tarefas: o tronco compartilhado teria sido forçado a uma representação de compromisso, não especializada para nenhuma das duas. Insuficiência da representação: ela não seria rica o bastante para codificar propriedade semântica e dinâmica sequencial ao mesmo tempo. Rigidez da topologia: um único bloco compartilhado seria restritivo demais. Uma precisão sobre o rodapé, que eu faço questão de dizer: a transferência negativa foi hipotetizada aqui, não foi observada.</sub>

---

## S25 · A null with three suspects does not close the investigation
`slide **24** · PDF p.28` · **20 s** · fim previsto **16:30**

> ### ▶ Um nulo com três suspeitos não encerra a investigação: ele desenha o próximo experimento.

**● COBRE** — RETOMA a bifurcação de 3.5

**✕ NUNCA** — que o próximo capítulo responde os três suspeitos. · Ele condena um.

<sub>Um nulo com três suspeitos não encerra a investigação: ele desenha o próximo experimento. Congelar a arquitetura e mover apenas a entrada.</sub>

---


# ST-MTLNet — Cap. 4 (CoUrb)


## S26 · Architecture or representation?
`slide **25** · PDF p.30` · **40 s** · fim previsto **17:10**

> ### ▶ O segundo estudo pega a pergunta herdada e a transforma em experimento controlado.

**● COBRE** — INTRODUZ o desenho controlado do Cap. 4 · RETOMA MTLnet e FiLM de 3.2, e Nash-MTL de 3.3B

**✕ NUNCA** — ampliar crédito de autoria. · A linha do divisor é a redação da própria Introdução entregue, e não se acrescenta nada a ela.

<sub>O segundo estudo pega a pergunta herdada e a transforma em experimento controlado. O gargalo é a representação, ou é a topologia de compartilhamento? Para separar as duas, ele mantém o MTLnet sem alterar uma linha: o mesmo tronco, a mesma modulação FiLM, o mesmo balanceador de gradientes, os mesmos hiperparâmetros. Só a entrada se move. A entrada antiga é o embedding monolítico de 64 dimensões do DGI. A entrada nova é a concatenação de três codificadores independentes, um espacial, um temporal e um categórico, de 64 dimensões cada, o que dá 192. E os encoders de tarefa projetam qualquer entrada para a mesma largura latente de 256 nos dois braços. É esse congelamento que faz o resultado ser diagnóstico, e não apenas melhor.</sub>

---

## S27 · HGI: how it works | why it was used
`slide **27** · PDF p.32` · **55 s** · fim previsto **18:05**

> ### ▶ O segundo mecanismo. É o conceito que sustenta o resto da dissertação, então eu vou com calma.

**● COBRE** — INTRODUZ o mecanismo do HGI · RETOMA a ideia infomax de 2.1 e o degrau do HGI em 2.7

**✕ NUNCA** — introduzir o Check2HGI aqui, que é do Cap. · Nunca Space2Vec nem POI2Vec como componentes deste trabalho.

<sub>O segundo mecanismo. É o conceito que sustenta o resto da dissertação, então eu vou com calma. O HGI monta uma hierarquia de três níveis: lugar, região, cidade. Um codificador de categoria pré-treinado dá as features iniciais dos lugares; uma camada de convolução sobre um grafo de Delaunay da área acrescenta contexto espacial a cada um; uma atenção multi-cabeça agrega os embeddings dos lugares de uma região; e uma soma ponderada por área sobre as regiões produz um embedding de cidade. O que se maximiza é a informação mútua entre dois níveis adjacentes dessa hierarquia. A peça que faz isso é um discriminador bilinear, que combina dois embeddings por uma matriz aprendida e passa o resultado por uma função logística, e a perda premia pontuação alta para um par verdadeiro e baixa para um par falso. Nenhum rótulo de tarefa final entra nessa comparação. Há uma consequência do desenho que vai importar duas vezes mais adiante. O treino atualiza junto o codificador de lugar, a agregação e o codificador de região, e a pertinência a região ainda entra pelos pesos das arestas. Então a saída no nível de lugar não descreve o lugar isolado: ela já reflete a região a que o lugar pertence. E um limite que eu declaro junto: o HGI foi desenvolvido e avaliado para representação de região urbana, e esta dissertação reaproveita a saída de nível de lugar para predição sequencial, um uso que a avaliação original não cobre.</sub>

---

## S28 · Why these encoders
`slide **28** · PDF p.33` · **60 s** · fim previsto **19:05**

> ### ▶ Por que estes codificadores, e não outros quaisquer.

**● COBRE** — INTRODUZ SIREN, Sphere2Vec-M, Time2Vec e o canal categórico em duas fases

**✕ NUNCA** — Space2Vec ou POI2Vec como componentes deste trabalho. · Eles são arte prévia, e não estão no registro de termos.

<sub>Por que estes codificadores, e não outros quaisquer. O canal espacial existe porque o MTLnet codificava espaço só implicitamente, pela topologia do grafo, e nunca como coordenada contínua. O estudo compara dois com hipóteses diferentes: o SIREN, que modela uma função contínua das coordenadas normalizadas com ativações senoidais, e o Sphere2Vec-M, que é multiescala e opera direto em coordenadas esféricas, preservando propriedades de distância geodésica. Os dois são treinados com a mesma perda contrastiva sobre distância geográfica, com par abaixo de dez quilômetros como positivo e acima de setenta como negativo, e é isso que faz a comparação isolar a arquitetura. O canal temporal existe porque o MTLnet não tinha representação temporal nenhuma, e o Time2Vec combina um termo linear, de tendência global, com termos senoidais, para os padrões cíclicos de hora do dia e dia da semana. O canal categórico existe porque o DGI codificava categoria pela estrutura do grafo, sem capturar relação hierárquica ou regional entre elas, e ele vem em duas fases. Primeiro um codificador de lugar, que aprende coocorrência entre categorias a partir de caminhadas aleatórias sobre o grafo espacial, com amostragem negativa e um termo que amarra cada classe fina à categoria de topo dela. Depois o HGI, que acrescenta a hierarquia regional sobre esse resultado.</sub>

---

## S29 · The caveat, then the number
`slide **29** · PDF p.34` · **45 s** · fim previsto **19:50**

> ### ▶ Aqui a ordem importa mais que o número, então eu digo a ressalva primeiro, em uma cláusula, e sigo em frente.

**● COBRE** — INTRODUZ o resultado da tarefa estática do Cap. 4 e a ressalva de rótulo na entrada

**✕ NUNCA** — número antes da ressalva. · "macro-F1 subiu 20 a 22" (são médias de F1 por categoria). · A faixa sem dizer que é melhor-de-dois. · Deixar o ganho estático falar pela tarefa sequencial.

<sub>Aqui a ordem importa mais que o número, então eu digo a ressalva primeiro, em uma cláusula, e sigo em frente. Depois da publicação, nós estabelecemos que a entrada da tarefa estática deste capítulo contém o rótulo que ela prevê: a feature de tipo de local mapeia um-para-um nas sete categorias de topo. A acurácia reportada nessa tarefa mede essa consulta, e não inferência semântica aprendida. A consequência é direta e eu prefiro dizê-la eu mesmo: o ganho estático não diz nada sobre a tarefa sequencial. Dito isso, o número. Na tarefa estática a entrada decomposta lidera nas vinte e uma combinações de categoria e estado, com ganhos médios por estado de vinte vírgula dois a vinte e dois pontos percentuais. E eu declaro o que essa faixa é: é o melhor dos dois codificadores espaciais em cada combinação, não é nenhum dos dois sozinho.</sub>

---

## S30 · The diagnostic result is the sequential task
`slide **30** · PDF p.36` · **60 s** · fim previsto **20:50**

> ### ▶ Agora a tarefa que produz o diagnóstico, que é a sequencial, e a razão é uma só: o alvo dela nunca está na entrada.

**● COBRE** — INTRODUZ o resultado sequencial do Cap. 4

**✕ NUNCA** — "macro-F1". · Nunca "supera": este capítulo não tem teste pareado. · Nenhum número do Cap. · 5 nesta tela.

<sub>Agora a tarefa que produz o diagnóstico, que é a sequencial, e a razão é uma só: o alvo dela nunca está na entrada. Na tela está a Flórida, com os três modelos lado a lado; Califórnia e Texas eu tenho prontos se a banca quiser. O cenário aqui é mais heterogêneo do que na tarefa estática, e continua favorável à entrada decomposta. Contando o melhor dos dois codificadores espaciais por combinação, os modelos espaço-temporais ficam com a média mais alta em quinze das vinte e uma combinações de categoria e estado, e o MTLnet, com a entrada original, retém seis. Uma dessas seis o capítulo chama, nas palavras dele, de um empate técnico adicional: é Outdoors na Flórida, onde a média do MTLnet fica dois centésimos de ponto percentual acima da melhor variante, dentro de um desvio padrão. Os maiores ganhos estão em Food, em que as duas variantes ficam acima nos três estados, com melhoria consistente também em Shopping e Community. E o carimbo de novo: isto é F1 por categoria, não é macro-F1.</sub>

---

## S31 · What the decomposition moved, and where it did not
`slide **31** · PDF p.37` · **65 s** · fim previsto **21:55**

> ### ▶ Três limites, e eu ofereço os três antes que me peçam.

**● COBRE** — INTRODUZ os três limites declarados do Cap. 4

**✕ NUNCA** — "pareado em largura". · Deixar o ganho estático falar pela sequencial. · Ampliar crédito de autoria.

<sub>Três limites, e eu ofereço os três antes que me peçam. O primeiro é o Travel, e ele precisa de rótulo de tarefa, senão a sala se confunde: Travel na classificação de categoria melhora, Travel na próxima categoria não. Na tarefa sequencial o MTLnet mantém a liderança na Flórida e na Califórnia, e a razão está escrita no capítulo: movimento de longa distância é esparso, e a topologia de grafo preserva relação entre lugares geograficamente distantes melhor do que um codificador baseado em coordenada. O segundo limite é que não existe codificador espacial universalmente melhor. O SIREN se destaca mais na Flórida e na Califórnia, o Sphere2Vec-M no Texas, e a adequação depende de como os lugares se distribuem em cada território. O terceiro é o que eu esperaria que a banca perguntasse, então eu digo primeiro: a comparação não é pareada em largura. São 192 dimensões contra 64. O capítulo declara isso como limite e pede um controle de dimensão equalizada, e eu não vou defender o ponto: parte do ganho pode vir da largura. Junto com isso, os três componentes entram sempre juntos, então este capítulo não isola a contribuição de cada codificador.</sub>

---

## S32 · With the architecture fixed, the input moved the result
`slide **32** · PDF p.38` · **35 s** · fim previsto **22:30**

> ### ▶ Com a arquitetura fixa, a entrada moveu o resultado: a representação é o gargalo.

**● COBRE** — RETOMA o gargalo · INTRODUZ as três camadas que o Cap. 5 reconstrói (representação · topologia · protocolo)

**✕ NUNCA** — que a correção de um vazamento foi o pivô. · A direcionalidade das arestas entra em S37, como princípio de projeto, na redação do próprio Cap.

<sub>Com a arquitetura fixa, a entrada moveu o resultado: a representação é o gargalo. Mas o diagnóstico ainda é em nível de lugar, sob um protocolo que deixa o mesmo usuário dos dois lados da divisão. O terceiro estudo reconstrói as três camadas: representação, topologia e protocolo.</sub>

---


# Check2HGI — Cap. 5 (MobiWac)


## S33 · Three changes, each a consequence of the diagnosis
`slide **33** · PDF p.40` · **60 s** · fim previsto **23:30**

> ### ▶ As três mudanças do último estudo, e nenhuma delas é preferência minha: as três são consequência do diagnóstico do capítulo anterior.

**● DIZER EXATO**
- usuários disjuntos
- o par de tarefas muda

**✕ NUNCA** — creditar qualquer uma das três mudanças a uma correção de vazamento. · "Prevê o próximo lugar".

<sub>As três mudanças do último estudo, e nenhuma delas é preferência minha: as três são consequência do diagnóstico do capítulo anterior. A representação sai do nível de lugar para o nível de check-in, porque um vetor por lugar não distingue um almoço de quarta-feira de uma noite de sábado no mesmo lugar. A topologia sai do compartilhamento rígido para atenção cruzada entre fluxos por tarefa, com um caminho espacial privado na saída de região. E o protocolo sai do estratificado por amostra para validação cruzada com **usuários disjuntos**, quatro sementes, e testes fixados antes de qualquer resultado ser lido. Aqui eu cumpro o aviso que dei na Seção 2: **o par de tarefas muda**. Com uma entrada por visita, a classificação estática vira um par pouco natural, e o par passa a ser próxima categoria mais próxima região, dois alvos finais sequenciais. A restrição da abertura continua valendo: um artefato, uma passagem, duas respostas.</sub>

---

## S34 · Next region: the task, and why it is worth predicting
`slide **34** · PDF p.41` · **35 s** · fim previsto **24:05**

> ### ▶ Trabalho relacionado deste estudo, que os dois primeiros não têm, e a primeira metade é a tarefa nova.

**● COBRE** — INTRODUZ a tarefa de próxima região e as suas motivações · RETOMA o eixo meio × fim de 2.6 e as contagens de região da Tab. 8 (2.3)

**# NÚMEROS** — classes em Istambul a 8.501 na Califórnia: mais

**✕ NUNCA** — especulação sobre erro geográfico ou desempenho de serviço (§8 regra 16). · Afirmação de ineditismo mais forte que a entregue, que é escopada a "to our knowledge" e "underexplored".

<sub>Trabalho relacionado deste estudo, que os dois primeiros não têm, e a primeira metade é a tarefa nova. Próxima região é classificação sobre as regiões candidatas do conjunto, de 520 classes em Istambul a 8.501 na Califórnia: mais grossa que lugar não quer dizer mais fácil. Prever sobre uma partição do mapa é a formulação padrão em mobilidade, com célula de grade como alvo; aqui entra no lugar dela a unidade administrativa de bairro. E onde a área já modela várias granularidades, categoria e região aparecem como sinais auxiliares de um alvo principal de próximo lugar. Eu estudo o par como objeto. O escopo vai junto com a motivação: preparação em nível de bairro, e nenhum serviço construído ou avaliado aqui.</sub>

---

## S35 · Why a per-visit representation is new in this line
`slide **35** · PDF p.42` · **25 s** · fim previsto **24:30**

> ### ▶ Segunda metade: por que uma representação por visita é nova nesta linha.

**● COBRE** — INTRODUZ CTLE como a arte prévia mais próxima · RETOMA a escada de representações de 2.7

**✕ NUNCA** — que o CTLE foi superado aqui. · O número do CTLE fica em S45, e o que ele estabelece é uma ordenação entre famílias de representação.

<sub>Segunda metade: por que uma representação por visita é nova nesta linha. A arte prévia mais próxima é o CTLE, que também dá um vetor por visita, aprendido mascarando e reconstruindo partes da sequência de check-ins do usuário. A diferença é de construção. O CTLE é um modelo de sequência, um Transformer que lê a própria sequência; o Check2HGI continua um modelo de grafo, com a mesma hierarquia de lugar, região e cidade e o mesmo objetivo infomax, agora um nível mais fundo. E o CTLE pré-treina só sobre identificador de lugar e marca de tempo, então o vocabulário de categoria nunca entra no treino dele. A novidade que eu reivindico é a combinação.</sub>

---

## S36 · Check2HGI: a fourth level below the place
`slide **36** · PDF p.43` · **80 s** · fim previsto **25:50**

> ### ▶ O Check2HGI, e ele se apoia direto no HGI do capítulo anterior.

**● DIZER EXATO**
- um quarto nível abaixo do lugar
- o grafo nunca vê a próxima categoria nem a próxima região

**# NÚMEROS** — auxiliares pequenos, de pesos 0,3 e 0,1, e

**✕ NUNCA** — nenhum p-valor nesta subseção. · "Substrate" (palavra de repositório).

<sub>O Check2HGI, e ele se apoia direto no HGI do capítulo anterior. O HGI tinha três níveis: lugar, região e cidade. O Check2HGI acrescenta **um quarto nível abaixo do lugar**, que é o próprio check-in. As arestas ligam cada nível ao de cima, ligam lugares próximos no nível de lugar, e ligam os check-ins consecutivos de um mesmo usuário, com um peso que decai conforme o intervalo entre as visitas cresce. Duas visitas ao mesmo lugar se encontram pelo nó de lugar, um nível acima. O treino é o objetivo infomax da Seção 2, agora um nível mais fundo: cada vetor aprende a reconhecer a vizinhança verdadeira e a rejeitar uma embaralhada. Junto com ele vão dois termos auxiliares pequenos, de pesos 0,3 e 0,1, e nenhum dos dois usa rótulo. Este é o ponto que eu quero deixar assentado antes de qualquer resultado: **o grafo nunca vê a próxima categoria nem a próxima região**. E do grafo treinado saem duas tabelas: um vetor de 64 dimensões por visita, e um vetor por região. São essas duas tabelas que o modelo da próxima tela vai ler.</sub>

---

## S37 · What each visit contributes
`slide **37** · PDF p.44` · **70 s** · fim previsto **27:00**

> ### ▶ O que cada visita contribui na entrada, e é aqui que está a informação que um vetor por lugar não consegue carregar.

**● COBRE** — INTRODUZ as features de nó por visita · INTRODUZ a aresta só para frente, como princípio de projeto

**✕ NUNCA** — a aresta só para frente como conserto, correção ou descoberta. · Ela é decisão de projeto que o documento explica. · Se perguntarem por que a direcionalidade importa, a resposta é o princípio; · se alguém perguntar por um episódio de correção no repositório, é o slide B2, com a proveniência primeiro.

<sub>O que cada visita contribui na entrada, e é aqui que está a informação que um vetor por lugar não consegue carregar. Três grupos. O semântico: a categoria do lugar visitado, como indicador sobre as classes. O tempo cíclico: hora do dia e dia da semana pelo seno e pelo cosseno, para que o fim e o começo de cada ciclo fiquem vizinhos, e não em pontas opostas de uma escala. E os tempos decorridos: o intervalo desde a visita anterior e o intervalo desde a primeira visita daquele usuário, os dois comprimidos por logaritmo, mais o intervalo dentro do mesmo dia e um indicador de primeira visita. É isso que dá ritmo à representação: distinguir a visita que vem minutos depois da anterior daquela que abre um passeio novo. E aqui um princípio de projeto, na redação do próprio capítulo: as arestas entre visitas consecutivas correm numa direção só, da visita anterior para a posterior. A razão está na mesma frase: o alvo é predito do passado do usuário, então a representação é construída só do passado. Todo valor é medido até a própria visita.</sub>

---

## S38 · The geometry of the vectors
`slide **38** · PDF p.45` · **90 s** · fim previsto **28:30**

> ### ▶ E este é o resultado da representação sozinha, antes de qualquer modelo.

**● DIZER EXATO**
- 0,57
- 0,00
- 0,98
- 0,78

**# NÚMEROS** — Ela dá cerca de 0,57 para a representação · check-in contra cerca de 0,00 para o embedding · próximos dá cerca de 0,98 contra 0,78. As

**✕ NUNCA** — nenhum p-valor aqui, e nenhuma afirmação de significância sobre esta figura. · Nunca chamar a diferença de representação de "margem": margem é do TOST, e isto é uma diferença.

<sub>E este é o resultado da representação sozinha, antes de qualquer modelo. A pergunta é simples: esses vetores por visita separam as sete categorias? A silhueta por categoria mede quão compactos e quão separados estão os grupos rotulados, numa escala de menos um a um. Ela dá cerca de **0,57** para a representação em nível de check-in contra cerca de **0,00** para o embedding por lugar. A pureza de categoria dos dez vizinhos mais próximos dá cerca de **0,98** contra **0,78**. As duas médias são sobre os cinco estados americanos. Duas ressalvas, e eu faço as duas antes de alguém pedir. A primeira: a figura caracteriza a **família** da representação, e não a configuração exata que eu avalio depois. É por isso que ela não precisa de partição, de semente nem de pareamento, e é por isso que ela pode vir antes do protocolo. A segunda: a mesma geometria **não** separa regiões. O benefício é de categoria, e o fluxo espacial do modelo lê os vetores de região do mesmo grafo, não estes. Nenhum p-valor nesta tela: aqui é geometria, e o teste vem depois, no bloco de resultados.</sub>

---

## S39 · The architecture: sharing by exchange
`slide **39** · PDF p.46` · **90 s** · fim previsto **30:00**

> ### ▶ A arquitetura, e o que mudou no multitarefa. Cada tarefa tem a sua entrada.

**● DIZER EXATO**
- por troca de informação entre fluxos por tarefa

**✕ NUNCA** — creditar transferência entre tarefas a partir desta tela. · "Backbone", "dual-tower", o identificador de repositório do modelo.

<sub>A arquitetura, e o que mudou no multitarefa. Cada tarefa tem a sua entrada. A de categoria lê a janela de vetores por visita, que é o fluxo semântico. A de região lê a mesma janela de visitas, só que cada visita agora representada pelo vetor treinado do nó de região dela, que é o fluxo espacial. As duas passam por encoders privados, sem peso nenhum compartilhado. E o tronco compartilhado é uma pilha de dois blocos de atenção cruzada: em cada bloco a atenção deixa um fluxo ler as features do outro, enquanto cada um mantém os próprios pesos feed-forward. É esta a frase que eu quero que fique da tela: as tarefas compartilham **por troca de informação entre fluxos por tarefa**, e não por possuírem camadas ocultas em comum. Comparem com o Capítulo 3, onde tudo atravessava um tronco único e as tarefas só se separavam nas saídas. É a mesma família de modelos, com a topologia de compartilhamento trocada, e a topologia era um dos três suspeitos do nulo.</sub>

---

## S40 · The private spatial path, and what the evidence does not separate
`slide **40** · PDF p.47` · **90 s** · fim previsto **31:30**

> ### ▶ Duas coisas fecham a arquitetura, e depois uma posição que eu preciso enunciar com precisão.

**● DIZER EXATO**
- dentro do mesmo modelo
- modelo dedicado de categoria recebe o mesmo ajuste

**✕ NUNCA** — "não podemos provar que não contribuiu, portanto provavelmente contribuiu". · Creditar Texas e Califórnia a transferência entre tarefas. · Nenhuma afirmação de Pareto sobre estes modelos.

<sub>Duas coisas fecham a arquitetura, e depois uma posição que eu preciso enunciar com precisão. A primeira é o caminho espacial privado: a saída de região tem, além do tronco, um ramo pequeno **dentro do mesmo modelo**, e não um segundo modelo, que lê a janela espacial e contorna o tronco. A tarefa de categoria não toca nesse ramo. A segunda é a perda: uma soma de peso fixo, meio a meio entre as duas tarefas, e o peso é fixo **de propósito**, para que qualquer melhora sobre os dedicados venha da representação compartilhada e não de um esquema adaptativo de ponderação. A saída de categoria treina com ajuste de logit, que empurra a fronteira de decisão para o posterior balanceado que a macro-F1 premia, e o **modelo dedicado de categoria recebe o mesmo ajuste**, então a comparação entre os dois não é afetada por ele. Agora a posição. A evidência aqui **não separa** as contribuições do tronco compartilhado e do caminho espacial privado. Ela não estabelece que o compartilhamento ajuda, e não o descarta. A afirmação que eu faço é sobre o desenho: esta combinação produz uma saída de região acima de dois modelos dedicados, nos dois conjuntos com os maiores números de regiões. Não é uma afirmação sobre transferência entre tarefas.</sub>

---

## S41 · Protocol, step 1 of 4: the unit of data
`slide **41** · PDF p.48` · **60 s** · fim previsto **32:30**

> ### ▶ O protocolo, e ele é o degrau que sustenta tudo o que vem depois.

**● DIZER EXATO**
- disjunta por usuário
- sobrepostas, de passo um
- estas não são as mesmas janelas

**✕ NUNCA** — "as mesmas janelas" para os três estudos. · "Fold" como palavra solta na fala: a superfície em português é partição.

<sub>O protocolo, e ele é o degrau que sustenta tudo o que vem depois. São quatro passos, e cada um responde a uma pergunta que o anterior deixa aberta. Primeiro passo: qual é a unidade de dados. Validação cruzada de cinco partições, **disjunta por usuário**: todas as janelas de um usuário ficam na mesma partição, então as visitas de um usuário de teste nunca aparecem no treino. Isto é exatamente a reparação da limitação que eu declarei no Capítulo 3. A estratificação é pelo rótulo da próxima categoria, porque as sete classes são desbalanceadas. As janelas são **sobrepostas, de passo um**: para cada usuário com pelo menos dez visitas, começa uma janela de nove visitas em cada visita, e a visita seguinte é o alvo; janelas curtas duplicadas, que terminam no mesmo alvo, são removidas. Repito o aviso da Seção 2: **estas não são as mesmas janelas** dos Capítulos 3 e 4, que usaram janelas não sobrepostas. E uma coisa que eu digo agora para não parecer descoberta depois: a partição retida é também a de validação, não há um terceiro corte, e é daí que sai o segundo dos meus limites.</sub>

---

## S42 · Protocol, step 2 of 4: what is measured
`slide **42** · PDF p.49` · **65 s** · fim previsto **33:35**

> ### ▶ Segundo passo: o que se mede. Na categoria, macro-F1, como eu defini na Seção 2, e o ponto de referência dela é o piso de classe majoritária, que fica entre 5,7 e 7,3 conforme o conjunto.

**● DIZER EXATO**
- 24,7% das visitas

**# NÚMEROS** — majoritária, que fica entre 5,7 e 7,3 conforme · e ainda assim marca 5,7 de macro-F1, porque

**✕ NUNCA** — Acc@10 sem o desconto OOD. · Nenhum número sem o seu ponto de referência.

<sub>Segundo passo: o que se mede. Na categoria, macro-F1, como eu defini na Seção 2, e o ponto de referência dela é o piso de classe majoritária, que fica entre 5,7 e 7,3 conforme o conjunto. Um exemplo concreto mostra por que essa métrica existe e a acurácia simples não: na Flórida esse piso acerta **24,7% das visitas** e ainda assim marca **5,7** de macro-F1, porque as outras seis categorias ele nunca acerta. Na região, acurácia em dez: a fração de visitas de teste cuja região verdadeira está entre as dez predições de maior pontuação. Ela não distingue o primeiro lugar do décimo, e eu declaro isso. E ela vem com um desconto, que é o ponto que mais gera pergunta: uma região que não aparece na partição de treino conta como **erro**. Então o que eu reporto é a acurácia em dez medida nas visitas dentro da distribuição, multiplicada por um menos a fração fora da distribuição. Os pontos de referência da região são dois. O modelo dedicado, que é a comparação controlada. E um piso de Markov de primeira ordem sobre transições de região, calculado sob as mesmas janelas e as mesmas partições, que alcança de 51 a 72 de acurácia em dez. Esse piso é alto de propósito: janelas de passo um fazem da última região visitada um preditor forte da próxima, e é exatamente esse sinal que uma tabela de transição lê.</sub>

---

## S43 · Protocol, step 3 of 4: what is compared
`slide **43** · PDF p.50` · **65 s** · fim previsto **34:40**

> ### ▶ Terceiro passo: o que se compara. A comparação é entre o modelo conjunto e os modelos dedicados, lendo a mesma representação, as mesmas janelas e as mesmas partições.

**● DIZER EXATO**
- vinte modelos ajustados por configuração
- um único modelo salvo por partição
- Todo veredito que eu vou dar é o da convenção estrita

**# NÚMEROS** — modelo conjunto, em até 0,23 de macro-F1 e · 0,23 de macro-F1 e 0,93 de acurácia em

**✕ NUNCA** — "n = 20 repetições pareadas". · Misturar joint-best com a leitura por tarefa dos Caps. · 3 e 4 sem declarar. · "As mesmas partições" entre sementes: vale dentro de uma semente.

<sub>Terceiro passo: o que se compara. A comparação é entre o modelo conjunto e os modelos dedicados, lendo a mesma representação, as mesmas janelas e as mesmas partições. Começo definindo semente, porque a palavra é ambígua na literatura. Aqui uma **semente** é uma repetição completa do experimento de cinco partições: ela fixa a inicialização aleatória **e** a divisão dos usuários, então cada semente sorteia a sua própria divisão. Dentro de uma semente, os modelos comparados leem a mesma partição, e é isso que licencia o pareamento. São quatro sementes, zero, um, sete e cem, vezes cinco partições, o que dá **vinte modelos ajustados por configuração**. Mas a unidade inferencial é **quatro**, as quatro médias por semente, porque partições dentro de uma semente não são independentes. E a convenção de leitura é a *joint-best*: as duas notas vêm de **um único modelo salvo por partição**, na época escolhida pela nota conjunta de validação, que é a média geométrica das duas métricas. Eu escolhi a convenção mais estrita de propósito. A alternativa, ler cada tarefa na melhor época dela, não descreve nenhum modelo salvo; ela é mais favorável ao modelo conjunto, em até 0,23 de macro-F1 e 0,93 de acurácia em dez numa semente, e viraria mais quatro resultados de categoria e mais dois de região em melhoras sob a mesma correção. **Todo veredito que eu vou dar é o da convenção estrita.**</sub>

---

## S44 · Protocol, step 4 of 4: how it is decided
`slide **44** · PDF p.51` · **65 s** · fim previsto **35:45**

> ### ▶ Quarto passo: como se decide. A primeira frase é a que organiza tudo: afirmar ganho e afirmar equivalência exigem testes diferentes, e uma diferença não significativa não é evidência de equivalência.

**● DIZER EXATO**
- afirmar ganho e afirmar equivalência exigem testes diferentes
- reportado ao lado e concorda
- resultados secundários, fora do plano

**# NÚMEROS** — consegue ficar abaixo de 0,0625, qualquer que seja

**✕ NUNCA** — apresentar o desvio como confissão. · Aplicar a margem de dois pontos ao eixo de categoria. · "Significativo" sem nomear o teste.

<sub>Quarto passo: como se decide. A primeira frase é a que organiza tudo: **afirmar ganho e afirmar equivalência exigem testes diferentes**, e uma diferença não significativa não é evidência de equivalência. Havia um plano de análise escrito, fixado durante o desenvolvimento e antes de qualquer resultado ser lido. Ele atribuiu um teste de superioridade à próxima categoria e um teste de não-inferioridade à próxima região, numa margem de dois pontos. A margem tem razão declarada: uma variação desse tamanho na acurácia em dez fica abaixo do nível em que a preparação em escala de bairro se comportaria de outro jeito. O teste primário é um t pareado sobre as quatro médias por semente, com intervalo de confiança de noventa por cento, e correção de Holm sobre os seis conjuntos, separadamente dentro de cada família de tarefa. Aqui eu declaro um desvio, porque ele existe e está no código: o plano registrava um Wilcoxon pareado sobre as vinte diferenças por partição. Ele é **reportado ao lado e concorda** com o t. O t carrega o veredito porque partições dentro de uma semente não são independentes, e porque, nesse apoio, o Wilcoxon exato unilateral não consegue ficar abaixo de 0,0625, qualquer que seja o efeito. Não é confissão: são dois apoios com o mesmo veredito. E duas consequências do plano, ditas antes dos números: ele não definiu teste de superioridade para região, então os dois ganhos de região que eu vou mostrar são **resultados secundários, fora do plano**; e ele não registrou margem de equivalência na categoria, então uma diferença de categoria que falha a superioridade é **não resolvida**, e é reportada pelo limite que o intervalo dela sustenta.</sub>

---

## S45 · Result 1: the representation, at every dataset
`slide **45** · PDF p.52` · **110 s** · fim previsto **37:35**

> ### ▶ Primeiro resultado, e ele é sobre a representação sozinha, não sobre o multitarefa.

**● DIZER EXATO**
- Só a entrada muda
- à frente nos seis
- unânime nas cinco partições em todos eles
- cinco dos seis
- direção consistente
- o que eles não fazem é isolar a hierarquia

**# NÚMEROS** — a p igual a 0,07, onde a direção · salto da tabela, mais 0,23. A faixa vai · faixa vai desse mais 0,23 na Flórida a · na Flórida a mais 6,29 em Istambul. O

**✕ NUNCA** — "o nível de check-in bate o de lugar nos seis" no sentido de teste: o teste separa em cinco. · Nunca generalizar a cláusula do capítulo "under a tenth of the place-to-check-in gap": ela é dita por estado, e generalizá-la é aritmeticamente falso contra a própria Tabela 9, na mesma página. · Nunca chamar a diferença de representação de "margem".

<sub>Primeiro resultado, e ele é sobre a representação sozinha, não sobre o multitarefa. A comparação é controlada: mesmo alvo, mesmo modelo de tarefa única, mesma configuração de treino, mesmas partições, mesmas janelas, mesmo orçamento de épocas, mesmo ajuste de logit. **Só a entrada muda.** A convenção desta tabela é a semente zero, com cinco partições pareadas, e o desvio é entre partições; guardem isso, porque a próxima tabela tem outra convenção. A leitura é a da própria tabela, não a minha: o nível de check-in está **à frente nos seis** conjuntos, e é **unânime nas cinco partições em todos eles**; um teste pareado sobre as cinco partições separa as duas colunas em **cinco dos seis**, e a Flórida é a exceção, a p igual a 0,07, onde a direção é unânime mas a diferença não alcança significância. E a Flórida ser a exceção não é acaso: ela é o menor salto da tabela, mais 0,23. A faixa vai desse mais 0,23 na Flórida a mais 6,29 em Istambul. O que isso estabelece é uma **direção consistente**, não um efeito grande. À direita, os dois controles que separam esse ganho de duas explicações mais baratas. O CTLE, que é a contextualização mais próxima, fica cerca de dois pontos abaixo do embedding por lugar na Flórida sob a mesma regra, e repete a ordenação com pesos fixos no Alabama, no Arizona e em Istambul. E a concatenação de features cruas ao embedding por lugar levanta esse embedding em 2,0, 1,7 e 0,8 ponto de macro-F1, no Alabama, no Arizona e na Flórida. Os dois controles limitam explicações mais baratas; **o que eles não fazem é isolar a hierarquia**. O controle de concatenação foi refeito depois do envio, na escala da Tabela 9, e lá ele fecha a maior parte da diferença — num conjunto, a ultrapassa. Tenho o slide, se quiserem vê-lo.</sub>

---

## S46 · Result 2: one model, two tasks
`slide **46** · PDF p.53` · **120 s** · fim previsto **39:35**

> ### ▶ Segundo resultado. E uma frase sobre a coluna do meio, antes de eu ler as duas nossas, porque ela muda o que as outras duas significam: a coluna Dedicated é o sistema mais forte desta tabela.

**● DIZER EXATO**
- pelo menos três vírgula zero seis pontos
- entre sementes

**# NÚMEROS** — POI-RGNN em pelo menos 3,06 pontos nos seis · na tela, fica entre 5,7 e 7,3. Na · forte em pelo menos 3,55 pontos de acurácia

**✕ NUNCA** — "empata", "matches", "ties", "em todos os conjuntos supera". · "Beats" ou "wins" para os métodos externos: o verbo é excede. · Nunca um número do Cap. · 3 nesta tela.

<sub>Segundo resultado. E uma frase sobre a coluna do meio, antes de eu ler as duas nossas, porque ela muda o que as outras duas significam: a coluna **Dedicated** é o sistema mais forte desta tabela. Ela está acima de todo baseline externo reportado, nos seis conjuntos e nas duas tarefas, e acima do piso de Markov também. Então o que vocês vão ver não é o modelo conjunto igualando um espantalho: é ele igualando a régua mais dura que eu tenho. E o capítulo põe número nisso: o conjunto fica **pelo menos três vírgula zero seis pontos** acima do baseline externo mais forte em todos os conjuntos, enquanto a diferença entre um modelo e dois é **meio ponto**. É o menor dos três efeitos em jogo aqui, e é nesse sentido que um modelo pode substituir dois. Em cima, a próxima categoria; embaixo, a próxima região; os mesmos seis conjuntos, na mesma ordem, nos dois blocos. Primeiro a convenção, porque ela mudou: aqui são quatro sementes vezes cinco partições, e o desvio é **entre sementes**, não entre partições como na tabela anterior. A ressalva vem antes da leitura: o modelo dedicado de categoria teve busca de configuração em todos os seis conjuntos, e o conjunto **não** teve busca no Texas nem na Califórnia, que carregam configuração transferida. Onde a busca do dedicado é a mais ampla, o resíduo favorece o dedicado, o que torna a diferença de categoria que eu reporto conservadora ali. A comparação que sustenta a minha afirmação é entre as colunas **Dedicated** e **Joint**, porque essas duas leem a mesma representação, as mesmas janelas e as mesmas partições. As colunas externas estão aqui como comparação com desenhos publicados, e elas rodam com as representações delas, então trazem junto a vantagem de representação do slide anterior. Na categoria, o conjunto excede o POI-RGNN em pelo menos 3,06 pontos nos seis, e o piso de classe majoritária, que não está na tela, fica entre 5,7 e 7,3. Na região, o conjunto excede a referência externa mais forte em pelo menos 3,55 pontos de acurácia em dez. O STAN e o ReHDM não estão na tela, e a razão é a ressalva de protocolo: o STAN roda nas nossas partições mas constrói as próprias representações e as próprias sequências, e o ReHDM roda sob o protocolo publicado dele. E eu vou dizer uma coisa contra mim mesmo, porque ela está no capítulo: o piso de Markov, que é um método não aprendido, fica **acima** desses três sistemas externos na maioria dos conjuntos. É por isso que eu trato o piso, e não os externos, como a referência que a tarefa de região tem de exceder. O conjunto e o dedicado estão acima do piso nos seis.</sub>

---

## S47 · The verdict, dataset by dataset
`slide **47** · PDF p.54` · **100 s** · fim previsto **41:15**

> ### ▶ E este é o veredito, com o intervalo de cada diferença.

**● DIZER EXATO**
- próxima região
- permanece dentro da margem de dois pontos
- déficits, não empates
- próxima categoria
- supera na Flórida
- não resolvidas

**# NÚMEROS** — dedicado no Texas, mais 1,21, e na Califórnia · e na Califórnia, mais 1,06; nos dois, as · os p corrigidos são 0,00013 e menos de · quero evitar: Alabama menos 0,87; Arizona menos 0,44

**✕ NUNCA** — "empata", "matches", "ties", "em todos". · Aplicar a margem de dois pontos ao eixo de categoria, ou meio ponto ao eixo de região. · Chamar as quatro diferenças dentro da margem de empates. · Dizer que uma diferença "exclui zero" sem dizer para que lado.

<sub>E este é o veredito, com o intervalo de cada diferença. Na **próxima região**, o modelo conjunto **supera** o dedicado no Texas, mais 1,21, e na Califórnia, mais 1,06; nos dois, as vinte partições favorecem o conjunto, e os p corrigidos são 0,00013 e menos de dez elevado a menos quatro. Nos outros quatro conjuntos ele **permanece dentro da margem de dois pontos**, registrada antes de qualquer resultado ser lido. E eu enuncio os quatro, porque citar três e omitir Istambul seria justamente o erro que eu quero evitar: Alabama menos 0,87; Arizona menos 0,44; Flórida menos 0,16; Istambul menos 0,08. Os quatro são **déficits, não empates**, e os quatro intervalos ficam inteiramente abaixo de zero. A direção é declarada, não arredondada. Na **próxima categoria**, ele **supera na Flórida**, mais 0,19, com p corrigido de 0,011 e dezenove das vinte partições a favor. As outras cinco são **não resolvidas**, e elas não apontam para o mesmo lado: Istambul, mais 0,08, exclui zero a favor do conjunto; Texas, menos 0,13, e Alabama, menos 0,19, excluem zero a favor do dedicado; Arizona e Califórnia ficam sobre o zero. Nenhuma das cinco sobrevive a Holm. O que se pode dizer sobre magnitude vem dos intervalos: o mais largo alcança 0,34 de ponto a partir de zero, no Alabama, e isso limita as seis diferenças de categoria a **meio ponto de zero de uma vez só**. Duas últimas coisas, ditas por mim antes de serem pedidas. Os dois ganhos de região são **resultados secundários, fora do plano registrado**. E Istambul, o único conjunto fora dos Estados Unidos, fica a menos de um décimo de ponto nos dois eixos, que é o teste de validade externa deste capítulo.</sub>

---

## S48 · The measured trade, and four declared limits
`slide **48** · PDF p.55` · **75 s** · fim previsto **42:30**

> ### ▶ A troca, medida, e depois quatro limites que eu ofereço antes de alguém pedir.

**● DIZER EXATO**
- operacional, não aritmético
- todo escore absoluto que eu reportei é otimista

**# NÚMEROS** — capítulo reporta cerca de 4,2 milhões de parâmetros · parâmetros no Alabama contra 1,1 milhão dos dois · dois dedicados somados, e 5,2 contra 2,0 na · pequenos, o maior deles 0,87 no Alabama: é

**✕ NUNCA** — repetir a razão de parâmetros como se tivesse sido re-medida, e nunca citar uma recontagem. · Se a pergunta vier, a resposta é que a razão de parâmetros não foi re-medida. · Nunca citar as duas porcentagens de parâmetros impressas no Apêndice G do suplemento: elas estão erradas, e o assunto é do slide B3.

<sub>A troca, medida, e depois quatro limites que eu ofereço antes de alguém pedir. A troca primeiro: o modelo conjunto é **maior**. O capítulo reporta cerca de 4,2 milhões de parâmetros no Alabama contra 1,1 milhão dos dois dedicados somados, e 5,2 contra 2,0 na Califórnia; uma passagem custa mais computação do que rodar os dois modelos pequenos. O que o modelo único entrega é **operacional, não aritmético**: um artefato para treinar, versionar e implantar, e uma passagem cujas entradas produzem as duas respostas de uma vez. E os quatro resultados de região dentro da margem são déficits pequenos, o maior deles 0,87 no Alabama: é uma troca medida, não uma substituição de graça. Os quatro limites. Primeiro: a representação é treinada uma vez sobre todos os lugares; uma reconstrução por partição, só com usuários de treino, mudou os resultados em no máximo 0,33 de acurácia em dez e 0,29 de macro-F1, em três conjuntos e numa semente, e a metade de categoria dessa verificação cobre de 67 a 87 por cento dos dados de validação. Segundo: a seleção de época consulta a mesma partição em que a nota é depois lida, então **todo escore absoluto que eu reportei é otimista**; a comparação entre conjunto e dedicado é bem menos afetada, porque a regra é a mesma para os dois nas mesmas partições e porque o dedicado de categoria recebe a busca mais ampla, mas daí não segue que o viés se cancele exatamente. Terceiro: eu não construo nem avalio serviço nenhum. Quarto: cada nó de visita se apoia só nas visitas que o precedem, e o grafo não passa informação de uma visita posterior para uma anterior, nem no treino nem na leitura.</sub>

---

## S49 · The ladder: three studies, three layers
`slide **49** · PDF p.57` · **30 s** · fim previsto **43:00**

> ### ▶ Uma tela em que a coletânea inteira cabe. Três linhas, os três estudos.

**● COBRE** — INTRODUZ a leitura conjunta dos três estudos lado a lado | RETOMA a linhagem de 2.1 e as três camadas de 5.1

**✕ NUNCA** — nenhum número nesta tela, em nenhuma célula. · Nenhum "fomos de X para Y" atravessando protocolos (§8 regra 7). · Nunca "supera" nesta tela: a licença é por célula de resultado, e aqui não há resultado.

<sub>Uma tela em que a coletânea inteira cabe. Três linhas, os três estudos. Três colunas, as três camadas. Mais uma quarta coluna: o que moveu. O Capítulo 3 não separou nada, e é isso que ele entrega, um nulo com três suspeitos. O Capítulo 4 manteve a arquitetura fixa de propósito, e é esse congelamento que faz a troca de entrada valer como diagnóstico. O Capítulo 5 mexeu nas três camadas. Um veredito condicional, medido sob o protocolo mais estrito dos três. O que os três estudos, juntos, estabelecem, e o que não estabelecem?</sub>

---


# CONCLUSÃO — a resposta condicional


## S50 · The conditional answer
`slide **50** · PDF p.58` · **55 s** · fim previsto **43:55**

> ### ▶ A resposta consolidada, e ela é condicional de propósito.

**● COBRE** — RETOMA o veredito de 5.5, a pergunta de 1.3 e o protocolo disjunto por usuário de 5.4 ("o protocolo mais estrito dos três")

**✕ NUNCA** — "MTL funciona" sem condição. · Re-caminhar a cadeia dos três estudos, que acabou de estar na tela em S49. · Creditar os ganhos de região a transferência entre tarefas. · Nenhum número novo.

<sub>A resposta consolidada, e ela é condicional de propósito. O aprendizado multitarefa ajuda a previsão da próxima categoria e da próxima região sob o desenho final e o protocolo de avaliação desenvolvidos nesta dissertação. O que isso não autoriza é dizer que o multitarefa sempre ajuda: ao longo dos três estudos, relação entre tarefas e treino conjunto não bastaram por si sós. Uma condição está estabelecida por comparação controlada, e é a representação de entrada. Duas outras a evidência sugere sem isolar: a arquitetura, e a escala do conjunto de dados. Sobre escala eu sou explícito. Ela continua sendo condição possível, não causa estabelecida, por duas razões que estão no próprio texto: a ordenação não se mantém dentro do par Texas e Califórnia, e estados com mais regiões também tendem a ter mais check-ins. Identificar essas condições é o achado principal desta dissertação.</sub>

---

## S51 · The contribution, in one block
`slide **51** · PDF p.59` · **45 s** · fim previsto **44:40**

> ### ▶ Esta é a mesma tela que eu mostrei no começo, com as mesmas palavras, e agora ela tem a evidência atrás.

**● COBRE** — RETOMA a contribuição de 1.5 (§8 regra 13: segunda das duas aparições, redação idêntica)

**✕ NUNCA** — redação diferente da de S7, mesmo que melhor. · Nenhum número novo. · Nunca a razão de parâmetros como verificada: o slide diz "maior", que é o que a página imprime, e nada além.

<sub>Esta é a mesma tela que eu mostrei no começo, com as mesmas palavras, e agora ela tem a evidência atrás. A metade prática: um modelo, uma passagem, duas predições. O ganho é operacional, não computacional, um artefato para treinar, versionar e implantar. E o preço vai junto: o modelo conjunto é maior que os dois dedicados que ele substitui. A metade científica: as condições, não um sim universal. A representação de entrada e a topologia de compartilhamento decidem se o multitarefa ajuda nestas tarefas. É por isso que o nulo do Capítulo 3 e o resultado positivo do Capítulo 5 não se contradizem.</sub>

---

## S52 · Six limitations, six next steps (1 of 2)
`slide **52** · PDF p.60` · **65 s** · fim previsto **45:45**

> ### ▶ As limitações, cada uma amarrada ao passo que ela pede.

**● COBRE** — INTRODUZ as limitações 1, 2 e 3 e os três trabalhos futuros correspondentes

**✕ NUNCA** — apresentar uma limitação sem o passo que ela pede. · ablation na tela ou na fala: o termo não está no GLOSSARY, e a superfície usada é "teste controlado". · As datas não são resultado, e nenhum número de resultado entra aqui.

<sub>As limitações, cada uma amarrada ao passo que ela pede. Eu prefiro dizê-las antes de serem perguntadas. Não são desculpas: são seis experimentos que alguém pode rodar. Primeira, a idade dos dados. Os cinco estados vão de janeiro de 2009 a agosto de 2011, e Istambul tem check-ins em dois blocos, 2012 a 2013 e 2017 a 2018, com cerca de sete em cada dez no bloco mais antigo. O passo é direto: rastros mais novos e mais densos. Segunda, a taxonomia é grossa. Sete classes de topo, e uma divisão mais fina pode mudar o efeito do treino conjunto. Terceira, e esta é a que eu mais gostaria de ver feita: a representação é transdutiva. Treinada sobre o grafo de check-ins de cada conjunto, ela não representa lugar nem usuário novo sem retreinar. O passo é uma variante indutiva, que sustentaria uso numa cidade que cresce. E, na mesma representação, dois testes controlados: variar o acoplamento com a tabela de vetores de lugar pré-treinada, e uma formulação em hipergrafo, em que uma aresta junta as várias visitas de uma sessão.</sub>

---

## S53 · Six limitations, six next steps (2 of 2)
`slide **53** · PDF p.61` · **65 s** · fim previsto **46:50**

> ### ▶ Quarta: eu não predigo o próximo lugar exato, então as conclusões valem para próxima categoria e próxima região.

**● COBRE** — INTRODUZ as limitações 4, 5 e 6 e os três trabalhos futuros correspondentes

**✕ NUNCA** — que o confundimento de par de tarefas foi removido; · ele é limitado pelo controle de par fixo. · pipeline: a superfície é "como as entradas são construídas". · ablation.

<sub>Quarta: eu não predigo o próximo lugar exato, então as conclusões valem para próxima categoria e próxima região. Acrescentar o próximo lugar como terceiro alvo reusa a representação que já existe: muda a construção da entrada e acrescenta uma saída. Quinta: fora dos Estados Unidos, a evidência se apoia numa cidade só. Mais cidades ampliam a base. Sexta, e é a mais honesta das seis. O par de tarefas mudou junto com a representação e com a topologia, e nenhuma comparação controlada isolada separa as duas mudanças no resultado final. O que eu tenho é o Capítulo 4 como controle de par fixo para o diagnóstico. E eu sei por que a comparação óbvia não serve: rodar a tarefa estática sob a representação em nível de check-in não é limpa, porque a categoria do lugar visitado é atributo de entrada do nó de check-in, e o alvo ficaria parcialmente legível da própria entrada. Isso decorre do desenho, e não foi medido. O confundimento fica limitado pelo controle de par fixo, não removido. O passo que resolveria é um alvo estático que a representação não carregue já como entrada.</sub>

---

## S54 · Closing
`slide **54** · PDF p.62` · **70 s** · fim previsto **48:00**

> ### ▶ Eu abri esta apresentação dizendo que antecipar o quê e o onde da próxima visita sustenta recomendação, navegação, planejamento de transporte e alocação de recursos por área.

**● COBRE** — RETOMA as aplicações de 1.1 e a restrição de modelo único de 1.3 | fecho

**✕ NUNCA** — nenhum número novo. · Nenhum "MTL funciona" sem condição. · Nunca ler a lista de agradecimentos da tela: ela não está na tela.

<sub>Eu abri esta apresentação dizendo que antecipar o quê e o onde da próxima visita sustenta recomendação, navegação, planejamento de transporte e alocação de recursos por área. Fecho no mesmo lugar, um nível acima. O produto prático desta dissertação é um modelo único que prediz duas propriedades da próxima visita numa passagem: que tipo de lugar a pessoa vai visitar, e em que parte da cidade. E a contribuição metodológica é a sequência de evidência que levou até ele: um resultado negativo publicado, o diagnóstico que identificou a representação de entrada como o gargalo, e uma solução desenhada a partir desse diagnóstico. O resultado negativo não foi obstáculo à contribuição. Ele foi a primeira metade dela. Antes de encerrar, os agradecimentos. Ao meu orientador, o professor Fabrício Silva, pela liberdade de explorar as minhas ideias e pela confiança para levá-las adiante. Ao Germano Santos, que trabalhou ao meu lado em todos os artigos deste mestrado. Ao Tarik Paiva. À Ingred F. Almeida. Ao Pedro Maia. À Universidade Federal de Viçosa e aos professores que fizeram parte deste caminho. E à minha família. Obrigado. Fico à disposição da banca.</sub>

---

## S55 · Acknowledgements
`slide **55** · PDF p.63` · **30 s** · fim previsto **48:30**

> ### ▶ os agradecimentos, terminando na banca: *"E, por fim, aos senhores da banca: obrigado por lerem o trabalho e por estarem aqui.

**● COBRE** — INTRODUZ nada (fecho social)

**✕ NUNCA** — nada de resultado aqui. · É fecho social, e é o slide que fica na tela durante a arguição.

<sub>os agradecimentos, terminando na banca: *"E, por fim, aos senhores da banca: obrigado por lerem o trabalho e por estarem aqui. Fico à disposição para as perguntas."*</sub>

---
