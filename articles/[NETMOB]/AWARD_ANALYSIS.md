# Best papers e trabalhos de destaque da NetMob (2023–2025)

**Atualização:** 18 de junho de 2026.  
**Critério de inclusão:** prêmios confirmados pela página oficial ou por página institucional/autoral;
quando a fonte não é oficial, isso é indicado. Não foi encontrada uma lista oficial consolidada de
awards para todas as edições.

## 1. Resumo dos prêmios confirmados

| Ano | Categoria | Trabalho | Por que se destaca |
|---|---|---|---|
| 2025 | Best Main Conference Paper | **Causal inference in the city: Improving Urban Policy Evaluation Through Mobility-Aware Methods** — Bijin Joseph, Hamish Gibbs, Takahiro Yabe, Esteban Moro | conecta método causal a avaliação de política urbana e explicita o papel da mobilidade |
| 2025 | Best Main Conference Poster | **Modeling Base Station Metadata Geolocation** — Orlando E. Martínez-Durive, Stefanos Bakirtzis, Cezary Ziemlicki, Marco Fiore | problema técnico concreto, relevante para qualidade/localização de dados de rede |
| 2025 | Best Data Challenge Paper | **On the Relationship between Space-Time Accessibility and Leisure Activity Participation** — Yuan Liao et al. | nova métrica individual de acessibilidade, dados GNSS e mecanismo comportamental interpretável |
| 2025 | Best Data Challenge Poster | **Beyond Willpower: Structural Constraints and Inequality in Mobility-Related CO₂ Emissions** — Yuxi Zhang et al. | desigualdade e emissões enquadradas como restrições estruturais, não apenas escolha individual |
| 2024 | Best Paper | **Individual Alternative Routing based on Road Popularity** — Luca Pappalardo, Giuliano Cornacchia | algoritmo de rotas alternativas com objetivo mensurável de distribuir tráfego e reduzir emissões |
| 2024 | Best Data Challenge Paper | **Assessing Urban and Rural Routing Inefficiencies with Aggregated Mobility Patterns** — Nandini Iyer, Massimiliano Luca, Riccardo Di Clemente | comparação urbano-rural, mobilidade agregada e relevância direta para eficiência/planejamento |
| 2024 | Best Data Challenge Poster | **The Role of Mobility Infrastructure at Risk for Carbon Efficiency** — Mauricio Rada Orellana, Markus Schläpfer | mobilidade, infraestrutura e carbono em quatro países do Sul Global |
| 2023 | Best contribution / Best Paper* | **Mixing Individual and Collective Behaviors in Mobility Models** — Massimiliano Luca et al. | regra simples e interpretável que mistura informação individual/coletiva e é robusta a mudança de regime |
| 2023 | Best Challenge Contribution runner-up | **Unmasking Socioeconomic Disparities: A Study of Urban Segregation through the Lens of Mobile App Usage Patterns** — Yuya Shibuya et al. | extrai desigualdade socioeconômica de padrões de uso de apps, com forte fit ao challenge |

\* A nomenclatura aparece como “best contribution” em uma fonte institucional e “Best Paper Award” na
página do autor. Sem página oficial consolidada de 2023, convém preservar essa ressalva.

## 2. Leitura dos premiados de 2025

### Causal inference in the city

O título faz três coisas que a NetMob valoriza: declara a ferramenta (causal inference), o domínio
(city/urban policy) e o avanço (mobility-aware evaluation). Mesmo sem rubrica publicada, a escolha
sinaliza que **predição não é o único eixo de qualidade**: identificar efeitos e melhorar decisões
públicas pode ser mais relevante que aumentar uma métrica de forecast.

Lição de escrita: se houver claim causal, explicitar tratamento, outcome, contrafactual/identificação,
confounders e teste de sensibilidade. “Causal” no título aumenta o padrão de evidência exigido.

### Space-Time Accessibility and Leisure Activity Participation

O trabalho introduz uma métrica de acessibilidade espaço-temporal baseada em capabilities. Usa GPS de
2.415 residentes da região de Paris, modo de transporte e infraestrutura; relaciona acessibilidade,
tempo de viagem e diversidade de locais de lazer por structural equation modeling. Os efeitos diretos
e indiretos têm sinais opostos, o que produz uma conclusão mais rica que “mais acessibilidade é melhor”.

Padrões fortes:

- construto teórico claro e operacionalizado;
- dados individuais ricos e adequados à pergunta;
- validação do construto antes do modelo explicativo;
- heterogeneidade por modo/estrutura familiar;
- efeito direto, mediação e limites, não apenas correlação bruta;
- implicação para planejamento e equidade.

Fonte: [preprint no arXiv](https://arxiv.org/abs/2510.10307).

### Posters premiados

Os dois posters cobrem extremos complementares do escopo: um problema de infraestrutura/metadata de
rede e um problema social-ambiental. Isso mostra que poster não significa trabalho menor; a seleção
parece favorecer uma pergunta visualizável, resultado concreto e conversa produtiva durante a sessão.

Fonte oficial: [programa e awards de 2025](https://netmob.org/www25/program/).

## 3. Leitura dos premiados de 2024

### Individual Alternative Routing based on Road Popularity

O método usa popularidade de vias para produzir rotas alternativas e redistribuir veículos. A ideia é
fácil de explicar, tem baseline natural (shortest path/alternativas) e produz outcomes operacionais
(congestionamento/emissões). O trabalho também foi publicado em versão curta no workshop SUMob 2024,
coerente com a permissão da NetMob para apresentar trabalho publicado em outro lugar.

Lição: um método relativamente simples pode vencer quando a motivação, o mecanismo e o impacto são
diretos. Complexidade arquitetural não substitui uma hipótese clara.

Fontes: [ISTI-CNR](https://www.isti.cnr.it/en/announcements/achievements/2534-best-paper-award-at-netmob-2024),
[registro bibliográfico/DOI](https://doi.org/10.1145/3681779.3696836).

### Routing Inefficiencies (Data Challenge)

O trabalho vencedor do challenge transforma dados agregados em uma comparação urbano-rural de
ineficiência de rotas. A força é alinhar o método ao objetivo declarado da edição de 2024: mobilidade,
política pública e Sul Global, usando o dataset oficial do desafio.

Fonte: [FBK](https://howto.fbk.eu/en/comunicazioni/brand-new-award-mobs-unit-2/).

### Mobility Infrastructure at Risk for Carbon Efficiency (poster)

Analisa Colômbia, México, Indonésia e Índia. A abrangência multinacional ajuda a sustentar validade
externa, enquanto a pergunta une risco de infraestrutura e eficiência de carbono.

Fonte: [Columbia Urban Systems Engineering Lab](https://urbansystems.civil.columbia.edu/news/best-poster-award-netmob-2024).

## 4. Leitura dos destaques de 2023

### Mixing Individual and Collective Behaviors in Mobility Models

O modelo mistura matrizes de transição individuais e coletivas usando entropia como medida de
incerteza. A publicação posterior testa cinco cidades e o período de ruptura da COVID-19: o componente
individual perde desempenho sob mudança comportamental, enquanto o coletivo e a mistura permanecem
mais estáveis. O método é compacto, interpretável e associado a uma explicação comportamental.

Padrões fortes:

- mecanismo legível em uma equação;
- múltiplas cidades;
- comparação com componentes individuais, coletivos e baselines neurais;
- avaliação fora de rotina/mudança de distribuição;
- resultado quantitativo e explicação de quando cada componente ajuda.

Fontes: [artigo posterior em PNAS/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12054799/),
[menção institucional FBK](https://magazine.fbk.eu/en/author/massimiliano-luca/).

### Unmasking Socioeconomic Disparities (runner-up do challenge)

O trabalho usa consumo de apps como lente para segregação urbana. A contribuição é mais do que uma
classificação: converte uma forma nova de dado móvel em leitura socioeconômica espacial.

Fonte: [página do autor](https://www.yuyashibuya.com/2023/10/07/Honor-to-receive-an-award.html).

## 5. Padrões transversais

### O que aparece repetidamente

1. **Problema primeiro.** Os títulos vencedores começam com causalidade, acessibilidade, desigualdade,
   roteamento ou infraestrutura — raramente com uma sigla de modelo.
2. **Mobilidade como mecanismo.** Os dados não são somente features; mobilidade explica acesso,
   exposição, resiliência, ineficiência ou desigualdade.
3. **Método interpretável.** Mesmo quando há ML, o leitor entende por que o método deve funcionar.
4. **Validação além de uma média.** Múltiplas cidades, grupos, regimes temporais, caminhos diretos e
   indiretos, ou comparações urbano-rural.
5. **Impacto concreto.** Política pública, transporte, emissões, acesso, crise ou operação de rede.
6. **Equidade e estrutura.** Trabalhos recentes evitam atribuir outcomes apenas a escolhas individuais.
7. **Narrativa visual.** A descoberta cabe em uma figura/tabela e em um talk de 12 minutos.

### O que não é sustentado pelas evidências

- Não há base para afirmar pesos oficiais de avaliação.
- Não há base para afirmar que deep learning, causal inference ou um dataset específico recebem
  preferência automática.
- Awards não permitem inferir taxa de aceitação.
- Ausência de um tema entre premiados não significa falta de interesse; os programas são mais amplos.
- Menções em páginas pessoais são úteis para preencher lacunas, mas têm confiança menor que uma lista
  oficial.

## 6. Heurística para escolher o headline deste projeto

Para `netcore`, ordenar as contribuições assim:

1. descoberta comportamental/metodológica: MTL pode ser Pareto sem sacrificar região;
2. mecanismo: substrato contextual beneficia semântica e torre privada protege geografia;
3. evidência: múltiplos estados, seeds, baselines STL e ablações do confound;
4. implicação: sistemas móveis multiobjetivo não precisam escolher entre semântica e espaço;
5. arquitetura detalhada apenas depois.

Essa ordem se alinha melhor aos vencedores do que abrir com “propomos uma nova rede”.

