# Best papers e papers de destaque da MobiWac

## Critério de inclusão

- **Premiado confirmado:** anúncio oficial ou institucional inequívoco.
- **Finalista confirmado:** lista oficial de candidatos.
- **Destaque analítico:** paper recente escolhido pela força metodológica e aderência;
  não implica prêmio.

## Visão anual

| Ano | Status | Paper | Por que importa |
|---:|---|---|---|
| 2020 | Best Paper confirmado | *The Impact of COVID-19 Confinement on Regional Mobility of Spatial-Temporal Social Networks* — Munairah Aljeri | problema urgente, dados reais de mobilidade e interpretação regional |
| 2021 | Best Paper confirmado | *Minimizing Rate Variability with Effective Resource Utilization in 5G Networks* — Fidan Mehmeti, Thomas F. La Porta | trade-off operacional claro, formulação e validação orientada a recursos/QoS |
| 2022 | Best Paper confirmado | *Environment-Aware Link Quality Prediction for Millimeter-Wave Wireless LANs* — Yuchen Liu, Douglas M. Blough | geometria + DNN + ray tracing + experimento 802.11ad |
| 2023 | Destaque analítico; prêmio não localizado | *Gated Recurrent Units for Blockage Mitigation in mmWave Wireless* — Ahmed Almutairi, Alireza Keshavarz-Haddad, Ehsan Aryafar | decisão de mitigação ponta a ponta, >93% e dados/código públicos |
| 2025 | Destaques analíticos; prêmio não localizado | *REMLAB*; *Backpack-LoRa*; *On the Design of Mobility-Aware Systems* | framework full-stack, eficiência energética e projeto explícito de sistemas mobility-aware |

Não houve MobiWac 2024 no histórico oficial.

## 2020 — mobilidade regional durante COVID-19

O anúncio oficial de 2021 identifica o trabalho de Munairah Aljeri como vencedor de 2020.
O paper tem sete páginas (pp. 29–35) nos proceedings.

### Sinais editoriais

- pergunta imediatamente relevante e compreensível;
- dados de redes sociais usados como proxy de mobilidade;
- comparação espacial e temporal antes/durante confinamento;
- resultado com implicação social, não apenas métrica de modelo;
- escopo curto e história única.

### Cautela

É um vencedor atípico pelo contexto pandêmico. Não conclua que a MobiWac prefere estudos
sociais genéricos; o paper ainda trata mobilidade, localização e dados de sistemas móveis.

Fontes: [anúncio oficial](https://mobiwac-symposium.org/2021/index.html),
[DBLP MobiWac 2020](https://dblp.org/db/conf/mobiwac/mobiwac2020).

## 2021 — variabilidade de taxa e uso de recursos em 5G

Três finalistas foram publicados oficialmente:

1. *Minimizing Rate Variability with Effective Resource Utilization in 5G Networks*
   — vencedor;
2. *Desynchronization and MitM Attacks Against Neighbor Awareness Networking Using
   OpenNAN*;
3. *An Innovative Neuro-Genetic Algorithm and Geometric Loss Function for Mobility
   Prediction*.

### Por que o vencedor é um bom modelo

O título explicita o trade-off. O problema é formulado em termos de experiência do usuário
e eficiência da operadora. A linha de trabalho foi depois ampliada e validada com trace 5G,
simulações e OpenAirInterface, reportando redução de desperdício e ganho contra estado da
arte. Isso mostra uma preferência por contribuição que combine teoria, política de
alocação e evidência de sistema.

### O que os finalistas revelam

A shortlist cobre três estilos aceitos pela conferência:

- otimização e gestão de recursos;
- segurança prática de protocolo com implementação open-source;
- aprendizagem para previsão de mobilidade.

O denominador comum é uma decisão concreta de sistema e avaliação verificável.

Fontes: [shortlist oficial](https://mobiwac-symposium.org/2021/index.html),
[confirmação do prêmio](https://www.ce.cit.tum.de/lkn/aktuelles/single-view/article/best-paper-award-for-fidan-mehmeti/).

## 2022 — previsão de link mmWave com consciência do ambiente

O paper separa LoS e NLoS por análise geométrica, usa modelo analítico no caso simples e
regressão/DNN no caso difícil. O treinamento usa cenários de ray tracing; a validação inclui
cenários sintéticos adicionais e um ambiente real 802.11ad.

### Por que é exemplar

- o problema vem antes da IA;
- a arquitetura incorpora conhecimento de domínio;
- o método é modular e explicável;
- a avaliação testa generalização entre cenários;
- há ponte entre simulador e medição real;
- o resultado habilita alocação proativa, não termina na previsão.

Fontes: [confirmação institucional](https://ece.gatech.edu/news/2023/12/top-prize-awarded-blough-liu-international-symposium-mobility-management-and-wireless),
[PDF público do autor](https://blough.ece.gatech.edu/research/papers/mobiwac22.pdf),
[proceedings](https://doi.org/10.1145/3551660).

## 2023 — destaques sem vencedor publicamente verificável

Não foi localizado anúncio oficial/institucional inequívoco do Best Paper de 2023. O site
oficial fornece programa e proceedings, mas não lista prêmio. Portanto, nenhum paper deste
ano é rotulado como vencedor neste dossiê.

O destaque selecionado, *Gated Recurrent Units for Blockage Mitigation in mmWave
Wireless*, é útil para estudar estilo porque:

- pergunta qual ação tomar, não apenas se haverá blockage;
- escolhe entre beam switching, handoff e beam widening;
- usa mensagens periódicas já existentes, evitando overhead adicional;
- reporta acurácia superior a 93% e dados/software públicos;
- mede volume de dados transferidos contra políticas alternativas.

Fontes: [programa 2023](https://mobiwac-symposium.org/2023/program.html),
[PDF público](https://web.cecs.pdx.edu/~aryafare/2023-MobiWac-GRU.pdf),
[repositório experimental](https://mmw.cs.pdx.edu/repository.html).

## 2025 — transição IEEE e papers de destaque

Não houve edição em 2024. A 22ª edição ocorreu em 2025 e seus papers aparecem nas sessões
MobiWac do volume IEEE MSWiM 2025. Não foi localizado anúncio de Best Paper.

Três trabalhos são especialmente úteis como referência:

- **REMLAB:** framework ns-3 full-stack para beam management 5G-NR baseado em radio
  environment maps; sinaliza valor de artefato integrado e reprodutível.
- **Backpack-LoRa:** protocolo multi-hop LoRaWAN com foco explícito em energia e operação;
  exemplo de contribuição de protocolo medida em objetivo de sistema.
- **On the Design of Mobility-Aware Systems: A Tourist's Perspective:** aproxima-se do tema
  deste repositório e mostra como mobilidade precisa virar requisito/decisão de sistema.

Fontes: [programa 2025](https://mobiwac-symposium.org/2025/program.html),
[sumário do volume IEEE](https://www.proceedings.com/content/083/083700webtoc.pdf).

## Padrão agregado

O que mais se repete entre premiados e destaques:

1. uma tensão operacional clara;
2. conhecimento de domínio embutido no método;
3. decisão de rede mensurável;
4. avaliação em múltiplas camadas;
5. baselines/políticas, não apenas arquiteturas de ML;
6. resultado quantitativo que muda QoS, energia, robustez, segurança ou utilização;
7. história comunicável em uma sessão curta.

O paper ideal para MobiWac é um paper de **sistema/rede móvel com método rigoroso**, não um
paper de ML que por acaso usa dados móveis.

