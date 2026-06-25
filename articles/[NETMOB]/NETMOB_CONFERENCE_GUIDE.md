# Guia de submissão e leitura editorial da NetMob

**Atualização:** 18 de junho de 2026  
**Escopo empírico da pesquisa:** edições 2023, 2024 e 2025; regras correntes de 2026; histórico
oficial desde 2010.  
**Nota de método:** este documento separa regras publicadas pela organização de inferências feitas
a partir dos programas, livros de abstracts e trabalhos premiados. A NetMob não publica uma rubrica
detalhada de revisão; qualquer “critério de avaliação” mais específico que as regras oficiais é,
portanto, uma reconstrução fundamentada, não uma política oficial.

## 1. O que é a NetMob

A NetMob se apresenta como a principal conferência dedicada ao uso de dados móveis e comportamentais
em problemas sociais, urbanos, societais e industriais. O núcleo histórico é a análise de dados de
telefonia móvel, mas o escopo atual inclui CDR/xDR, localização móvel e GNSS, Wi-Fi, uso de apps,
redes sociais e outros dados comportamentais em larga escala.

O formato é incomum e deve orientar toda a estratégia de escrita:

- **single track**, com comunidade interdisciplinar;
- **talks curtos e posters**, em vez de sessões paralelas de artigos completos;
- submissão principal como **abstract de duas páginas**;
- aceitação explícita de **pesquisa em estágio inicial** e de **trabalho publicado ou submetido em
  outro lugar**;
- publicação em um **Book of Abstracts**, e não, pelo que as páginas oficiais indicam, em proceedings
  arquivais convencionais com artigo completo.

Consequência: o abstract deve vender uma descoberta científica clara e discutível em poucos segundos.
Não tente comprimir um paper de oito páginas para duas; selecione uma pergunta, um mecanismo, uma
evidência principal e uma implicação.

Fontes: [NetMob 2026](https://netmob.org/),
[NetMob 2025](https://netmob.org/www25/),
[NetMob 2024](https://netmob.org/www24/),
[NetMob 2023](https://netmob.org/www23/).

## 2. Regras formais correntes (NetMob 2026)

### Main Conference

- PDF eletrônico.
- Máximo de **duas páginas**, incluindo figuras, tabelas e referências.
- Sem apêndices.
- Título, autores, afiliações e e-mails na primeira página.
- Uso obrigatório do template oficial em
  [LaTeX/Overleaf](https://pt.overleaf.com/read/hzrxdnwxvwtp) ou
  [Word](https://docs.google.com/document/d/15tJsLFyojAyJoF69g_5eBhZXpkq6-ob7/edit).
- O site alerta que a não conformidade pode causar rejeição **antes da revisão**.

O texto não é anônimo: a regra exige identificação dos autores na primeira página. O site não publica,
contudo, uma descrição do regime de revisão, número de revisores, critérios pontuados ou taxa de
aceitação. Não assuma double blind nem uma rubrica semelhante à de ACM/IEEE.

### Datas anunciadas para 2026

| Evento | Data no site em 18/06/2026 | Estado |
|---|---:|---|
| abertura do sistema | 15 maio 2026 | anunciada |
| submissão do abstract | **1 julho 2026** | tentativa |
| notificação da Main Conference | 29 julho 2026 | tentativa |
| camera-ready da Main Conference | 15 agosto 2026 | tentativa |
| conferência, UFF, Niterói | **14–16 outubro 2026** | anunciada |

O sistema indicado é o JEMS/SBC. Como o próprio site chama as datas de “tentative”, deve-se verificar
novamente a [página oficial](https://netmob.org/) e o sistema de submissão antes do envio.

### Data Challenge 2026

O desafio usa dados de ônibus de Niterói: rastreamento veicular de alta frequência, demanda de
passageiros, linhas e embarques individuais, com dados auxiliares. A primeira etapa também exige um
extended abstract de duas páginas com resultados preliminares. Selecionados entregam um relatório
confidencial completo e um resumo público de duas páginas.

**Inconsistência oficial encontrada:** a descrição textual informa relatório final em **20/08/2026**,
enquanto o cronograma da mesma página informa **06/09/2026**. Não escolher uma das datas por conta
própria; confirmar com `netmob2026@midiacom.uff.br`.

## 3. O que a conferência provavelmente avalia

Não há rubrica oficial pública. A matriz abaixo é uma ferramenta interna, reconstruída a partir do
escopo declarado, da seleção talk/poster e dos trabalhos premiados de 2023–2025.

| Dimensão inferida | O que demonstra força | Evidência histórica |
|---|---|---|
| **Aderência** | dado móvel/comportamental é central para responder a pergunta, não apenas decorativo | escopo oficial e composição de todas as sessões |
| **Pergunta relevante** | problema social, urbano, operacional ou de política pública bem delimitado | prêmios em causalidade urbana, acessibilidade, roteamento e desigualdade |
| **Novidade do insight** | mecanismo, medida ou método simples de explicar e diferente de mera aplicação | mistura adaptativa individual/coletiva; acessibilidade baseada em capacidades |
| **Credibilidade empírica** | dados descritos, desenho de validação, baselines, incerteza e testes de robustez | premiados reportam comparações, múltiplas cidades/grupos ou caminhos causais |
| **Impacto interpretável** | efeito quantificado e implicação concreta, não somente ganho de métrica | política urbana, emissões, lazer, eficiência e resiliência |
| **Consciência de viés/ética** | representatividade, privacidade, cobertura e limites declarados | sessões inteiras sobre bias, fairness e privacy em 2023 e 2025 |
| **Comunicação visual** | uma figura legível que contém a descoberta principal | formato de duas páginas e apresentação oral de 12 minutos em 2023/2025 |

### Rubrica interna recomendada

Use-a para revisar o abstract; ela **não é da organização**.

| Item | Peso sugerido |
|---|---:|
| fit com NetMob e importância da pergunta | 20% |
| contribuição/novidade claramente formulada | 20% |
| rigor e adequação dos dados/métodos | 25% |
| força, robustez e interpretabilidade dos resultados | 20% |
| impacto social/urbano/industrial e limitações éticas | 10% |
| clareza, figura e conformidade | 5% |

Uma submissão forte deve alcançar pelo menos 4/5 em fit, contribuição, rigor e resultado. Um ganho de
predição sem explicação comportamental ou implicação urbana tende a ser menos memorável que um ganho
associado a um mecanismo e a uma decisão prática.

## 4. Estilo editorial preferido

### Estrutura eficaz para duas páginas

1. **Título com descoberta**, não apenas com o nome do modelo.
2. **Abertura (3–5 frases):** problema, por que importa, lacuna e pergunta.
3. **Dados:** população/amostra, período, cobertura espacial e temporal, unidade de análise, filtros,
   privacidade e viés de seleção.
4. **Método:** apenas o necessário para entender a comparação ou o mecanismo; inclua desenho de
   validação e baselines.
5. **Resultado central:** números absolutos e deltas, incerteza/variabilidade, ablação ou robustez.
6. **Implicação e limites:** o que muda para ciência/política/operação e onde a conclusão não vale.
7. **Referências mínimas:** priorize o trabalho mais próximo e a fonte dos dados.

### Alocação prática de espaço

- 15–20%: motivação, lacuna e contribuições;
- 20–25%: dados e desenho experimental;
- 30–35%: resultados;
- 10–15%: discussão, limitações e implicações;
- restante: título/autores, figura/tabela e referências.

### Figuras e tabelas

Prefira **uma figura principal** com uma frase conclusiva no caption. Ela deve continuar legível em
100% de zoom no PDF e responder a pergunta do paper. Uma tabela compacta pode conter baselines,
média ± dispersão e delta; evite cinco gráficos pequenos que exigem uma apresentação oral para serem
entendidos.

### Tom

- interdisciplinar: defina siglas e evite jargão de arquitetura sem função argumentativa;
- orientado a resultado: coloque a descoberta antes da implementação;
- preciso: “paridade dentro de X pp” é melhor que “comparável”; “associação” não é “causalidade”;
- honesto sobre cobertura e representatividade;
- conciso, mas não telegráfico: o leitor deve reconstruir o desenho do estudo.

## 5. Talk, poster e awards

Em 2023 e 2025, talks receberam **12 minutos + 3 minutos de perguntas**. Posters foram A0 vertical.
A organização decide talk versus poster; no Data Challenge, o relatório completo também participa
dessa decisão. A edição de 2025 distinguiu quatro categorias: Best Main Conference Paper, Best Main
Conference Poster, Best Data Challenge Paper e Best Data Challenge Poster, além de divulgar top-3.

O formato de fala reforça a necessidade de uma narrativa de três atos: pergunta, evidência, consequência.
Um abstract selecionável para talk deve conter um resultado suficientemente fechado para sustentar
12 minutos, mesmo quando a pesquisa geral ainda está em andamento.

## 6. Temas que ganharam centralidade recentemente

Os programas de 2023–2025 mostram uma expansão de “analisar CDRs” para ciência comportamental e urbana
mais ampla:

- desigualdade, segregação, pobreza e acessibilidade;
- crises, desastres, epidemias e resiliência;
- mobilidade sustentável, emissões e transporte;
- viés, representatividade, fairness e privacidade;
- previsão de trajetórias e modelos generativos;
- dados móveis para estatística oficial e política pública;
- tráfego de rede, consumo de apps e infraestrutura celular.

O fio comum é: **dados móveis em larga escala + questão substantiva + método defensável + implicação**.

## 7. Riscos e pontos que precisam de confirmação

- A organização não publica rubrica, taxa de aceitação ou política detalhada de revisão.
- A exigência de template começou explicitamente em 2025 e permanece em 2026; não reutilize um layout
  antigo apenas porque aparece em livros anteriores.
- O status não arquival e a permissão para trabalho publicado em outro lugar devem ser checados contra
  a política do outro venue e de direitos autorais.
- Datas de 2026 são tentativas, e o Data Challenge tem conflito de datas no próprio site.
- “Best paper” e “best contribution” variam na nomenclatura entre edições; não trate todas as menções
  institucionais como uma categoria oficial idêntica.
- Como o Book of Abstracts limita detalhes, qualquer claim causal, de generalização ou de superioridade
  precisa caber com a evidência mínima no próprio abstract.

## 8. Recomendação para o paper deste repositório

O projeto `netcore` combina next-category e next-region com dados de check-ins. Para a NetMob, a história
mais adequada não é uma enumeração de componentes de deep learning. A melhor formulação é:

> **Pergunta:** um único modelo pode aprender simultaneamente a semântica e a geografia da próxima
> visita sem sacrificar a tarefa espacial?  
> **Descoberta:** uma representação hierárquica por check-in gera ganho de categoria enquanto uma
> torre espacial privada preserva a região; o suposto trade-off era parcialmente um confound de loss.  
> **Consequência:** separar substrato contextual de compartilhamento de tarefas oferece uma regra de
> desenho para sistemas de mobilidade multiobjetivo.

Para a submissão, reduza a quatro resultados: baseline STL, modelo conjunto, ablação do substrato e
ablação/confound da loss. Deixe a exploração extensa de otimizadores MTL para uma linha ou material
posterior. Inclua uma figura Pareto category × region e uma tabela mínima com estados/seeds.

