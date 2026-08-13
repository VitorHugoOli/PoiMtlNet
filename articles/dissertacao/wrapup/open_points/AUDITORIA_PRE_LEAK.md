# Auditoria de proveniencia: o que foi medido antes da correcao do vazamento

Pergunta do autor, 2026-08-13: o Q15 caiu porque descrevia uma auditoria feita sobre uma construcao
anterior da representacao. **Quantos outros pontos que eu levantei caem pelo mesmo motivo?**

A pergunta e boa e o risco e real: nove dos vinte estudos em `docs/results/closing_data/` rodaram na
preparacao anterior (`check2hgi_dk_ovl`) ou na receita v17, e **sete** deles ancoram afirmacoes
**vivas** do texto. Esta auditoria rastreia cada um e diz, por estudo, se o vazamento alcanca a
afirmacao.

## O criterio

O vazamento era especifico: sob v17, o vetor de uma visita era convolvido sobre uma vizinhanca que
incluia a visita **seguinte** (`METHODOLOGY.md:23`). Uma medicao so e contaminada se **le vetores**.
Uma medicao que le apenas rotulos, o stream de check-ins, ou o mapa POI-para-regiao passa por
convolucao nenhuma e e imune.

O proprio registro da o teste: um numero de categoria v18 que aterrissa perto do valor v17 indica
que o caminho forward-only esta quebrado (`METHODOLOGY.md:30-31`).

## Estudos pre-leak que sustentam afirmacoes vivas

| estudo | afirmacao viva que ele ancora | le vetores? | veredito |
|---|---|---|---|
| `markov_floor_stride1` | o piso de Markov de regiao, 51 a 72 Acc@10, excedido por 4,1 a 10,0 pontos (Cap. 5, p. 82) | **nao** | **IMUNE** |
| `h2_v17_cat_ceiling`, `catx_v17_n20` | o benchmark de historico de rotulos (Apendice D) | **nao** | **IMUNE** |
| `capacity_matched_stl_cat` | o controle de capacidade em categoria, $6.5\times$ os parametros baixa $0.53$ (Apendice G) | sim, mas **ambos os bracos** na mesma preparacao | **VALIDO como comparacao interna** |
| `baseline_compare` | os quatro baselines externos da tabela do Cap. 5 | sim, os proprios | **VALIDO**, e o texto declara que rodam nos proprios embeddings |
| `v18_place_level` | a coluna de place embedding da tabela de representacao | sim, `hgi_dk_ovl` | **VALIDO por desenho**: e o braco de comparacao, e o texto o nomeia |
| `horizon_stride1` | o horizonte de predicao, mediana de 0,4 h em Florida a 5,5 h em Istanbul, e 5 a 27 por cento dos alvos mais de tres dias depois (Cap. 5, secao 5.5.1) | **nao** | **IMUNE** |
| `a40`, `h100` | nenhuma afirmacao viva casa | — | sem exposicao |

### Por que o piso de Markov e imune, com a medicao

O registro declara o que ele le: `checkin_stream`, `poi_region_mapping` e a contagem de check-ins
descartados por POI sem mapeamento. Nenhum vetor. E uma tabela de transicao de primeira ordem sobre
rotulos de regiao.

E o corpus nao mudou entre as preparacoes. Alabama, no registro do piso: **113.846** check-ins,
**96.326** janelas, **1.109** regioes. A tabela de datasets do Cap. 5, linha de Alabama: 113.846,
96.326, 1.109. Identicos. A janela tambem: tamanho 9, stride 1, sequencia minima 10, sem cauda.

### Por que o teto de categoria e imune, com a citacao

O proprio artefato declara o escopo no campo `what`: *"label-only autocorrelation ceiling for
next-category (no embeddings read)"*. O protocolo confirma: `GroupKFold(5)` por usuario, macro-F1,
e o teto e o melhor entre persistencia, ultima categoria, contagens de janela e posicional. Todos
sao funcoes de rotulos.

### Por que o horizonte de predicao e imune, com a medicao

**Correcao de classificacao.** Numa primeira passagem eu agrupei este estudo com os que nao tinham
exposicao. Estava errado: ele ancora uma afirmacao viva na secao 5.5.1, e a sonda que eu mesmo rodei
havia impresso tres correspondencias que eu nao examinei antes de classificar.

A quantidade medida e `prediction_horizon_hours`, definida no artefato como
`timestamp(target) - timestamp(last of the 9 window visits)` por janela. E uma **diferenca de
timestamps entre visitas**: nenhum vetor e lido, entao a convolucao de vizinhanca nao a alcanca. O
artefato tambem carrega um `markov_floor_crosscheck` com `exact_match: true` contra o registro do
piso, o que amarra as duas contagens de janela.

As duas clausulas do texto conferem contra o artefato: mediana de **0,4375 h** em Florida (texto:
0,4) e **5,5000 h** em Istanbul (texto: 5,5); e a fracao de alvos alem de 72 horas vai de **4,91 por
cento** em Florida a **27,03 por cento** em Istanbul (texto: 5 a 27).

### Por que o controle de capacidade em categoria continua valendo

Ele roda na preparacao anterior, sim. Mas a comparacao e **interna e pareada**: os dois bracos, o
dedicado estreito e o dedicado alargado, leem a mesma representacao. O que ele mede e o efeito de
largura com representacao fixa, e essa quantidade nao depende de qual preparacao esta fixa. O
proprio apendice declara o limite: *"it holds the representation fixed and varies width"*.

## O que esta auditoria mudou

**Nada no texto.** Nenhuma afirmacao viva perde suporte. O Q15 era o caso genuinamente afetado, e
caiu porque a auditoria que ele descrevia **le vetores** e foi medida em construcoes anteriores.

**Uma correcao ao meu proprio trabalho desta sessao.** Verifiquei que os tres arms de P1 usaram
exatamente as duas fontes da celula bancada, `--engine-override check2hgi_v18` e
`--region-emb-source check2hgi_design_k_resln_mae_l0_1`, identicas a `ENG` e `V14` do driver v18
(`run_wave.sh:29-30`). Se tivessem divergido, a comparacao pareada de P1 seria invalida e eu nao
teria notado: o arm nao registra a engine no seu proprio JSON de config.

## O que esta auditoria NAO cobriu

- Os numeros dos Caps. 3 e 4. Aqueles capitulos usam DGI e HGI sobre place embeddings, uma
  representacao diferente, e o prefacio de cada um data suas conclusoes. Nao foram rastreados aqui.
- `apxi_v18`, `reg_ceiling_n20` e `v18_place_level` nao carregam marca de engine nos JSON que eu li;
  os dois primeiros nao casam com afirmacao viva, e o terceiro foi resolvido pelo campo `engine`.
### Os dois sem exposicao: por que, e como foi testado

`a40` e `h100` **nao** ficaram sem exposicao por ausencia de correspondencia numerica. Testados por
valor, 23 e 17 dos seus numeros casam com algum numero em prosa viva, o que e esperado: sao estudos
sobre as mesmas quantidades, na mesma faixa, e uma busca por valor sozinha nao distingue coincidencia
de citacao.

O teste que decide e de **proveniencia**, nao de valor. A celula reportada de California, semente 0,
declara `rundir: results/check2hgi_v18/california/mtlnet_lr1.0e-04_bs8192_ep50_20260809_185016_843391`
e `lane_host: local:NVIDIA A40`, ou seja: vem da execucao v18, nao do diretorio `a40/` nem do `h100/`.
O `h100/` carrega um `california_s0_board_partial.json` com `status: PARTIAL (2/5 folds)`, que por
construcao nao pode ser a fonte de uma celula de cinco folds.

Os dois diretorios sao registros de execucoes de pareamento de hardware, e a afirmacao viva que os
usa esta em `ORA-5` do registro de lacunas, respondida como divulgacao e nao como resultado.

## O corpus nao mudou: verificado nos seis datasets

A duvida legitima e se o corpus ou as janelas mudaram entre as preparacoes, porque isso invalidaria
o piso mesmo sendo ele imune ao vazamento. Conferido registro contra a tabela de datasets do Cap. 5:

| dataset | check-ins (piso / tabela) | janelas (piso / tabela) | regioes | confere |
|---|---|---|---|---|
| Alabama | 113.846 / 113.846 | 96.326 / 96.326 | 1.109 | sim |
| Arizona | 236.450 / 236.450 | 200.895 / 200.895 | 1.547 | sim |
| Florida | 1.407.034 / 1.407.034 | 1.274.418 / 1.274.418 | 4.703 | sim |
| Texas | 4.089.892 / 4.089.892 | 3.830.414 / 3.830.414 | 6.553 | sim |
| California | 3.171.380 / 3.171.380 | 2.925.466 / 2.925.466 | 8.501 | sim |
| Istanbul | 462.615 / 462.615 | **270.217 / 271.666** | 520 | check-ins e regioes sim; janelas divergem 0,53 por cento |

**A divergencia de Istanbul ja estava medida pelo proprio registro**, que carrega um campo
`window_count_gate` com os dois valores, a razao de 0,9947 e o veredito `within_1pct: true`, citando
a tabela do manuscrito como fonte de comparacao. Ou seja: quem produziu o piso antecipou a duvida,
mediu a diferenca e a declarou dentro do proprio limite de tolerancia. Nao e um achado novo desta
auditoria; e uma discrepancia conhecida e registrada, de meio por cento, no menor dataset.

## O que esta auditoria NAO cobriu

- Os numeros dos Caps. 3 e 4. Aqueles capitulos usam DGI e HGI sobre place embeddings, uma
  representacao diferente, e o prefacio de cada um data suas conclusoes. Nao foram rastreados aqui.
- `apxi_v18` e `reg_ceiling_n20` nao carregam marca de engine nos JSON que eu li; nenhum dos dois
  casa com afirmacao viva.
- A causa da divergencia de 0,53 por cento em Istanbul. O registro a declara dentro da tolerancia e
  eu nao a rastreei ate a linha que a produz.
