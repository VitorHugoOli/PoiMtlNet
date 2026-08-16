# Q13 -- o controle de concatenacao na escala da Tabela 9

**Estado: EM EXECUCAO.** O controle de fidelidade ja passou, que era a barreira. Este documento
registra o que foi construido, a receita que reproduz, e os numeros conforme chegam.

## A pergunta

A Tabela 9 compara a representacao por check-in contra o place embedding. O controle de concatenacao
junta ao place embedding as mesmas features por visita que o grafo le, e pergunta quanto do intervalo
isso fecha. O estudo que a dissertacao cita rodou em outra preparacao e com outro harness, entao seus
valores nao estao na escala da Tabela 9: em Alabama ele reporta um ganho de $+2{,}0$, quando o
intervalo inteiro da Tabela 9 ali e $+1{,}62$. Este estudo roda na mesma escala.

## O desenho

Tres bracos por dataset, mesma semente, mesmas dobras, mesma receita, variando so a entrada:

| braco | entrada | serve para |
|---|---|---|
| `place` | place embedding, 9 blocos de 64 | **controle de fidelidade**: tem de reproduzir a coluna Place level |
| `feat` | place embedding + as 11 features por visita, 9 blocos de 75 | o controle de concatenacao |
| `checkin` | representacao por check-in, 9 blocos de 64 | tem de reproduzir a coluna Check-in level |

## A receita, e por que ela custou a ser achada

O valor reportado so reproduz com **`bs 8192`, `max_lr 0,0025` e `logit_adjust_tau 0,5`**, lidos do
sidecar do proprio resultado. Tres caminhos plausiveis nao reproduzem:

| caminho | fold 1 de Alabama | reproduz? |
|---|---|---|
| `run_hgi_ovl_cat_cell.sh`, o driver que nomeia esta celula (`bs2048`, `max-lr 3e-3`, sem logit adjustment) | $26{,}35$ | nao |
| o mesmo, com `bs8192` | $26{,}77$ | nao |
| o harness `p1_region_head_ablation.py` com a receita completa | $28{,}83$ | nao |
| **`train.py` com a receita do sidecar** | $\mathbf{29{,}7261}$ na epoca 16 | **sim, exato** |

O valor alvo e $29{,}7261$ na epoca 16. A convencao de selecao tambem importa: macro-F1 na epoca de
melhor macro-F1, lido dos CSVs de validacao. O JSON do harness traz o snapshot de `top10_acc`, que
daria $23{,}27$ no mesmo run.

## O que foi construido, e o que nao foi tocado

**`next_region.parquet` para as engines de place embedding.** O carregador exige essa tabela mesmo
quando o alvo e categoria, e ela nao existia para elas. Foi copiada de `check2hgi_dk_ovl`, mesmas
linhas na mesma ordem, sob o janelamento atual, com uma guarda de alinhamento verificada na escrita.
Isso e legitimo porque o carregador le dessa tabela apenas `region_idx` e `last_region_idx`: rotulos
derivados do mapa POI-para-regiao e do janelamento, nao do substrato. Verificado: as tabelas de
`check2hgi_dk_ovl` e `check2hgi_v18`, engines com embeddings diferentes, carregam `region_idx` e
`userid` identicos linha a linha. O proprio codigo ja usava esse argumento para as engines do estudo
`integrity_v2`.

**O construtor de features passou a aceitar o janelamento.** Ele reconstruia as janelas sempre com a
regra nao sobreposta, do estudo original, e abortava com razao ao encontrar 12.709 janelas onde as
sequencias atuais tem 96.326. Agora aceita `stride`, `min_sequence_length` e `emit_tail`; os
defaults reproduzem o comportamento anterior byte a byte, verificado.

**O braco de concatenacao virou engine propria, `hgi_ovl_feat`.** Seu `next.parquet` ja traz o place
embedding com as features concatenadas. Assim o braco roda pelo caminho de treino normal e
**nenhum codigo compartilhado foi alterado**: a alternativa seria adicionar uma flag a `train.py`,
que gera todos os resultados da dissertacao.

## Resultados

A preencher conforme os bracos terminam. O braco `place` de Alabama ja reproduziu o valor reportado.
