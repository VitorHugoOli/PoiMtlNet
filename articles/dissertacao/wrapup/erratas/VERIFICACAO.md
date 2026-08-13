# Verificacao dos numeros das erratas

Medido em 2026-08-13 contra os artefatos, nao contra prosa. Fonte de cada arm:
`docs/results/P1/region_head_{california,texas}_region_5f_50ep_v18_*_capmatched*_s0.json`;
comparador do modelo conjunto: `docs/results/closing_data/v18/joint_best_perfold.json`,
celulas `{state}_s0_joint`, campo `top10_full` por fold, mesma semente e mesma engine.

| afirmacao na errata | medido | confere |
|---|---|---|
| California: dedicado pareado acima do conjunto por $0.43$ Acc@10 | $+0.4277$ | sim |
| California: $p = 0.008$, unanime | $p = 0.0082$, unanime nos 5 folds | sim |
| Texas: diferenca de $0.21$ Acc@10 | $+0.2136$ | sim |
| Texas: $p = 0.12$, nao separado de zero | $p = 0.1162$, 4 de 5 folds | sim |
| braco estreito a $57$ por cento do orcamento | $5{,}014{,}942 / 8{,}809{,}533 = 56.9$ por cento | sim |
| passo de $352$ para $528$ nao separado de zero | $+0.021$, $p = 0.40$ | sim |
| categoria: $6.5\times$ os parametros baixa o macro-F1 | registro do controle de categoria, $-0.53$, $p = 0.0011$ | sim |

## Convencao de metrica

A metrica de regiao e `top10_acc` por fold. Verificado antes de qualquer comparacao: o arquivo do
dedicado reportado reproduz $63{,}4456$ a partir dos seus proprios folds por esse campo, que e o
valor da coluna Dedicated da tabela do Cap. 5 em California. Isso fixa qual campo usar e descarta a
leitura errada, que teria usado o valor bruto sem desconto de cobertura.

## Ausencias medidas

Nas tres extracoes completas de PDF e no fonte vivo com comentarios removidos:
`capacity-matched`, `confounded with capacity`, `several times the size`, `on three grounds`,
`fourth ground`, `forward-edge` tem **zero** ocorrencias no volume principal. `linear probe` tem
uma, dentro da tabela de errata. `separate study` tem zero nos dois volumes.

## Desvio registrado nos tres arms

`MTL_STRICT` omitido: o certificado de empate aborta a celula inteira por uma linha ambigua em
$585{,}091$ (0,00017 por cento). Os logs registram o aviso em vez do abort.
