# erratas

Erratas escritas **depois** de o texto ser concluido e enviado. Nada aqui entra em nenhum build.

Cada errata e um `.tex` autocontido, escrito na forma em que entraria no documento, com um
cabecalho de comentario que diz o que ela corrige, onde a frase original vive, e como cada numero
foi verificado. Quando o deposito final for montado, cada arquivo tem um destino declarado no
proprio cabecalho.

| arquivo | corrige | estado |
|---|---|---|
| `errata_Q13_concatenation_scope.tex` | a frase do Cap. 5, p. 79, que afirma uma separacao que o artigo submetido recusa sobre os mesmos numeros | escrita; o ponto so fecha de vez com um estudo de concatenacao na representacao final |
| `errata_Q14_capacity_region.tex` | a lista de limites do Cap. 5, que perdeu o confundimento de capacidade no eixo de regiao | escrita, e ja traz a medicao que o artigo declarava faltar |
| `errata_resumo_escopo_categoria.tex` | o Resumo em portugues, que dizia que o modelo conjunto superou os dedicados na proxima categoria "em todos os conjuntos" quando o resultado entregue e superioridade em UM conjunto | escrita e **ja aplicada ao fonte** (2026-08-21, decisao do autor). O `dissertacao.pdf` entregue nao foi reconstruido |
| `RESPOSTAS_ORAIS.md` | a frase a ser dita em pe para cada uma, mais Q15 | — |
| `VERIFICACAO.md` | cada numero das erratas contra o artefato de origem | — |

## Q24 nao gerou errata

Eu havia registrado como defeito o fato de o volume principal nao apontar para o material
extra. Nao e defeito: o texto depositado cita apenas o proprio texto e o repositorio, sem
referencias externas, e isso e a politica do documento. O material extra foi escrito como apoio
a defesa e agora vive em `../material_extra/`.

## Material extra

`material_apx_static_scope.tex` veio de `src_fix/chapters/`, onde existia sem ser incluido em
nenhum build. Ele esta aqui porque **parte dele serve a argumentacao oral** e nada dele esta nos
dois volumes: documenta um canal de autovazamento medido no embedding do Cap. 3, com peso medio de
0,10 contra 0,39 da propria categoria, e uma sonda caindo de 0,46 para 0,30 macro-F1 contra piso de
0,07.

Duas ressalvas antes de usar qualquer numero dele: a medicao e sobre a **preparacao anterior** da
representacao, e o proprio arquivo carrega um pedido de aval do autor que nunca foi respondido.
Nenhum arguidor faz essa pergunta sem o repositorio, mas com o repositorio ela e a unica coisa que
os dois documentos nao antecipam.

## Q15 nao gerou errata

A linha de errata do suplemento que descreve um quarto fundamento de integridade, com sonda linear
em Florida, descreve uma auditoria sobre construcoes anteriores da representacao, e a propria linha
declara esse limite. A correcao e retirar a linha, nao adicionar o fundamento ao texto. Detalhe em
`RESPOSTAS_ORAIS.md`.
