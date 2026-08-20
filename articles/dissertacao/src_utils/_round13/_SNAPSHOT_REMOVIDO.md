# _snapshot/ foi removido em 2026-08-20

A rodada 13 guardava aqui uma **copia completa da arvore de fonte** (`_snapshot/src/`, 53 arquivos,
2,3 MB) para comparar antes-e-depois. Ela foi apagada, e por um motivo que vale registrar.

O caminho dela terminava em `src/tables/mobiwac/results.tex` — **identico ao caminho da arvore
entregue**. E o conteudo eram os numeros da geracao v17, os que o vazamento de rotulo inflava:

    AL 56.82 / 64.51      FL 74.51 / 79.84      TX 69.79 / 77.24      CA 70.60 / 77.05

contra os entregues, que sao `AL 30.77 / 30.59`, `FL 37.35 / 37.55`, `TX 36.33 / 36.19`,
`CA 35.63 / 35.63`. Depois que `src_fix` virou `src`, um grep pela tabela passava a achar as duas
copias, e a errada nao se anunciava como errada em lugar nenhum.

**Nenhum portao lia o snapshot**: a varredura por `_round13/_snapshot` em `src_utils/*.py` e
`src_utils/*.sh` so encontra dois comentarios. Os achados da rodada 13 continuam aqui, nos relatorios
`.md` irmaos, que sao o registro de verdade — o snapshot era o insumo deles, nao a conclusao.

Recuperar, se algum dia for preciso:

    git checkout dissertacao-pre-reorg -- articles/dissertacao/src_utils/_round13/_snapshot
