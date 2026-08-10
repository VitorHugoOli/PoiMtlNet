# Future work — migrar o loop de época/val do p1 para `src/training/`

**Origem:** investigação da alternância CPU/GPU na célula reg dedicada
([`docs/studies/closing_data/v18/INVESTIGATION_gpu_cpu_alternation.md`](../studies/closing_data/v18/INVESTIGATION_gpu_cpu_alternation.md)).
**Status:** proposto, NÃO iniciar mid-board — é trabalho de **freeze boundary** (a mesma
regra do log v18 §54: nunca trocar o caminho de scoring no meio de um pool bancado).

## O problema estrutural

`scripts/p1_region_head_ablation.py` carrega uma **cópia própria** do loop de treino/val
(`_train_single_task`, ~280 linhas no script). Foi essa cópia que ficou meses atrás dos
fixes de `src/` e produziu a classe de bug investigada: o scoring de validação CPU-full-logit
(por-batch `.cpu()` síncrono + 4 passes de CPU sobre `[N_val, C]`) sobreviveu no p1 depois
de o MTL já ter sido corrigido (S1 em 2026-06-24; S2 sempre-GPU), custando ~39 % da wall
da célula AZ e segundos por época em CA/TX, com a GPU serrilhando 10-57 % de utilização.

O sintoma se repete porque a causa é estrutural: **qualquer driver de experimento que
reimplemente época/val à mão pode divergir de `src/` silenciosamente**. O refactor de
2026-08-10 centralizou o *scoring* (`StreamingClsMetrics` é hoje o único caminho de alta
cardinalidade, consumido por mtl_eval/mtl_cv/p1/`_single_task_train` + 5 baselines e2e),
mas o **loop** em si continua duplicado no p1.

## O que migrar

1. **Extrair um helper de val streamado** em `src/training/shared_evaluate.py` —
   `score_val_streaming(model, val_dl, n_classes, top_k) -> dict` — encapsulando:
   decisão `should_stream`, o loop `no_grad` por batch, o certificado de ambiguidade
   por época, e os knobs de device (`P1_STREAM_GPU`-equivalente). O p1, os baselines
   e2e e futuros drivers passam a chamar isso em vez de re-escrever o loop de val.
2. **(Mais ambicioso) aposentar o `_train_single_task` do p1** delegando ao
   `train_single_task` de `src/training/runners/_single_task_train.py`, que já streama
   para C>256 pós-refactor. Exige reconciliar diferenças reais entre os dois loops:
   - otimizador: p1 usa AdamW(lr=1e-4, wd=0.01) vs helpers de src (wd=0.05);
   - loss: p1 usa `build_calibrated_loss` com stats do train-fold;
   - RNG: p1 re-semeia por fold DENTRO do `_train_single_task` (`seed + fold_idx`) —
     é isso que torna o fan-out `--only-fold` byte-idêntico; o runner de src não tem
     esse contrato;
   - tracking: p1 escreve JSON de checkpoint próprio (`per_metric_best`), não MLHistory.

## Critérios de aceite (inegociáveis)

- **Paridade byte-idêntica em eager** contra uma célula bancada do board (AZ s0 reg:
  provado nesta investigação que um rerun no protocolo reproduz o JSON bancado
  bit a bit com `MTL_SKIP_INERT_PRIOR=0` — esse é o golden de referência).
- O contrato de reseed por fold preservado (fan-out `--only-fold` continua 12-casas
  idêntico ao sequencial).
- Smoke de integração em `tests/test_integration/` que EXECUTA o caller de ponta a
  ponta (1 fold × 1 época, dados sintéticos) — a lição do `UnboundLocalError` que
  passou por unit test + review porque nada rodava o loop real.

## Quando

No próximo freeze boundary do closing_data (com o board §1 fechado), junto com a
homogeneização já adiada do scoring (log v18 §54). Ganho esperado: zero em wall
(o scoring já é GPU); o valor é impedir a próxima recorrência da classe de bug.
