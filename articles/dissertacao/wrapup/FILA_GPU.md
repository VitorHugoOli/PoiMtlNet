# Fila de GPU — o que corre quando houver máquina

> Criada a 2026-10-01 a pedido do autor, "para não se perder". O mesmo conteúdo está no dossiê do
> depósito. **O nespedgpu não serve:** reiniciou quatro vezes a 2026-09-30, sempre segundos depois
> de a GPU arrancar com a nossa carga real (05:19, 05:22, 16:46, 21:28). Um teste de matmul constante
> a 300 W passou, e a nossa carga irregular a ~100 W derruba a máquina. Registos em
> `/dados/poimtlnet/integrity_v18/logs/`. Precisa de diagnóstico de hardware feito por quem o gere.

Cada item diz porquê, o que correr, o que muda no texto e o que fica se nunca correr.

## G1 · Verificação de integridade no modelo entregue — BLOQUEIA O DEPÓSITO

- **Porquê.** A verificação "treinar a representação com todos os utilizadores dá vantagem?" do
  Cap. 5 vem do A4 dos pre-freeze gates. Esse A4 usou outra versão da representação e outro
  protocolo: janelas stride-9, prior de transição ligado, 30 épocas em CPU e outra divisão de
  utilizadores, em que 74–82% dos utilizadores de validação de cada fold entregue estavam do lado
  de treino do A4 (AL fold 0: 183 de 222; medido contra os folds reais do `train.py`).
  A 2026-09-30, por decisão do autor, saiu do texto o "on an earlier build". Os números impressos
  (região −0,33…+0,01, categoria 0,00…+0,29) continuam os antigos até esta execução terminar.
- **O que correr.** Reconstruir a representação por fold só com os utilizadores de treino, em
  AL/AZ/FL com a seed 0. Categoria: ~4–6 GPU-h. Região: ~3–5 GPU-h. Enviar ~18 GB e ter ≥100 GB
  livres. Começar por um fold de AL para medir o tempo.
- **Muda no texto.** `05_setup` ("Whole-dataset training") e o primeiro limite de `07_discussion`,
  mais a linha da B.5 (desatualizada de propósito) e a sua classe na prosa do Apêndice B.
- **Se nunca correr.** Repor a menção à versão anterior como exceção declarada. O estado atual não
  pode ir para o depósito.
- **Material.** `docs/studies/closing_data/v18/integrity_rerun/`: plano, 8 verificações e três
  patches por aplicar. ⚠ **As 15 divisões em `/dados/poimtlnet/integrity_v18/splits/` NÃO são os
  folds entregues.** O `freeze_split.py` passava os userids como texto, e o `train.py` passa-os como
  int64, o que dá outra partição (0/5 folds iguais, reproduzido). Regenerar com
  `--group-dtype int` (patch no pacote) e confirmar com a verificação #5.

## G2 · CTLE em AL/AZ/IST com a divisão certa — decidido (opção 1 se houver GPU)

- **Porquê.** O braço CTLE "congelado" foi pré-treinado com uma divisão escolhida sobre o ficheiro
  stride-9, enquanto a avaliação usa as linhas stride-1. Resultado: 75–83% dos utilizadores
  avaliados estavam no pré-treino (reproduzido em AL e AZ; IST ~80%). Isso favorece o CTLE. O braço
  da Florida, treinado de ponta a ponta, está certo.
- **O que correr.** Pré-treinar o CTLE por fold sobre a divisão das nossas células e reavaliar a
  categoria, na seed 0. ~2–5 GPU-h, mais 2 GB.
- **Muda no texto.** Pode voltar a frase de que o CTLE repete a ordem nos três conjuntos. Nenhum
  número impresso muda, porque só se imprime o CTLE da Florida.
- **Se nunca correr.** A regra do autor: a conclusão vem da Florida, dita de forma simples. A
  proposta de texto está pronta e espera aprovação.
- **Material.** Mudanças em `scripts/baselines/build_ctle_substrate.py` (`get_fold_indices`),
  `src/configs/paths.py` (enum novo) e `scripts/closing_data/mac_baseline_compare.py`. Escrever
  num engine novo e nunca por cima de `output/check2hgi_ctle/`.

## G3 · HGI de Istambul pareado — talvez neste Mac

- **Porquê.** A célula por lugar de Istambul (29,07) correu sobre `output/hgi_dk_ovl/istanbul`
  (343 795 janelas, 16 348 utilizadores, 2026-06-26), e não sobre as janelas do v18 (271 666 /
  14 530). É ela que sustenta o +6,29 e o "32×". Os cinco estados dos EUA são pareados.
- **O que correr.** Colocar os vetores HGI nas janelas do v18 de Istambul e correr a mesma receita
  do braço por check-in. A viabilidade no M2 Pro (32 GB) está a ser avaliada.
- **Muda no texto.** A célula de Istambul, o +6,29 e o 32× (no resumo, no Cap. 1, no Cap. 5 e no
  Cap. 6), o VEREDITOS V7/V8 e uma linha na B.5.
- **Se nunca correr.** Decisão do autor: o texto fica como está, porque a direção do ganho
  mantém-se e só a magnitude de Istambul fica em causa.

## G4 · STAN em Arizona — se houver GPU

- **Porquê.** O fold 2 falhou o treino (Acc@10 26,84 contra ~55 nos outros; um Markov-1 tira 50,37
  no mesmo fold). A média impressa (49,86) puxa o STAN para baixo, o que nos favorece.
- **O que correr.** Repetir o STAN de AZ nos 5 folds da seed 0.
- **Muda no texto.** A célula de AZ, o "STAN abaixo do piso em quatro" e o "at least 3.55".
- **Se nunca correr.** Decisão do autor: fica como está, sem acrescentar detalhe ao texto.
- **Material.** `docs/results/baselines/faithful_stan_arizona*` e `research/baselines/stan/`.

## G5 · HMT-GRN lido na melhor época — se não houver registos por época

- **Porquê.** As células impressas foram lidas na época 50, sem seleção, e só na seed 0. O autor
  quer os melhores resultados do HMT-GRN, o que não muda a conclusão sobre os nossos modelos.
- **O que correr.** Voltar a correr com seleção de época nos seis conjuntos (seed 0, 5 folds; o
  piso a comparar é o Markov-1 stride-1). **Atualizado a 2026-10-01:** os JSONs foram encontrados
  no SSD externo (`/Volumes/Vitor's SSD/ingred/results/baseline_b3_hmt_grn_style/`), e as seis
  células impressas reproduzem exatamente (IST = `istanbul_stride1/`, 60,42). Mas os ficheiros só
  guardam o valor final de cada fold, sem histórico por época. A melhor época não se lê sem voltar
  a correr. A correr é pouco: AL/AZ cabem em minutos.
- **Muda no texto.** As seis células do HMT-GRN e a frase sobre os modelos lidos na época escolhida
  pela validação.
- **Material.** `scripts/baselines/b3_hmt_grn.py`.
