# Fila de GPU — o que corre quando houver máquina

> Criada a 2026-10-01 a pedido do autor, "para não se perder". O mesmo conteúdo está no dossiê do
> depósito. **O nespedgpu não serve:** reiniciou quatro vezes a 2026-09-30, sempre segundos depois
> de a GPU arrancar com a nossa carga real (05:19, 05:22, 16:46, 21:28). Um teste de matmul constante
> a 300 W passou, e a nossa carga irregular a ~100 W derruba a máquina. Registos em
> `/dados/poimtlnet/integrity_v18/logs/`. Precisa de diagnóstico de hardware feito por quem o gere.

Cada item diz porquê, o que correr, o que muda no texto e o que fica se nunca correr.

> ## Estado a 2026-10-02 de manhã (noite de execuções no M2 Pro)
>
> - **G1 integridade:** Alabama e Arizona FEITOS, 5 folds cada (`5cd62589`, `cb35cd62`). Categoria,
>   só-treino − todos: AL −0,11 ± 0,10, AZ −0,17 ± 0,11 (p 0,024). Região: AL −0,01, AZ +0,13 (n.s.).
>   **Florida:** a parte de CPU está feita (construções, leitura, materialização), mas os braços de
>   treino não cabem no Mac (+6 GB de swap em 16 s ao carregar 1,27M × 576). Opções para o autor:
>   (a) máquina com mais RAM, (b) carregador mais leve, que é uma mudança de código, (c) ficar com
>   AL + AZ e o texto passa a "dois conjuntos de dados". Recomendado: (c).
> - **G2 CTLE:** FEITO (`ded26e96`). Com a divisão certa: AL 16,31 (junho 17,77), AZ 17,67 (junho
>   19,30); todos os folds mais baixos. A ordem a nosso favor fica reforçada. Istambul provavelmente
>   limpo, não provado.
> - **G5 HMT-GRN melhor época:** AL/AZ/IST/FL FEITOS (`073a7e73`, FL 5/5 folds verificados). CA e TX
>   pendentes: ~2,5 h + ~3 h no Mac, sozinhos; precisam de uma janela do autor. A coluna só muda com
>   os seis.
> - **G3:** feito. **G4:** só em GPU remota com compile; se não houver, fica como está.


## G1 · Verificação de integridade no modelo entregue — BLOQUEIA O DEPÓSITO

> **Estado a 2026-10-02: Alabama FEITO, Florida a correr, Arizona preparado.** Os três patches foram
> aplicados no commit `d01cd2a8` (com uma guarda que exige `group_dtype == 'int'`). Alabama, seed 0,
> 5 folds, neste Mac: treinar a representação só com os utilizadores de treino contra a mesma
> construção com todos dá categoria **−0,108 ± 0,099** (p = 0,07) e região **−0,014 ± 0,376**
> (p = 0,94); o braço entregue, re-corrido, reproduz as médias impressas a +0,003 e +0,05. Evidência:
> `docs/results/closing_data/v18/g1_integrity_al/` (commit `5cd62589`). A Florida está a correr no
> SSD externo (`ingred_g1fl`, fold 0 desde 2026-10-02 00:50), e o Arizona tem as entradas conferidas
> por md5 (`gpu_queue_scratch/logs/AZ_G1_INPUT_MD5S.txt`, 2026-10-01). O texto ainda não mudou.

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
  patches, aplicados a 2026-10-01 (`d01cd2a8`; dizia "por aplicar" até 2026-10-02). ⚠ **As 15 divisões em `/dados/poimtlnet/integrity_v18/splits/` NÃO são os
  folds entregues.** O `freeze_split.py` passava os userids como texto, e o `train.py` passa-os como
  int64, o que dá outra partição (0/5 folds iguais, reproduzido). Regenerar com
  `--group-dtype int` (patch no pacote) e confirmar com a verificação #5.

## G2 · CTLE em AL/AZ com a divisão certa — decidido (opção 1 se houver GPU)

> **2026-10-01: Istambul provavelmente está limpo.** O CTLE de Istambul correu no M4 Pro, cujo
> ficheiro canónico de Istambul já tinha as 271 666 linhas stride-1 (idênticas às do `dk_ovl`).
> Isso não está provado, porque não há marcador `CTLE_FOLD.txt` de Istambul. AL e AZ mantêm-se
> (74–82% e 78–82%, medidos contra os folds reais do `train.py`).

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

## G3 · HGI de Istambul pareado — FEITO (2026-10-01)

> **Feito.** Corrido neste Mac (`398b904f`) e aplicado ao Cap. 5 (`915a0199`): a célula por lugar
> passa a **32,54 ± 1,03**, a vantagem a **+2,81 ± 0,51**, os cinco folds a favor, p = 0,00025, e o
> braço por check-in em MPS reproduz a célula CUDA impressa a 0,03 por fold. A faixa fica **+0,23 a
> +2,81** e a razão "cerca de catorze vezes" (era trinta e duas). Nenhum sítio da moldura (Resumo,
> Abstract, Caps. 1, 2 e 6) carregava a magnitude de Istambul. Linha nova na B.5; VEREDITOS V7/V8
> atualizados. Evidência: `docs/results/closing_data/v18/istanbul_pair/`. O resto desta secção é o
> registo de quando estava por correr.

- **Porquê.** A célula por lugar de Istambul (29,07) correu sobre `output/hgi_dk_ovl/istanbul`
  (343 795 janelas, 16 348 utilizadores, 2026-06-26), e não sobre as janelas do v18 (271 666 /
  14 530). É ela que sustenta o +6,29 e o "32×". Os cinco estados dos EUA são pareados.
- **O que correr.** Colocar os vetores HGI nas janelas do v18 de Istambul e correr a mesma receita
  do braço por check-in. **A correr neste Mac (M2 Pro), autorizado pelo autor a 2026-10-01:** os
  dois braços em MPS, com a mesma receita e sem compile. A comparação só conta se o braço por
  check-in reproduzir a célula CUDA impressa (35,35). ~5–6 GB e ~0,5–1,5 h por braço, estimado.
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

> **Estado a 2026-10-02: AL, AZ e IST feitos; FL, CA e TX por correr.** Corrido neste Mac (MPS) com
> `b3_hmt_grn.py --epoch-select per_task`, a opção do commit `82dbe843`, nos folds entregues
> (verificado 5/5). Região na melhor época: **AL 63,19 / AZ 52,11 / IST 68,06** (impresso, época 50:
> 57,05 / 43,70 / 60,42); categoria na melhor época: 20,99 / 22,09 / 24,85. A época 50 de hoje
> reproduz as células impressas, portanto as impressas leram uma época sobretreinada. Registo:
> `/Volumes/Vitor's SSD/gpu_queue_scratch/G5_SUMMARY.md` (ainda fora do repositório). O texto ainda não
> mudou.

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
