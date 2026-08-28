# Pares faixa ↔ corpo da Série B — para o teste "a faixa contradiz alguma linha do corpo?"
# Gerado pela sessao ppt em 27/08 depois do defeito do B6-6.

## B-KARPATHY
FAIXA: An open problem the field cannot design in advance
  corpo: 1$$ {B-KARPATHY} $$ An open problem the field cannot design in advance
  corpo: Karpathy (2019), on designing a multitask network: {``how much feature sharing is there''}, {``tasks fight for the sam
  corpo: Standley et al.\ (ICML 2020): the task-affinity matrix, which asks the same question empirically.
  corpo: PCGrad and GradNorm, which that discussion names, are {already in this dissertation} (Chapter 2).

## B-P1
FAIXA: Matched capacity removes the advantage at California; Texas is unresolved
  corpo: 1$$ {B-P1} $$ Matched capacity removes the advantage at California; Texas is unresolved
  corpo: Give the dedicated region model the joint model's entire parameter budget. Seed 0, five folds.
  corpo: Dataset & ded.\ (narrow) & ded.\ (matched) & joint & joint $-$ matched & {p} & unanimous
  corpo: California & 63.446 & {64.931} & 64.503 & {$-$0.428} & 0.0082 & 5/5

## B-Q13
FAIXA: Stands: representation over architecture. Falls: which part carries the gain
  corpo: 1$$ {B-Q13} $$ Stands: representation over architecture. Falls: which part carries the gain
  corpo: The deposited sentence is wrong in its direction. There is a written errata.} Control redone on the scale of Table 9 -
  corpo: Dataset & gap (place $$ check-in) & concatenation gain & share of the gap
  corpo: Alabama & $+$1.56 & {$+$1.73} ({p} $=$ 0.003) & the whole gap

## B-Q14
FAIXA: The paper lists it; the dissertation does not
  corpo: 1$$ {B-Q14} $$ The paper lists it; the dissertation does not
  corpo: alert}{{It should not have gone out. It returns as an errata, and it now carries the measurement the paper said was mi
  corpo: Submitted paper, p.~9, limit 4 of 5:} the region advantage at Texas and California {``is therefore confounded with cap
  corpo: Main volume, p.~85:} {``Four limits qualify these results''} --- capacity on the region axis is not one of them.

## B1-3
FAIXA: We report the convention that yields fewer improvements
  corpo: 2$$ {B1-3} $$ We report the convention that yields fewer improvements
  corpo: Reported convention:} each dedicated model at its own task's best epoch; the joint model at the epoch its joint valida
  corpo: The alternative} (each task at its own best epoch) is more favorable to the joint model:
  corpo: at most {0.23 macro-F1} and {0.93 Acc@10} at any one seed;

## B1-1
FAIXA: All four intervals lie entirely below zero
  corpo: 2$$ {B1-1} $$ All four intervals lie entirely below zero
  corpo: alert}{{All four are deficits. All four intervals lie entirely below zero.
  corpo: Alabama  & $-$0.87 & $-$1.00 to $-$0.75
  corpo: Arizona  & $-$0.44 & $-$0.62 to $-$0.25

## B1-2
FAIXA: Category has no margin; the intervals carry the bound
  corpo: 2$$ {B1-2} $$ Category has no margin; the intervals carry the bound
  corpo: Superiority} registered on next category, {non-inferiority} on next region. {alert}{{No equivalence margin on the cate
  corpo: Istanbul   & $+$0.08 & $+$0.01 to $+$0.15 & joint model
  corpo: Arizona    & $-$0.00 & $-$0.04 to $+$0.03 & no direction

## B2-3
FAIXA: Every absolute score here is optimistic
  corpo: 2$$ {B2-3} $$ Every absolute score here is optimistic
  corpo: alert}{{Yes --- the second of the four declared limits, in the chapter's own words:}} {``epoch selection consults the 
  corpo: Why the {comparison} is affected far less, stated rather than assumed:
  corpo: same selection rule for both models on the same folds, each selected on its own validation objective;

## B2-1
FAIXA: A per-fold rebuild moves at most 0,33 Acc@10 — three datasets, one seed
  corpo: 2$$ {B2-1} $$ A per-fold rebuild moves at most 0,33 Acc@10 — three datasets, one seed
  corpo: The representation objective {never reads the next-category or next-region targets}.
  corpo: Control: a fresh representation built {per fold, from that fold's training users only}. Three datasets, one seed. Diff
  corpo: Declared limit of that control: a training-only graph has no visit vectors for validation users, so the category compa

## B6-5
FAIXA: Yes on category; on region the floor is the reference
  corpo: 2$$ {B6-5} $$ Yes on category; on region the floor is the reference
  corpo: primaryshade}Next category --- the comparison is clean.} POI-RGNN is
  corpo: native to the task}, re-implemented from its published architecture, and
  corpo: above the tuned Markov-K floor at all six}. The chapter's own sentence:

## B6-6
FAIXA: No published model treats region as an end target
  corpo: 2$$ {B6-6} $$ No published model treats region as an end target
  corpo: 1.0}{tabular}{@{}p{1.8cm} p{1.5cm} p{5.5cm} p{4.8cm}@
  corpo: Sistema} & {Tarefa} & {Como rodou} & {A ressalva
  corpo: re-implemented from its {published architecture and

## B6-7
FAIXA: Nothing was adapted for category; on region, nothing runs unchanged
  corpo: 3$$ {B6-7} $$ Nothing was adapted for category; on region, nothing runs unchanged
  corpo: 1.2}{tabular}{@{}p{2.0cm} p{11.4cm}@
  corpo: Sistema} & {O que foi adaptado
  corpo: 2}{@{}l}{{{primaryshade}NEXT CATEGORY

## B6-7
FAIXA: Two axes, two answers
  corpo: 3$$ {B6-7} $$ Two axes, two answers
  corpo: block}{Em categoria, nada foi adaptado
  corpo: O POI-RGNN roda da arquitetura publicada; no Markov só a ordem $K$ é escolhida por
  corpo: conjunto --- sintonia, não adaptação.

## B-GEO
FAIXA: The vectors separate categories — regions they do not
  corpo: 2$$ {B-GEO} $$ The vectors separate categories — regions they do not
  corpo: primary}{$$}~{Silhouette by category} --- how tight and how
  corpo: well separated the seven labeled groups are, on a scale of $-1$ to $1$;{2pt
  corpo: primary}{$$}~{Nearest-neighbor category purity} ($k = 10$) ---

## U6
FAIXA: No clean ablation separates it; the fixed-pair control bounds it
  corpo: 3$$ {U6} $$ No clean ablation separates it; the fixed-pair control bounds it
  corpo: Limitation 6, p.~90: no controlled ablation separates the change of representation and topology from the change of tas
  corpo: The ablation that would separate them --- {static category classification under the check-in-level representation} ---
  corpo: Chapter 4 is the fixed-pair control}: same architecture, same task pair, only the input moves.

## Q8
FAIXA: The claim is the design, not transfer between the tasks
  corpo: 3$$ {Q8} $$ The claim is the design, not transfer between the tasks
  corpo: Chapter 5, p.~84: {``The evidence here does not separate their contributions''}. The surviving claim is about the {des
  corpo: A one-fold screen} (seed 0, three arms; one number per arm, so it detects only a large effect): at California the regi
  corpo: alert}{{The five-fold trunk ablation at those two datasets does not exist.}} Five-fold ablations ran only at Alabama a

## B4-LEAK
FAIXA: Exact lookup in Chapter 4; one-hop average in Chapter 3
  corpo: 3$$ {B4-LEAK} $$ Exact lookup in Chapter 4; one-hop average in Chapter 3
  corpo: Chapter 4:} the venue-type feature maps {one to one} onto the seven top-level categories, 284 to 365 distinct values p
  corpo: Chapter 3:} the input feature is the {average of its neighbors' categories}, own one-hot excluded by construction. {Th
  corpo: Measured: own category returns at {mean weight 0.10} against {total own-category weight 0.39}; removing it lowers a pr

## B2-4
FAIXA: Forward-only by design; still transductive by construction
  corpo: 3$$ {B2-4} $$ Forward-only by design; still transductive by construction
  corpo: In the delivered text} the direction is a {design decision}, stated as the fourth of the four limits (p.~85) and in th
  corpo: In the repository}, the generation that produced every delivered cell is the one in which that channel is closed. Clos
  corpo: What the closure does not buy:} the representation is still trained {once over the whole graph} --- transductive by co

## B4-3
FAIXA: Best-of-two per row, and not width-matched
  corpo: 3$$ {B4-3} $$ Best-of-two per row, and not width-matched
  corpo: alert}{{The range is best-of-two per row, and saying so is the answer.}} Category gains of {20.2 to 22.0} points per s
  corpo: Read {by isolated variant}, SIREN alone at Texas averages {$+$17.89}, outside the announced range.
  corpo: On the sequential task the same rule gives {15 of 21} category-state combinations to the variants, with one technical 

## B7-3
FAIXA: Four steps, and one deliberate gradient cut
  corpo: 3$$ {B7-3} $$ Four steps, and one deliberate gradient cut
  corpo: 1.}~{Two check-in graph-convolution layers} over the succession edges, residual update. Output: one 64-dimensional vec
  corpo: 2.}~{Pool visits at their place} with four attention heads, one learned query shared across places, keys and values fr
  corpo: 3.}~{Add the spatial place neighborhood:} the pooled place representation combined with a trainable place table initia

## B7-6
FAIXA: Training batches pair rows at random; validation rows are record-aligned
  corpo: 3$$ {B7-6} $$ Training batches pair rows at random; validation rows are record-aligned
  corpo: alert}{{During training: yes.}} The category and region loaders use the {same user-disjoint fold} and {shuffle indepen
  corpo: At validation: no.} {``Validation rows are record-aligned.''
  corpo: The appendix calls it what it is: {``This operational detail is unusual, but it is part of the reported training proto

## B7-4
FAIXA: Private encoders and heads; only activations meet
  corpo: 3$$ {B7-4} $$ Private encoders and heads; only activations meet
  corpo: Private encoders, same shape, different parameters:} each history goes through 64 $$ 256 $$ 256 $$ 256, ReLU and layer
  corpo: Two bidirectional cross-attention blocks.} In each block Next Category queries Next Region first; Next Region then que
  corpo: Jointly optimized, but the directional projections are not tied:} {``The model therefore differs from classical hard p

## B7-5
FAIXA: Two routes, and one prior fixed at zero
  corpo: 3$$ {B7-5} $$ Two routes, and one prior fixed at zero
  corpo: Category head:} a four-layer unidirectional GRU, width 256, reading the category-context sequence; the top-layer state
  corpo: Region head keeps two routes on purpose:
  corpo: private tower}: the raw $964$ region history, spatio-temporal attention, four heads, dropout 0.3;

## Q5
FAIXA: Declared without a number; the only bound is architectural
  corpo: 3$$ {Q5} $$ Declared without a number; the only bound is architectural
  corpo: Declared, without a number}, in Chapter 2, p.~27: the joint model reads two tables exported from the same check-in-lev
  corpo: The only {quantified} boundary is architectural: on the spatial route the pooled place representation is {detached} (A
  corpo: That bounds gradient flow {inside representation training}. It does not quantify the {information overlap} between the

## B1-4
FAIXA: n = 4; the exact Wilcoxon floors at 0,0625
  corpo: 3$$ {B1-4} $$ n = 4; the exact Wilcoxon floors at 0,0625
  corpo: 4 seeds $$ 5 folds = {20 fitted models} per configuration. the test compares {four numbers}: one mean per seed.
  corpo: Primary:} paired {t} on the four per-seed means, with the 90
  corpo: Registered:} paired Wilcoxon signed-rank over the 20 matched fold differences. {Reported alongside, and it agrees.

## B6-4
FAIXA: Computed under our own windows — the chapter declines one explanation
  corpo: 3$$ {B6-4} $$ Computed under our own windows — the chapter declines one explanation
  corpo: HMT-GRN falls below the floor at {all six} datasets, the ReHDM reference at {three}, STAN at {four}.
  corpo: The floor is computed under {our own sliding windows and folds}, advancing one visit at a time, {``so the region of th
  corpo: They do not meet the floor on equal terms: HMT-GRN on the same data, folds and inits; STAN on the same folds, own repr

## B6-1
FAIXA: One sentence, isolated, corrected in the source
  corpo: 3$$ {B6-1} $$ One sentence, isolated, corrected in the source
  corpo: Resumo (delivered)}        & superiority on next category {``em todos os conjuntos''
  corpo: English Abstract, §2.5, Ch.~5, Ch.~6} & superiority {at one dataset
  corpo: The delivered result}      & {Florida only}, $+$0.19, Holm {p} 0.011

## B6-3
FAIXA: The user column is the raw corpus, not the test
  corpo: 3$$ {B6-3} $$ The user column is the raw corpus, not the test
  corpo: check-ins {} users {} POIs & {raw corpus
  corpo: windows & {after} the minimum-length filter (ten check-ins), stride 1
  corpo: Verified by direct count on the raw files: Alabama 113,846 check-ins, {3,858 users}, 11,848 places; Arizona 236,450, {

## B4-2
FAIXA: Category gains; the sequential task loses
  corpo: 3$$ {B4-2} $$ Category gains; the sequential task loses
  corpo: Travel, Florida & MTLnet & ST-MTLNet (SIREN)
  corpo: category} (Table 6)      & 45.49 $$ 1.20 & {64.89 $$ 1.20
  corpo: next category} (Table 7) & {64.47 $$ 1.02} & 45.00 $$ 1.10

## B7-1
FAIXA: Fitted first and frozen --- the forecast labels never reach it
  corpo: 3$$ {B7-1} $$ Fitted first and frozen --- the forecast labels never reach it
  corpo: validate the records, order each user's visits by time, map each place to a polygon;
  corpo: build temporal, place and region graphs linked by the check-in / place / region / city hierarchy;
  corpo: train Check2HGI, export separate {64-dimensional} check-in and region tables;

## U8
FAIXA: Registered before results; Alabama would fail one point
  corpo: 4$$ {U8} $$ Registered before results; Alabama would fail one point
  corpo: The justification is a declared judgment: a service acts on which region will be busy, not on a single rank position, 
  corpo: The empirical support that does exist is the dispersion: the sd of the paired difference across the four user partitio
  corpo: Why it does not overturn the thesis:} the margin was registered {before any result was read} (p.~76), and the four cel

## U1
FAIXA: The five-fold trunk ablation at Texas and California does not exist
  corpo: 4$$ {U1} $$ The five-fold trunk ablation at Texas and California does not exist
  corpo: What exists: the one-fold screen of {Q8}, where every arm moves under {0.15} point.
  corpo: The five-fold ablation at those two datasets {does not exist}. Five-fold arms were run at {Alabama} (dcat $-$0.015 / d
  corpo: Why it does not overturn the thesis:} the thesis does not claim the trunk carries the result. The claim on p.~84 is ab

## U4
FAIXA: Four of six measured; the two largest label spaces are not
  corpo: 4$$ {U4} $$ Four of six measured; the two largest label spaces are not
  corpo: Appendix D of the main volume covers {four of the six} datasets, and says so: {``Texas and California are not measured
  corpo: It also limits itself by architecture: {``Nothing here says the gradients stay orthogonal in a model that shares more 
  corpo: At the four measured datasets, equivalence to zero holds within a {$$0.05} margin, with every mean inside it and {99.6

## U5
FAIXA: Not retained by the evaluation path
  corpo: 4$$ {U5} $$ Not retained by the evaluation path
  corpo: Chapter 5, p.~85: {``Where the shortlist misses, the geographic size of the error is the quantity that would matter to
  corpo: The service framing is explicitly motivation, not result, and is the {third} of the four declared limits: {``we do not
  corpo: Why it does not overturn the thesis:} no claim in the document is about service performance. The shortlist reading on 

## U7
FAIXA: Not evaluated, and transductive by construction
  corpo: 4$$ {U7} $$ Not evaluated, and transductive by construction
  corpo: Chapter 6, p.~88: the result {``also supports testing Check2HGI in other mobility prediction architectures, although i
  corpo: Limitation 3, p.~90: the representation is {transductive}, trained on each dataset's check-in graph, {``so it cannot r
  corpo: Why it does not overturn the thesis:} every comparison that carries the thesis holds the consuming model fixed and var

## B4-4
FAIXA: Every record of the earlier extraction reappears in the current
  corpo: 4$$ {B4-4} $$ Every record of the earlier extraction reappears in the current
  corpo: Chapters 3 and 4    & 20,301 & 65,009 & 990,518
  corpo: Chapter 5           & 21,052 & 76,544 & 1,407,034
  corpo: The mechanism is declared: the category-mapping table was extended about eleven months after the earlier extraction, a

## B4-5
FAIXA: 2,3 times the two single-task models, in wall time
  corpo: 4$$ {B4-5} $$ 2,3 times the two single-task models, in wall time
  corpo: Model & Time (s) & Epochs & MFLOPs
  corpo: Category      & 16.26 & 3.8 & 2.315
  corpo: Next          & 18.71 & 3.2 & 0.012

## B4-DGI
FAIXA: It may be degenerate, and it was never re-measured
  corpo: 4$$ {B4-DGI} $$ It may be degenerate, and it was never re-measured
  corpo: An audit recorded, incidentally and outside the leak question, that the contrastive objective {as implemented appears 
  corpo: If confirmed, {``trained DGI embedding''} may not describe what Chapter 3 actually used.
  corpo: alert}{{What I cannot say is that it was re-measured. It was not.}} The artifacts of that audit are not in the reposit

## B-APXG
FAIXA: The published 100,2\% was counted at the wrong depth
  corpo: 4$$ {B-APXG} $$ The published 100,2
  corpo: Printed (supplement, Appendix G, Table 8):} joint 4,197,621 (AL) and 5,151,189 (CA); original dedicated 644,359; wider
  corpo: Recounted against an independent implementation of the head:} 1,433,863 {} 9,634,471 ({230
  corpo: The conclusion does not fall. It gets stronger.} The wider arm was not capacity-matched: it received {more than double

## B-MTLCHECK
FAIXA: Eight cells, mean delta of one thousandth
  corpo: 4$$ {B-MTLCHECK} $$ Eight cells, mean delta of one thousandth
  corpo: A clean reimplementation, written without reusing code from the old repository, running from raw check-ins to trained 
  corpo: Eight cells at Alabama and Arizona, under {the chapter's own protocol} (five flat folds): {mean delta $-$0.001 pp}, la
  corpo: Two caveats the sentence must carry: {one seed} on the new side against four on the chapter's; and the two columns are

## B-NOM
FAIXA: It names a confound; it does not change the inference
  corpo: 4$$ {B-NOM} $$ It names a confound; it does not change the inference
  corpo: It does not change the inference.} The reported test was always the paired {t} on the {four per-seed means}: $n = 4$, 
  corpo: It changes the name, and it names a limit.} Each seed is one {repetition} of the cross-validation: one integer drives 
  corpo: Measured: between folds $${1.2 pp} {} between repetitions {0.02 to 0.07 pp} {} paired band over repetitions {0.05 to 0

## B1-6
FAIXA: The ten differences, with their intervals
  corpo: 4$$ {B1-6} $$ The ten differences, with their intervals
  corpo: minipage}[b]{0.775}{black!70}{{figures/mobiwac/fig4\_deltas.pdf} --- Fig.~7, Cap.~5 (volume principal). {A figura impr
  corpo: minipage}[b]{0.20} {b0}{{B0}}{minipage

## B2-2
FAIXA: Sample-stratified, one repetition, declared in the chapter
  corpo: 4$$ {B2-2} $$ Sample-stratified, one repetition, declared in the chapter
  corpo: Declared in Chapter 3 itself (p.~46): a stratified splitter {over the samples}, so one user's check-ins may appear in 
  corpo: Both prefaces date their conclusions: p.~36, {``Its conclusions are the conclusions of the time, for the configuration
  corpo: What each chapter carries forward is {internal and directional}: Chapter 3 delivers a {null result}; Chapter 4 compare

## B6-2
FAIXA: Always name the volume: B is Disclosure here, Errata there
  corpo: 4$$ {B6-2} $$ Always name the volume: B is Disclosure here, Errata there
  corpo: main volume} (119 pp) & {supplement} (27 pp)
  corpo: A & Other Scientific Contributions & not present
  corpo: B} & {AI-Use Disclosure} & {Errata to the Reproduced Articles

## B2-5
FAIXA: Two errata, offered rather than defended
  corpo: 4$$ {B2-5} $$ Two errata, offered rather than defended
  corpo: ERR-6.} The clause {``at Florida and California it was not varied, so those two carry the value the smaller searches s
  corpo: ERR-7.} The same sentence grades {Florida} with Texas as {``fewer folds''} in the batch-size search. In the dedicated 
  corpo: Neither changes a number or a verdict. Both {reduce} what the sentence claims.

