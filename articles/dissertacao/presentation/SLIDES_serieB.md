> **Onde esta parte entra no `SLIDES.md`.** Depois do slide "Obrigado" da trilha principal, entre
> `\miniframesoff` e `\miniframeson` (plano §6 e §11.1). Os slides desta série **não entram no
> orçamento de 48 min** e **não registram ponto na barra de navegação**. O `\insertframenumber`
> congela sob `\miniframesoff`, então **o rótulo `B-n` vai no conteúdo de cada slide**, nunca no
> rodapé. Numeração interna aqui: `SB1 … SB46`, só para referência de redação. No cabeçalho de cada bloco
> o formato é `SB<n> · <código B> · <a pergunta>`; **na tela o título é só a pergunta**, e o código
> `B-n` fica no canto do conteúdo, como a primeira linha de **Na tela**.
>
> **Contrato herdado do plano §6, aplicado a todos os 46:** uma pergunta = um slide · o título é a
> pergunta, em português, como a banca a faria · rodapé de proveniência · números copiados de célula
> impressa ou do `ladder_recompute.json`, nunca re-derivados · onde a resposta honesta é "não foi
> medido", o limite é a manchete · carimbo de convenção métrica em toda arte reproduzida dos Caps. 3/4.

---

### SB1 · B0 · Se a pergunta for uma destas, o slide já existe
- **Seção/subseção:** Série B · índice (B0)   **Tempo:** sob demanda · ~15 s
- **LEDGER:** INTRODUZ o índice clicável da série de reserva | RETOMA nada
- **Na tela:**
  `B0 · Index`
  Seven families, one question per slide. Every entry is a `\hyperlink`.
  - **B1 · Verdict and statistics**: the four region cells and their direction · the five category cells and their sign · checkpoint convention · what `n` is · why a two-point margin
  - **B2 · Protocol and leakage**: training the representation on all places · the two studies' split · choosing the epoch on the fold that is reported · the forward-only edge · search coverage
  - **B3 · Post-submission**: capacity control on region · the concatenation control · the limit the paper carries · the supplement errata line · naming · capacity in the literature · the reimplementation · Appendix G counts
  - **B4 · Chapters 3 and 4**: label in the input, by mechanism · the contrastive objective · width · Travel by task · best-of-two · the Florida corpus · convergence cost
  - **B5 · Not measured**: U1 to U8, the limit as the headline
  - **B6 · Document and scope**: the Portuguese Resumo · appendix letters · the user column · the region Markov floor
  - **B7 · How Check2HGI and the joint model work**: Appendix E of the main volume
- **Fala (PT):** "Posso ir direto ao slide da pergunta." E clico. Nada mais.
- **Proveniência:** rodapé: `série de reserva · fora da contagem principal`.
- **Nunca dizer:** nada aqui é falado como conteúdo; o B0 é navegação.

---

### SB2 · B1-1 · Um teste de não-inferioridade não é um empate. Nas quatro células de região dentro da margem, qual é a direção?
- **Seção/subseção:** Série B · família B1 (Q4)   **Tempo:** sob demanda · ~45 s
- **LEDGER:** RETOMA o ladder de veredito (5.5) | RETOMA a margem de dois pontos (5.4)
- **Na tela:**
  `B1-1`
  **All four are deficits. All four intervals lie entirely below zero.**
  | Dataset | Δ Acc@10 | 90% CI |
  |---|---:|---|
  | Alabama | −0.87 | −1.00 to −0.75 |
  | Arizona | −0.44 | −0.62 to −0.25 |
  | Florida | −0.16 | −0.19 to −0.13 |
  | Istanbul | −0.08 | −0.16 to −0.002 |
  - A reverse-direction test, post hoc over the same six comparisons and corrected across them, resolves three of the four. Istanbul is not resolved.
  - Each of the four clears the registered two-point margin. **None of them is a tie.**
- **Fala (PT):** "O senhor tem razão, e o texto diz isso na mesma página. As quatro são déficits, e os quatro intervalos ficam inteiramente abaixo de zero: Alabama menos zero vírgula oitenta e sete, Arizona menos zero vírgula quarenta e quatro, Flórida menos zero vírgula dezesseis, Istambul menos zero vírgula zero oito. Um teste na direção contrária, aplicado depois às mesmas seis comparações e corrigido entre elas, resolve três das quatro; o de Istambul não, o intervalo dele chega a dois milésimos de zero. As quatro vencem a margem com folga, que é o que a análise registrada pediu delas. Nenhuma é um empate, e eu não as chamo assim em lugar nenhum."
- **Proveniência:** Cap. 5, p. 82 (volume principal, `src/banca.pdf`); precisão cheia em `wrapup/evidence/ladder_recompute.json`, bloco `reg`.
- **Nunca dizer:** "empata", "matches", "ties", "semelhante", "a par". Nunca aplicar a margem de dois pontos ao eixo de categoria.

---

### SB3 · B1-2 · A margem de equivalência foi registrada só para região. O que o senhor usa em categoria?
- **Seção/subseção:** Série B · família B1 (Q2)   **Tempo:** sob demanda · ~50 s
- **LEDGER:** RETOMA o ladder de veredito (5.5) | RETOMA o plano de análise registrado (5.4)
- **Na tela:**
  `B1-2`
  **The plan assigned a superiority test to next category and a non-inferiority test to next region.** No equivalence margin was registered on the category axis.
  | Dataset | Δ macro-F1 | 90% CI | favors |
  |---|---:|---|---|
  | Istanbul | +0.08 | +0.01 to +0.15 | joint model |
  | Arizona | −0.00 | −0.04 to +0.03 | no direction |
  | California | −0.00 | −0.03 to +0.02 | no direction |
  | Texas | −0.13 | −0.19 to −0.08 | dedicated |
  | Alabama | −0.19 | −0.33 to −0.04 | dedicated |
  - Florida (+0.19) is the one cell that survives Holm.
  - The other five are **unresolved**, and the bound is read off the intervals: the widest reaches 0.34 from zero, at Alabama. **Equivalent to zero within half a point.**
- **Fala (PT):** "O plano registrou superioridade em categoria e não-inferioridade em região, e não registrou margem no eixo de categoria. Então uma diferença que falha o teste de superioridade é relatada como não resolvida, nunca como empate. O que eu posso dizer sobre a magnitude vem dos próprios intervalos: o mais largo chega a zero vírgula trinta e quatro de zero, em Alabama, o que limita as seis diferenças a meio ponto de zero de uma vez. E a direção viaja junto, porque as cinco não apontam para o mesmo lado: Istambul favorece o modelo conjunto, Texas e Alabama favorecem o dedicado, e Arizona e Califórnia ficam sobre o zero."
- **Proveniência:** Cap. 5, §5.5.3 (p. 76) e p. 82; legenda da Tabela 10 (p. 81), `src/tables/mobiwac/results.tex:26-30`; intervalos em `wrapup/evidence/ladder_recompute.json`, bloco `cat`.
- **Nunca dizer:** "equipara-se", "empata". Nunca citar meio ponto no eixo de região (lá o limite derivado é 1,372 pp).

---

### SB4 · B1-3 · O senhor escolheu a convenção de checkpoint mais restrita, que remove seis melhorias que poderia estar reclamando. Quem decide isso depois de ver os resultados?
- **Seção/subseção:** Série B · família B1 (Q1)   **Tempo:** sob demanda · ~45 s
- **LEDGER:** RETOMA a convenção joint-best (5.4)
- **Na tela:**
  `B1-3`
  **Reported convention:** each dedicated model is read at its own task's best epoch; the joint model is read at the epoch its joint validation score selects, both tasks from the one saved model.
  **The alternative** (each task at its own best epoch) is more favorable to the joint model:
  - at most **0.23 macro-F1** and **0.93 Acc@10** at any one seed;
  - averaged over the four seeds, **+0.03 to +0.17** category and **+0.19 to +0.90** region;
  - enough to turn **four further category cells and two further region cells** into improvements that survive the same Holm correction.
  **That is exactly why it is not the one reported.** One checkpoint per fold is what a deployed system can serve.
- **Fala (PT):** "A convenção que eu reporto é a única que um sistema em produção consegue servir, porque se compromete com um checkpoint por dobra. A outra leitura favoreceria o modelo conjunto em até zero vírgula vinte e três de macro-F1 e zero vírgula noventa e três de Acc@10 na pior semente, e viraria mais seis células a meu favor sob a mesma correção. É por isso que eu não a uso. E ela está declarada no parágrafo imediatamente acima da tabela, não num apêndice."
- **Proveniência:** Cap. 5, p. 80-81; definição da convenção em `src/chapters/5_mobiwac/06_results.tex:141-142`; médias por semente em `articles/dissertacao/src_fix/REVISION_PLAN.md:93-94` (registro de revisão).
- **Nunca dizer:** apresentar os números da convenção alternativa como resultado.

---

### SB5 · B1-4 · O que é o n do seu teste? E por que o senhor trocou o Wilcoxon registrado pelo t pareado?
- **Seção/subseção:** Série B · família B1 (plano §7, item 4)   **Tempo:** sob demanda · ~50 s
- **LEDGER:** RETOMA n = 4, sementes, TOST, Holm (5.4)
- **Na tela:**
  `B1-4`
  - 4 seeds × 5 folds = **20 fitted models** per configuration. **Inferential unit: n = 4**, the per-seed means.
  - **Primary:** paired *t* on the four per-seed means, with the 90% CI, because folds inside a seed are not independent.
  - **Registered:** paired Wilcoxon signed-rank over the 20 matched fold differences. **Reported alongside, and it agrees.**
  - At the seed-level footing the exact one-sided Wilcoxon cannot fall below **0.0625**, whatever the effect size. That is why the *t* carries the verdict.
  - The departure is stated in the chapter and both tests ship in the code release.
- **Fala (PT):** "Vinte modelos ajustados, quatro médias por semente, e a unidade inferencial é quatro. O plano tinha registrado o Wilcoxon pareado sobre as diferenças por dobra, e ele continua reportado ao lado, e concorda. O primário é o t pareado sobre as quatro médias porque as cinco dobras dentro de uma semente compartilham a maior parte do treino e não são cinco observações independentes. E há uma razão aritmética: no nível de semente o Wilcoxon exato de uma cauda não desce abaixo de zero vírgula zero seiscentos e vinte e cinco, por maior que seja o efeito. Não é confissão de desvio, é desvio declarado, com os dois apoios e o mesmo veredito."
- **Proveniência:** Cap. 5, §5.5.3, p. 76 (`src/chapters/5_mobiwac/05_setup.tex:115,:117`); `GLOSSARY.md` §4, linhas `n = 20` / `n = 4`.
- **Nunca dizer:** "n = 20 repetições pareadas" (proibido pelo `GLOSSARY`).

---

### SB6 · B1-5 · Por que dois pontos? A margem não foi escolhida para caber no resultado?
- **Seção/subseção:** Série B · família B1 (Q2 / justificativa da margem)   **Tempo:** sob demanda · ~45 s
- **LEDGER:** RETOMA a margem de dois pontos registrada (5.4)
- **Na tela:**
  `B1-5`
  - Fixed in the written analysis plan, **before any result was read**.
  - The reason is the use: a mobility-aware service acts on **which region will be busy**, not on a single rank position. A two-point change in Acc@10 is below the level at which that service would behave differently.
  - What the data say about the margin, stated in the chapter: the sd of the paired difference across the four user partitions runs **0.02 to 0.16**; the intervals at **Istanbul, Arizona and Florida** are narrow enough to support a margin as small as **one point**; **Alabama's is not**, and Alabama has the largest region difference.
  - **The chapter names its own weakest case.**
- **Fala (PT):** "Ela é uma escolha, e está declarada como escolha, fixada antes de qualquer resultado ser lido. A justificativa é de uso: o serviço age sobre qual região vai ficar movimentada, não sobre uma posição de ranking, e dois pontos de Acc@10 ficam abaixo do nível em que ele se comportaria diferente. E o texto vai além: diz que Istambul, Arizona e Flórida têm intervalos estreitos o bastante para sustentar uma margem de um ponto, e que o de Alabama não, que é justamente o conjunto com a maior diferença de região. Eu nomeio o meu pior caso na mesma página."
- **Proveniência:** Cap. 5, §5.5.3, p. 76-77 (`src/chapters/5_mobiwac/05_setup.tex:119`).
- **Nunca dizer:** que a margem foi derivada dos dados (ela é registrada; o limite **derivado** é outro objeto, e vale 1,372 pp simultâneo em região).

---

### SB7 · B2-1 · A representação foi treinada uma vez sobre o conjunto inteiro, incluindo os usuários de validação. Isso não vaza?
- **Seção/subseção:** Série B · família B2 (Q12)   **Tempo:** sob demanda · ~50 s
- **LEDGER:** INTRODUZ a cobertura de 67 a 87 por cento do controle de reconstrução por dobra | RETOMA o split disjunto por usuário (5.4) e os quatro limites declarados (5.6)
- **Na tela:**
  `B2-1`
  - The representation objective **never reads the next-category or next-region targets**.
  - Control: a fresh representation built **per fold, from that fold's training users only**. Three datasets, one seed.
  - Differences: **−0.33 to +0.01 Acc@10** (region), **0.00 to +0.29 macro-F1** (category).
  - The declared limit of that control: for category, a graph built from training users only has no visit vectors for validation users, so the comparison used one vector per place and kept only windows whose input places occurred in training. **Those windows cover 67 to 87 percent of the validation data**, and the comparison does not cover per-visit information or places unseen in training.
  - Forecast evaluation is **user-disjoint**; representation learning is **transductive** with respect to the graph, and the chapter says so.
- **Fala (PT):** "O objetivo que treina a representação nunca vê os rótulos das duas tarefas. E eu rodei o controle: reconstruir a representação por dobra, só com os usuários de treino daquela dobra, move o resultado no máximo zero vírgula trinta e três de Acc@10 e zero vírgula vinte e nove de macro-F1, em três conjuntos numa semente. O que esse controle não cobre está escrito na mesma seção: no lado da categoria ele roda sobre sessenta e sete a oitenta e sete por cento das janelas de validação, porque um grafo só de treino não tem vetor de visita para usuário de validação. A avaliação é disjunta por usuário; o aprendizado da representação é transdutivo, e o texto declara os dois."
- **Proveniência:** Cap. 5, §5.5.2, p. 75-76 (`src/chapters/5_mobiwac/05_setup.tex:65-77`); primeiro dos quatro limites, p. 85.
- **Nunca dizer:** que a representação é causal, ou que o controle cobre visitas e lugares não vistos.

---

### SB8 · B2-2 · Os dois primeiros estudos usam divisão estratificada por amostra, não disjunta por usuário. Os resultados deles ainda valem?
- **Seção/subseção:** Série B · família B2 (Q19)   **Tempo:** sob demanda · ~45 s
- **LEDGER:** RETOMA o protocolo dos dois primeiros estudos (2.5)
- **Na tela:**
  `B2-2`
  - Declared in Chapter 3 itself (p. 46): a stratified splitter **over the samples**, so one user's check-ins may appear in both training and validation. For the category task the sample unit is the place, so **no place spans two folds**. One pinned seed, so the five folds are **one repetition**.
  - Both prefaces date their conclusions: p. 36, *"Its conclusions are the conclusions of the time, for the configuration studied here"*; p. 52, *"The conclusions reported here are those of the time, for that configuration"*.
  - What each chapter carries forward is **internal and directional**: Chapter 3 delivers a **null result**; Chapter 4 compares against **one baseline under the same protocol on both arms**.
  - No conclusion of this dissertation rests on an absolute number from either.
  `Next-POI Prediction = next category (Def. 2.7)` · Caps. 3/4 report **per-category F1**, not macro-F1
- **Fala (PT):** "Eles valem como o que são: comparações internas sob um protocolo mais fraco, declarado na página quarenta e seis e datado nos dois prefácios. O Capítulo 3 entrega um nulo, e um nulo não fica melhor com um protocolo pior. O Capítulo 4 compara contra uma única baseline, com o mesmo protocolo nos dois braços, então o que muda entre eles é a entrada. Nenhuma conclusão do documento depende de um número absoluto desses dois capítulos."
- **Proveniência:** Cap. 3, p. 46-47; prefácios p. 36 e p. 52; delimitação de moldura no Cap. 2, p. 35.
- **Nunca dizer:** chamar as médias por categoria dos Caps. 3/4 de macro-F1; comparar um número do Cap. 3 com um do Cap. 5.

---

### SB9 · B2-3 · O senhor escolheu a época no mesmo conjunto em que reporta o número?
- **Seção/subseção:** Série B · família B2   **Tempo:** sob demanda · ~50 s
- **LEDGER:** RETOMA os quatro limites declarados (5.6)
- **Na tela:**
  `B2-3`
  **Yes, and it is the second of the four declared limits, in the chapter's own words:** *"epoch selection consults the fold that the score is then read on, so every absolute score reported here is optimistic"*.
  Why the **comparison** is affected far less, stated rather than assumed:
  - the selection rule is **the same for both models on the same folds**, each model selected on its own validation objective;
  - for the category comparison the **dedicated** model receives the **wider search**: batch size at all six datasets and learning rate at four, against a joint-model search covering four of six, with Texas and California carrying a transferred configuration;
  - both sides of the **region** comparison run one fixed configuration, so that mitigation does not apply there.
  - The chapter closes it: *"It does not follow that the bias cancels exactly"*.
- **Fala (PT):** "Sim, e está declarado como segundo dos quatro limites: todo escore absoluto que eu reporto é otimista. O que eu posso defender é a comparação, e por duas razões que o texto declara em vez de supor. A regra de seleção é a mesma nos dois modelos, nas mesmas dobras, cada um selecionado no próprio objetivo de validação. E no eixo de categoria é o braço dedicado que recebe a busca mais ampla, o que torna a diferença que eu reporto conservadora, exceto em Flórida e Califórnia, onde as duas buscas ficam próximas. No eixo de região os dois lados rodam configuração fixa, então essa mitigação não vale ali. E eu fecho dizendo que não se segue que o viés se cancele exatamente."
- **Proveniência:** Cap. 5, p. 85, limite 2; cobertura por botão em §5.5.2, p. 75-76.
- **Nunca dizer:** que o viés se cancela.

---

### SB10 · B2-4 · As arestas entre visitas consecutivas só correm para frente. Isso foi sempre assim, ou foi corrigido?
- **Seção/subseção:** Série B · família B2   **Tempo:** sob demanda · ~55 s
- **LEDGER:** INTRODUZ a correção do canal entre visitas consecutivas no repositório e o que ela valeu | RETOMA a aresta só para frente como princípio de projeto (5.2)
- **Na tela:**
  `B2-4` · **provenance first**
  - **In the delivered text** the direction is a **design decision**, stated as the fourth of the four limits (p. 85) and in the method (p. 26): each visit node draws on the visits that precede it, in training and at readout, *"which is what keeps a node from carrying a feature of the target it is used to predict"*.
  - **In the repository**, the generation that produced every delivered cell is the one in which that channel is closed. Closing it was worth **28.63 macro-F1 at Alabama** on the category axis; the region axis moved by under two points.
  - **What the closure does not buy:** the representation is still trained **once over the whole graph**, so it is transductive by construction. The forecast splits stay user-disjoint.
  - Nothing in either volume reports a before-and-after of this channel, and this slide does not present one as a result.
- **Fala (PT):** "No documento a direcionalidade é decisão de projeto, e está nas duas páginas que a explicam: o alvo é predito do passado do usuário, então a representação é construída só do passado, no treino e na leitura. No repositório, a geração que produziu todas as células entregues é aquela em que esse canal está fechado, e fechá-lo valeu vinte e oito vírgula sessenta e três de macro-F1 em Alabama. Isso não torna a representação causal: ela continua treinada uma vez sobre o grafo inteiro, e eu digo isso. Se o senhor quiser o rastro completo, ele está no repositório, e nenhum dos dois volumes o narra."
- **Proveniência:** Cap. 5, p. 26 e p. 85 (limite 4). O valor de 28,63 macro-F1: `articles/dissertacao/CLAUDE.md` §0.2 e `wrapup/NEW_VERSION.md` §8. **pós-submissão / repositório: não consta em nenhum dos dois volumes.**
- **Nunca dizer:** levantar isto se ninguém perguntar; apresentá-lo como descoberta ou como conserto da história do arco. O plano §2 tira o vazamento da narrativa principal.

---

### SB11 · B2-5 · A frase de cobertura de busca do Cap. 5 está correta?
- **Seção/subseção:** Série B · família B2 (ERR-6, ERR-7)   **Tempo:** sob demanda · ~45 s
- **LEDGER:** INTRODUZ as duas erratas de cobertura de busca (ERR-6, ERR-7)
- **Na tela:**
  `B2-5` · **two errata, offered rather than defended**
  - **ERR-6.** The clause *"at Florida and California it was not varied, so those two carry the value the smaller searches selected"* credits the small-dataset searches with a value they did not select. The small searches selected **0.0025 and 0.0005**; Florida and California carry **0.005**, the large-dataset tier. The coverage itself is right; the **provenance clause** is not.
  - **ERR-7.** The same sentence grades **Florida** with Texas as *"fewer folds"* in the batch-size search. In the dedicated category family the single-fold annotation is **Texas only**; Florida's single-fold screen belongs to the **joint-model** search, described one sentence later.
  - Neither changes a number or a verdict. Both **reduce** what the sentence claims.
- **Fala (PT):** "Duas frases de proveniência, não de resultado, e eu as trago em vez de esperar. A cobertura por botão está certa: taxa de aprendizado buscada em cinco dobras nos três conjuntos menores, em uma dobra no Texas, e não variada em Flórida e Califórnia. O que a frase atribui mal é a origem do valor que esses dois carregam. E a graduação de dobra única em Flórida pertence à busca do modelo conjunto, não à do dedicado; as duas buscas estão em sentenças vizinhas e a anotação migrou de uma para a outra. As duas correções enfraquecem a frase, e é por isso que eu as declaro."
- **Proveniência:** `wrapup/open_points/LACUNAS.md` §ERR-6, §ERR-7; texto vivo `src/chapters/5_mobiwac/05_setup.tex:52-55`. Correções destinadas ao depósito final.
- **Nunca dizer:** que a cobertura declarada está errada; o que está errado é a atribuição de origem.

---

### SB12 · B3-1 · B-P1 · A vantagem de região sobrevive a um controle de capacidade pareada?
- **Seção/subseção:** Série B · família B3 (Q14, U2)   **Tempo:** sob demanda · ~70 s   **OFERECER PROATIVAMENTE**
- **LEDGER:** INTRODUZ o controle de capacidade pareada no eixo de região
- **Na tela:**
  `B-P1` · **OFFER THIS BEFORE IT IS ASKED**
  **The control: give the dedicated region model the joint model's entire parameter budget, change nothing else. Seed 0, five folds.**
  | Dataset | dedicated (narrow) | dedicated (matched) | joint | joint − matched | *p* | unanimous |
  |---|---:|---:|---:|---:|---:|:--:|
  | California | 63.446 | **64.931** | 64.503 | **−0.428** | 0.0082 | 5/5 |
  | Texas | 64.951 | **66.330** | 66.117 | −0.214 | 0.1162 | 4/5 |
  - The width curve **saturates early**: an arm at **57 percent** of that budget already reaches the same level (352 → 528 is +0.021, *p* = 0.40).
  - **The reported region advantage measures capacity, not exchange between the tasks.**
  - What survives: one model produces both predictions in one forward pass, at no measured cost on either task.
  - **Category is untouched:** there a dedicated model with 6.5 times the parameters scores **lower** by 0.53 macro-F1 (*p* = 0.0011).
- **Fala (PT):** "Esta é a pergunta que eu quero fazer no lugar do senhor. O artigo submetido lista o confundimento de capacidade como um dos seus cinco limites e diz que o controle pareado não havia sido rodado. Eu o rodei. Dando ao dedicado de região o orçamento inteiro do modelo conjunto, ele fica acima do modelo conjunto na Califórnia por quatro décimos de Acc@10, com direção unânime nas cinco dobras; no Texas os dois ficam a zero vírgula vinte e um de distância, e o mesmo teste não separa essa diferença de zero. E um braço com cinquenta e sete por cento daquele orçamento já chega no mesmo nível. Ou seja: a vantagem de região que eu reporto mede capacidade, não troca entre as tarefas. O que sobrevive é o resultado operacional, um modelo produz as duas predições numa passada sem custo mensurável em nenhuma das duas. E o eixo de categoria, onde a tese vive, não é tocado: lá multiplicar por seis e meio os parâmetros do dedicado baixa o macro-F1 dele."
- **Proveniência:** `wrapup/post_submission_studies/P1_capacity_region.md` (medido 2026-08-13); verificação célula a célula em `wrapup/erratas/VERIFICACAO.md`; errata redigida em `wrapup/erratas/errata_Q14_capacity_region.tex`. **pós-submissão: não consta em nenhum dos dois volumes.**
- **Nunca dizer:** `+2,12 / +2,05` (números de uma representação superada). As margens entregues são **TX +1,21** e **CA +1,06**. Nunca dizer que o Texas favorece o dedicado: lá a diferença não se separa de zero.

---

### SB13 · B3-2 · B-Q13 · O senhor afirma que o ganho vem da estrutura hierárquica e não das features. Como sabe?
- **Seção/subseção:** Série B · família B3 (Q13)   **Tempo:** sob demanda · ~70 s   **OFERECER PROATIVAMENTE**
- **LEDGER:** INTRODUZ o controle de concatenação refeito na escala da Tabela 9
- **Na tela:**
  `B-Q13` · **OFFER THIS BEFORE IT IS ASKED**
  **The deposited sentence is wrong in its direction, and there is a written errata.**
  Control redone on the scale of Table 9. Three arms per dataset, one seed, five folds, one training configuration, only the input changes.
  | Dataset | gap (place → check-in) | concatenation gain | share of the gap |
  |---|---:|---:|---:|
  | Alabama | +1.56 | **+1.73** (*p* = 0.003) | the whole gap |
  | Arizona | +2.50 | **+1.70** (*p* < 0.001) | 68 percent |
  | Florida | +0.21 | **+1.02** (*p* < 0.001) | past it |
  - Arizona is the one dataset where the check-in-level representation still leads the concatenation arm, by **0.80** (*p* = 0.03). At Alabama the two are within **0.18** (*p* = 0.53). At Florida the concatenation arm leads by **0.81** (*p* = 0.001).
  - **What Table 9 establishes stands:** the input representation dominates the architecture. **What falls** is the finer claim about which part of the representation carries the gain.
  - Fidelity: Alabama reproduces fold by fold and epoch by epoch. Arizona and Florida reproduce the mean within 0.07 and 0.03 but **not** fold by fold, so their scale is **open**.
- **Fala (PT):** "Não sabemos, e a afirmação no texto está errada. A fração de um décimo que ela cita vem de outro estudo, em outra variante do grafo e outro código, e não está na escala da Tabela 9. Refiz o controle na escala da própria tabela, em três conjuntos. Concatenar as features por visita ao place embedding fecha o intervalo inteiro em Alabama, sessenta e oito por cento em Arizona, e em Flórida vai muito além dele. São as features que carregam a maior parte do ganho de categoria. Há errata escrita. O que a tabela estabelece continua de pé: a representação de entrada domina a arquitetura. O que cai é a afirmação mais fina sobre qual parte da representação carrega o ganho. E digo o limite: Alabama reproduz exatamente, fold a fold; Arizona e Flórida reproduzem a média mas não os folds, e essa fidelidade fica em aberto."
- **Proveniência:** `wrapup/post_submission_studies/Q13_concatenation_control.md` (medido 2026-08-16); errata em `wrapup/erratas/errata_Q13_concatenation_scope.tex`; frase substituída no Cap. 5, p. 79. **pós-submissão: não consta em nenhum dos dois volumes.**
- **Nunca dizer:** *"sob um décimo do salto place→check-in"* (aritmeticamente falso contra a Tabela 9 na mesma página). Nunca a frase retratada sobre representação hierárquica × injeção de features.

---

### SB14 · B3-3 · B-Q14 · O artigo submetido lista o confundimento de capacidade entre seus limites. A lista da dissertação não o carrega. Por que ele saiu?
- **Seção/subseção:** Série B · família B3 (Q14)   **Tempo:** sob demanda · ~50 s   **OFERECER PROATIVAMENTE**
- **LEDGER:** INTRODUZ a divergência de limites entre o artigo submetido e o volume principal
- **Na tela:**
  `B-Q14` · **OFFER THIS BEFORE IT IS ASKED**
  - **Submitted paper, p. 9, fourth of five limits:** the joint model carries more parameters than the two dedicated models combined, its region pathway several times the dedicated one, so *"the region advantage at Texas and California is therefore confounded with capacity"*, and the capacity-matched control *"has not been run"*.
  - **Main volume, p. 85:** *"Four limits qualify these results"*, and capacity on the region axis is not one of them.
  - What the main volume **does** carry: the parameter cost on p. 73 (about **4.2 million** at Alabama against **1.1 million** for the two dedicated models combined) and an attribution on p. 84 that credits neither the shared trunk nor transfer.
  - The only capacity control in either volume is **Appendix G of the supplement**, which covers **next category only**.
  - **It should not have gone out. It returns as an errata, and it now carries the measurement the paper said was missing** (→ `B-P1`).
- **Fala (PT):** "Ele não deveria ter saído. O artigo lista cinco limites e o quarto é exatamente esse; a dissertação lista quatro e ele não está entre eles. O custo de parâmetros está reportado na página setenta e três, e a atribuição da página oitenta e quatro já não credita o tronco compartilhado nem transferência, mas o limite nomeado saiu. Entra como errata, e agora vem com a medição que o artigo declarava faltar. É a página seguinte do meu material de reserva."
- **Proveniência:** artigo submetido `articles/[mobiwac]/src_fix/sections/07_discussion.tex:105-109`, p. 9; volume principal p. 85 e p. 73; Apêndice G do **suplemento**, p. 24-26. Errata: `wrapup/erratas/errata_Q14_capacity_region.tex`.
- **Nunca dizer:** "Apêndice B" ou "Apêndice G" sem nomear o volume (§8 regra 14).

---

### SB15 · B3-4 · B-Q15 · A errata do suplemento descreve um quarto fundamento de integridade com sonda linear em Florida. Onde ele está no volume de defesa?
- **Seção/subseção:** Série B · família B3 (Q15)   **Tempo:** sob demanda · ~45 s
- **LEDGER:** INTRODUZ a divergência entre a linha de errata do suplemento e o texto que ela descreve
- **Na tela:**
  `B-Q15`
  - **Supplement, p. 18, Table 4** describes a correction that adds *"a fourth ground"*, reporting a development audit of the forward-edge channel, with its three limits stated: **a linear probe**, **Florida at one random initialization**, and **earlier builds of the representation**.
  - **Main volume:** the integrity paragraph at pp. 75-76 does not enumerate grounds. Measured across the three full PDF extractions and the comment-stripped source: *"on three grounds"*, *"fourth ground"*, *"forward-edge"* have **zero** occurrences in the main volume; *"linear probe"* has one, inside the errata table itself.
  - **This is not an errata to write. It is an errata to correct, and the correction is to remove the line.** The audit it describes measured an earlier build of the representation, not the one the results use.
  - What the main volume does carry: the forward-edge channel as the **fourth declared limit**, p. 85, in prose and without a probe.
- **Fala (PT):** "Aquela linha da tabela de errata descreve uma auditoria de desenvolvimento feita sobre uma construção anterior da representação, e a própria linha declara os três limites dela. O texto que eu depositei não carrega esse fundamento, e não deveria: ele mediria uma preparação que não é a que os resultados usam. A linha de errata é que está sobredeclarando, e ela sai. O que está no volume de defesa é o canal de aresta como quarto limite, na página oitenta e cinco, em prosa e sem sonda."
- **Proveniência:** suplemento, p. 18, Tabela 4 (`wrapup/material_extra/`, `tables/mobiwac/errata_scope.tex:31-38`); volume principal, pp. 75-76 e p. 85. Ausências medidas em `wrapup/erratas/VERIFICACAO.md`.
- **Nunca dizer:** "Apêndice B" sem o volume. Aqui é **Apêndice B do suplemento** (Errata), não o do volume principal (Declaração de Uso de IA).

---

### SB16 · B3-5 · B-NOM · Vocês chamaram de semente o que a literatura chama de repetição. Isso muda os resultados?
- **Seção/subseção:** Série B · família B3   **Tempo:** sob demanda · ~55 s
- **LEDGER:** INTRODUZ a distinção repetição × réplica e o confundimento de um botão só
- **Na tela:**
  `B-NOM`
  - **It does not change the inference.** The reported test was always the paired *t* on the **four per-seed means**: n = 4, three degrees of freedom. The registry already separates `n = 20` (fitted models) from `n = 4` (inferential unit) and forbids writing "n = 20 paired repetitions".
  - **It changes the name, and it names a limit.** Each seed is one **repetition** of the cross-validation: one integer drives **two** things, the user partition and the initialization. So variance "between seeds" is partition plus initialization, and **no result produced so far separates them**.
  - Measured magnitudes: between folds **≈1.2 pp** · between repetitions **0.02 to 0.07 pp** · paired band over repetitions **0.05 to 0.15 pp**.
  - The fold term is **20 to 50 times** the repetition term **and is common to both arms**, which is why pairing detects what an unpaired analysis does not.
  - The later work did not correct an error. **It named a confound nobody had named.**
- **Fala (PT):** "Não muda a inferência: o teste reportado sempre foi o t pareado sobre as quatro médias por semente, e o registro de termos já distingue vinte modelos ajustados de unidade inferencial quatro. Muda o nome, e o nome nomeia um limite. Cada semente é uma repetição da validação cruzada, e um único inteiro governa duas coisas, a partição e a inicialização. Então variância entre sementes é partição mais inicialização, e nenhum resultado que eu já produzi separa as duas. As magnitudes ajudam a ver por que o pareamento importa: entre dobras é cerca de um vírgula dois ponto, entre repetições é dois a sete centésimos, e o termo de dobra é comum aos dois braços, então o pareamento o remove."
- **Proveniência:** `mtlcheck/docs/NOMENCLATURE.md` §§1-5 (repositório da reescrita); `GLOSSARY.md` §4, linhas `seed`, `n = 20` / `n = 4`; Cap. 5, §5.5.3, p. 76. **pós-submissão: não consta em nenhum dos dois volumes.**
- **Nunca dizer:** "n = 20 repetições pareadas"; "réplicas independentes" para as quatro sementes.

---

### SB17 · B3-6 · B-KARPATHY · Como se decide quanta capacidade cada tarefa recebe num modelo multitarefa?
- **Seção/subseção:** Série B · família B3   **Tempo:** sob demanda · ~45 s
- **LEDGER:** INTRODUZ o problema aberto de alocação de capacidade em aprendizado multitarefa (contexto de literatura)
- **Na tela:**
  `B-KARPATHY` · **context for `B-P1`**
  - Karpathy (2019), on designing a multitask network: *"how much feature sharing is there"*, *"tasks fight for the same shared capacity"*, *"there's finite capacity to go around... I don't really have language to describe how to correctly allocate capacity to tasks"*.
  - Standley et al. (ICML 2020): the task-affinity matrix, which asks the same question empirically.
  - PCGrad and GradNorm, which that discussion names, are **already in this dissertation** (Chapter 2).
  - **The point:** the capacity control in `B-P1` is exactly the measurement this literature says the field does not know how to design in advance. Running it is the answer to an open problem, not a wound.
- **Fala (PT):** "Isso é um problema aberto reconhecido da área, e não uma fragilidade só deste trabalho. Quando o Karpathy descreve como se desenha uma rede multitarefa, ele diz que não tem linguagem para descrever como alocar capacidade corretamente entre tarefas, e a matriz de afinidade entre tarefas do Standley e colegas ataca a mesma pergunta empiricamente. Os dois métodos que ele discute, PCGrad e GradNorm, já estão no meu Capítulo 2. O controle de capacidade que eu rodei é exatamente a medição que essa literatura diz que ninguém sabe desenhar de antemão."
- **Proveniência:** Karpathy (2019), palestra sobre desenho de redes multitarefa; Standley et al., ICML 2020; Cap. 2, §2.3 (PCGrad, GradNorm). **contexto de literatura, fora dos dois volumes.**
- **Nunca dizer:** nunca na trilha principal. Nunca usar *"tasks fight for capacity"* para **explicar** os resultados entregues: é exagero na direção oposta à posição simétrica de §5.3, e concederia a posição sobre o tronco.

---

### SB18 · B3-7 · B-MTLCHECK · Vocês reescreveram o sistema. Os números mudaram?
- **Seção/subseção:** Série B · família B3   **Tempo:** sob demanda · ~60 s
- **LEDGER:** INTRODUZ a reimplementação independente e a paridade de oito células
- **Na tela:**
  `B-MTLCHECK`
  - A clean reimplementation, written without reusing code from the old repository, running from raw check-ins to trained models. **No file produced by the old repository enters the path.**
  - Eight cells at Alabama and Arizona, under **the chapter's own protocol** (five flat folds): **mean delta −0.001 pp**, largest single deviation **0.421 pp**, which sits inside the seed spread the chapter itself prints (0.04 to 0.22).
  - Two caveats the sentence must carry: **one seed** on the new side against four on the chapter's; and the two columns are **not the same configuration** (the representation was rebuilt, the region tower unified, one component corrected).
  - **This is extra material. It does not correct Chapter 5.**
- **Fala (PT):** "O sistema foi reescrito do zero, sem reaproveitar código, e reproduz a tabela do Capítulo 5 com desvio médio de um milésimo de ponto em oito células; o maior desvio individual é de quatro décimos, e fica na ordem do desvio entre sementes que a própria tabela publica. Duas ressalvas viajam junto: uma semente do meu lado contra quatro do lado dela, e as duas colunas não são a mesma configuração, porque a representação foi reconstruída e a torre de região unificada. É material extra, e não corrige o Capítulo 5."
- **Proveniência:** `wrapup/NEW_VERSION.md` §3 (tabela de oito células) e §11 (procedência); evidência em `studies/porting_validation/evidence/estado_vs_dissertacao.json`. **pós-submissão: não consta em nenhum dos dois volumes.**
- **Nunca dizer:** misturar um número da reescrita com um da dissertação na mesma frase. Os protocolos são outros (divisões aninhadas 70/10/20, métricas agrupadas fora de dobra, margem derivada de 0,4 pp contra os 2 pp registrados). Sob aquele protocolo, Alabama/região **vira inferior**, e isso vem de outro livro de regras.

---

### SB19 · B3-8 · As contagens de parâmetros do Apêndice G do suplemento estão certas?
- **Seção/subseção:** Série B · família B3   **Tempo:** sob demanda · ~50 s
- **LEDGER:** INTRODUZ as contagens corrigidas do Apêndice G do suplemento
- **Na tela:**
  `B-APXG`
  - **Printed (supplement, Appendix G, Table 8):** joint 4,197,621 (AL) and 5,151,189 (CA); original dedicated 644,359; wider dedicated 4,207,399 and 5,249,719, labeled 100.2% and 101.9% of the joint budget.
  - **Recounted against an independent implementation of the head:** 1,433,863 · 9,634,471 (**230%**) · 12,044,791 (**234%**). The published figures counted the same widths at the wrong depth (two layers instead of the four the cell inherits).
  - **The conclusion does not fall. It gets stronger.** The wider arm was not capacity-matched: it received **more than double** the joint budget and still scored **lower** than the narrow model.
  - What does not survive: the labels *"100.2% / 101.9% matched"*.
  - The result columns of Table 7 (Alabama −0.53, *p* = 0.0011; California, one seed, three arms within 0.06) are unaffected.
- **Fala (PT):** "As larguras estão certas; as contagens foram feitas pela profundidade errada, duas camadas em vez das quatro que a célula executada herda. Recontadas contra uma implementação independente da cabeça, os três números sobem, e o braço alargado passa a ter mais que o dobro do orçamento do modelo conjunto, não cem por cento dele. A conclusão do controle fica mais forte: eu dou ao dedicado mais que o dobro e ele não recupera. O que não sobrevive é o rótulo de pareado. Há errata."
- **Proveniência:** suplemento, Apêndice G, Tabelas 7 e 8, pp. 24-26; recontagem em `wrapup/NEW_VERSION.md` §10.6. **pós-submissão: não consta em nenhum dos dois volumes.**
- **Nunca dizer:** `100,2%` e `101,9%` como se fossem pareamento de capacidade (§8 regra 9, proibição literal).

---

### SB20 · B4-LEAK · O embedding do Capítulo 3 também devolve a própria categoria do lugar?
- **Seção/subseção:** Série B · família B4   **Tempo:** sob demanda · ~75 s
- **LEDGER:** INTRODUZ o canal indireto de rótulo medido no embedding do Cap. 3 (auditoria de código)
- **Na tela:**
  `B4-LEAK` · **the two chapters differ by mechanism, not by innocence**
  - **Chapter 4:** the venue-type feature maps **one to one** onto the seven top-level categories, 284 to 365 distinct values per state, **not one ambiguous**. That is an **exact lookup**.
  - **Chapter 3:** each place's input feature is built from the **average of its neighbors' categories**, with the place's own one-hot excluded by construction. **The exclusion holds exactly, and it does not close the channel.** Each neighbor's feature is itself an average over that neighbor's neighborhood, and the place belongs to those neighborhoods.
  - Measured: the place's own category returns at a **mean weight of 0.10** against a **total own-category weight of 0.39**. Removing that contribution and changing nothing else lowers a probe of the place's own category from **0.46 to 0.30 macro-F1** (0 to 1 scale), against a **majority-class floor of 0.07**.
  - Confirmed by causal intervention: relabeling one place, without touching that place's own input row, changes the embedding of every place that receives a message.
  - **The formulation is the code audit's: an exact lookup in Chapter 4, a diluted average recovered through one hop in Chapter 3.**
  - **Does it invalidate the chapters? No.** The **sequential** task of both never has the target in its input, and that is where the arc's conclusions come from. Chapter 4's preface: *"every claim this chapter makes about the sequential task [...] stands as published"*.
  `Next-POI Prediction = next category (Def. 2.7)` · Caps. 3/4 report **per-category F1**
- **Fala (PT):** "Os dois capítulos têm o problema, e a diferença é de mecanismo, não de inocência. No Capítulo 4 a feature de tipo de local mapeia um para um nas categorias: é consulta direta, e o prefácio já declara isso. No Capítulo 3 o vetor de entrada de cada POI exclui o one-hot dele próprio, por construção, e eu verifiquei que essa exclusão vale exatamente. Ela não fecha o canal. Cada vizinho tem, no próprio vetor, a média do bairro dele, e o lugar está nesse bairro, então um salto de agregação devolve o rótulo. Medido: peso médio de zero vírgula dez contra zero vírgula trinta e nove de peso total da própria categoria, e uma sonda do próprio rótulo cai de zero vírgula quarenta e seis para zero vírgula trinta de macro-F1, contra um piso de classe majoritária de zero vírgula zero sete. E confirmei por intervenção causal. A pergunta que importa vem depois: isso invalida os capítulos? Não, porque a tarefa sequencial dos dois nunca teve o alvo na entrada, e é dela que vêm as conclusões que o arco carrega."
- **Proveniência:** auditoria de código do repositório (`DGI-leak-audit`), texto registrado em `wrapup/erratas/material_apx_static_scope.tex`; prefácio do Cap. 4, p. 52; range 284 a 365 reproduzido sobre os cinco parquets de estado. ⚠ **o texto que registra isto não chegou a nenhum dos dois volumes**: o apêndice que o carrega não é chamado por nenhum `\input` vivo. **Resposta oral apoiada no repositório, não citação do documento.**
- **Nunca dizer:** **"o DGI não vaza"**. A auditoria foi encomendada precisamente para não depender dessa crença, e mediu que o canal não fecha. O que se afirma é a diferença de mecanismo e de grau.

---

### SB21 · B4-DGI · O objetivo contrastivo do DGI está fazendo o que vocês pensam?
- **Seção/subseção:** Série B · família B4   **Tempo:** sob demanda · ~45 s
- **LEDGER:** INTRODUZ a constatação incidental sobre o objetivo contrastivo do Cap. 3
- **Na tela:**
  `B4-DGI` · **the limit is the headline**
  - An audit recorded, incidentally and outside the leak question, that the contrastive objective **as implemented appears degenerate**: positive and negative score sets identical as multisets, and a measured loss floor matching `2·ln2` to six decimals.
  - If confirmed, *"trained DGI embedding"* may not describe what Chapter 3 actually used.
  - **What I cannot say is that it was re-measured. It was not.** The artifacts of that audit are not in the repository and the observation was never acted on.
  - **What it would limit:** Chapter 3 is the **null result** of the arc. A weaker representation than assumed makes the null **less** surprising, not more. And neither later chapter inherits DGI: Chapter 4 replaces it, Chapter 5 builds on the check-in-level representation.
- **Fala (PT):** "Uma auditoria registrou, de passagem, que o objetivo contrastivo daquele capítulo, como implementado, parece degenerado: os conjuntos de escore positivo e negativo são idênticos como multiconjuntos, e o piso de perda medido bate dois ln dois em seis casas. Se isso se confirmar, chamar aquilo de embedding treinado pode não descrever o que o capítulo usou. O que eu não posso dizer é que foi re-medido, porque não foi. E o que isso limitaria: o Capítulo 3 é o nulo do arco, e uma representação mais fraca que a suposta torna o nulo menos surpreendente, não mais. Os dois capítulos seguintes não herdam aquele embedding."
- **Proveniência:** ledger §F da auditoria de código, registrado no comentário de aval aberto em `wrapup/erratas/material_apx_static_scope.tex`. **não consta em nenhum dos dois volumes.**
- **Nunca dizer:** que foi re-medido. **Se ninguém perguntar, não levantar.**

---

### SB22 · B4-1 · A entrada decomposta tem 192 dimensões e a baseline tem 64. Quanto do ganho é só largura?
- **Seção/subseção:** Série B · família B4 (Q17)   **Tempo:** sob demanda · ~45 s
- **LEDGER:** RETOMA a comparação não pareada em largura (4.4)
- **Na tela:**
  `B4-1`
  - MTLnet projects **any** input to the same 256-dimensional shared space through the task-specific encoders, *"so that the capacity of the shared layers and task heads remains unchanged across the evaluated models"* (p. 61).
  - The **input** width differs: 192 against 64. The published chapter says so on the same page: *"the difference in input dimensionality may influence part of the observed gains"*, and asks for a control that equalizes dimensionality.
  - Chapter 6, p. 87, repeats the requirement: *"Chapter 4 therefore calls for an equal-dimension control to separate the semantic contribution of the encoders from the effect of the additional width"*.
  - **The equal-dimension control has not been run. It is declared, not defended.**
  `Next-POI Prediction = next category (Def. 2.7)` · Cap. 4 reports **per-category F1**
- **Fala (PT):** "As camadas compartilhadas e as cabeças têm capacidade idêntica nos dois braços, porque a arquitetura projeta qualquer entrada ao mesmo espaço de duzentos e cinquenta e seis dimensões. O que difere é a largura de entrada, cento e noventa e dois contra sessenta e quatro, e o próprio capítulo publicado diz que isso pode influenciar parte dos ganhos e pede o controle de dimensão igual. O Capítulo 6 repete a exigência. O controle não foi executado, e eu declaro isso como limite em vez de defender."
- **Proveniência:** Cap. 4, p. 61; Cap. 6, p. 87.
- **Nunca dizer:** "pareado em largura".

---

### SB23 · B4-2 · Em Travel a baseline continua ganhando. O senhor tem uma explicação ou uma desculpa?
- **Seção/subseção:** Série B · família B4 (Q18)   **Tempo:** sob demanda · ~50 s
- **LEDGER:** RETOMA Travel rotulado por tarefa (4.4)
- **Na tela:**
  `B4-2` · **label Travel by task, because it moves in opposite directions**
  | Travel, Florida | MTLnet | ST-MTLNet (SIREN) |
  |---|---:|---:|
  | **category** (Table 6) | 45.49 ± 1.20 | **64.89 ± 1.20** |
  | **next category** (Table 7) | **64.47 ± 1.02** | 45.00 ± 1.10 |
  - California repeats the pattern on the sequential task: 46.05 ± 0.84 against 36.94 ± 1.70 and 37.82 ± 1.04. **Texas does not**: there the variants lead (34.26 ± 0.90 against 29.71 ± 1.32).
  - The chapter's own reading, p. 64: Travel *"tends to involve sparser movements between distant regions"*, and *"the graph topology used by DGI may be more efficient for preserving relationships between geographically distant POIs"*, with the two representation types capturing complementary aspects.
  - The chapter declares this as a limitation (p. 66), together with the fact that the three encoders are used jointly and no individual contribution is isolated.
  `Next-POI Prediction = next category (Def. 2.7)` · Cap. 4 reports **per-category F1**
- **Fala (PT):** "Primeiro rotulo a tarefa, porque Travel se move em direções opostas nas duas e a sala se confunde. Em categoria, Travel é onde a decomposição mais ganha: em Flórida vai de quarenta e cinco vírgula quarenta e nove para sessenta e quatro vírgula oitenta e nove. Na tarefa sequencial, é onde ela mais perde: sessenta e quatro vírgula quarenta e sete da baseline contra quarenta e cinco. A explicação do capítulo é que Travel envolve movimentos esparsos entre regiões distantes, e ali a topologia de grafo preserva melhor as relações entre POIs geograficamente distantes, enquanto codificadores de coordenada capturam padrão local. E o capítulo declara isso como limitação, junto com o fato de que os três codificadores são usados em conjunto e ele não isola a contribuição de cada um."
- **Proveniência:** Cap. 4, Tabela 6 e Tabela 7, p. 65; leitura em p. 64; limitação em p. 66.
- **Nunca dizer:** deixar o ganho da tarefa estática falar pela sequencial.

---

### SB24 · B4-3 · O melhor codificador espacial depende do estado. Isso não é escolha de hiperparâmetro disfarçada?
- **Seção/subseção:** Série B · família B4   **Tempo:** sob demanda · ~50 s
- **LEDGER:** INTRODUZ a acusação de overfitting de seleção e o ganho por variante isolada (SIREN no Texas) | RETOMA a declaração de melhor-de-dois por linha (4.2)
- **Na tela:**
  `B4-3`
  - **The range is best-of-two per row, and saying so is the answer.** Category gains of **20.2 to 22.0** points per state count, for each category-state cell, the better of SIREN and Sphere2Vec-M.
  - Read **by isolated variant**, SIREN alone at Texas averages **+17.89**, outside the announced range.
  - On the sequential task the same rule gives **15 of 21** category-state combinations to the variants, with one technical tie at Outdoors in Florida (the baseline mean above the best variant by 0.02, inside one standard deviation).
  - The chapter states the pattern as an observation: SIREN stands out in Florida and California, Sphere2Vec-M more often in Texas. **There is no universally better spatial encoder here, and no rule for choosing one without seeing the evaluation.**
  `Next-POI Prediction = next category (Def. 2.7)` · Cap. 4 reports **per-category F1**
- **Fala (PT):** "É melhor de dois por linha, e a resposta é dizer isso. O intervalo de vinte vírgula dois a vinte e dois pontos conta, em cada célula, o melhor dos dois codificadores. Lido por variante isolada, o SIREN sozinho no Texas rende dezessete vírgula oitenta e nove em média, fora do intervalo anunciado. Na tarefa sequencial, o mesmo critério dá quinze de vinte e uma combinações às variantes, com o que o capítulo chama de um empate técnico em Outdoors na Flórida, a dois centésimos e dentro de um desvio padrão. E o capítulo declara o padrão como observação: não há codificador espacial universalmente melhor, e eu não tenho regra para escolher um sem olhar a avaliação."
- **Proveniência:** Cap. 4, p. 44 do texto vivo (`chapters/4_courb/results.tex:44,:62`), Tabelas 6 e 7, p. 65; a leitura por variante isolada em `articles/CoUrb_2026/slides/judge_feedback.md:11`.
- **Nunca dizer:** apresentar a faixa sem declarar que é melhor de dois por linha.

---

### SB25 · B4-4 · Os números do corpus de Florida mudam entre capítulos. 990.518 no Cap. 3 e 1.407.034 no Cap. 5. Qual está certo?
- **Seção/subseção:** Série B · família B4 (Q20)   **Tempo:** sob demanda · ~45 s
- **LEDGER:** RETOMA a base de evidência e as duas extrações de Florida (2.3)
- **Na tela:**
  `B4-4` · **two extractions of one state of one public dataset, not a discrepancy**
  | | users | POIs | check-ins |
  |---|---:|---:|---:|
  | Chapters 3 and 4 | 20,301 | 65,009 | 990,518 |
  | Chapter 5 | 21,052 | 76,544 | 1,407,034 |
  - The mechanism is declared: the category-mapping table was extended about eleven months after the earlier extraction, and the added places fall mostly in Entertainment, Outdoors and Travel.
  - A controlled comparison confirms that **every POI, user and check-in of the earlier extraction reappears in the current one**, which adds others.
  - The three Chapter 3 figures are themselves a declared errata: the published article left placeholders, and the values come from the published CoUrb table.
- **Fala (PT):** "São duas extrações do mesmo estado do mesmo conjunto público, não um conflito. A tabela de mapeamento de categorias foi estendida cerca de onze meses depois da extração anterior, e os lugares acrescentados caem majoritariamente em Entertainment, Outdoors e Travel. Uma comparação controlada confirma que cada POI, usuário e check-in da extração anterior reaparece na atual, que adiciona outros. Cada capítulo reporta o corpus como o pipeline da sua época o produziu, e os três números do Capítulo 3 são, eles próprios, uma errata declarada."
- **Proveniência:** suplemento, §B.4, pp. 13-14; registro em `src_utils/cbic_recompute_result.md`.
- **Nunca dizer:** "superconjunto". Não há evidência de contenção declarada nesses termos no texto; o que há é a reaparição verificada de cada registro.

---

### SB26 · B4-5 · Quanto custou o modelo conjunto do Capítulo 3?
- **Seção/subseção:** Série B · família B4   **Tempo:** sob demanda · ~45 s
- **LEDGER:** INTRODUZ o custo de convergência do Cap. 3 (Tabela 4)
- **Na tela:**
  `B4-5` · **the table, never the published prose**
  | Model | Time (s) | Epochs | MFLOPs |
  |---|---:|---:|---:|
  | Category | 16.26 | 3.8 | 2.315 |
  | Next | 18.71 | 3.2 | 0.012 |
  | **MTL** | **80.88** | 3.2 | 0.234 |
  - Wall time to reach the target F1 scores: **80.88 s against the cumulative 34.97 s** of the two single-task models, about **2.3 times**.
  - MFLOPs do not follow the same pattern, and the chapter says so.
  - **The published prose of that article carries two defective statements about this measurement**, both registered in its own errata. **I quote the table, not the prose.**
  `5-fold cross-validation` · targets: Category F1 47, Next F1 32.2 · `Next-POI Prediction = next category (Def. 2.7)`
- **Fala (PT):** "A tabela é esta, e eu leio dela, não da prosa publicada. Para alcançar os alvos de F1, o modelo conjunto levou oitenta vírgula oitenta e oito segundos de tempo de parede, contra trinta e quatro vírgula noventa e sete somados dos dois modelos de tarefa única, cerca de duas vírgula três vezes. Em MFLOPs o padrão não se repete, e o capítulo diz isso. A prosa publicada daquele artigo carrega duas frases defeituosas sobre esta medição, e as duas estão registradas na errata dele."
- **Proveniência:** Cap. 3, Tabela 4, e o parágrafo de leitura na mesma seção; erratas em `articles/CBIC___MTL/ERRATA.md` e Apêndice B do **suplemento**.
- **Nunca dizer:** "quase quatro vezes" e "cerca do dobro" em MFLOPs (as duas frases defeituosas da prosa publicada). Nunca MFLOPs sem o enquadramento de tempo de parede (§8 regra 16).

---

### SB27 · Q5 · As duas entradas do modelo conjunto vêm do mesmo grafo. Elas não são independentes. Qual o tamanho dessa dependência?
- **Seção/subseção:** Série B · família B5 (Q5, `[ABERTO]`)   **Tempo:** sob demanda · ~45 s
- **LEDGER:** INTRODUZ o limite sobre a dependência entre as duas entradas do modelo conjunto
- **Na tela:**
  `Q5` · **not measured, and the limit is the headline**
  - **Declared, without a number**, in Chapter 2, p. 27: MTLnet derives both inputs from one place embedding, whereas the joint model reads two tables exported from the same check-in-level representation, and *"The two tables share an origin by construction, so they are not independent views"*.
  - The only **quantified** boundary is architectural: on the spatial route the pooled place representation is **detached**, so the place-region and region-city objectives cannot update the check-in encoder through that branch (Appendix E of the main volume, p. 112).
  - That bounds gradient flow **inside representation training**. It does not quantify the **information overlap** between the two exported tables.
  - **Why it does not overturn the thesis:** the comparison that carries the conclusions is paired and internal, the joint model against the dedicated models under the same representation, windows and folds. Shared origin bears on the mechanistic reading of what the trunk does, which p. 84 already leaves open, not on the validity of the measured difference.
- **Fala (PT):** "As duas tabelas partilham origem por construção, e eu digo isso no Capítulo 2, página vinte e sete. A única fronteira quantificada é arquitetural: na rota espacial a representação de lugar entra com gradiente interrompido, e está no Apêndice E, página cento e doze. A magnitude da sobreposição de informação entre as duas entradas não foi medida. Isso afeta a leitura mecanística do que o tronco faz, que o texto já deixa aberta na página oitenta e quatro, e não a validade da diferença medida, porque a comparação é pareada e interna."
- **Proveniência:** Cap. 2, p. 27 (`chapters/2_fundamentals.tex:813`); Apêndice E do volume principal, p. 112 (`chapters/apx_h_check2hgi_joint_model.tex:172`); Cap. 5, p. 82 e p. 84.
- **Nunca dizer:** inventar um número para a dependência.

---

### SB28 · Q8 · O resultado de região é transferência entre tarefas, ou é a arquitetura e os parâmetros que o senhor acrescentou?
- **Seção/subseção:** Série B · família B5 (Q8, `[ABERTO]`)   **Tempo:** sob demanda · ~60 s
- **LEDGER:** INTRODUZ a triagem de uma dobra sobre o tronco em California e Texas
- **Na tela:**
  `Q8` · **I do not claim transfer, and the text does not either**
  - Chapter 5, p. 84: *"The evidence here does not separate their contributions"*, and the surviving claim is about the **design**: this combination, shared representation and private spatial path together, produces a joint region output above two dedicated models at the two datasets with the largest region vocabularies.
  - **A one-fold screen** (seed 0, `--only-fold 0`, three arms; **one number per arm, so it detects only a large effect**): at California the region advantage survives severing the trunk (**−0.099**) and survives deleting the category task as well (**−0.077**); at Texas it survives severing the trunk (**−0.120**). All under 0.15 point.
  - **The five-fold trunk ablation at those two datasets does not exist.** Five-fold ablations were run only at Alabama and Florida.
  - **Post-submission**, a capacity-matched control answers the other half: the reported region advantage **measures capacity** (→ `B-P1`).
- **Fala (PT):** "Eu não reivindico transferência, e o texto não reivindica. A afirmação da página oitenta e quatro é sobre o desenho completo, representação compartilhada mais caminho espacial privado. Uma triagem de uma dobra mostra que a vantagem de região na Califórnia sobrevive a severar o tronco e a deletar a tarefa de categoria, movendo menos de zero vírgula quinze ponto, e no Texas sobrevive a severar o tronco. É uma triagem: uma dobra dá um número por braço, então ela só detecta efeito grande, e tinha poder para a hipótese de que o tronco carrega os dois pontos. Não a confirmou. A ablação de cinco dobras nesses dois conjuntos não existe, e eu declaro isso. E a outra metade da pergunta, os parâmetros, eu respondo no slide seguinte, com o controle de capacidade."
- **Proveniência:** Cap. 5, p. 84; triagem em `docs/studies/closing_data/v18/region_1fold_triage/FINDING.md`; ausência da ablação de cinco dobras em `docs/studies/closing_data/v18/SWEEP_PLAN.md:285-290`. **os números da triagem são de repositório: não constam em nenhum dos dois volumes.**
- **Nunca dizer:** creditar TX/CA a transferência entre tarefas. Nunca apresentar os valores absolutos da triagem ao lado das células da Tabela 10: são convenções diferentes (uma dobra, uma semente).

---

### SB29 · U1 · O tronco compartilhado contribui algo em Texas e California?
- **Seção/subseção:** Série B · família B5 (U1)   **Tempo:** sob demanda · ~40 s
- **LEDGER:** RETOMA a triagem de uma dobra (Q8) | RETOMA a posição sobre o tronco (5.5)
- **Na tela:**
  `U1` · **not measured**
  - What exists: the one-fold screen of `Q8`, where every arm moves under **0.15** point.
  - The five-fold ablation at those two datasets **does not exist**. Five-fold arms were run at **Alabama** (dcat −0.015 / dreg −0.138, and dcat −0.154 / dreg −0.004) and **Florida** (dcat +0.002 / dreg +0.026).
  - **Why it does not overturn the thesis:** the thesis does not claim the trunk carries the result. The claim on p. 84 is about the full design, and the same page states that the evidence does not separate the two parts. The screen had power for the hypothesis that the trunk carries the two points and did not confirm it, which **supports** the cautious wording rather than contradicting it.
- **Fala (PT):** "Não foi medido em cinco dobras nesses dois conjuntos, e eu digo isso. O que existe é a triagem de uma dobra, onde todos os braços se movem menos de zero vírgula quinze ponto, e ablações de cinco dobras em Alabama e Flórida. Isso não derruba a tese porque a tese não afirma que o tronco carrega o resultado: a afirmação é sobre o desenho completo, e a mesma página declara que a evidência não separa as duas partes."
- **Proveniência:** `region_1fold_triage/FINDING.md`; `SWEEP_PLAN.md:275-277,:285-290`; Cap. 5, p. 84. **repositório: não consta em nenhum dos dois volumes.**
- **Nunca dizer:** "não podemos provar que não contribuiu, portanto provavelmente contribuiu".

---

### SB30 · U2 · A vantagem de região sobrevive a um controle de capacidade pareada?
- **Seção/subseção:** Série B · família B5 (U2)   **Tempo:** sob demanda · ~45 s
- **LEDGER:** RETOMA o controle de capacidade pareada (B-P1) | RETOMA o Apêndice G do suplemento
- **Na tela:**
  `U2` · **not measured on the region axis, in either volume**
  - The only capacity control in the two volumes is **Appendix G of the supplement**, and it covers **next category**: at Alabama, multiplying the dedicated model's trainable parameters by 6.5 **lowers** its macro-F1 by **0.53** (*p* = 0.0011, direction unanimous over the twenty folds); at California, one seed, the three arms lie within **0.06**. The appendix states its own limit: *"What the control does not do is decompose the joint model: it holds the representation fixed and varies width"*.
  - **Post-submission it was measured on the region axis**, and the answer is unfavorable to the region reading (→ `B-P1`).
  - **Why it does not overturn the thesis:** on the **category** axis, where the representation thesis lives, the parameter-count explanation is tested and not supported. The two region gains are **secondary results outside the registered analysis plan**, declared on p. 76 and repeated on p. 88, and the parameter cost is reported on p. 73 rather than hidden.
- **Fala (PT):** "Nos dois volumes, não. O único controle de capacidade que eles carregam é o Apêndice G do suplemento, e ele cobre next category: em Alabama, multiplicar por seis e meio os parâmetros do dedicado baixa o macro-F1 dele em meio ponto, com direção unânime nas vinte dobras. Depois do envio eu rodei o controle no eixo de região, e a resposta não me favorece: está no slide do P1. Isso não derruba a tese porque a tese vive no eixo de categoria, e os dois ganhos de região são resultados secundários, fora do plano registrado, o que o texto declara em duas páginas."
- **Proveniência:** suplemento, Apêndice G, Tabela 7, pp. 24-26; Cap. 5, p. 73, p. 76, p. 88; `wrapup/post_submission_studies/P1_capacity_region.md`.
- **Nunca dizer:** "100,2% / 101,9% pareado" (contagens corrigidas em `B-APXG`).

---

### SB31 · U3 · Qual o tamanho da dependência entre as duas entradas do modelo conjunto?
- **Seção/subseção:** Série B · família B5 (U3)   **Tempo:** sob demanda · ~30 s
- **LEDGER:** RETOMA o limite sobre a dependência entre as duas entradas (Q5)
- **Na tela:**
  `U3` · **not measured** · same answer as `Q5`
  - Non-independence is **declared without a number** (Chapter 2, p. 27). The only quantified boundary is the detached copy on the spatial route (Appendix E of the main volume, p. 112).
  - **What would close it:** a single overlap quantity between the semantic and spatial windows on the same sequence, for example representational similarity or the information one window carries about the other under a linear probe.
  - **Why it does not overturn the thesis:** the comparison that carries the conclusions is paired and internal, under the same representation, windows and folds (p. 82).
- **Fala (PT):** "É a mesma pergunta do Q5, e a resposta é a mesma: não foi medido. O que fecharia isso é uma quantidade única de sobreposição entre as duas janelas na mesma sequência. E não derruba a tese porque a comparação que a sustenta é pareada e interna."
- **Proveniência:** Cap. 2, p. 27; Apêndice E do volume principal, p. 112; Cap. 5, p. 82.
- **Nunca dizer:** oferecer um número aproximado.

---

### SB32 · U4 · O cosseno entre os gradientes das duas tarefas continua ortogonal em Texas e California?
- **Seção/subseção:** Série B · família B5 (U4)   **Tempo:** sob demanda · ~45 s
- **LEDGER:** RETOMA o Apêndice D do volume principal (cosseno)
- **Na tela:**
  `U4` · **not measured at those two**
  - Appendix D of the main volume covers **four of the six** datasets, and says so: *"Texas and California are not measured here"*. Those are precisely the two where the joint model outperforms on region, and the appendix names that too: the diagnosis *"leaves the largest label spaces untested"*.
  - It also limits itself by architecture: *"Nothing here says the gradients stay orthogonal in a model that shares more of its depth, couples the tasks in a cascade, or shares an output layer"*.
  - At the four measured datasets, equivalence to zero holds within a **±0.05** margin, with every mean inside it and **99.6 percent** of individual epochs (Figure 8, Table 11).
  - **Why it does not overturn the thesis:** the appendix explains why a gradient balancer had nothing to balance, and that explanation is supported independently by the screen of nineteen balancers at two datasets (p. 70). No conclusion of Chapter 5 depends on the cosine at Texas or California.
- **Fala (PT):** "Não foi medido nesses dois, e o apêndice diz isso na própria tabela: ele cobre quatro dos seis conjuntos, e deixa os maiores espaços de rótulo sem teste. Nos quatro medidos, a equivalência a zero vale com margem de cinco centésimos, com todas as médias dentro dela. E o apêndice também se limita por arquitetura: ele não diz nada sobre um modelo que compartilhe mais profundidade ou acople as tarefas em cascata. Isso não derruba nada porque a função dele é explicar por que um balanceador de gradiente não tinha o que balancear, e essa explicação é sustentada de forma independente pela triagem de dezenove balanceadores."
- **Proveniência:** Apêndice D do volume principal, p. 106-108; Figura 8, p. 107; Tabela 11; triagem de balanceadores, Cap. 5, p. 70.
- **Nunca dizer:** chamar um cosseno próximo de zero de "nenhum conflito detectado" (o `GLOSSARY` exige a formulação de equivalência).

---

### SB33 · U5 · Quão longe fica a região predita quando o modelo erra?
- **Seção/subseção:** Série B · família B5 (U5)   **Tempo:** sob demanda · ~40 s
- **LEDGER:** RETOMA o terceiro dos quatro limites declarados (5.6)
- **Na tela:**
  `U5` · **not measured, and declared as such**
  - Chapter 5, p. 85: *"Where the shortlist misses, the geographic size of the error is the quantity that would matter to such a service, and measuring it requires the per-visit predictions that the evaluation path does not retain, so it is left to future work"*.
  - The service framing is explicitly motivation, not result, and is the **third** of the four declared limits: *"we do not build or evaluate a mobility-aware service"*.
  - **Why it does not overturn the thesis:** no claim in the document is about service performance. The shortlist reading on p. 84 is presented with the number that supports it, the Acc@10 of Table 10 itself: **California 64.54 percent** in ten tracts out of 8,501; **Texas 66.15 percent** in ten out of 6,553.
- **Fala (PT):** "Não foi medido, e o texto declara por quê: medir o tamanho geográfico do erro exige as predições por visita, que o caminho de avaliação não retém. Isso é trabalho futuro. E o enquadramento de serviço é motivação, não resultado: é o terceiro dos quatro limites. Nenhuma afirmação do documento é sobre desempenho de serviço; a leitura de lista curta vem com o número que a sustenta, que é o próprio Acc@10 da tabela."
- **Proveniência:** Cap. 5, p. 84 e p. 85; Tabela 10.
- **Nunca dizer:** especular sobre erro geográfico ou desempenho de serviço (§8 regra 16).

---

### SB34 · U6 · O senhor consegue separar a mudança de par de tarefas da mudança de representação e de topologia?
- **Seção/subseção:** Série B · família B5 (U6)   **Tempo:** sob demanda · ~45 s
- **LEDGER:** RETOMA a sexta limitação e o trabalho futuro amarrado a ela (6.3)
- **Na tela:**
  `U6` · **not measured, and the reason is structural**
  - Limitation 6, p. 90: no controlled ablation separates the change of representation and topology from the change of task pair.
  - The ablation that would separate them, running **static category classification under the check-in-level representation**, *"is not clean under that representation: the category of the visited place is an input feature of a check-in node, so that target would be partly readable from its own input"*. That follows from the design and was not measured, **so the confound is bounded by the fixed-pair control rather than removed**.
  - **Chapter 4 is the fixed-pair control**: same architecture, same task pair, only the input moves.
  - **Why it does not overturn the thesis:** the thesis is conditional by construction (p. 89), and it is Chapter 4, not Chapter 5, that carries the claim that the representation is the dominant factor.
  - Future work tied 1:1 to the limitation, p. 91: a static target the representation does not carry as an input feature.
- **Fala (PT):** "Não, e a razão é estrutural, não uma omissão. A ablação que separaria as duas seria rodar classificação estática de categoria sob a representação por check-in, e essa ablação não é limpa: a categoria do lugar visitado é feature de entrada de um nó de check-in, então o alvo seria parcialmente legível da própria entrada. Como não é limpa, o confundimento fica limitado pelo controle de par fixo em vez de removido, e esse controle é o Capítulo 4, que mantém o par e move só a entrada. A tese é condicional por construção, e é o Capítulo 4 que sustenta a afirmação sobre representação."
- **Proveniência:** Cap. 6, limitação 6, p. 90; trabalho futuro, p. 91; tese condicional, p. 89.
- **Nunca dizer:** "MTL funciona" sem condição.

---

### SB35 · U7 · A representação serve modelos que não são o seu?
- **Seção/subseção:** Série B · família B5 (U7)   **Tempo:** sob demanda · ~40 s
- **LEDGER:** RETOMA a terceira limitação (transdutividade) e o trabalho futuro (6.3)
- **Na tela:**
  `U7` · **not measured**
  - Chapter 6, p. 88: the result *"also supports testing Check2HGI in other mobility prediction architectures, although its benefit in those architectures has not yet been evaluated"*.
  - Limitation 3, p. 90: the representation is **transductive**, trained on each dataset's check-in graph, *"so it cannot represent unseen places or users without retraining"*.
  - **Why it does not overturn the thesis:** every comparison that carries the thesis holds the consuming model fixed and varies only the representation. Table 9 is exactly that design: same single-task model, same training configuration, same folds, same sliding windows, same epoch budget; only the input changes.
- **Fala (PT):** "Não foi avaliado, e o Capítulo 6 diz isso com essas palavras. E há um limite anterior: a representação é transdutiva, treinada no grafo de check-ins de cada conjunto, então ela não representa lugares ou usuários novos sem retreino. Isso não derruba a tese porque toda comparação que a sustenta mantém o modelo consumidor fixo e varia só a representação, que é exatamente o desenho da Tabela 9."
- **Proveniência:** Cap. 6, p. 88 e limitação 3, p. 90; Tabela 9, p. 79, e sua legenda.
- **Nunca dizer:** afirmar transferência da representação para outras arquiteturas.

---

### SB36 · U8 · A margem de dois pontos é o limiar em que um serviço se comportaria diferente. Isso foi medido em um serviço?
- **Seção/subseção:** Série B · família B5 (U8)   **Tempo:** sob demanda · ~45 s
- **LEDGER:** RETOMA a justificativa da margem de dois pontos (B1-5)
- **Na tela:**
  `U8` · **not measured in a service, and it is stated as a judgment**
  - The justification is a declared judgment: a service acts on which region will be busy, not on a single rank position, and *"A two-point change in Acc@10 is below the level at which this service would behave differently"* (p. 77).
  - The empirical support that does exist is the dispersion: the sd of the paired difference across the four user partitions runs **0.02 to 0.16**, and the intervals at **Istanbul, Arizona and Florida** would support a **one-point** margin. **Alabama's would not.**
  - **Why it does not overturn the thesis:** the margin was registered **before any result was read** (p. 76), and the four cells inside it clear it comfortably, the largest deficit being **0.87** (p. 82). Under a one-point margin three of the four datasets still hold by their own intervals, and the document names the exception.
- **Fala (PT):** "Não, e o texto apresenta a margem como julgamento, não como medição de serviço. O apoio empírico que existe é a dispersão: o desvio padrão da diferença pareada entre as quatro partições de usuários vai de dois a dezesseis centésimos, e os intervalos de Istambul, Arizona e Flórida sustentariam uma margem de um ponto. O de Alabama não, e o texto nomeia isso. E ela foi registrada antes de qualquer resultado ser lido, com as quatro células vencendo-a com folga, o maior déficit em zero vírgula oitenta e sete."
- **Proveniência:** Cap. 5, §5.5.3, pp. 76-77; p. 82.
- **Nunca dizer:** apresentar a margem como derivada dos dados.

---

### SB37 · B6-1 · O Resumo em português diz que o modelo conjunto superou os dedicados na próxima categoria em todos os conjuntos. O senhor sustenta isso?
- **Seção/subseção:** Série B · família B6   **Tempo:** sob demanda · ~40 s
- **LEDGER:** INTRODUZ o defeito do Resumo entregue e a errata aplicada
- **Na tela:**
  `B6-1` · **no. It is a defect, it was isolated, and it is corrected**
  | | what it says |
  |---|---|
  | **Resumo (delivered)** | superiority on next category *"em todos os conjuntos"* |
  | **English Abstract, §2.5, Ch. 5, Ch. 6** | superiority **at one dataset** |
  | **The delivered result** | **Florida only**, +0.19, Holm *p* 0.011 |
  - The defect was **isolated to the Portuguese Resumo**. Every other surface of the document already said one dataset.
  - Corrected in the source for the final deposit, with a registered errata. The delivered PDF is kept frozen as the record of what the banca received, and it differs from the source in this one sentence.
- **Fala (PT):** "Não sustento. É um defeito, e é isolado: o Abstract em inglês, a seção dois ponto cinco, o Capítulo 5 e o Capítulo 6 já diziam em um conjunto. O resultado entregue é superioridade na Flórida, mais zero vírgula dezenove, com p de Holm de zero vírgula zero onze. Está corrigido no fonte para a versão final, com errata registrada, e o PDF entregue fica congelado como o registro do que a banca recebeu."
- **Proveniência:** `wrapup/erratas/errata_resumo_escopo_categoria.tex`, aplicada ao fonte em 2026-08-21; `articles/dissertacao/CLAUDE.md` §1.2.
- **Nunca dizer:** a frase do Resumo entregue, em voz alta, exceto se perguntado (§8 regra 9 e decisão do autor).

---

### SB38 · B6-2 · Quando o senhor diz "Apêndice B", de qual dos dois documentos está falando?
- **Seção/subseção:** Série B · família B6   **Tempo:** sob demanda · ~35 s
- **LEDGER:** INTRODUZ a colisão de letras de apêndice entre os dois volumes
- **Na tela:**
  `B6-2` · **two volumes, colliding letters. Always name the volume.**
  | | **main volume** (119 pp) | **supplement** (27 pp) |
  |---|---|---|
  | A | Other Scientific Contributions | not present |
  | **B** | **AI-Use Disclosure** | **Errata to the Reproduced Articles** |
  | C | Data Ethics and Governance | not present |
  | **D** | **Why the Two Tasks Do Not Compete on the Shared Trunk** | **A Label-History Benchmark for the Next-Category Task** |
  | **E** | **How Check2HGI and the Joint Model Work** | **The Human-Subjects Question** |
  | F | not present | Adaptation of the HGI Baseline |
  | G | not present | A Parameter-Count Control for Next-Category Prediction |
  - The deposited text **deliberately does not point at the supplement**: it cites only itself and the repository. That is the document's policy.
- **Fala (PT):** "Preciso nomear o volume toda vez, porque as letras colidem. No volume principal, o Apêndice B é a Declaração de Uso de IA; no suplemento, é a Errata dos artigos reproduzidos. O D e o E também colidem. E o texto depositado não aponta para o suplemento de propósito: ele cita apenas a si próprio e o repositório."
- **Proveniência:** `articles/dissertacao/CLAUDE.md` §1; política declarada em `wrapup/erratas/README.md` (Q24).
- **Nunca dizer:** "Apêndice B" sem o volume (§8 regra 14).

---

### SB39 · B6-3 · Quantos usuários entram de fato no seu teste?
- **Seção/subseção:** Série B · família B6   **Tempo:** sob demanda · ~45 s
- **LEDGER:** INTRODUZ a distinção entre a coluna de usuários do corpus e a população pós-filtro
- **Na tela:**
  `B6-3` · **the table's user column is the raw corpus, and only the window column crosses the filter**
  | Table 8 column | population |
  |---|---|
  | check-ins · users · POIs | **raw corpus** |
  | windows | **after** the minimum-length filter (ten check-ins), stride 1 |
  - Verified by direct count on the raw files: Alabama 113,846 check-ins, **3,858 users**, 11,848 places; Arizona 236,450, **7,869**, 20,666. Those are exactly the printed values.
  - Counted on the representation actually used, the users **present in the pooled predictions** are **1,101** (AL), **2,136** (AZ) and **14,530** (Istanbul). The window column agrees exactly in all three (96,326 · 200,895 · 271,666), which is what shows it is the same representation.
  - Arithmetic closes both ways at Alabama: the 1,101 qualifying users hold 106,235 of the 113,846 check-ins, leaving 7,611 for the 2,757 who do not qualify, about 2.8 each.
  - **A caption clause fixes this without touching a number.**
- **Fala (PT):** "A coluna de usuários é do corpus bruto, e só a coluna de janelas atravessa o filtro de comprimento mínimo. Conferi por contagem direta nos arquivos brutos: os três números impressos de Alabama e Arizona batem ao caractere. Contando sobre a representação que roda, os usuários que efetivamente entram nas predições são mil cento e um em Alabama, dois mil cento e trinta e seis em Arizona. Três colunas de corpus e uma de experimento na mesma linha, com uma legenda que não distingue. Resolve-se com uma cláusula na legenda, sem tocar em nenhum número, e é uma errata que eu declaro."
- **Proveniência:** Tabela 8, p. 75 do volume principal (`src/tables/mobiwac/datasets.tex`); reconciliação e contagem direta em `wrapup/NEW_VERSION.md` §10.1. **a contagem pós-filtro é pós-submissão: não consta em nenhum dos dois volumes.**
- **Nunca dizer:** que os números impressos estão errados. Eles descrevem o corpus; o que falta é a legenda que distingue as duas populações.

---

### SB40 · B6-4 · O piso de Markov de região fica acima de três sistemas publicados. Isso não diz que a sua tarefa é fácil, ou que os externos foram mal executados?
- **Seção/subseção:** Série B · família B6 (Q22)   **Tempo:** sob demanda · ~50 s
- **LEDGER:** RETOMA o piso Markov-1 (5.4) e a comparação externa (5.5)
- **Na tela:**
  `B6-4` · **the chapter takes this head on, and declines a single explanation**
  - HMT-GRN falls below the floor at **all six** datasets, the ReHDM reference at **three**, STAN at **four**.
  - Two facts are declared about how the numbers were produced:
    - the floor is computed under **our own sliding windows and folds**, and the windows advance one visit at a time, *"so the region of the last visit is a strong predictor of the next one, and a first-order transition table reads exactly that signal"*. At Alabama the target is the last visited region in **32.9 percent** of windows.
    - the three systems do not meet the floor on equal terms: HMT-GRN on the same data, folds and initializations; STAN on the same folds but with its own representations and sequence construction; ReHDM under its own published protocol.
  - The chapter's closing: *"Neither fact establishes why the floor lies above the three systems, and we do not claim a single explanation. We treat the floor, not the external systems, as the reference the region task has to clear."*
  - The joint model exceeds the floor by **4.1 to 10.0** points at all six datasets.
- **Fala (PT):** "O piso é computado sob as minhas janelas, que avançam uma visita por vez, e em Alabama o alvo é a última região visitada em trinta e dois vírgula nove por cento das janelas, de modo que uma tabela de transição de primeira ordem lê exatamente esse sinal. E os três sistemas não encontram o piso em termos iguais: um roda nos mesmos dados, dobras e inicializações; outro nas mesmas dobras mas com representações próprias; o terceiro sob o protocolo publicado dele. Nenhum desses dois fatos estabelece por que o piso fica acima, e eu não reivindico uma explicação única. Eu trato o piso, e não os sistemas externos, como a referência que a tarefa de região tem de ultrapassar, e o modelo conjunto fica de quatro vírgula um a dez pontos acima dele nos seis conjuntos."
- **Proveniência:** Cap. 5, pp. 82-83; Tabela 10 e sua nota de rodapé.
- **Nunca dizer:** atribuir a ordenação a má execução dos sistemas externos.

---

### SB41 · B7-1 · Como o Check2HGI aprende sem ver os rótulos das duas tarefas?
- **Seção/subseção:** Série B · família B7 (Apêndice E do volume principal)   **Tempo:** sob demanda · ~60 s
- **LEDGER:** INTRODUZ o pipeline de cinco estágios e os três discriminadores bilineares do Apêndice E | RETOMA a ideia infomax (2.1)
- **Na tela:**
  `B7-1` · **Figure 9 (main volume, p. 111)**
  Five stages, and the boundary between the third and the fourth is the point:
  1. validate the records, order each user's visits by time, map each place to a polygon;
  2. build temporal, place and region graphs linked by the check-in / place / region / city hierarchy;
  3. train Check2HGI, export separate **64-dimensional** check-in and region tables;
  4. build stride-one windows, nine observed visits, the tenth as target;
  5. train one joint model on the two representation sequences.
  - **Check2HGI is fitted first and its exported tables stay fixed during supervised training.** The joint model does not rebuild the graph or update the representation encoder. **Check2HGI never receives the two forecast labels.**
  - Learning signal: **bilinear discriminators at three hierarchy boundaries** (check-in to place, place to region, region to city), each separating a true pairing from a corrupted one.
  - Two auxiliary terms: masked-place reconstruction (15 percent hidden, neighbors aggregated, category distribution reconstructed) and a place-table anchor.
  - Optimization: full-batch Adam, 500 epochs, learning rate 1e-3, clip norm 0.9; the saved state is the epoch with the smallest complete training loss.
- **Fala (PT):** "O objetivo que treina a representação distingue um pareamento verdadeiro de um corrompido, em três fronteiras da hierarquia: visita para lugar, lugar para região, região para cidade. Nenhum rótulo de próxima categoria ou próxima região entra ali. E a fronteira que importa é entre o terceiro e o quarto estágio: a representação é ajustada primeiro, e as duas tabelas exportadas ficam fixas durante o treino supervisionado. O modelo conjunto não reconstrói o grafo nem atualiza o codificador."
- **Proveniência:** Apêndice E do volume principal, §§ "The complete method at a glance" e "How Check2HGI learns and what it exports", pp. 109-113; Figura 9, p. 111; Tabela 12, p. 117.
- **Nunca dizer:** que a representação é causal ou indutiva. Ela é **transdutiva**, e o próprio capítulo diz.

---

### SB42 · B7-2 · O que exatamente entra num nó de check-in, e por que as arestas só correm para frente?
- **Seção/subseção:** Série B · família B7   **Tempo:** sob demanda · ~50 s
- **LEDGER:** INTRODUZ o decaimento exponencial de uma hora na aresta de sucessão e a ausência de coocorrência no conjunto de arestas | RETOMA as features de nó por visita e a aresta só para frente como princípio de projeto (5.2)
- **Na tela:**
  `B7-2`
  - Each check-in starts with **15 values**: seven category indicators, sine and cosine of hour of day, sine and cosine of day of week, and four elapsed-time values. Cyclic encoding makes the endpoints of the daily and weekly clocks adjacent.
  - **Coordinates are not appended.** They determine polygon membership, Delaunay edges and region adjacency instead.
  - **Check-in succession:** consecutive visits by the same user are connected **in one direction only**, earlier to later, with an edge weight decaying exponentially with the time interval, one-hour decay constant. The appendix states the reason as design: *"a target is predicted from a user's past, so a representation built for that target is constructed from the past alone, in training and at readout alike"*.
  - **Category co-occurrence is absent from the reported edge set.** Repeated visits to one place receive no extra check-in edge; they meet through their common place node.
- **Fala (PT):** "Cada nó de check-in começa com quinze valores: sete indicadores de categoria, seno e cosseno da hora do dia, seno e cosseno do dia da semana, e quatro tempos decorridos. As coordenadas não entram nesse vetor; elas decidem a que polígono o lugar pertence e quais arestas de Delaunay existem. E as arestas entre visitas consecutivas do mesmo usuário correm numa direção só, da anterior para a posterior, com peso caindo exponencialmente no intervalo de tempo. A razão está escrita como projeto: o alvo é predito do passado do usuário, então a representação é construída só do passado, no treino e na leitura."
- **Proveniência:** Apêndice E do volume principal, § "From check-in records to a heterogeneous mobility graph", p. 109-110; Tabela 12, p. 117, linha "Check-in input".
- **Nunca dizer:** "coocorrência" como canal deste grafo. E não narrar a direcionalidade como conserto: no documento ela é decisão de projeto (§2 do plano).

---

### SB43 · B7-3 · Como a hierarquia sobe de uma visita até a cidade?
- **Seção/subseção:** Série B · família B7   **Tempo:** sob demanda · ~55 s
- **LEDGER:** INTRODUZ a cópia com gradiente interrompido na rota espacial e o resumo de cidade ponderado por área (Apêndice E) | RETOMA o Check2HGI e o diagrama de níveis (5.2)
- **Na tela:**
  `B7-3` · **bottom-up, four steps (Figure 9)**
  1. **Two check-in graph-convolution layers** over the succession edges, residual update. Output: one 64-dimensional vector per visit. A row that described only the visit's own category and time now also reflects the visits that precede it.
  2. **Pool visits at their place** with four attention heads, one learned query shared across places, keys and values from that place's visits. Result: how a place is used, not only where it is.
  3. **Add the spatial place neighborhood:** combine the pooled place representation with a trainable place table initialized from a pretrained one, then one weighted graph convolution over the Delaunay place graph. **The pooled place representation is detached on this route**, so the place-region and region-city objectives cannot rewrite the check-in encoder through the spatial branch.
  4. **Region and city:** a second four-head pooling over the places of a region, a 64 to 64 convolution between adjacent region polygons, then an **area-weighted city summary**. The city vector serves the highest objective and is **not** an input to the joint model.
- **Fala (PT):** "O passe é de baixo para cima. Duas camadas de convolução sobre as arestas de sucessão dão o vetor por visita, e a partir dali uma linha que descrevia só a própria categoria e o próprio tempo passa a refletir as visitas que a precedem. Depois as visitas de um mesmo lugar são resumidas por atenção, com quatro cabeças, o que produz como um lugar é usado e não apenas onde ele fica. A rota espacial recebe esse resumo com o gradiente interrompido, e essa fronteira é deliberada: impede que as perdas geográficas mais altas reescrevam a representação temporal da visita. Em cima, região e cidade. O vetor de cidade serve ao objetivo mais alto e não entra no modelo conjunto."
- **Proveniência:** Apêndice E do volume principal, §§ "Step 1" a "Step 4", pp. 110-112; a cópia com gradiente interrompido em p. 112.
- **Nunca dizer:** nada aqui é resultado; é descrição de método.

---

### SB44 · B7-4 · O modelo conjunto é compartilhamento rígido com outro nome?
- **Seção/subseção:** Série B · família B7   **Tempo:** sob demanda · ~55 s
- **LEDGER:** INTRODUZ as projeções direcionais não amarradas do módulo de interação (Apêndice E) | RETOMA compartilhamento rígido (2.2) e o tronco de atenção cruzada (5.3)
- **Na tela:**
  `B7-4` · **Figure 10 (main volume, p. 114)**
  - **Private encoders, same shape, different parameters:** each history goes through 64 → 256 → 256 → 256, ReLU and layer normalization after every transformation, dropout 0.1 after the first two. Equal tensor widths do not imply equal meaning: a check-in vector describes one visit in a trajectory, a region vector a geographic area after spatial aggregation.
  - **Two bidirectional cross-attention blocks.** In each block the category stream queries the region stream first; the region stream then queries the already updated category stream. Four heads, padding positions excluded, each direction with its own projections, residual connections, normalizations and a 256 → 256 → 256 GELU feed-forward network.
  - **The interaction subsystem is jointly optimized, but the directional projections are not tied.** The appendix states the difference plainly: *"The model therefore differs from classical hard parameter sharing: the tasks keep private encoders and heads, while their activations meet in a fixed trainable interaction module that receives gradients from both losses."*
- **Fala (PT):** "Não. Os dois históricos entram em codificadores próprios, de mesma forma e parâmetros diferentes, porque largura igual não é significado igual: um vetor de check-in descreve uma visita numa trajetória, e um vetor de região descreve uma área depois de agregação espacial. O que compartilha é um módulo de interação: dois blocos de atenção cruzada bidirecional, em que a corrente de categoria consulta a de região e depois a de região consulta a de categoria já atualizada. As projeções das duas direções não são amarradas. As tarefas mantêm codificadores e cabeças privados, e o que se encontra são as ativações, num módulo que recebe gradiente das duas perdas."
- **Proveniência:** Apêndice E do volume principal, §§ "Step 1" e "Step 2" do modelo conjunto, pp. 113-115; Figura 10, p. 114; Tabela 12, p. 117, linhas "Private encoders" e "Interaction".
- **Nunca dizer:** chamar o desenho de compartilhamento rígido, e também não afirmar que o compartilhamento é o que produz o resultado (§5.3 do plano: a evidência não separa as contribuições).

---

### SB45 · B7-5 · Por que a região tem uma torre privada, e o que o β faz?
- **Seção/subseção:** Série B · família B7   **Tempo:** sob demanda · ~55 s
- **LEDGER:** INTRODUZ a fusão de duas torres na cabeça de região e o prior de transição inativo
- **Na tela:**
  `B7-5`
  - **Category head:** a four-layer unidirectional GRU, width 256, reading the category-context sequence; the top-layer state at the last valid position, then layer normalization, dropout, seven logits.
  - **Region head keeps two routes on purpose:**
    - the **private tower** reads the **raw 9×64 region history** with a spatio-temporal attention model, four heads, dropout 0.3;
    - the **shared-context tower** reads the 9×256 sequence produced by cross-attention, eight heads, dropout 0.1.
    - Each returns one 128-dimensional feature, fused as **f_R = f_priv + β · W_shr f_shr**, with **β trainable, initialized at 0.1**.
  - **This lets category context help region prediction without removing the direct spatial sequence model.**
  - **One term is inactive in the reported configuration:** an additive region-transition prior, whose scalar weight is **fixed at zero and not trained**, so it reaches neither the logits nor the gradients. The same table is never used as a training signal, and the category output is never an input to the region output. **Region prediction depends only on the two towers.**
- **Fala (PT):** "A cabeça de categoria é uma GRU de quatro camadas sobre a sequência de contexto. A de região mantém duas rotas de propósito: uma torre privada, que lê o histórico de região cru, com um modelo de atenção espaço-temporal, e uma torre de contexto, que lê a saída da atenção cruzada. As duas devolvem um vetor de cento e vinte e oito dimensões, e a fusão é aditiva, com um escalar treinável inicializado em zero vírgula um. Isso deixa o contexto de categoria ajudar a região sem remover o modelo sequencial espacial direto. E há um termo que fica inativo na configuração reportada: um prior aditivo de transição entre regiões, com peso fixado em zero e não treinado, então ele não chega nem aos logits nem aos gradientes."
- **Proveniência:** Apêndice E do volume principal, § "Step 3", pp. 115-116, e o parágrafo sobre os caminhos inativos; Tabela 12, p. 117, linhas "Category head", "Region head" e "Inactive region paths".
- **Nunca dizer:** escrever `log_T` ou qualquer nome de repositório para o prior; o nome do documento é **region-transition prior**.

---

### SB46 · B7-6 · O cross-attention atende histórico de região de um usuário com histórico de categoria de outro?
- **Seção/subseção:** Série B · família B7   **Tempo:** sob demanda · ~50 s
- **LEDGER:** INTRODUZ o pareamento aleatório de linhas entre os dois carregadores durante o treino
- **Na tela:**
  `B7-6` · **the sharpest question in the appendix, and the appendix answers it**
  - **During training: yes.** The category and region loaders use the **same user-disjoint fold** and **shuffle independently**, so their batch rows have compatible shapes but *"need not describe the same user or window in one optimizer step; cross-attention operates on that random pairing"*. The shorter loader cycles until the longer one is exhausted.
  - **At validation: no.** *"Validation rows are record-aligned."*
  - The appendix calls it what it is: *"This operational detail is unusual, but it is part of the reported training protocol."*
  - Objective: **fixed 0.50 / 0.50** cross-entropy, logit adjustment τ = 0.5 on the category term at train time only, region term unadjusted. AdamW with three parameter groups (category, region, shared), one backward pass through the connected model. Fifty epochs, five user-disjoint folds, seeds {0, 1, 7, 100}. **The checkpoint maximizes the geometric mean of category macro-F1 and region Acc@10, so selection depends on both tasks.**
- **Fala (PT):** "No treino, sim, e o apêndice declara isso em vez de esconder. Os dois carregadores usam a mesma dobra disjunta por usuário e embaralham de forma independente, então as linhas de um passo de otimização não descrevem necessariamente o mesmo usuário ou a mesma janela, e a atenção cruzada opera sobre esse pareamento. Na validação as linhas são alinhadas por registro. O próprio apêndice chama isso de detalhe operacional incomum, e diz que é parte do protocolo reportado. Se o senhor quiser a leitura mais dura: durante o treino a atenção cruzada aprende a usar contexto de região em geral, não o contexto daquele usuário, e o resultado é medido na validação alinhada."
- **Proveniência:** Apêndice E do volume principal, § "How the joint model is optimized and evaluated", pp. 116-117; Tabela 12, p. 117, linhas "Objective", "Optimization" e "Evaluation".
- **Nunca dizer:** apresentar isto como defeito corrigido. É o protocolo reportado, declarado no documento.
