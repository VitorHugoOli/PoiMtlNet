# Germano — anotações no PDF revisado (`_MOBIWAC__26_revisado.pdf`, 2026-07-10)

> Fonte: PDF anotado à mão (Notewise), 8 páginas; anotações nas páginas 1–5 (6–8 limpas).
> Notas transcritas verbatim (PT/EN como escritas); edições de tinta (riscos/substituições) descritas entre colchetes.

## Página 1 — Abstract

Qoute: "Two coarse questions about the next visit are usually enough: its category (the kind of place) and its
region (which part of the city)."
germano.santos: [reescrita na margem] "Two categories are usually enough: the place category and its region"

Qoute: "We therefore ask whether one model can learn both"
germano.santos: [risca "ask", escreve acima] "test"

Qoute: ", and what sharing a single model costs."
germano.santos: [inserção em azul após "costs"] "show the trade-off between shared training and performance"

Qoute: "and most of the gain comes from the per-visit context"
germano.santos: [palavra na margem, apontando para "per-visit context"] "characteristics"

Qoute: "a dedicated category model (about +5 to +9 macro-F1)"
germano.santos: [risca o "+9"]

Qoute: "and on region it outperforms the dedicated region model at four of the six datasets"
germano.santos: [margem "next" + inserção "task" sobre "region"] → sugerindo "on the next-region task it outperforms…"

Qoute: "the joint model outperforms on category and comes out slightly ahead on region."
germano.santos: [inserção em azul no fim do abstract] "the joint model outperforms on category "and achieves superior
performance" ...."

## Introdução

Qoute: "staging content where a user is heading [3] or planning capacity there before demand arrives."
germano.santos: [circula "there"] retirar

Qoute: "Predicted handovers already let cellular services adapt in advance at the network level [4];"
germano.santos: achei deslocado

Qoute: "the next category, the kind of place such as food or shopping"
germano.santos: [circula "kind of place"] "the place category/type"

Qoute: "One captures intent, the other geography;"
germano.santos: which one capture intent? It's better to use: "the first… the latter" → "the first task…"

Qoute: "Sharing one representation across tasks is not free;"
germano.santos: [circula "is not free"] "is not straightforward"

Qoute: "one task while hurting the other [6]. Our earlier work [7] observed exactly this for next-category and
next-region;"
germano.santos: não gosto de auto-citação assim. Cite o 6 e o 7 juntos: [6,7]
vitor: De fato citar nosso artigo parece um pouco estranho e o leitor não vaia abrir o nosos outros artigo para avaliar

Qoute: "the useful question is where sharing helps, where it costs, and how to share so the gains hold and the
cost stays small."
germano.santos: [circula "where it costs"] é isso mesmo. "where it costs" é um phrasing estranho

Qoute: "We introduce two changes."
germano.santos: [reescrita na margem] "Therefore, we propose two enhancements"

Qoute: "Per-visit context is not new on its own; what is new is obtaining it from inside this hierarchical graph
representation, a choice we test directly (Section VI-A)."
germano.santos: [trecho todo marcado] Sem necessidade

Qoute: "We evaluate on two deliberately different settings"
germano.santos: [risca "deliberately"]

Qoute: "On region, it outperforms at four of the six datasets and stays statistically non-inferior"
germano.santos: [reescrita] "Outperforms 4 out of 6 scenarios"

Qoute: "The category gain holds everywhere, and across the U.S. states the region result moves monotonically"
germano.santos: [circula "everywhere"] "holds on Istanbul and U.S. states"; [circula "and" em azul] "while"

Qoute: "from a non-inferior match (TOST, ±2 percentage points, or pp) at the small region counts to outperforming
the dedicated model at the large"
germano.santos: não entendi muito bem o que você quis dizer

Qoute: "Istanbul, the smallest dataset, also comes out ahead on region. Four of the six datasets are measured at
four seeds over five folds on both arms"
germano.santos: acho que dá para melhorar esse phrasing

## Background and Related Work

Qoute: "A place embedding turns a POI into a vector so that similar or nearby places sit close together, rather
than each being a bare, arbitrary index."
germano.santos: [risca "turns" → "transforms"; escreve "where?" após "close together"; risca "rather than each
being a bare, arbitrary index"]

Qoute: "Deep Graph Infomax (DGI) [9] learns such vectors self-supervised"
germano.santos: [em azul] DGI is not a place embedding model. It's a graph self-supervised model instead.
[+ marca "ly," em "self-supervised" → "self-supervisedly"]

Qoute: "training them to tell the real place network, linked by similarity, time, and distance, from a shuffled copy."
germano.santos: [risca "from a shuffled copy"]

Qoute: "they are what a mobility-aware service can act on, not to make the task easier;"
germano.santos: [risca "not to make the task easier"]

Qoute: "The category task itself follows our earlier line of work [7]."
germano.santos: Reviewers doesn't know about our earlier work and they aren't supposed to read to understand
this work

Qoute: "The nearest exceptions stop short of our pairing:"
germano.santos: this phrasing is weird

Qoute: [§II-C inteiro: "MCARNN [19] jointly predicts activity and location … we test the choice directly
(Section VI-B)."]
germano.santos: Revisa esse parágrafo. Está difícil de ler. [circula "headline"?, "instrumental step",
"instrumentally", "co-equal"?, "instrumental"? — repetição/jargão]

Qoute: "We confirm this: none of the balancers we tried, including the two named above, improved on a tuned fixed
task weighting in our model."
germano.santos: [frase riscada] Instead of this sentence is preferable to create a hypothesis

Qoute: "We order each user's check-ins in time, form windows of nine consecutive visits"
germano.santos: [risca "form" → "build"]

Qoute: "from about five hundred classes (Istanbul) to about eight thousand five hundred (California)."
germano.santos: Write the raw number or use a table for each city

## Method

Qoute: "Edges tie the levels together and link a user's consecutive check-ins, weighted by closeness in time"
germano.santos: Edges tie the levels? [sublinha "weighted by closeness in time" →] "weighted by time difference
of following visits on ordered sequence"

Qoute: "Each visit's category, time of day, and day of week enter as input features of its node"
germano.santos: [risca "enter" → "act"]

Qoute: "We train the graph mainly with an infomax objective"
germano.santos: Consegue colocar uma equação?

Qoute: "Each ingredient appears in prior work on its own; as Section II-A argued, the contribution is their
combination, and we show later that the combination, not extra supervision, makes the category task far easier
to learn."
germano.santos: [parágrafo circulado, "ingredient" circulado] Retirar

Qoute: "We then train one model that reads a window of recent per-visit vectors"
germano.santos: [risca "one" → "a"]

Qoute: "Two input streams, a semantic stream … the two streams exchange information … attention lets each stream
read the other's features, while each stream keeps its own feed-forward weights."
germano.santos: [circula "stream(s)" 6×] muita repetição

Qoute: "The training loss is a fixed-weight sum of the two outputs, the category output weighted 0.75 and the
region output 0.25, and both outputs use plain unweighted cross-entropy."
germano.santos: Pergunta clássica: foi escolhido α1 e α2 ou esses parâmetros foram tunados?

Qoute: "This is ordinary fixed-weight joint training"
germano.santos: [sublinha "ordinary"] ?

Qoute: "joint training [6], kept deliberately plain so that any improvement over the dedicated single-task models
comes from the shared representation, not from an adaptive weighting scheme."
germano.santos: não entendi o que é "deliberately plain"

Qoute: "The cost of serving both tasks this way is not free, and we state it plainly."
germano.santos: [circula "plainly"]

Qoute: "about 4.2 million parameters at Alabama against 1.1 million for the two combined (5.2 against 2.0 at
California)."
germano.santos: What about the multitasking model? How many parameters does it have?

## EXPERIMENTAL SETUP

Qoute: "The states range from about 114 thousand check-ins and 1,109 regions in Alabama to several million
check-ins and 8,501 regions in California."
germano.santos: Senti falta de uma tabela com essa análise descritiva
Vitor: Uai, não seria a tabela 3 ?

Qoute: "For each user with at least ten visits we form time-ordered overlapping nine-visit windows, one starting
at each visit, with the next visit as the target, giving both tasks more examples."
germano.santos: [risca "form" → "build"; risca "giving both tasks more examples"]

Qoute: "we keep one full-context window ending there and drop the padded duplicates."
germano.santos: [frase riscada] Não tem necessidade de explicar todo o pré-processamento

Qoute: "overlap cannot leak: a test user's visits never appear in training."
germano.santos: [risca "a test user's visits never appear in training"] Isso é desnecessário. Assume-se que o
modelo publicado não tem leakage
vitor: De fato colocamos isso e parte do Integrity of the representation, para provar parte do argumento do revisor do
Bracis que não estamos tendo data leak, mas acredito qeu nào vale ocupar espaço do texto com isso, podemos abri espaçõa
outros conteutod mais importantes, e quando ao Integrity of the representation, podemos resumir em uma frase

Qoute: "Food is the majority class in every dataset, from about 25 percent of visits in Florida to 34 percent in
Alabama."
germano.santos: Senti falta de um gráfico descritivo

Qoute: "we argue, and verify, that it carries no usable information about the test visits, on three grounds."
germano.santos: [risca "about the test visits"] linka com a tabela ou gráfico que diz isso

Qoute: "Second, because the graph is built over all places, the only test-side exposure is graph structure."
germano.santos: Não entendi

Qoute: "We measured it directly, rebuilding the representation per fold from its training users only and
re-running both tasks against the full-corpus version on the same folds. The effect is within fold noise"
germano.santos: Não entendi

Qoute: "This is a place-level proxy and the one residual we cannot fully measure"
germano.santos: [circula "we cannot fully measure"] Porque não?

Qoute: "the representation cannot score a place never seen in training"
germano.santos: [circula "cannot"] a gente não deveria nem usar contração em artigo

Qoute: "on the measurable visits, the large majority, the audit finds no inflation. Third,"
germano.santos: [circula "audit"] audit?

Qoute: [parágrafo inteiro "Integrity of the representation." — da abertura até "…held to a weaker standard than
ours."]
germano.santos: Eu não entendi esse parágrafo. Acho um bom candidato para remover
vitor: Concordo, podemos resumir esse paragrafo de forma mais clara e abri espaço par aoutros conteudos

Qoute: "C. Metrics, superiority versus non-inferiority"
germano.santos: [risca ", superiority versus non-inferiority" no título da seção]

## Página 5 — Metrics / Baselines

Qoute: "a paired test over the per-seed means with a Holm correction across the cells"
germano.santos: [circula "Holm correction"] Você criou isso? Tem citação? Nunca ouvi falar disso, tem alguma
definição?
Vitor: Hum, vamos avlaiar o quanto o Holm e usado nesse tipo de artigos e em artigos do mobiwas, e o que faz sentido fazer. Apesar qeuyeu acho que usar o holm da uma autoridade a mais para nós.

Qoute: "at the two single-seed states, a paired Wilcoxon signed-rank test over the folds"
germano.santos: [circula "single-seed"] o que é single-seed?

Qoute: "We fix the two-point margin in advance, on deployment grounds;"
germano.santos: deployment grounds?
vitor: Outra palavra vinda de um texto criado por IA

Qoute: "Where the joint model is meant to outperform, … giving power near 1.0 to reject a true two-point gap, so a
non-inferiority verdict reflects a real match."
germano.santos: Não entendi esse parágrafo também. Está um pouco confuso

Qoute: "For next-category this is a faithful re-implementation of POI-RGNN [29]"
germano.santos: [risca "faithful re-implementation"]
vitor: De fato explicar que é um faithful re-implementation talvez seja disnecearrio, já s esperae que seja, não ?

Qoute: "which serve the next-place head we do not predict"
germano.santos: [insere "that"] → "the next-place head that we do not predict"

Qoute: "We also run a faithful STAN [12], trained from raw with its own embeddings and sequence construction under
one audited recipe"
germano.santos: [circula "audited"] ?

## Páginas 6–8

Sem anotações (Tabelas I–III, Results, Discussion, Conclusion, References limpas).
