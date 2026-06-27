# Review — Germano Barcelos (markup on `_MOBIWAC_.pdf`)

Internal co-author review pass extracted from the PDF highlight/annotation layer of
`_MOBIWAC_.pdf` (paper: *"Predicting the Next Category and Region of a Visit: A Check-in-Level
Multi-Task Study on Mobility Data"*). **70 highlight comments**, all by **Germano Barcelos**.

Each entry shows the **exact highlighted span** (recovered via word-level bounding boxes) and
the reviewer's **verbatim** comment (mostly Portuguese), plus an English **Action** gloss.

Recurring themes: (1) swap poetic phrasing for standard ML jargon; (2) soften/justify novelty
claims; (3) move the MTL definition earlier and keep Related Work about *others'* work; (4) add
citations to strong claims; (5) check the references aren't outdated.

> **Co-author response (added 2026-06-27, advisor-reviewed).** A multi-agent pass evaluated all 70 comments
> against the current draft and our prior decisions, with two advisors checking for missing interpretations and
> conflicts. Verdicts: **Accept 29, Partial 29, Reject 12.** Each comment below carries a **Response** (verdict +
> rationale) and the **Edit** to make. Rejections push back honestly where a comment is taste-only, factually
> wrong, or conflicts with a decision already made (e.g. the softened abstract, the plain-words audience).

---

## Abstract (p1)

**1.**
> **Highlight:** "coarse"
> **Germano:** retirar
> **Action:** Remove the word "coarse".

> **Response — ❌ REJECT:** Keeping 'coarse' — it's the load-bearing scoping term, not filler, and it pre-empts the 'you just simplified the problem' attack (your own #48). The real risk is a connotation misread (coarse = crude/low-quality) for a non-ML reader, which the intro's '(less detailed)' gloss (#18) guards against; the abstract pairs it with the two named questions so it can't read as 'low-quality predictions'. No change to the word.
> 
> **Edit:** none (rejected)

**2.**
> **Highlight:** "city)." (…the next region (which part of the **city**))
> **Germano:** two questions of what? / Frase sem sentido
> **Action:** The sentence is unclear — next category/region *of what?* (of the next visit).

> **Response — ✅ ACCEPT:** Fair — the two questions aren't anchored to anything. Binding them to 'the next visit' fixes the 'of what?' ambiguity and also resolves #3 (suffice) and #4 (the same of-what concern on the category clause) in one rewrite.
> 
> **Edit:** Replace 'Two coarse questions usually suffice: the next category (the kind of place) and the next region (which part of the city).' with 'Two coarse questions about the next visit are usually enough: its category (the kind of place) and its region (which part of the city).'

**3.**
> **Highlight:** "suffice:"
> **Germano:** suffice????
> **Action:** Avoid "suffice" — reword.

> **Response — ✅ ACCEPT:** Agreed, 'suffice' reads oddly; 'are usually enough' is plainer for the networking audience. Folded into the #2 rewrite. Your deeper worry (is two questions really enough?) is the paper's thesis, argued in §3 and §5.3, not the abstract.
> 
> **Edit:** Within the #2 rewrite, 'usually suffice:' becomes 'are usually enough:'.

**4.**
> **Highlight:** "the next category (the kind of place) and the next region"
> **Germano:** next category and the next region of what?
> **Action:** Specify "of the next visit".

> **Response — ✅ ACCEPT:** Same valid point as #2, on the category clause. The #2 rewrite makes both targets explicit properties of the next visit ('its category ... and its region'), answering 'of what?' once for both — no separate edit.
> 
> **Edit:** Covered by the #2 rewrite.

**5.**
> **Highlight:** "what sharing a single representation costs."
> **Germano:** estranho
> **Action:** Awkward phrasing — rephrase.

> **Response — 🟡 PARTIAL:** The 'cost of sharing' is the paper's central research question, so I won't cut the idea, but the gerund subject reads heavy. Smoothing keeps the meaning. Note: #27 must stay consistent with this — it will NOT say 'without paying that cost' (that would contradict this line and overclaim).
> 
> **Edit:** Replace '... so we ask whether one model can learn both, and what sharing a single representation costs.' with '... so we ask whether one model can learn both, and what it costs to share one representation.'

**6.**
> **Highlight:** "instead of giving every place one fixed vector."
> **Germano:** retirar. O leitor pode ficar perdido aqui
> **Action:** Remove — too much detail for the abstract; reader can get lost.

> **Response — ❌ REJECT:** Keeping it — this is the one plain-language line that tells a non-ML reader what 'check-in-level representation' means: per-visit context versus one fixed vector per place. That contrast is the core idea and the paper's stated novelty, so for the MobiWac audience it clarifies rather than confuses. Cutting it leaves the term undefined exactly where it first appears.
> 
> **Edit:** none (rejected)

**7.**
> **Highlight:** "five U.S. states,"
> **Germano:** which ones?
> **Action:** Name the states (or defer to Setup).

> **Response — ❌ REJECT:** Enumerating the five states in the abstract is clutter abstracts conventionally avoid; they're named and tabulated in §5.1 / Table I. Your 'which ones?' is answered there, where it belongs.
> 
> **Edit:** none (rejected)

**8.**
> **Highlight:** "beats" (…this lifts… / it **beats** a dedicated category model)
> **Germano:** outperforms
> **Action:** Use "outperforms" instead of "beats".

> **Response — 🟡 PARTIAL:** I'm rejecting the swap to 'outperforms', Germano. 'Beats/matches' is the paper's DEFINED verdict vocabulary — §5.3 sets up 'which a beat and which a match is a result' (paired Wilcoxon for beats, TOST for matches) and all eight sections plus the Table III caption use it. Swapping only the abstract introduces a third superiority verb (the body also has 'wins') decoupled from the test that defines it. Your real point — three verbatim 'beats' in the abstract (#25) — is valid, so I'll vary sentence structure to break the triple while keeping the defined verb.
> 
> **Edit:** Keep 'beats' as the defined verdict verb. Restructure the abstract's region/Istanbul sentences to avoid three verbatim 'beats' (e.g., '...it beats a dedicated single-task category model (+4.7 to +7.7 macro-F1); on region it again beats the single-task model where regions are many, and matches it — statistically, within two points — where they are few.'). Do NOT introduce 'outperforms'.

**9.**
> **Highlight:** "dedicated category model"
> **Germano:** what is a dedicated model? Try to use more classical jargons
> **Action:** Define / replace "dedicated model" with standard terminology (single-task baseline).

> **Response — ✅ ACCEPT:** Good catch — 'dedicated model' is vague (dedicated to what?). But the paper's anchor name is 'dedicated single-task model' (§5.3) and Table III's column header is literally 'Dedicated', so the fix is to EXPAND on first use, not strip 'dedicated' (which would orphan the table). I'll write 'dedicated single-task category model' at first mention and keep 'dedicated model' as the established shorthand thereafter.
> 
> **Edit:** First use (abstract): 'dedicated category model' -> 'dedicated single-task category model'; thereafter 'dedicated model' as shorthand. Do NOT rename to bare 'single-task model'. Apply the same first-use expansion in the intro contribution bullets; leave Table III's 'Dedicated' column and the body shorthand intact.

**10.**
> **Highlight:** "beats" (…on region it **beats** the dedicated model)
> **Germano:** exceeds in performance
> **Action:** Use "exceeds in performance" instead of "beats".

> **Response — 🟡 PARTIAL:** Accept the spirit but keep 'beats', not 'outperforms' (it's the §5.3-defined verb; see #8). 'Beats' is honest here because region genuinely BEATS at the large counts (paired Wilcoxon superiority at FL/TX/CA), distinct from the small-count MATCH. I'll only fix the imprecise noun phrase 'region space is large' (see #11).
> 
> **Edit:** Replace 'on region it beats the dedicated model where the region space is large' with 'on region it beats the single-task model where there are many regions' (the 'many regions' part resolves #11).

**11.**
> **Highlight:** "region space"
> **Germano:** region area
> **Action:** Prefer "region area".

> **Response — 🟡 PARTIAL:** Pushing back on 'region area' — it's technically wrong here: it reads as geographic area (km^2), but the paper means the number of region classes (cardinality, ~500 in Istanbul up to ~8,500 in California), the opposite kind of 'size'. The clean fix is 'many regions', which also matches the next sentence ('grows with the number of regions').
> 
> **Edit:** Replace 'where the region space is large' with 'where there are many regions' (folded into the #10 edit).

**12.**
> **Highlight:** "matches it (statistically, within two points) where it is small."
> **Germano:** não entendi. O que seria *statistically, within two points*? Fez teste de hipótese?
> **Action:** Clarify "statistically, within two points" — state explicitly that a TOST hypothesis test was run.

> **Response — ❌ REJECT:** This conflicts with a prior decision: the abstract was deliberately softened ('matches it, statistically, within two points') with the TOST machinery deferred. Your underlying question — did we run a hypothesis test? — is fully answered: §5.3 names a 'two one-sided tests (TOST)' non-inferiority procedure (confirmed in 05_setup.tex L72) at a ±2 pp margin, applied per-cell in §6.2 with 90% CIs. Keeping the acronym out of the abstract; pointing you to §5.3/§6.2.
> 
> **Edit:** none (rejected)

**13.**
> **Highlight:** "spatial gain"
> **Germano:** spatial gain?
> **Action:** Define / rephrase "spatial gain".

> **Response — ✅ ACCEPT:** Agreed, 'spatial gain' is a coined phrase that's easy to misread. Plain 'the gain on region' is clearer for both audiences and ties directly to the region task. No information lost.
> 
> **Edit:** Replace 'The spatial gain grows with the number of regions.' with 'The gain on region grows with the number of regions.'

**14.**
> **Highlight:** "result holds:" (On a non-U.S. city (Istanbul) the **result holds**…)
> **Germano:** Our model also demonstrate superior performance on a non-U.S city (Istanbul) when comparing the next-category and next-region.
> **Action:** Suggested rewrite of the Istanbul sentence (use Germano's phrasing as a base).

> **Response — 🟡 PARTIAL:** I must reject your suggested wording — it claims 'superior performance' on BOTH tasks, which is factually wrong. Istanbul has ~500 mahalle (small count), so region is a statistical MATCH (TOST, within two points), NOT a beat; claiming superiority re-triggers exactly the overclaim a reviewer would flag. I accept that 'the result holds' is vague and make it explicit while keeping the beats-on-category / matches-on-region split, and keep the defined verb 'beats' (not 'outperforms', per #8).
> 
> **Edit:** Replace 'On a non-U.S. city (Istanbul) the result holds: it beats on category and stays within two points on region.' with 'On a non-U.S. city (Istanbul, where regions are the mahalle) the same pattern holds: it beats on category and statistically matches the single-task model on region (within two points).'
---

## Introduction (p1)

**15.**
> **Highlight:** "because a service that can anticipate the next move can prepare ahead of time instead of reacting after the fact."
> **Germano:** confuso. o que quis dizer aqui?
> **Action:** Confusing — clarify the intended meaning.

> **Response — 🟡 PARTIAL:** The proactive-vs-reactive framing is exactly what a wireless reviewer cares about, so the intent is right, but you're correct that 'prepare ahead of time instead of reacting after the fact' is abstract (prepare what? react to what?). The next sentence already grounds it, so I'll tighten the hook by naming what gets prepared rather than cut the idea.
> 
> **Edit:** Reword to: '...because anticipating the next move lets a mobility-aware service provision for demand in advance rather than react once it has already arrived.'

**16.**
> **Highlight:** "Predicting the exact next place, a single point of interest (POI), is hard,"
> **Germano:** predicting the next place exactly is hard.
> **Action:** Reword to "predicting the next place exactly is hard."

> **Response — 🟡 PARTIAL:** Happy to adopt your adverb order, but the load-bearing element is the inline POI definition on first use, which the non-ML audience needs and your literal rewrite drops. So I take the reorder and keep the appositive that defines POI.
> 
> **Edit:** Reword to: 'Predicting the next place exactly, a single point of interest (POI), is hard,' (keep the appositive).

**17.**
> **Highlight:** "and it is often more than a service needs."
> **Germano:** retirar. Senão colocar uma citação
> **Action:** Remove, or add a citation.

> **Response — ❌ REJECT:** This clause is load-bearing motivation — the reason we predict coarser targets, which you yourself probe in #48. It's a design rationale grounded by the paper's own use cases (content staging and capacity planning need the region, not the exact POI), so an external citation wouldn't fit and removal would weaken the central why-coarser argument. (Consistent with #57: cite concrete application examples, never the framing sentence.)
> 
> **Edit:** none (rejected)

**18.**
> **Highlight:** "coarser"
> **Germano:** o que é coarser questions?
> **Action:** Define "coarser questions".

> **Response — 🟡 PARTIAL:** The contrast with 'the exact next place' already implies coarser = lower-resolution, so it isn't broken — but a one-word plain gloss on first use is cheap, matches how the rest of the paper glosses terms for the non-ML reader, and doubles as the connotation guard for #1 (coarse != low-quality).
> 
> **Edit:** Reword to: 'Two coarser (less detailed) questions are easier to answer and usually enough:'.

**19.**
> **Highlight:** "part of the city"
> **Germano:** region
> **Action:** Use "region".

> **Response — ❌ REJECT:** 'part of the city' IS the plain-language gloss of 'region' here ('the next region, the part of the city a user is heading to'). Replacing the gloss with 'region' makes it circular ('the next region, the region') and deletes exactly the plain-words definition the clarity panel asked for. The jargon swap hurts the actual MobiWac reader.
> 
> **Edit:** none (rejected)

**20.**
> **Highlight:** "(we use the census tract, or the mahalle in Istanbul)."
> **Germano:** sem necessidade de colocar aqui essa parte. Você está definindo o problema somente
> **Action:** Unnecessary here — this is just problem definition; cut from the intro.

> **Response — 🟡 PARTIAL:** I won't fully cut this — it grounds 'region' for the networking reader and, via the mahalle, previews the Istanbul external-validity result. But you're right the full operational definition belongs in Problem and Tasks (confirmed present in §3), so I trim to keep just the grounding and the international preview.
> 
> **Edit:** Reword to: '(a neighborhood-scale unit, the census tract in the U.S. and the \emph{mahalle} in Istanbul)' and let the detailed operational definition live in Section~\ref{sec:problem}.

**21.**
> **Highlight:** "We do not predict the exact next place."
> **Germano:** retirar
> **Action:** Remove.

> **Response — 🟡 PARTIAL:** Don't drop it entirely — an explicit 'we do not predict the exact next place' is a deliberate scope boundary that heads off the 'why not predict the exact POI?' reaction (a BRACIS sore point), and it's also stated in §3. But as a bare standalone sentence it reads choppy, so I'll fold it into the flow.
> 
> **Edit:** Merge into the preceding clause: '...the \emph{mahalle} in Istanbul); we do not target the exact POI.'

**22.**
> **Highlight:** "The" (**The** two questions pair naturally…)
> **Germano:** These
> **Action:** Replace "The" with "These".

> **Response — ✅ ACCEPT:** Agreed, 'These' is more precise — it points back to the two questions just named. Harmless, correct micro-fix.
> 
> **Edit:** Reword to: 'These two questions pair naturally,'.

**23.**
> **Highlight:** "and they are more learnable than"
> **Germano:** não entendi
> **Action:** Unclear — rephrase.

> **Response — ✅ ACCEPT:** Fair, 'more learnable' is borderline jargon. The point is that one-of-seven categories or one-of-N regions is easier to predict accurately than one-in-thousands POIs, so I'll say that plainly, which also helps the networking reader.
> 
> **Edit:** Reword to: 'and they are easier to predict accurately than a single POI while still being useful.'

**24.**
> **Highlight:** "Our earlier work [3] found that joint training can help one task while quietly hurting the other, because the shared part is pulled toward a compromise that suits neither perfectly [4]."
> **Germano:** Aqui você afirma que nosso trabalho identificou isso, mas não foi. A gente só chegou nessa conclusão para essas duas tarefas, mas não foi a gente quem identificou isso. … pode parafrasear o pareto. Otimalidade de pareto que discute esse ponto.
> **Action:** Don't credit *us* with discovering the trade-off — we only observed it for these two tasks. Reframe via **Pareto optimality** (paraphrase that concept).

> **Response — ✅ ACCEPT:** Right to attribute the phenomenon — negative transfer is a known MTL property, not our discovery. But I'm dropping the word 'Pareto': Caruana 1997 is the canonical MTL-helps / hard-parameter-sharing reference, not a Pareto / multi-objective source, and there is no multi-objective citation in references.bib to carry it (an ML reviewer would catch 'Pareto [Caruana 1997]'). I'll credit the general trade-off to the MTL literature and reserve our earlier work for the specific observation.
> 
> **Edit:** Reword to: 'joint training can help one task while hurting the other, a well-known trade-off in multi-task learning where the shared parameters settle on a compromise that suits neither task perfectly~\cite{caruana1997multitask}; our earlier work~\cite{silva2025mtlnet} observed exactly this for next-category and next-region.' (drop 'Pareto').

**25.**
> **Highlight:** "useful question"
> **Germano:** repetindo muito *useful question*.
> **Action:** "useful question" repeated too often — vary the wording.

> **Response — ✅ ACCEPT:** Agreed, 'useful question' appears in para 1 and again in para 3. Vary the second one. (This and #8's restructuring together kill the abstract's repetition complaint.)
> 
> **Edit:** In para 3 reword 'So the useful question is not whether...' to 'So the question that matters is not whether...'.

**26.**
> **Highlight:** "multi-task learning (MTL)"
> **Germano:** apresenta o que é multi-task learning primeiro.
> **Action:** Introduce/define MTL **before** using it.

> **Response — ✅ ACCEPT:** Agreed, and defining MTL in plain words on first use serves the networking audience directly (it also feeds #64, where 'shared trunk' is used before MTL is explained). Introduce and gloss it at the top of the paragraph, then use the bare acronym afterward.
> 
> **Edit:** Change the paragraph opener to: 'Sharing one representation across the two tasks, an approach known as multi-task learning (MTL), is not free.' and drop the parenthetical from the later sentence so it reads '...is not whether MTL is good in general...'.

**27.**
> **Highlight:** "what to do about it."
> **Germano:** não entendi direito. Fazer o que aqui?
> **Action:** Unclear — clarify what "what to do about it" refers to.

> **Response — ✅ ACCEPT:** Agreed the phrase is vague, but I will NOT write 'without paying that cost' — that overclaims (our region result is non-inferiority, a bounded cost <=2 pp, not zero) and contradicts the line two sentences up (#5: 'what it costs to share one representation'). The honest concrete version names the goal: keep the gains while keeping the cost small.
> 
> **Edit:** Edit line 36: '...where sharing helps, where it costs, and what to do about it.' -> '...where sharing helps, where it costs, and how to share so the gains hold and the cost stays small.'

**28.**
> **Highlight:** "We make two changes"
> **Germano:** we make? *We propose two changes* é melhor.
> **Action:** Prefer "We propose two changes".

> **Response — ✅ ACCEPT:** Agreed, minor register fix. 'We introduce two changes' reads better than 'We make' alongside '...and measure each one carefully'. Adopting it.
> 
> **Edit:** Edit line 38: 'We make two changes and measure each one carefully.' -> 'We introduce two changes and measure each one carefully.'

**29.**
> **Highlight:** "instead of one fixed vector per place, where two visits to the same coffee shop look identical, each check-in gets its own vector that carries its context (the time, the nearby places, and the recent trail)."
> **Germano:** isso aqui me parece muito natural. Qual artigo que faz diferente disso? Não me parece uma contribuição nossa.
> **Action:** Feels too natural to claim as a contribution — which prior work does it differently? Justify or soften.

> **Response — 🟡 PARTIAL:** Fair challenge, and it matches our novelty discipline: per-visit context is not ours (CTLE already provides it). I'll add ONE light clause in the intro and keep the full scoping in Related Work (#46) and Method (#61/#62) so the same sentence doesn't appear three times. Audience caveat: CTLE is never expanded in the paper, so in the intro I'll present it as 'sequence-model embeddings such as CTLE [cite]' rather than a bare acronym.
> 
> **Edit:** After '...from the place down to the individual visit (Fig.~\ref{fig:dataflow}).' add a brief, differentiated clause: 'Per-visit context is not new on its own — sequence-model embeddings such as CTLE~\cite{lin2021ctle} provide it — but we obtain it from inside the hierarchical-graph-infomax representation and test that choice against CTLE directly.' Keep the fuller treatment in Related Work/Method; do not duplicate verbatim.

**30.**
> **Highlight:** "This extends hierarchical graph representations of places [1],"
> **Germano:** representar assim estende o trabalho de hierarchical graph representation?
> **Action:** Verify/justify the claim that this *extends* the hierarchical graph representation work.

> **Response — 🟡 PARTIAL:** The 'extends' claim is accurate and verified — Method states a four-level graph (check-in/place/region/city) and Fig 1 draws it, so 'a fourth, check-in level' is correct, not off-by-one. I'll make the hierarchy-extension explicit. But I'll keep the intro ACRONYM-FREE: 'HGI' isn't expanded until Related Work, so injecting bare 'HGI' into the intro would hit the networking reader with undefined jargon.
> 
> **Edit:** Edit line 42: '...from the place down to the individual visit' -> '...by adding a fourth, check-in level beneath the hierarchy of nested place graphs, from the place down to the individual visit'. Do NOT use the bare 'HGI' acronym in the intro (first expanded in Related Work).

**31.**
> **Highlight:** "semantic trunk"
> **Germano:** semantic trunk? use classical jargon
> **Action:** Replace "semantic trunk" with standard terminology (shared layers / backbone).

> **Response — 🟡 PARTIAL:** Half-agree: I'll drop the 'semantic trunk' compound that reads poetic, but I'm keeping the word 'trunk' — Method, results, and discussion all say 'shared trunk', so switching the intro to 'shared layers' would create an intro/body mismatch. I keep the semantic-vs-spatial contrast (it's load-bearing: what is shared vs. what stays private). Note 'backbone' would be worse for our audience — it collides with 'network backbone'.
> 
> **Edit:** Edit line 46: 'with a shared semantic trunk and a private spatial path for the region task' -> 'with a shared trunk that carries the semantic context and a private spatial path for the region task'. Keep 'trunk'; the hard-parameter-sharing gloss lands once at first use (#64).

**32.**
> **Highlight:** "international city"
> **Germano:** African city / western cities versus eastern cities
> **Action:** Consider framing external validity as Western vs. Eastern cities.

> **Response — ❌ REJECT:** Istanbul is one city, so it can only serve as an external-validity check (do findings survive outside the U.S.?), not a 'Western vs Eastern' cultural contrast. Claiming a cultural axis from n=1 overclaims and would invite a reviewer to demand a balanced multi-city cultural study we don't have ('African city' also doesn't apply — Istanbul isn't African). Keeping the honest 'international / non-U.S. city' framing.
> 
> **Edit:** none (rejected)

**33.**
> **Highlight:** "(a balanced score that counts every category equally)"
> **Germano:** retirar
> **Action:** Remove the parenthetical.

> **Response — ❌ REJECT:** Keeping the gloss — this is a deliberate clarity decision for our actual audience. MobiWac reviewers are wireless-networking people, and the clarity panel specifically asked that ML metrics be glossed in plain words. Removing '(a balanced score that counts every category equally)' serves an ML reader at the expense of the reader we're writing for.
> 
> **Edit:** none (rejected)

**34.**
> **Highlight:** "+28 to +40"
> **Germano:** huge. É esse número mesmo?
> **Action:** Double-check the +28 to +40 figure — it looks huge.

> **Response — ✅ ACCEPT:** Checked and the number holds: AL +29.3, AZ +27.6, FL +39.6, CA +38.0, TX +37.5 (Table II), and the abstract already hedges with 'about +28 to +40', so AZ's +27.6 rounds in honestly. It's large because a single fixed vector per place floors category macro-F1 in the high-20s/30s — co-located, multi-activity places collapse into one representation — and per-visit context restores the separation. I'll add one clause saying why and point it at the Fig 3 evidence so the claim is visibly backed.
> 
> **Edit:** Keep 'about $+28$ to $+40$'. Append to the first contribution bullet: '...this margin is large because a single fixed vector per place cannot separate the categories of places that host several kinds of activity, which per-visit context recovers (Section~\ref{sec:results}, Fig.~\ref{fig:embquality}).'

**35.**
> **Highlight:** "beats" (…it **beats** a dedicated category model at every state)
> **Germano:** outperforms
> **Action:** Use "outperforms".

> **Response — 🟡 PARTIAL:** Same call as #8: keeping 'beats' (the §5.3-defined verdict verb used across all sections), NOT swapping to 'outperforms' — that would fragment the vocabulary. Category genuinely beats at every state, so the verb is honest. The noun I'll normalize via the naming sweep: first use becomes 'dedicated single-task category model', then 'dedicated model' as shorthand (#9).
> 
> **Edit:** Line 63: keep 'beats'; normalize the noun on first use to 'dedicated single-task category model' (then 'dedicated model'). Do NOT change 'beats' to 'outperforms'.
---

## Contributions (p2)

**36.**
> **Highlight:** "An empirical account,"
> **Germano:** empirical account?
> **Action:** Rephrase "empirical account".

> **Response — ✅ ACCEPT:** Agree it reads a bit literary. 'An empirical analysis' is clearer and standard, and loses nothing. Swapping it.
> 
> **Edit:** Edit line 68: 'An empirical account, across five states...' -> 'An empirical analysis, across five states...'.

**37.**
> **Highlight:** "±2 pp)"
> **Germano:** pp?
> **Action:** Spell out "pp" (percentage points) on first use.

> **Response — ✅ ACCEPT:** Right, 'pp' is unglossed on first use. I'll expand it to 'percentage points (pp)' at intro line 72 and use 'pp' thereafter. Convention note: the softened abstract deliberately stays in plain 'points' (no 'pp'); the body uses 'percentage points (pp)' for the TOST margin and metric deltas — so I'm fixing the first-use expansion, not converting the abstract.
> 
> **Edit:** Edit intro line 72: '(TOST, $\pm2$~pp)' -> '(TOST, $\pm2$ percentage points (pp))'; use 'pp' after. Leave the abstract's 'within two points' as the plain-language form.
---

## Related Work (p2)

**38.**
> **Highlight:** "by raw identifiers."
> **Germano:** what is raw identifiers?
> **Action:** Define "raw identifiers".

> **Response — ✅ ACCEPT:** Fair point — for the networking audience 'raw identifiers' is undefined jargon for the bare place ID, and a one-clause gloss helps both audiences without bloating the sentence.
> 
> **Edit:** Edit to: '...so that a model can reason about places by their geometry rather than by raw identifiers, the arbitrary ID numbers that carry no meaning on their own.'

**39.**
> **Highlight:** "Deep Graph Infomax (DGI) [5] learns such vectors by training them to tell a real network of places, linked by similarity, time, and distance, from a shuffled copy of that network."
> **Germano:** DGI is not a work for places. It is a technique for self-supervised learning on graphs
> **Action:** Correct the description — DGI is a self-supervised graph learning technique, not a place-specific method.

> **Response — ✅ ACCEPT:** Correct and important. DGI is a task-agnostic self-supervised method for node representations on graphs; the current sentence reads as if DGI were place-specific, which a reviewer could rightly flag. Make explicit that DGI is the general technique and the place network is our application of it.
> 
> **Edit:** Edit to: 'Deep Graph Infomax (DGI)~\cite{velickovic2019dgi}, a general self-supervised method for learning node representations on graphs, is applied here to a network of places: it learns such vectors by training them to tell the real place network, linked by similarity, time, and distance, from a shuffled copy.'

**40.**
> **Highlight:** "Both methods emit one vector per place,"
> **Germano:** what is to emit a vector? Do you mean to yield?
> **Action:** Replace "emit" a vector (→ produce/yield).

> **Response — ✅ ACCEPT:** Agreed, 'emit' is mildly odd; 'produce' is plainer and standard. The three occurrences are 02_related L18 and L28 and the Method novelty sentence (04_method L35, 'emitting'). I'll change all three — including the Method one, which #61's merge must also carry as 'producing', so we don't end up with 'produced' and 'emitting' for the same action.
> 
> **Edit:** Replace 'emit'/'emits'/'emitting' with 'produce'/'produces'/'producing' at 02_related L18, 02_related L28, and 04_method L35. Coordinate with #61 so the merged novelty sentence reads 'producing a vector per check-in'.

**41.**
> **Highlight:** "but it gets there a different way."
> **Germano:** but in a different way
> **Action:** Reword to "but in a different way".

> **Response — 🟡 PARTIAL:** Minor register tweak, not a real defect. 'Gets there a different way' is clear plain English the MobiWac reader parses fine, but I'll tighten it slightly since your version reads marginally cleaner. I would not call the original wrong.
> 
> **Edit:** Optional tightening: '...also works at the check-in level, but it gets there differently.'

**42.**
> **Highlight:** "now extended down to a fourth, check-in level (Fig. 1),"
> **Germano:** não entendi. Não sei se é a melhor posição aqui para explicar isso
> **Action:** Unclear and possibly the wrong place to explain this — consider moving/rewriting.

> **Response — 🟡 PARTIAL:** The clarity complaint is fair, the relocation suggestion is not. The phrase is dense because it assumes the reader has counted place/region/city as three levels — and the four-level structure is verified (Method L20, Fig 1) — so I'll spell it out, but this embeddings subsection is the right place to explain the HGI lineage (HGI is already expanded here), so I keep it here.
> 
> **Edit:** Reword in place to: '...trains it with the same infomax objective, now extended one level deeper, from individual places down to individual check-ins, a fourth level beneath place, region, and city (Fig.~\ref{fig:dataflow}), rather than switching to a sequence model.'

**43.**
> **Highlight:** "rather than switching to a sequence model."
> **Germano:** como assim sequence model?
> **Action:** Clarify what "sequence model" means here.

> **Response — 🟡 PARTIAL:** The term itself is standard, but you're right the contrast floats: the 'sequence model' we contrast against is CTLE, named two sentences earlier. Pinning it to CTLE removes the ambiguity for both audiences without adding a definition.
> 
> **Edit:** Edit to: '...rather than switching to a sequence model, as CTLE does.'

**44.**
> **Highlight:** "Put another way,"
> **Germano:** Put another way?????
> **Action:** Drop "Put another way".

> **Response — 🟡 PARTIAL:** Low stakes. 'Put another way' is a harmless reader aid signaling a plain-language restatement, which the non-expert audience benefits from, so I won't cut the restatement. But the connective can be swapped to something less conversational since it bothers you.
> 
> **Edit:** Replace 'Put another way,' with 'In short,' (or drop the connective and keep the sentence).

**45.**
> **Highlight:** "earlier representations"
> **Germano:** which earlier representations?
> **Action:** Specify *which* earlier representations.

> **Response — ✅ ACCEPT:** Specifying the antecedent answers your question and fixes a subtle inaccuracy: DGI and HGI changed which features are encoded, whereas CTLE already changed how often a vector is produced (per visit), so lumping all three is imprecise. Naming them sharpens the contrast (and this occurrence is one of the emit->produce fixes, #40).
> 
> **Edit:** Edit to: '...earlier place embeddings (DGI, HGI) changed \emph{which} features a place vector encodes; CTLE and we change \emph{how often} a vector is produced, once per place or once per visit.'

**46.**
> **Highlight:** "representations changed which features a place vector encodes; we change how often the representation emits a vector, once per place or once per visit. The novelty is this specific combination, per-visit context inside a hierarchical graph-infomax representation, and we compare against CTLE directly to show that the combination, not per-visit context on its own, is the source of the gain."
> **Germano:** redigir outro parágrafo
> **Action:** Rewrite this paragraph.

> **Response — 🟡 PARTIAL:** A wholesale rewrite isn't warranted — this paragraph does exactly what the BRACIS post-mortem requires: it scopes novelty to the specific combination and commits to the CTLE comparison, so weakening it re-triggers the novelty attack. I'll apply only light copyedits (emit->produce #40, antecedent fix #45) and keep the combination-novelty framing, differentiated from the intro pointer (#29) and the fuller Method statement (#61/#62) so it doesn't read verbatim three times.
> 
> **Edit:** Keep the combination-novelty framing; apply only the copyedits from #40/#45. Do not rewrite the paragraph wholesale; ensure it is not a verbatim duplicate of the intro (#29) or Method (#61) versions.

**47.**
> **Highlight:** "We deliberately predict two coarser properties of the next visit instead: its category, what kind of place it is, and its region, where it falls, at the granularity of a census tract."
> **Germano:** não tem necessidade de falar o que a gente faz aqui. é só falar o que está sendo feito por outros trabalhos e apresentar o gap
> **Action:** In Related Work describe only *others'* work and present the gap — don't describe our method here.

> **Response — 🟡 PARTIAL:** Half right. The method specifics (census-tract granularity) belong in Problem/Tasks (confirmed present in §3), so I'll defer them. But the contrast sentence itself is the gap statement — 'others predict the exact place; we deliberately do not' is exactly how Related Work presents a gap — so I won't strip it.
> 
> **Edit:** Trim to a gap statement and defer the granularity: 'In contrast, we target two coarser properties of the next visit, its category and its region, rather than the exact place.' (Census-tract definition stays in §3.)

**48.**
> **Highlight:** "Coarser targets are easier to get right and still useful to act on, and they line up with two kinds of preparation a mobility-aware service can make."
> **Germano:** então a gente simplifica o problema? isso não é um drawback?
> **Action:** Address the implied criticism — does coarsening simplify the problem (a drawback)? Frame as a deliberate, justified choice.

> **Response — ✅ ACCEPT:** This is the strongest comment in your batch, and a reviewer will ask the same thing: did we just make the problem easier? We rebut head-on — coarser is not trivial here (region still spans hundreds to ~8,500 classes), and the choice is operationally motivated (a service acts on category and region, not on an exact POI), not score inflation.
> 
> **Edit:** Add justification: 'Coarser does not mean trivial: the region target alone spans hundreds to several thousand classes. We choose these targets because they are what a mobility-aware service can actually act on, not to make the problem easier, and they line up with two kinds of preparation such a service can make.'

**49.**
> **Highlight:** "The category task appears in our own earlier line of"
> **Germano:** retirar / se aparece em outro trabalho por que não colocar aqui?
> **Action:** Remove — or if it appears in other work, cite it here.

> **Response — 🟡 PARTIAL:** Your either/or is already half-satisfied — the sentence carries the citation (\cite{silva2025mtlnet}, our earlier MTLnet work), so 'remove' would erase honest lineage and the 'cite it' branch is met. I'll keep the self-citation but tighten the phrasing so it doesn't read as a stranded fragment.
> 
> **Edit:** Keep the citation; integrate: 'The category task itself follows our earlier line of work~\cite{silva2025mtlnet}.'

**50.**
> **Highlight:** "We also note that next region over census tracts has little precedent:"
> **Germano:** predict over census tracts is a novelty?
> **Action:** Is census-tract prediction really novel? Justify the claim or cite prior work.

> **Response — 🟡 PARTIAL:** Fair challenge, accepting the scoping per our novelty discipline. The honest claim is not that the census-tract unit is novel — it's that treating fine-grained region as a co-equal END target (rather than an auxiliary coarse cell, as HMT-GRN does) is underexplored. I'll reword so we don't overclaim the unit, but won't drop the scoping the comparison rests on.
> 
> **Edit:** Reframe to: 'Fine-grained region as an end target, rather than an auxiliary coarse cell, has little precedent: HMT-GRN predicts a coarse geohash cell only as an aid, so we define our region unit carefully when we state the tasks.'

**51.**
> **Highlight:** "uses category in the same spirit,"
> **Germano:** in the same spirit?
> **Action:** Reword the vague "in the same spirit".

> **Response — ✅ ACCEPT:** Agreed, 'in the same spirit' is vague, and the precise meaning is already in the next clause ('to filter candidate places'), so name it: category is used instrumentally, as in the CSLSL cascade. Clean fix.
> 
> **Edit:** Edit to: 'CatDM~\cite{yu2020catdm} likewise uses category instrumentally, to filter candidate places.'

**52.**
> **Highlight:** "balancer" (yet another gradient **balancer**…)
> **Germano:** regularizer
> **Action:** Terminology — consider "regularizer".

> **Response — ❌ REJECT:** Technically wrong, Germano. PCGrad, GradNorm, and Nash-MTL are gradient-balancing MTL optimizers — they manipulate per-task gradients/loss weights, not regularizers (which add a penalty term). Renaming them 'regularizers' is a substantive error a reviewer would catch. The first-use term 'gradient balancer' is already glossed at 02_related L70, so the clarity need is met; at most an optional tighten to 'gradient-balancing method', never 'regularizer'.
> 
> **Edit:** None (optional: 'gradient balancer' -> 'gradient-balancing method' for precision; do NOT use 'regularizer').

**53.**
> **Highlight:** "balancers" (such **balancers** rarely beat…)
> **Germano:** regularizers
> **Action:** Terminology — "regularizers".

> **Response — ❌ REJECT:** Same issue as #52: 'such balancers rarely beat a well-tuned fixed weighting' is correct — these methods balance gradients, they don't regularize, and the line is anaphoric to 'gradient balancer' just above (L70). 'Regularizers' would mislabel the entire MTL-optimizer literature being cited (xin2022domtl, kurin2022defense). Keep 'balancers'.
> 
> **Edit:** None (optional: 'such balancers' -> 'such gradient-balancing methods'; do NOT use 'regularizers').
---

## Problem & Tasks (p2–3)

**54.**
> **Highlight:** "We work with check-in sequences."
> **Germano:** deveria vir na introdução
> **Action:** This belongs in the Introduction.

> **Response — ❌ REJECT:** Keeping the line — a Problem and Tasks section conventionally re-anchors its data substrate, and 'We work with check-in sequences' sets up the very next sentence (ordering check-ins, forming nine-visit windows); cut it and the windowing starts abruptly. I take your redundancy point, though: if it reads as a restatement of the intro, the low-churn option is to fold it into the windowing sentence rather than stand alone. Leaving placement as-is but flagging that merge as available.
> 
> **Edit:** None (optional: merge into the next sentence — 'For each user we order their check-in sequence in time and form short windows of nine consecutive visits.').

**55.**
> **Highlight:** "The first is its category, the kind of place, one of seven fixed labels: Community, Entertainment, Food, Nightlife, Outdoors, Shopping, and Travel."
> **Germano:** o outro dataset tem essas mesmas categorias?
> **Action:** Confirm the other dataset (Istanbul) uses these same categories.

> **Response — ✅ ACCEPT:** Good question, and the answer is yes: Istanbul (Massive-STEPS) is collapsed to the same seven root categories, so the category task is identical across all datasets. Right now the reader has to assume that, so I'll state it explicitly.
> 
> **Edit:** After '...Shopping, and Travel.' add: 'Istanbul uses these same seven categories via the Massive-STEPS seven-root mapping, so the category label set is identical across all datasets.'

**56.**
> **Highlight:** "Region is a large one, ranging from about five hundred classes (Istanbul) to about eight thousand five hundred (California),"
> **Germano:** classes for region???
> **Action:** "classes" for regions is confusing — clarify the region-as-classification framing.

> **Response — ✅ ACCEPT:** Fair — 'classes' reads oddly attached to region until the reader registers that we frame next-region as classification, where each candidate census tract is one class. §3 half-says this ('a fine-grained choice over many candidate regions') but 'classes' lands first, so I'll make the region-as-classification framing explicit at first mention. This also helps the non-ML reader.
> 
> **Edit:** Reword to: 'Region is a large set: each candidate census tract is one class, ranging from about five hundred (Istanbul) to about eight thousand five hundred (California),' then keep the existing 'fine-grained choice over many candidate regions' clause.

**57.**
> **Highlight:** "The motivation is practical, and the two predictions line up with two kinds of preparation a mobility-aware service makes."
> **Germano:** afirmação bem forte sem nenhuma citação
> **Action:** Strong claim with no citation — add a reference.

> **Response — 🟡 PARTIAL:** This is motivational framing, and §3 already self-disclaims it ('the application is the reason the predictions matter, not a result we claim'), so it doesn't strictly need a citation. Per the rule we apply at #17, we cite concrete application examples, never the framing sentence — but references.bib has NO mobility-aware content-staging/capacity-planning entry, and I will not fabricate one. So: leave the sentence as motivation now; attach a single verified reference to the concrete examples only IF a real one is sourced and added with a %verified provenance note.
> 
> **Edit:** No cite added now (no verified mobility-aware-service reference exists in references.bib). If one is sourced, attach it to the 'content staging / capacity planning' examples only (not the framing sentence), with a provenance note; otherwise leave as motivation, consistent with #17.

**58.**
> **Highlight:** "in hand" (With both **in hand** a service can act…)
> **Germano:** , (insert a comma)
> **Action:** Punctuation fix — add a comma.

> **Response — ✅ ACCEPT:** Correct — 'With both in hand' is an introductory phrase and needs a trailing comma. Trivial, uncontroversial fix.
> 
> **Edit:** Change to 'With both in hand, a service can act ahead of regional demand...'

**59.**
> **Highlight:** "a census tract is a neighborhood, not a radio cell,"
> **Germano:** radio cellll???
> **Action:** "radio cell" feels out of place — reconsider.

> **Response — ❌ REJECT:** Keeping 'radio cell'. This paper targets a wireless/MobiWac audience, and the radio cell is exactly the spatial unit those reviewers think in — explicitly saying a census tract is NOT a radio cell is the deliberate scoping that stops a networking reviewer from expecting handover or cell-association results. It reads as stray jargon to an ML eye, but for the actual reader it's audience-serving and load-bearing.
> 
> **Edit:** none (rejected)

**60.**
> **Highlight:** "and even the ten most likely tracts are far too coarse to drive cell association or handover, so we keep the motivation at the level of regional demand and load anticipation, content staging, and capacity planning."
> **Germano:** deslocado do texto
> **Action:** Out of place — relocate or trim.

> **Response — 🟡 PARTIAL:** The scoping content stays — it's load-bearing per the section plan, right-sizing the motivation to tract granularity (demand/load anticipation, content staging, capacity planning) and ruling out handover/cell-association, which preempts the wireless reviewer's overclaim question. It only feels 'out of place' through an ML lens that doesn't value the cell-scoping. I'll grant the flow point: it's a single run-on, so I'll split it while preserving every clause.
> 
> **Edit:** Split for flow, keep all content: 'These examples are sized to what we actually measure. A census tract is a neighborhood, not a radio cell; even the ten most likely tracts are far too coarse to drive cell association or handover. We therefore keep the motivation at the regional level: demand and load anticipation, content staging, and capacity planning.'
---

## Method (p3–4)

**61.**
> **Highlight:** "The pieces here are individually standard. Per-visit context, hierarchical place graphs, and the infomax objective all exist in prior work."
> **Germano:** sem necessidade
> **Action:** Unnecessary — cut.

> **Response — 🟡 PARTIAL:** I won't cut the concession — 'the pieces are individually standard' is exactly the BRACIS-defense that disarms the 'not novel' attack, and deleting it makes the combination claim float. But two standalone meta-sentences are heavy, so I'll compress into one. And I'll apply emit->produce here (#40): the merged sentence must say 'producing a vector per check-in', not 'emitting', or Related Work and Method disagree on the same action.
> 
> **Edit:** Merge 04_method L33-39 into: 'Each ingredient is individually standard -- per-visit context, hierarchical place graphs, and the infomax objective all appear in prior work -- but the novelty is their combination: producing a vector per check-in from inside a hierarchical-graph-infomax representation, which earlier contextual embeddings built from sequence models (CTLE) do not provide. We show later that this combination, not extra supervision, is what makes the category task far easier to learn.'

**62.**
> **Highlight:** "Our one honest novelty is the combination: emitting a vector per check-in from inside a hierarchical-graph-infomax representation, a combination that earlier contextual embeddings built from sequence models do not provide, and we show later that this combination, not extra supervision, is what makes the category task far easier to learn."
> **Germano:** não tem necessidade. Pensa que você está estendendo um artigo. Não precisa de *honest novelty*
> **Action:** You're extending a paper — drop the "honest novelty" framing.

> **Response — 🟡 PARTIAL:** Agreed on tone, not on substance: 'our one honest novelty' reads over-apologetic, so I'll cut it to a plain 'the novelty is their combination'. But I'm keeping the scoping itself (combination, not bare per-visit context; the CTLE/sequence-model contrast; 'not extra supervision') because that honest scope is what protected us at BRACIS — dropping it re-opens the wound. 'Emitting' becomes 'producing' (#40).
> 
> **Edit:** Within the #61 merge, 'Our one honest novelty is the combination: emitting...' -> 'the novelty is their combination: producing...'; retain the CTLE/sequence-model contrast and the 'not extra supervision' clause verbatim.

**63.**
> **Highlight:** "private per-task encoders"
> **Germano:** o que é private per task encoder?
> **Action:** Define "private per-task encoder".

> **Response — ✅ ACCEPT:** Fair — for the MobiWac/mobile-systems audience 'private' can read as a private network, so a one-clause gloss on first use helps both audiences. Define it inline rather than assume the reader knows the MTL idiom.
> 
> **Edit:** Line 46-47: 'pass through private per-task encoders into a shared trunk' -> 'pass through private per-task encoders (a small input network dedicated to each task, with no weights shared between them) into a shared trunk'.

**64.**
> **Highlight:** "shared trunk,"
> **Germano:** shared trunk? você não explicou nada de MTL
> **Action:** "shared trunk" used before any MTL explanation — introduce MTL / hard-parameter-sharing first.

> **Response — ✅ ACCEPT:** Correct — 'shared trunk' is used before hard parameter sharing is introduced (the citation lands two paragraphs later). Glossing it at first use names the standard MTL setup for the networking reader and removes the forward reference; this pairs with defining MTL in the intro (#26). Then trim the later repeat so the definition lands exactly once.
> 
> **Edit:** Line 47: 'into a shared trunk, where the two streams exchange information' -> 'into a shared trunk -- a stack of layers used by both tasks, the standard multi-task setup known as hard parameter sharing~\cite{caruana1997multitask} -- where the two streams exchange information'; then at line 65 trim 'This is ordinary hard parameter sharing~\cite{caruana1997multitask}' to 'kept deliberately plain...' so the definition is not repeated.

**65.**
> **Highlight:** "We want to be precise about what that private path is, because it is easy to misread."
> **Germano:** Sem necessidade dessa frase
> **Action:** Unnecessary sentence — cut.

> **Response — ✅ ACCEPT:** Agreed, this is throat-clearing — the actual content is in the next sentence; the lead-in announces an intention without adding information. Cutting it and going straight to the definition is tighter and loses nothing; the single-model clarification it precedes stays fully intact.
> 
> **Edit:** Delete the sentence and start the paragraph at the content: 'The private spatial path is a task-specific branch inside the one model, not a second model, not a second representation, and not a route that swaps models per task.'

**66.**
> **Highlight:** "swaps models per task. The whole system is a single model: one set of shared parameters, one forward pass, two predictions. This is the deployment property we care about, since a service that wants both answers runs one model once, not two models in sequence."
> **Germano:** não entendi
> **Action:** Unclear — rephrase.

> **Response — 🟡 PARTIAL:** Keeping this — the single-model property (one set of shared parameters, one forward pass, two predictions) is the paper's primary deployment thesis and can't be softened. Your 'nao entendi' is likely 'two models in sequence', which is imprecise; the real contrast is two separate single-task models. I'll rephrase for clarity while preserving the claim.
> 
> **Edit:** Replace 04_method L55-58 with: 'The whole system is a single model: one set of shared parameters and one forward pass produce both predictions. This is the deployment property we care about -- a service that wants both answers runs a single model once, instead of two separate dedicated single-task models.'

**67.**
> **Highlight:** "The cost of serving both tasks this way is small. One joint model has about five percent more parameters and compute than a single dedicated model, far less than the two separate models the usual setup would deploy, so two answers come at roughly the price of one."
> **Germano:** tentar reescrever essa subseção. Me pareceu bem deslocado
> **Action:** Rewrite this subsection — reads as misplaced.

> **Response — 🟡 PARTIAL:** I won't rewrite the subsection, but the +5% figure is a genuine selling point for a MobiWac/edge audience and backs the single-model thesis, so it stays. The 'misplaced' feel is real — it reads orphaned after the loss paragraph — so I'll move it to sit right after the single-model paragraph (#66), and normalize 'single dedicated model' to 'dedicated single-task model' (#9).
> 
> **Edit:** Relocate the cost paragraph to immediately follow the single-model paragraph (#66) and edit: 'One joint model has about five percent more parameters and compute than a single dedicated model' -> '...than a dedicated single-task model'.
---

## Results (p7)

**68.**
> **Highlight:** "Check-in level | Place level" (Table II — Check2HGI vs HGI columns)
> **Germano:** é uma comapração quase injusta. Porque a ideia aqui é bem diferente não?! São dois targets completamente diferentes. São duas modelagens diferentes
> **Action:** Comparison may be almost unfair — the two are very different ideas/targets/models. Justify comparability or reframe.

> **Response — 🟡 PARTIAL:** Rejecting the substance — this is a controlled comparison by construction, not unfair: both columns predict the SAME next-category macro-F1, through the SAME matched single-task head, on the SAME folds and windowing; only the input representation changes (vector-per-visit vs vector-per-place HGI). The targets and model are not different — that's what makes the margin attributable to the representation. But your misreading signals a reviewer could misread too, so I'll add one explicit comparability sentence to §6.1.
> 
> **Edit:** Add to §6.1 after the comparison sentence: 'This is a controlled, like-for-like comparison: the target (next-category macro-F1), the single-task head, and the cross-validation folds are identical across the two columns; only the input representation changes (a vector per visit versus a vector per place), so the margin isolates the representation rather than a difference in task or model.'

**69.**
> **Highlight:** "Category separability of the two representations. The check-in-level representation cleanly separates the seven categories where the place embedding does not, which is exactly why it helps the category task and is neutral on region." (Fig. 3 caption)
> **Germano:** how many clusters? kNN accuracy em category separability?
> **Action:** State the number of clusters; clarify why kNN accuracy measures "category separability".

> **Response — ✅ ACCEPT:** Fair question, and worth fixing. There are seven groups, not discovered clusters — points are colored by the seven ground-truth category labels, not a k-means k. Separability is quantified two ways in §6.1: silhouette-by-category (0.56 vs 0.00) and nearest-neighbor category purity (0.98 vs 0.78), the latter being the kNN-style measure you're asking about. The caption asserts separability without the numbers; I'll lift them in so the figure is self-contained.
> 
> **Edit:** Replace the Fig. 3 caption with: 'Category separability of the two representations (per-visit and per-place vectors projected to two dimensions, colored by the seven categories). The check-in-level representation cleanly separates the categories where the place embedding does not: silhouette-by-category $\approx 0.56$ versus $\approx 0.00$, and nearest-neighbor category purity (the share of each vector's nearest neighbors that share its category, a kNN-style separability measure) $\approx 0.98$ versus $\approx 0.78$. This is exactly why it helps the category task and is neutral on region.' (Match the projection-method wording to whatever fig3_embquality.py renders.)
---

## References (p9)

**70.**
> **Highlight:** "Joint Conference on Artificial Intelligence (IJCAI), pp. 3081–3089, 2025." (Ref [26])
> **Germano:** Estou com medo das referências estarem ultrapassadas
> **Action:** Worried the references may be outdated — review for recency.

> **Response — ✅ ACCEPT:** Good instinct to spot-check, and the recency pass is done — but the references aren't outdated. I counted the actual citations: 26 unique \cite keys across all sections/tables/figures (references.bib holds 33 entries; 7 are uncited), so the paper cites 26 references, all web-verified with %verified provenance notes. The span you flagged is in fact the NEWEST entry — ReHDM (li2025rehdm, IJCAI 2025), our next-region reference — alongside Massive-STEPS (2025), MCMG (TOIS 2024), and CSLSL (2024). No edit needed; the 26-vs-27 ambiguity resolves to 26 actually-cited.
> 
> **Edit:** No paper edit. Recency review complete: 26 cited references (grep-verified; 33 bib entries, 7 uncited), all web-verified and current; newest = ReHDM (IJCAI 2025) = the highlighted entry. Reference list stands.
---

*All 70 highlight comments, in reading order. Highlighted spans recovered via word-level
bounding boxes (PyMuPDF) so each maps to exactly the text under the highlight. Verbatim
Portuguese preserved with English action glosses.*
