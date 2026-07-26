# Full-arc re-review — five personas on the complete approved narrative

> **What this is.** The author-requested final pass before drafting: five reviewers (cold reader,
> claim honesty, banca simulator, adversarial advisor, excellence assessor) ran fresh-eyes on the
> COMPLETE package — the spine, all storyline files, both sign-off rounds with the author's inline
> decisions, and the drafted 2.1/2.5 — asking "are we overlooking something, forgetting something;
> is this now a good narrative?"
>
> **The one-line verdict: 5/5 `ready_with_fixes`.** Every persona independently answers "yes, this
> is now a good narrative" — and every persona independently found the same category of defect:
> not the story, but its **transmission**. The approvals lived in the storyline sign-off documents
> while the spine (NORTH_STAR/GLOSSARY) that drafting agents expand still carried pre-correction
> sentences. Most of those are now FIXED (see §2); the remaining items are the author's.

---

## 1. What the five lenses agreed on

**The narrative is good.** From the verdicts: the cold reader — "a genuinely good story that would
carry a cold reader once the frame narrates it"; claim honesty — "an honest one that got MORE
honest under pressure"; the banca — the correction-trail structure its question bank rewards, and
conditional on the fixes "this defends as aprovado com correções menores"; the adversarial
advisor — "a rare one … the arc's most dangerous confound converted into its most satisfying
beat"; excellence — "close to an outstanding one."

**The convergent findings (each found independently by 3–5 personas):**
1. **[BLOCKER] The UW-3 protocol retraction never propagated** to NORTH_STAR:139/:262 and
   GLOSSARY:64 — a Ch.4 preface drafted from the spine would assert an unverified protocol claim.
2. **[MAJOR] The spine predated the sign-offs** — NORTH_STAR §6 carried no task-pair
   acknowledgment, no three-legged defense, no N2/N3 framing, no §3.4 concession in the
   limitations list.
3. **[MAJOR] The author's N3 question was unanswered** — the package treated N3 as closed while
   its "Author:" line contains a direct question and an experiment offer (§3 below).
4. **[MAJOR] The PANORAMA logline contradicted the mechanism** — "via atenção cruzada em vez de um
   tronco comum" denied the shared trunk the freeze control credits.
5. **[MAJOR] The intro's "lower cost" promise (NORTH_STAR §6.1 beat 2)** was never redefined to the
   operational form the arc actually delivers (honesty flag F3 / thread T4).

## 2. Fixed in this pass (governance sync, applied to the repo)

| Fix | Where |
|---|---|
| CoUrb protocol claim downgraded to [VERIFY] with verification path (codebase / judge_feedback) | `NORTH_STAR.md` §4 honesty items + §6 Ch.4 preface beat |
| GLOSSARY protocol note downgraded to [VERIFY] | `GLOSSARY.md` user-disjoint-split row |
| Item 6 one-sentence boundary floor written into the Ch.4 preface beat ("brevity yes, omission no") | `NORTH_STAR.md` §6 Ch.4 preface |
| Intro beat 2 rewritten: operational simplicity promised, F3 guard added (never "lower cost") | `NORTH_STAR.md` §6.1 beat 2 |
| Arc beat 4 extended with the five signed-off additions (task-pair acknowledgment; three-legged defense with leg-2 fallback; corrected corollary; N2 caution-form-only with F4 guard; mechanism-as-hypothesis) | `NORTH_STAR.md` §6.1 beat 4 |
| Ch.6 limitations beat gains the §3.4 confound concession, tied 1:1 to the fixed-pair-ablation future-work item; N3 mechanism beats appended with full scope + licensed vocabulary ("gate" translated) | `NORTH_STAR.md` §6.4 beats |
| PANORAMA logline corrected (shares THROUGH a cross-attention trunk; correction note kept visible) | `storyline/PANORAMA_ptBR.md` §1 |
| PANORAMA "previu" softened to "levantou a hipótese e os resultados a sustentaram" (licensed strength) | `storyline/PANORAMA_ptBR.md` §2 |
| Stale OpenAlex status updated (connector needs in-app authorization) | `storyline/PANORAMA_ptBR.md` §6 |

**N2 drafting rule (recorded here, binding):** draft ONLY from N2's caution paragraph — CBIC
hypothesized three causes and its future work proposed the *architecture* door (soft
sharing/Cross-Stitch/MoE, optimizers, task-relatedness); it did NOT propose a representation
program. The claim-honesty auditor verified this against `conclusion.tex` this session. Never
write "CBIC's future work called for better representations."

**Item 2 rendering rule (banca finding):** the tasks are "coarser-grained than next place," never
"simpler"/"easier" — the author's "mais simples" note must not be rendered literally, or it
collides with the approved "harder, not easier" cardinality fact.

## 3. The N3 answer the author asked for (my opinion, as requested)

You wrote under N3: the current evidence for *where the MTL improvement comes from* feels not very
convincing; should we run more experiments (locally or on nespedgpu)?

**My honest assessment.** The evidence you have is better than you are giving it credit for, but it
has one real hole, and the banca simulator posed it verbatim as the one question the package cannot
yet answer: *"Se o ganho vem do tronco e não da interação entre as tarefas, um modelo dedicado de
tarefa única com a mesma capacidade não recuperaria o mesmo ganho? O senhor rodou esse baseline?"*
The freeze control proves the gain is a trunk effect (not task-teaching); what it does not prove is
that a **capacity-matched dedicated model** would not do the same. Today the defense is an honest
disclosure (params as cost) plus a concession — defensible at a master's, but it is the weakest
point under arguição.

**The options, ranked:**
1. **Run the capacity-matched dedicated baseline (recommended if time allows).** One experiment:
   the dedicated category model (and optionally region) scaled to ~the joint model's parameter
   count, same protocol (user-disjoint 5-fold, seeds, same tuning budget). It answers the exact
   banca question with a number instead of a concession. Feasibility: the repo has the full
   training pipeline (`src/`, `scripts/train.py`, closing-data protocol), and nespedgpu (A40 46GB,
   128GB RAM, 32 cores) is connected — this is well within reach. **Licensing rule (mandatory,
   from the adversarial advisor):** the new numbers do NOT enter Ch.5 (the MobiWac version of
   record is under review); they live in the frame (a Ch.5-adjacent discussion or an appendix) as
   post-submission analysis, with their own fact gate, clearly dated. Either outcome strengthens
   you: if capacity-matched dedicated ≈ dedicated, the joint win is not a capacity artifact and
   the trunk story is confirmed; if it recovers part of the gain, you report it honestly and the
   two-factor story gains a third, quantified nuance — still your finding, not a reviewer's.
2. **Write the concession (the floor, already signed off).** The §3.4 concession is now in the
   spine's limitations beat, tied to the fixed-pair ablation as future work. This is the
   defensible minimum if no experiment runs.
3. **Do not** run broad new mechanism studies (symmetric freezes, representation probes, etc.)
   before the defense — scope creep against an August deadline; the capacity-matched baseline is
   the one experiment that answers the one live question.

**Decision needed from you:** option 1 (schedule the run; I prepare the scripts and dispatch to
nespedgpu) or option 2 (concession only). The Ch.6 §6.4 mechanism paragraph is drafted differently
under each, so this decision gates Ch.5/Ch.6 drafting — Ch.1–Ch.4 work is unaffected.

## 4. Still open (the author's list)

1. **The N3 decision** (§3 above) — gates Ch.6.
2. **CoUrb split verification** — one look at the CoUrb codebase (github.com/TarikSalles/
   Spatial_Embeddings) or `slides/judge_feedback.md` settles UW-3; until then no protocol
   difference is drafted anywhere.
3. **OpenAlex authorization** — the reconnected literature server needs its in-app authorization
   (Configurações → Conectores); then the beyond-mobility + N1-leg-2 sweep runs (open-and-verify,
   with the pre-approved fallback if no anchor supports the comparative form).
4. **The Ch.2 page fixes** (already catalogued in `10_specialist_check/`, still unapplied): the 2.1
   scope sentence (cold-reader escalation: it reads as false at Ch.3 — edit, not only add), the
   93% scoping, the under-review marker in 2.5, the map-partition sentence, the name-mapping seam.
   These ride the first drafting wave as its first work order.
5. **Ch.1 beat budget** (cold-reader MAJOR): before drafting Ch.1, one page assigning each approved
   move exactly one home (Ch.1 vs Ch.4 preface vs Ch.5 recap vs Ch.6) so the Introduction narrates
   rather than litigates. The "and/or" placements in the aval items are resolved there.
6. **Title shortlist** (cold-reader MINOR): two of the three NORTH_STAR §5 candidates fail the F1
   two-factor test ("Representation-Driven…", "Check-in-Level Representations for…"); the first
   candidate is the only survivor as written. Decide before front matter.
7. **Products/artifacts surface** (excellence MINOR): where the committee finds the products list
   (2 DOIs + 1 under review + code + protocol) — a short appendix or per-chapter footnotes.

## 5. Where the five full verdicts live

The complete structured verdicts (each persona's overlooked-items list, per-move soundness check on
all 11+3 approved items, narrative-quality paragraph, and top priority) are archived verbatim in
this folder as `five_verdicts.txt`.
