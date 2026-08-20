# CLAUDE.md — Dissertação de Mestrado (UFV / PPGCC) — working folder

> **What this folder is.** The working folder for Vitor H. O. Silva's **master's dissertation** at
> UFV / PPGCC (Ciência da Computação, Campus Florestal / NESPeD-LAB), advisor Fabrício A. Silva.
> Format: **coletânea de artigos** (UFV Normas §2.3(iii)/§2.6), **English frame**.
> **Status: DELIVERED to the banca. Defense 2026-08-28.** The text is frozen; changes reach it only
> as erratas, at the final deposit.
>
> Title, exactly as the folha de rosto prints it:
> *Multitask Learning for Point-of-Interest Classification and Prediction Tasks: The Role of the
> Check-in-Level Representation.*

---

## 0 · Read this section before you write any number

The recurring failure in this project has never been the science. It is **pulling a number from the
wrong generation of the experiment**, or from the wrong one of two volumes that both have an
"Appendix B". Three rules, and then the map.

> **Rule 1 — the delivered numbers are v18, joint-best convention. Nothing else.**
> **Rule 2 — a next-category macro-F1 outside 30–38 is a leaked pre-v18 number. Stop.**
> **Rule 3 — "Appendix B" is ambiguous in this project. Always say *of which volume*.**

### 0.1 · Where the delivered numbers come from

| what you want | the ONE source | not this |
|---|---|---|
| any printed cell of Ch. 5 | the table file itself, `src/tables/mobiwac/*.tex` — each value carries its provenance in a comment beside it, and that comment is what the fact gate follows | never re-derive from a JSON without reading the comment first |
| joint (multi-task) per-fold cells | `docs/results/closing_data/v18/joint_best_perfold.json`, `cells['<state>_s<seed>_joint'].folds[<fold>].joint_best` | `V18_RESULTS.md` §1 — it tabulates **diag-best**, a different and more flattering convention |
| dedicated (single-task) per-fold cells | `docs/studies/closing_data/v18/data/v18_results.json`, `per_run[].stl_cat_folds` / `stl_reg_folds` | — |
| full-precision ladder (all statistics) | `wrapup/evidence/ladder_recompute.json` — the only file in the repo carrying more than two decimals | — |
| the aggregation rule | mean the 5 folds inside a seed, **then** mean the 4 seeds. The printed ± is the sd **across the 4 per-seed means**, not across folds | — |
| CBIC (Ch. 3) / CoUrb (Ch. 4) numbers | the published tables in `../CBIC___MTL/` and `../CoUrb_2026/` — reproduce, never recompute | — |

**`docs/studies/closing_data/RESULTS_BOARD.md` is DEAD for this dissertation.** It calls itself a
single source of truth, and it is one — for **v17**, whose category cells are leak-inflated by 25–45
points (it prints AL 63.56 / FL 79.85 / CA 77.05 against the delivered 30.59 / 37.55 / 35.63). It
was last touched 2026-07-20 and never mentions v18. Several documents still route you there,
including `AGENT_GUARDRAILS.md` §N1 — **that pointer is stale; this table wins.**

### 0.2 · Why the generation matters so much

`v18` closed a label leak in the consecutive-visit graph (it is now forward-only, `src < tgt`, in
training *and* at readout). At Alabama the leak was worth **28.63 macro-F1**. That is why every
category number moved so far, and why the range check in Rule 2 works. Region barely moved (< 2 pp),
so **the range check does not protect the region axis — check the file path instead.**

Methodology of record: `docs/studies/closing_data/v18/METHODOLOGY.md`; the frozen recipe (author-
approved 2026-08-09): `docs/studies/closing_data/v18/FINAL_SETTINGS.md`.

> ⚠ **A bare `python scripts/train.py --task mtl` does NOT reproduce this dissertation.**
> `src/configs/canon.py` still pins `DEFAULT_CANON = "v17"`, on the leaked substrate. To reproduce a
> delivered cell, copy the command verbatim from `cell_joint()` in
> `docs/studies/closing_data/v18/run_wave.sh`. Ignore the B9 recipe block in the repo-root
> `/CLAUDE.md`; it is three generations old.

### 0.3 · The delivered verdict ladder — the exact wording the law allows

| axis | verdict |
|---|---|
| **next-category** | **outperforms at Florida only** (+0.19, Holm *p* 0.011). The other five differences are **unresolved** — never "matches", never "empata", never "everywhere". |
| **next-region** | **non-inferior at all six** (TOST at the registered 2-point margin), with **Texas +1.21** (Holm *p* 0.00013) and **California +1.06** (*p* 0.0000063) **outperforming**. |

The equivalence margin is registered for the **region** axis only; category equivalence is expressed
as the derived bound "within half a point". Word-level law: [`WRITING_LAW.md`](WRITING_LAW.md) §3.

> `NORTH_STAR.md` still carries the **superseded** ladder ("category everywhere, region at four of
> six", "+28…+40 macro-F1 over place-level" — the true range is +0.23…+6.29) at lines 26, 43, 175,
> 330 and 394. Do not copy a claim from it until those are fixed.

---

## 1 · Two volumes, and the appendix letters collide

The banca received **two** documents. Both have an Appendix B, an Appendix D and an Appendix E, and
they are different texts. **Never write "Appendix B" without naming the volume.**

| | **main volume** (deposited) | **supplement** (defense support only) |
|---|---|---|
| built from | `src/` | `wrapup/material_extra/` |
| PDF | `src/dissertacao.pdf` — **119 pp**, md5 `5be69d1b…` | `wrapup/material_extra/main_extra.pdf` — **27 pp**, md5 `512b372c…` |
| A | Other Scientific Contributions | — |
| **B** | **AI-Use Disclosure** | **Errata to the Reproduced Articles** |
| C | Data Ethics and Governance | — |
| **D** | **Why the Two Tasks Do Not Compete on the Shared Trunk** | **A Label-History Benchmark for the Next-Category Task** |
| **E** | **How Check2HGI and the Joint Model Work** | **The Human-Subjects Question** |
| F | — | Adaptation of the HGI Baseline |
| G | — | A Parameter-Count Control for Next-Category Prediction |

The deposited text **deliberately does not point at the supplement**: it cites only itself and the
repository. That is the document's policy, not an omission (`wrapup/erratas/README.md`, "Q24").

### 1.1 · The builds

One source, three builds, from `src/`:

| command | output | pages | what it is |
|---|---|---|---|
| `make defense` | `build/main.pdf` → copied to `dissertacao.pdf` | **119** | the banca document |
| `make academico` | `build/main_academico.pdf` | **114** | the AcademicoPG deposit body |
| `make ppgc` | `build/main_ppgc.pdf` | **120** | defense document + approval sheet |
| `cd ../wrapup/material_extra && make extra` | `build/main_extra.pdf` | **27** | the supplement |

> 🔴 **FIVE make targets overwrite the tracked `dissertacao.pdf`**: `defense`, the default `all`
> (so a bare `make`), `all3`, `fast` / `fast-defense`, and `fast3` — each ends in
> `cp build/main.pdf dissertacao.pdf`. **Never "just run make" to check something.** Use
> `make check` (runs no build) or `make academico` / `make ppgc` (they do not copy).
> To verify a rebuild, compare `pdftotext` output, **not** md5: there is no `SOURCE_DATE_EPOCH`, so
> every rebuild differs in `/CreationDate` and md5 can never match.

---

## 2 · Folder map — and whether each part is still live

| folder | what it is | still live? |
|---|---|---|
| [`src/`](src/) | **THE delivered dissertation.** LaTeX source + `chapters/` + `figures/` + `tables/` + the tracked `dissertacao.pdf` | **yes, and frozen.** Prose changes reach it only through `wrapup/erratas/` at the final deposit |
| [`wrapup/`](wrapup/) | everything that happened **after** the submission: the supplement, the erratas, the open points, the post-submission studies, the rescued evidence | **yes — this is the front line.** Start here for anything defense-related |
| [`src_utils/`](src_utils/) | the build + gate toolchain, and the round-by-round audit trail | **yes, load-bearing.** `check.sh` executes `_round9` code and delivered `.tex` files cite `_round6…_round14` by path. **Do not prune the underscore dirs** |
| [`science/`](science/) | internal scientific records (integrity studies, trunk-gain attribution, the technical appendix) + cited article PDFs | yes. The delivered source cites `science/` paths 19× |
| [`docs/`](docs/) | official UFV PDFs (submission manual) + the 2026-07-18 research records | yes — the deposit is still ahead |
| [`reviewers/`](reviewers/) | 19 invocable reviewer personas | yes — several fire again before the defense and the deposit |
| [`fundamentals/`](fundamentals/), [`storyline/`](storyline/) | frozen chapter drafts | **frozen, but do not move them.** The delivered text cites paths inside both (21 `fundamentals/_bib` provenance hits in `references.bib`; `storyline/audit/` from `preamble.tex:216` and three chapters) |
| [`archive/`](archive/) | spent planning + single-use scaffolding | **no — nothing here is a source.** See its README |
| [`exemples/`](exemples/) | exemplar dissertations (Viegas, Germano, …) used as the quality bar | yes as reference. ⚠ **gitignored — 49 MB that exist only on disk.** Backed up 2026-08-20 to `~/Backups/dissertacao_exemples_2026-08-20.tgz` |

**The boundary rule**, which is what makes this folder navigable:

> `wrapup/` is what came **after** the submission. `archive/` is what was left **behind** it.
> Everything outside those two is the live document.

### 2.1 · The law documents

| doc | what it governs |
|---|---|
| [`WRITING_LAW.md`](WRITING_LAW.md) | register, canonical names, honesty rules, the verdict ladder's permitted wording, AI-tell bans |
| [`GLOSSARY.md`](GLOSSARY.md) | the term registry. **Fail-closed: a term not in the registry may not be used** |
| [`AGENT_GUARDRAILS.md`](AGENT_GUARDRAILS.md) | process law: citation / number / claim protocols, review gates, and **§4b**, the meta-claim protocol — the most valuable text in this folder |
| [`NORTH_STAR.md`](NORTH_STAR.md) | thesis question, the three-paper arc, chapter map. ⚠ carries the superseded ladder (see §0.3) |
| [`UFV_COMPLIANCE.md`](UFV_COMPLIANCE.md) | UFV/PPGCC norms, defense prerequisites, the AcademicoPG deposit pipeline |

---

## 3 · The three articles (the coletânea)

Chapters 3, 4 and 5 are re-typeset reproductions of three papers. **Each has its own folder**, and
the delivered `references.bib` cites all three as provenance of record.

| ch. | paper | venue / status | folder |
|---|---|---|---|
| 3 | *An Investigation into Multi-Task Learning for POI Category Classification and Next-POI Prediction* | **CBIC 2025, published.** DOI `10.21528/CBIC2025-1191324`. **Satisfies Art. 21** | [`../CBIC___MTL/`](../CBIC___MTL/) |
| 4 | *ST-MTLNet: Representações Espaço-Temporais de POIs para Aprendizado Multitarefa* | **CoUrb 2026 (SBRC), published.** DOI `10.5753/courb.2026.22960`. Tarik S. Paiva 1st author, Vitor 2nd + presenter. Translated to EN for the chapter | [`../CoUrb_2026/`](../CoUrb_2026/) (see `src_en/`) |
| 5 | *Predicting the Next Category and Region of a Visit* | **MobiWac 2026, submitted** (EDAS #1571313639) | [`../[mobiwac]/`](../%5Bmobiwac%5D/) — ⚠ the paper of record is **`src_fix/`**, not `src/` |

Each article folder carries its own `ERRATA.md`, which is the subject matter of **Appendix B of the
supplement**. BRACIS 2026 is **not** a chapter: rejected, superseded by MobiWac.

---

## 4 · What is still open

The live registry is [`wrapup/open_points/LACUNAS.md`](wrapup/open_points/LACUNAS.md) — 42 items,
each remedied against the live source and the built PDF. **17 blocks are open**, in four classes:

| class | open | what closes it |
|---|---:|---|
| **ERRATA** — changes at the final deposit | 7 | ERR-1…ERR-7. Drafted erratas live in [`wrapup/erratas/`](wrapup/erratas/) |
| **DECISÃO DO AUTOR** — no agent closes these | 7 | incl. NSO-46 (the last open sign-off marker), LO-11 (authorship credit on the CoUrb article), LO-12 (Ch. 4's temporal-input description) |
| **EXECUÇÃO** — only an experiment closes it | 3 | P4, P6, GAPS-D. P1 is **closed** (2026-08-13) |
| **ORAL** — answered standing up, text untouched | 0 open | six answers drafted in `wrapup/erratas/RESPOSTAS_ORAIS.md` |

Two more registries: [`wrapup/NEW_VERSION.md`](wrapup/NEW_VERSION.md) (the `mtlcheck` rewrite — see
§5) and `src_utils/PENDENCIAS.md` (the older author queue).

> **The sign-off marker count does not live in one place any more.** Measured 2026-08-20:
> `src/` **24** + `wrapup/material_extra/` **9** + `wrapup/erratas/` **1** = **34**. A command that
> greps only the main tree misses ten, **including NSO-46, the one still open.**

---

## 5 · Traps — each one has already cost a session

1. **Strip LaTeX comments before grepping this source.** Every rewritten table and appendix carries
   a dated `EVIDENCE BASE REPLACED` header that **quotes the superseded values verbatim**, so an
   unfiltered grep always over-reports. Filter the *file* (`grep -vn '^[[:space:]]*%'`), not the
   grep output.
2. **Read the exit code, not the output.** `make check` prints a green-looking table and then exits
   2. This has produced a full day of commit messages claiming gates pass. The folder's own history
   records it twice; this session hit it a third time.
3. **The printed baseline column does not reproduce from the per-fold baseline JSONs.**
   `docs/results/baselines/faithful_poi_rgnn_*.json` are the **pre-bugfix May-2 draw**. The printed
   values are correct and the JSONs are the trap — the chain is written into the header comment of
   `src/tables/mobiwac/results.tex`.
4. **The post-submission studies are NOT in either delivered PDF.** Q13 (concatenation) and P1
   (capacity) were verified absent by text extraction. They are oral-defense material with drafted
   erratas. And **P1's own record cites `+2.12 / +2.05` from a superseded substrate** — the
   delivered margins are **TX +1.21 / CA +1.06**. Quote the delivered ones.
5. **Never mix an `mtlcheck` number with a dissertation number.** The rewrite uses a different
   evaluation protocol (nested 70/10/20 user splits, out-of-fold pooled metrics, a derived 0.4 pp
   margin instead of the registered 2 pp). Under it, **Alabama/region flips to inferior.** Those
   numbers answer defense questions; they do not correct Chapter 5. Read `wrapup/NEW_VERSION.md`
   §2 first.
6. **One live number defect sits in the supplement**, not the main volume: Appendix G's parameter
   columns (644,359 / 4,207,399 "100.2%" / 5,249,719 "101.9%") are wrong; `NEW_VERSION.md` §10.6
   measures 1,433,863 / 9,634,471 (230%) / 12,044,791 (234%). The macro-F1 results are fine and the
   conclusion strengthens. **Do not say "100.2%" aloud.**
7. **`git status` does not see `docs/results/`.** `.git/info/exclude` carries a bare `results`
   pattern. Before deleting anything, run `git check-ignore -v <path>`.

---

## 6 · Version history of this folder

| when | what |
|---|---|
| 2026-08-11 | `src_fix/` branched from `src/` at `de040d3c` as the revision working copy |
| 2026-08-16 | delivery build (`264c7996`); the supplement moved to `wrapup/material_extra/` |
| **2026-08-20** | **the folder went from three copies of the text to one.** `src_fix/` → `src/`; the pre-revision v1 `src/` and the comment-stripped mirror `src_clean/` were deleted |

**Recovering the pre-reorg state:** `git checkout dissertacao-pre-reorg -- articles/dissertacao/src`

What the 2026-08-20 reorg established, measured rather than assumed: `src_clean/` was prose-identical
to the delivered tree in **54 of 54** shared `.tex` files after comment stripping, so it held no text
of its own; the seven files the old `src/` had and the delivered tree lacked are all preserved under
`wrapup/` (four identical, three in corrected newer form). The one edit where the old `src/` was
newer — `content.tex`, made inside the delivery commit itself — **reinstated a retracted claim** and
was discarded deliberately.

It also fixed a live defect: `check.sh:7` and `fastbuild.sh:40` resolve `../src` by construction, so
from 2026-08-11 to 2026-08-20 **every `make check` and `make fast` run from `src_fix/` gated and
built the v1 tree, not the delivered one.** Nine days of green gates on the wrong document.
