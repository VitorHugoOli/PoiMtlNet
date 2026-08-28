# Dissertação de Mestrado — Vitor Hugo Oliveira

**Multitask Learning for Point-of-Interest Classification and Prediction Tasks:
The Role of the Check-in-Level Representation**
PPGCC/UFV · formato coletânea de artigos (CBIC → CoUrb → MobiWac)
**Defendida e aprovada em 28/08/2026.**

---

## Onde está cada coisa

| quero… | está em |
|---|---|
| **o texto entregue** | `src/` — `content.tex` monta `src/chapters/` |
| **o PDF do depósito** | `src/dissertacao.pdf` |
| **o PDF que a banca recebeu** | `src/banca.pdf` — **congelado, nunca rebuildar** (ver abaixo) |
| **as figuras do volume** | `src/figures/` (`.png`, `.tex`, `courb/`, `mobiwac/`) |
| **as tabelas de resultados** | `src/tables/` |
| **a bibliografia** | `src/references.bib` |
| **o suplemento** (material extra) | `wrapup/material_extra/` → `main_extra.pdf` |
| **os slides da defesa** | `presentation/slides/` — o deck congelado é `slide_final.pdf` |
| **as chapas de arquitetura** (TikZ) | `presentation/figures/` — fontes em `src/`, PDFs em `plates/`, método no README de lá |
| **o roteiro da fala** | `presentation/SPEECH.pdf`, gerado de `presentation/SLIDES.md` |
| **as notas de pesquisa e a proveniência dos números** | `science/` |
| **as ferramentas de conferência** | `src_utils/` — `check.sh` é a porta de entrada |
| **o que ficou por fazer** | `ACHADOS.md` (raiz) e `wrapup/open_points/LACUNAS.md` |

Fora do caminho principal: `reviewers/` (19 perfis de revisão usados na redação),
`archive/`, `docs/` (manuais da UFV), `exemples/` (dissertações de exemplo, fora do git).

---

## O que foi decidido — o resultado, em quatro linhas

A pergunta: **MTL ajuda predição de POI (próxima categoria + próxima região), e do que depende
a resposta?** A resposta entregue, medida com CV por usuário disjunto, n=20, correção de Holm:

- **next-category:** o modelo conjunto **supera o dedicado em Florida** (+0,19, Holm *p* 0,011).
  As outras cinco diferenças são **não resolvidas** — nunca "em todos os datasets";
- **next-region:** **não-inferior nos seis** (TOST, margem registrada de 2 pontos), com
  **Texas +1,21** e **California +1,06** superando;
- **a representação é o fator dominante:** trocar embedding place-level por check-in-level move
  mais que qualquer mudança de arquitetura — **+0,23 a +6,29** macro-F1;
- o arco é uma **trilha de correção**, não três artigos grampeados: resultado negativo publicado
  (CBIC) → diagnóstico (CoUrb) → resolução (MobiWac).

> 🛑 **Circula pelo repositório uma escada de veredito SUPERADA** — *"category everywhere,
> region at four of six"* e *"+28…+40 macro-F1"*. **Não é o resultado entregue.** O `NORTH_STAR.md`
> abre com um banner que marca os sítios; a linha 67 dele ainda não está marcada (ver `ACHADOS.md §A4`).
> Fonte do número entregue: `src/tables/mobiwac/results.tex` e `wrapup/evidence/ladder_recompute.json`.

---

## Cinco coisas que quem chega precisa de saber antes de mexer

1. **`src/banca.pdf` não reproduz do `src/` — de propósito.** É o registo congelado do que a banca
   recebeu (md5 `5be69d1b…`). O do depósito é o `src/dissertacao.pdf` (md5 `d7e85bb7…`).
   Quem os comparar sem saber isto vai reportar um defeito que não existe.

2. **Um `make` pelado sobrescreve o PDF do depósito.** Cinco alvos copiam para `dissertacao.pdf`:
   `defense`, `all`, `all3`, `fast`/`fast-defense`, `fast3`. Para conferir sem buildar: `make check`.

3. **`src_utils/_round6` … `_round14` parecem pastas de trabalho velhas e não são.**
   `check.sh` **executa** `_round9/35_wave_a_render_check.py`, e `check_audit_claims.py` lê os
   `.md` do `_round9` **como dados** (uma tabela de regexes). Apagar um deles faz o gate falhar.

4. **Os `.tex` do volume estão cheios de comentários de proveniência** que citam os valores
   superados **verbatim**. Qualquer `grep` sem filtro sobre-reporta. Filtrar o ficheiro, não a
   saída: `grep -v '^[[:space:]]*%' ficheiro.tex | grep <padrão>`.

5. **`git check-ignore -v` mente em caminhos já rastreados** (consulta o índice). Para saber se
   uma regra apanha um caminho, use `git check-ignore --no-index -v <path>`; para ver o que existe
   no disco e o git não mostra, `git status --ignored`.

---

## Reconstruir

```bash
cd src && make check      # confere sem buildar (ler o exit code, não a saída)
cd src && make academico  # builda sem tocar no dissertacao.pdf
cd presentation/figures && ./build.sh   # regenera as chapas TikZ
```

O `presentation/Makefile.speech` **só compila o `SPEECH.tex` que já existe** — não chama os dois
extractores (`build_speech_1_extract.py`, `build_speech_2_emit.py`). Ele imprime `OK` na mesma.
Para regenerar o roteiro a partir do `SLIDES.md`, correr os extractores primeiro.
