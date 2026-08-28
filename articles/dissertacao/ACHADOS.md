# ACHADOS — pasta da dissertação

> **Escopo:** `articles/dissertacao/` e **só**. O resto do repositório fica para a fase seguinte;
> o que apareceu de fora está na **Parte B**, apenas anotado, sem ação.
>
> **Data:** 2026-08-28 (dia da defesa — aprovada). **Estado:** nada foi apagado, movido ou
> commitado. Este documento é a primeira passagem: **inventário e verificação**.
>
> **Como ler a coluna de origem:** `[V]` = eu verifiquei contra os ficheiros agora.
> `[R]` = relatado pela sessão `gate` ou `ppt` e **não** reverificado. Onde eu corrigi um
> relato, está marcado **CORREÇÃO**.

---

## 0 · Decisões do autor — tomadas em 28/08

| # | assunto | decisão | estado |
|---|---|---|---|
| **D1** | `presentation/exemples/` — gravação de 6,6 GB | **apagar de vez**, reafirmada com os factos corrigidos | ✅ feito · 6,6 GB libertados · ver **A2** |
| **D2** | rastrear o deck congelado e as figuras | **sim** (implícito em *"vamos subir tudo isso no git"*) | ✅ feito · ver **A1** |
| **D3** | erratas ERR-1…ERR-7, Apêndice G, proveniência do `.bib` | **só registar, não mexer no texto** | ✅ registado · ver **A3** |
| **D4** | profundidade desta passagem | **README de entrada + commit de tudo**, sem apagar nem mover | ✅ `README.md` escrito |

**Adiado por escolha do autor:** destilar o `considerations.md` e podar o `presentation/` item a
item. A base para isso — inventário, citações e armadilhas — está neste documento.

---

# PARTE A — dentro da pasta

## A1 · 🔴 O achado mais grave: o deck congelado não existe para o git

`presentation/slides/slide_final.pdf` — o deck que a banca viu, congelado a pedido do autor e
registado em `slides/SLIDE_FINAL.md` — está **gitignorado e não rastreado**. `[V]`

```
slides/slide_final.pdf      >>> UNTRACKED + ignored <<<
figures/plates/*.pdf        >>> UNTRACKED <<<     (visíveis, mas nunca adicionados)
SPEECH.pdf                  TRACKED               (este está bem)
```

Ele desaparece de `git status`. Um `git clean -xfd` apaga-o **sem aviso e sem diff**.

**E isto já aconteceu uma vez nesta mesma pasta.** O próprio `presentation/.gitignore` documenta
o acidente, nas linhas 27-30:

> *"Em 2026-08-21 ele foi apagado por engano justamente porque a regra `*.pdf` acima o tornava
> invisível ao `git status`."*

E as linhas 33-35 do mesmo ficheiro dão a instrução para evitar a repetição:

> *"Quando o deck final existir, rastreie-o explicitamente: `git add -f presentation/<nome-do-deck>.pdf`"*

**A instrução nunca foi executada.** As chapas TikZ têm o mesmo problema por outra via: a exceção
`!figures/plates/*.pdf` existe no `.gitignore` (linha 41), portanto elas **não** estão ignoradas —
mas também nunca foram adicionadas. Estão como `??`, recuperáveis; o `slide_final.pdf` não.

✅ **Feito nesta passagem** (aditivo e reversível, alinhado com *"vamos subir tudo isso no git"*):

```bash
git add -f presentation/slides/slide_final.pdf   # o deck congelado, 1,0 MB
git add    presentation/figures/plates/          # as 4 chapas TikZ, 88 KB
```

### E o espelho do mesmo problema: as figuras existiam sem as suas fontes `[V]`

A `tikz` apontou, e confirmei: antes desta passagem o git tinha **quatro PDFs e mais nada** em
`presentation/figures/`. Os `.tex` que os geram, os scripts de build e o README **não estavam
ignorados — estavam por adicionar**.

> Quem clonasse recebia quatro imagens e **nenhuma forma de as alterar** — o oposto exato do que
> o autor quer (*"outras pessoas podem querer usar as imagens"*).

✅ Adicionados: `src/*.tex` (8), `preview.tex`, `standalone.tex`, `build.sh`, `export.sh`,
`nesped.sty`, `README.md`. **Fora, de propósito:** `build/` (ignorado) e `png/` (3 MB de renders,
regeneráveis por `build.sh`).

**Duas notas sobre o relato da `tikz`:**

- ✅ **correta** na largura: `slides/main.tex:841` diz `width=0.80\textwidth` (alvo ~302 pt),
  não os 0,85/~321 pt que o `gate` tinha dito;
- ❌ **CORREÇÃO** — ela marcou `figures/superseded/_hgi_flow_v1_376pt.pdf` como *"duplicata do
  `plates/hgi_flow.pdf` vivo"* e candidato a apagar. **Os dois ficheiros diferem**
  (`b57b48b2…` vs `97964125…`). É uma versão anterior genuína. **Não apagar** com esse argumento.

---

## A2 · 🔴 Um erro meu, e o que se perdeu com ele

**Eu disse ao autor que nada no repositório citava a gravação de 6,6 GB. Estava errado**, e ele
decidiu apagá-la com base nisso. Verifiquei tarde, antes de executar, e voltei a perguntar com os
factos corretos — **ele reafirmou o apagar com o quadro completo**, e só então apaguei.

**O que a gravação era** `[V]`: a defesa de **Henrique de Souza Santana** (mesmo programa, mesmo
template NESPeD), 61 min, 4K. Era a **base empírica** contra a qual o deck foi reescrito a
2026-08-24 — as medições de forma (mediana 21 palavras de ecrã, média 28, máx ~71) saem dela.
Citada em quatro documentos:

- `BOAS_PRATICAS_SLIDES.md §3` — a referência de forma medida;
- `APRESENTACAO_DEFESA_GUIDE.md §4.0` — a tabela de timing slide-a-slide, lida do relógio do Meet;
- `HANDOFF_SLIDES.md` — o alvo empírico do deck;
- `HANDOFF.md` — o método de extração de frames.

**Não havia cópia** `[V]`: o tarball `~/Backups/dissertacao_exemples_2026-08-20.tgz` (43 MB) **não**
a continha, e `/Volumes/linux/VIDEO/` já não a tinha. O próprio `BOAS_PRATICAS_SLIDES.md:254`
avisava: *"se o arquivo sumir do disco, some de vez."*

> ⚠ **Consequência viva, para quem chegar depois:** aqueles quatro documentos citam agora um
> ficheiro que **não existe**. As medições continuam válidas e o método está escrito em cada um
> deles — mas **a evidência primária não é mais reverificável**. Quem quiser refazer a medição
> precisa de outra gravação. **Isto devia ser anotado nos quatro documentos** (não o fiz: são do
> `gate`, e a decisão de redação é dele).
>
> Argumento a favor da decisão, para o registo: era a defesa **de outra pessoa** e continha
> **conteúdo pessoal do autor** (61 min do ecrã dele, com outras janelas). Sob o critério novo
> — *"colegas do departamento vão poder ler e usar isto"* — não é um ficheiro que se queira
> preservar e partilhar.

### O resto do disco `[V]`

Depois do apagar, `presentation/` passou de **6,6 G para 17 M**; o espaço livre da máquina foi de
13 GiB para 19 GiB. O que resta ignorado no disco, dentro do escopo, é pequeno e regenerável:

| tamanho | caminho | o que é |
|---|---|---|
| 49 M | `exemples/` | dissertações de exemplo · **tem** backup (`~/Backups/…2026-08-20.tgz`) |
| 16 M | `src/build/` | saída de build |
| ~4 M | builds de `presentation/` e `wrapup/material_extra/` | saída de build |
| 3 M | `presentation/figures/png/` | renders das chapas, regeneráveis por `build.sh` |
| ~1 M | `.DS_Store` (19×), `__pycache__` (2×) | cruft |

## A3 · Pendências reais do depósito

### ERR-1 … ERR-7 — sete erratas, todas abertas `[V]`

Em `wrapup/open_points/LACUNAS.md`, todas marcadas `[ABERTO]`, com rascunhos `.tex` em
`wrapup/erratas/`. A `AUT-35` decidiu adiá-las para depois da defesa — **é agora**.

| | linha | assunto |
|---|---|---|
| ERR-1 | 336 | normalização de "multi-task" entra na coluna de prosa **publicada** |
| ERR-2 | 356 | títulos dos cap. 3 e 5 desidratam o hífen de títulos citados |
| ERR-3 | 373 | nota de anotação interna imprime **dentro** de uma referência |
| ERR-4 | 386 | 2 entradas do `.bib` não citadas (`belkin2003laplacian`, `santos2024urban`) |
| ERR-5 | 402 | 6 entradas sem identificador; 3 resolvíveis já |
| ERR-6 | 427 | atribuição errada da origem da taxa de aprendizado de FL e CA |
| ERR-7 | 450 | a mesma frase grada Florida como "fewer folds" |

`LACUNAS.md:51` regista que **ERR-6 e ERR-7 não vinham de registo nenhum** — foram encontrados na
auditoria. Vale ler antes de fechar os outros cinco.

### Apêndice G do suplemento imprime colunas de parâmetros erradas `[V]`

`wrapup/material_extra/chapters/apx_i_parameter_count_control.tex:101-102` ainda imprime:

```
Alabama    & 4,197,621 & 644,359 & 4,207,399 (h=672)   -> rotulado "100,2%"
California & 5,151,189 & 644,359 & 5,249,719 (h=752)   -> rotulado "101,9%"
```

O defeito **já está registado** em `CLAUDE.md §5 item 6`, que mede outra coisa
(1.433.863 / 9.634.471 = 230 % / 12.044.791 = 234 %) e conclui: *"Do not say 100.2% aloud."*
Os resultados de macro-F1 estão bem e a conclusão **fortalece**. O que falta é o `.tex` refletir
isso. **Pendência viva do suplemento, não do volume principal.**

### `banca.pdf` não reproduz do `src/` — e isso é de propósito `[V]`

```
src/banca.pdf        md5 5be69d1bf589e6fe5e794898a06306cf   <- congelado, o que a banca recebeu
src/dissertacao.pdf  md5 d7e85bb7a00911aa5a8a37eb4f51f76a   <- o que vai ao depósito
```

São ficheiros diferentes **por desenho** (`CLAUDE.md §1.2`). Quem auditar sem saber disto vai
reportar um defeito que não existe. **Nunca rebuildar o `banca.pdf`.**

---

## A4 · Documentos que induzem em erro

### NORTH_STAR.md — **CORREÇÃO ao relato do `gate`** `[V]`

O `gate` disse que a escada superada está nas linhas **26, 43, 175, 330, 394** e que "ninguém deve
copiar uma alegação de lá". **Duas correções:**

1. **O ficheiro não mente em silêncio.** Tem um banner nas linhas 15-27 com o veredito entregue
   (categoria: **só Florida**, +0,19, Holm p 0,011; região: **não-inferior nos seis**, TX +1,21,
   CA +1,06) e diz explicitamente que a faixa real é **+0,23…+6,29**, não "+28…+40".
2. **As linhas que ele deu não são onde o texto está hoje.** Os sítios vivos são **21, 22**
   (o próprio banner), **49** (marcada `[SUPERADO 2026-08-20]`) e **67**.

🔴 **A linha 67 é o problema real, e ninguém a tinha apontado.** É a linha da tabela dos três
papers — o sítio mais copiável do ficheiro — e carrega `+28…+40 macro-F1` **e** *"category
outperforms the dedicated model at all six datasets (+5.3…+9.4)"* **sem marcador `[SUPERADO]`
nenhum**. As outras estão todas marcadas; esta escapou.

> **Não a alterei** — é decisão de conteúdo do `gate`/autor se marca, reescreve, ou deixa
> (a linha descreve o que o *paper submetido* alega, o que pode ser legítimo). Mas **hoje ela é
> a única fonte não marcada da alegação superada dentro da pasta.**

### `references.bib` — 20 comentários de proveniência apontam para um caminho que não existe `[V]`

```
citado:   articles/dissertacao/fundamentals/_bib/new_references_ch2.bib   <- NÃO EXISTE
real:     articles/dissertacao/science/fundamentals/_bib/new_references_ch2.bib
```

20 ocorrências exatas. A pasta `fundamentals/` na raiz da dissertação não existe — o ficheiro
vive sob `science/`. A trilha de proveniência do `.bib` está **partida desde a reorganização**.
Correção mecânica e segura (é tudo comentário `%`; não toca em campo nenhum do BibTeX), mas é
edição no source entregue — **fica para depois de D3**.

### `AGENT_GUARDRAILS.md §N1` aponta para fora `[R, não verificado]`

O ponteiro é da pasta; o alvo (`docs/studies/closing_data/RESULTS_BOARD.md`) é da raiz do repo e
está **fora do escopo**. Ver **B1**.

---

## A5 · O que parece lixo e **não é** — armadilhas confirmadas

**`src_utils/_round6` … `_round14` — NÃO APAGAR.** `[V]` Confirmado por dois caminhos:
- `src_utils/check.sh:345-346` **executa** `python3 $UTILS/_round9/35_wave_a_render_check.py`;
- `check_audit_claims.py` **lê os `.md` como dados** — tem uma tabela de regexes que casa contra
  `_round9/37_reviewer_gate_round9.md`, `_round9/47_applied_check.md` e outros. Apagar um `.md`
  do `_round9` faz o gate de auditoria falhar, não avisar.

**`science/` — CORREÇÃO ao relato do `gate`.** `[V]` Ele disse "o source entregue cita `science/`
19 vezes". A contagem está certa (19), mas o que importa é isto: **as 19 estão todas dentro de
comentários LaTeX (`%`)**. Não há uma única referência em código ativo.

> **Consequência prática:** apagar `science/` **não parte o build**. O argumento para manter é
> "preserva a proveniência", não "senão quebra". É um argumento mais fraco — e portanto uma
> decisão diferente da que o briefing sugeria. `science/` são 15 MB.
>
> Nota separada: as **ferramentas** (`check_wordcount_claims.py`, `sync_deliverables.py`,
> `verify_format.py`, `mkformat.py`, …) referem `science/AGENT_HANDOFF.md`, e o
> `sync_deliverables.py:40` **mapeia-o**. Esse ficheiro é vivo. `science/` inteiro não é o mesmo
> que `science/AGENT_HANDOFF.md`.

**`src/figures/` — o deck da defesa depende dela.** `[V]` `presentation/slides/main.tex:14`:

```latex
\graphicspath{{img/}{../figures/plates/}{../../src/figures/}{../../src/figures/mobiwac/}{../../src/figures/courb/}}
```

Quatro das figuras do deck vivem na árvore entregue. `\includegraphics` que não resolve produz
**caixa vazia, não erro** — parte em silêncio. `[R]` do `ppt`, caminho verificado por mim.

**`src/banca.pdf`** — ver A3. **`exemples/`** — ver A2.

---

## A6 · Armadilhas de mecânica — uma delas está mal documentada

### 🔴 `git check-ignore` **mente em caminhos já rastreados** — CORREÇÃO importante `[V]`

A regra escrita em `CLAUDE.md §5 item 7` e repetida pelo `gate` é *"antes de apagar, corre
`git check-ignore -v <path>`"*. **Essa forma do comando dá a resposta errada:**

```
docs/results  (default):     NOT IGNORED
docs/results (--no-index):   .git/info/exclude:9:results   docs/results
```

`git check-ignore` consulta o índice: se o caminho **já está rastreado**, reporta "não ignorado"
mesmo estando coberto por uma regra. Foi exatamente isto que me deu leituras contraditórias
(`docs/results` "não ignorado" com 4 169 ficheiros rastreados, mas `.../rundirs/results`
"ignorado" — a diferença é o índice, não a regra).

**A sonda correta é uma destas duas:**
```bash
git check-ignore --no-index -v <path>     # a regra, sem o índice
git status --ignored --porcelain <dir>    # o que existe no disco e o git não vê
```

✅ **E a boa notícia:** **não há nenhuma pasta chamada `results` dentro de `articles/dissertacao/`**
`[V]`. **A armadilha do `docs/results/` não dispara no nosso escopo.** O que dispara é a lista da
**A2** — e é essa que interessa antes de qualquer limpeza aqui.

### Cinco alvos do make sobrescrevem `dissertacao.pdf` `[V]` — `gate` correto

`src/Makefile`: `defense` (36), `all` → `defense` (33), `all3` → `defense` (52),
`fast`/`fast-defense` (59-60), `fast3` (66-69, `cp build/main.pdf dissertacao.pdf`).

**Um `make` pelado sobrescreve o PDF do depósito.** Seguro para conferir: `make check` (não
builda), `make academico`, `make ppgc` (não copiam).

### `make check` sai com código ≠ 0 mesmo imprimindo verde `[R]`

Relatado pelo `gate`; **não executei** (evito correr o harness sem necessidade). **Ler o exit
code, não a saída.** Registado como aviso, não como facto verificado.

### `Makefile.speech` diz `OK` sobre um intermediário velho `[V]` — `ppt` correto

```make
SPEECH.pdf: SPEECH.tex
	xelatex ... && echo "OK -- SPEECH.pdf (N paginas)"
```

O cabeçalho diz *"o roteiro de fala, **gerado do SLIDES.md**"*, mas a regra **só compila o
`SPEECH.tex` que já está no disco** — nunca chama `build_speech_1_extract.py` nem
`build_speech_2_emit.py`. Imprime `OK` na mesma. **Ou se corrige a dependência, ou se documenta.**

### Comentários LaTeX inflam qualquer grep `[V]` — `gate` correto

Cada tabela reescrita carrega um cabeçalho `EVIDENCE BASE REPLACED` que **cita os valores
superados verbatim**. Filtrar o **ficheiro**, não a saída:
`grep -v '^[[:space:]]*%' ficheiro.tex | grep <padrão>`. Foi assim que separei os 19 `science/`
(todos comentário) dos 0 ativos.

---

## A7 · Fio solto que o `gate` deu como aberto e **já está fechado** — CORREÇÃO `[V]`

O briefing lista como pendência: *"o `6.909.789` … não reconcilia com os `5.151.189`"*.

**Duas coisas:**

1. **Os dois números são de estados diferentes** — `6.909.789` é **Alabama**, `5.151.189` é
   **California**. Não deviam reconciliar. O par correto é **CA 8.809.533 (reconstruído) vs
   CA 5.151.189 (medido)**.
2. **O ficheiro já resolve isto**, e ontem. `wrapup/open_points/AUDITORIA_PRE_LEAK.md:149-157`
   tem um bloco `🔴 CORRIGIDO 2026-08-27`, com o `6.909.789` **riscado**:

   > *"O 6.909.789 **não foi medido** — é uma reconstrução com defaults assumidos, e os logs de
   > execução v18_2 a contradizem… **O 4.197.621 é o que o modelo entregue de facto executa.**"*

**Não é pendência.** O que sobra é consequência conhecida: a largura pareada `h=672` pareia o
orçamento da arquitetura **anterior** — que é precisamente o defeito do Apêndice G (A3).

**Duplicação real, essa sim:** `wrapup/post_submission_studies/EXECUTION_WAVE.md` e
`docs/studies/closing_data/v18/EXECUTION_WAVE.md` são **byte-idênticos**
(md5 `55070d377d380f33ecf27c7630ea92fb`). Um está dentro do escopo, o outro fora. Ver **B2**.

---

## A8 · `presentation/` — o que é registo e o que é andaime

A pasta do `gate` terminou o trabalho. **Nada aqui foi tocado**; é proposta para D4.

**Registo — fica** (`[R]` do `gate`/`ppt`, com o meu voto):
- `considerations.md` (394 KB) — decisões **AUT-1…AUT-36** com a razão de cada uma. **Destilar,
  não apagar.** O `gate` sabe quais foram revogadas;
- `HANDOFF_GATE.md` — 12 classes de erro + 6 regras de medição que **não são sobre slides**;
  generalizam. **Candidato a subir para a raiz da pasta;**
- `HANDOFF_PPT.md`, `BOAS_PRATICAS_SLIDES.md` — a forma medida do template;
- `slides/slide_final.pdf` + `slides/SLIDE_FINAL.md` — ver **A1**;
- `figures/plates/` — **zero ficheiros sem uso** `[R, ppt]`. Nada a podar.

**Instrumentos vivos — ficam** `[R, ppt]`:
- `canaria_coluna.py` — apanha duas classes que o `Overfull` dá como zero: coluna de `columns`
  que transborda (~11 frames de duas colunas) e botões do índice que não renderizam;
- `build_speech_1_extract.py` + `build_speech_2_emit.py` — a cadeia que gera o `SPEECH.pdf`;
- 🛑 **`SLIDES.md` (320 KB) não é andaime** — é a **fonte** do `SPEECH.pdf`. Apagá-lo não parte
  o deck: parte o roteiro, **em silêncio**, porque o `SPEECH.tex` continua no disco a compilar.

**Provável andaime — confirmar item a item com o `gate`:**
`CORTES_FALA.md`, `SPEC_RODADA_27AGO.md`, `FAIXA_VS_CORPO.md`, `hgi_draw.txt`, `andrej_mtl.md`,
`ink_sweep.py`, `diff_fala.py`, `archive/`, `slides_ux/`, `HANDOFF.md`, `HANDOFF_SLIDES.md`,
`PLANO_FLUXO_DEFESA.md`, `APRESENTACAO_DEFESA_GUIDE.md`.

> O `gate` avisou que **alguns têm consumidores que não se veem**. Confirmado no caso do
> `SLIDES.md`. **Nenhum destes sai sem passar por ele primeiro.**

---

## A9 · Estado do git no escopo `[V]`

`19 modificados` + `16 não rastreados`, nenhum commitado. Inclui trabalho de conteúdo real
(`GLOSSARY.md`, `WRITING_LAW.md`, `AGENT_GUARDRAILS.md`, `SPEECH.*`, `slides/main.tex`,
`ESTUDOS_DEFESA.html`).

**Antes de qualquer limpeza, este trabalho devia estar commitado** — senão a limpeza e o trabalho
do dia misturam-se no mesmo diff e deixam de ser reversíveis em separado. **Push só no fim, com
o autor.**

---

# PARTE B — fora do escopo (só anotado, para a fase seguinte)

> **Nada aqui foi tocado, e nada aqui deve ser tocado nesta fase.**

**B1 · `docs/studies/closing_data/RESULTS_BOARD.md` está morto para a dissertação** `[R, gate]`
É v17, com categoria inflada por vazamento em 25-45 pontos (imprime AL 63,56 contra os 30,59
entregues) — e **chama-se a si próprio "single source of truth"**. `AGENT_GUARDRAILS.md §N1`
(dentro do escopo) aponta para ele. **O ponteiro é nosso, o alvo não.** `[V]` o ficheiro existe
e está rastreado; o conteúdo não foi auditado por mim.

**B2 · `EXECUTION_WAVE.md` duplicado** `[V]` Byte-idêntico em
`articles/dissertacao/wrapup/post_submission_studies/` (dentro) e `docs/studies/closing_data/v18/`
(fora). Decidir qual é o canónico quando a fase seguinte abrir — **não desduplicar agora**, porque
metade da decisão está fora do escopo.

**B3 · A armadilha do `results` nu é real fora do escopo** `[V]` `.git/info/exclude:9` tem um
padrão `results` sem barra. Não dispara em `articles/dissertacao/`, **dispara em `docs/`**: por
exemplo `docs/results/closing_data/v18_2/modal_runs/*/rundirs/results` está ignorado e invisível.
Levar a **A6** (a forma correta da sonda, `--no-index`) para a fase seguinte.

**B4 · `ARMADILHAS_DE_MEDICAO.md` na raiz do repositório** `[R, ppt]` Doze classes de erro em que
um instrumento devolve verde e mede outra coisa. **§7** (âncora ambígua devolve a primeira
ocorrência), **§9** (uma cópia é outro artefacto no instante em que a original muda) e **§12**
(uma alegação vive em várias superfícies) aplicam-se diretamente a uma limpeza. Não verificado
por mim; fica como leitura obrigatória da fase seguinte.

---

## Procedência deste documento

Verificado por mim `[V]`: A1, A2, A3 (erratas, Apêndice G, md5s), A4 (NORTH_STAR, references.bib),
A5 (`_round9`, `science/`, `graphicspath`), A6 (exceto o exit code do `make check`), A7, A9, B2, B3.

Relatado e **não** reverificado `[R]`: o backup do `exemples/` da raiz, o exit code do
`make check`, a cobertura das canárias, a ausência de órfãos em `figures/plates/`, o conteúdo do
`RESULTS_BOARD`, o `ARMADILHAS_DE_MEDICAO.md`.

Correções aos briefings: **A4** (linhas do NORTH_STAR — a real é a 67, não marcada),
**A5** (`science/` é proveniência, não dependência de build), **A6** (`git check-ignore` precisa
de `--no-index`), **A7** (o `6.909.789` já está fechado; os números eram de estados diferentes).
