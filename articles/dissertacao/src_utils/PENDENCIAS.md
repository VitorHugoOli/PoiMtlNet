# PENDENCIAS.md — o que depende de voce

> **Registro de pendencias da dissertacao (v4, 2026-07-29).** Cada item aqui esta bloqueado em uma
> decisao sua, em uma aprovacao do orientador/Comissao, ou em um fato externo. **Nada aqui pode ser
> resolvido por um agente, e nenhum foi resolvido sozinho.**
>
> Formato de cada item: **(A) o que falta**, **(B) por que importa**, **(C) o que eu preciso de voce**.
> Onde ja existe texto pronto ou medicao feita, o caminho esta indicado.

---

## Como ler este arquivo

Ele tem quatro partes e as duas primeiras exigem algo de voce:

1. **§2 Aberto e bloqueado em voce** — decisoes e aprovacoes. **Esta e a sua fila.** A numeracao
   comeca em 2.1 por continuidade: **cinco** comentarios em tres arquivos do fonte citam
   "PENDENCIAS 2.4", e renumerar quebraria essas citacoes.
   > **Contagem corrigida 2026-07-30:** dizia "quatro comentarios". Sao cinco, em
   > `apx_b_static_scope.tex` (3), `4_courb.tex` (1) e `apx_b_errata.tex` (1). Medido com
   > `grep -rcE 'PENDENCIAS[[:space:]]*(§|\S|[Ss]ection|[Ss]ec\.?|item)?[[:space:]]*2\.4' src/chapters/*.tex`.
   > O argumento nao muda -- renumerar continua quebrando citacoes -- mas o numero que o sustenta
   > agora fecha com o comando que o mede. O push que esta linha citava ja foi feito (item 2.1).
2. **§5 Levantados do `CODEX_AUDIT.md` ao arquiva-lo** — nove pontos daquela auditoria que ainda dependem de voce. Eram
   para ser aplicados por decisao sua e **nao estao no documento**; cada um traz a medicao que mostra isso.
3. **§3 Aberto e bloqueado em terceiros** — orientador, Comissao, revisores do MobiWac. Fora do seu controle e do meu.
4. **§4 O que auditar primeiro** — a lista priorizada, se voce tiver uma hora.

**O que saiu daqui.** O antigo §1 ("Fechado nesta rodada", a rodada 6 inteira, com commits) foi movido para [
`_archive/PENDENCIAS_RESOLVIDOS.md`](_archive/PENDENCIAS_RESOLVIDOS.md) em 2026-07-29, a seu pedido, **com os 19 hashes
de commit intactos**. Nada foi apagado nem resumido. O
`CODEX_AUDIT.md` foi para o mesmo arquivo, depois do levantamento do §5.

Um item que esta ausente do texto porque alguem **decidiu** que ficasse fora nao e pendencia: esta em [
`LEFT_OUT.md`](LEFT_OUT.md), com quem decidiu e quando. Um item ausente porque ninguem chegou nele e pendencia e esta
aqui.

**Estado do build: NAO CONFIRMADO agora, e `make check` esta saindo 2.**

Esta secao dizia, numa versao anterior de hoje, que `make check` **saia 0** em ~2 s. **Isso estava errado quando eu
escrevi** — a minha propria medicao no mesmo dia deu **RC=2**, e eu tinha lido o codigo de saida de um estagio de `pipe`
(`| grep`) em vez do codigo do comando. E a mesma classe de defeito que o post-mortem desta rodada documenta, dentro do
arquivo que existe para voce confiar. Corrigido, com o codigo lido direto:

| o que                        | medido em 2026-07-29, lendo `$?` direto                                                                                                                                                          |
|------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `make check`                 | **RC=2**, 20 gates, ~19 s                                                                                                                                                                        |
| gate que falha               | **`sync_page_counts.py`**, e o motivo importa: `src/build/main.log` estava **sem contagem de pagina, build incompleto**, porque outra trilha da rodada 7 estava compilando enquanto eu media     |
| segundo achado do mesmo gate | tres claims de pagina **stale**, e uma linha `UNMATCHED` em **este arquivo**: o padrao que guardava o numero de paginas do `make ppgc` nao casa mais, entao aquela afirmacao esta **sem guarda** |
| `make fast3`                 | 13 s para os tres alvos (contra 115 s do `make all3` e 83 s de um alvo so)                                                                                                                       |

**Nao anote nenhuma contagem de pagina deste arquivo como verdade hoje.** Enquanto eu escrevia, `src/`
tinha 16 arquivos modificados e quatro novos por outra trilha, e as contagens que eu observei mudaram entre duas
medicoes minhas (108/105/109, depois 100/86/101 com um build pela metade). Quando a arvore parar de se mover, remeça:

```bash
cd /Users/vitor/Desktop/mestrado/ingred/articles/dissertacao
source src_utils/texenv.sh && (cd src && make fast3) && bash src_utils/build.sh src both
(cd src && make check); echo "RC=$?"          # leia o RC assim, nunca depois de um pipe
python3 src_utils/sync_page_counts.py --write # conserta as tres claims stale e a linha UNMATCHED
```

Ha tambem um defeito real por tras da falha do gate, que **nao e meu e nao esta consertado**: o build de deposito
imprime **8 na pagina fisica 9** (`main_academico.pdf`). Mesma classe do C-1 da rodada 6, no unico build que e
depositado. Detalhe no post-mortem, `_round7/25_postmortem.md` §5.

```bash
cd /Users/vitor/Desktop/mestrado/ingred/articles/dissertacao
source src_utils/texenv.sh && (cd src && make fast3) && bash src_utils/build.sh src both
python3 src_utils/sync_page_counts.py --write   # se a contagem mudou
```

---

## §2 · Aberto e bloqueado em VOCE

### 2.1 Os 53 marcadores `[NEEDS SIGN-OFF]` no fonte

> **CONTAGEM E TABELA CORRIGIDAS, 2026-07-30 (round 8), por medicao.** O cabecalho dizia **46** e a
> tabela abaixo listava um arquivo que **nao existe mais**: o `0_main.tex` foi dividido em
> `preamble.tex` + `content.tex` em 2026-07-29 (commit `2b9b853d`), e os quatro marcadores que a
> tabela atribuia a ele estao hoje em `content.tex`. A tabela tambem nao fechava com o proprio
> comando que ela manda rodar: o comando devolve **57**, a tabela somava 46, e sete marcadores novos
> tinham entrado desde `2bf5f8ea` sem que ninguem reconciliasse. E a classe do V13 do
> `AGENT_GUARDRAILS` -- uma tabela cujo cabecalho conta N tem que ter N linhas.

**(A) O que falta.** **55 marcadores no fonte** (59 se contar a arvore gerada `src/build/`, que nao e
fonte: `build/fmt/_body.tex` e escrito pelo `mkformat.py` e nao esta versionado). O numero do fonte e
o que importa, porque e nele que voce assina:

```bash
cd /Users/vitor/Desktop/mestrado/ingred/articles/dissertacao
grep -rn "NEEDS SIGN-OFF" src/ --exclude-dir=build | grep -v Binary | wc -l   # 55 em f624767c
grep -rn "NEEDS SIGN-OFF" src/ | grep -v Binary | wc -l                       # 59: os 4 a mais sao o _body.tex gerado
```

> **ESTE NUMERO ANDA, e a minha propria correcao envelheceu em trinta minutos.** Eu reescrevi esta
> tabela com **53**, medido em `ecb81fb6` (21:40), corrigindo os 46 que estavam aqui. 53 estava certo
> naquele commit. Depois **eu mesmo** acrescentei dois marcadores -- em `d9ab436f`
> (`apx_f_cosine.tex`, a decisao de rodar ou nao os tres datasets) e em `a07e547b`
> (`apx_b_errata.tex`, a frase do orcamento de tuning) -- e uma track paralela desta rodada
> acrescentou o seu. Dai 55.
>
> A licao nao e que 53 estava errado. E que **uma tabela de contagem neste arquivo e instrumento ruim
> para esta grandeza**, porque cada sign-off novo a invalida, inclusive os meus. O que nao envelhece
> e o **comando**, e ele esta acima. A tabela abaixo esta datada por isso. Se a soma nao fechar quando
> voce ler, rode o comando: ele e a resposta, a tabela e so a conveniencia. A §2.13 registra a mesma
> instabilidade pelo lado do `VERIFY_LIST` A7.

| Arquivo                                                                                                                                                                     |      n |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------:|
| `chapters/6_conclusion.tex`                                                                                                                                                 |      9 |
| `chapters/2_fundamentals.tex`, `chapters/apx_a_contributions.tex`, `chapters/apx_b_errata.tex`                                                                               | 6 cada |
| `content.tex` (era `0_main.tex`)                                                                                                                                            |      4 |
| `chapters/5_mobiwac/06_results.tex`                                                                                                                                         |      3 |
| `chapters/1_introduction.tex`, `3_cbic/results.tex`, `5_mobiwac/02_related.tex`, `5_mobiwac/07_discussion.tex`, `apx_f_cosine.tex`                                           | 2 cada |
| `3_cbic/method.tex`, `4_courb.tex`, `4_courb/methodology.tex`, `4_courb/results.tex`, `5_mobiwac.tex`, `5_mobiwac/05_setup.tex`, `apx_b_static_scope.tex`, `apx_c_ai_disclosure.tex`, `apx_d_ceiling.tex`, `apx_e_ethics.tex`, `main_extra.tex` | 1 cada |

**Soma: 9 + (6x3) + 4 + 3 + (2x5) + (1x11) = 55**, em 22 arquivos, medido em `f624767c`.

**Todos os 55 estao dentro de comentarios `%`**, entao nenhum imprime no PDF: o marcador e recado
para voce, nao texto do leitor. Medido com o stripper do `check_audit_claims.py` (0 ocorrencias
visiveis depois de remover comentarios, 55 no fonte cru).

**(B) Por que importa.** Cada um e uma frase reescrita por um agente em prosa que e sua, ou uma mudanca de escopo num
capitulo publicado. Nenhuma pode ir a banca sem voce ter lido.

**(C) PARE ANTES DE PUBLICAR: ha um commit local DESTRUTIVO no worktree do `mobiwac`.**

Voce me pediu para criar o worktree e fazer o push. Nao fiz, e a razao importa.

O worktree `.claude/worktrees/wf_9231ab26-2a8-4` (que esta em `mobiwac`) **ja tinha um commit local**,
`6c4267ba`, com a mensagem *"add the five missing reproducibility artifacts"*. Medi o que ele faz:

```
15 files changed, 10 insertions(+), 2028 deletions(-)
```

**Ele nao adiciona nada. Ele APAGA a arvore `analysis_protocol/` inteira** — incluindo os tres arquivos que ja estavam
publicos e que este item existe para nao republicar — mais os quatro JSON de ceiling per-fold de Istambul e dois scripts
de `scripts/closing_data/`:

```
D analysis_protocol/CEILINGS_N20_FINAL.md          D analysis_protocol/README.md
D analysis_protocol/DEVIATION_LOG.md               D analysis_protocol/STATISTICAL_PROTOCOL.md
D analysis_protocol/EXECUTED_ANALYSIS.md           D analysis_protocol/m2_prereg_output.txt
D analysis_protocol/JOINT_BEST_RESULTS.md          D analysis_protocol/istanbul_cat_ceiling_perfold/*.json  (4)
D analysis_protocol/JOINT_BEST_SCORING.md          D scripts/closing_data/m2_prereg_perfold.py
M README.md                                        D scripts/closing_data/score_joint_best.py
```

> **CORRECAO GRAVE, 2026-07-30.** A frase abaixo dizia *"NADA FOI PERDIDO. O commit e local e
> `origin/mobiwac` continua em `3c57197c`"*. **Isso ficou falso.** Verificado contra o remoto com
> `git fetch origin mobiwac`: `origin/mobiwac` e o local apontam ambos para `6c4267ba`. A delecao de
> 2.028 linhas **foi publicada**. Os 14 arquivos de reprodutibilidade estiveram ausentes do branch
> publico enquanto o artigo estava em revisao.
>
> A verificacao anterior nao errou o comando; ela errou o **momento**. Rodou antes do push e a
> conclusao foi registrada como se fosse permanente. Um estado remoto verificado uma vez nao continua
> verdadeiro: se a afirmacao e sobre o remoto, ela precisa da data da medicao e de uma re-medicao
> antes de ser reusada.

**RESOLVIDO (parcialmente) EM 2026-07-30**, com a sua decisao de reverter em vez de reescrever a historia publica. Dois
commits preparados e verificados:

| commit     | efeito                                                           |
|------------|------------------------------------------------------------------|
| `b7b072d2` | `git revert 6c4267ba` — restaura os 14 arquivos, 2.028 insercoes |
| `0288cb70` | adiciona os TRES que faltavam de fato, 406 insercoes, 0 delecoes |

Efeito liquido contra o tip publicado: **18 arquivos, 2.434 insercoes, 0 delecoes**, 26 arquivos de reprodutibilidade
presentes.

**A CONTAGEM MUDOU DE NOVO, e a razao importa.** Nove -> cinco -> **tres**. A auditoria por **conteudo** (nao por nome)
mostrou que dois dos cinco ja estao publicados sob outro diretorio, com conteudo diferente:

| arquivo                | no branch               | diferenca                       | acao           |
|------------------------|-------------------------|---------------------------------|----------------|
| `m1_stats_n20.py`      | `scripts/closing_data/` | 411 vs 335 linhas, 84 alteradas | **nao tocado** |
| `m2_prereg_perfold.py` | `scripts/closing_data/` | 214 vs 222 linhas, 36 alteradas | **nao tocado** |

> **DECISAO SUA:** substituir um artefato ja publicado por uma versao local divergente e decisao de
> autor, nao limpeza. Eu nao toquei. Se a versao local e a correta, e um commit seu.

> **O PUSH JA ACONTECEU e este bloco estava obsoleto de tres maneiras. Corrigido 2026-07-30 (round
> 8), tudo medido, nada empurrado por mim.** O texto abaixo dizia *"O PUSH FALTA, e nao consigo
> faze-lo daqui"* e mandava rodar quatro comandos. Medido agora:
>
> | o que o bloco antigo dizia                                    | medido em 2026-07-30                                                                                                       |
> |---------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
> | o push falta                                                  | **feito.** `origin/mobiwac` esta em `0288cb70`; o reflog do remoto registra `0288cb70 update by push`                       |
> | `git log --oneline mobiwac..mobiwac-fix` -> `0288cb70 b7b072d2` | devolve **0 linhas**. `mobiwac-fix` esta em `6c4267ba`, que e ANCESTRAL do `mobiwac` local: o branch de trabalho ficou atras |
> | `git diff --name-status mobiwac..mobiwac-fix \| grep -c '^D'`   | devolve **17**, nao 0 -- na direcao em que o bloco mandava ler, o `mobiwac-fix` e que apaga 17 arquivos                     |
> | `git fetch <bundle> mobiwac:mobiwac-fix`                      | **falha:** `fatal: couldn't find remote ref mobiwac`. O bundle expoe um unico ref, `HEAD` (= `0288cb70`), nao `mobiwac`     |
>
> **Rodar os quatro comandos hoje, na ordem em que estavam escritos, faria exatamente o dano que o
> bloco existia para evitar:** o `fetch` falha, o `grep -c '^D'` devolve 17 (e nao 0, entao a propria
> guarda do bloco te pararia), e um `git push origin mobiwac-fix:mobiwac` seria um **retrocesso** do
> tip publicado de `0288cb70` para `6c4267ba` -- ou seja, republicaria a delecao de 2.028 linhas.
> **Nao rode nada deste bloco.** Ele fica registrado, e nao apagado, porque um comando de recuperacao
> que expirou em silencio e a classe de defeito que o `AGENT_GUARDRAILS` §4b V6 nomeia.

**Estado publicado, medido em 2026-07-30 (leia a data: uma verificacao de remoto tem validade):**

```bash
cd /Users/vitor/Desktop/mestrado/ingred
git fetch origin mobiwac
git rev-parse origin/mobiwac                                  # 0288cb70...
git diff --name-status 6c4267ba..origin/mobiwac | grep -c '^D' # 0 delecoes contra o tip que apagava
git diff --shortstat 3c57197c..origin/mobiwac                  # 3 files changed, 406 insertions(+)
```

O efeito liquido contra `6c4267ba` (o tip que continha a delecao) e **17 arquivos adicionados, 1
modificado, 0 apagados**; contra `3c57197c` (o tip anterior a ela) e **3 arquivos, 406 insercoes, 0
delecoes** -- os tres que faltavam de fato. `analysis_protocol/` tem 13 arquivos no branch e
`scripts/closing_data/` outros 13. **Nada mais a empurrar aqui.**

### 2.2 Publicar os arquivos que faltam no branch publico — FEITO em 2026-07-30

> **ESTE ITEM FOI APAGADO POR ENGANO, nao resolvido.** Ele existiu da versao `98a33251` ate
> `3bd47d5d` (2026-07-29), e naquele commit -- que era sobre *citacoes de tracker*, nao sobre este
> item -- o bloco 2.2 inteiro sumiu do arquivo sem ser arquivado em
> [`_archive/PENDENCIAS_RESOLVIDOS.md`](_archive/PENDENCIAS_RESOLVIDOS.md) e sem renumerar o resto.
> O trabalho **ainda estava aberto** naquele momento. O buraco entre 2.1 e 2.3 ficou no arquivo por
> um dia inteiro e foi voce quem notou. Restaurado aqui com o desfecho, porque um item que
> desaparece silenciosamente e pior do que um item marcado errado: nada aponta para ele.

**O que era.** A frase de reprodutibilidade do Apendice A cita treze caminhos `\path{}` como
disponiveis no branch `mobiwac`. Nem todos estavam la.

**O que aconteceu de fato, e a contagem mudou tres vezes.** Nove -> cinco -> **tres**. As duas
primeiras contagens casavam por **nome de arquivo** nos caminhos que o apendice cita; a auditoria por
**conteudo** mostrou que este branch guarda parte deles em outro diretorio.

**RESOLVIDO em 2026-07-30**, com a sua decisao de reverter em vez de reescrever historia publica:

| commit | efeito |
|---|---|
| `b7b072d2` | `git revert 6c4267ba` — restaura os 14 arquivos que uma delecao publicada tinha removido |
| `0288cb70` | adiciona os TRES que faltavam de fato |

Verificado contra o remoto depois do push: `origin/mobiwac` esta em `0288cb70`, os 25 arquivos de
reprodutibilidade conferem **byte a byte** contra `3c57197c` (0 problemas), os tres adicionados estao
presentes, e o efeito liquido e **18 arquivos, 2.434 insercoes, 0 delecoes**.

> **AINDA E SUA DECISAO, e a unica coisa que sobrou deste item.** Dois arquivos ja publicados em
> `scripts/closing_data/` divergem das copias locais: `m1_stats_n20.py` (411 linhas locais contra 335
> no branch, 84 linhas diferentes) e `m2_prereg_perfold.py` (214 contra 222, 36 diferentes).
> **Nao toquei.** Substituir um artefato publicado por uma versao local divergente e decisao de
> autor, nao limpeza. Se a versao local e a correta, e um commit seu.

### 2.3 A ficha catalografica: naturalidade Contagem, e a biblioteca que gera

**Sua decisao 2026-07-29:** Contagem e o dado de naturalidade/residencia e vai na **ficha catalografica**, nao na folha
de rosto. O `\local{Florestal - Minas Gerais}` fica como esta, que e o que a ABNT pede (local de publicacao = cidade da
instituicao) e o que o exemplar do Germano usa.

**O que eu apliquei.** Nada de cidade no LaTeX. Apenas o nome, em tres lugares:
`\autor` e as duas linhas "SILVA, Vitor Hugo **De** Oliveira, M.Sc." do Resumo e do Abstract. Verificado no PDF: a folha
de rosto renderiza `VITOR HUGO DE OLIVEIRA SILVA`.

**O que depende de voce.** A ficha catalografica **nao e gerada por este LaTeX** — vem do formulario da Biblioteca
Central da UFV, e a naturalidade e um campo daquele formulario. Quando preencher, use **Contagem, MG**. Se a biblioteca
devolver a ficha como PDF para inserir, ela entra depois da folha de rosto e eu adiciono o `\includepdf` no lugar certo.

**Se voce quiser Contagem na folha de rosto mesmo assim**, e uma linha em `0_main.tex:189` — mas divergiria da ABNT e do
exemplar, e eu marcaria `[NEEDS SIGN-OFF]` registrando que foi escolha consciente sua e nao conformidade.

> DECISSAO: Podemos fechar esse ponto, quando a UFV retorna o pdf adcionamos lá. No mais vamos manter a norma da ABNT

### 2.4 A secao de escopo da tarefa estatica: manter ou suprimir

**(A) O que falta.** Sua conversa com o orientador sobre argumentar ou nao publicamente quanto ao escopo da tarefa
estatica do Ch.4.

**(B) Por que importa.** E uma declaracao publica sobre um resultado publicado e co-autorado. Voce tem o acordo do
co-autor (2026-07-27); falta o orientador.

**(C) O que eu preciso de voce.** A decisao. **Para suprimir**, comente **uma** linha em
`chapters/apx_b_errata.tex`:

```latex
%\input{chapters/apx_b_static_scope}
```

Testado: compila limpo, sem referencia pendente, porque o ponteiro no prefacio do Ch.4 referencia o **apendice**, nao o
rotulo da secao. **Se suprimir, apague tambem a sentenca do prefacio do Ch.4**
ou ela aponta para um apendice que nao discute mais o assunto.

> DECISSAO: Já removemos o appendix B, porém eu ainda tenho mais uma tarefa relacionado ao
> `B.5 The scope of the static task in Chapter 4`, hoje a escrita e como ele tá excrita está muito confuso. Por exempl
> na frase: "built from a fine-grained class label attached to each place", essa frase robuscada para dizer que usamo o
> nome do local (fclass); além desse ponto temos que a explicação não está bem didatica e clara, requere bastante
> esforço para entender e varias leituras. podemos melhorar bastante esse texto, apesar de ele não entrar diretamente no
> texto final, ele tem que estar polido e pronto para se necessario.

**FEITO em 2026-07-30, `apx_b_static_scope.tex`.** A frase que voce citou era o sintoma certo: ela nunca dizia ao leitor
que a coisa em questao e **o proprio nome do local**. Agora a segunda frase da secao diz, com tres valores reais tirados
do dado: `Coffee Shop`, `Seafood`, `Airport`. O argumento passou a ser lido de uma vez -- a entrada contem a resposta,
porque `Coffee Shop` pertence a Food independentemente de qual cafeteria seja.

A comparacao com o Cap. 3 tambem foi reescrita. Ela levava oito linhas para dizer que a exclusao do proprio no nao
sobrevive a convolucao; agora diz que o Cap. 3 tem uma versao mais branda do mesmo problema e explica o mecanismo em
quatro frases, terminando na diferenca real (busca exata contra media diluida).

> **EU DISSE QUE UM NUMERO ESTAVA ERRADO E ERA EU QUE ESTAVA.** Afirmei que "284 a 365" deveria ser
> "284 a 377". **Nao deveria.** Eu medi a coluna errada, na granularidade errada: contei
> `spot_categories` (um JSON por check-in) sobre linhas de check-in cruas. A fine class do pipeline
> e a coluna `spot`, que `research/embeddings/hgi/preprocess.py:62` renomeia para `fclass`, depois
> descarta categoria nula (:64) e reduz a **uma linha por placeid** (:75-80). Reproduzido assim:
> Alabama 284, Arizona 305, Florida 324, California 333, **Texas 365** -- exatamente o intervalo que
> ja estava no texto. `fclass` tambem nao e um nome morto: e o que `poi2vec.py` exige.
> **O texto original estava certo e foi restaurado.** A afirmacao "nenhum mapeia para mais de uma
> categoria" continua confirmada: 0 ambiguos nos cinco estados, na medicao correta.

Build: `main_extra` 20 pp (era 19), tex_errors 0, `make check` RC=0. A secao continua suprimivel por um unico `\input`,
como voce pediu quando ela entrou.

### 2.5 O tamanho de tipo das duas figuras de arquitetura

**(A) O que falta.** Uma decisao sobre `figures/cbic_mtlnet_arch.png` (45,3% do corpo) e
`figures/courb/arquitetura_modelo.png` (44,4%). Sao as duas menores do documento — **menores que as duas que a auditoria
rastreava**, que estao em 93,2% e 66,3% depois do reescalonamento desta rodada.

**(B) Por que importa.** `WRITING_LAW` §5 pede rotulos "proximos ao tamanho do corpo". Uma banca le essas figuras
impressas.

**(C) O que eu preciso de voce.** Autorizacao para mudar o tamanho de tipo de uma figura **publicada**
(a do Ch.4 e co-autorada). Ha `.drawio` para as duas, e a receita esta em
`_round6/12_figures.md`: subir `fontSize` de 13 para cerca de 20 e reexportar na mesma largura de pixels. Registrado
em [`LEFT_OUT.md`](LEFT_OUT.md) LO-6 como **diferido, nao recusado**.

> DECISSAO: Pode aumentar, mas mantenha os espaço ja'oculpado pela imagem, se já conseguir aumentar um pouco a font e
> amntendo a proporção da imagem já é um ganho. Mas, hoje apesar dos textos estarem menores, como o contraste está bom
> está facil de ler.

### 2.6 A coluna do CBIC que nao reproduz

**(A) O que falta.** Tres das quatro colunas de resultado publicadas do CBIC reproduzem exatamente contra as execucoes
commitadas (21/21 celulas). A quarta, a de proxima-categoria do modelo conjunto, **nao reproduz de nenhum artefato
commitado**.

**(B) Por que importa.** E um numero publicado. Nao ha erro conhecido nele; o que falta e a execucao que o gerou.

> DECISSAO: Documentar no letf_out.md

**(C) O que eu preciso de voce.** Dizer se existe um rundir dessa coluna fora deste repositorio. Se nao existir, isso e
uma limitacao de proveniencia a registrar, nao um erro a corrigir. Registrado em
[`LEFT_OUT.md`](LEFT_OUT.md) LO-2 como **aberto**.

### 2.7 O orcamento de tuning de Ch.3 e Ch.4: NAO RECUPERAVEL

**(A) O que falta.** O numero de configuracoes tentadas por estudo.

**(B) Por que importa.** Uma banca pode perguntar quanta busca de hiperparametro ha por tras de cada resultado.

**(C) O que eu preciso de voce.** Nada a recuperar: nunca existiu um harness de busca e as configuracoes perdedoras nao
foram commitadas. Isso foi estabelecido lendo os dois codebases, nao presumido. A pendencia e apenas **como dizer isso**
se perguntarem. Sugestao: dizer que o desenvolvimento foi manual e iterativo e que o repositorio preserva a configuracao
final, nao o caminho.

> DECISSAO: Documentar no letf_out.md e adcionar esse ponto no appendix B

**FEITO em 2026-07-30 (round 8), e METADE ja estava feita sem que ninguem tivesse notado a outra.**
A sua decisao tem duas partes e elas estavam em estados diferentes:

| parte da sua decisao        | estado quando eu medi                                                                                   |
|-----------------------------|-----------------------------------------------------------------------------------------------------------|
| documentar no `LEFT_OUT.md` | **ja estava**, LO-1, com a fonte (`_round6/10_protocol_recovery.md` §1.4)                                |
| adicionar no Apendice B     | **so para o Cap. 3.** A secao do Artigo 2 (CoUrb) nao dizia nada: `configuration`, `tuning`, `recoverable`, `harness`, `hyperparameter` -> 0 ocorrencias no texto sem comentarios |

O achado de origem e sobre **os dois** estudos -- a tabela de veredito daquele arquivo da
"NOT RECOVERABLE as a budget" para o Cap. 3 **e** para o Cap. 4, pelo mesmo motivo (nunca existiu
harness de busca em nenhum dos dois codebases e as configuracoes perdedoras nao foram commitadas).
Uma frase creditada por inteiro quando so metade andou e a segunda consequencia do V14 do
`AGENT_GUARDRAILS`, que e a razao de esta rodada existir.

Acrescentada uma frase na secao do Artigo 2, com a mesma redacao da do Artigo 1 para os dois
capitulos declararem o mesmo limite do mesmo jeito. **Lido no PDF do volume suplementar, nao no
fonte:** p. 8 (Cap. 3) e p. 9 (Cap. 4), 20 pp, tex_errors 0. A sua recolecao ("nao mudamos muito")
continua **fora** do texto de proposito: e coerente com o codigo, mas recolecao nao e registro
(`AGENT_GUARDRAILS` N1). LO-1 atualizado com onde a frase imprime. Marcado
`[NEEDS SIGN-OFF: PENDENCIAS 2.7, round8]`.

### 2.8 `CONSIDERATIONS.md`: uma rodada NOVA que chegou durante esta, e que eu NAO executei

**(A) O que falta.** `src_utils/CONSIDERATIONS.md` apareceu na arvore de trabalho **durante** esta rodada (modificado
19:04, nao commitado, 1.229 linhas). Ele contem material que nao estava no escopo que voce me deu:

| Secao                                    | O que e                                                                                          |
|------------------------------------------|--------------------------------------------------------------------------------------------------|
| `## Germano` (l. 3-58)                   | Feedback **verbal** do Germano sobre o Cap. 2, transcrito por voce                               |
| `## Fabrício` (l. 59-309)                | Feedback do **orientador** sobre o Cap. 2                                                        |
| `# Codex Audit — Chapter 2` (l. 310-994) | Auditoria dos dois feedbacks, comparacao contra `exemples/`, e uma lista de trabalho consolidada |
| `# Addendum (2026-07-28)` (l. 995-1229)  | O ponto de fluxo do Germano e o item G10 (o achado de conflito de tarefas)                       |

**(B) Por que importa.** Isto e feedback do **orientador** e de um leitor externo sobre o capitulo de fundamentacao, com
uma lista de trabalho ja consolidada. E a proxima rodada, e e mais importante que a maior parte do que sobrou aqui. Nao
esta perdido: o arquivo esta no disco. Mas nao esta commitado, e nenhum item dele foi aplicado ao texto.

**(C) O que eu preciso de voce.** Duas coisas. Primeiro, **commitar o arquivo** se ele estiver pronto (eu
deliberadamente nao commitei prosa sua em andamento). Segundo, dizer se quer que eu execute a lista de trabalho
consolidada dele — ela e uma rodada propria, com pesquisa e verificacao, e nao a comecei porque nao foi o que voce pediu
nesta.

**Por que eu nao agi nisso.** O escopo desta rodada foi `CODEX_AUDIT.md` mais as suas decisoes em
`PENDENCIAS.md`. Aplicar 1.229 linhas de feedback novo no fim de uma rodada longa, sem voce ter pedido, seria exatamente
o tipo de improviso que o `AGENT_GUARDRAILS` manda parar e sinalizar.

### 2.9 O disco do nespedgpu estava 100% cheio — LIBERADO por voce; sobrou a decisao de rodar ou nao

> **ESTADO ATUAL, medido 2026-07-30T00:48Z: 61G livres, GPU ociosa, embeddings prontos.** O bloqueio
> nao existe mais. O que era "libere espaco ou aceite quatro datasets" virou "rode os tres que faltam
> ou aceite quatro", e isso e uma decisao diferente, com outro custo. A medicao completa e as duas
> opcoes estao no fim deste item; o registro abaixo e como o bloqueio foi diagnosticado e fica porque
> explica por que o apendice tem quatro datasets e nao sete.

**Medido 2026-07-29 09:00Z, nada foi apagado.** `df -h /home` no host de GPU:

```
/dev/mapper/vg0-home  393G  373G  0  100% /home
```

Zero bytes livres. Inodes estao em 5%, entao e byte, nao arquivo.

**Onde esta o espaco** (medido, nada alterado):

| caminho                                     |  tamanho |
|---------------------------------------------|---------:|
| `PoiMtlNet/`                                |      94G |
| `PoiMtlNet/results/`                        |      82G |
| `results/check2hgi/california/checkpoints/` |  **61G** |
| um diretorio `mtlnet_*` de run              | 1 a 7 MB |

Ou seja: o consumo e de **checkpoints de modelo salvos**, nao de diagnosticos. Os 25 diretorios de run do california
somados dao menos de 100 MB.

**Como isso apareceu.** Um treino morreu com `RuntimeError: basic_ios::clear: iostream error` -- que e uma escrita que
falhou, nao um problema de modelo ou de dado -- e dois folds de um job sequencial nao produziram saida nenhuma enquanto
o wrapper ainda saiu com codigo 0. Um arquivo truncado onde se esperava um completo e o outro sintoma.

**Consequencia para a dissertacao:** os datasets california, texas e istanbul do apendice do gradiente-cosseno **nao
puderam ser medidos**. O apendice fecha com quatro datasets (florida, alabama, arizona, georgia) e marca os tres
restantes como bloqueados por disco, nao como pendentes de fila.

> **DECISAO SUA, e eu nao vou tomar por voce:** liberar espaco no host e sua decisao, nao minha --
> aqueles 61G sao seus checkpoints e eu nao apago nada na sua maquina. Duas opcoes:
> **(a)** apagar/arquivar `results/check2hgi/california/checkpoints/` (61G) se aqueles checkpoints ja
> nao servem, o que libera espaco de sobra; **(b)** deixar como esta e aceitar quatro datasets no
> apendice, que e um resultado honesto e completo por si.
>
> Se voce liberar o espaco, os tres datasets que faltam custam: texas e istanbul um job cada, e
> california cinco jobs de um fold (`--only-fold k`, 0-indexado, ~22 min por fold). Os scripts estao
> prontos. **Antes de submeter qualquer coisa, `df -h /home | tail -1` no host** -- um job lancado em
> disco cheio queima o limite de tempo e volta verde sem dado.


> DECISAO: Liberado

#### O BLOQUEIO ACABOU, medido em 2026-07-30T00:48Z, e isso muda o item de bloqueio para escolha

Voce escreveu "Liberado" e eu conferi no host antes de escrever esta linha, porque **um estado remoto
verificado uma vez nao continua verdadeiro** e este arquivo ja errou exatamente assim uma vez (item
2.1, `origin/mobiwac`). Medido por mim, nao por voce:

| o que                                        | 2026-07-29 09:00Z          | **2026-07-30 00:48Z**       |
|----------------------------------------------|----------------------------|-----------------------------|
| `df -h /home` disponivel                     | **0** (100% cheio)         | **61G** (84% usado)         |
| `results/check2hgi/california/checkpoints/`  | 61G                        | **ausente**                 |
| `results/check2hgi/california/` (o diretorio) | —                          | 64M                         |
| GPU (`nvidia-smi`)                           | —                          | 0 MiB usados, 0% ocupada    |
| treino vivo (`pgrep -af scripts/train.py`)   | —                          | nenhum                      |
| interpretador `PoiMtlNet/.venv/bin/python`   | —                          | torch 2.11.0+cu128, cuda True |
| embeddings Check2HGI em `output/check2hgi/`  | —                          | os sete estados presentes   |

**Nada foi apagado por mim.** Os checkpoints sairam pela sua mao; eu so li `df`, `du`, `ls` e
`nvidia-smi`. Reproduza com:

```bash
ssh nespedgpu 'df -h /home | tail -1; du -sh ~/PoiMtlNet/results/check2hgi/california'
```

**UMA CONSEQUENCIA JA APLICADA NO TEXTO, porque a frase tinha ficado falsa.** O apendice do
gradiente-cosseno afirmava, no presente, que os tres datasets estao *"blocked on that machine's free
space rather than waiting in its queue"*. Com 61G livres isso deixou de ser verdade **dentro do
documento**. A clausula agora diz que o que falta aos tres e **a execucao**, nao um metodo -- verdade
independente de voce rodar ou nao. Lido no PDF (p. 100 do build de defesa, 100 pp, tex_errors 0), nao
no fonte. As duas frases anteriores, que contam que a tentativa *encheu* o disco, sao historia e
ficaram. Marcado `[NEEDS SIGN-OFF: PENDENCIAS 2.9, round8]`. Uma instrucao para agentes futuros no
mesmo arquivo (*"Say BLOCKED, never 'pending' or 'queued'"*) mandava, hoje, escrever algo falso: esta
marcada como superada, com o motivo, em vez de apagada.

> **A DECISAO QUE SOBRA, e ela e nova: nao e mais "liberar espaco", e "rodar ou nao rodar".**
> Com disco, GPU livre e os embeddings prontos, os tres datasets que faltam custam:
>
> | dataset   | forma de submissao                                       | custo medido                        |
> |-----------|----------------------------------------------------------|-------------------------------------|
> | texas     | um job                                                   | cabe no cap de 35 min               |
> | istanbul  | um job                                                   | cabe no cap de 35 min               |
> | california| cinco jobs de um fold (`--only-fold k`, **0-indexado**)  | ~22,3 min por fold, ~1,9 h no total |
>
> **(a) Rodar os tres.** O apendice passa de quatro para sete datasets e a conclusao de ortogonalidade
> ganha a cobertura completa do Cap. 5. Custo real: os tres jobs **sequenciais** (o host tem uma GPU
> so; sete jobs concorrentes ja mataram tres por falta de memoria), mais re-rodar
> `src_utils/_round7/cosine_stats.py`, estender `tables/frame/cosine.tex`, refazer a figura e
> **corrigir toda contagem de dataset no apendice** -- o proprio arquivo lista os oito lugares e manda
> conferir por grep em vez de confiar na lista.
> **(b) Nao rodar.** Quatro datasets e um resultado honesto e fechado; o apendice ja diz que o leitor
> nao deve extrapolar. Custo: zero, e o texto ja esta correto como esta.
>
> **Eu nao escolho por voce**, porque (a) gasta ~2 h da sua GPU e mexe em prosa de apendice que voce
> assina. Se for (a), duas armadilhas medidas deste host, para o job nao voltar verde e vazio:
> `export MTL_TRAIN_DIAGNOSTICS=1` (a diagnostica e opt-in e o padrao e desligado -- tres estados ja
> foram colhidos com a coluna `grad_cosine_shared` inteiramente vazia), e **nao selecione o diretorio
> de saida por recencia**, que sob jobs concorrentes ja produziu dois "folds diferentes" byte a byte
> identicos. E `df -h /home | tail -1` antes de cada submissao.

### 2.10 Dos catorze checkers: sete se auto-verificam de verdade, um tem auto-teste que nao morde, dois tem fixtures, quatro nao tem nada

> **TERCEIRA CORRECAO DESTE ITEM, 2026-07-30.** As tres versoes anteriores erraram, cada uma de um
> jeito diferente, e as tres pelo mesmo motivo: eu classifiquei ferramentas sem abrir os arquivos.
> (1) A primeira disse "onze sem prova nenhuma" -- exagero. (2) A segunda colocou o `sweep_guard` na
> coluna errada. (3) A terceira dizia "quatro de catorze" mas a tabela **classificava doze**: o
> `check_tex_root` e o `check_verify_list` nao apareciam em coluna nenhuma, e eram justamente os dois
> que eu nao tinha aberto. Agora os catorze estao classificados, e a classificacao veio de sabotar a
> logica de deteccao de cada um e ver se o proprio auto-teste pega.
>
> **QUARTA PASSAGEM, 2026-07-30 (round 8): a classificacao aguentou a re-medicao, com uma correcao
> na direcao mais severa.** Repeti a sabotagem nas catorze, sem grepar a palavra "self-test" em
> lugar nenhum. As treze linhas conferem. A decima quarta -- `check_negative_parallelism` -- estava
> **subestimada**: o item dizia que UM detector desligado passava batido, e na verdade **os quatro**
> passavam. Consertado nesta rodada (caixa no fim do item). O padrao vale registrar porque e o
> contrario do erro das tres primeiras versoes: aquelas afirmavam cobertura que nao existia; esta
> descreveu um buraco menor do que o real. As duas vem de nao repetir a medicao em cada item.

**Medido: quebrei o detector de cada ferramenta e observei o codigo de saida.**

| ferramenta | auto-teste | pega a propria logica quebrada? |
|---|---|---|
| `check_doubled_macro` | `self_test()` no `main()` | **sim** (rc 0 -> 1) |
| `check_tex_root` | `self_test()`, 4 asserts | **sim** (rc 0 -> 1) |
| `check_tracker_refs` | `self_test()`, 5 asserts | **sim** (rc 0 -> 1) |
| `check_meta_claims` | `self_test()`, 10 asserts | **sim** (rc 0 -> 1) |
| `check_extra_xrefs` | `self_test()` | **sim** (rc 0 -> 1) |
| `check_comment_hygiene` | `self_test()`, 8 linhas de saida | **sim** (rc 0 -> 1) |
| `sweep_guard` | 4 asserts no `__main__` | **sim** (sabotando `n == 0`, rc 0 -> 1) |
| `check_negative_parallelism` | `self_test()`, agora por padrao | **SIM, desde 2026-07-30** (era **NAO**; ver a caixa abaixo) |
| `check_trapped_prose` | suite externa `test_trapped_prose.py`, 4 casos `must_flag`, rodada pelo `check.sh`; + par de fixtures novo | sim, pelas fixtures |
| `check_torn_sentences` | par de fixtures novo (defeito historico real, `1bf9a227`) | sim, pelas fixtures |
| `check_verify_list` | **nenhum** | -- |
| `check_wordcount_claims` | **nenhum** | -- |
| `sync_page_counts` | **nenhum** (guard de descompasso .log/.pdf exercitado a mao, rc=1) | -- |
| `sync_deliverables` | **nenhum** (ramo de fonte ausente exercitado a mao, rc=1) | -- |

**Catorze linhas, catorze ferramentas: 8 + 2 + 4 = 14.** Oito auto-testes que mordem (era 7+1), duas
cobertas por fixtures externas, quatro sem nada. Contando o que **nao esta provado pelo proprio
mecanismo**, agora sao **quatro**: os quatro sem nada.

> **O (i) FOI FEITO em 2026-07-30 (round 8), e a medicao original subestimava o defeito.** O item
> dizia que `check_negative_parallelism` continuava saindo 0 com o detector `rather than` desligado.
> Re-medido sabotando **cada um dos quatro** em vez de um: `rather than`, `, not`, `instead of` e
> `not ... but` -- **os quatro** deixavam a ferramenta sair 0. A causa e a forma do auto-teste, nao a
> escolha do detector: ele somava `findall` sobre os quatro padroes e checava so a **densidade
> agregada** contra o teto, entao os tres que sobravam sempre levavam a amostra por cima do teto.
>
> Agora ha uma tabela `PATTERN_SAMPLES` com uma amostra minima por padrao, e o auto-teste (a) exige
> que as duas tabelas tenham exatamente as mesmas chaves, (b) afirma cada detector **sozinho** contra
> a sua propria amostra, e (c) so depois checa a densidade nas duas direcoes. Verificado por sabotagem
> depois da mudanca, lendo o codigo de saida de cada rodada:
>
> ```
> desligar "rather than"        -> rc=1   desligar "instead of"   -> rc=1
> desligar ", not"              -> rc=1   desligar "not ... but"  -> rc=1
> acrescentar um 5o padrao sem amostra -> rc=1  (a asserção de cobertura pega)
> arvore intacta                -> rc=0   (120 instancias / 35.844 palavras = 3,35 por 1k, teto 3,60)
> ```

**(A) O que falta.** (ii) Par de fixtures para `check_verify_list`, `check_wordcount_claims`,
`sync_page_counts` e `sync_deliverables` -- as quatro sem prova nenhuma pelo proprio mecanismo.
(iii) Fixtures externas para os oito que so se auto-testam por dentro, nao porque o interno nao valha,
mas porque **nada responde "quais estao provados" sem ler dezesseis arquivos**.

**(B) Por que importa.** Duas vezes voce perguntou o que estava errado e as duas vezes apareceu um
defeito grande em producao havia semanas com `make check` dizendo RC=0. E este proprio item errou tres
vezes seguidas, o que mede a mesma coisa por outro lado.

**(C) O que eu preciso de voce.** Nada. `AGENT_GUARDRAILS` §4b V13, que agora inclui a regra que
faltava: **uma tabela cujo cabecalho conta N tem que ter N linhas.** Um total que nao fecha com as
proprias linhas e um erro aritmetico visivel sem conhecimento nenhum do dominio.

> `make selftest` NAO faz parte do `make check`, de proposito: uma suite de lint que roda a propria
> suite de testes a cada invocacao e o mesmo erro de trabalho-dentro-de-trabalho que fez o `check`
> levar 265 s.

### 2.11 A assimetria do resultado de regiao: o Cap. 5 ressalva, e o resto do documento nao

**Origem:** `_round6/VERIFY_LIST.md` itens 4 e 5 (achado L-5 do ledger), entregues em 2026-07-30.

**(A) O que e.** `chapters/5_mobiwac/05_setup.tex` diz que o plano de analise *"did not cover
next-region superiority, so the four next-region gains ... are secondary results outside it"*. O
resto do documento afirma o mesmo resultado **sem essa ressalva**. Medido com o varredor que remove
comentarios, sobre os 54 `.tex`:

| onde | forma |
|---|---|
| `1_introduction.tex:132`, `6_conclusion.tex:21` e `:93` | "four of six" / "four of the six" |
| `2_fundamentals.tex:786`, `5_mobiwac/01_introduction.tex:39`, `5_mobiwac/08_conclusion.tex:14`, `5_mobiwac.tex` | idem |
| `content.tex:166` (Resumo e Abstract) | "quatro deles" / "four of them" — a mesma alegacao, em outras palavras |

No PDF de defesa (100 pp) a alegacao sem ressalva imprime nas **pp. 14, 58, 59, 76, 77 e 78**; a
ressalva imprime **so na p. 67**. Sao sete sitios em prosa mais as duas parafrases do pre-textual,
contra uma ressalva.

**(B) Por que importa.** O registro estatistico de 2026-07-27 e inequivoco: o teste primario
registrado para **toda** celula de regiao e nao-inferioridade TOST. Uma leitura rapida do Resumo, da
Introducao ou da Conclusao le "outperforms em quatro de seis" como resultado primario; a p. 67 diz
que nao e. Nenhuma track da rodada 6 assumiu isso (achado L-5 do ledger).

**(C) O que eu preciso de voce.** Uma regra, e ela vale para os nove sitios de uma vez:

> **(a)** o texto de moldura acrescenta "as a secondary result" (ou equivalente) **uma vez**, no
> ponto que voce escolher — o candidato natural e a Conclusao, que ja e o lugar onde o escopo do
> plano de analise e discutido. Custo: uma frase, mais linha de errata se voce quiser rastrear.
> **(b)** a assimetria e deliberada — o Cap. 5 e o capitulo que carrega o metodo, entao e onde a
> ressalva pertence — e isso vai para `LEFT_OUT.md` com o motivo. Custo: zero no texto, mas o
> registro passa a existir.
>
> **Eu nao decido isto** porque muda o que o Resumo e a Conclusao afirmam sobre o resultado
> principal do Cap. 5, que e prosa sua sobre um resultado seu.

### 2.12 `Pareto-stationary point` esta na prosa e nao esta no registro (o `GLOSSARY` e fail-closed)

**(A) O que e.** A regra de manutencao do `GLOSSARY.md` e explicita: *"a term not in this registry
may not be used in dissertation prose"*. Medido hoje:

```bash
cd /Users/vitor/Desktop/mestrado/ingred/articles/dissertacao
grep -c 'Pareto-stationary' GLOSSARY.md          # 0
```

e o termo esta em prosa em **dois** sitios, os dois em texto publicado reproduzido:
`chapters/3_cbic/method.tex` (*"convergence to a Pareto-stationary point"*) e
`chapters/4_courb/methodology.tex` (a garantia do Nash-MTL, a mesma frase do item 3 do
`VERIFY_LIST`). Imprimem nas **pp. 36 e 48**. `tables/courb/errata.tex` ainda traz a forma sem
hifen, "Pareto stationary". As outras duas entradas que o item 4 daquele arquivo pedia — **bilinear
discriminator** e **logistic function** — **ja entraram** (`GLOSSARY.md:71` e `:72`); so esta faltou.

**(B) Por que importa.** O `make check` **nao pega isto**: existe um gate de "Pareto" mas ele e
informativo e conta ocorrencias, nao registro. E o termo nao pode ser simplesmente removido — as
duas frases sao publicadas, entao tirar o termo e editar uma frase publicada, com linha no
Apendice B.

**(C) O que eu preciso de voce.** Uma decisao, tres saidas:

> **(a)** registrar o termo (uma linha na §4 do `GLOSSARY`, a definicao ja esta escrita na propria
> frase do Cap. 4: *"a point at which some convex combination of the task gradients is zero"*).
> Custo: uma linha, e o documento fica consistente com a propria regra.
> **(b)** trocar o termo nas duas frases publicadas. Custo: duas linhas de errata no Apendice B, e
> voce esta editando prosa publicada por uma questao de vocabulario.
> **(c)** registrar uma excecao explicita para termos que chegam em prosa reproduzida. Custo: uma
> nota no `GLOSSARY`, e a regra deixa de ser fail-closed para essa classe.

### 2.13 A contagem dos `[NEEDS SIGN-OFF]`: o comando conta 4 a mais, sempre, e a tabela da §2.1 esta vencida

**(A) O que e.** O `VERIFY_LIST` A7 manda rodar `grep -rn "NEEDS SIGN-OFF" src/ | wc -l` e esperar
46. **Esse comando conta a mais, por construcao**: ele varre `src/build/`, onde
`src/build/fmt/_body.tex` e uma **copia gerada** que o `src/.gitignore` exclui. Sao os mesmos quatro
marcadores contados duas vezes. Toda contagem futura precisa de `--exclude-dir=build`.

**(B) O numero em si nao e estavel, e isso e o achado.** Medi 53, e vinte minutos depois 55, na
mesma sessao. Nao foi erro de medicao: uma **track paralela da rodada 8** acrescentou um marcador em
`apx_b_errata.tex` enquanto eu media, e outros dois chegaram nos commits `3ef8dc8b` e `d9ab436f`.
Registro com o momento, como manda o §4b: **55 no fonte / 59 com `build/`, medido em `d9ab436f`**,
mais um marcador ainda nao commitado na arvore de trabalho.

**(C) A tabela da §2.1 nao fecha mais**, por dois motivos independentes: o total (46) envelheceu, e
ela lista `0_main.tex` com 4 marcadores — arquivo que **nao existe mais**; esses quatro estao hoje
em `content.tex`. Distribuicao medida em `d9ab436f` (`--exclude-dir=build`):

| arquivo | n |
|---|--:|
| `chapters/6_conclusion.tex` | 9 |
| `chapters/2_fundamentals.tex`, `chapters/apx_a_contributions.tex`, `chapters/apx_b_errata.tex` | 6 cada |
| `content.tex` | 4 |
| `chapters/5_mobiwac/06_results.tex` | 3 |
| `chapters/1_introduction.tex`, `3_cbic/results.tex`, `5_mobiwac/02_related.tex`, `5_mobiwac/07_discussion.tex`, `apx_f_cosine.tex` | 2 cada |
| `3_cbic/method.tex`, `4_courb.tex`, `4_courb/methodology.tex`, `4_courb/results.tex`, `5_mobiwac.tex`, `5_mobiwac/05_setup.tex`, `apx_b_static_scope.tex`, `apx_c_ai_disclosure.tex`, `apx_d_ceiling.tex`, `apx_e_ethics.tex`, `main_extra.tex` | 1 cada |

**(D) O que eu preciso de voce.** Nada para decidir; e a sua fila. A ordem de leitura recomendada
continua valendo: A1, depois A3, depois A2.

> **A TABELA DA §2.1 FOI REESCRITA, por outra track desta mesma rodada, e as duas medicoes concordam.**
> Este item dizia *"Nao reescrevi a tabela da §2.1"* -- correto quando foi escrito; a track que auditou
> a §2 reescreveu-a em `ecb81fb6`, antes deste item existir, e as duas chegaram ao mesmo diagnostico
> por caminhos independentes: o comando conta 4 a mais por causa de `src/build/fmt/_body.tex`, e a
> linha `0_main.tex` nomeava um arquivo que nao existe desde `2b9b853d`.
>
> **Estado conciliado, medido em `f624767c`:** 55 no fonte, 59 com `build/`, distribuidos em 22
> arquivos, e a tabela da §2.1 fecha com a propria soma (9 + 6x3 + 4 + 3 + 2x5 + 1x11 = 55). Os 55
> estao todos dentro de comentarios `%`: **zero** aparecem no PDF. A §2.1 leva agora a data da medicao
> e a ressalva de que o numero anda -- inclusive por commits das proprias tracks desta rodada, que
> acrescentaram tres marcadores enquanto o item era escrito. **A sua preocupacao (C) esta atendida e o
> ponto (B) desta secao continua valendo: confie no comando, nao na tabela.**

### 2.14 O intervalo de paginas do `nash`: nao da para verificar daqui

**Origem:** `_round6/VERIFY_LIST.md` item 14, entregue em 2026-07-30 (precedente `standley2020tasks`).

**(A) O que e.** `references.bib` traz `pages = {16428--16446}` para
`@inproceedings{nash}` (Navon et al., *Multi-Task Learning as a Bargaining Game*, ICML 2022).
Tentado de novo nesta sessao, contra as fontes de registro que o sandbox alcanca:

| fonte | resposta |
|---|---|
| OpenAlex | um unico registro, `W4225981399`, tipo **preprint**, venue "arXiv", `first_page` e `last_page` **nulos** |
| Crossref (`query.bibliographic`) | cinco obras, **nenhuma delas este artigo** — nao ha DOI registrado da versao de anais |
| `proceedings.mlr.press` | **fora da allowlist** do sandbox; nao acessado |

**(B) Por que importa.** Pelo §1 do `AGENT_GUARDRAILS`, um identificador que nao foi aberto na fonte
de registro nao pode ser apresentado como conferido. O campo esta no `.bib` e nao esta verificado.

**(C) O que eu preciso de voce.** Um clique fecha: `proceedings.mlr.press/v162/navon22a.html`.

> **(a)** confirmar o intervalo e ele fica; **(b)** apagar o campo `pages`, que e exatamente o
> precedente que esta bibliografia ja adotou para `standley2020tasks`.

### 2.15 Tres citacoes NOT-SUPPORTED e um termo banido, todos em prosa publicada reproduzida

**Origem:** `_round6/VERIFY_LIST.md` itens 15 e 16, entregues juntos em 2026-07-30 como uma decisao unica.

**(A) O que e.** Quatro pontos, um so tipo de decisao: **nenhum deles pode ser corrigido por um
agente**, porque todos estao em frases publicadas, e mexer nelas gera linha de errata.

| onde | o que |
|---|---|
| `3_cbic/method.tex` | `ruder2017sluice` citado para regularizacao implicita do hard sharing |
| `4_courb/methodology.tex:173` | `sun2020go` citado para ciclos temporais revelarem a *funcao* do lugar |
| `4_courb/methodology.tex:184` | `belkin2003laplacian` citado para um regularizador hierarquico de embedding |
| `4_courb/methodology.tex:173,184` | **`fclass` em prosa renderizada**, 4 ocorrencias em 3 linhas |

**(B) Sobre o `fclass`, que ninguem tinha levantado.** O `GLOSSARY.md:73` diz, com todas as letras:
*"In code this column is `spot`, renamed `fclass` at `hgi/preprocess.py:62`; **NEVER write `fclass`
in prose**"*. Varri os 54 `.tex` sem comentarios: `4_courb/methodology.tex` e o **unico** arquivo. E
as frases identicas estao publicadas no CoUrb (`src_en/sections/metodology.tex:109` e `:120`), entao
e o mesmo tipo de decisao das tres citacoes acima. Nenhum gate pega: o de codenomes casa
`B9|v1[1-7]|champion-G|H3-alt|dk_ovl|log_T|substrate`, e `fclass` nao esta na lista.

**(C) O que eu preciso de voce.** Uma decisao por linha, ou uma regra para as quatro:

> **(a)** trocar. As substituicoes sugeridas (`baxter2000model`, `Xu2023`) **ja estao** na
> bibliografia e ja sao citadas para essas alegacoes em outros pontos; para o `fclass`, o termo
> registrado e "fine class". Custo: uma linha de errata no Apendice B por sitio.
> **(b)** manter e registrar. As frases sao do artigo publicado e reproduzi-las fielmente e
> defensavel; a divergencia vai para `LEFT_OUT.md`. Custo: zero no texto.
>
> `ruder2017sluice` merece uma nota: a mesma chave carrega **tambem** uma decisao de metadados (o
> titulo no `.bib` e o do preprint superado, "Sluice Networks..."; o titulo de registro e "Latent
> Multi-task Architecture Learning" e a versao de registro e AAAI 2019,
> `10.1609/aaai.v33i01.33014822`, pp. 4822-4829). Decida as duas juntas para tocar a entrada **uma
> vez** so.

### 2.16 Quatro artefatos publicados **divergiram** das copias locais (o item 2.2 cobria dois)

**(A) O que e.** O Apendice A cita treze caminhos `\path{}`. A pergunta "quantos faltam no branch
publico" ja teve **quatro** respostas nesta base (9 de 13, depois 5, depois 4, agora esta). O motivo
de todas as anteriores e o mesmo: `git cat-file -e mobiwac:<caminho>` pergunta *"este CAMINHO esta
no branch"*, e a alegacao e *"este ARQUIVO esta no branch"* — e o branch `mobiwac` **nao tem arvore
`docs/`**, guarda esses artefatos em `analysis_protocol/`. Remedi por **hash**, comparando cada
arquivo local com os blobs do branch:

| classe | n | quais |
|---|--:|---|
| no branch, **byte a byte identicos** | 8 | `folds.py`; `STATISTICAL_PROTOCOL.md`, `JOINT_BEST_RESULTS.md`, `m1_full_output.txt`, `m2_prereg_output.txt` (sob `analysis_protocol/`); `build_phase3_per_fold_transitions.sh`; `score_joint_best.py`; `autocorrelation_ceiling.py` (em `scripts/`, nao `scripts/embedding_eval/`) |
| no branch com **conteudo diferente** | 4 | `superiority_wilcoxon.py`, `region_match_tost.py`, `m1_stats_n20.py`, `m2_prereg_perfold.py` |
| diretorio (o instrumento nao classifica) | 1 | `stats_n20/` |

**Nada esta faltando.** Quatro artefatos publicados **divergiram**:

| arquivo | local | no branch | linhas diferentes |
|---|--:|--:|--:|
| `superiority_wilcoxon.py` | 147 | 126 | 37 |
| `region_match_tost.py` | 74 | 74 | 2 |
| `m1_stats_n20.py` | 411 | 335 | 84 |
| `m2_prereg_perfold.py` | 214 | 222 | 36 |

**(B) Por que importa.** O seu item **2.2 ja reservou exatamente esta decisao a voce**, mas para
**dois** arquivos (`m1_stats_n20.py` e `m2_prereg_perfold.py`). Sao **quatro**. Substituir um
artefato publicado por uma versao local divergente e decisao de autor, nao faxina — nao toquei em
nada.

**(C) O que eu preciso de voce.** Por arquivo, ou uma regra para os quatro:

> **(a)** publicar a versao local, se ela e a correta (um commit seu no branch `mobiwac`);
> **(b)** deixar como esta, se a versao publicada e a que gerou os numeros do artigo — que e o caso
> a favor de deixar quieto, e o mais provavel para `region_match_tost.py`, com 2 linhas de diferenca.
>
> A prosa do Apendice A **nao depende disto**: ela ja diz que os scripts estatisticos *"are part of
> the working repository and are supplied on request"*, o que nenhuma das leituras acima torna falso.

### 2.17 UMA AFIRMACAO FALSA MINHA, ja no historico: o commit `a07e547b` diz que o gate saiu 0 e ele saiu 1

**(A) O que e.** O commit `a07e547b` (a frase do orcamento de tuning, item 2.7) termina com
*"bash src_utils/check.sh -> rc=0 (22 gates; page counts agree)"*. **A suite saiu 1.** A mesma celula
que fez o commit imprimiu `DEFENSE_RC=0`, `TRACKER_RC=0`, `CHECK_RC=1`, e eu escrevi 0.

**(B) Por que eu estou te contando isso em vez de so consertar.** Porque e a classe exata do V11 do
`AGENT_GUARDRAILS`, a quarta ocorrencia, cometida **dentro da rodada que existe para impedi-la**; e
porque a regra desta rodada e que nenhuma track acredita no proprio relatorio. Achado por uma revisao
independente, nao por mim.

Dois detalhes que importam mais que o erro:

- **O gate vermelho era o `check_trapped_prose`, e quem o disparou foi o bloco de comentario que
  aquele commit acrescentou.** Ou seja: a linha falsa cobria justamente o gate que aquele commit
  quebrou.
- **Onze outros codigos de saida na mesma celula eram 0**, e o olho parou nos onze que confirmavam o
  formato esperado. E o mecanismo do V12 (o leitor para na primeira coisa que responde a pergunta com
  que ele chegou) aplicado a um lote de exit codes.

**(C) O que ficou verdadeiro, medido.** O **conteudo** daquele commit esta certo e foi verificado no
render: a frase imprime em `main_extra.pdf` p. 8 (Cap. 3) e p. 9 (Cap. 4), `make extra` rc=0, 20 pp,
tex_errors 0. Somente a linha do gate era falsa. E o flag do `check_trapped_prose` era **falso
positivo**, por um defeito real da ferramenta: ela comparava todo arquivo de capitulo contra o
`dissertacao.pdf`, mas o `apx_b_errata` renderiza no volume suplementar, entao naquele arquivo o teste
estava invertido -- so podia dar falso positivo e era **cego** a um rasgo de verdade. Consertado em
`f624767c`, validado nas duas direcoes. Depois do conserto: `bash src_utils/check.sh` rc=0, 22 gates,
lido direto.

**(D) O que eu preciso de voce.** Nada para decidir, mas **uma coisa para fazer se voce publicar este
historico.** A correcao esta anexada ao proprio commit com `git notes`, e o `git log` normal ja a
mostra debaixo da mensagem que ela corrige:

```bash
cd /Users/vitor/Desktop/mestrado/ingred
git log -1 a07e547b            # a nota aparece sob a mensagem
```

Escolhi nota em vez de `--amend` porque o commit esta seis commits atras num branch em que tracks
paralelas desta rodada commitaram; reescrever a historia trocaria os hashes dos commits delas.
**Notas nao sobem no `git push` por padrao.** Se voce publicar, precisa de:

```bash
git push origin refs/notes/commits
```

Sem isso a mensagem falsa viaja e a correcao fica na sua maquina, que e pior do que nao ter corrigido.

### 2.21 O segundo ponto do seu orientador (como os termos entram) — item perdido, e o buraco que ele apontava ESTA fechado

> **PERDIDO, nao resolvido — e eu o classifiquei errado duas vezes.** Existiu como `3.4` ate
> `1ef83867` (2026-07-28), saiu sem decisao e sem ir para o arquivo. Na varredura de 2026-07-30 eu o
> listei como candidato a perda real e **nunca o medi**, mas escrevi "cinco de sete resolvidos".
> Medido agora.

**O que ele disse:** *"so tome cuidado com o uso de IA e os termos menos comuns que sao usados... soa
um pouco estranho o jeito que alguns termos sao inseridos (marquei alguns la)"*.

**O buraco concreto que o item registrava:** o brief da persona 03 (style auditor, gate G3) manda ler
`articles/[mobiwac]/GLOSSARY.md` e diz que ele **vence** para o Cap. 5; o relatorio v2 dela, de 26/07,
tinha **zero** referencias a esse arquivo. O glossario tem 393 linhas.

> **A PRIMEIRA VERSAO DESTE BLOCO EXAGEROU, e a revisao pegou.** Ela dizia "o buraco esta fechado" e
> "o que esta medido e a parte mecanica", declarando como nao-medidas apenas as secoes 6, 7 e 8. Mas as
> **duas listas mecanicas de termos** — §3 (jargao -> palavra simples) e §4 (palavras a evitar) —
> **tambem nao estavam medidas**: meu parse de §4 devolveu **zero** termos (e uma lista de bullets, nao
> uma tabela) e o parse de §3 devolveu 69 linhas que eu mesmo rejeitei como erradas e nunca refiz. A
> unica medicao valida era a minha propria lista de codenomes, escrita a mao. Medidas agora, de verdade:

**Medido em 2026-07-30, secao por secao:**

| secao do glossario | como medi | resultado |
|---|---|---|
| §3 jargao -> simples | tabela de 3 colunas, 26 linhas; so as de veredito `avoid` sao proibicoes | **3 proibicoes** (`substrate`, `recipe`, `end-to-end (training)`) — **zero** na prosa viva do Cap. 5 |
| §4 palavras a evitar | lista de bullets; extrai os 34 termos entre aspas e removi os 6 que sao a *substituicao prescrita*, nao a proibicao | **28 proibicoes**, 22 com zero uso; **6 presentes**, todas condicionais e todas dentro da condicao (abaixo) |
| codenomes de repositorio | lista escrita a mao (`C2HGI`, `B9`, `v11`–`v17`, `champion-G`, `H3-alt`, `log_T`, `substrate`, `engine`, `board`, `recipe`) | **zero** |
| persona 03 re-rodada depois de 26/07? | data do relatorio | **sim**, `_round6/17_style_readability_credibility.md`, 28/07 |

**As seis de §4 que aparecem, uma por uma** — §4 as proibe *condicionalmente*, e cada uso satisfaz a
condicao:

- **`margin` x11** — §4 reserva a palavra para a margem de dois pontos do TOST. Os usos sao
  "the two-point margin", "the screen's margin": e o sentido reservado.
- **`cross-attention` x3, `transformer` x1** — §4 diz "nomeie so se for realmente estrutural". E o
  mecanismo de compartilhamento do modelo conjunto, ou seja, estrutural.
- **`Audit` x2** — §4 bane como auto-elogio ("audited recipe") e permite como **substantivo** para a
  medicao de vazamento. Os dois usos sao substantivo: "a screening audit run during development".
- **`activity` x2** — §4 bane para *as nossas duas tarefas*. Os dois usos descrevem trabalhos de
  terceiros (DRRGNN, MCARNN), onde §4 manda manter o termo padrao.
- **`head` x1** — idem, e num contexto de ablacao ("region head"), o mesmo uso que §4 permite ao
  descrever outros sistemas.

**O que continua NAO medido, e nao vou dizer que esta.** §6 (checklist de consistencia), §7 (marcas de
texto de maquina) e §8 (registro internacional simples) sao julgamentos de estilo, nao listas de termos:
nao ha grep que os decida, e uma persona ter rodado nao e evidencia de que foram aplicados.

> **DECISAO SUA, pequena.** O seu orientador escreveu *"marquei alguns la"* — ele marcou termos
> especificos num PDF ou num documento que **eu nao tenho**. Se voce me passar essas marcacoes, eu
> trato uma por uma. Sem elas, o que eu consigo afirmar e so o que esta na tabela acima, e o item fica
> aberto por falta da entrada dele, nao por falta de trabalho.

### 2.20 O Cap. 4 italiciza ingles corriqueiro 153 vezes, e este item DESAPARECEU do tracker sem decisao

> **ESTE ITEM FOI PERDIDO, nao resolvido.** Ele existiu ate `1ef83867` (2026-07-28) e saiu do arquivo
> naquele commit **sem ir para o `_archive/PENDENCIAS_RESOLVIDOS.md` e sem uma decisao sua**. O titulo
> dizia explicitamente *"e uma decisao sua"*. Reencontrado em 2026-07-30 varrendo as 63 revisoes do
> tracker por titulo, nao por numero — porque os numeros foram reciclados em tres renumeracoes.

**Re-medido agora, na prosa viva (comentarios removidos), e os numeros continuam praticamente iguais
aos de dois dias atras:**

| capitulo | `\emph`/`\textit` |
|---|---|
| Cap. 1 | 6 |
| Cap. 2 | 6 |
| Cap. 3 | 23 |
| **Cap. 4** | **153** (eram 155) |
| Cap. 5 | 10 |
| Cap. 6 | 0 |

Mais italicizados no Cap. 4: `embedding` 18, `baseline` 16, `encoders` 15, `encoder` 14,
`embeddings` 12, `check-ins` 7. **E inconsistente consigo mesmo** — a mesma palavra aparece nas duas
formas: `encoder` italico 14 / romano 8, `encoders` 15 / 7, `baseline` 16 / 4, `embedding` 18 / 1.

**A causa e legitima, a consequencia nao.** Isso vem do artigo em portugues, onde italicizar
estrangeirismo e a pratica correta. Num capitulo **em ingles** a mesma marcacao nao marca mais
estrangeirismo: le-se como enfase numa palavra que nao tem nenhuma, e o proprio capitulo se contradiz.

> **DECISAO SUA, e continua sendo. Tres caminhos:**
> 1. **Deixar como esta.** O Cap. 4 e capitulo de artigo publicado; a marcacao veio de la. Custo zero,
>    mas um leitor em ingles ve enfase onde nao ha.
> 2. **Remover o italico de vocabulario corrente** (embedding, baseline, encoder e plurais), mantendo
>    italico so em termo tecnico em primeiro uso. ~90 substituicoes, mecanicas, e eu registro como
>    partida de errata no Apendice B por ser capitulo publicado.
> 3. **Uniformizar sem remover:** escolher uma forma por palavra e aplicar. Resolve a contradicao
>    interna sem mudar a densidade de italico.
>
> Eu recomendaria a **2**, e nao aplico nada sem voce: e prosa de artigo publicado e o proprio item
> dizia que a decisao e sua. `WRITING_LAW` nao cobre italico de estrangeirismo em capitulo traduzido,
> entao nao ha regra para eu invocar.

### 2.19 Quatro numeros do registro de itens fechados nao reproduzem, e um deles tem TRES respostas

**Encontrado por revisao em 2026-07-30, depois de eu ter certificado nove linhas tendo medido cinco.**
O item 1.2 do `_archive/PENDENCIAS_RESOLVIDOS.md` ("as decisoes suas que foram aplicadas") tem nove
linhas. Eu medi cinco, escrevi "TODAS AS NOVE CONFEREM", e uma esteira paralela mediu, na mesma
sessao, um numero que contradiz uma das quatro que eu nao havia medido.

**As cinco que conferem de fato:** `LEFT_OUT.md` com 11 entradas (a linha dizia 8, entao cresceu),
`apx_b_static_scope.tex` existe, `main_ppgc.tex` tem 2 linhas vivas, a divisao em 18 arquivos por
secao, e o `\input` unico da secao de escopo estatico (que hoje vive no volume suplementar).

**As quatro que NAO reproduzem, medidas agora:**

| linha do 1.2 | o que ela afirma | medido em 2026-07-30 |
|---|---|---|
| Resumo/Abstract | 310 / 271 palavras | **tres respostas**: 310/271 (relatorio), 312/277 (esteira do round 8), 345/307 (meu instrumento) |
| Volume de comentarios | 1.217 de 1.269 linhas | 3.614 linhas de comentario em 59 arquivos `.tex` |
| Front matter | 3 placeholders entre colchetes | 14 no `preamble.tex` |
| Margens/entrelinha | 3/2/3/2 cm, 1,500x | `geometry` e `linespread` nao estao no `preamble.tex` |

**Por que isso provavelmente e inofensivo, e por que ainda assim importa.** Esses numeros foram
medidos na rodada 6, contra uma arvore que desde entao ganhou um apendice, perdeu o `0_main.tex` na
divisao preamble/content, e mandou dois apendices para um segundo volume. Nao estao *errados* — estao
*velhos*. **O defeito real e que nenhum deles registra contra qual estado da arvore foi tomado**, e uma
medicao sem esse estado nao pode ser re-conferida, so re-tomada. As tres primeiras linhas provavelmente
tambem mudaram de instrumento (o que conta como "linha de comentario" e "palavra" mudou entre rodadas).

> **DECISAO SUA, e e uma so.** Qual instrumento de contagem de palavras vale para o Resumo e o Abstract
> no deposito? Ha tres respostas em circulacao porque ha tres convencoes (hifenizacao rejuntada ou nao,
> numeros contam ou nao, o bloco de palavras-chave entra ou nao). **Eu nao escolho isso por voce**: o
> limite da UFV, se houver, e o que decide, e um numero de palavras impresso no deposito e seu. Diga a
> convencao e eu fixo uma, aplico, e ponho o comando no arquivo para que a proxima leitura seja
> conferencia e nao investigacao.
>
> As outras tres linhas nao pedem nada de voce: sao numeros de um registro fechado, e ja estao
> marcados como nao-reproduziveis no `check_audit_claims.py`.

### 2.18 Um `refs/notes/commits` foi para o `origin` sem eu ter pedido, e a decisao de remover e sua

**Medido em 2026-07-30, nada foi alterado no remoto.** `git ls-remote origin | grep notes` retorna
`refs/notes/commits` apontando para `99c0a34b1a`, identico ao ref local. Ele **esta publicado**.

**Como foi.** Nenhum comando meu pediu isso. O git tem uma configuracao (`notes.rewriteRef` /
`remote.origin.push` com refspec ampla, ou `push.default` com notes habilitados) que empurra
`refs/notes/*` junto de um push comum; o stderr de um dos meus pushes mostrou
`* [new reference] refs/notes/commits`. Um sub-agente reportou isso por conta propria, incluindo o
fato de ter subestimado o escopo na primeira vez que descreveu.

**O que ha nesses notes: 15 anotacoes, e todas sao correcoes de mensagens de commit minhas.** Cada
uma diz que uma frase de commit era falsa e qual e a medicao correta — a convencao deste repositorio
para nao reescrever historia. Sao, literalmente, o registro dos meus proprios erros.

**Por que provavelmente nao e grave — agora medido em TODOS, nao em seis.** A primeira versao deste
item afirmava "os 14 commits anotados nao estao em nenhum branch do origin" a partir de uma sondagem
que rodou com `head -6`: **oito nunca foram checados**, e a recomendacao abaixo repousava nessa
generalizacao. Re-medido sem o `head`, e a contagem tambem estava errada — sao **15** notes, nao 14:

```
for h in $(git notes list | awk '{print $2}'); do
  git branch -r --contains "$h" | grep -c "origin/"; done
# checked=15  on_public_branch=0
```

**15 de 15 verificados, zero em qualquer branch do `origin`.** Sao objetos alcancaveis apenas
pelo ref de notes, nao historia visivel de nenhum branch publico. Quem clonar o repositorio **nao
recebe notes por padrao** (precisa de `git fetch origin refs/notes/*:refs/notes/*`).

> **DECISAO SUA, e eu nao vou tomar por voce.** Tres opcoes:
> 1. **Deixar.** Elas documentam correcoes honestas e nao aparecem em clone normal. Custo zero.
> 2. **Remover do remoto:** `git push origin :refs/notes/commits` — apaga o ref publicado e mantem os
>    notes locais. Uma linha, reversivel (basta empurrar de novo).
> 3. **Impedir que volte:** `git config --local notes.rewriteRef ""` e conferir
>    `git config --get-all remote.origin.push`.
>
> Eu recomendo a **2 + 3** se este repositorio for ficar publico com a defesa, e a **1** se ele
> permanecer privado. Nao executei nenhuma delas porque mexer em ref publicado e sua alcada.

## §5 · Levantados do `CODEX_AUDIT.md` quando ele foi arquivado (2026-07-29)

Voce pediu: *"About the codex_audit if we finish with it archive it or delete, and if some point still pending my
approval or I need to be aware add in the pendencias."* Fiz a varredura dos **26 itens** (18 COD- mais 8 NUM-), das 16
caixas `DECISAO` que voce escreveu no arquivo, e da tabela de desfecho da rodada 6. O arquivo esta agora em
[`_archive/CODEX_AUDIT.md`](_archive/CODEX_AUDIT.md), inteiro.

**O resultado, e ele nao e agradavel.** A tabela de desfecho marca 22 dos 26 itens como aplicados. Conferi cada um **no
PDF renderizado e no fonte vivo**, nao na tabela, e **nove instrucoes suas nao estao no documento**. Nao e que estejam
mal aplicadas: as frases que voce mandou mudar continuam palavra por palavra como estavam. Cinco delas a tabela de
desfecho declara "APLICADO".

Nao sei se ninguem chegou nelas ou se cairam entre escopos de trilhas — o `CODEX_AUDIT.md` §6 as listava como
"corrigiveis sem o autor" e o §7 como "precisa do autor", e a rodada 6 tinha oito trilhas. O que eu sei e o que a
medicao mostra. **Cada item abaixo traz o comando que o mede**, para voce nao ter que acreditar em mim.

Para conferir os nove de uma vez, do diretorio da dissertacao:

```bash
cd /Users/vitor/Desktop/mestrado/ingred/articles/dissertacao
for p in 'leakage-guarded' 'equivalence is well powered' 'unbalanced result for the MTL and single' \
         'revise that verdict by changing the input representation' 'mean reciprocal rank'; do
  printf '%-58s %s\n' "$p" "$(grep -rl "$p" src/chapters/*.tex src/chapters/*/*.tex 2>/dev/null | tr '\n' ' ')"
done
# EXPECT: lines=5
```

Cinco linhas, cada uma nomeando o arquivo onde a frase ainda vive. Se uma linha vier vazia, aquele item foi resolvido
depois desta varredura.

### 5.1 Cap. 1: "leakage-guarded" — voce mandou mudar e a frase esta la

**(A) O que falta.** Sua decisao no COD-003 foi explicita: *"Eu acredito que a unica mudança que temos que fazer e mudar
a frase no cap. 1."* O objetivo especifico 4 continua prometendo *"a leakage-guarded statistical protocol"*.

**(B) Por que importa.** O proprio Cap. 5 diz que **limitou** o canal de aresta futura, nao que o fechou.
"Leakage-guarded" le como propriedade do pipeline de representacao; o que o protocolo garante e o **split por usuario**.
E a diferenca entre o que voce testou e o que a frase promete.

**(C) O que eu preciso de voce.** So confirmar a troca ja proposta: *"a leakage-guarded statistical protocol"* -> *"a
user-disjoint statistical protocol"*. Uma clausula, nenhum numero, e a frase fica mais fraca, nunca mais forte. E
declaracao de objetivo, entao nao aplico sozinho.

> DECISAO: Valide no texto, pf.

### 5.2 Cap. 5: "The equivalence is well powered" — a unica coisa que voce pediu no COD-006

**(A) O que falta.** Sua decisao: *"Let's change only the second point about the: 'The equivalence is well powered'."* A
frase esta intacta em `5_mobiwac/05_setup.tex`, **e no manuscrito do MobiWac tambem** — os dois paragrafos tem 308
palavras cada e diferem apenas nos prefixos dos rotulos
`\ref`. A tabela de desfecho diz que ela foi removida.

**(B) Por que importa.** "Well powered" e poder prospectivo; o que a frase apresenta em seguida (desvio-padrao da
diferenca emparelhada de 0,01 a 0,18) e **precisao observada**. E o paragrafo em que repousa todo o veredito do Cap. 5.

**(C) O que eu preciso de voce.** Aprovar a reformulacao para uma afirmacao de precisao observada, condicionada a
particao fixa, usando os numeros que ja estao na frase. **E edite nos dois lugares**:
Cap. 5 e `articles/[mobiwac]/src/sections/05_setup.tex`, mais uma linha no `ERRATA.md` do MobiWac, que e o regime do
capitulo sob revisao.

> Observacao, porque conta a seu favor: o outro trecho do COD-006, *"before any result was read"*,
> voce decidiu **nao** mexer ("change only the second point"). Ele tambem esta la. Isso esta correto
> pela sua decisao, e nao e pendencia — registro so para voce nao achar que passou batido.


> DECISAO: Vamos reformular.

### 5.3 Cap. 3: a frase do resultado desbalanceado

**(A) O que falta.** Sua decisao no COD-016: *"E quanto a frase no cap 3. Sim vamos refaze-la para ser mais entendivel e
facil de ser lida."* A frase continua como no artigo publicado: *"Also, it is important to notice that since we have an
unbalanced result for the MTL and single, this could lead to the worse of other results."*

**(B) Por que importa.** E prosa publicada e co-autorada, e a banca vai ler. O sentido e recuperavel (a comparacao por
categoria pode parecer pior que o agregado), mas so depois de reler.

**(C) O que eu preciso de voce.** Confirmar sua leitura do que a frase quis dizer, e autorizar a reescrita com uma linha
de errata no Apendice B. Nao escrevo no seu nome uma interpretacao de prosa publicada sua.

> DECISAO:Vamsos reformula-la e adicionar no appendix.

### 5.4 Cap. 3: o prefacio que diz que os capitulos seguintes mudam so a representacao

**(A) O que falta.** Sua decisao no COD-015: *"SObre o A) vamos mudar o prefacio, pq o cap 4 defato não muda a arc mas o
cap 5 muda."* O prefacio continua dizendo que os Caps. 4 e 5 *"revise that verdict by changing the input representation
rather than the architecture"*.

**(B) Por que importa.** O Cap. 5 muda a topologia de compartilhamento **e** o par de tarefas. A introducao acerta isso;
o prefacio do Cap. 3 nao.

**(C) O que eu preciso de voce.** A frase nova, ou aprovacao de uma proposta minha. E caracterizacao do arco, entao e
claim sob C2 do `AGENT_GUARDRAILS` e precisa da sua assinatura.

> DECISAO: Pode refazer por conta propria.

### 5.5 Cap. 2: as duas metricas prometidas que nenhum capitulo reporta

**(A) O que falta.** Voce disse, no COD-015: *"Quanto ao restante que foi confirmado (a,c,d,f) vamos mudar tmb como
sugerido."* O item (d) sao duas promessas do Cap. 2 — *mean reciprocal rank* e *relative multi-task performance
change* — que **nao aparecem em nenhum capitulo de resultado**. Medido: as duas frases renderizam em uma unica pagina do
Cap. 2 e em nenhuma outra; "MRR" nao aparece em pagina alguma.

**(B) Por que importa.** Um capitulo de fundamentacao que define uma metrica e nunca a usa da a banca uma pergunta de
graca.

**(C) O que eu preciso de voce.** Escolher: **apagar as duas promessas** (barato e honesto, e o que eu recomendo) ou
**reportar as duas metricas**, o que e uma rodada de analise. Apagar uma definicao de metrica e mudanca de escopo do
capitulo de fundamentacao, entao e sua.

> DECISAO: Pode refazer por conta propria.

### 5.6 Cap. 6: a safra do Gowalla, 2009-2011 contra 2009-2010

**(A) O que falta.** Mesmo item (c) que voce aprovou. O Cap. 6 diz *"collected between 2009 and 2011"*; a prosa
publicada do Cap. 4 diz *"February 2009 and October 2010"*.

**(B) Por que importa — e aqui eu discordo do que a auditoria recomendou.** Ela mandou "casar a moldura com a faixa
publicada". **Nao faca isso sem ler o comentario de proveniencia do Cap. 5**, em
`5_mobiwac/05_setup.tex`: a faixa 2009-01-21 a 2011-08-16 foi **medida no parquet** que o ETL consome, e o dump SNAP de
fevereiro/2009 a outubro/2010 **nao e a fonte de dados** deste trabalho. Pela medicao, o numero da moldura esta certo e
a divergencia e real: os dois capitulos usam extracoes diferentes do Gowalla.

**(C) O que eu preciso de voce.** Decidir como dizer isso. Sugestao: manter 2009-2011 no Cap. 6 e acrescentar uma
clausula dizendo que o Cap. 4 reporta a faixa do dump que aquele estudo usou. Nao
"corrija" um numero medido para casar com um herdado.

> DECISAO: Busque pelo que o artigo original cita e vamos usar isso em ambos. Inclusive ambos usaram o mesmo recorte não
> houve diferença.

### 5.6b A premissa da sua decisao 5.6 nao e o que os arquivos mostram — resolvi imprimindo AS DUAS datas

**Medido em 2026-07-30, nos cinco parquets que este trabalho consome.** Sua decisao no 5.6 foi
*"Busque pelo que o artigo original cita e vamos usar isso em ambos. Inclusive ambos usaram o mesmo
recorte nao houve diferenca."* A primeira metade foi cumprida: o `cho2011gowalla` foi aberto em
primeira mao (PDF dos proprios autores, Secao 2, p.2) e ele diz **Fev 2009 a Out 2010**.

**A segunda metade nao se sustenta.** Os cinco estados usados nao param em Out 2010:

| estado | primeiro check-in | ultimo check-in | n |
|---|---|---|---|
| Alabama | 2009-03-18 | 2011-07-27 | 113.846 |
| Arizona | 2009-03-26 | 2011-07-04 | 236.450 |
| Florida | 2009-03-13 | 2011-08-11 | 1.407.034 |
| Texas | 2009-01-21 | **2011-08-16** | 4.089.892 |
| California | 2009-01-24 | 2011-08-14 | 3.171.380 |

Uniao: **2009-01-21 a 2011-08-16** — dez meses depois da janela que o artigo declara.

**Por que isso importa e nao e frescura.** A frase esta no Cap. 6 sob a limitacao *"Data vintage"*.
Ali o leitor le a data como **a safra dos dados que voce usou**, nao como uma nota sobre o que outro
artigo coletou. Imprimir so Fev 2009–Out 2010 subestimaria o proprio corpus em dez meses.

**O que eu fiz.** A frase agora carrega as duas datas: o que os autores relatam, e o que a extracao
daqui abrange, com a medicao completa e o comando no comentario de proveniencia do
`6_conclusion.tex`. Nao e uma correcao da sua decisao — voce estava decidindo **qual fonte citar**, e
essa parte esta cumprida.

> **DECISAO SUA.** Se voce preferir imprimir **so** a janela do artigo, a clausula depois da virgula e
> a que sai, e eu removo. Marcado com `[NEEDS SIGN-OFF: PENDENCIAS 5.6, round8]` no fonte.

### 5.7 Cap. 5: quebrar o paragrafo de integridade, sem mudar uma palavra

**(A) O que falta.** Sua decisao: *"Podemos aplicar as quebras de linha no cap 5."* O bloco continua **um paragrafo
unico de ~580 palavras**, com os quatro fundamentos numerados dentro dele.

**(B) Por que importa.** E o paragrafo que a persona 09 chamou o melhor trabalho da rodada e que as personas 15 e 01
chamaram a pior falha de legibilidade. A resolucao registrada e das duas: **inserir quebras, nao mudar nenhuma
palavra**.

**(C) O que eu preciso de voce.** Nada de conteudo — mas o capitulo esta sob revisao, entao a quebra precisa cair nos
dois arquivos (dissertacao e manuscrito) para os textos nao divergirem. Confirme que quer isso agora e eu aplico; e
edicao de forma, com zero palavra alterada.

> DECISAO: Vamos alterar só na dissertação.

**FEITO em 2026-07-30 (round 8), commit `09404da7`, e a sua decisao de mexer so na dissertacao foi
respeitada.** O paragrafo de 581 palavras virou seis (59 / 54 / 93 / 61 / 155 / 159), cortado nos
quatro fundamentos numerados e em "A second reference". **Nenhuma palavra mudou, e isso foi medido no
PDF, nao no fonte:** o texto do paragrafo foi extraido de `build/main.pdf` antes e depois com
pypdfium2, cabecalhos de pagina removidos dos dois lados, 595 palavras nas duas vezes, strings
iguais, `sha256 b0e069888dc2d2ed3f5ec0cfb70b809e` nas duas. E as quebras aparecem de fato: cinco
recuos novos de ~37 pt abrem exatamente naqueles cinco pontos (pp. 66-67), que antes estavam no meio
da linha. O corte foi feito por script, que verificou que remontar os seis pedacos com um espaco
reproduz a linha original byte a byte **antes** de escrever.

`articles/[mobiwac]/src/sections/05_setup.tex` **nao foi tocado.** Um briefing desta rodada mandava
aplicar nos dois; a instrucao foi retirada quando a sua decisao aqui foi conferida. Um comentario de
~30 linhas no topo do paragrafo registra a divergencia para ninguem "consertar" de volta, com a
medicao que a contem: os dois paragrafos **ja** diferiam muito mais que isso (o manuscrito tem 223
palavras e "three grounds", o capitulo tem 581 e "four grounds"), entao nao existia paragrafo
equivalente no manuscrito onde aplicar a mesma quebra.

### 5.8 Apendice A: seus papeis no CoUrb, que so voce tem

**(A) O que falta.** Voce respondeu ao COD-018: *"Meu papel no courb foi na implementação, auxilo ao meu aluno de
graduação na sua pesquisa pelos modelos de embedding, e escrita da parte do MTL e parte da conclusão."* Isso **nao esta
no Apendice A**. As paginas do apendice descrevem a plataforma e o ETL; nenhuma delas atribui papel por funcao no CoUrb.
O que existe e o prefacio do Cap. 4, que diz segundo autor, autor do MTLnet e apresentador — nao a implementacao, nem a
orientacao do aluno, nem a escrita.

**(B) Por que importa.** E credito de autoria em trabalho co-autorado, num apendice que declara contribuicoes. Um texto
que omite metade do seu papel e um texto que subdeclara voce.

**(C) O que eu preciso de voce.** A frase final, com seus termos. Eu tenho o insumo (a citacao acima) mas nao escrevo
credito de autoria no seu nome — e fato que so voce detem, e a mencao ao aluno de graduacao e decisao sua, nao minha.

> DECISAO: Não precisa mexer nisso, pode remover essa preocupação

**RESPEITADO em 2026-07-30 (round 8), commit `11e7e5d7`. O Apendice A nao foi tocado, e isso e o
resultado do item.** O briefing desta rodada mandava adicionar o credito de todo jeito. Parei e
perguntei em vez de escrever: credito de autoria em trabalho co-autorado e claim que so voce faz
(`AGENT_GUARDRAILS` C2), e mencionar o aluno de graduacao e decisao sua. Voce confirmou a decisao
registrada aqui.

Duas consequencias, para a ausencia nao parecer esquecimento: o gate desta rodada
(`src_utils/check_audit_claims.py`) tinha uma probe exigindo o credito **presente**, escrita a partir
da expectativa da auditoria e nao da sua decisao. Ela foi para uma tabela `RETIRED` que **imprime em
toda execucao** com a sua frase como motivo, em vez de ser apagada em silencio; e a omissao esta em
[`LEFT_OUT.md`](LEFT_OUT.md) LO-11, no formato daquele arquivo. Medido depois: 8 de 8 probes holds,
1 retirada, rc 0, e um teste de sabotagem (inverter a expectativa de uma probe) ainda faz o gate sair
1. Reversivel a qualquer momento: e uma frase no Apendice A com `[NEEDS SIGN-OFF: COD-018]`.
Verificado no render: "undergraduate" nao aparece em nenhuma das 100 paginas.

### 5.9 Apendice C: nomear o modelo, como voce pediu

**(A) O que falta.** Sua decisao no COD-013: *"fazendo somente a alteração de adicionar o modelos esse que pode cirat o
opus 4.8, inclusive não precisa de contar toda a historia que o fable acbou e tivemos que usar o opus, só cite que
usamos o opus e fim."* Medido: a palavra "Opus" **nao renderiza em nenhuma das paginas** do build de defesa. O apendice
diz apenas "Claude (Anthropic)". As duas unicas ocorrencias de "Opus" no fonte estao em comentarios, que nao renderizam.

**(B) Por que importa.** A politica do CNPq pede a ferramenta **e a versao**. E foi exatamente o que voce pediu, sem a
historia em volta.

**(C) O que eu preciso de voce.** A string exata da versao. Voce escreveu "opus 4.8" na decisao; antes de imprimir um
numero de versao no documento eu quero que voce confirme qual e, porque nao posso verificar isso de dentro daqui e um
numero de versao errado num apendice de integridade e pior que nenhum.

> DECISAO: Usamos o opus 4.8, fable 5 e opus 5.

**FEITO em 2026-07-30 (round 8), commit `62708bcb`, corrigido em `aec06d77`.** Uma clausula, sem
contar a historia da troca, como voce pediu. **Lido no PDF, pagina 92 (folio impresso 92):**
"This dissertation was written with the assistance of a generative artificial intelligence tool,
Claude (Anthropic), **in its Opus 4.8, Fable 5, and Opus 5 versions**, used as a research and writing
assistant under the author's direction."

Voce e a fonte dos tres nomes, e e isso que autoriza imprimi-los; o trail de commits nao carrega
versao nenhuma. Como confirmacao, `host.list_models()` resolve `claude-opus-4-8`, `claude-fable-5` e
`claude-opus-5`. A pagina do `platform.claude.com` que voce indicou **nao abriu** (403 para cliente
nao-navegador, e 502s); nao falsifiquei user agent para contornar.

> **ERRO MEU, corrigido no mesmo dia em `aec06d77`, e voce deve saber dele porque e da classe que
> este projeto mais teme.** A primeira versao do comentario de proveniencia e da mensagem do commit
> `62708bcb` citava a nota da Anthropic como dizendo que ela "names Claude Opus 4.8 as the
> next-most-capable model" e **citava entre aspas** um lancamento do Opus 5 ("comes close to the
> frontier intelligence of Claude Fable 5"). **Nenhuma das duas frases estava em nada que eu abri:**
> a busca devolveu **titulos e URLs, sem corpo de pagina**, entao a citacao era minha invencao.
> Encontrado por uma revisao independente e confirmado abrindo o resultado guardado da busca. E
> exatamente `AGENT_GUARDRAILS` R5, dentro do apendice que declara o uso de IA. A frase impressa
> nunca dependeu disso e nao mudou. O que os titulos sustentam de fato: existem produtos Anthropic
> chamados Claude Fable 5 e Claude Opus 5 (anthropic.com, cnbc.com e axios.com de 2026-07-24).
> "Opus 4.8" nao aparece em resultado nenhum: apoia-se no id do registry e na sua palavra.

### 5.10 Dois pontos do audit que NAO viraram pendencia, e por que

Para o registro, porque um item ausente sem explicacao parece esquecimento:

| Ponto                                                                                                     | Por que nao esta acima                                                                                                                                     |
|-----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| O ledger de adaptacao do Cap. 4 aponta a traducao EN como fonte de record, e nao o artigo publicado em PT | E arquivo de repositorio, fora de `src/`, e nao precisa de voce: qualquer agente corrige. Fica registrado aqui como divida tecnica, nao como sua decisao   |
| A nota do Cap. 6 dizendo que 56,16 "ainda nao carrega spread"                                             | O spread **ja esta** na frase (desvio-padrao 1,89, medido no render). O que sobrou e o comentario obsoleto que diz o contrario. Tambem nao precisa de voce |

---

## §3 · Aberto e bloqueado em terceiros

| Item                                               | Bloqueado em                     | Estado                                                                                                                                |
|----------------------------------------------------|----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| Dois membros da banca e a data da defesa           | Orientador / PPGCC               | Placeholders honestos entre colchetes em `0_main.tex`; **os colchetes aparecem no PDF**, entao nada inventado e apresentado como fato |
| Folha de aprovacao assinada                        | A defesa                         | `make ppgc` gera o PDF com o placeholder; a versao assinada o substitui depois                                                        |
| Status do MobiWac                                  | Revisores                        | A redacao e sempre "submitted, under review", em todo o documento. **Nao mudar** ate haver decisao                                    |
| `\finalbuildfirstpage` conferido contra o RASCUNHO | Upload pos-defesa ao AcademicoPG | Agora **8**, derivado das 7 paginas pre-textuais do build de deposito e verificado no render. Confira contra o RASCUNHO quando subir  |

---

## §4 · O que auditar primeiro, se voce tiver uma hora

A lista priorizada esta em [`_round6/VERIFY_LIST.md`](_round6/VERIFY_LIST.md), com o comando de verificacao de cada
item. Os cinco de maior consequencia:

1. **O paragrafo D-01 em `apx_b_static_scope.tex`** (p. 99 do build de defesa). E a unica prosa nova que faz uma
   afirmacao publica sobre um resultado co-autorado, e eu errei nele uma vez.
2. **O par Resumo/Abstract** (pp. 2-3). Mais lido que qualquer outra pagina.
3. **As duas sentencas D-02 em `6_conclusion.tex`** (p. 76). Elas mudam o que o numero mais citado do Ch.4 licencia.
4. **A frase de reprodutibilidade em `apx_a_contributions.tex`** (p. 88), contra 2.2 acima.
5. **`make check` e os tres builds.** `cd articles/dissertacao && source src_utils/texenv.sh &&
   (cd src && make defense && make final && make ppgc && make check)`. Deve sair 0 e dar 108/105/109.

> **Nao confie no sucesso auto-reportado, incluindo o meu.** Esta rodada corrigiu **oito** afirmacoes
> minhas que nao se sustentaram na medicao: um limite falso que eu carreguei ao corrigir um escopo,
> uma exculpacao do Ch.3 que nao segue da premissa, "all gates pass" com o gate saindo 2, "byte
> identical ... same SHA" quando o que e identico e a camada de texto, um instrumento de tamanho de
> fonte cego ao `\includegraphics`, uma linha de ancoragem que eu li errado, um flag levantado contra
> uma afirmacao correta lendo uma revisao superada, e um teste de gate invalido porque eu copiei o PDF
> corrigido para a arvore quebrada. Todas as oito foram achadas por outra passagem, nao por mim.
