# PENDENCIAS.md — o que depende de voce

> **Registro de pendencias da dissertacao (v3, 2026-07-28).** Cada item aqui esta bloqueado em um
> fato externo, em uma decisao sua, ou em uma aprovacao do orientador/Comissao. **Nada aqui pode ser
> resolvido por um agente, e nenhum foi resolvido sozinho.**
>
> Formato de cada item: **(A) o que falta**, **(B) por que importa**, **(C) o que eu preciso de voce**.
> Onde ja existe texto pronto ou medicao feita, o caminho esta indicado.
>
> **Estado do build agora** (medido em `29c7629c`, os tres alvos reconstruidos):
>
> | alvo | paginas | tex_errors | overfull hbox/vbox | undef cite/ref | bibtex | floats grandes | avisos Hfootnote |
> |---|---:|---:|---:|---:|---:|---:|---:|
> | `make defense` -> `main.pdf` | **108** | 0 | 0 / 0 | 0 / 0 | 0 | 0 | 0 |
> | `make final` -> `main_final.pdf` | **105** | 0 | 0 / 0 | 0 / 0 | 0 | 0 | 0 |
> | `make ppgc` -> `main_ppgc.pdf` | **109** | 0 | 0 / 0 | 0 / 0 | 0 | 0 | 0 |
>
> `make check`: **exit 0**. Pela primeira vez nesta rodada. Ver o item fechado C-6 abaixo: durante
> toda a rodada o gate saiu com codigo 2 enquanto seis mensagens de commit, minhas inclusive,
> afirmavam "all gates pass".

---

## Como ler este arquivo

Ele tem tres partes e apenas a segunda exige algo de voce:

1. **§1 Fechado nesta rodada** — o que entrou, com o commit. Nada a fazer; esta aqui para voce
   auditar.
2. **§2 Aberto e bloqueado em voce** — decisoes, aprovacoes e um push de repositorio. **Esta e a
   sua fila.**
3. **§3 Aberto e bloqueado em terceiros** — orientador, Comissao, revisores do MobiWac. Fora do seu
   controle e do meu.

Um item que esta ausente do texto porque alguem **decidiu** que ficasse fora nao e pendencia: esta
em [`LEFT_OUT.md`](LEFT_OUT.md), com quem decidiu e quando. Um item ausente porque ninguem chegou
nele e pendencia e esta aqui.

---

## §1 · Fechado nesta rodada (round 6)

### 1.1 O bloqueador que ninguem viu: o fonte nao compilava

**Fechado em `ba90aa6d`.** De `6d780b58` a `a880632b` a chave de abertura do grupo `{\small ...}` em
`tables/frame/bib_errata.tex` estava ausente e a de fechamento sobreviveu. Todo build morria com
`! Extra }, or forgotten \endgroup`. Seis mensagens de commit reportaram "104/99 pp, 0 overfull, 0
undefined" nesse periodo.

**Por que passou:** os dois caminhos de build discordavam e o que foi acreditado era o cego. O
`Makefile` usa `-halt-on-error` e nao produzia nada; `src_utils/build.sh` roda em
`-interaction=nonstopmode`, onde o pdflatex se recupera do erro e **escreve um PDF completo**, que o
script media e certificava limpo — porque nunca procurava por erros. Agora procura (`tex_errors`), e
o gate foi validado nas duas direcoes antes de ser aceito.

**Registrado em:** `science/AGENT_HANDOFF.md` §2.3b e `AGENT_GUARDRAILS.md` §7 (duas linhas novas de
vies: desconfie da ferramenta que reporta sucesso quando duas discordam; valide todo gate novo contra
uma arvore onde o defeito esta presente).

### 1.2 As decisoes suas que foram aplicadas

| Sua decisao | O que foi feito | Commit |
|---|---|---|
| COD-007: recuperar protocolo de Ch.3/Ch.4 | Eixo de split, seeds e regra de checkpoint recuperados do codigo e dos artefatos de execucao, e adicionados como **adicoes declaradas** com trilha em Apendice B | `519de348`, `a7ab2eaa` |
| COD-002: registrar o que fica fora do texto | [`LEFT_OUT.md`](LEFT_OUT.md) criado, 8 entradas, cada uma com achado, o que o texto diz em vez disso, por que esta fora, e **quem decidiu com data** | `e9370222` |
| 2.2: a tarefa estatica do Ch.4 em Apendice B, "facil de ser comentado" | Secao nova em `chapters/apx_b_static_scope.tex`, incluida por **um** `\input`. Caminho de supressao **testado** em arvore copiada: compila limpo, sem referencia pendente | `28097d93` |
| Split do `main.tex` | `main.tex` = build da defesa; `main_ppgc.tex` = o mesmo PDF **mais** a folha de aceite, em duas linhas de conteudo, para os dois nao divergirem. Terceiro alvo `make ppgc` | `7a91b720` |
| Chapters "corridos": dividir como nos artigos originais | Os tres capitulos de artigo divididos em 18 arquivos por secao, espelhando os nomes de arquivo de cada artigo. Verificado mecanico: camada de texto dos tres builds **identica byte a byte** antes e depois | `4e84cf7a` |
| Resumo/Abstract: cortar e refazer | 500 -> 310 palavras e 423 -> 271, refeitos como par de paridade de 11 sentencas, 19 claims em ambas as linguas | `40ed8e7b` |
| Margens e formatacao local | **Medido, nada a mudar.** Sondei a geometria real compilando uma pagina com o preambulo do documento: 3/2/3/2 cm e entrelinha exatamente 1,500x, todos exatos ao manual §7 | `2d117c7a` |
| Volume de comentarios | **Medido, recomendo nao comprimir.** 1.217 de 1.269 linhas de comentario (95%) carregam um fato rastreavel; as 52 restantes sao banners estruturais e a sua propria fila de sign-off | `2d117c7a` |
| Nomes de exemplo no front matter | **Nao existem.** Todos os campos reais, exceto tres placeholders honestos entre colchetes (dois membros da banca e a data), que e o estado correto | `18b817d9` |

### 1.3 Os defeitos que as revisoes acharam, e que foram corrigidos

Oito trilhas de revisao rodaram sobre o texto desta rodada. Nenhuma tinha visto o que a outra fez.
O que elas acharam:

| ID | Gravidade | O defeito | Commit |
|---|---|---|---|
| N-1 | BLOCKER | O limite "dentro de ±0,003" do cosseno de gradientes e **falso para Alabama** (+0,0032). Criado nesta rodada: o **escopo** da frase foi corrigido de tres para quatro datasets e o limite foi carregado sem reverificar a grandeza que depende do escopo. Corrigido na dissertacao **e** no manuscrito | `fecc7fb1` |
| N-2 | MAJOR | Ch.2 afirmava que Ch.3 "nao identifica o eixo de split", o que a adicao COD-007 desta rodada tornou falso no mesmo dia. O reparo foi **previsto por escrito** e redigido pela trilha de protocolo, e caiu entre dois escopos | `fecc7fb1` |
| D-01 | BLOCKER | Minha propria secao de Apendice B concluia que "o rotulo de um lugar nunca entra na sua propria representacao" no Ch.3. **A premissa e verdadeira e a conclusao nao segue:** o grafo e nao dirigido e a convolucao agrega o no com a vizinhanca, entao o rotulo volta no primeiro salto. Reproduzido em grafo de 4 nos: h_0[Food] = 0,667 contra x_0[Food] = 0,000 | `4b609643` |
| D-02 | BLOCKER | Ch.6 citava o ganho de 20,2 a 22,0 pontos do Ch.4 **sem rotular a tarefa**, como o diagnostico do arco inteiro. O ganho e da tarefa **estatica**, que o Apendice B desta rodada desqualifica. O numero fica (e a figura auditada do capitulo publicado), agora com tarefa e qualificacao, e o diagnostico repousa na tarefa sequencial | `4b609643` |
| F-01 | BLOCKER | **9 de 13** caminhos de reprodutibilidade do Apendice A **nao estao** no branch publico que o Ch.5 aponta em nota de rodape. Todos os 13 existem nesta maquina: a promessa estava errada, nao o codigo. (A primeira contagem dizia 8 de 12; ver `c6e62c62` -- um `grep` por linha perdeu `m1_full_output.txt`, que divide a linha com outro caminho) | `ec1cea0d`, `c6e62c62` |
| F-02 | MAJOR | A pagina 77, secao 6.2, tinha uma sentenca **sem sujeito**: "California run, completed since, repeats the pattern". O artigo "The" terminava a linha anterior e foi absorvido por um bloco de comentario inserido depois. Recuperado do commit original | `ec1cea0d` |
| C-1 | MAJOR | O build de **deposito** (AcademicoPG) imprimia 11 na pagina fisica 8. `\finalbuildfirstpage` estava fixo no offset do build de defesa, e o deposito tem tres paginas pre-textuais menos. Nao conformidade de numeracao no unico build que e depositado | `29c7629c` |
| E-5 | MAJOR | **Dez marcas de nota de rodape eram hyperlinks vivos para a pagina 1** em todos os tres builds. A persona mediu **onde os links caem**, nao apenas se os destinos resolvem. Corrigido com `hyperfootnotes=false` passado em tempo de carga (em `\hypersetup` **nao** funciona: o abntex2 ja carregou o hyperref) | `29c7629c` |
| E-2 | MAJOR | Seis arquivos sem diretiva `% !TeX root`, e depois do split esses seis incluiam os tres masters de capitulo. Segunda instancia na semana. Gate novo `check_tex_root.py` achou **18 outros** | `29c7629c` |
| STY-01 | MAJOR | Sete termos em uso que o registro fail-closed nao tinha, **dois deles em portugues no Resumo** que a minha propria passagem de registro seis horas antes nao cobriu | `a8865214` |
| AIC-01 | MAJOR | A densidade de paralelismo negativo foi **congelada** por uma revisao anterior e esta rodada a levantou de 67 para 79. O diagnostico da persona e o que importa: *"um guard que vive so no relatorio de uma rodada anterior e um guard que ninguem esta checando."* Movido para `check_negative_parallelism.py` | `a8865214` |
| C-6 | MAJOR | **`make check` saiu com codigo 2 durante toda a rodada** enquanto seis commits diziam "all gates pass". Dois falsos positivos ("this article" no apendice de errata, que e correto; "Pareto", que e o termo tecnico). Ambos isentos com a justificativa no lugar | `6ee23ca7` |
| L-9 | MINOR | O Apendice B imprimia "todos os 25 lugares" com uma decomposicao que soma 25 apenas contando um cabecalho de subsecao onde ha dois. Reenumerado: 28 | `6ee23ca7` |
| M-1, M-2 | MAJOR | O Resumo/Abstract perdeu o indice temporal do diagnostico do CoUrb e usava um universal sem escopo ("em todos os estados") cujo antecedente mais proximo e o numero errado de estados. Corrigidos **em paridade** nas duas linguas | `6ee23ca7` |

### 1.4 Gates novos, todos validados nas duas direcoes antes de serem aceitos

| Gate | A classe silenciosa que ele pega | Por que nenhum outro gate a via |
|---|---|---|
| `build.sh` `tex_errors` | O fonte nao compila | `nonstopmode` se recupera e escreve um PDF completo, que o script media |
| `check_doubled_macro.py` | `\\ref{...}` com barra dobrada, que imprime o rotulo cru | O pdflatex nao avisa (as duas metades sao legais) e `undef_ref` fica em 0, corretamente |
| `check_negative_parallelism.py` | Densidade de paralelismo negativo acima do teto | Vivia so num relatorio de revisao |
| `check_tex_root.py` | Diretiva `% !TeX root` ausente ou apontando para arquivo inexistente | Invisivel para o `make`, que le o `main.tex` e nunca olha um comentario magico |

---

## §2 · Aberto e bloqueado em VOCE

### 2.1 Os 46 marcadores `[NEEDS SIGN-OFF]` no fonte

**(A) O que falta.** 46 marcadores, distribuidos assim:

| Arquivo | n |
|---|---:|
| `chapters/6_conclusion.tex` | 7 |
| `chapters/apx_a_contributions.tex` | 6 |
| `chapters/2_fundamentals.tex` | 5 |
| `chapters/apx_b_errata.tex` | 5 |
| `0_main.tex` | 4 |
| `chapters/5_mobiwac/06_results.tex` | 3 |
| `chapters/1_introduction.tex`, `5_mobiwac/02_related.tex`, `5_mobiwac/07_discussion.tex` | 2 cada |
| `3_cbic/method.tex`, `3_cbic/results.tex`, `4_courb.tex`, `4_courb/methodology.tex`, `4_courb/results.tex`, `5_mobiwac.tex`, `apx_b_static_scope.tex`, `apx_c_ai_disclosure.tex`, `apx_d_ceiling.tex`, `apx_e_ethics.tex` | 1 cada |

Para listar: `grep -rn "NEEDS SIGN-OFF" src/ | grep -v Binary`.

**(B) Por que importa.** Cada um e uma frase reescrita por um agente em prosa que e sua, ou uma
mudanca de escopo num capitulo publicado. Nenhuma pode ir a banca sem voce ter lido.

**(C) O que eu preciso de voce.** Ler e aprovar ou reescrever. **Tres tem prioridade sobre os
outros 43:**

1. **`apx_b_static_scope.tex`** — o paragrafo D-01 corrigido. Ele agora diz algo **mais fraco sobre o
   Ch.3** do que a sua decisao de 2026-07-27 assumia. Voce decidiu que "esse nao se aplica ao DGI que
   usamos no cbic"; a medicao diz que o canal do Ch.3 e **indireto**, nao ausente. A secao continua
   concentrando o achado no Ch.4, onde esta a identidade exata. **Leia esse paragrafo especificamente.**
2. **`apx_a_contributions.tex`** — a frase de reprodutibilidade foi **enfraquecida** para "supplied on
   request" porque 9 de 13 caminhos nao estao no branch publico. Uma banca pode perguntar por que. Ver
   2.2 abaixo: se voce publicar os arquivos, a frase forte volta.
3. **`6_conclusion.tex`** — as duas sentencas D-02 adicionadas, que qualificam o numero do Ch.4.

### 2.2 Publicar os 9 arquivos que faltam no branch publico

**(A) O que falta.** O Apendice A cita **13** caminhos. Estes **nove** existem nesta maquina e
**nao** em `github.com/VitorHugoOli/PoiMtlNet/tree/mobiwac` (os quatro nomes sem barra resolvem
dentro de `stats_n20/`, e o caminho completo esta dado aqui para o push):

```
docs/studies/closing_data/v17_completion/STATISTICAL_PROTOCOL.md
scripts/build_phase3_per_fold_transitions.sh
docs/studies/closing_data/joint_best/JOINT_BEST_RESULTS.md
docs/studies/closing_data/v17_completion/stats_n20/
docs/studies/closing_data/v17_completion/stats_n20/m1_stats_n20.py
docs/studies/closing_data/v17_completion/stats_n20/m2_prereg_perfold.py
docs/studies/closing_data/v17_completion/stats_n20/m1_full_output.txt
docs/studies/closing_data/v17_completion/stats_n20/m2_prereg_output.txt
scripts/embedding_eval/autocorrelation_ceiling.py
```

Ja presentes, **nao republicar**: `src/data/folds.py`,
`scripts/closing_data/score_joint_best.py`, `scripts/closing_data/superiority_wilcoxon.py`,
`scripts/closing_data/region_match_tost.py`.

> **Correcao 2026-07-28 (`c6e62c62`).** A primeira versao desta lista dizia **8 de 12** e omitia
> `m1_full_output.txt`. A contagem veio de um `grep` linha a linha, e esse arquivo divide a linha com
> outro `\path{}` no fonte. Recontado extraindo cada `\path{}` da **prosa** com comentarios
> removidos: **13 entradas, 4 sem barra, 9 ausentes**. Confira com:
>
> ```bash
> cd /Users/vitor/Desktop/mestrado/ingred
> F=articles/dissertacao/src/chapters/apx_a_contributions.tex
> grep -v '^[[:space:]]*%' "$F" | grep -o 'path{[^}]*}' | wc -l          # 13 entradas na prosa
> grep -v '^[[:space:]]*%' "$F" | grep -o 'path{[^}]*}' | grep -cv '/'   #  4 sem barra
> 
> # EXPECT: lines=2
```
>
> **O `grep -v '%'` e obrigatorio.** Sem ele o comando retorna **15**, porque os proprios comentarios
> de proveniencia deste apendice escrevem a palavra `\path{}` ao explicar a contagem, e o `grep` casa
> com essas duas ocorrencias vazias tambem. Uma versao anterior desta nota trazia o comando sem o
> filtro e anotava `# 13` ao lado — um comando que **parece** certo e devolve outro numero, que e
> exatamente a classe de defeito que esta rodada passou o tempo todo corrigindo. Se voce rodar e vier
> 15, o filtro caiu.

**(B) Por que importa.** A frase de reprodutibilidade mais carregada do documento nao resolvia para
um leitor que seguisse a nota de rodape do proprio capitulo. Eu escopei a prosa para o que e
verdadeiro, mas **a correcao melhor e publicar**, nao enfraquecer a frase.

**(C) O que eu preciso de voce.** Um push desses **nove** para o branch publico. Feito isso, reverta o
paragrafo de `apx_a_contributions.tex` para a versao forte e apague o comentario `[round6, F-01]`
que esta la — ele contem a instrucao.

### 2.3 A secao de escopo da tarefa estatica: manter ou suprimir

**(A) O que falta.** Sua conversa com o orientador sobre argumentar ou nao publicamente quanto ao
escopo da tarefa estatica do Ch.4.

**(B) Por que importa.** E uma declaracao publica sobre um resultado publicado e co-autorado. Voce
tem o acordo do co-autor (2026-07-27); falta o orientador.

**(C) O que eu preciso de voce.** A decisao. **Para suprimir**, comente **uma** linha em
`chapters/apx_b_errata.tex`:

```latex
%\input{chapters/apx_b_static_scope}
```

Testado: compila limpo, sem referencia pendente, porque o ponteiro no prefacio do Ch.4 referencia o
**apendice**, nao o rotulo da secao. **Se suprimir, apague tambem a sentenca do prefacio do Ch.4**
ou ela aponta para um apendice que nao discute mais o assunto.

### 2.4 O tamanho de tipo das duas figuras de arquitetura

**(A) O que falta.** Uma decisao sobre `figures/cbic_mtlnet_arch.png` (45,3% do corpo) e
`figures/courb/arquitetura_modelo.png` (44,4%). Sao as duas menores do documento — **menores que as
duas que a auditoria rastreava**, que estao em 93,2% e 66,3% depois do reescalonamento desta rodada.

**(B) Por que importa.** `WRITING_LAW` §5 pede rotulos "proximos ao tamanho do corpo". Uma banca le
essas figuras impressas.

**(C) O que eu preciso de voce.** Autorizacao para mudar o tamanho de tipo de uma figura **publicada**
(a do Ch.4 e co-autorada). Ha `.drawio` para as duas, e a receita esta em
`_round6/12_figures.md`: subir `fontSize` de 13 para cerca de 20 e reexportar na mesma largura de
pixels. Registrado em [`LEFT_OUT.md`](LEFT_OUT.md) LO-6 como **diferido, nao recusado**.

### 2.5 A coluna do CBIC que nao reproduz

**(A) O que falta.** Tres das quatro colunas de resultado publicadas do CBIC reproduzem exatamente
contra as execucoes commitadas (21/21 celulas). A quarta, a de proxima-categoria do modelo conjunto,
**nao reproduz de nenhum artefato commitado**.

**(B) Por que importa.** E um numero publicado. Nao ha erro conhecido nele; o que falta e a execucao
que o gerou.

**(C) O que eu preciso de voce.** Dizer se existe um rundir dessa coluna fora deste repositorio. Se
nao existir, isso e uma limitacao de proveniencia a registrar, nao um erro a corrigir. Registrado em
[`LEFT_OUT.md`](LEFT_OUT.md) LO-2 como **aberto**.

### 2.6 O orcamento de tuning de Ch.3 e Ch.4: NAO RECUPERAVEL

**(A) O que falta.** O numero de configuracoes tentadas por estudo.

**(B) Por que importa.** Uma banca pode perguntar quanta busca de hiperparametro ha por tras de cada
resultado.

**(C) O que eu preciso de voce.** Nada a recuperar: nunca existiu um harness de busca e as
configuracoes perdedoras nao foram commitadas. Isso foi estabelecido lendo os dois codebases, nao
presumido. A pendencia e apenas **como dizer isso** se perguntarem. Sugestao: dizer que o
desenvolvimento foi manual e iterativo e que o repositorio preserva a configuracao final, nao o
caminho.

---

### 2.7 `CONSIDERATIONS.md`: uma rodada NOVA que chegou durante esta, e que eu NAO executei

**(A) O que falta.** `src_utils/CONSIDERATIONS.md` apareceu na arvore de trabalho **durante** esta
rodada (modificado 19:04, nao commitado, 1.229 linhas). Ele contem material que nao estava no escopo
que voce me deu:

| Secao | O que e |
|---|---|
| `## Germano` (l. 3-58) | Feedback **verbal** do Germano sobre o Cap. 2, transcrito por voce |
| `## Fabrício` (l. 59-309) | Feedback do **orientador** sobre o Cap. 2 |
| `# Codex Audit — Chapter 2` (l. 310-994) | Auditoria dos dois feedbacks, comparacao contra `exemples/`, e uma lista de trabalho consolidada |
| `# Addendum (2026-07-28)` (l. 995-1229) | O ponto de fluxo do Germano e o item G10 (o achado de conflito de tarefas) |

**(B) Por que importa.** Isto e feedback do **orientador** e de um leitor externo sobre o capitulo de
fundamentacao, com uma lista de trabalho ja consolidada. E a proxima rodada, e e mais importante que
a maior parte do que sobrou aqui. Nao esta perdido: o arquivo esta no disco. Mas nao esta commitado,
e nenhum item dele foi aplicado ao texto.

**(C) O que eu preciso de voce.** Duas coisas. Primeiro, **commitar o arquivo** se ele estiver pronto
(eu deliberadamente nao commitei prosa sua em andamento). Segundo, dizer se quer que eu execute a
lista de trabalho consolidada dele — ela e uma rodada propria, com pesquisa e verificacao, e nao a
comecei porque nao foi o que voce pediu nesta.

**Por que eu nao agi nisso.** O escopo desta rodada foi `CODEX_AUDIT.md` mais as suas decisoes em
`PENDENCIAS.md`. Aplicar 1.229 linhas de feedback novo no fim de uma rodada longa, sem voce ter
pedido, seria exatamente o tipo de improviso que o `AGENT_GUARDRAILS` manda parar e sinalizar.

---

## §3 · Aberto e bloqueado em terceiros

| Item | Bloqueado em | Estado |
|---|---|---|
| Dois membros da banca e a data da defesa | Orientador / PPGCC | Placeholders honestos entre colchetes em `0_main.tex`; **os colchetes aparecem no PDF**, entao nada inventado e apresentado como fato |
| Folha de aprovacao assinada | A defesa | `make ppgc` gera o PDF com o placeholder; a versao assinada o substitui depois |
| Status do MobiWac | Revisores | A redacao e sempre "submitted, under review", em todo o documento. **Nao mudar** ate haver decisao |
| `\finalbuildfirstpage` conferido contra o RASCUNHO | Upload pos-defesa ao AcademicoPG | Agora **8**, derivado das 7 paginas pre-textuais do build de deposito e verificado no render. Confira contra o RASCUNHO quando subir |

---

## §4 · O que auditar primeiro, se voce tiver uma hora

A lista priorizada esta em [`_round6/VERIFY_LIST.md`](_round6/VERIFY_LIST.md), com o comando de
verificacao de cada item. Os cinco de maior consequencia:

1. **O paragrafo D-01 em `apx_b_static_scope.tex`** (p. 99 do build de defesa). E a unica prosa nova
   que faz uma afirmacao publica sobre um resultado co-autorado, e eu errei nele uma vez.
2. **O par Resumo/Abstract** (pp. 2-3). Mais lido que qualquer outra pagina.
3. **As duas sentencas D-02 em `6_conclusion.tex`** (p. 76). Elas mudam o que o numero mais citado do
   Ch.4 licencia.
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
