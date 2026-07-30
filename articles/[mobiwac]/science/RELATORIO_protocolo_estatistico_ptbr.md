# RELATORIO — auditoria do protocolo estatistico (MobiWac 2026)

**Data:** 2026-07-25 · **Alvo:** `articles/[mobiwac]/src/` · **Base:** branch `main` do repositorio
**Nenhum arquivo foi editado nesta passagem.** Este documento resume o relatorio completo
(`AUDIT_statistical_protocol.md`) em portugues e lista as decisoes que dependem de voce.

Todos os numeros abaixo foram lidos dos artefatos ou reproduzidos rodando as convencoes de teste do
proprio repositorio (`scripts/closing_data/superiority_wilcoxon.py`, `region_match_tost.py`) sobre os
vetores por dobra ja versionados. Nada foi calculado de memoria.

---

## 1 · Resumo executivo

O arquivo `ISSUE_statistical_protocol.md` esta **correto nos tres defeitos que aponta**, **errado em um
ponto** e **deixou passar o defeito mais grave**.

| ID | Defeito | Severidade | Situacao |
|---|---|---|---|
| D1 | Desvio Wilcoxon para teste t emparelhado, registrado no repositorio mas nao divulgado no artigo | Baixa | **CONFIRMADO**, com duas correcoes |
| D2 | Superioridade em regiao nunca foi pre-registrada e nao esta em nenhuma familia de multiplicidade | Real | **CONFIRMADO**, e subestimado |
| D3 | A frase "fixed in an analysis plan during development" excede o registro | Real | **CONFIRMADO** |
| **D4** | **"Released with the code" e falso: o pacote publicado nao contem o plano nem o log de desvios** | **Real, o pior** | **Confirmado aqui.** O arquivo ISSUE levantou o item como pergunta aberta (Step 1(e) e criterio 6 da §7), sem afirmar a resposta; a resposta e negativa |

**A boa noticia, verificada e nao apenas presumida: nenhum numero, intervalo ou veredito do artigo muda.**
Reconstrui os dois lados da comparacao a partir dos vetores por dobra versionados e reproduzi cada delta,
cada intervalo de confianca de 90 por cento e cada valor-p do artigo, inclusive a celula mais fina
(regiao em Istambul, +0,1936, IC +0,154 a +0,234) e o pior p ajustado por Holm (1,0e-06). O problema esta
inteiramente nos **rotulos epistemicos**, nao nos resultados.

---

## 2 · D4, o defeito novo (e o mais grave)

A secao 5.3 afirma que o plano de analise "are released with the code". A nota de rodape 1
(`01_introduction.tex:22`) aponta para `github.com/VitorHugoOli/PoiMtlNet/tree/mobiwac`.

Essa branch (local `689b0d6e`, remota `673fbd27`, commit unico de 2026-07-08) **nao publica nenhum arquivo
sob `docs/`**: nao ha `STATISTICAL_PROTOCOL.md`, nao ha `stats_n20/RESULTS.md`, nao ha `log.md`, nao ha
`joint_best/`. Ela publica apenas os tres scripts de teste. Um revisor que siga a nota de rodape **nao
consegue ler o plano nem o log de desvios**.

Pior: o `README.md` publicado, secao 6, diz ao revisor que o `superiority_wilcoxon.py` incluido implementa
"the pre-registered tests", listando entre elas "region superiority (FL/CA/TX)". Ou seja, a alegacao nao
registrada aparece justamente no unico artefato que o leitor alcanca.

**Veredito da sentenca P5 (`05_setup.tex:42`): UNSUPPORTED** na metade sobre publicacao. A metade sobre
"fixed during development" se sustenta: o protocolo foi commitado em 2026-06-21 (`c96c67e3`) e as primeiras
celulas do board entraram em 2026-06-22, portanto o documento antecede a abertura dos resultados.

---

## 3 · Correcao ao arquivo ISSUE

O arquivo diz que o desvio Wilcoxon para t foi "forcado pela disponibilidade de artefatos", porque os
vetores por dobra do lado MTL estariam apenas no A40. **Isso era verdade em 2026-07-13; nao e mais verdade
hoje.** O re-scoring joint-best versionou os vetores por dobra dos seis conjuntos no lado MTL
(`joint_best/data/j1_results.json` e `catx_v17_n20/joint_best/`, ambos de 2026-07-13). A restricao mudou de
lugar: a unica lacuna restante esta no lado **dedicado**, em Istambul (secao 5 deste relatorio). Qualquer
texto de remedio que repita "os valores por dobra nao estao disponiveis" ficaria desatualizado.

Segunda correcao, menor: o motivo registrado no repositorio para reportar o t e o **piso de potencia**
(com quatro sementes o Wilcoxon exato nao desce de 0,0625, qualquer que seja o tamanho do efeito), e nao o
argumento de pseudo-replicacao que o arquivo ISSUE oferece. O argumento de pseudo-replicacao e defensavel e
vale para a carta-resposta, mas nao e o que o artefato registra, e o artigo nao deve atribui-lo ao plano.

---

## 4 · Evidencia mais forte para D2 do que a do arquivo ISSUE

O arquivo ISSUE argumenta que a superioridade em regiao e post-hoc por silencio do protocolo. Existe um
fato mais duro: o script `superiority_wilcoxon.py`, que nomeia FL/CA/TX como "the beats", foi commitado pela
primeira vez em `1e3449e6`, **2026-06-25**. Os tetos dedicados de regiao de FL e CA entraram em **2026-06-22**
e o de TX em **2026-06-24**. Ou seja, a escolha dos tres conjuntos foi feita **depois** que essas celulas
ja podiam ser lidas. Esse script esta no pacote publicado.

A mesma afirmacao nao registrada aparece em outros dois lugares versionados:
`stats_n20/m1_stats_n20.py:333` (que imprime "the pre-registered reg-'beats' family", e a frase chega ao
`m1_full_output.txt:83,91`) e `stats_n20/RESULTS.md` secao 1b, que trata o script como autoridade de
registro. Nenhum dos tres tem respaldo no protocolo.

---

## 5 · As quatro perguntas

**P1. Algum numero reportado foi afetado? NAO, confirmado.** Reproducao independente dos seis conjuntos nas
duas tarefas. Duas notas de arredondamento, nenhuma delas defeito: o limite inferior do IC de FL reproduz
0,665 (o artigo imprime 0,67) e o delta de AZ e -0,0031 (o artigo imprime 0,00, e o ledger proibe mexer em AZ
em qualquer direcao).

**P2. As quatro alegacoes de "outperforms" em regiao estao dentro de alguma familia de correcao, registrada
ou aplicada? NAO, nos dois sentidos.** O protocolo, secao 5.2, enumera exatamente uma familia de
superioridade, as seis celulas de categoria, e exclui explicitamente as celulas TOST de regiao. Na execucao,
`stats_n20/RESULTS.md` e `m1_full_output.txt` aplicam Holm somente as seis celulas de categoria; os
valores-p de regiao saem sem correcao. Logo, o "with a Holm correction across the comparisons" da secao 5.3
sugere uma cobertura que a analise nao tem.

*Quanto custaria corrigir, se voce quiser:* na familia propria de quatro celulas (Holm m=4, teste t no
nivel de semente), os p ajustados ficam em 7,2e-04 (Istambul), 4,9e-05 (FL), 9,4e-08 (TX) e 5,0e-08 (CA).
Juntando as dez celulas em uma familia unica (m=10), o pior p ajustado e 7,2e-04. **Nenhum veredito se move
em nenhum dos esquemas testados.** Essa e a defesa mais barata e completa, e nao exige o teste pre-registrado.

**P3. A frase "fixed in advance" excede o que o protocolo fixou? SIM, de duas formas.** (i) Escopo: o plano
fixou a atribuicao **por tarefa** (categoria para superioridade, regiao para nao-inferioridade), e a
gramatica da frase ("where the joint model was expected to outperform") sugere um julgamento a priori **por
conjunto de dados** que nao esta registrado em lugar algum. (ii) Cobertura: o plano nao cobriu superioridade
em regiao, portanto as quatro alegacoes de "outperforms" ficam fora da atribuicao que a frase anuncia.

Detalhe relevante: essa frase e **nova**. O `git log -S` datou "analysis plan during development" em
`158de7d1`, **2026-07-20**, o mesmo commit que trocou superioridade para o teste t. O texto anterior dizia
apenas "we test superiority with a paired Wilcoxon signed-rank test over the folds", mais proximo do
registro no nome do teste e sem nenhuma alegacao de publicacao.

**P4. O desvio Wilcoxon para t esta divulgado em algum lugar visivel ao leitor? NAO.** O artigo nomeia
somente o teste t e nunca menciona o Wilcoxon registrado, de modo que o leitor nao tem como perceber a
substituicao. A divulgacao existe apenas em `v17_completion/stats_n20/RESULTS.md`, que nao esta no pacote
publicado. O log obrigatorio pelo proprio protocolo (`docs/studies/closing_data/log.md`) tambem nao o
carrega, e a citacao "per protocol §8's powered-t deviation" no cabecalho do `RESULTS.md` e uma referencia
pendente, porque a secao 8 do protocolo nao tem essa entrada.

---

## 6 · Viabilidade do teste pre-registrado hoje

**Cinco de seis para categoria, seis de seis para regiao. A familia registrada de seis conjuntos NAO pode
ser rodada hoje.**

| Conjunto | MTL joint-best por dobra | Teto categoria por dobra | Teto regiao por dobra |
|---|---|---|---|
| AL, AZ, FL, CA, TX | sim | sim | sim |
| **Istambul** | sim | **NAO** | sim |

**Artefato exatamente ausente.** O teto dedicado de categoria de Istambul existe no repositorio apenas como
quatro escalares por semente, em
`docs/studies/closing_data/v17_completion/h3_istanbul/step3_runs/cat_ceil_s{0,1,7,100}.txt`
(54,7063 / 54,8632 / 54,7705 / 54,6101; media 54,7375, o valor do board 54,74). Os vetores por dobra ficam
no sidecar `stl_cat_ceiling_score.json` (chave `cat_per_fold`), escrito por
`scripts/closing_data/score_stl_cat_ceiling.py` dentro de cada rundir
`results/check2hgi_dk_ovl/istanbul/next_*_<pid>/`. Esses rundirs estao no gitignore e nao estao nesta
maquina. **Atencao:** os dois JSONs de teto de categoria de Istambul que estao versionados sao celulas
diferentes e nao podem substituir (medias 53,20 e 52,10, nao os 54,74 do board).

**Custo para obter.** (a) Copiar os quatro sidecars `stl_cat_ceiling_score.json` da maquina que guarda os
rundirs de Istambul, se ainda existirem: minutos, sem GPU, sem retreino. (b) Re-rodar as quatro execucoes
dedicadas de categoria de Istambul (`next_gru`, 5 dobras, 50 epocas, sementes {0,1,7,100}, engine
`check2hgi_dk_ovl`) e re-pontuar: trabalho de GPU da ordem da passagem original do H3 step-3. Antes de um
prazo de camera-ready, so a opcao (a) vale a pena.

**O que o teste pre-registrado entrega onde ele roda** (Wilcoxon emparelhado unilateral exato, por dobra,
n=20, vetores joint-best): categoria em AL/AZ/FL/CA/TX e regiao em Istambul/FL/TX/CA, todas com **20/20
dobras positivas e p = 9,54e-07**, que e o piso exato de n=20 (1/2^20). Isso confirma o "p ~ 9,5e-07,
20/20 positive" citado no arquivo ISSUE para a opcao R2. **Nao reporte uma familia de cinco de seis como "o
teste pre-registrado": a familia registrada tem seis conjuntos, e uma familia parcial e outra familia.**

---

## 7 · Remedio recomendado

Custos de linha medidos, nao estimados: o paragrafo da secao 5.3 renderiza a 51,9 caracteres de fonte por
linha tipografica (1.920 caracteres em 37 linhas). A folga medida na compilacao atual de 8 paginas e de
**85,3 pt na ultima coluna, cerca de 7 linhas de corpo** no passo de 11,9 pt.

| Opcao | Veredito |
|---|---|
| **R1** reparo minimo de honestidade | **RECOMENDADA**, estendida por R6 |
| R2 rodar o teste pre-registrado | **Nao agora.** Bloqueada em Istambul (secao 6). Alem disso troca um desvio conservador e documentado por uma objecao de pseudo-replicacao que o artigo passaria a ter de responder. Manter como trabalho de camera-ready se os sidecars forem recuperados. |
| R3 reportar as duas bases | **Nao agora**, mesmo bloqueio, e gasta linhas para dizer duas vezes o que uma oracao diz uma vez. |
| R4 manifesto completo do REV-007 | **Correta para a dissertacao e o camera-ready.** Nao e um patch de texto. |
| R5 remover as alegacoes de "outperforms" em regiao | **Rejeitada.** Subestima um resultado com 20/20 dobras positivas e ICs longe de zero, e colide com a lei de verbos (essas celulas **tem** teste de superioridade que passa; o que falta e registro, que e questao de rotulo). |
| **R6 (nova)** R1 mais correcao aplicada a familia de regiao | **RECOMENDADA como opcao sua.** Converte "em nenhuma familia" em "em familia propria, divulgada". Custo +0,6 linha. |
| **R7 (nova)** consertar o pacote publicado | **OBRIGATORIA em qualquer cenario.** Sem ela, P5 continua falso e o desvio continua invisivel. Zero linhas de artigo. |

**Recomendacao: R1 + R6 no artigo, R7 no repositorio, R4 para a dissertacao e o camera-ready.**

### Texto exato proposto (em ingles, como entra no `.tex`)

**Edicao 1 — `articles/[mobiwac]/src/sections/05_setup.tex:42`.** Substituir o trecho que vai de
"We fix the assignment in advance" ate "the paired difference." por:

> A written analysis plan, fixed during development and before any result was read, assigned one test to each
> task: superiority for next-category, non-inferiority for next-region, with the two-point margin pinned there.
> The plan assigned the tests per task, not per dataset, and it did not cover next-region superiority, so the
> four next-region gains of Section~\ref{sec:results-part2} are secondary results outside it. The plan
> registered a paired Wilcoxon signed-rank test; at four seeds its exact one-sided $p$ cannot fall below
> $0.0625$, so we report a paired $t$ on the per-seed means instead, with the 90\% confidence interval of the
> paired difference. The plan and this departure are released with the code.

**Edicao 2 — mesmo paragrafo.** Trocar "with a Holm correction~\cite{holm1979} across the comparisons." por:

> with a Holm correction~\cite{holm1979} across the six next-category comparisons and, separately, across the
> four next-region comparisons.

**Edicao 3 (somente se R6 for adotada) — `06_results.tex:77-78`.** Acrescentar apos
"(paired $t$, corrected $p<0.001$)":

> ; the four next-region gains hold under their own Holm correction as well (corrected $p<0.001$).

**Edicao 4 (opcional) — legenda da `tbl3_results.tex`.** Acrescentar ao final:

> The next-region improvements are secondary results, outside the analysis plan
> (Section~\ref{sec:setup-metrics}).

**Conformidade com a lei de escrita:** sem travessao, ingles americano, sem contracoes, nomes canonicos das
tarefas, "dedicated" intocado, AZ nao promovido, alegacao de escala intocada, resumo e lista de contribuicoes
intocados (portanto a linha do ledger sobre o TOST suavizado no resumo nao e reaberta). O verbo "outperforms"
permanece ligado as quatro celulas de regiao, que **tem** teste de superioridade aprovado; o que a Edicao 1
acrescenta e que o teste nao foi registrado previamente. **Nenhum numero muda.**

### Custo e o corte que financia

| Item | Linhas |
|---|---:|
| Edicao 1 (712 caracteres substituindo 475) | +4,6 |
| Edicao 2 (escopo do Holm) | +0,6 |
| Edicao 3 (opcional, R6) | +0,6 |
| **Corte de financiamento** | **-2,4** |
| **Liquido com Edicoes 1+2** | **+2,8** |
| **Liquido com Edicoes 1+2+3** | **+3,4** |
| Folga medida na compilacao atual | cerca de 7 |

**O corte.** Remover de `05_setup.tex:42` a frase "For scale, with 520 to 8,501 regions, a random top-ten
guess includes the true region at most about two percent of the time." (125 caracteres, 2,4 linhas). E uma
duplicata: `06_results.tex:35-36` ja diz a mesma coisa, e a secao 6.2 e onde o leitor precisa da ancora de
escala. Se quiser mais folga, a Edicao 4 pode cair (e opcional) e a floreada de potencia da secao 5.3 pode
encurtar, mas nada disso e necessario na folga medida.

### Dissertacao, capitulo 5

`articles/dissertacao/src/chapters/5_mobiwac.tex:355` carrega a frase identica e aceita as Edicoes 1 e 2
literalmente, trocando os rotulos para `\ref{sec:mobiwac:results-part2}` e `\ref{sec:mobiwac:setup-metrics}`.
Sem pressao de paginas, valem duas adicoes que o artigo nao pode pagar: o caminho do plano no repositorio
publicado e a mencao explicita ao log de desvios.

### Carta-resposta (versao curta, em ingles)

> Our analysis plan, fixed during development and before any result was read, assigned superiority to
> next-category and non-inferiority to next-region, and pinned the two-point equivalence margin; it registered
> a paired Wilcoxon signed-rank test. Because the exact one-sided Wilcoxon cannot fall below 0.0625 at four
> seeds, we report a paired $t$ on the per-seed means, a departure recorded in the deviation log released with
> the code. The four next-region improvements are secondary results outside that plan: we now say so in
> Section V-C, and we report them under their own Holm correction, where every one of them holds at a corrected
> $p<0.001$. No reported estimate, interval, or verdict changes.

---

## 8 · Decisoes que dependem de voce

Nada foi editado. As seis decisoes abaixo sao suas.

1. **Aprovar ou nao as Edicoes 1 e 2** em `05_setup.tex:42` (e o corte da frase duplicada que as financia).
   E o remedio minimo que torna verdadeira cada afirmacao do artigo sobre pre-registro.
2. **Adotar R6 ou nao (Edicao 3).** Aplicar Holm a uma familia de regiao que o protocolo **nao registra** e
   uma escolha metodologica, nao um ajuste de redacao. Divulgar uma familia aplicada nova precisa da sua
   assinatura. Todos os p ajustados ficam abaixo de 0,001, portanto nada se move.
3. **Executar R7 no repositorio.** Adicionar `STATISTICAL_PROTOCOL.md` e `stats_n20/RESULTS.md` (ou um
   `ANALYSIS_MANIFEST.md` de uma pagina) a branch `mobiwac`, e corrigir a linha do `README.md` secao 6 que
   anuncia "region superiority (FL/CA/TX)" como pre-registrada. Sem isso a frase "released with the code"
   continua falsa, com qualquer texto que voce escolha.
4. **Os sidecars de Istambul existem ainda?** Se os quatro `stl_cat_ceiling_score.json` estiverem na maquina
   que guardou os rundirs, R2 deixa de estar bloqueada e o teste pre-registrado completo passa a ser
   possivel para o camera-ready. Se nao existirem, R2 sai de cena a menos que voce autorize o re-treino.
5. **Ledger.** Nao existe linha no ledger sobre linguagem de pre-registro. A Edicao 1 muda como o artigo
   descreve o proprio plano, o que fica proximo o bastante da linha do verbo de veredito para merecer sua
   aprovacao explicita em vez de tratamento editorial. **Nenhuma linha do ledger foi reaberta por mim.**
6. **Contradicao entre capitulos da dissertacao.** `2_fundamentals.tex:448-450` diz que o Wilcoxon
   emparelhado "is the test that licenses the verb outperforms", contradizendo o teste t do capitulo 5. Esta
   fora do escopo desta passagem, mas precisa se mover na mesma onda de edicao, senao o documento se
   contradiz entre capitulos. **Sinalizado, nao redigido.**

---

## 9 · [VERIFICAR] — o que nao consegui resolver pelos artefatos

1. **Vetores por dobra do teto de categoria de Istambul.** Fora do repositorio (secao 6). Se os sidecars
   ainda existem na maquina de execucao e pergunta para voce.
2. **Procedencia dos numeros da entrada de 2026-07-18 na secao 8 do protocolo.** Ela imprime exatamente os
   ICs que o artigo carrega, e eu reproduzi todos eles a partir dos vetores versionados, portanto os valores
   sao solidos. Mas nenhum script versionado emite essa entrada: o `m1_stats_n20.py` le as fontes diag-best e
   o `score_joint_best.py` apenas pontua celulas. O gerador da rodada estatistica joint-best nao esta na
   arvore. Isso nao afeta numero algum; e a forma concreta do "no single manifest" do REV-007 e e o que R7
   resolveria.
3. **A frase "released with the code" ja foi verdadeira em algum momento?** As duas pontas da branch
   `mobiwac` (local e remota) nao publicam `docs/`, e a branch tem um commit unico de 2026-07-08, anterior a
   frase (2026-07-20). Nao encontrei evidencia de upload suplementar separado, mas nao posso descartar que
   voce planeje um (por exemplo, um arquivo suplementar no EDAS). Se esse upload existir e contiver o
   protocolo, a segunda metade de P5 passa a ter respaldo e resta apenas divulgar o desvio.
4. **Docstrings de `m1_stats_n20.py` e `superiority_wilcoxon.py`.** Ambas afirmam um registro que o protocolo
   nao contem, e ambas estao no pacote publicado. Corrigi-las e trabalho de repositorio fora desta passagem;
   nao editei nada. O `stats_n20/RESULTS.md` secao 1b repete a mesma afirmacao, e a citacao "per protocol §8's
   powered-t deviation" no cabecalho e referencia pendente.


---

## 10 · ADENDO (2026-07-25, apos sua aprovacao) — o que foi executado

Voce aprovou R1 + R6 (artigo), R7 (repositorio) e o espelho na dissertacao. Durante a execucao o bloqueio
da secao 6 desapareceu, o que melhorou o resultado.

**Istambul nao esta mais faltando, e nao precisou de GPU.** Os quatro sidecars
`stl_cat_ceiling_score.json` estavam no A40 (`ssh:nespedgpu`), nos rundirs
`next_lr1.0e-04_bs2048_ep50_20260706_*_{3856035,3861493,3866919,3872209}`, com as tags
`h3ist_cat_s{0,1,7,100}`. As medias por semente reproduzem exatamente os escalares versionados
(54,7063 / 54,8632 / 54,7705 / 54,6101; media n=20 = 54,7375, o valor 54,74 do board), portanto **nenhum
re-treino foi necessario** e a celula do board fica intacta. Versionados em
`docs/studies/closing_data/v17_completion/h3_istanbul/step3_runs/cat_ceiling_perfold/`. O LIMITS #2 do
`stats_n20/RESULTS.md` esta fechado.

**R2 rodou, e a secao 6 deste relatorio esta superada.** O teste pre-registrado, na base registrada
(Wilcoxon emparelhado unilateral por dobra, n=20, protocolo §2; Holm m=6, §5.2), cobre agora a familia
completa de seis conjuntos: **20/20 dobras positivas** em todas, p exato = 9,5367e-07 (o piso de n=20,
1/2^20), **Holm ajustado 5,7220e-06, todas rejeitam a α = 0,05**. As quatro celulas de regiao rejeitam na
familia propria m=4 (ajustado 3,8147e-06). Gerador com portao de reproducao 24/24:
`stats_n20/m2_prereg_perfold.py`; saida `m2_prereg_output.txt`. **Nenhum veredito, estimativa ou intervalo
se moveu.**

**Consequencia para o remedio.** A recomendacao de adiar R2 caiu; em vez disso, R1 ganhou uma oracao de
corroboracao. O artigo agora nomeia o desvio **e** informa que o teste registrado concorda, o que responde
a objecao do revisor de forma mais completa do que qualquer das duas opcoes isoladas. O risco de
pseudo-replicacao que tornava R2 pouco atraente fica neutralizado ao manter o teste t no nivel de semente
como base primaria (os ICs de regiao sao calculados nela) e o Wilcoxon por dobra como corroboracao, em vez
de trocar a base primaria.

**Custo de linhas, medido no PDF compilado.** Minha estimativa (+2,8 liquido) estava errada: a primeira
compilacao saiu com **9 paginas**, porque a folga projetada de 7 linhas nao sobreviveu ao refluxo dos
floats. Fechar isso exigiu seis cortes adicionais de duplicacao, cada um removendo um fato ja dito em outro
lugar do artigo: a oracao final da convencao de epoca, a frase "All six datasets are measured with four
seeds" da §6.2 (a §5.3 ja define), a clausula "both models average four seeds over five folds" da §6.3, a
floreada de potencia da §5.3, mais dois aprimoramentos do texto novo. **Build final: 8 paginas, 0
referencias indefinidas, 0 overfull, bibtex limpo.** Licao para a proxima passagem: em um build IEEE no
limite, aritmetica de caracteres por linha nao prediz numero de paginas; compile antes de prometer
orcamento.

**Onde o registro vive agora (R7 feito).** O commit `09b01923` na branch `mobiwac` adiciona
`analysis_protocol/` (protocolo, log de desvios, analise executada, registro da convencao de epoca, tetos
dedicados, saida do teste registrado e os vetores por dobra de Istambul), mais `m2_prereg_perfold.py` e
`score_joint_best.py`, e reescreve a secao 6 do README. O script publicado foi testado a partir de um
checkout limpo (`git archive`) da branch e reproduz todos os numeros. **O commit e local: nao foi enviado
para o remoto.**

**O §8 do protocolo esta fechado.** As entradas **D-1 a D-4** estao agora em
`docs/studies/closing_data/log.md`, o local que o §8 exige, e o §8 aponta para elas. A referencia pendente
do `RESULTS.md:20` tem, pela primeira vez, um referente.

**Continua aberto.** O gerador das estatisticas joint-best no nivel de semente ainda nao esta na arvore (só
o `m2_prereg_perfold.py` cobre a familia registrada por dobra), e o `ANALYSIS_MANIFEST.md` de uma pagina
segue como item de trabalho do REV-007. As tres afirmacoes em docstrings (`superiority_wilcoxon.py`,
`m1_stats_n20.py:333`, `RESULTS.md §1b`) estao sinalizadas no README publicado e na entrada D-4, mas nao
foram editadas: corrigir a prosa de um registro de resultados e uma passagem separada e neutra em relacao
as alegacoes.

**Uma decisao sua permanece:** o `git push` da branch `mobiwac` para o remoto. Deixei o commit local.
