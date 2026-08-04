# PENDENCIAS.md — o que falta, e de quem depende

**Fila viva. Se um item nao espera nada de ninguem, ele nao mora aqui.**

## Como este arquivo funciona

**Cada item tem a mesma forma, e ela e curta:**

```
### N.M Titulo de uma linha
**O que e.** Uma a tres frases: o achado, com o numero medido.
> **DECISAO SUA:** o que falta, com as opcoes e o custo de cada uma.
*Forense: ponteiro para o relatorio de rodada.*
```

**Onde cada coisa vive.** O tracker carrega a **decisao**; a **forense** (como o defeito foi descoberto, qual
instrumento mentiu, o que cada commit mediu) vai para `_round8/`. Em 2026-07-30 seis itens carregavam 34 mil dos 55 mil
caracteres do arquivo, quase tudo forense: foi para
[`_round8/29_pendencias_detail.md`](_round8/29_pendencias_detail.md), **nada apagado**, e o arquivo caiu de 67 mil para
37 mil.

**Para ADICIONAR um ponto seu:** escreva embaixo do item, comecando a linha com `> DECISSAO:` (ou
`> DECISAO:`). Eu leio isso como sua palavra final e nao reinterpreto. Se voce nao tiver numero de item, escreva no fim
do §2 com um titulo qualquer — eu numero e coloco no lugar.

**Para FECHAR um item:** ele sai daqui e vai para `_archive/PENDENCIAS_RESOLVIDOS.md` **com o motivo de saida no topo do
bloco**. O gate `check_tracker_refs.py` falha se um item desaparecer sem chegar ao arquivo — tres foram perdidos assim,
e voce achou dois deles lendo o arquivo. **Nao renumere:**
comentarios no fonte citam estes numeros, e um buraco na numeracao e melhor que um ponteiro errado.

**Ordem das secoes:** §2 (voce) -> §3 (terceiros) -> §4 (o que auditar primeiro). Deliberada: o que depende de voce vem
antes. **§5 removida em 2026-08-03** (retirado; os onze itens que apontava continuam em
`_archive/PENDENCIAS_RESOLVIDOS.md`, re-medidos e intactos).

**O §6 saiu deste arquivo em 2026-08-03, a seu pedido, e nao foi perdido.** Ele entrou em 2026-07-30 substituindo o §2.8
e carregou vinte e seis itens vindos do `CONSIDERATIONS.md`; **os vinte e seis foram respondidos por voce** e estao em
`_archive/PENDENCIAS_RESOLVIDOS.md`, cada um com o motivo da saida no topo do bloco e sob o cabecalho que registra o
encerramento da secao inteira. A numeracao 6.1 a 6.26 **nao** foi reaproveitada, e as dezenove citacoes que apontavam
para ca foram repontadas para o arquivo na forma historica que o `check_tracker_refs.py` reconhece. O §6 seguiu a mesma
trajetoria do §2.8: deixou de pedir decisao e virou registro.

---

## §2 · Aberto e bloqueado em VOCE

> **LIMPO EM 2026-07-30, a seu pedido.** Cinco itens desta secao estavam **de fato fechados** e foram
> movidos para `_archive/PENDENCIAS_RESOLVIDOS.md` com o motivo de saida no topo de cada bloco:
> **2.2** (push publicado, verificado por hash contra o remoto — o resto virou 2.16), **2.3** (fechado
> pela sua frase *"podemos fechar esse ponto"*), **2.7** (orcamento de tuning nao-recuperavel,
> registrado em `LEFT_OUT.md`), **2.13** (o comando contava 4 a mais por ser cego a comentarios;
> corrigido) e **2.17** (afirmacao falsa minha, corrigida com nota de git em `a07e547b`).
>
> **Os buracos na numeracao — 2.2, 2.3, 2.7, 2.13, 2.17 — sao esses cinco, e nao perdas.** Nao
> renumerei os que ficaram: seis comentarios no fonte e o `_round6/VERIFY_LIST.md` citam estes numeros,
> e renumerar transformaria cada citacao num ponteiro para o item errado, que e pior que um buraco.
> O gate `check_tracker_refs.py` agora falha se um item sair daqui sem chegar ao arquivo.
>
> **O que sobrou aqui espera VOCE, nao a mim.** Onde a medicao esta completa, o bloco `(A)/(B)/(C)`
> diz exatamente o que falta e quanto custa cada saida.

### 2.1 Os marcadores `[NEEDS SIGN-OFF]` no fonte — **56** medidos em 2026-08-02, agora com mapa item por item

**O que e.** Pontos do fonte marcados como precisando do seu aval. Nenhum bloqueia build, e **nenhum aparece no PDF**:
todos vivem em comentario `%`. **O numero anda** — tracks paralelas removem marcadores conforme voce decide.

**Novo em 2026-08-02: [`src_utils/NEEDS_SIGN_OFF.md`](NEEDS_SIGN_OFF.md)** traduz os 56 marcadores para PT-BR, um por
um, com contexto, a pergunta exata e um espaco `> **SUA DECISAO:**` para voce responder — o mesmo padrao deste arquivo.
Cada item foi conferido contra o fonte vivo (`grep` na linha exata) antes de entrar no mapa. Quando um item for
resolvido la, ele sai daquele arquivo e o `[NEEDS SIGN-OFF]` correspondente sai do `.tex`. Confie no comando, nao no
titulo:

```bash
cd /Users/vitor/Desktop/mestrado/ingred/articles/dissertacao
grep -rc "\[NEEDS SIGN-OFF" src --include="*.tex" --exclude-dir=build | grep -v ":0$" | sort -t: -k2 -rn
```

Medido assim em 2026-07-30 sobre `5c074a2a` mais a arvore de trabalho: **54 marcadores em 21 arquivos** (52 com corpo
`[NEEDS SIGN-OFF: ...]` e 2 retrovisores nus `[NEEDS SIGN-OFF]`); 58 se `src/build/` entrar, porque
`build/fmt/_body.tex`
e copia gerada — dai o `--exclude-dir=build`.

*(O comando que estava aqui — `grep -rn ... | grep -v ":\s*%"` — imprimia **zero linhas** e saia `rc=1`: o `-v` casa o
`%` do comentario em que cada marcador vive, entao removia justamente tudo o que devia contar. E nenhum gate conta estes
marcadores: `check_verify_list` executa blocos documentados, nao mede esta contagem, ao contrario do que este item
afirmava.)*

**Tres tem prioridade** (afirmam algo sobre trabalho publicado ou co-autorado): o paragrafo corrigido do Apendice B
sobre o Cap. 3, o numero limitado do Cap. 4 na conclusao, e a frase de reprodutibilidade enfraquecida. Estao detalhados
em `_round6/VERIFY_LIST.md` A1, A2 e A3.

> **DECISAO SUA:** ler os 53 e me dizer quais aprova. Nao precisa ser de uma vez — se me der os tres
> prioritarios, eu removo os marcadores deles e mantenho os outros 50.

*Forense (a tentativa de push destrutiva, o worktree, os artefatos divergentes): agora e o item 2.16 e o corpo integral
esta em [`_round8/29_pendencias_detail.md`](_round8/29_pendencias_detail.md).*

### 2.5 O tamanho de tipo das duas figuras de arquitetura — autorizado, mas eu nao consigo executar

**Voce autorizou:** *"pode aumentar, mas mantenha o espaco ja ocupado pela imagem... mantendo a proporcao"*, e observou
que o contraste hoje ja deixa legivel.

**Nao consigo fazer daqui:** nao ha `drawio` nem `inkscape` neste ambiente. **Os dois `.drawio` estao no repositorio** —
`figures/mtlnet_poi_new.drawio` (13.640 B, `fontSize=14`) e `figures/courb/arquitetura_modelo.drawio`
(14.588 B, `fontSize=13`), medidos em 2026-07-30 com `find . -name '*.drawio'` (quatro no repo inteiro). A receita esta
em `_round6/12_figures.md` (subir `fontSize` para ~20 e reexportar na mesma largura em pixels).

*(Este bloco dizia **"so 1 dos 2"**. Era falso, e o commit `b89a9876` ja tinha diagnosticado exatamente isso — o
instrumento era `ls src/figures/*.drawio`, glob nao-recursivo que nao ve `figures/courb/` — mas a correcao nao chegou ao
arquivo. Tamanhos de tipo medidos, no `LEFT_OUT.md` LO-6: **45,3%** do corpo no do Cap. 3 e **44,4%** no do Cap. 4,
contra corpo de 11,96 pt. O raster do Cap. 3 e byte-identico ao publicado do CBIC, conferido por sha256.)*

> **Seu, quando quiser:** reexportar as duas no Draw.io e me passar os PNG — eu troco e remeco o tipo na
> pagina. **Opcional**, pela sua propria observacao sobre o contraste.

### 2.27 A arvore revisada do autor entrou no `src`, e o que ficou aberto nela

**(A) O que e.** Em 2026-08-02 o autor entregou `src_clean`, lido e editado por ele. O merge esta em
`src_utils/_round9/49_clean_tree_merge.md`. A prosa dele entrou byte a byte nos 54 arquivos; a camada de comentario do
`src` (4.114 linhas, 275 blocos, 54 marcadores `[NEEDS SIGN-OFF]`) foi reancorada por cima. 228 dos 275 blocos
reancoraram exatamente.

**(B) O que fica aberto para voce.**

1. **28 blocos marcados `[ORPHANED 2026-08-02]`** — eram 47, e voce resolveu 19 no commit `45c75611`
   ("remove orphaned comments and clean up LaTeX files"). Medido em 2026-08-02:
   `grep -rho 'ORPHANED 2026-08-02' src --include='*.tex' --exclude-dir=build | wc -l` = **28**. Cada um anota uma frase
   que a sua revisao reescreveu ou cortou; nenhum foi apagado por mim. A tabela original dos 47 esta no relatorio 49.
   Sao seus para manter, reescrever ou deletar; um agente nao deve decidir isso.

2. **54 marcadores `[NEEDS SIGN-OFF]` continuam abertos**, distribuidos em 21 arquivos, com 7 em
   `2_fundamentals.tex`, 8 em `6_conclusion.tex` e 6 em `apx_a_contributions.tex`. Sao afirmacoes que nenhum artigo
   publicado sustenta e que dependem da sua assinatura.

3. **A grafia do termo central foi uniformizada em "multitask"**, como manda `GLOSSARY.md:130`. As 36 ocorrencias
   hifenizadas que restam sao TITULOS CITADOS no `references.bib` e nao podem ser alteradas sem falsear as fontes.

4. **`apx_g_hgi_tuning.tex` e um apendice novo seu**, que recebeu a varredura do peso do HGI que saiu do capitulo 2.
   Renderiza na p. 106 da defesa. Ele nao esta no `main_extra`, so no volume principal.

**(C) Status.** Builds 106/103/107/22 pp, zero erros, zero referencias indefinidas; 25 gates e o selftest em rc=0, lidos
diretamente.

### 2.28 Varredura de auditoria de 2026-08-02: 14 itens fechados, 5 abertos, 2 surpresas

**(A) O que foi feito.** Voce pediu para auditar cada item do §2 e do §5, medindo o estado do documento em vez de ler o
cabecalho do proprio item. Os 19 itens em escopo foram medidos contra a arvore em `45c75611`
mais a arvore de trabalho. **14 fecharam e foram para
[`_archive/PENDENCIAS_RESOLVIDOS.md`](_archive/PENDENCIAS_RESOLVIDOS.md)** com a evidencia e a sua decisao preservadas
verbatim; 51 citacoes a esses itens foram reapontadas para o arquivo, mais 3 no `GLOSSARY.md`
e neste arquivo, e o gate `check_tracker_refs` voltou a rc=0.

**(B) As duas surpresas, e as duas vao nas duas direcoes.**

1. **`2.26` estava dado por resolvido e nao estava.** Voce escreveu "Aplique o R15-10 e o R15-09" e nenhum dos dois
   havia sido aplicado: `"Two patterns stand out in the data."` e `"Settling that needs"` continuavam na prosa viva do
   apendice do cosseno. **Aplicados agora** (2026-08-02): `"The figure shows two patterns."` e
   `"Answering that question needs the same diagnostic"`.

2. **`EX-9` dentro do `2.23`: a sua revisao desfez a sua propria decisao.** Voce escreveu "nao aplique o EX-9", cuja
   familia eram quatro frases (`deserves one statement`, `worth reporting`, `needs saying`,
   `worth stating`). Todas as quatro **sairam** da prosa viva; `git log -S` mostra duas saindo no seu proprio
   `src_clean` (`807183c1`). Voce foi consultado e decidiu que a sua leitura com o texto na mao superseda a decisao
   anterior. Registrado como SUPERSEDIDO, nao como aplicado. **E o meu probe nao pegou isso:** o `A23-EX9` vigiava
   `"Pareto front"`, que continua no texto, em vez das frases que a decisao protegia — passava enquanto a decisao era
   desfeita. Reapontado para a definicao de fronteira de Pareto que voce de fato manteve, e validado nos dois sentidos.

**(C) Um item que o tracker dava por aberto e estava aplicado.** O `2.20` (italico em ingles corriqueiro no Cap. 4): a
sua opcao 2 esta aplicada. `\textit` na prosa viva do Cap. 4 = **48**, contra 157 no fonte em `5c074a2a`; os
sobreviventes sao os 7 nomes de categoria, nomes de modelo e substantivos proprios. Duas formas arguveis sobraram
(`one-hot`, `skip-gram`) e nao mexi nelas.

**(D) Segunda passagem, 2026-08-02: o §5 retirado, e o 2.21 e o 2.24 fechados.** O §5 foi **re-medido** depois da fusao
e virou ponteiro: os onze itens estao no arquivo e as conclusoes sobreviveram (o comando do proprio banner ainda
reproduz o que ele afirmava). O **2.21** fechou — o termo que o seu orientador marcou,
`license the verbs`, ja tinha saido do Cap. 2 na sua revisao, e a metafora foi trocada por `supports` nos tres sitios
vivos restantes mais a glosa do `GLOSSARY`; os usos em `apx_e_ethics.tex` ficaram, porque ali
`license` e licenca de software de verdade. O **2.24** fechou nas duas metades: a norma ABNT NBR 10520:2023 esta na §1
do `WRITING_LAW` com gate e self-test, dois fragmentos foram corrigidos e a citacao de frase completa ficou por sua
isencao; e o `towards` fica como esta por sua decisao, com a entrada do
`OPEN_REGISTER` como registro permanente dela.

**Sobram tres itens seus:** `2.1`, `2.5` e `2.27`.

*Forense: [`_round9/50_pendencias_audit.md`](_round9/50_pendencias_audit.md), com a medicao de cada um dos 19.*

### 2.29 Rodada 12, 2026-08-03 — o §6 fechou inteiro, as duas linhas do `GLOSSARY` entraram, e voce mesmo escreveu a D2

**Registro, nao pedido.** Nada aqui espera voce; esta secao existe para que nada disto seja reaberto.

**AS DUAS LINHAS DA §1.1, aplicadas por mim sob a sua autorizacao explicita.** Voce disse "eu autorizo voce a colar elas
no glossary", que e a **opcao 3** do antigo §6.26 e nao a 1 — a diferenca importa, porque a regra da casa e que a tabela
de notacao e sua e um agente **propoe** linhas. As duas fecham uma lacuna fail-closed medida: o $\mathbf{e}_{x_i}$
estava em uso vivo na Definicao 2.4 e o $f_{\mathrm{place}} (H_i)$ na 2.9, e nenhum dos dois estava registrado. Isso
completa a **AD-5**. **E a consequencia que a propria opcao 3 previa foi cumprida no mesmo commit:** o comentario do
`2_fundamentals.tex` afirmava, verbatim, que a linha
"is PROPOSED to the author and is not written by an agent; the notation table is his" — verdadeiro quando escrito e
**tornado falso pelo ato de escrever a linha**. Ele agora cita a frase antiga como superada, diz que voce autorizou a
excecao, e mantem a regra geral de pe.

**O §6 SAIU INTEIRO, a seu pedido.** Os vinte e seis itens foram respondidos por voce, os dez `______` que restavam eram
**residuo de formatacao** (cada um ja respondido em outra secao, conferido um por um), e a
`h3` e o cabecalho `## §6` foram removidos. **Duas coisas que eu conferi porque este arquivo manda:**

1. **Chegada antes de apagar.** Para cada bloco eu confeti que o cabecalho **e** uma linha interior do corpo estavam no
   `_archive/PENDENCIAS_RESOLVIDOS.md` antes de remover. Tres itens desta lista se perderam no passado exatamente por
   apagar antes de conferir.
2. **Ponteiros.** Remover a secao orfanou **dezenove** citacoes no fonte e **quatro probes**. As citacoes foram
   repontadas para a forma historica que o `check_tracker_refs.py` reconhece
   (`PENDENCIAS_RESOLVIDOS <n>.<m> (arquivado 2026-08-03)`), e os quatro probes (`R9-pend6`, `R9-blq4`,
   `R9-blq5`, `R12-extra`) passaram a ler o arquivo — cada string **verificada presente lá** antes do repoint, nao
   suposta. O `R9-pend6` deixou de pinar um cabecalho que voce mandou remover e passa a pinar o registro do
   encerramento, com o `R9-pend6b` guardando o cabecalho citado verbatim para quem encontrar um comentario antigo
   dizendo "§6". A numeracao 6.1 a 6.26 **nao** foi reaproveitada.

**A SUA D2, afiada em cima e nao reescrita (AD-6).** Voce substituiu a frase vaga de retencao pela sua, que nomeia o
alvo por tarefa e enuncia a posse do rotulo. Tres afiamentos, nenhum tocando o seu conteudo:

- **Referencia para frente.** A sua frase era a **primeira** ocorrencia viva de "next-category prediction" e
  "next-region prediction" no capitulo, e as duas sao definidas ~130 linhas adiante. Medido, nao suposto. Os simbolos
  estavam bem (a D1 vincula o $c_i$ e o $r_i$), entao e mais leve que um simbolo-antes-da-definicao, mas e a propriedade
  que a ordem dos passos do redesenho existia para proteger. Resolvido apontando para frente **explicitamente**, em vez
  de tirar os nomes das tarefas que voce escolheu.
- **A metade positiva.** Excluir o rotulo do $x_i$ afasta o vazamento; faltava dizer que as categorias e as regioes das
  visitas **passadas** sao entrada legitima — que e exatamente a duvida que gerou a sua edicao. Agora esta dito numa
  oracao.
- Largura de linha de volta as ~85 colunas do arquivo.

**Nada mudou na 2.5, e isso e deliberado.** A sua leitura estava certa: um "historico de regioes" e uma **projecao**
de $H_i$ e nao uma entrada diferente, porque a regiao ja esta dentro do check-in pela D1, e o que o modelo le
e $\rho (H_i)$. Se a definicao da tarefa dissesse "recebe um historico de regioes", ela passaria a descrever a escolha
de representacao do Cap. 5 e o Cap. 3 nao caberia mais nela.

**Um defeito meu, apanhado por um revisor:** eu publiquei "os oito probes novos validados por sabotagem"
quando eram **sete**. O oitavo era justamente o probe de **ausencia** — o unico cuja falha e o silencio. Corrigido,
validado nos dois ramos, e a regra que evita a repeticao esta no `_round9/34`: reconciliar os nomes dos probes validados
contra os adicionados **como conjuntos**, nao pela contagem de linhas.

## §3 · Aberto e bloqueado em terceiros

| Item                                               | Bloqueado em                     | Estado                                                                                                                                                                                                                                        |
|----------------------------------------------------|----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Dois membros da banca e a data da defesa           | Orientador / PPGCC               | Placeholders entre colchetes em `preamble.tex:217-219`. **Nao imprimem em nenhum dos tres builds** (`\folhadeaprovacao` esta comentada em `abntex2-UFV.sty:166-170`), entao nao ha nada inventado no PDF — nem os nomes reais quando chegarem |
| Folha de aprovacao assinada                        | A defesa                         | `make ppgc` gera o PDF com o placeholder; a versao assinada o substitui depois                                                                                                                                                                |
| Status do MobiWac                                  | Revisores                        | A redacao e sempre "submitted, under review", em todo o documento. **Nao mudar** ate haver decisao                                                                                                                                            |
| `\finalbuildfirstpage` conferido contra o RASCUNHO | Upload pos-defesa ao AcademicoPG | Agora **9** (`main.tex:95`), das 8 paginas pre-textuais do build de deposito; a primeira pagina de corpo do `main_academico.pdf` e a fisica 9 e imprime 9. Confira contra o RASCUNHO quando subir                                             |

---

## §4 · Pensamentos e considerações do Autor

> **AUDITADO EM 2026-08-03/04.** Os 37 pontos que voce escreveu aqui foram medidos um por um contra o
> fonte VIVO, nao contra o build que voce leu quando os escreveu. Cada um agora carrega ID estavel,
> estado da citacao, minha avaliacao, plano, sobreposicoes, disposicao e o commit da medicao.
>
> **A NUMERACAO DOS IDs NAO E A DOS ITENS, e isso foi sua decisao.** `AUT-01` ja existia
> (`CONSIDERATIONS.md`:668, a sua pergunta sobre otimalidade de Pareto, ainda aberta, defendida pelo
> probe `R9-schema`). Reusar `AUT-01` seria reciclar um ID, o que a regra deste arquivo proibe. Voce
> escolheu **continuar a serie**, entao **item N vira AUT- (N+1)**:
>
> | item | ID | item | ID | item | ID | item | ID |
> |---|---|---|---|---|---|---|---|
> | 1 | AUT-02 | 11 | AUT-12 | 21 | AUT-22 | 31 | AUT-32 |
> | 2 | AUT-03 | 12 | AUT-13 | 22 | AUT-23 | 32 | AUT-33 |
> | 3 | AUT-04 | 13 | AUT-14 | 23 | AUT-24 | 33 | AUT-34 |
> | 4 | AUT-05 | 14 | AUT-15 | 24 | AUT-25 | 34 | AUT-35 |
> | 5 | AUT-06 | 15 | AUT-16 | 25 | AUT-26 | 35 | AUT-36 |
> | 6 | AUT-07 | 16 | AUT-17 | 26 | AUT-27 | 36 | AUT-37 |
> | 7 | AUT-08 | 17 | AUT-18 | 27 | AUT-28 | 37 | AUT-38 |
> | 8 | AUT-09 | 18 | AUT-19 | 28 | AUT-29 | | |
> | 9 | AUT-10 | 19 | AUT-20 | 29 | AUT-30 | | |
> | 10 | AUT-11 | 20 | AUT-21 | 30 | AUT-31 | | |
>
> **O seu texto original nao foi apagado.** Esta byte por byte em
> [`_round13/_aut_original.md`](_round13/_aut_original.md) (sha256 `e2a44fea...`, 162 linhas, 37 itens),
> junto com o snapshot da arvore medida em `_round13/_snapshot/` com `MANIFEST.tsv` por arquivo.
>
> **Passagem de citacoes obsoletas, contada:** 29 ancoras citaveis nos 37 itens.
> **12 EXATAS, 12 ALTERADAS, 5 DESAPARECIDAS.** Por item: 22 itens tem ancora citavel
> (**10 exatas, 10 alteradas, 2 desaparecidas**), 14 itens pedem um conceito ou uma secao e nao tem
> string para localizar, e o item 37 esta vazio no fonte. Recontado contra `c13fe4d2` depois que a
> outra esteira commitou: **zero deriva** nas 29 ancoras.
>

*Forense completa: [`_round13/60_terminology_audit.md`](_round13/60_terminology_audit.md),
[`61_check2hgi_audit.md`](_round13/61_check2hgi_audit.md),
[`62_literature_audit.md`](_round13/62_literature_audit.md),
[`63_conclusion_audit.md`](_round13/63_conclusion_audit.md),
[`59_my_own_measurements.json`](_round13/59_my_own_measurements.json).*

---

> **RODADA 13, 2026-08-04 — 26 DOS 37 ITENS FORAM FECHADOS E ESTAO ARQUIVADOS.** Voce respondeu 21 em
> `§4.1` e os 15 `[YOU APPLY]` vinham de `§4.2`. Os 26 fechados sairam daqui, cada um com o commit em
> que foi aplicado, para
> [`_archive/PENDENCIAS_RESOLVIDOS.md`](_archive/PENDENCIAS_RESOLVIDOS.md), secao
> "§4 (os itens `AUT-`) — 26 DE 37 FECHADOS". Nada foi perdido: o seu §4 original continua byte por
> byte em [`_round13/_aut_original.md`](_round13/_aut_original.md).
>
> **Os 11 abaixo continuam abertos**, e nenhum deles espera uma decisao que voce ja tomou: para a
> maioria a sua palavra chegou e a edicao ainda nao foi aplicada. Tres pedem trabalho maior (AUT-14 a
> secao de Contribuicoes, AUT-37 a reordenacao de §6.2, AUT-29 a re-hierarquizacao de §2.3), um espera
> o seu orientador (AUT-26), um espera a sua leitura de uma validacao que voltou REFUTADA (AUT-35), e
> um esta vazio no fonte (AUT-38).
>
> **A numeracao dos IDs continua a mesma**, e os IDs arquivados **nunca serao reciclados**: item N do
> seu texto original e AUT-(N+1).

### AUT-02 — especificidades no Resumo e na introducao

- **§4 item:** 1
- **Source status:** [GONE] nas duas ancoras do Resumo, [CHANGED] nas outras duas. As duas frases que voce cita do
  Resumo ("cinco estados dos Estados Uni- dos... Massive-STEPS" e "vinte modelos ajustados por configuracao...") **nao
  existem mais**: o Resumo vivo (`content.tex`) ja diz "seis conjuntos de dados de diferentes contextos geograficos,
  incluindo um conjunto nao estadunidense", que e quase exatamente a redacao generica que voce propoe. A ancora "pelo
  procedimento TOST" tambem ja saiu; sobrou "margem de dois pontos de Acc@10". A lista das sete categorias continua
  viva, em 2.1.1.3 (p.19), nao em 2.1.1.2.
- **Minha leitura e avaliacao:** **Voce estava certo e metade ja foi feita por outra esteira.** O que sobra e uma
  decisao de escopo, nao de redacao: a margem de dois pontos e a convencao que liga o verbo "equipara-se" ao teste, e
  WRITING_LAW §3 exige que todo numero carregue a sua convencao. Tirar "dois pontos de Acc@10" do Resumo remove
  exatamente esse vinculo. O mesmo argumento aparece no FAB-22, que voce ja tem em aberto.
- **Plano de resolucao proposto:** Confirmar que o Resumo vivo ja satisfaz o pedido (uma linha sua fecha isso). Depois
  decidir apenas sobre a margem: (A) manter como esta, (B) tirar a margem do Resumo e deixa-la so no Cap.5/Cap.6. A
  lista das sete categorias e o AUT-18, tratada separadamente.
- **Sobreposicoes e dependencias:** AUT-18 (a mesma lista de categorias), AUT-16, FAB-14, FAB-15, FAB-22 (todos sobre
  detalhe de dados e de resultado no texto de moldura).
- **Disposicao alvo:** **[I DECIDE]** — a parte generica ja esta feita; a margem colide com uma regra de honestidade.
- **Onde renderiza:** Resumo/Abstract p.6-7; §2.1.1.3 p.19
- **Build medido contra:** `c13fe4d2` (build/main.pdf, 105 pp, mtime 2026-08-03 21:21:31)

### AUT-08 — por que o par de tarefas mudou: o argumento da literatura, e o vazamento de categoria

- **§4 item:** 7
- **Source status:** [EXACT] — a frase esta viva em `1_introduction.tex`. Sao **duas** perguntas suas num item, e elas
  se separam.
- **Minha leitura e avaliacao:** **Primeira metade (o argumento da literatura): REFUTADA.** Voce quer dizer que na
  literatura a proxima categoria e a proxima regiao "possuem mais forcas" que a classificacao de POI. Nenhuma fonte
  aberta sustenta a forma comparativa, e a contagem no OpenAlex aponta para o outro lado. Isto **fecha em negativo** a
  bandeira `[VERIFY]` que o `NORTH_STAR` §6 Cap.1 beat 4 (b) tinha deixado aberta exatamente para esta frase, e a
  propria fallback sancionada la esta disponivel. O eixo que **e** defensavel, e que a literatura sustenta caso por
  caso, e outro: nesses trabalhos a categoria costuma ser tarefa **auxiliar** de um objetivo de proximo lugar, nao alvo
  final. **Segunda metade (o vazamento): CONFIRMADA, e documentada no codigo.** O vetor de check-in codifica sim a
  categoria da visita **atual**, por construcao: a categoria entra como feature do no de check-in, e ha um termo de
  perda de reconstrucao de categoria com peso 0.3 no objetivo do Check2HGI. Isso e exatamente por que a tarefa
  **estatica** deixa de ser um par limpo sob esse regime, que e a sua intuicao original.
- **Plano de resolucao proposto:** Duas edicoes independentes. (1) Trocar a perna comparativa pela redacao fallback do
  `NORTH_STAR`, ou pelo eixo auxiliar-versus-alvo-final se voce quiser o enquadramento de literatura (uma a tres frases
  em §1.2). (2) Decidir se o argumento do vazamento entra no texto. Ele e **novo** (C2) e hoje nao esta em nenhum
  `.tex`; entra como afirmacao de **projeto**, nunca como resultado, porque nada no repositorio isola o efeito. Se
  entrar, `apx_b_static_scope.tex`:83-85 precisa de reescrita coordenada, porque hoje aponta na direcao contraria.
- **Sobreposicoes e dependencias:** **AUT-20 e AUT-08 compartilham a base factual** (categoria atual versus futura) e
  vao em direcoes opostas: um quer dizer que nao ha rotulo, o outro que ha. Resolver um sem o outro cria contradicao.
  Tambem: AUT-35 (c) (o confound do par de tarefas, mesmo raciocinio de vazamento), AUT-14, FAB-28.
- **Disposicao alvo:** **[I DECIDE]** — uma perna refutada com fallback pronta, e uma afirmacao nova que precisa do seu
  aval.
- **Onde renderiza:** §1.2 p.13-14; Apendice B (escopo estatico) p.81
- **Build medido contra:** `c13fe4d2` (build/main.pdf, 105 pp, mtime 2026-08-03 21:21:31)

### AUT-09 — a frase do "correction trail" ficou pior que a anterior

- **§4 item:** 8
- **Source status:** [EXACT] na versao atual; [GONE] na versao anterior que voce prefere (esperado: ela foi
  substituida).
- **Minha leitura e avaliacao:** Concordo com a sua leitura, e a comparacao e justa. A versao viva ("It presents a
  sequence in which a result valid for one configuration leads to a diagnosis and then to a different, explicitly
  bounded solution") e abstrata: "a sequence in which a result leads to a diagnosis" faz o resultado agir, o que e
  exatamente a forma que WRITING_LAW §1 proibe (substantivo abstrato como agente). A anterior nomeia o sujeito (cada
  estudo revisou o anterior) e diz o que os capitulos fazem. **Mas** a anterior contem "correction trail", e o
  `NORTH_STAR` §6 Cap.1 beat 4 (d) tem uma guarda F4 explicita contra fazer o resultado nulo parecer ato um de um
  roteiro escrito de antemao.
- **Plano de resolucao proposto:** Reescrever uma a duas frases combinando as duas: o sujeito nomeado e os capitulos
  como agentes (da anterior), sem a metafora de trilha e mantendo o enquadramento time-indexed (da guarda F4). Nao e
  restaurar a antiga literalmente.
- **Sobreposicoes e dependencias:** AUT-37 (a conclusao pede o mesmo arco no Cap.6, com o mesmo cuidado F4).
- **Disposicao alvo:** **[I DECIDE]** — a frase e uma afirmacao de arco (C2) e a redacao anterior colide com a guarda
  F4.
- **Onde renderiza:** §1.2 p.14
- **Build medido contra:** `c13fe4d2` (build/main.pdf, 105 pp, mtime 2026-08-03 21:21:31)

### AUT-14 — a secao de Contribuicoes: os seus quatro candidatos, e o que falta

- **§4 item:** 13
- **Source status:** [N/A] como citacao (pedido de avaliacao). A secao viva e `1_introduction.tex`:294-331.
- **Minha leitura e avaliacao:** Auditei os seus quatro candidatos separadamente, porque tem riscos muito diferentes.
  **(a) Check2HGI como avanco reutilizavel: SUSTENTAVEL, e hoje sub-declarado.** Pode entrar.
- **(b) O modelo conjunto como modular/extensivel: PARCIALMENTE SUSTENTAVEL, o mais fraco dos quatro.** Modularidade e
  uma propriedade de projeto que o documento pode afirmar; "pode ser expandido para outras tarefas" e uma previsao que
  nenhum experimento sustenta. Se entrar, entra como projeto.
- **(c) "as tarefas parecem nao ser conflitantes": o seu proprio aviso esta correto.** A evidencia e uma media de
  cosseno de +0.001, quatro sementes, numa preparacao de dados anterior, mais o Apendice F em sete conjuntos. Uma media
  nao distingue "consistentemente ortogonal" de "forte conflito nos dois sentidos que se cancela". E o mesmo ponto do
  **GER-11**, que o Germano levantou.
- **(d) "nossos artigos sao pioneiros":
  REFUTAVEL COMO ESCRITO, e foi testado para ser refutado.** Oito sondagens booleanas no OpenAlex mais cinco buscas por
  palavra-chave, com o instrumento validado antes. MTL com alvo de categoria e uma literatura **populada** (HAMTL,
  DRRGNN, Hgarn, HMT-GRN, MCMG, CSLSL, MCARNN), e proxima categoria como alvo isolado **antecede** este trabalho. O que
  nenhuma fonte encontrada faz e tratar proxima categoria e proxima regiao como alvos finais **co-iguais** de um modelo
  conjunto sem alvo de proximo lugar. **O texto vivo ja e mais estreito que a sua frase** e ja esta protegido por probe
  (`R10-novelty`).
- **O maior buraco nao esta nos seus quatro:** o **Capitulo 4 nao aparece nenhuma vez** na secao (zero referencias a
  `ch:courb`, contra duas ao Cap.3 e duas ao Cap.5), embora o diagnostico de que a representacao e o gargalo seja a
  dobradica do arco e o bullet Teorico afirme exatamente esse achado. O bullet Software lista MTLnet e Check2HGI e omite
  ST-MTLNet, que e artefato registrado no GLOSSARY §2.
- **Plano de resolucao proposto:** Fazer em duas ondas. **Mecanico primeiro** (e a parte que eu aplicaria): acrescentar
  o Cap.4 e o ST-MTLNet aos bullets Software e Teorico. **Depois as decisoes:** (a) reforcar, (b) so como projeto, (c)
  redigir com o escopo completo viajando junto ou rebaixar, (d) usar a forma estreita que o texto ja tem, com "to our
  knowledge", e **nunca** "pioneering"/"the first" no Cap.1 — a forma mais forte viva esta no Cap.5, que esta em
  revisao, e a moldura nao deve exceder o capitulo. Alem disso, quatro contribuicoes ja conquistadas e nao declaradas: a
  triagem de dezenove balanceadores (com escopo anexado), a medicao de ortogonalidade do Apendice F, as adaptacoes dos
  baselines para regiao, e as margens sobre os metodos externos.
- **Sobreposicoes e dependencias:** **AUT-28 (Pareto) e AUT-14 (c) e (d) se cruzam:** os tres sao afirmacoes sobre o que
  os resultados autorizam. **GER-11** e o mesmo ponto que (c). **FAB-28** e o mesmo ponto que (d) e foi DESBLOQUEADO por
  esta rodada (ver §4.1). AUT-08 (a mesma questao de literatura), AUT-11.
- **Disposicao alvo:** **[I DECIDE]** — quatro decisoes de risco diferente, duas delas (c, d) afirmacoes sob C2. A parte
  mecanica (Cap.4 + ST-MTLNet) pode ser destacada como [YOU APPLY] se voce quiser.
- **Onde renderiza:** §1.6 p.16
- **Build medido contra:** `c13fe4d2` (build/main.pdf, 105 pp, mtime 2026-08-03 21:21:31)

### AUT-26 — renomear o modelo conjunto para MTLChkNet, e se ja e tarde

- **§4 item:** 25
- **Source status:** [N/A] como citacao (pergunta). `MTLChkNet` aparece **0** vezes na arvore.
- **Minha leitura e avaliacao:** Medi o raio de impacto antes de opinar, porque a resposta depende dele.
  `the joint model` tem **49** ocorrencias em prosa viva, e elas nao sao equivalentes entre si: **15 estao no Cap.5**,
  que e o manuscrito **submetido**; **5 estao dentro de tabelas de errata**, onde a string citada **e** a evidencia de
  uma correcao (mudar ali altera a citacao); **22 estao em prosa de moldura e no `content.tex`**, que sao livres; 2 na
  tabela de linhagem; 2 no Cap.3 publicado. Ou seja: **nao e um search-and-replace**, sao tres regimes diferentes. Alem
  disso `the joint model` esta **registrado** no GLOSSARY §2 como o nome canonico, com a nota de que o id de repositorio
  `mtlnet_crossattn_dualtower` nunca aparece no texto. Uma opiniao que eu devo dar mesmo sem voce pedir: um nome proprio
  novo compra pouco aqui, porque "the joint model" ja e contrastivo com "dedicated single-task model", que e o par que o
  documento usa em todo comparativo.
- **Plano de resolucao proposto:** Se voce quiser o nome: (A) so na moldura (Caps.1, 2, 6 + `content.tex`), deixando o
  Cap.5 como esta — barato, mas cria duas nomenclaturas no mesmo documento, o que e pior que nenhuma; (B) moldura +
  Cap.5, **sem** tocar as tabelas de errata, com nova linha no GLOSSARY §2 e paragrafo no Apendice B; (C) tambem no
  manuscrito do MobiWac, que e **edicao cross-boundary num artigo em revisao** e precisa da sua autorizacao explicita (o
  precedente e a correcao B.1 do CBIC, autorizada por voce e registrada na ERRATA do MobiWac); (D) nao renomear.
  **Recomendo (D)**, e se for renomear, (B) ou (C), nunca (A).
- **Sobreposicoes e dependencias:** AUT-03 (a mesma familia de arquivos de nome). O GLOSSARY §2 precisa de linha nova em
  qualquer opcao menos (D), e so voce pode adicionar.
- **Disposicao alvo:** **[I DECIDE]** — escolha de nome, com custo em manuscrito submetido e em registro.
- **Onde renderiza:** moldura p.13-16, p.83-87; Cap.5 p.64-80; tabelas de errata p.48
- **Build medido contra:** `c13fe4d2` (build/main.pdf, 105 pp, mtime 2026-08-03 21:21:31)

### AUT-29 — §2.3.2 e §2.3.3 estao mal organizados e repetitivos

- **§4 item:** 28
- **Source status:** [N/A] como citacao (pedido estrutural).
- **Minha leitura e avaliacao:** Concordo com o sintoma e tenho uma ressalva estrutural sobre a solucao. **O sintoma:**
  medido, §2.3 tem §2.3.1 (242 palavras), §2.3.2 (129), §2.3.2.1 (159), §2.3.2.2 (225), §2.3.3 (267), §2.3.4 (233); a
  ordem atual apresenta o formalismo antes do problema que ele formaliza, que e a inversao que faz voce voltar e reler.
  **A ressalva:** a sua ordem proposta move o §2.3.2.2 para **antes do seu proprio pai** §2.3.2 e o divide em duas (
  'part B'), o que nao e uma reordenacao e sim uma **re-hierarquizacao** — as duas subsubsecoes deixariam de ser filhas
  de "Multi-objective optimization". Isso e legitimo, mas e uma mudanca de arvore de secoes, com renumeracao de todos os
  `\ref` internos, e nao a troca de ordem que o item descreve.
- **Plano de resolucao proposto:** Se voce quiser a ordem "problema -> formalismo -> literatura -> garantias": promover
  "Gradient conflict" a subsecao irma de "Multi-objective optimization" e coloca-la antes, em vez de manter a
  aninhamento atual. Isso preserva a hierarquia coerente e da o fluxo que voce quer. Custo: renumeracao de §2.3.x,
  revisao dos `\ref` que apontam para `def:fund:conflict` e para as subsubsecoes, e conferencia dos probes que citam
  strings dessa regiao (`R9-conflict`, `R10-cosine`, `R12-dwa*`, `R11-aligned*`). Nenhum deles casa em numero de secao,
  entao a reordenacao **nao** os quebra; e preciso confirmar depois de mover.
- **Sobreposicoes e dependencias:** **GER-09 e GER-10 pedem exatamente a reestruturacao de §2.3 e estao em aberto com
  voce** — este item deve ser decidido junto com eles, ou a secao sera reorganizada duas vezes. AUT-27 (a glosa
  do $\\mathcal{L}_k$ vive nessa regiao).
- **Disposicao alvo:** **[I DECIDE]** — mudanca estrutural, e a forma que voce propos re-hierarquiza em vez de
  reordenar.
- **Onde renderiza:** §2.3.2 p.26, §2.3.2.1 p.26, §2.3.2.2 p.27, §2.3.3 p.27
- **Build medido contra:** `c13fe4d2` (build/main.pdf, 105 pp, mtime 2026-08-03 21:21:31)

### AUT-32 — a abertura do Cap.6 esquece a classificacao de POI

- **§4 item:** 31
- **Source status:** [CHANGED] — a sua citacao nao e o texto vivo. Hoje (`6_conclusion.tex`:15-17): *This dissertation
  asked whether multitask learning helps point-of-interest prediction for the next category and the next region of a
  visit, and which design choices determine the answer.*
- **Minha leitura e avaliacao:** A omissao e real, e ela e **deliberada e coerente** com tres lugares aprovados. A
  pergunta de pesquisa do documento, no `NORTH_STAR` §1 e no §1.2, e sobre proxima categoria e proxima regiao; a
  classificacao estatica e uma tarefa **dos estudos 1 e 2**, nao da pergunta. Acrescenta-la a abertura do Cap.6 faria o
  capitulo declarar uma pergunta de pesquisa que os outros dois lugares nao declaram, o que propaga alem de uma correcao
  de redacao. **Porem** ha um fato a seu favor: a estrutura de tres tarefas **e** dita no Cap.6, duas vezes, mais
  adiante — entao o leitor nao fica sem ela, so a encontra depois.
- **Plano de resolucao proposto:** Duas saidas: (A) manter a abertura como esta (ela espelha a pergunta de pesquisa) —
  custo zero; (B) uma oracao no Cap.6, **sem** mexer na pergunta de pesquisa, dizendo que os dois primeiros estudos
  incluiam tambem uma tarefa estatica de classificacao e que o par mudou no estudo final. Recomendo (B): custa uma
  oracao, e satisfaz o seu desconforto sem alargar a pergunta em tres lugares.
- **Sobreposicoes e dependencias:** **AUT-35 (c) e AUT-36 giram na mesma mudanca de par de tarefas.** Uma resolucao
  consistente trata a tarefa estatica como **historica**: presente nos recaps dos Caps.3/4 e na limitacao 6, ausente da
  pergunta de pesquisa e do trabalho futuro. AUT-16, AUT-07.
- **Disposicao alvo:** **[I DECIDE]** — (A) ou (B); alargar a pergunta de pesquisa e mudanca no `NORTH_STAR` §1 e em
  dois pontos do Cap.1, e so voce autoriza.
- **Onde renderiza:** §6 abertura p.83
- **Build medido contra:** `c13fe4d2` (build/main.pdf, 105 pp, mtime 2026-08-03 21:21:31)

### AUT-35 — as tres limitacoes de §6.3: vintage, transdutividade, e o confound do par de tarefas

- **§4 item:** 34
- **Source status:** [EXACT] nas duas ancoras que voce cita (`Transductive representation`, `The task-pair confound.`) —
  as duas estao vivas em `6_conclusion.tex`.
- **Minha leitura e avaliacao:** Tres sub-pontos, tres vereditos diferentes.
- **(a) vintage: REFUTADO, e a sua frase cai numa armadilha.** A limitacao viva ja e **escopada ao Gowalla** ("The five
  state datasets come from Gowalla... January 2009 to August 2011") e nao diz nada sobre Istambul, entao ela nao afirma
  o que voce esta rebatendo. E o Massive-STEPS **nao e dado de 2025**: 2025 e o ano de **publicacao** (ainda preprint,
  arXiv:2505.11239), e os check-ins de Istambul sao de **2017-2018** conforme o proprio resumo do benchmark. Publicacao
  e vintage sao coisas diferentes. **(b) transdutividade: PARCIALMENTE CONFIRMADO.** A limitacao viva esta correta e
  estreita. O seu acrescimo ("isso afeta varias abordagens da literatura") e uma **afirmacao sobre o trabalho de
  outros** e por isso precisa de citacao localizada, nao pode ser generalidade nua. O polo indutivo ja tem citacao
  verificada (`hamilton2017graphsage`); o polo transdutivo precisaria de pelo menos `huang2023hgi` e, idealmente, uma
  frase de survey lida em primeira mao. **(c) o confound do par de tarefas: o seu raciocinio de vazamento SE SUSTENTA, e
  ele NAO apaga a limitacao — ele a reforca.**
  Voce esta certo que a ablacao que resolveria o confound (classificacao estatica sob o Check2HGI) vazaria, porque o
  vetor de check-in codifica a categoria da visita atual e ha um termo de reconstrucao de categoria no objetivo. Mas dai
  segue que o confound **nao e resolvivel de forma limpa**, o que e uma limitacao **mais** forte, nao menos. O que a sua
  objecao invalida e o **item de trabalho futuro** amarrado a ela, que hoje propoe rodar essa ablacao. E ha um cuidado
  de registro: esta limitacao tem **aval registrado** (`NORTH_STAR` §6, adicao assinada em 2026-07-22), entao remove-la
  precisa de novo aval. Uma ressalva de honestidade sobre a minha propria conclusao: o argumento do vazamento e
  **analitico**, derivado do spec, nao medido — nenhum experimento demonstrou o vazamento.
- **Plano de resolucao proposto:** (a) Se voce quiser Istambul nomeado para a limitacao ler como escopada: manter a
  janela do Gowalla **literalmente** (o probe `R8-vintage` exige a string "August 2011") e acrescentar **uma** frase com
  a janela 2017-2018, que e **numero novo** e precisa de linha de ledger e de comentario de claim no `.bib`. (b) Fazer a
  citacao **antes** da frase. (c) Quatro saidas: manter como esta; manter e acrescentar por que a ablacao nao e limpa
  (marcado como inferencia, nao medicao); enfraquecer; remover. **Recomendo a segunda**, e nesse caso o item de trabalho
  futuro correspondente muda de "rodar a ablacao" para "a ablacao nao e executavel de forma limpa sob esta
  representacao".
- **Sobreposicoes e dependencias:** **AUT-36 esta amarrado 1:1 a (c)**: mexer na limitacao sem mexer no trabalho futuro
  deixa um orfao. AUT-32 (a tarefa estatica como historica). AUT-08 e AUT-20 (a mesma base factual do vazamento).
  **Gate:** `R8-vintage` em (a).
- **Disposicao alvo:** **[I DECIDE]** — (a) premissa errada com edicao que introduz numero e toca um gate; (b) afirmacao
  nova sobre terceiros; (c) limitacao com aval registrado, acoplada ao trabalho futuro.
- **Onde renderiza:** §6.3 p.86
- **Build medido contra:** `c13fe4d2` (build/main.pdf, 105 pp, mtime 2026-08-03 21:21:31)

### AUT-36 — os sete itens de trabalho futuro

- **§4 item:** 35
- **Source status:** [N/A] como citacao (lista de propostas). O §6.4 vivo esta em `6_conclusion.tex` (p.86).
- **Minha leitura e avaliacao:** Conferi os sete contra o §6.4 vivo, um por um; o detalhe esta em
  `_round13/63_conclusion_audit.md`. Parte deles ja esta la em alguma forma, parte esta ausente, e dois tem restricao
  estrutural. **As duas restricoes que valem para qualquer acrescimo:** (i) o `NORTH_STAR` §6 exige que cada item de
  trabalho futuro esteja amarrado **1:1** a uma limitacao de §6.3, entao acrescentar itens significa acrescentar ou
  re-amarrar limitacoes, e isso muda a frase que conta as limitacoes e a numeracao delas; (ii) o seu item mais
  promissor, a cabeca de **proximo lugar** acoplada ao modelo conjunto, colide com o escopo declarado: proximo lugar e a
  tarefa que o documento diz explicitamente que **nao** preve, e isso e registrado formalmente no GLOSSARY §1.1 e
  defendido por dois probes (`R12-fplace`, `R12-fplace2`). Propor como trabalho futuro e **legitimo** e ja aparece na
  lista do `NORTH_STAR`; o cuidado e de redacao, para nao soar como algo que a dissertacao fez.
- **Plano de resolucao proposto:** Ate quatro frases novas em §6.4 (integracao do Check2HGI, soft-sharing moderno,
  hypergraphs, e a metade de tuning do cascade), cada uma com a sua limitacao de ancoragem, mais uma oracao para o
  mecanismo do item de proximo lugar. Se limitacoes novas forem criadas, a frase de contagem e a numeracao mudam e os
  gates precisam ser reconferidos.
- **Sobreposicoes e dependencias:** **AUT-35 (c) esta amarrado 1:1 a este item.** AUT-21 (o acoplamento do POI2Vec e a
  base do seu item 1). AUT-32. **Gates:** `R12-fplace`, `R12-fplace2`.
- **Disposicao alvo:** **[I DECIDE]** — acrescentar trabalho futuro mexe na estrutura 1:1 e na contagem de limitacoes.
- **Onde renderiza:** §6.4 p.86; §6.3 p.86
- **Build medido contra:** `c13fe4d2` (build/main.pdf, 105 pp, mtime 2026-08-03 21:21:31)

### AUT-37 — avaliacao critica da conclusao, e o fluxo que voce quer para §6.2

- **§4 item:** 36
- **Source status:** [N/A] como citacao (pedido de avaliacao).
- **Minha leitura e avaliacao:** Os exemplares foram medidos (extensao, seccionamento, densidade de numeros, ordem de
  movimentos) e a comparacao esta em `_round13/63_conclusion_audit.md` §36. O seu diagnostico sobre §6.2 se sustenta: a
  secao tem 53 numerais e restabelece a manchete mais de uma vez, o que a torna um segundo capitulo de resultados em vez
  do fechamento do arco. O seu fluxo alvo (pergunta e tese -> cadeia de causa e efeito -> o que erramos na tese
  inicial -> a licao pela lente das descobertas) e implementavel como uma **reordenacao de quatro movimentos**, nao uma
  reescrita. O movimento 3 (o que erramos) e o que a secao hoje nao tem, e o material dele **existe**: e exatamente o
  que os controles descartaram (o AUT-33).
- **Plano de resolucao proposto:** Reordenar §6.2 em quatro movimentos, promovendo a cadeia causal a um paragrafo
  narrado e acrescentando o movimento do "o que erramos". O movimento 3 e uma **afirmacao nova de moldura** mesmo com
  todos os componentes ja sourceados, e o tamanho e a posicao do paragrafo do baseline de capacidade sao uma decisao sua
  explicitamente reservada no cabecalho do proprio arquivo.
- **Sobreposicoes e dependencias:** **AUT-33 fornece o material do movimento 3.** **AUT-34 pede o mesmo reequilibrio de
  numeros** e a mesma edicao serve aos dois.
- **Disposicao alvo:** **[I DECIDE]** — reordenacao estrutural com uma afirmacao nova de moldura.
- **Onde renderiza:** §6.2 p.84-86
- **Build medido contra:** `c13fe4d2` (build/main.pdf, 105 pp, mtime 2026-08-03 21:21:31)

### AUT-38 — item vazio no fonte

- **§4 item:** 37
- **Source status:** [N/A] — o item 37 existe como "37." seguido de duas linhas em branco. Nao ha texto.
- **Minha leitura e avaliacao:** Nao ha nada para auditar. Reservei o ID para que a numeracao nao seja reciclada se voce
  escrever aqui depois, que e a regra deste arquivo.
- **Plano de resolucao proposto:** Escrever o pensamento, ou apagar o marcador. Se escrever, ele entra na proxima
  passada com este ID.
- **Sobreposicoes e dependencias:** Nenhuma.
- **Disposicao alvo:** **[BLOCKED]** — vazio no fonte; aguarda o seu texto.
- **Onde renderiza:** n/a
- **Build medido contra:** `c13fe4d2` (build/main.pdf, 105 pp, mtime 2026-08-03 21:21:31)

---

## §4.1 · Decisões Pendentes do Autor

> **So os itens `[I DECIDE]` do §4 moram aqui: 21 dos 37.** Cada bloco acima carrega o diagnostico
> completo; este e o formulario de decisao, com opcoes, troca e esforco. Os 15 `[YOU APPLY]` nao
> estao aqui de proposito: eles nao esperam nada de voce. O `AUT-38` esta vazio no fonte.
>
> **Para responder:** escreva `> DECISSAO:` (ou `> DECISAO:`) embaixo do item. Eu leio como palavra
> final e nao reinterpreto.
>
> **Custo em horas e o meu, nao o seu.** "P" = menos de 30 min, "M" = 30 min a 2 h, "G" = mais de 2 h
> ou toca gate/manuscrito submetido.

### As tres que eu levaria primeiro

**1. AUT-03 · tres citacoes textuais estao danificadas, e uma e o titulo do seu artigo. [P]**
Uma varredura anterior tirou o hifen de `Multi-Task` **dentro** de citacoes textuais, em tres lugares:

| onde                                                                   | o documento imprime                                     | a fonte de registro diz                                                                 |
|------------------------------------------------------------------------|---------------------------------------------------------|-----------------------------------------------------------------------------------------|
| `3_cbic.tex`:17, preambulo do Cap.3 (p.33)                             | *An Investigation into **Multitask** Learning for...*   | **Multi-Task** — Crossref, DOI `10.21528/CBIC2025-1191324`, e `CBIC___MTL/main.tex`:31  |
| `5_mobiwac.tex`:29, preambulo do Cap.5 (p.64)                          | *A Check-in-Level **Multitask** Study on Mobility Data* | **Multi-Task** — `[mobiwac]/src/main.tex`:52, o manuscrito submetido (EDAS #1571313639) |
| `tables/cbic/errata_wording.tex`:24, coluna "Published wording" (p.48) | *investigating advanced **multitask** optimizers...*    | **multi-task** — `CBIC___MTL/sections/conclusion.tex`:17                                |

O primeiro e o urgente: **o mesmo PDF imprime dois titulos diferentes para um artigo so.** A p.33 diz
`Multitask` e a bibliografia, na p.88 referencia [1], diz `multi-task`, que e a forma correta. O terceiro e
auto-anulante: a coluna cuja funcao e mostrar o que o artigo publicou deixou de mostrar o que o artigo publicou. Isto
viola R2 (atributos copiados da fonte) e a ressalva do WRITING_LAW §1 sobre citacao textual.

- **(A) Restaurar o hifen nas tres.** Recomendada. 3 strings, 3 arquivos.
- **(B) Restaurar so no titulo do CBIC** (o que sai em dois lugares do mesmo PDF).
- **(C) Nao mexer.** Nao recomendo: sao atributos de registro, nao prosa da casa.
- E, independente disso: registrar a normalizacao da **prosa reproduzida** no Apendice B, seguindo o paragrafo
  `MTLNet`->`MTLnet` que ja esta la como precedente. [P]

> DECISSAO: A.

**2. AUT-20 · "trained without category or region labels" nao e verdade como esta escrito. [M]**
Verificado no spec e no codigo: ha um termo de **reconstrucao de categoria de POI mascarado** com peso **0.3** de cinco
termos no objetivo do Check2HGI, e um termo L2 de categoria no POI2Vec a montante (peso `1e-8`, que o proprio comentario
do codigo chama de *"the only explicit category-label path"*). O que **e** verdade e mais estreito e ainda suficiente:
nenhum objetivo de representacao le a categoria ou a regiao de uma visita **futura**.

- **(A) Trocar a afirmacao de ausencia por uma de escopo.** Recomendada: diz o que protege (o alvo futuro) e admite o
  que usa (a categoria observada da visita atual).
- **(B) Manter e acrescentar uma nota de rodape** com os dois termos. Mais barato, menos honesto: a frase principal
  continua prometendo mais do que entrega.
- **(C) Manter.** Nao recomendo. E a classe de defeito que o WRITING_LAW §3 chama de bug, nao de estilo.
- **Atencao:** resolva junto com o **AUT-08**, que quer afirmar o **oposto** (que o vazamento da categoria atual e
  desejavel). Os dois no mesmo capitulo, decididos em separado, se contradizem.

> DECISSAO: A. Read the: articles/dissertacao/science/check2hgi_v17_complete_picture.md if you need more context, but we
> use the category and the region (lat and long) on the graph node.

**3. AUT-14 · o Capitulo 4 nao aparece nenhuma vez nas Contribuicoes. [M, com uma parte P]**
Medido: zero referencias a `ch:courb` na secao, contra duas ao Cap.3 e duas ao Cap.5 — embora o diagnostico de que a
representacao e o gargalo seja a dobradica do arco e o bullet Teorico afirme exatamente esse achado. O `ST-MTLNet`,
artefato registrado, esta fora do bullet Software.

- **(A) Destacar a parte mecanica agora** (acrescentar Cap.4 e ST-MTLNet) e decidir o resto depois. Recomendada: e a
  unica parte sem juizo de conteudo. [P]
- **(B) Reescrever a secao inteira de uma vez**, com os seus quatro candidatos resolvidos. [M/G]
- Sobre os seus quatro candidatos: (a) sustentavel e hoje sub-declarado; (b) so como afirmacao de projeto, nunca como
  previsao; (c) o seu proprio aviso esta certo, veja o **AUT-28** e o **GER-11**; (d) **"pioneiros" e refutavel como
  escrito** — a forma estreita que o texto ja tem ("Among the works reviewed here, none treats next category and next
  region as co-equal end targets of one joint model")
  e o teto, e **nunca** "pioneering"/"the first" no Cap.1. Uma varredura de refutacao com treze buscas encontrou dez
  trabalhos vizinhos e nenhum que refute a forma estreita.

> DECISSAO: Vamos seguir com o B. Nessa sessão de contribuições temos que colocar o que é contribuição real do ponto de
> vista de todo o trablaho, e o Cap. 4,
> apesar de ter seido uma ótima base téorica, como citada na introdução, ele não posui contribuições praticas para
> literatura, a não ser os aprendizados que já foram consumidos nos artigos em sequencias.

### As decisoes de afirmacao cientifica

**4. AUT-28 · afirmar a propriedade de Pareto do Cap.5. [P na prosa, mas toca 4 probes]**
Aplicando a definicao que o proprio documento registra, com os numeros citados da fonte de registro:
dominancia vale em **Istambul, FL, TX, CA** e esta **bloqueada em AL e AZ**, onde regiao e apenas nao-inferior por TOST.
O whitelist do MobiWac proibe "Pareto-dominates everywhere" nominalmente.

- **(A) Duas frases no Cap.6, sem a palavra "Pareto".** Recomendada.
- **(B) Nomear o conceito explicitamente.** Admissivel, mas o Apendice F e o lugar **errado** (ele mede cosseno de
  gradiente, nao desfecho de tarefa); o lugar e o Cap.6.
- **(C) Nao afirmar nada.** Custo zero, e o documento continua coerente.
- **Se (A) ou (B):** a frase "claims no Pareto property" do §2.3 fica falsa como escrita e precisa ser estreitada para
  **otimalidade**. Quatro probes a guardam (`R9-pareto`, `R9-pareto2`, `R9-pareto3`,
  `R9-conflict`), e o probe tem de ser reapontado no **mesmo commit** da edicao.

> DECISSAO: Opção C.

**5. AUT-08 · por que o par de tarefas mudou. [P na perna 1; M na perna 2]**
A perna comparativa ("na literatura essas duas tarefas tem mais forca") **nao tem ancora aberta que a sustente**, e isso
fecha em negativo a bandeira `[VERIFY]` que o `NORTH_STAR` §6 Cap.1 beat 4 (b) deixou exatamente para esta frase.

- **(A) Usar a fallback ja sancionada** no proprio `NORTH_STAR`: *"both are established end targets, and next region
  feeds a broader family of downstream problems"*. Recomendada, custo minimo.
- **(B) Usar o eixo que a literatura sustenta caso por caso:** nesses trabalhos a categoria costuma ser tarefa
  **auxiliar** de um objetivo de proximo lugar, nao alvo final.
- **(C) Acrescentar tambem o argumento do vazamento** (a sua segunda metade). Afirmacao nova (C2), entra como projeto e
  nunca como resultado, e obriga a reescrever `apx_b_static_scope.tex`:83-85, que hoje aponta na direcao contraria.
  Decida junto com o **AUT-20**.

> DECISSAO: Opção A.

**6. AUT-35 · as tres limitacoes. [(a) M e toca gate; (b) M; (c) G]**

- **(a) vintage: a sua premissa esta errada.** A limitacao ja e escopada ao Gowalla e nao fala de Istambul; e o
  Massive-STEPS **nao e dado de 2025** (2025 e publicacao, ainda preprint; os check-ins de Istambul sao de
  **2017-2018**). Opcoes: manter; ou acrescentar uma frase com a janela de Istambul, o que introduz **numero novo**
  (linha de ledger + comentario de claim) e exige manter a string
  "August 2011" literal, porque o probe `R8-vintage` a exige.
- **(b) transdutividade:** o acrescimo e afirmacao sobre trabalho de terceiros e precisa de citacao localizada **antes**
  da frase. Nao pode ser generalidade nua.
- **(c) o confound do par de tarefas: o seu raciocinio se sustenta e NAO apaga a limitacao.** Se a ablacao que
  resolveria o confound vazaria, entao o confound nao e resolvivel de forma limpa, o que e uma limitacao **mais** forte.
  O que a sua objecao invalida e o **item de trabalho futuro** amarrado a ela. Quatro saidas: manter; **manter e
  acrescentar por que a ablacao nao e limpa** (recomendada, marcada como inferencia e nao medicao); enfraquecer;
  remover — e remover precisa de **novo aval**, porque a limitacao tem aval registrado de 2026-07-22.

> DECISSAO:Vamos aplica o A,B e commentar a limitação 6, com as alterações do C. Ainda sobre o ponto A, tem um detalhe
> que voce não pegou e eu tmb não citei, mas o masive step é o conjunto mais moderno que temos na literatura publica,
> pode
> validar.

### As decisoes estruturais

**7. AUT-29 · reordenar §2.3.2 e §2.3.3. [M/G]** Concordo com o sintoma. Mas a ordem que voce propoe move o §2.3.2.2
para **antes do seu proprio pai** e o divide em dois, o que e **re-hierarquizacao**, nao reordenacao. Saida coerente:
promover "Gradient conflict" a subsecao irma e coloca-la antes. **Decida junto com GER-09 e GER-10**, que pedem a mesma
reestruturacao e estao abertos.

> DECISSAO: Eu concordo com essa sugestão, mas ainda assim teremos que mudar o inicio do 2.3.2, porque hoje o texto
> atual, já começa direto na formalização antes da problematica.

**8. AUT-31 · reorganizar §2.4 com "Preparation and data split". [M]** Concordo; a estrutura proposta e mais coerente
que a atual, e o material a promover ja esta dentro do §2.4.3. Cuidado: o
`\label{sec:fund:eval}` e citado de fora do capitulo e deve continuar apontando para onde as outras referencias esperam.

> DECISSAO: OK

**9. AUT-18 · as sete categorias numa definicao tecnica. [P]** Concordo com a primeira metade (a cardinalidade e
propriedade do dado) e discordo da segunda (setor censitario e mahalle sao o que torna
"regiao" concreto). **Decida numa unica passada com FAB-14 e FAB-15**, que sao a mesma decisao no Cap.1.

> DECISSAO: Sobre o ponto que voce discorda, a questão e que esses não devem ser citados em todo momento do texto, eles
> devem ser citados com setories adimistrativos ou um nome generico que caiba para ambos e que caiba para qualquer outro
> datasetr no futuro, o ponto da minha critica é a especificidade, não precismoas dizer que é o seto censitario dos USA
> ou e mahalle que é o orgão de istambul, conseguiimos se refeir a eles de forma generica. E ser especifico na descrição
> do dataset. Outra questão e que sim os artigos não são generalistas é isso e um problema, não devemos tratar e nem
> apontar.

**10. AUT-37 · a conclusao e o fluxo de §6.2. [M]** O seu diagnostico se sustenta: 53 numerais em §6.2 e a manchete
restabelecida mais de uma vez. O fluxo que voce quer e uma **reordenacao de quatro movimentos**, e o movimento que falta
("o que erramos na tese inicial") tem material pronto: e o que os controles descartaram (AUT-33). **A mesma edicao serve
ao AUT-34.**

> DECISSAO: OK, tome cuidado na re-escrita, para usar agents que escrevam de forma natural, se preciso for use o codex
> com o gpt-sol.

**11. AUT-34 · menos numeros em §6.1. [P]** A proporcao que voce assume **ja vale** (21 numerais em §6.1 contra 53 em
§6.2). Dos sete numerais arabes de §6.1, so um e resultado que §6.2 repete; os outros sao convencao de protocolo, que
nao sai sozinha da frase.

> DECISSAO: Então vamos deixar em como está.

**12. AUT-36 · os sete itens de trabalho futuro. [M/G]** Cada item precisa de uma limitacao amarrada 1:1, entao
acrescentar mexe na contagem de limitacoes. O seu item mais promissor (cabeca de proximo lugar) e legitimo e ja esta na
lista do `NORTH_STAR`, mas a redacao nao pode soar como algo que a dissertacao fez: dois probes guardam essa exclusao de
escopo.

> DECISSAO: Sobre o proximo lugar podemos adicionar ele no trabalho futuro relacionado ao uso do check2hgi para outros
> contextos. Esse eu faço questão de estar de alguma froma. Quanto ao resto adicione os que voê analisou e cabem estar
> lá.

### As menores, agrupadas

**13. AUT-02 · especificidades no Resumo. [P]** Metade ja foi feita: o Resumo vivo ja diz "seis conjuntos de dados...
incluindo um conjunto nao estadunidense". Sobra decidir sobre a margem de dois pontos, que e a convencao que liga o
verbo ao teste. Mesmo argumento do FAB-22.

> DECISSAO: A.

**14. AUT-09 · a frase do arco. [P]** Concordo que a atual e pior. Mas a anterior traz "correction trail", e o
`NORTH_STAR` tem guarda explicita (F4) contra fazer o resultado nulo parecer ato um de um roteiro. A saida e combinar as
duas, nao restaurar a antiga.

> DECISSAO: Vamos combinar as duas cuidado com a reescrita não soar natural.

**15. AUT-11 · alargar o objetivo 1. [P]** Discordo: acrescenta um segundo objetivo a um item que declara um so, e o
`NORTH_STAR` fixa objetivos 1:1 com capitulos. A construcao do MTLnet ja e reivindicada nas Contribuicoes. Recomendo
manter.

> DECISSAO: A.

**16. AUT-12 · o objetivo 4 aponta para o Cap.6. [P]** Defeito real e invisivel aos gates: o `\ref`
resolve, entao 0 referencias indefinidas, e o PDF imprime "(Chapter 6)" para "the protocol used in the final study".
Duas leituras, duas edicoes: **(A)** `ch:conclusion`->`ch:mobiwac`, um token (recomendada); **(B)** manter a referencia
e trocar a redacao, se a intencao era a consolidacao na Conclusao.

> DECISSAO: A.

**17. AUT-21 · citar o POI2Vec. [P/M]** Ele **esta** no pipeline (inicializacao + ancora L2). Mas o que o nosso codigo
implementa **nao e** o metodo do POI2Vec publicado (Feng et al., AAAI 2017), e cita-lo seria **misatribuicao** — a mesma
classe do erro do POI-RGNN que ja e errata deste projeto. Recomendada: nomear o mecanismo sem citar nada novo.

> DECISSAO: Vamos de i) Nomear o mecanimos citado e nada novo.

**18. AUT-23 · §2.2.3.2 contra o spec. [M]** Tres afirmacoes nao conferem, e sao o mesmo excesso do AUT-20 com outras
cabecas (a equacao mostrada e o nucleo hierarquico, nao a perda de cinco termos). **Resolva numa unica passada com
AUT-20 e AUT-21**, ou o capitulo se contradiz.

> DECISSAO: Beleza vamos resolver em uma unica passada, mas depois dessa apssada averigue se está tudo certo.

**19. AUT-25 · as duas entradas e a "correlacao". [P/M]** A sua analise tecnica confere nos dois pontos que dependiam da
codebase. O terceiro (correlacao) e defensavel como **fato de construcao** (copia com stop-gradient da mesma origem),
nao como quantidade: afirmar correlacao pediria um numero que ninguem mediu. Decida entre hedge e medicao.

> DECISSAO: hedge, deixar no left_out.md a medeição.

**20. AUT-26 · renomear para MTLChkNet. [G]** Raio medido: das 49 ocorrencias de "the joint model", 15 estao no Cap.5
**submetido**, 5 dentro de tabelas de **errata** (onde a string citada e a evidencia) e 22 em prosa livre. Nao e
search-and-replace. **Recomendo nao renomear**; se renomear, moldura **e** Cap.5 juntos, nunca so a moldura, e mexer no
manuscrito precisa da sua autorizacao explicita.

> DECISSAO: Deixar esse ponto aberto, vou perguntar para meu orientador.

**21. AUT-32 · a abertura do Cap.6 e a tarefa estatica. [P]** A omissao e coerente com a pergunta de pesquisa em tres
lugares aprovados. **(A)** manter; **(B)** uma oracao no Cap.6 dizendo que os dois primeiros estudos incluiam a tarefa
estatica, sem mexer na pergunta (recomendada).

> DECISSAO: B.
---

## §4.2 · Plano de execucao dos `[YOU APPLY]`

> **15 itens, ordenados. Nada aqui foi aplicado:** esta fase e auditoria, e nenhum `.tex` foi tocado
> (verificado: `git diff --name-only HEAD -- 'articles/dissertacao/src/**/*.tex'` volta vazio).
>
> **Regra de ouro para a Onda 1:** cinco destes itens sao "fechar como ja satisfeito", ou seja, **nao ha
> edicao**, so registro com a citacao que prova. Eles nao podem quebrar nada, e devem sair primeiro para
> tirar ruido da fila.

### Onda 1 — fechar sem editar (independentes, paralelizaveis, risco nulo)

| item       | o que fazer                                                                                                         | onde renderiza               | evidencia                                                                      |
|------------|---------------------------------------------------------------------------------------------------------------------|------------------------------|--------------------------------------------------------------------------------|
| **AUT-13** | fechar: "the joint setting" ja virou "The joint model operates under..."                                            | §1.4 p.15                    | `1_introduction.tex`:222-223                                                   |
| **AUT-15** | fechar: a frase viva ja carrega a clausula de escopo "at a coarse spatial resolution"                               | §2.1 p.18                    | `2_fundamentals.tex`, §2.1                                                     |
| **AUT-17** | fechar: as definicoes de `x_i` e `H_i` existem e ligam os cinco simbolos                                            | §2.1.1.1 p.18                | Definitions "Check-in" e "Check-in history"; probes `R12-s1bind`, `R12-s2type` |
| **AUT-33** | fechar: o Cap.6 ja faz a distincao entre sinal de treino e arquitetura; varredura com zero defeitos                 | §6.2 p.84                    | `6_conclusion.tex`, "does not require training transfer"                       |
| **AUT-22** | **nao executar aqui:** e o mesmo defeito do **GER-06**, que ja esta aprovado. Executar via GER-06 para nao duplicar | §2.2.3.1 p.23 -> §2.3.1 p.25 | GER-06 em `CONSIDERATIONS.md`                                                  |

### Onda 2 — edicoes de uma frase ou um token (independentes entre si)

| ordem | item       | a edicao                                                                                 | onde renderiza                          | cuidado                                                                                                                |
|-------|------------|------------------------------------------------------------------------------------------|-----------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| 1     | **AUT-07** | `static place categories` -> `category classification` (nome canonico registrado)        | §1.2 p.13-14                            | varrer o resto da arvore pelo mesmo padrao antes de fechar: uma ocorrencia so e o caso que a amostragem perde          |
| 2     | **AUT-27** | glosar `\mathcal{L}_k` como a perda da tarefa `k` (meia oracao)                          | §2.3.2 p.26                             | so a glosa; a reorganizacao e o AUT-29, que e `[I DECIDE]`                                                             |
| 3     | **AUT-30** | expandir `OOD` no primeiro uso + reescrever a frase de equivalencia                      | §2.4.2.2 p.29                           | corrigir tambem o comentario de `content.tex`:262-264, que afirma que OOD "never appears in prose" e hoje isso e falso |
| 4     | **AUT-04** | religar a frase das categorias semanticas ao que vem depois                              | §1.1 p.13                               | fazer junto com o 5: e um paragrafo so                                                                                 |
| 5     | **AUT-05** | cortar as duas frases de "neighboring geospatial tasks" (o conteudo ja esta em §2.2.3.1) | §1.1 p.13                               | fazer junto com o 4                                                                                                    |
| 6     | **AUT-06** | nomear as duas tarefas em vez de "both tasks"                                            | §1.1 p.13                               | usar os nomes canonicos, nao o rascunho do item                                                                        |
| 7     | **AUT-10** | glosar `hard parameter sharing` no primeiro uso do Cap.1                                 | §1.2 p.13-14 (`1_introduction.tex`:131) | a definicao formal fica no Cap.2; nao duplicar                                                                         |
| 8     | **AUT-16** | glosar "sequential" / "static" no primeiro uso do Cap.1                                  | §1.1 p.13                               | o Cap.2 ja resolve; o defeito e so no Cap.1                                                                            |
| 9     | **AUT-19** | uma frase de entrada em §2.2.2 dizendo o que infomax **e**                               | §2.2.2 p.21                             | os termos ja estao no GLOSSARY §3; sem bloqueio fail-closed                                                            |
| 10    | **AUT-24** | reescrever a transicao de abertura de §2.2.4 (nomear o sujeito)                          | §2.2.4 p.24                             | o fechamento da mesma subsecao e o AUT-25, que e `[I DECIDE]`                                                          |

### Sequenciamento e conflitos

- **Os itens 4, 5 e 6 tocam o mesmo paragrafo** (§1.1, `1_introduction.tex`). Aplicar em uma unica passada, nao em tres,
  ou o terceiro reescreve o contexto dos dois primeiros.
- **Os itens 1, 7 e 8 tocam a mesma regiao do Cap.1** (§1.1-§1.2). Sequenciais, nao paralelos.
- **AUT-24 e AUT-25 sao a abertura e o fechamento de §2.2.4.** O AUT-24 e `[YOU APPLY]` e o AUT-25 e
  `[I DECIDE]`; se voce decidir o AUT-25, faca os dois juntos, senao a subsecao fica com metade nova.
- **AUT-19 e AUT-20 estao na mesma subsecao** (§2.2.2), e o AUT-20 e a **primeira frase do corpo**. Aplicar o AUT-19
  antes de decidir o AUT-20 e seguro; o contrario tambem, mas nao simultaneamente.
- **Nenhum item da Onda 1 ou 2 casa com a string de nenhum probe.** Medido: das 29 ancoras, so as dos AUT-08 e AUT-09
  tem probe a menos de 600 caracteres (`R10-fab22`, que exige a presenca de "Istanbul as a non-United-States dataset"),
  e **os dois sao `[I DECIDE]`**, entao a Onda 2 nao chega perto de um gate.
- **Depois de cada onda:** `cd articles/dissertacao/src && make check` e `make defense`, lendo o codigo de saida de cada
  um por gate, nunca um veredito unico (§4b V11/V12). As paginas citadas acima sao do
  `build/main.pdf` de 105 paginas com mtime 2026-08-03 21:21:31; elas **andam** se a outra esteira soltar prosa, e por
  isso cada linha registra o build contra o qual foi medida.

### O que esta fora deste plano, e por que

- Os **21 `[I DECIDE]`** do §4.1: esperam a sua palavra.
- **AUT-38**: vazio no fonte.
- As quatro tarefas de infraestrutura que o AUT-03 abriu e que **nao** sao edicao de prosa: registrar a normalizacao no
  Apendice B, registrar a regra de hifen de POI no GLOSSARY (linha sua), implementar o probe `R9-poihyphen` reservado
  desde o FAB-20, e reparar o `[VERIFY]` de descricao obsoleta do probe
  `R8-vintage` (o probe passa, a descricao dele deixou de descrever o texto que ele guarda).
