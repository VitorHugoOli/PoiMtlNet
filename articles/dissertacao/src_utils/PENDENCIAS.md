# PENDENCIAS.md — o que depende de voce

> **Registro de pendencias da dissertacao (v2, 2026-07-26).** Cada item aqui esta bloqueado em um
> fato externo, uma decisao sua, ou uma aprovacao do orientador/Comissao. Nada aqui pode ser
> resolvido por um agente, e nenhum foi resolvido sozinho.
>
> A rodada de correcoes de 2026-07-26 fechou 26 dos 39 achados da revisao. O que sobrou esta
> abaixo. Auditoria completa: [`dissertation_review_v2.md`](dissertation_review_v2.md).
>
> Formato de cada item: **(A) o que falta**, **(B) por que importa**, **(C) o que eu preciso de
> voce**. Onde ja existe texto pronto ou pesquisa feita, o caminho esta indicado.
>
> **Estado do build agora:** defesa 94 pp, final 89 pp, 0 caixas estouradas, 0 citacoes indefinidas,
> 0 referencias indefinidas, 0 erros de BibTeX, lint exit 0.

---

## BLOCO 1 — bloqueiam a entrega, nao a ciencia

### 1.1 Banca, data, capa e folha de aprovacao (REV-023)

**(A)** `0_main.tex:122-124` tem tres placeholders entre colchetes (membros da banca e data). O
build de defesa comeca na folha de rosto: nao ha capa (`\imprimircapa` existe no `.sty` mas nunca e
chamado), `\campus{}` nunca e setado, e nao ha ficha catalografica. A folha de aprovacao e um
placeholder literal.

**(B)** Um documento cientificamente correto nao pode ser depositado com front matter incompleto.
Isso independe de tudo o mais nesta lista.

**(C)** Preciso de: nomes e afiliacoes dos membros da banca, a data marcada da defesa, e a decisao
sobre a capa. Sobre a folha de aprovacao, a decisao 3.9 do doc anterior continua valendo e a minha
recomendacao nao mudou: manter o placeholder honesto, que e o que o precedente do Germano de fato
faz (ele deixou o `\includepdf` do modelo COMENTADO).

**Ja feito nesta rodada:** a macro `\imprimirfolhadeaprovacao` no `abntex2-UFV.sty` tinha o nome de
**outro aluno** (`Gabriel Vita Silva Franco`) hardcoded. Estava inofensivo porque a macro nao e
chamada, mas quem trocasse o placeholder pela macro imprimiria o autor errado na folha de
assinaturas. Corrigido para usar `\imprimirautor`.

> DECISAO / DADOS: __________________________________________________

### 1.2 Pacote de aprovacoes do orientador (uma conversa so)

**(A)** Quatro decisoes que so o orientador (e possivelmente a Comissao) fecha, e que e melhor
levar juntas: (i) o **frame em ingles**; (ii) a **inclusao do capitulo CoUrb** traduzido, em que
voce e segundo autor; (iii) o **titulo final** (a opcao 1 esta ativa como titulo de trabalho, as
alternativas estao comentadas no `0_main.tex`); (iv) a **politica de errata** adotada.

**(B)** A politica de errata e a que mais trava trabalho: quase toda correcao em texto publicado
desta rodada entrou pelo mecanismo do Apendice B, e ele so fica legitimo com o aval dele.

**(C)** Uma conversa, quatro respostas.

> DECISAO: __________________________________________________

### 1.3 Fonte da bibliografia: 12 pt ou `\footnotesize`? (REV-024)

**(A)** `0_main.tex:369-370` envolve a bibliografia em `{\footnotesize ...}`. As paginas de
referencia medem **9,96 pt** contra **11,96 pt** do corpo. A regra (`UFV_COMPLIANCE.md:32`, Manual
§8) diz 12, sem excecao para bibliografia.

**(B)** O que muda o quadro: o construto foi herdado do esqueleto do **Germano**, que defendeu em
2024 com o mesmo orientador e cujas paginas de referencia medem os mesmos 9,96 pt. O Viegas, ao
contrario, usa 11,96. Ou seja: um exemplar fez e passou, outro nao fez, e a regra escrita diz 12. O
seu proprio doc de compliance antecipa esse dilema e responde **"comply, don't gamble"**.

**(C)** A edicao e uma delecao de uma linha. O motivo de precisar de voce e a consequencia: a
secao de referencias cresce cerca de duas paginas, e isso interage com a paginacao do AcademicoPG,
que so fecha depois do rascunho no portal. Vale levar junto com o item 1.2.

> DECISAO: __________________________________________________

---

## BLOCO 2 — exposicao cientifica real

### 2.1 Etica e governanca de dados — **o item mais exposto da lista** (REV-026)

**(A)** Uma varredura dos nove arquivos de capitulo por
`ethic|privacy|re-identif|anonym|consent|GDPR|LGPD|IRB|licen[cs]e|terms of use` retorna quatro
ocorrencias, e **todas as quatro sao o verbo "license"** em "the test that licenses the verb
outperforms". Nao existe **uma unica frase renderizada** sobre etica, privacidade,
re-identificacao, consentimento ou licenciamento, numa dissertacao cujo objeto sao trajetorias de
movimento por usuario.

**(B)** O simulador de banca desta rodada perguntou isso diretamente e classificou como
*obrigatoria*. Disponibilidade publica nao elimina risco de re-identificacao, e a banca vai
perguntar por que processar trajetorias individuais foi eticamente aceitavel. E muito mais provavel
perguntarem pela ausencia do que contestarem o paragrafo depois de escrito.

**(C)** A pesquisa de licenca **ja esta feita** e esta em
[`DATASET_LICENSING_FINDINGS.md`](DATASET_LICENSING_FINDINGS.md). O que ficou verificado:

- **Gowalla (Cap. 5):** o ETL consome o dump do Figshare (registro 22126586, DOI
  `10.6084/m9.figshare.22126586.v2`), rotulado **CC0**, e os tres arquivos batem com o que o
  `src/etl/gowalla/main.py:22-24` le. **Ressalva que voce precisa saber:** o CC0 foi aplicado por um
  **depositante terceiro**, e a origem que o registro cita (`yongliu.org/datasets/`) nao abriu. Nada
  prova que o depositante tinha o direito de aplicar CC0.
- **Discrepancia real:** `docs/context/DATASETS.md:187-199` documenta o **SNAP** como fonte do
  Gowalla e nao tem linha de licenca nenhuma. Mas o SNAP e um artefato diferente: 6.442.890
  check-ins sem categoria, contra 36.001.959 com a anotacao de sete categorias no dump do Figshare.
  O registro documenta uma fonte que o pipeline nao le.
- **Massive-STEPS (Istanbul):** o "Open-source; academic research" do repo pode ser substituido por
  uma licenca identificada, **Apache-2.0**, no card do Hugging Face e no `LICENSE` do GitHub.

**Preciso de voce, tres fatos, e nenhum eu posso inventar:**

1. A UFV/PPGCC exige determinacao de **CEP** para analise secundaria de dados publicos
   de-identificados? (Se exige, o numero/parecer. Se dispensa, sob qual regra.) **Nao vou fabricar
   aprovacao nem dispensa de CEP.**
2. Voce confirma o dump do Figshare como a fonte de registro do Gowalla, ciente da ressalva do
   depositante terceiro?
3. Quer que eu corrija `docs/context/DATASETS.md` para registrar a fonte que o pipeline realmente
   le, com a linha de licenca?

Com esses tres, o paragrafo de governanca se escreve sozinho, curto e factual.

> DECISAO: __________________________________________________

### 2.2 Escopo da tarefa estatica do Cap. 4 (REV-002) — **medido nesta rodada, e o resultado nao ajuda**

**(A)** Voce escreveu: *"se nao me engano usou o fclass e nao a categoria ... vamos avaliar o
tamanho do problema, porque os numeros ficaram bem proximos do DGI."* A premissa esta certa. **Eu
medi, e ela nao ajuda.**

Em `data/checkins_by_state/Alabama.parquet` (113.846 linhas): o corpus tem **275 valores distintos
de `fclass`** (a categoria fina: Airport, Coffee Shop, Seafood) e **7 categorias de topo**, que sao
o alvo. **Cada um dos 275 mapeia para exatamente uma categoria. Zero mapeiam para mais de uma.**

A cadeia, cada elo verificado em codigo: `poi2vec.py:486-487` faz
`poi_embeddings[valid] = fclass_embeddings[fclass_values[valid]]`, entao o vetor do lugar e funcao
pura do `fclass`; o `fclass` determina a categoria deterministicamente; por composicao, **o
embedding de lugar determina exatamente o rotulo alvo**. Usar `fclass` em vez de `category` deixa a
entrada *mais* informativa sobre o alvo, nao menos. E os numeros "proximos do DGI" sao consistentes
com isso, e nao um alivio: a entrada do DGI e a media one-hot dos vizinhos da mesma taxonomia.

**Importante, e a favor do documento:** isso vale para a tarefa **estatica**. A tarefa
**sequencial** dos dois capitulos e limpa (`3_cbic.tex:161-167`, `4_courb.tex:125`) e nao e afetada.
A revisao v1 nao fez essa distincao, e ela e a diferenca entre uma frase de escopo e uma retratacao.

**(B)** `apx_a_contributions.tex:91-93` conta a banca que uma submissao anterior foi atacada por
vazamento de rotulo e que o Cap. 5 responde com uma auditoria dedicada. Quem ler isso e depois ler
o Cap. 4 vai perguntar por que o Cap. 4 nao recebeu tratamento equivalente.

**(C)** Duas coisas: (i) o Cap. 4 e artigo **publicado e co-autorado**, com o Tarik como primeiro
autor, entao a frase de escopo precisa de **aviso de cortesia** a ele antes de entrar; (ii) voce
sugeriu um apendice para o tema, e concordo, e o lugar certo para a medicao acima. Autoriza?

> DECISAO: __________________________________________________

### 2.3 Conflito: sua decisao sobre Nash x instrucao do NORTH_STAR (REV-005)

**(A)** Voce decidiu: *"vamos ignorar esse erro ... para o cap. 3 nao adicionamos caveat nem
errata."* Aplicado exatamente. Mas o `NORTH_STAR.md:146` lista **"Nash-MTL caveat as in Ch.3"** como
item de honestidade **do Cap. 4**, escrito e nunca executado. Sua decisao (posterior) esta contra
uma instrucao escrita (anterior).

**(B)** Escopo do que esta em jogo, para a decisao ser barata: **so a alegacao de preferencia de
otimizador** morre. O resultado principal do Cap. 3 (paridade MTL x single-task) nao depende de qual
balanceador estava ativo, e o Cap. 5 nao usa Nash.

**(C)** Ou (a) mantenho como esta e **corrijo o NORTH_STAR** para registrar que a instrucao foi
revogada por decisao sua, ou (b) adiciono a frase de caveat no prefacio do Cap. 4. Nao resolvi
sozinho porque e uma contradicao entre duas ordens suas.

> DECISAO: __________________________________________________

### 2.4 PCGrad continua nomeado (REV-011)

**(A)** Apliquei sua redacao, "at their default configurations", em `5_mobiwac.tex:185`. Mas o
qualificador **nao cobre o PCGrad**: pelo audit (`T4_audit_and_verdict.md:26-31`), a exclusao dele e
um resultado de **fiacao**, nao de configuracao (sob a torre dupla a torre privada treina em peso
unitario de qualquer jeito, entao o metodo colapsa para peso igual). Um resultado de fiacao e
invariante a configuracao. O Nash-MTL, o outro metodo nomeado, estava corretamente ligado
(`T4:37-39`).

**(C)** Minha recomendacao, **nao aplicada**: remover `PCGrad \cite{yu2020pcgrad}, ` da frase que
cita em `:183` e deixar o Nash-MTL carregar a evidencia nomeada. Sua instrucao prevalece; o nome
fica ate voce decidir.

> DECISAO: __________________________________________________

---

## BLOCO 3 — assinaturas e itens adiados

### 3.1 Os 24 marcadores `[NEEDS SIGN-OFF]`

Voce pediu a lista. Sao 24 marcadores em 8 arquivos, todos comentarios LaTeX (**nenhum renderiza**,
entao nao ha sujeira no PDF). O risco nao e visual: e que o **Apendice C afirma** que o autor leu e
aprovou cada palavra, enquanto o proprio apendice esta marcado como nao aprovado. Voce ja decidiu
manter o Apendice C como esta, o que torna esta lista o caminho para tornar a afirmacao verdadeira.

| Arquivo | Qtd | O que e |
|---|---|---|
| `0_main.tex` | 6 | Resumo e Abstract: **par de paridade**, incluindo as mudancas de unidade inferencial desta rodada |
| `chapters/6_conclusion.tex` | 5 | Escopos de alegacao: joint model qualificado, largura 64→192, California completa, parametro escopado, convencao 64,51 |
| `chapters/apx_a_contributions.tex` | 4 | Apendice inteiro, mais as tres correcoes desta rodada |
| `chapters/5_mobiwac.tex` | 3 | Prefacio, recap, figura restaurada |
| `chapters/1_introduction.tex` | 2 | Correcao de gate L3, unidade inferencial |
| `chapters/2_fundamentals.tex` | 2 | Escopo dos 93% do Song, de-duplicacao L3 |
| `chapters/apx_b_errata.tex` | 1 | Apendice inteiro |
| `chapters/apx_c_ai_disclosure.tex` | 1 | Apendice inteiro |

**Regra que nao da para contornar:** os 6 do `0_main.tex` sao **um par**. Resumo e Abstract carregam
as mesmas alegacoes, e aprovar um sem o outro quebra a paridade. Leia os dois lado a lado.

**Um termo novo precisa entrar no GLOSSARY antes de virar canonico:** usei **"modelos ajustados"**
como equivalente PT de "fitted models" no Resumo. O GLOSSARY §6 nao tem essa entrada, e a regra e
fail-closed (o termo entra no registro **antes** de entrar no texto). Confirma o termo?

> DECISAO: __________________________________________________

### 3.2 Figura 2 do Cap. 4: rotulos em portugues (REV-022)

**(A)** A figura da arquitetura na p. 48 tem `Encoder Espacial`, `Encoder Temporal`, `Encoder
Categorico`, `Coordenadas (lat, lon)`, `Timestamps (hora, dia)`, `Categorias (POI graph)` dentro de
um capitulo em ingles, sob legenda em ingles.

**(B)** Duas personas classificaram como bloqueador visual.

**(C)** **Bloqueado por falta do fonte.** As Figuras 1, 2 e 3 existem so como PNG achatado; nao ha
`.drawio`, `.svg` nem `.py` em lugar nenhum sob `articles/dissertacao/`. Preciso de uma de duas
coisas: o arquivo fonte (com os autores do CoUrb, provavelmente com o Tarik), ou autorizacao para
**recriar** a figura do zero. Recriar levanta questao de fidelidade, porque a figura pertence a um
artigo publicado co-autorado, entao nao faco sozinho.

**Ja feito:** o rotulo do eixo da Figura 6 dizia "Score (0-1)" para uma silhueta definida em
[-1, 1]; corrigido e a figura foi regerada (o resto do PDF e byte-identico).

> DECISAO: __________________________________________________

### 3.3 Resumo e Abstract: tamanho (REV-018)

**(A)** Abstract 429 palavras, Resumo 505. No build de defesa o Resumo enche a p. 3 e deixa **duas
palavras-chave sozinhas** numa p. 4 praticamente em branco (61 caracteres na pagina inteira).

**(B)** **Nenhuma norma esta sendo violada** — verifiquei: nem o `UFV_COMPLIANCE.md` nem o Manual
04/2026 impoem limite de palavras ou paginas; a unica regra de palavra e "uma palavra por linha" nas
palavras-chave. E polimento, nao compliance.

**(C)** Sua instrucao foi deixar por ultimo, depois que o texto assentar, e concordo. Quando quiser,
ha duas rotas: (i) comprimir os dois em paridade, o que mexe em alegacao e portanto e seu; ou (ii)
uma alternativa **puramente mecanica**, um ajuste de `\clearpage`/espacamento que tira a pagina
quase-branca sem tocar em uma palavra. A (ii) da para fazer agora se quiser resolver o efeito
visual e adiar o resto.

> DECISAO: __________________________________________________

### 3.4 A auditoria de vazamento nos outros cinco datasets (REV-001) — opcional, e escopada

**(A)** O Cap. 5 agora **cita** a auditoria de aresta-futura, com os tres limites declarados: a
sonda e **linear**, rodou **so na Florida** numa inicializacao, e rodou em **builds ancestrais** da
representacao, nao na que produziu os resultados.

**(B)** Voce perguntou se da para extrapolar da Florida para os outros estados. **Nao da, e quem
proibe e o registro do proprio repositorio:** (i) o gate e **por encoder** e ja desqualificou dois
encoders, entao um teste cuja funcao e separar encoders nao pode ser assumido transferivel; (ii) a
sonda e linear e o `RESCREEN.md:94` **documenta um caso em que ela falhou** (um encoder passou no
gate linear e vazou sob o modelo sequencial); (iii) a linhagem entregue nunca foi medida.

**(C)** O que eu levantei sobre a execucao, para a decisao ser informada:

- Os inputs da sonda (`output/check2hgi/<estado>/input/next.parquet`) estao **na sua maquina local**,
  seis estados disponiveis, nao no nespedgpu. O `nespedgpu` esta acessivel (confirmei) mas o
  `PoiMtlNet` de la nao tem os parquets, so o script.
- A sonda e barata: regressao linear sobre 64 dimensoes, sem retreino.
- **O que ela fecharia:** a coberturas de datasets, um dos tres limites declarados.
- **O que ela NAO fecharia:** os outros dois. Continua linear (o limite documentado como falho), e
  continua medindo os mesmos builds a menos que a linhagem entregue seja re-exportada.
- **Ressalva importante que descobri e que muda o valor do exercicio:** tentei derivar o teto de
  autocorrelacao (a referencia contra a qual a sonda e lida) a partir dos proprios parquets e **nao
  consegui**. A hipotese de janela stride-1 nao se sustenta nos dados (checando
  `last_region_idx` contra o `shift(1)` do `region_idx` por usuario, so **18%** concordam). Ou seja:
  rodar a sonda nos outros estados produz numeros, mas o **teto** contra o qual eles se leem teria
  que ser reconstruido primeiro, e eu nao vou inventa-lo.

**Minha recomendacao:** nao rodar agora. O texto ja e honesto sobre a cobertura, e um numero novo
sem o teto correto e pior do que a declaracao de limite que ja esta la. Vale como resposta de
arguicao ("temos a auditoria na Florida, e o gate esta implementado; estender e trabalho de sonda,
nao de retreino"). Se quiser rodar mesmo assim, o caminho e reconstruir o teto primeiro.

> DECISAO: __________________________________________________

### 3.5 Higiene do repositorio, herdada (REV-007)

Dois itens claim-neutros que a auditoria estatistica anterior deixou abertos e que nao toquei
porque estao dentro de registros de resultado:

1. **Falta um gerador.** A entrada §8 de 2026-07-18 imprime os ICs exatos que os capitulos carregam,
   e todos reproduzem a partir dos arrays commitados, entao os valores sao solidos. Mas nenhum
   script commitado emite aquela entrada: o gerador da rodada joint-best nao esta na arvore.
2. **Docstrings.** `superiority_wilcoxon.py` e `m1_stats_n20.py` ainda afirmam um registro que o
   protocolo nao contem, e o `stats_n20/RESULTS.md` §1b repete. Higiene, sem efeito em alegacao.

> DECISAO: __________________________________________________

---

## Notas de rodape uteis

- **Ordem de aplicacao**, se voce for reabrir algo: registro fail-closed (GLOSSARY) → governanca →
  texto de moldura → capitulos publicados sob errata → layout → build → revisao. A ordem importa: o
  layout depende das quebras de pagina que o texto move, e o Apendice B so fica correto se escrito
  **depois** das correcoes que ele declara.
- **Um defeito meu, declarado.** Meu verificador de build reportou "0 citacoes indefinidas" enquanto
  quatro citacoes renderizavam como `(??)` nos dois PDFs. O grep era ancorado por linha e o LaTeX
  quebra os avisos em varias linhas. Tres revisores independentes pegaram o que eu tinha certificado
  como limpo. O verificador agora achata o log, le o `.blg` e falha alto em erro de BibTeX. A causa
  raiz eram arquivos `.aux` velhos **commitados** na raiz do `src/`, que o BibTeX le antes do
  `build/`; foram removidos e entraram no `.gitignore` com o motivo registrado.
- **O que nao esta pendente, embora pareca:** a discrepancia 87/83 x 89/84 de paginas era real e
  esta corrigida (agora 94/89 apos as correcoes, medido). As duas violacoes de margem foram
  eliminadas. As tabelas do Cap. 5 que estavam em 8 pt agora renderizam em 11,96 pt, tamanho de
  corpo.
