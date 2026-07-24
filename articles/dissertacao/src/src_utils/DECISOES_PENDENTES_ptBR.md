# Decisoes pendentes do autor -- dissertacao v1 (pt-BR)

> Documento de decisoes para o Vitor preencher. Cada item traz o contexto, o que ja foi feito
> nesta rodada, e o campo **DECISAO:** para voce escrever a sua escolha. Onde houver texto de
> reparo pronto, o caminho de aplicacao esta indicado. Nada aqui foi auto-aprovado; voce decide.
>
> Referencias uteis: `src_utils/HANDOFF_v1.md` (mapa completo + ordem), `src_utils/_review_v1/`
> (relatorios das 18 personas), `src_utils/cbic_recompute_result.md` (numeros do CBIC).
> O PDF de defesa atual e `src/dissertacao.pdf` (87 pp).

---

## TIER 1 -- itens que bloqueiam a versao para a banca

### 1.1 Titulo da dissertacao
**Contexto.** O titulo estava em placeholder em todos os pontos (folha de rosto, cabecalho do
Resumo e do Abstract, metadados do PDF).
**Feito nesta rodada.** Setei a **opcao 1** como titulo de trabalho, ativo em todos os pontos:
*"From Representations to a Single Joint Model: Multi-Task Learning for Point-of-Interest Category
and Region Prediction"*. As tres alternativas ficaram como comentario em `src/0_main.tex` (bloco
do `\titulo`), para a conversa com o orientador.
**DECISAO (com o orientador):** confirmar a opcao 1, ou trocar por uma das alternativas comentadas
(ou uma nova). Se trocar, e so editar o `\titulo` e os dois cabecalhos-catalogo no `0_main.tex`.
> DECISAO: __________________________________________________

### 1.2 Numeros do dataset do CBIC (Cap. 3)
**Contexto.** A frase do dataset do CBIC ("This subset comprises ... users ... POIs ...
check-ins") estava com placeholders [VERIFY] nunca preenchidos no paper publicado.
**Feito nesta rodada (com correcao apos o fact-gate).** Os numeros vem do artefato CBIC-era
committado que os modelos de fato consumiram -- `data/output/florida_dgi.zip::filtrado.csv` --
recontado de forma independente: **10.460 usuarios / 64.454 POIs / 960.520 check-ins**. E esse o
registro fiel da epoca do CBIC. Preenchi o Cap. 3 com esses valores.
> IMPORTANTE: o codigo atual do ETL NAO reproduz esses numeros. Os arquivos de mapeamento de
> categoria foram expandidos em 2026-04-14 (~11 meses depois da extracao CBIC-era), entao rodar o
> ETL hoje super-conta (bruto 21.052/76.544/1.407.034; filtrado-<5 13.935/76.266/1.392.262) sobre
> o mapeamento de 2026, nao o da epoca. Por isso o `filtrado.csv` committado e a fonte correta,
> nao uma rodada nova de codigo. (Eu tinha inicialmente preenchido com os numeros do parquet
> fresco -- base errada; o fact-gate pegou, corrigido.)
Cross-check (NAO e fonte, regra do ERRATA do CBIC): a linha FL publicada do CoUrb e
990.518 / 65.009 / 20.301. POIs (64.454 vs 65.009) e check-ins (960.520 vs 990.518) batem a poucos
%. Analise completa: `src_utils/cbic_recompute_result.md`.
**DECISAO ABERTA (fail-closed -- por isso continua [VERIFY]):** o **N_users** tem um gap de ~2x
que os artefatos committados nao resolvem: `filtrado.csv` diz 10.460, o CoUrb diz 20.301. E uma
diferenca de convencao/mapeamento entre 2025 (CBIC) e 2026 (CoUrb) que nao da para reconstruir do
repo. Voce precisa: (a) confirmar a base 10.460 (CBIC-era, recomendada), OU (b) escolher outra
convencao e deixar consistente/time-indexed entre os capitulos. POIs e check-ins ficam como estao.
> DECISAO: __________________________________________________

### 1.3 Erro de atribuicao do CBIC no Cap. 5 (item B.1)  --  JA CORRIGIDO nesta rodada
**Contexto.** O Cap. 5 (herdado do paper MobiWac) dizia que o trabalho anterior (CBIC) estudou
"next-category e next-region" e "observou" transferencia negativa -- ambos falsos: o CBIC pareou
classificacao estatica de categoria com next-category (sem tarefa de regiao) e *hipotetizou* a
transferencia negativa sobre um resultado nulo de paridade.
**Feito nesta rodada (com sua autorizacao para editar o paper).** Corrigi nos DOIS lugares: no
Cap. 5 da dissertacao E no paper (fonte de referencia) `articles/[mobiwac]/src/`
(01_introduction.tex, 02_related.tex). Registrado no `articles/[mobiwac]/ERRATA.md`, no Apendice B
e no ledger do Cap. 5. O texto de reparo e o da persona 14 (aprovado por voce). Reforca a novidade
do paper (e o primeiro a adicionar regiao) e nao altera nenhum resultado.
**DECISAO:** confirmar que o texto de reparo esta bom e **enviar essa correcao junto da proxima
submissao do MobiWac** (ainda da tempo na revisao). Se quiser reformular alguma frase, o texto
esta nos dois arquivos-fonte.
> DECISAO: __________________________________________________

---

## TIER 2 -- conteudo redigido que precisa do seu OK (compila como esta)

### 2.1 Resumo (PT) + Abstract (EN)
Par redigido com paridade de alegacoes a partir dos Cap. 1 e 6 (certificado paralelo pelas
personas 03/08). Em `src/0_main.tex`. **DECISAO:** ler e aprovar/ajustar.
> DECISAO: __________________________________________________

### 2.2 Apendice C -- declaracao de uso de IA
Uma pagina, a partir do historico git. Ja inclui a linha do desvio de modelo (suite de revisao
rodou em Opus 4.8 porque os tokens do Fable acabaram). **DECISAO:** confirmar o escopo e aprovar.
> DECISAO: __________________________________________________

### 2.3 Apendice A (BRACIS) + Apendice B (errata)
Apendice A trata o BRACIS como iteracao intermediaria nao publicada (regra de contencao C4).
Apendice B lista as erratas aplicadas (inclui agora a correcao 1.3). **DECISAO:** ler e aprovar.
> DECISAO: __________________________________________________

### 2.4 Cap. 5 -- prefacio + recap duplo + figura embquality restaurada
Texto novo-no-capitulo, obrigatorio pelo desenho do coletanea. **DECISAO:** aprovar as insercoes.
> DECISAO: __________________________________________________

### 2.5 Reformulacoes das correcoes de gate  --  DEIXAR COMO ESTA (sua decisao)
**Contexto.** Reformulacoes neutras em relacao a alegacao, aplicadas nos gates: escopo dos 93% do
Song (`2_fundamentals.tex`), convencao do 64,51 (`6_conclusion.tex`), de-duplicacao L3
(`1_introduction.tex` + `2_fundamentals.tex`). Todas com caminho de reversao nos comentarios.
**Sua instrucao nesta rodada: deixar como esta.** Nenhuma acao necessaria; registrado aqui so para
constar. (Se um dia quiser reverter, os comentarios `[NEEDS SIGN-OFF]` no fonte indicam o texto
original.)
> DECISAO: mantido como esta (por decisao do autor)

---

## TIER 3 -- correcoes da suite de revisao que precisam da sua chamada

> Estas NAO foram aplicadas ainda (exceto onde indicado). Precisam de decisao porque tocam
> alegacao, escopo ou voz. Detalhe por item nos relatorios em `src_utils/_review_v1/`.

### 3.1 (MJ-2) Nome do teste de superioridade
Cap. 2 diz Wilcoxon; Cap. 5 usa t pareado com n=4. **DECISAO:** escopar para o frame+Cap.5 e
defender a escolha, OU uniformizar. > DECISAO: ______________________________

### 3.2 (MJ-3) Escopo da validacao cruzada usuario-disjunta
Cap. 2 vende no documento inteiro; so o Cap. 5 usa. **DECISAO:** escopar + 1 frase de aviso no
Cap. 3. > DECISAO: ______________________________

### 3.3 (MJ-4) Enquadramento de pre-registro da regiao
Declarar que regiao foi pre-registrada como nao-inferioridade; superioridade confirmada post-hoc.
**DECISAO:** aplicar essa frase? > DECISAO: ______________________________

### 3.4 (MJ-5) Vintage dos dados  --  CONFIRMADO nesta rodada: 2009-2011
O recompute mostrou que os check-ins de Florida vao de **2009 a 2011** (2009: 40.304; 2010:
769.792; 2011: 596.938). O Cap. 6 diz "2009 and 2010" -- incompleto. **DECISAO:** trocar para
"2009 to 2011"? (recomendado, e o que os dados mostram). > DECISAO: ______________________________

### 3.5 (MJ-8) Ponte do termo "next-POI"
Uma frase de prefacio no Cap. 3/Cap. 4 ligando "next-POI" (uso dos papers) a distincao de tarefas
do frame. **DECISAO:** aplicar? > DECISAO: ______________________________

### 3.6 (MJ-17) Ponderacao de classe (class-weighting)
Cap. 2 diz entropia cruzada ponderada por classe; Cap. 5 usa nao-ponderada e relata que ponderar
PIORA. Contradicao. **DECISAO:** corrigir o Cap. 2 para bater com o Cap. 5 (metodo real).
> DECISAO: ______________________________

### 3.7 (MJ-18) Costura do nome "MTLnet"
O nome "MTLnet" e citado como artefato central "introduzido no Cap. 3", mas o Cap. 3 nao usa o
nome (fiel ao CBIC publicado). **DECISAO:** adicionar o nome no prefacio do Cap. 3, OU ajustar a
frase do frame. > DECISAO: ______________________________

### 3.8 Visual (persona 18)
Fig. 2 com rotulos em portugues (regenerar em EN); Fig. 3 codificada so por cor Food/Shopping
(regenerar segura para tons de cinza); Tabela 1 com ~1cm de overflow. **DECISAO:** regenerar os
assets? (sao regeneracoes de figura, nao de texto -- posso fazer se voce autorizar).
> DECISAO: ______________________________

### 3.9 Melhoria de calibracao (exemplares Locus) -- pagina de aprovacao
Os 3 exemplares do Locus mostram uma folha de aprovacao real; a v1 usa um placeholder entre
colchetes. A arvore do Germano ja traz `pdfs/Modelo-pgs-de-assinaturas.pdf`. **DECISAO:** incluir o
modelo de folha de assinaturas no build de defesa (deixa o PDF mais "completo"), OU manter o
placeholder honesto (a folha assinada e inserida na/apos a defesa). > DECISAO: ____________________

### 3.10 Movimentos opcionais de excelencia (persona 17, lente SBC-CTD)
Nao sao defeitos: tabela contribuicoes->alegacoes no cap.1 (§1.6); tabela consolidada de resultados
cross-chapter no cap.6; apendice de artefatos/reprodutibilidade. **DECISAO:** quer algum desses?
> DECISAO: ______________________________

---

## Itens que NAO precisam de decisao (so para constar)
- Vetado (persona 14): a varredura mecanica de preposicoes "at [dataset]" -- colide com escopos de
  veredito congelados; nao rodar em massa.
- Guarda: se algum relatorio recomenda ADICIONAR uma citacao nova (ex.: arXiv:2311.04698), NAO foi
  adicionada; verificar na fonte antes de incluir (protocolo fail-closed).
