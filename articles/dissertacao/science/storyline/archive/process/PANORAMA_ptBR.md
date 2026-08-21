# Panorama da dissertação — a visão geral antes dos primeiros rascunhos

> **O que é este documento.** Você pediu a visão geral do todo antes de partirmos para os primeiros
> rascunhos. Este arquivo reúne, em português, a história completa que a dissertação conta, o estado
> de cada decisão que você já tomou, o que ainda falta o seu aval, e o caminho até o primeiro
> rascunho. Ele aponta para os arquivos de detalhe quando você quiser aprofundar. É
> meta-documentação (não é texto da dissertação), então é livre da lei de escrita; as frases do
> capítulo em si serão em inglês e só depois do seu aval.
>
> **Estado geral.** A revisão de narrativa (10 arquivos em `storyline/`) e a checagem de quatro
> especialistas (`10_specialist_check`) estão feitas. Você respondeu aos 11 itens de aval em
> `AVAL_NECESSARIA_ptBR.md`. Os seus "APROVO COM AJUSTE" trouxeram direções novas e muito boas — três
> delas geram sub-afirmações que pedem um último aval seu, reunidas em `AVAL_NECESSARIA_2_ptBR.md`.
> Fora isso, estamos prontos para começar a redigir.

---

## 1. A história em uma frase (a logline honesta)

**Pergunta de pesquisa:** *o aprendizado multitarefa (MTL) ajuda a predição de POI — próxima
categoria e próxima região — e do que depende a resposta?*

**A resposta, em uma frase:** um único modelo que prevê ao mesmo tempo *que tipo de lugar* uma pessoa
visitará em seguida e *em que região* deveria ser possível; um modelo conjunto ingênuo sobre um
embedding no nível do lugar **não** supera dois modelos dedicados (CBIC); a razão é a **representação**,
não a arquitetura de compartilhamento (CoUrb); e quando cada visita passa a ter o seu próprio vetor e
as duas tarefas passam a compartilhar **através de um tronco de atenção cruzada** — trocando informação
entre os dois fluxos por atenção, em vez de possuírem camadas ocultas em comum, com um caminho espacial
privado para a região — um único modelo **finalmente supera** os dois dedicados — categoria em todas as
bases, região em quatro de seis e não inferior (TOST, margem de dois pontos) nas outras duas (MobiWac).

> *Correção da re-revisão (auditor de honestidade): a versão anterior desta frase dizia "via atenção
> cruzada em vez de um tronco comum", o que negava o tronco compartilhado que o próprio mecanismo do
> MobiWac credita (o controle de congelamento atribui o ganho a "um tronco compartilhado mais forte";
> o GLOSSÁRIO define a pilha de atenção cruzada COMO o tronco). A forma licenciada é "compartilham
> através de um tronco de atenção cruzada, sem possuir camadas ocultas em comum". Nunca redigir o
> Cap. 1 a partir da frase antiga.*

Duas coisas tornam essa frase forte e honesta:
1. **É um arco de verdade** (null → diagnóstico → resolução), não um "propusemos X e funcionou". Isso é
   raro e valioso: a literatura de dissertações premiadas trata um null bem diagnosticado como sinal de
   excelência, não como fraqueza.
2. **A resolução é de dois fatores**, não de um. O ganho final veio de uma representação no nível do
   check-in **e** de uma topologia de compartilhamento redesenhada. Creditar só à representação
   contradiria o próprio texto do MobiWac. Essa é a regra de honestidade que atravessa tudo.

---

## 2. O arco em três atos (a espinha da coletânea)

| Ato | Artigo | O que fez | O resultado honesto | O que forçou em seguida |
|---|---|---|---|---|
| **Setup** | **CBIC** (2025, publicado) | Primeiro modelo conjunto: embedding no nível do lugar (DGI) + compartilhamento rígido | **Null honesto** — o conjunto não supera consistentemente dois dedicados, e custa mais para treinar. O artigo **levantou a hipótese** desse limite e os resultados a **sustentaram** ("lend weight to"), para aquela configuração. | Três hipóteses para o null; a dissertação testa a da representação primeiro |
| **Diagnóstico** | **CoUrb** (2026/SBRC, publicado) | Segura a arquitetura, **decompõe/enriquece a entrada** (espaço + tempo + categoria) | A categoria sobe muito — evidência de que a **representação** é a alavanca. (Comparação controlada: só a entrada mudou.) | Se enriquecer ajuda, qual o teto de enriquecer *no nível do lugar*? |
| **Resolução** | **MobiWac** (2026, submetido, sob revisão) | Representação no **nível do check-in** (Check2HGI) + modelo conjunto de atenção cruzada com caminho espacial privado | Um modelo conjunto **supera** os dois dedicados: categoria em todas as seis bases; região em 4/6 + não inferior (TOST ±2pp) nas outras duas | Resolve a pergunta; abre o trabalho futuro (next-POI, representação indutiva) |

**O mecanismo que faz o arco girar** (o coração intelectual): um lugar não tem identidade fixa entre
visitas — o mesmo café é almoço de quarta e balada de sábado, e um único vetor por lugar não pode
estar certo para os dois. É por isso que a representação no nível do lugar é o limite, e por isso
descer ao nível do check-in é a saída. Essa frase precisa aparecer cedo (na Introdução), como a
hipótese que a jornada testa.

---

## 3. A escolha das tarefas (a sua preocupação principal) — endossada

Este foi o ponto que você levantou, e a investigação confirmou que ele é mais profundo do que "não
prevemos o próximo lugar". **O par de tarefas evoluiu ao longo do arco:**

- CBIC e CoUrb: *classificação estática de categoria* + *próxima categoria*.
- MobiWac: *próxima categoria* + *próxima região*.

A constante é a **próxima categoria**; a segunda tarefa mudou de estática para próxima região. Isso
importa porque a reversão CBIC→MobiWac mudou **três** coisas ao mesmo tempo: a representação, a
topologia de compartilhamento **e** o par de tarefas. O arco credita as duas primeiras; a terceira
precisa ser dita em voz alta, senão um examinador lê um "gol contra".

**Como endossamos isso (a linha que você aprovou, com o seu ajuste):**
1. A mudança de tarefa é, em parte, um **corolário** da mudança de representação (sob uma
   representação por-visita, a tarefa estática vira a menos natural). — *Item 1, aprovado com ajuste.*
2. **Mais forte ainda (o seu ajuste):** as duas tarefas são escolhidas por **razão de literatura e de
   utilidade** — são mais presentes na literatura, têm mais uso em problemas reais, e alimentam a
   predição do **próximo lugar** (a tarefa mais citada), que seria um passo seguinte natural. Essa é a
   argumentação que dá força, e ela precisa aparecer junto com o corolário, não no lugar dele.
3. As duas tarefas são **coordenadas complementares da mesma próxima visita** (o tipo e a região), de
   modo que um único modelo prevê ambas para a próxima visita do mesmo usuário. — *Item 2, aprovado.*
4. **Não é mais fácil, é mais difícil:** a próxima região tem de centenas a milhares de classes (520 a
   8.501), contra sete da tarefa estática abandonada. — *fato desarmante, aprovado.*

O detalhe completo e verificado está em `02_task_choice_endorsement` (o corolário) e
`09_application_scope_breadth` (o além-da-mobilidade). As sub-afirmações novas do seu ajuste (a
comparação "mais presente na literatura" e o enquadramento "alimenta o próximo lugar") pedem um último
aval — estão no `AVAL_NECESSARIA_2_ptBR.md`, item N1.

---

## 4. O estado das 11 decisões (o que você já resolveu)

| # | Afirmação | Sua decisão | Estado / próximo passo |
|---|---|---|---|
| 1 | Mudança de tarefa é corolário da representação | **APROVO C/ AJUSTE** | Adotar a versão corrigida ("não natural", não "incoerente") **e** somar a razão de literatura/utilidade → sub-aval N1 |
| 2 | Categoria e região = duas coordenadas da próxima visita | **APROVO** | Escrever; enfatizar complementaridade e "difícil, não trivial" |
| 3 | Pergunta respondida sobre dois pares diferentes | **APROVO C/ AJUSTE** | Enquadrar via o **trabalho futuro do CBIC** + reviews de MTL → sub-aval N2 |
| 4 | Por que nível de check-in (o teto do vetor-por-lugar) | **APROVO** | Escrever a ponte na Introdução e/ou recap do Cap. 5 |
| 5 | Por que representação antes de arquitetura | **APROVO** (breve, 1 parágrafo) | Um parágrafo, como seguro contra a banca |
| 6 | CoUrb isola representação, não revisita MTL-vs-única | **APROVO C/ AJUSTE** | Trazer, mas **breve e com critério** (não quebrar o fluxo) — nota de redação |
| 7 | Nomear a reversão da transferência negativa | **APROVO C/ AJUSTE** | Adicionar o **porquê o MTL vence sem transferência positiva** → sub-aval N3 |
| 8 | Arco honesto como espinha da Introdução | **APROVO** | Escrever a Introdução em torno do arco |
| 9 | Subseção de recap CoUrb → MobiWac | **APROVO** | Escrever o recap na cabeça do Cap. 5 |
| 10 | Tabela-ponte na Introdução | **NÃO SEI** | **Resolvido:** não fazer tabela isolada; dobrar a lógica nos recaps (5/9) e no arco (8) — ver §5 |
| 11 | Motivar além da mobilidade | **APROVO C/ AJUSTE** | Buscar artigos externos para sustentar; **não** ficar só em mobilidade → sub-aval N1 + trabalho pendente (§6) |

**Regra que atravessa tudo (não precisa de aval, é lei):** a resolução é sempre de dois fatores; os
verbos ficam presos aos testes ("supera" só com superioridade pareada, "iguala" só com TOST ±2pp); o
AZ (0,00) nunca é promovido; conclusões do CBIC/CoUrb são datadas ("a conclusão daquele momento, para
aquela configuração").

---

## 5. Item 10 resolvido — a tabela-ponte

Você respondeu **"NÃO SEI"**, com receio de repetição e de ocupar espaço, e sem saber se é praxe. A
investigação resolve isso a seu favor: o exemplar do Viegas (defendido, mesmo orientador, mesmo
formato de coletânea) **não usa tabela-ponte** — ele usa **subseções de recap** (§4.2.4 e §5.2.1, "The
MTLnet framework"). Ou seja, a praxe do precedente mais próximo é exatamente o que você já aprovou nos
Itens 5, 6 e 9.

**Recomendação:** não fazer a tabela-ponte isolada. A lógica "o que cada artigo mudou → o que forçou"
já será carregada (a) pelo parágrafo do arco na Introdução (Item 8) e (b) pelas subseções de recap
(Itens 5, 6, 9). Isso respeita o seu receio de repetição e segue o precedente. A tabela que **fica** é
a de linhagem de modelos (DGI → HGI → MTLnet → ST-MTLNet → Check2HGI → modelo conjunto), que já existe
e é no nível dos *modelos*, não do *argumento* — não compete com o texto. **Confirme se concorda com
essa resolução** (é a única decisão sua ainda em aberto do documento original).

---

## 6. O que ainda não consegui fazer (e por quê)

**A busca de literatura além-da-mobilidade (Item 11) está parcial.** Você autorizou buscar artigos
externos para sustentar os usos além da mobilidade (recomendação, planejamento urbano, controle de
trânsito). Rodei a busca, mas:
- *(Atualizado na re-revisão)* Você restaurou a conexão do OpenAlex, mas o conector re-registrado
  ainda pede **autorização** no app (Configurações → Conectores → conectar o servidor de literatura).
  Assim que autorizado, a busca dedicada roda.
- O **arXiv** funcionou, mas devolveu sobretudo ruído ou trabalhos que **já estão no corpus**, e vários
  candidatos tentadores (predição de *fluxo de multidão* a nível de cidade, tipo ST-ResNet) são de uma
  **tarefa diferente** — densidade agregada, não a próxima região de um indivíduo. Citá-los como "por
  que a próxima região importa" seria uma conflação que um examinador de POI pega. Por isso, seguindo o
  fail-closed, **não proponho nenhuma citação externa nova ainda**.

**O que temos agora, verificado e seguro:** o material do próprio corpus (CBIC cita visão
computacional, PLN, saúde, recomendação; CBIC `basis` cita planejamento urbano; CoUrb cita
recomendação + análise de mobilidade urbana e a origem dos codificadores em ecologia/sensoriamento).
Detalhe em `09_application_scope_breadth`.

**Recomendação:** quando o OpenAlex reconectar, faço uma busca dedicada e **abro/verifico** cada
candidato antes de propor. Até lá, a problematização já pode ser redigida com o material do corpus, e
ampliada depois. Isso é o item N1 do `AVAL_NECESSARIA_2_ptBR.md`.

---

## 7. O mapa dos capítulos (onde cada movimento aprovado entra)

- **Cap. 1 — Introdução** (a redigir; maior alavancagem). Carrega: o arco como espinha (Item 8); as
  apostas além da mobilidade (Item 11); o mecanismo "mesmo lugar, visitas diferentes, mesmo vetor"
  (aprovado); a ponte "por que nível de check-in" (Item 4); o "por que representação antes de
  arquitetura" em um parágrafo (Item 5); a escolha das tarefas com razão de literatura/utilidade
  (Itens 1, 2, 11).
- **Cap. 2 — Fundamentos** (rascunhado; §2.1 e §2.5 são prosa real). Ajustes apontados pelos
  especialistas (ação sua): reconhecer a troca de par na página (§2.1/§2.5); escopar os 93% de
  predizibilidade no §2.1; migrar a frase "partição do mapa é a formulação padrão" para §2.1/§2.4;
  marcar o status "sob revisão" onde o resultado do MobiWac aparece.
- **Cap. 3 — CBIC** (reproduzido). Prefácio-cápsula-do-tempo (venue/status/o que é revisado depois).
- **Cap. 4 — CoUrb** (reproduzido, traduzido). Prefácio + a fronteira "isola representação, não
  revisita MTL-vs-única" (Item 6, breve); o "por que representação antes de arquitetura" pode morar
  aqui (Item 5).
- **Cap. 5 — MobiWac** (versão de registro). Subseção de recap CoUrb→MobiWac (Item 9); a ponte "por
  que check-in" pode morar aqui (Item 4).
- **Cap. 6 — Conclusão** (a redigir). A resposta em dois pares (Item 3); a reversão da transferência
  negativa + o porquê o MTL vence (Item 7); fecha o laço com a Introdução.

**Correções de governança (ação sua, apontadas pelos especialistas):** reconciliar o NORTH_STAR §4/§6
(ainda afirma o protocolo do CoUrb como fato) com a retração UW-3 ([VERIFICAR]); sincronizar o mapa de
citações do §2.3. Detalhe em `10_specialist_check` (resumo de ações no topo).

---

## 8. O caminho até o primeiro rascunho

1. **Você:** confirmar a resolução do Item 10 (§5) e avaliar os três sub-avais em
   `AVAL_NECESSARIA_2_ptBR.md` (N1 literatura/além-mobilidade; N2 o CBIC via trabalho futuro; N3 o
   porquê o MTL vence).
2. **Eu (quando o OpenAlex reconectar):** busca dedicada de âncoras além-da-mobilidade, abrindo e
   verificando cada uma; nada de citação de memória.
3. **Eu:** para cada afirmação aprovada, passar pelos revisores 07 (honestidade) e 14 (orientador
   adversarial) antes de virar frase.
4. **Eu:** redigir o **primeiro rascunho do Cap. 1 (Introdução)** em inglês — é o de maior alavancagem
   e onde a maioria dos movimentos aprovados mora. (Sugiro começar por ele; alternativa: os prefácios
   e recaps, que são curtos e destravam a unidade.)
5. **Eu:** rodar o portão de fatos (05/06/07) sobre a prosa nova; te devolver para o G4 (seu aval).

**Sugestão de ordem de escrita:** Introdução (Cap. 1) primeiro — ela fixa a voz e o arco que todo o
resto herda. Se preferir um começo menor e de baixo risco, os três prefácios-cápsula e as duas
subseções de recap são curtos, destravam a unidade da coletânea, e não dependem dos sub-avais
pendentes.

---

## Arquivos de detalhe (para aprofundar)

- `AVAL_NECESSARIA_ptBR.md` — os 11 itens originais com as suas decisões.
- `AVAL_NECESSARIA_2_ptBR.md` — **os três sub-avais novos** vindos dos seus ajustes (N1, N2, N3) + a
  confirmação do Item 10.
- `02_task_choice_endorsement` — a escolha das tarefas (o corolário, corrigido).
- `09_application_scope_breadth` — o além-da-mobilidade (material verificado do corpus).
- `10_specialist_check` — os quatro especialistas + o resumo de ações (o que está corrigido vs o que
  é ação sua).
- `01`…`08` — a revisão de narrativa completa (arco, coesão, beats, craft, honestidade,
  recomendações, lados subponderados).
