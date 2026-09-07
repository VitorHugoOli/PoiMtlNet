# Proposta: resumo e contribuições reequilibrados

> ## ⚠ CORRECÇÃO 2026-09-06, depois da revisão do Fable — DUAS AFIRMAÇÕES MINHAS ERAM FALSAS
>
> **1. O reequilíbrio NÃO poupa 233 palavras.** Poupa entre 8 e 60, e a contabilidade estava
> duplamente errada: os números propostos não batiam (o resumo proposto dá ~257 e não 214; a
> contribuição 1 **cresce** 40), e o `PLANO_ATAQUE §0` **já tinha contabilizado** o resumo (−80) e o
> *bullet* 3 (C12, −115). Contei os mesmos cortes duas vezes.
> **É a segunda vez que erro a aritmética a favor do plano** — a primeira foi a base da v1 (5 623 em
> vez de 5 499). **As ~2 700 palavras continuam a ter de sair da §5, §6 e §7.** O reequilíbrio
> justifica-se pela força da evidência, não pelo orçamento.
>
> **2. A linha "Ameaça conhecida: nenhuma no contraste interno" era FALSA**, e falsificada pela §6.1
> do próprio artigo: o controlo de concatenação sobe o embedding de lugar **+2,0 / +1,7 / +0,8**
> contra gaps de **+1,62 / +2,58 / +0,23** — em Alabama excede o gap e na Florida excede-o várias
> vezes. O artigo diz-o: *"what the check-in-level representation adds beyond them on this axis is
> small, and this control does not separate it"*. **A evidência não separa a extensão do grafo da
> injecção das features cruas.** O *bullet* 1 abre com *"We extend hierarchical graph infomax…"*,
> e essa atribuição não está estabelecida.
>
> **3. E o "30/30 folds" carrega dois quase-empates.** ✔ Verificado nos arrays: na Florida as cinco
> diferenças são **+0,276 · +0,434 · +0,029 · +0,001 · +0,425**. É verdade que todos os folds
> favorecem, mas dois por margens de um milésimo e de três centésimos. Some-se que **os três maiores
> datasets estão todos abaixo de um ponto** (FL 0,23 · CA 0,88 · TX 0,99) e que o topo da gama vem
> só de Istanbul.
>
> **O que sobrevive:** a **ordem** — a representação primeiro — continua justificada, porque o
> contraste é controlado e o dedicado já carrega toda a margem externa. O que não sobrevive é
> vendê-la como isenta de ameaça, nem como poupança de espaço.


> **Para o autor aprovar antes de o `mobiwac-writer` aplicar.** Nada foi escrito no `src/`.
>
> **O problema, medido.** O resumo dá ~49 palavras à representação e ~54 ao modelo conjunto — billing
> igual para dois resultados de força muito desigual. Os revisores não comentaram a novidade do
> Check2HGI porque nós não a destacámos.

## Porque é que o reequilíbrio é justificado pela evidência, e não por marketing

| | Check2HGI (representação) | Modelo conjunto |
|---|---|---|
| Datasets a favor | **6 de 6** | 3 vitórias em 12 células |
| Folds a favor | **30 de 30** | 8 dos 12 deltas negativos |
| Significância | 5 de 6 (FL, p = 0,067) | Holm em 3 células |
| Natureza do contraste | **controlado**: mesmas folds, mesma cabeça, mesmas janelas, mesma precisão — **só a entrada muda** | dois modelos diferentes |
| Ameaça conhecida | nenhuma no contraste interno | o controlo de capacidade inverte a California |

**E quem carrega a margem sobre a literatura:**

| | dedicado − POI-RGNN | conjunto − POI-RGNN | o conjunto acrescenta |
|---|---:|---:|---:|
| AL | +6,97 | +6,79 | −0,18 |
| AZ | +6,93 | +6,93 | +0,00 |
| Istanbul | +5,22 | +5,30 | +0,08 |
| FL | +2,86 | +3,06 | +0,20 |
| CA | +3,85 | +3,85 | +0,00 |
| TX | +3,30 | +3,16 | −0,14 |

**A representação carrega toda a margem externa. O conjunto acrescenta ruído em torno de zero.**

⚠ **Regra que governa a redacção abaixo.** O Check2HGI sobe pelo **contraste interno** (mesmo
protocolo), que não precisa de ressalva, e **não** pela margem externa, que carrega duas exposições:
o POI-RGNN corre janelas de passo 9 contra as nossas de passo 1, e na região o piso de Markov-1 está
acima do HMT-GRN nos seis datasets. A margem externa fica **uma oração nas contribuições** e sai do
resumo.

---

## A · RESUMO — proposta

**276 → 214 palavras (−62).** A ordem inverte-se: a representação primeiro, com o contraste
controlado; o modelo conjunto a seguir, como o que o sistema permite fazer com ela.

> Location-based social networks record where people go and what they do, one check-in at a time.
> If we can anticipate the next visit, mobile and urban services can cache content or reserve
> capacity ahead of demand. Two coarse questions are usually enough: the category of the next visit
> and its region.
>
> We give each check-in its own vector, describing that visit in its own context, instead of giving
> every place one fixed vector. Under a controlled comparison — the same model, folds, windows and
> training configuration, with only the input changed — this improves next-category prediction at
> every one of six datasets and in every fold, by $0.23$ to $6.29$ points of macro-averaged F1, and
> the difference is significant at five of the six. Five of the datasets are U.S. states and one is
> a non-U.S. city.
>
> On that representation we then train one model that answers both questions in a single forward
> pass. On next region it outperforms a dedicated single-task model at the two datasets with the
> largest region vocabularies, by about one point, and at the other four it stays within a two-point
> margin registered before any result was read; a control shows that a dedicated model of the same
> size reaches that gain at one of the two, so we do not attribute it to sharing. On next category
> it outperforms the dedicated model at one dataset, and the other five differences are equivalent
> to zero within half a point. One model therefore serves both tasks at a cost bounded on both axes.

### O que mudou, e porquê

| | |
|---|---|
| **Sai** o *gloss* *"(five U.S. states and one non-U.S. city, Istanbul)"* | Decisão do autor. Reaparece em meia frase no fim do parágrafo, sem nomear Istanbul — o nome está na Tabela I. **−9** |
| **Sai** *"These are normally handled by separate models. We therefore test whether one model can learn both tasks, and what sharing costs"* | Anunciava o modelo conjunto como a pergunta do artigo. Passa a ser a segunda metade, não a moldura. **−32** |
| **Entra** *"Under a controlled comparison — the same model, folds, windows and training configuration, with only the input changed"* | **É o coração da proposta.** Diz porque é que o resultado da representação é forte: não é um número maior, é um contraste limpo. **+21** |
| **Entra** *"and the difference is significant at five of the six"* | Honestidade obrigatória: na Florida é +0,23 com desvio de fold 0,42, p = 0,067. Sem isto, "todos os datasets" sobrevende. **+11** |
| **Nomes de datasets saem** (*Texas and California*, *Florida*) | Um resumo não precisa deles; a tabela tem-nos. **−12** |
| **Encolhe** a frase da capacidade, de 32 para 24 palavras | Mantém o facto e a recusa de atribuição, que a D4 exige. Perde só o "however" e o "remains open", que a frase seguinte já implica. **−8** |
| ⚠ **NÃO entra** a margem sobre a literatura | Ver a regra acima. O Fable derrubou a minha primeira proposta de a pôr aqui, e tinha razão. |

---

## B · CONTRIBUIÇÕES — proposta

**471 → 300 palavras (−171).** Três *bullets*, na ordem da força da evidência.

> - **A check-in-level representation.** We extend hierarchical graph infomax from the place to the
>   individual visit, so that each check-in carries its own vector rather than sharing one fixed
>   vector per place. Under a controlled comparison in which only the input changes, this improves
>   next-category prediction at every one of the six datasets and in every one of their folds, by
>   $+0.23$ to $+6.29$ points of macro-F1, significant at five of the six
>   (Table~\ref{tab:substrate}). A single vector per place cannot separate places that serve several
>   purposes; per-visit context recovers that distinction. The complete system is also above the
>   four external baselines re-run under our protocol, on both tasks and at every dataset.
>
> - **A single model for both tasks.** One model predicts the next category and the next region in
>   one forward pass, sharing semantic context while keeping a private spatial path for the region
>   task; to our knowledge, this is the first work to treat fine-grained region as an end target of
>   equal standing (Section~\ref{sec:related-tasks}). It outperforms a dedicated single-task region
>   model at the two datasets with the largest region vocabularies and stays within a registered
>   two-point margin at the other four; at one of the two, a dedicated model of the same size
>   reaches that gain, so we do not attribute it to sharing (Section~\ref{sec:discussion}). On
>   category it outperforms the dedicated model at one dataset and the five remaining differences
>   are equivalent to zero within half a point, even though the dedicated category model receives a
>   per-dataset search (Table~\ref{tab:results}).
>
> - **Next-region results under one protocol.** We report next-region prediction across six
>   datasets, four random initializations and five folds, against four external baselines under a
>   pre-registered statistical protocol, with the folds and region assignments released so that
>   others can report on the same footing (footnote~\ref{fn:code}). The two datasets where joint
>   training pays are the two with the largest region vocabularies, while the four smaller ones sit
>   inside the margin (Fig.~\ref{fig:deltas}); we report this as an observation and not a law, since
>   the ordering does not hold inside the pair and region count co-varies with corpus size.

### O que mudou, e porquê

| | |
|---|---|
| **Bullet 1 ganha** *"Under a controlled comparison in which only the input changes"* e *"significant at five of the six"* | O mesmo movimento do resumo: a força vem do desenho, não da magnitude. **+18** |
| **Bullet 1 ganha** a oração da margem externa | A oração delimitada que o Fable autoriza — *"re-run under our protocol"* é o que a torna defensável. **+17** |
| **Bullet 2 encolhe** de 155 para 130 palavras | Nomes de datasets saem; a cláusula da capacidade compacta-se. Nada de substância sai. **−25** |
| **Bullet 3 é reescrito** | Deixa de ser *"an empirical account of where the joint region gain appears"* — que é um sub-resultado do bullet 2 — e passa a ser **o resultado de próxima-região como recurso**: seis datasets, quatro sementes, cinco folds, quatro baselines, protocolo pré-registado, com folds e regiões publicados. O agrupamento por vocabulário sobrevive como segunda frase, com os dois confundidores. |
| **Sai** a frase sobre quantas sementes cada tabela usa | Está na §5 e na legenda da Tabela III. **−28** |

### ⚠ Duas condições do bullet 3

1. **A cláusula *"with the folds and region assignments released"* é uma promessa, e hoje é falsa.**
   O `origin/mobiwac` publica a geração retirada. **Este bullet não entra até o ramo estar
   republicado** — está com a sessão `mobiwac-branch-refactor`. Se o ramo não ficar pronto, a
   cláusula cai e o bullet fica só com o protocolo.
2. **Não afirmo escassez de literatura.** O autor sugeriu dizer que há poucos artigos na tarefa; o
   artigo já faz essa afirmação de forma delimitada na §2 (*"underexplored"*, com as excepções
   nomeadas). Repeti-la nas contribuições sem método de busca seria a classe de erro que este
   projecto já cometeu nove vezes esta semana. **O enquadramento de recurso não precisa dela.**

---

## C · Balanço

| | palavras hoje | proposta | Δ |
|---|---:|---:|---:|
| Resumo | 276 | 214 | **−62** |
| Contribuições | 471 | 300 | **−171** |
| | | | **−233** |

**O reequilíbrio poupa 233 palavras** — cerca de 8 % do corte necessário — em vez de custar. Reordenar
por força da evidência é mais curto do que dar billing igual a tudo.

E arrasta consigo: se a história se apoia na representação, o aparato do modelo conjunto na §7 (o
quinto limite, a convenção de época, a capacidade) passa a ser proporcionado em vez de dominante — que
é exactamente onde o corte tem de morder.
