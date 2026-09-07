# PLANO DE ATAQUE — camera-ready MobiWac 2026

> **Prazo: 2026-09-07 AoE.** **Tecto: 8 páginas** (decisão do autor; 10 é o máximo absoluto, com
> 2 páginas a 120 € cada). **Estado: 11 páginas, portão verde, build 0/0/0.**
>
> Este plano junta duas coisas que até agora corriam separadas: o **corte** para 8 páginas e a
> **resposta aos três pareceres do EDAS**. Junta-as porque a maior parte dos pedidos dos revisores
> ou não custa espaço ou **poupa** espaço — e um deles, o do R3, aponta exactamente para onde o
> orçamento já mandava cortar.
>
> Companheiros: [`EDAS_REVIEWS.md`](EDAS_REVIEWS.md) (os pareceres, ponto a ponto) ·
> [`CAMERA_READY.md`](CAMERA_READY.md) (números e vereditos) ·
> [`CONSOLIDATION_PLAN.md`](CONSOLIDATION_PLAN.md) (a Fase 2, de onde vem a lista base).

---

## 0 · A aritmética, calibrada e não estimada

A divisão ingénua (8 296 palavras ÷ 11 páginas) dá 754 palavras/página e **está errada**: as páginas
1–2 levam título, autores, resumo e dois flutuantes. A calibração honesta é a **v1**, que tinha o
**mesmo conjunto de flutuantes** (Tabelas I–III, Figuras 1, 2 e 4; a Fig. 3 estava cortada nas duas):

| | palavras vivas de secção | páginas |
|---|---:|---:|
| **v1** (o commit `0834419b`, o que os revisores leram) | **5 499** | **8** |
| hoje | **8 296** | 11 |

**Alvo: ~5 500 palavras. Há que largar ~2 800**, mais as adições — **~2 890 no total.**

> ✅ **Duas coisas que o autor fechou a 2026-09-06 e que aliviam a conta.** Leu as normas do venue e
> confirmou: **não há `\IEEEpubid`** (o formulário de copyright é enviado depois do upload) e **não
> há agradecimentos nem financiamento a declarar**. A página 1 não perde espaço nenhum, e o portão em
> fase 2 deixou de o exigir — as 8 páginas passam a ser o único critério. Poupa ~25 equivalentes e,
> mais importante, tira dois bloqueadores do caminho crítico.

> ⚠ **CORRECÇÃO (Fable, 2026-09-06): a minha base estava errada e o erro era a favor do plano.**
> Escrevi 5 623 para a v1; são **5 499** (verificado nos três candidatos de 8 páginas: `0834419b`
> 5 499, `97f01a50` 5 496, `f66f8a73` 5 447). Isso são 150 palavras a mais para cortar.
>
> ### 🔴 E o pior: **a lista de cortes da §2 NÃO chega às 8 páginas.**
> Somadas contra as passagens reais, as reduções de prosa dão **~1 850**, não 1 975 — várias
> estimativas estavam altas (C6 dá ~85 e não 150; C15 dá 34 e não 50; C17 dá ~45 e não 70; a "0,2
> página" de cabeçalho da C1 é ~0,03) e outras estavam baixas (C7 dá 285, C11 até 220), e os erros
> quase se anulam. Os flutuantes dão **~330** e não 800: a Fig. 4 vale ~0,20 página (~195 palavras),
> não ¼, e encolher a Fig. 1 para largura de coluna poupa ~0,13 página **mas põe as etiquetas TikZ a
> ~4,5 pt — ilegíveis em impressão a preto e branco.**
>
> **Total realista: ~2 180 contra ~2 950 necessárias. Falta uma página inteira.**
>
> **O que fecha as ~770 em falta, e o plano não tinha:**
> - **Podar a bibliografia.** 33 `\bibitem` ocupam ~0,95 de página. Cortar 8, pela ordem e com a
>   lista de nunca-podar do `PLAN_8PAGES §3` → **~210**.
> - **O resumo**, de 278 para ~200 palavras (a v1 tinha 233) → **~80**.
> - **Aparar 25 % dos blocos que ninguém tocou**: §4.1 (379), §4.2 (485) e os baselines da §5 (425)
>   → **~320**.
> Com estes três, chega-se a 8 — apertado.

Onde o texto cresceu, por secção — e é aqui que o corte tem de morder:

| secção | v1 | hoje | Δ |
|---|---:|---:|---:|
| §7 Discussão | 302 | **1 236** | **+934** ← quadruplicou |
| §5 Setup | 1 286 | 1 642 | +356 |
| §6 Resultados | 1 485 | 2 074 | +589 |
| §8 Conclusão | 102 | 507 | +405 |
| §4 Método | 604 | 868 | +264 |
| §2 Relacionado | 803 | 933 | +130 |
| §1 Introdução | 771 | 765 | −6 |
| §3 Problema | 270 | 271 | +1 |

**A §7 sozinha explica 35 % do excesso.** E não foi descuido: quase todo o crescimento é o **quinto
limite** (o controlo de capacidade, +284 palavras) que o autor mandou declarar, mais o parágrafo da
hipótese TX/CA. Foram decisões certas e são agora o alvo principal.

---

## 1 · O que ACRESCENTA, e porquê vale a pena

Total: **+117 palavras**. Menos de 5 % do corte. Cada uma responde a um revisor ou a uma decisão do
autor, e as três primeiras respondem à acusação mais séria dos três pareceres.

| # | Onde | O quê | Custo | Responde a |
|---|---|---|---:|---|
| **A1** | §3, a seguir à lista das sete categorias | A proveniência da taxonomia: são as **sete categorias de topo do próprio Gowalla**, distribuídas com o dataset; as etiquetas de Istanbul colapsam nas mesmas sete raízes pelos seus pais de topo. | **+40** | **R1, ponto 1** — o único ponto dos três pareceres em que um revisor relata **confusão** e não um desejo. É facto documentado, não argumento. |
| **A2** | **Só nas contribuições da §1, NÃO no resumo** | Uma oração delimitada: o sistema completo está acima dos quatro baselines externos re-executados sob o nosso protocolo. | **+15** | Decisão do autor, **recalibrada** — ver §1.1, e a razão não é a que eu dei |
| **A3** | §1, bullet 2 | Mover para a introdução a razão operacional do modelo único (um artefacto para treinar, versionar e servir; uma passagem em vez de duas). Hoje está enterrada na §4. | **+12**, e liberta 22 na §4 → **−10 líquido** | **R1, ponto 2** — ele leu a resposta na §4 da v1 e achou-a insuficiente. Repeti-la lá não fecha o ponto; movê-la para onde ele procura, fecha. |
| **A4** | §5, uma oração | Restaurar o período de recolha dos dados (2009–2011), que a reescrita deixou cair. | **+5** | **R3, reprodutibilidade** |

### 1.1 · Sobre a margem da literatura, e uma correcção que faço a mim próprio

Medido: a margem sobre a literatura aparece **zero vezes no resumo**, **zero vezes nas
contribuições**, e duas vezes no corpo. Entretanto os deltas conjunto-contra-dedicado (−0,19 a +1,21)
estão no resumo, na introdução, nos resultados, na discussão e na conclusão.

**O artigo lidera com as suas margens mais fracas e enterra as mais fortes, por um factor de cinco:**

| | margem |
|---|---|
| Conjunto vs. dedicado | **−0,19 a +1,21** |
| Sistema vs. literatura publicada | **+3,06 a +6,93** (cat.) · **+3,55 a +6,04** (reg.) |

Dois revisores escreveram *"has merit but mostly incremental"*. É a leitura que se tem de um artigo
que põe em destaque diferenças de décimas.

⚠ **EU CORRIGI-ME NO SÍTIO ERRADO, E O FABLE APANHOU-O. A posição final é intermédia.**

Eu tinha proibido a margem externa de liderar; depois "corrigi-me", dizendo que devia ter proibido a
**confusão** e não a **afirmação**. O argumento de atribuição está certo mas **não é o que interessa**.
A razão pela qual o número não pode abrir o resumo é **comparabilidade**, e é verificável:

1. ✔ **O piso de Markov-1 está acima do HMT-GRN nos SEIS datasets** (e acima do STAN em 4, do ReHDM
   em 3). A §6 do próprio artigo imprime isso. Um resumo que abra com *"≥ 3,55 acima da referência de
   região mais forte"* entrega ao leitor a frase seguinte de graça: *"os baselines deles perdem para
   uma tabela de transições de primeira ordem."*
2. **POI-RGNN e o piso Markov-K correm janelas stride-9 não sobrepostas contra as nossas stride-1.**
   O `CAMERA_READY §9 B1` di-lo: *"inofensivo a +46 pp, **exposto a +3,06**"*.
3. **O modelo dedicado também está acima de todas as externas** (+2,86 cat / +3,27 reg). Logo a
   margem externa **não diz nada sobre o modelo conjunto** — que é o que o *"incremental"* visava.
4. *"Incremental"* é uma nota de **Originalidade**, e o artigo já está aceite: **não há nota para
   mover**. O único leitor que ainda importa é quem compare a v1 com a versão final — e, sem nota aos
   chairs, trocar −0,19…+0,19 por +3…+7 no resumo lê-se como reposicionamento comercial.

**Decisão: A2 sai do resumo. Fica UMA oração delimitada nas contribuições** — *"the complete system,
representation and model, is above the four external baselines re-run under our protocol"* — que é
defensável e custa ~15 palavras em vez de 60. **A proibição original estava bem calibrada para o
resumo; o que estava a mais era a extensão dela às contribuições.**

### 1.2 · A próxima região como contribuição de recurso — parecer

O autor propõe uma terceira contribuição: **publicar resultados de próxima-região para a literatura
usar**, dado haver poucos artigos na tarefa. **Subscrevo, e é o mais forte dos dois pontos dele para
responder ao *"incremental"***, porque uma contribuição de recurso não compete com o que existe:
preenche uma lacuna.

O artigo já reivindica *"to our knowledge, the first work to treat fine-grained region as an end
target of equal standing"* — mas enquadra-a como novidade **arquitectural**. O enquadramento de
recurso é mais defensável e temos com que o sustentar: seis datasets, quatro sementes, cinco folds,
protocolo estatístico pré-registado, quatro baselines externos no mesmo protocolo.

**Duas ressalvas, e a segunda é bloqueante:**
1. *"não encontrámos muitos artigos"* tem de virar afirmação delimitada e verificável — quantos,
   procurados como, em que período. Uma ausência afirmada sem método é a classe de erro que este
   projecto já cometeu **nove vezes** esta semana.
2. **Uma contribuição de recurso só existe se o recurso existir.** Ver §4: o ramo público publica
   hoje a geração retirada. **Os dois pontos são um só.**

⚠ **Custo:** o enquadramento cabe na frase de novidade que já existe, portanto **+0 palavras** — é
reescrita, não adição. A afirmação sobre a escassez de literatura **não entra** sem o método da
busca, e esse não cabe no orçamento. Recomendo o enquadramento sem a afirmação de escassez.

---

## 2 · O que CORTA — ordenado, com o que cada corte também resolve

Alvo ~2 650 + 117 = **~2 770 palavras**.

| # | Onde | O quê | Poupa | Também resolve |
|---|---|---|---:|---|
| **C1** | §3 → §4 | **Fundir a §3 na §4** como primeira subsecção. | **~40** + ~0,2 pág. de cabeçalho | **R3, pedido explícito** (*"may be too short, so it may be merged with Section IV"*). Único sítio onde um pedido de revisor e o orçamento apontam no mesmo sentido. |
| **C2** | §7 | Comprimir o quinto limite: manter 97,4 % · +0,41 · p = 0,010 · cinco folds · *"Texas has no control of matched size"*; os braços de 528 e do Texas passam a uma oração cada. | ~150 | — |
| **C3** | §7 | O parágrafo da hipótese de categoria em TX/CA — explica um −0,13 que o próprio artigo chama equivalente a zero. | ~130 | R3 (narrativa) |
| **C4** | §6 | Reconciliação do piso de Markov: 2 parágrafos → 3 frases. | ~190 | R3 (narrativa) |
| **C5** | §2 | Rastreio de balanceadores + cosseno → uma frase de literatura + uma de achado. | ~210 | R3 (narrativa) |
| **C6** | §6 | Parágrafo da convenção de época → convenção + 0,23/0,93 + as inversões de veredito. | ~150 | — |
| **C7** | §8 | Reescrever a conclusão para ~220 palavras. | ~220 | R3 (narrativa) |
| **C8** | §5 | A cobertura da busca está dita em cinco sítios; manter §5 + rodapé, ponteiros nos outros. | ~120 | — |
| **C9** | §7 | Segundo limite. | ~110 | — |
| **C10** | §6 | Istanbul (§6.3) → uma frase dentro da §6.2. | ~100 | — |
| **C11** | §6 | Geometria + CTLE → dois números + delimitação. | ~120 | — |
| **C12** | §1 | Bullet 3 → duas linhas; cai a frase de roteiro. | ~115 | — |
| **C13** | §7 | Parágrafo do *trunk*. | ~75 | — |
| **C14** | §7 | *Shortlist* de serviço → 2 frases, mantendo 64,54 e 8 501. | ~80 | ⚠ **contraria R2(a)** — ver §3 |
| **C15** | §5 | Parágrafo TOST: cai a frase da margem de um ponto. | ~50 | — |
| **C16** | §6 | A frase *"For scale"*. | ~45 | ⚠ **contraria R3** — ver §3 |
| **C17** | §5 | Plano + Wilcoxon → ~50 palavras. ⚠ **A §3 lista isto como intocável e a §2 listava-o como corte — contradição minha.** O parágrafo tem de manter: teste registado → *t* pareado sobre quatro médias por semente → o piso de 0,0625 como razão → *"the code release includes both tests"*. O artigo **já teve uma errata** por descrever isto mal. | ~45 | — |
| | | **SUBTOTAL** | **~1 975** | |

**Faltam ~800 palavras**, e não saem de prosa. Saem dos flutuantes:
- **A Figura 4 duplica** os 12 deltas da Tabela II. Cortá-la liberta ~¼ de página. É o corte de maior
  rendimento e o de menor perda de informação.
- **A Figura 1** passa de `figure*` (largura dupla) a largura de coluna: ~⅓ de página.
- **A Tabela III** admite forma compacta.

---

## 3 · O que NÃO se corta — e dois cortes que contrariam revisores

**Intocável** (cada linha custa mais em credibilidade do que liberta em espaço):
Tabela II com as 24 células, os ± e as marcas · os seis intervalos de categoria e o limite de 0,34 ·
os quatro défices de região com a direcção · o núcleo do quinto limite (D4) · as frases *forward-only*
da §4 e o limite que nomeia o canal · a cobertura da busca **uma vez** · os números do treino com
todos os utilizadores com *"on an earlier build"* · o prior por fold e *"not a reproduction of the
complete published system"* · o plano estatístico e o desvio Wilcoxon→*t* · o par **P1** e a frase da
D4 colada · a recusa **P2** · as contagens de parâmetros · Istanbul.

⚠ **Dois cortes da lista contrariam directamente um revisor. Decisão do autor:**

- **C14 (shortlist de serviço).** O **R2(a)** critica precisamente que a motivação de *caching* é
  *"asserted but never connected to an actual system-level measurement"*. A v1 dava duas quantidades
  medidas ali; a reescrita já retirou uma. Cortar mais **agrava** o ponto dele. Recomendo **não
  cortar**, e recuperar as ~35 palavras da compacidade em vez disso.
- **C16 (a frase "For scale").** É o piso de classe maioritária, e é o que dá escala ao macro-F1 de
  30–38. O **R3** elogiou justamente *"The macro-averaged F1 for next category prediction shows good
  results"* — sem o piso, esse número fica sem referência. Recomendo **não cortar** (~45 palavras).

---

## 4 · O que não custa uma palavra e é o mais urgente

**O ramo público publica a geração retirada.** ✔ `origin/mobiwac`, HEAD `f9c50218`: 159 ficheiros
referem a geração antiga, **zero** a nova, e o README abre com *"Anonymous Code Release"* — numa
conferência de revisão simples, com o artigo aceite desde 2026-08-26.

O **R3** escreveu que um investigador **não conseguiria reproduzir** os resultados. A resposta do
artigo a essa frase é a nota de rodapé da página 1. **A promessa era verdadeira quando foi escrita e
tornou-se falsa por acção nossa**, ao corrigir os números sem republicar o pacote.

Passado à sessão `mobiwac-branch-refactor`. Fecha o ponto mais grave dos três pareceres com **zero**
palavras do orçamento — e é a condição para a contribuição de recurso da §1.2 existir.

---

## 5 · Sequência de execução

1. **A1–A4 primeiro** (+117). Acrescentar antes de cortar: é mais fácil cortar um texto completo do
   que lembrar-se de acrescentar a um texto já apertado.
2. **C1** (fundir §3 em §4) — muda a estrutura, e tudo o resto se mede depois dela.
3. **C2–C7**, por ordem. Medir páginas ao fim de cada um.
4. **Os flutuantes**, só se as ~800 palavras ainda faltarem. A Fig. 4 primeiro.
5. **C8–C17**, se ainda faltar.
6. **`GATE_PHASE=2 ./gate_v17.sh`** — exige 8 páginas e verifica as duas direcções de frase partida,
   que é a operação em que o corte mais falha.

**A regra que se manteve o dia todo:** um comentário vive **entre** frases, nunca a meio de uma.
Perdemos texto entregue **quatro vezes** hoje por a violarmos, e o corte é onde ela mais se viola.

---

## 6 · A MECÂNICA DE SUBMISSÃO — não estava no plano e tem de acontecer amanhã

Nenhum destes é prosa, e nenhum estava aqui. **Vários dependem de coisas que só o autor tem.**

| # | O quê | Quem |
|---|---|---|
| **M1** | **Confirmar o prazo.** A página do venue diz *"September 6th for All Submissions"*; o e-mail que o autor recebeu diz *"end of the day (extended), **September 7th AOE**"*. O e-mail é mais recente e diz "extended", portanto ganha — mas é o e-mail que manda, não este plano. | autor |
| **M2** | **O código de copyright do `\IEEEpubid`.** A página do venue não o dá; vem com as instruções da Sheridan. **Sem ele não se pode medir a página 1**, e o portão em fase 2 exige-o. | autor |
| **M3** | **O formulário electrónico de copyright (eCF)**, enviado ao autor de contacto depois do upload. Assinar **por último**. | autor |
| **M4** | **PDF eXpress**, Conference ID **71698X**. As fontes já passam (15 embebidas, zero Type 3), mas o PDF final deve passar lá. | writer |
| **M5** | **O upload vai para o link da Sheridan, não para o EDAS.** PDF **e** fonte LaTeX, juntos — *"No exceptions"*. | autor |
| **M6** | **O resumo registado no EDAS ainda é o rascunho antigo**, e o bloco de substituição do `EDAS_SUBMISSION.md` é da geração retirada (*"+28 to +40"*). Tem de ser substituído pelo resumo actual, e a ordem das keywords alinhada. | autor |
| **M7** | **Agradecimentos / financiamento.** A norma di-lo responsabilidade do autor de contacto. **Ninguém perguntou ao autor se CAPES / CNPq / FAPEMIG têm de aparecer.** Se sim, ~5 linhas, e contam para as páginas. | autor |
| **M8** | **O nome do meio do Germano ("Barcelos") não vem do EDAS** e continua por confirmar. Um nome legal errado no Xplore é permanente. | autor |

## 7 · Pontos dos revisores sem item, e custam quase nada

- **R2(a), correcção de zero palavras:** a §3 diz que o esboço de serviço *"quantifies"* algo, e a §7
  diz *"This remains motivation, not a measured service result"*. **`quantifies` → `states`.** É uma
  promessa quebrada que o R2 vai encontrar, e é exactamente a queixa dele.
- **R2(c)-i:** o controlo de congelamento que ele pediu para clarificar **desapareceu em silêncio**.
  Ou uma retracção declarada (+34 palavras), ou uma decisão explícita de o deixar em silêncio.
- **R2(c)-iii:** a cobertura de 67–87 % lê-se como se 13–33 % da avaliação ficasse por pontuar (+31).
- **R1-4 (eventos esporádicos):** a divisão é por **utilizador**, não temporal, e o artigo não o diz
  em lado nenhum. Decidir explicitamente omitir, em vez de omitir por esquecimento.
