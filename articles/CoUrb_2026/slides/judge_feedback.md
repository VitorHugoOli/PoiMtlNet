# Revisão crítica do deck — ST-MTLNet @ CoUrb 2026

## 1. Veredito em uma frase

**O deck não mente, mas faz cherry-picking de "best-of-two" sem deixar claro, vende a faixa "+20–24 pp" como se cada variante alcançasse isso (uma delas não chega) e repete o "16/21" do paper apesar de a contagem estrita das tabelas dar 15/21.**

## 2. Problemas científicos / inconsistências numéricas

Recontando as próprias tabelas `tabela_comparativa_f1_*.tex`:

- **Categoria — ganho médio por estado (best-of-SIREN/Sphere por linha):** FL +20,24 pp · CA +20,91 pp · TX +21,98 pp. O intervalo "20–24 pp" só fecha porque pega o melhor encoder em cada célula. Olhando por **variante isolada**, SIREN no Texas rende **+17,89 pp** em média — fora do intervalo declarado. Isso precisa estar no rodapé (ex.: "20–24 pp considerando o melhor dos dois encoders por linha") ou cair para "≈ +20 pp em média".
- **Next-POI — "16/21" / "76 %":** contagem estrita por média (best-of-two > baseline) dá **15/21 = 71,4 %**. O caso ambíguo é **Florida Outdoors**: baseline 21,61 vs Sphere 21,59 — baseline ainda vence por 0,02 pp (dentro de σ ≈ 1–2 pp). O paper e o slide tratam isso como vitória. Há duas opções honestas: (i) reclassificar como "empate técnico" e dizer "15 vitórias, 1 empate, 5 derrotas"; (ii) declarar explicitamente o critério (ex.: "vitória ou empate dentro de 1 σ"). Como está, o "76 %" é numericamente impreciso.
- **"Nightlife salta de ~30 % para >60 %"** (slide 7): no Texas e na Flórida ✓; na Califórnia o baseline é 26,8 %, não 30 %. Arredondamento generoso mas defensável; manter "de ~30 % a >60 %" é OK, mas evitar dizer "triplicamos" se alguém pedir precisão.
- **Travel é tratado de forma contraditória entre tarefas e o slide pode confundir:** Travel **vence** na categoria (+19 pp) e **perde** no next-POI (FL 64,47 → 45,00, CA 46 → 37). Os dois slides 7 e 8 usam Travel como exemplo com sinais opostos — é cientificamente correto, mas o público pode sair confuso. Use rótulos claros: "Travel (categoria) ✓" vs "Travel (next-POI) ✗".

## 3. Riscos de comunicação

- **Slide 4 está pesado:** figura + 5 bullets + frase de controle + menção a FiLM/ResBlocks em 1:30 é apertado. A própria figura `arquitetura_modelo.png` é densa.
- **Slide 5 (encoders espaciais)** é puro "background". Em 1 min você gasta 11 % do orçamento explicando algo que não é resultado. Considere fundir com slide 4 ou compactar para 30 s e abrir backup para Q&A.
- **Loss contrastiva binária** aparece no slide 5 ("par+ ≤10 km, par− ≥70 km") sem explicar por que esses thresholds — é jargão sem motivação. Cortar ou justificar em uma linha.
- **Slide 9** tenta apertar insights + limitações + 3 trabalhos futuros em 50 s — quase certamente vai estourar.
- **Slide 7** promete *"este é o resultado que eu quero que vocês levem"* e centra em "+20–24 pp / 21 de 21". Se um revisor olhar a tabela, vê SIREN Texas com 17,9 pp médio. **Coloque a janela "20–24" como faixa do *melhor* encoder por estado**, ou use uma única média global "≈ +21 pp".
- **Slide 10:** "76 % de vitórias em next-POI" deve virar "**~72 %** de vitórias" se a contagem estrita for adotada, ou "16/21 (com Outdoors-FL no limiar)" com transparência.

## 4. Problemas de tempo

Orçamento somado é **9:10** com **20 s de folga**. Pontos críticos:

- Slide 4 (1:30) é otimista para a figura densa + 5 itens.
- Slide 7 + 8 = 2:30 combinados; se houver pergunta intercalada não-prevista (raro mas acontece em workshop pequeno), estoura. O slot total do CoUrb costuma incluir transição de speaker → realístico é **~8 min de fala**, não 9. Ensaie para **8:30** e mantenha o último minuto como margem.

## 5. Limitações ausentes do deck

O slide 9 cobre **Travel/long-range** e **falta de ablação por encoder**. Faltam:

- **Escopo restrito:** apenas 3 estados dos EUA (FL/CA/TX), apenas 7 categorias (alto nível) — generalização para outras regiões/granularidades não foi avaliada.
- **Dataset Gowalla está envelhecido** (check-ins 2009–2010). Para uma plateia de computação urbana isso é uma fragilidade real e provável ponto de pergunta.
- **Custo computacional:** 3 encoders treinados separadamente + entrada de 1 728 dim no next-POI (vs 576 do baseline) — não há discussão sobre custo de treino/inferência. Para CoUrb (aplicações urbanas) isso importa.
- **Risco de leakage da divisão temporal:** o paper diz "5-fold estratificado 80/20" — mas em dados sequenciais com janela não-sobreposta, splits aleatórios sobre check-ins podem misturar trajetórias do mesmo usuário entre train/val. Não é claro se o split é por usuário ou por janela. Se for por janela, há vazamento conhecido.
- **Falta de baselines externos** — só compara com MTLNet original. Não há comparação com LSTPM, GETNext, STAN, HMT-GRN citados nos relacionados. Plateia adversarial vai notar.
- **NashMTL é referenciado como funcionando** — o `MEMORY.md` do projeto registra histórico de bug `cvxpy/ECOS` em que NashMTL colapsava para [1,1]; o paper assume o solver. Verificar se a versão dos experimentos do paper de fato usa Nash funcional (não é seu problema diretamente, mas é uma cautela).

## 6. Três perguntas adversariais que o deck não está pronto para responder

1. **"Por que comparar só com MTLNet 2025? Sua arquitetura tem +50 % mais parâmetros de entrada (192 vs 64); como sabemos que o ganho não é só capacidade?"** — o slide 4 menciona que MTLNet interno é igual, mas **não há controle de parâmetros totais nem ablação com 3×64 d aleatórios/zerados**. Falta resposta pronta.
2. **"O split é por usuário ou por janela? Como vocês evitam que um mesmo usuário apareça em treino e validação?"** — Nenhum slide aborda isso. Resposta deve estar pronta com 1 frase.
3. **"O encoder espacial 'melhor' depende do estado — isso não é hyperparameter tuning disfarçado? Como vocês escolhem em produção sem ver o teste?"** — slide 9 toca no tema mas como *insight*, não como vulnerabilidade. Plateia adversarial pode chamar de overfitting de seleção.

Bônus de quarta pergunta provável: **"Por que não fundir os três encoders por atenção em vez de concat?"** — o slide 9 lista isso como future work, mas não há justificativa para começar por concat.

## 7. Correções obrigatórias priorizadas

**P0 (antes da apresentação — bloqueantes):**
- Reescrever headline do slide 7 como `"+20 a 22 pp F1 (média por estado, melhor encoder)"` ou `"≈ +21 pp em média"` — não usar "20–24" sem qualificador.
- Reconciliar "16/21 / 76 %" com a contagem real: ou "**15/21 vitórias estritas + 1 empate técnico**", ou explicitar critério de empate dentro de σ. Aplicar nos slides 8 e 10.
- Adicionar 1 linha no slide 6 (Setup) declarando **como o split é feito** (por usuário? por janela? estratificado em quê?). Sem isso, qualquer revisor de mobilidade derruba o trabalho em Q&A.

**P1 (fortemente recomendado):**
- Slide 9 deve incluir bullets sobre: (a) ausência de baselines externos modernos, (b) Gowalla envelhecido, (c) escopo de 3 estados. Honestidade adicional reduz risco em Q&A.
- Slide 4 — cortar 1 bullet (mover "MTLNet interno inalterado" para nota de rodapé) para não estourar 1:30.
- Preparar slide-backup com **ablação plausível** (mesmo que retórica): "se zerássemos um encoder, o que esperaríamos perder?".

**P2 (polimento):**
- Slide 5: reduzir ou fundir; é o slide com menor densidade informativa por segundo.
- Trocar "Praia do Forte, BA" da capa por "CoUrb 2026 / SBRC 2026" se o local exato não estiver confirmado.
- Verificar se a citação "Silva et al. 2025" do MTLNet está atualizada — é trabalho do próprio laboratório.

## 8. Nota final

**6,0 / 10.**

Justificativa: a narrativa central é sólida (decomposição modular vs monolito) e o ganho em categoria é real e robusto. Mas há **três fragilidades concretas** que um revisor crítico pega em segundos: (i) a faixa "+20–24 pp" depende de escolha best-of-two por linha não declarada; (ii) o "16/21 / 76 %" não bate com a contagem estrita da própria tabela; (iii) limitações materiais (split, baselines externos, escopo de 3 estados, dataset envelhecido) estão ausentes do deck. Em workshop pequeno talvez passe; em revisão mais rigorosa, esses três pontos derrubam a confiança. **Corrigir os P0 sobe a nota para 7,5–8.**
