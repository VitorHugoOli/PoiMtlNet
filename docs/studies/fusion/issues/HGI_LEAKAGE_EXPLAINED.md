# HGI Leakage Audit — Glossário Explicativo

**Data:** 2026-04-15
**Escopo:** Explicações em linguagem acessível dos conceitos-chave do `HGI_LEAKAGE_AUDIT.md`.
**Quando usar:** Leia este documento **antes** de mergulhar no audit técnico se você é novo no pipeline HGI, ou quando precisar explicar o achado para um revisor/coautor.

---

## 1. O que é OSM (OpenStreetMap)

**OSM = OpenStreetMap** — uma base de dados geográfica aberta e colaborativa, tipo "Wikipedia de mapas". Qualquer um pode cadastrar ruas, prédios, lojas, parques, etc.

Quando alguém cadastra um POI (ponto de interesse) no OSM, preenche **tags** dizendo o que aquilo é — ex.: `amenity=restaurant`, `cuisine=pizza`, `shop=bakery`. O dataset Gowalla herda esse esquema de tags.

No nosso código, essas tags viram duas colunas:

- **`fclass`** (sub-tipo fino, herdado do OSM) — ex.: "Burgers", "Motel", "Church", "Bakery". Cada estado tem ~300 valores distintos.
- **`category`** (classe grossa, agregação manual em 7 grupos) — Food, Travel, Community, Entertainment, Nightlife, Outdoors, Shopping.

**Relação crítica:** cada `fclass` sempre pertence a **uma única** `category`. "Burgers" é sempre Food, "Motel" é sempre Travel, "Church" é sempre Community. Nunca se cruzam. Por isso `fclass → category` é **função determinística** (pureza = 1.0 em todos os 6 estados avaliados — Alabama, Arizona, California, Florida, Georgia, Texas).

Essa determinismo é uma **propriedade da taxonomia do dataset**, não um bug nosso.

---

## 2. POI2Vec no nível de fclass — por que isso importa

O pipeline HGI tem dois estágios de embedding:

### Estágio 1 — POI2Vec (Fase 3 do pipeline)

Gera embeddings **no nível do fclass**, não no nível do POI. Ou seja: existem ~300 vetores (um por fclass), não ~12 mil (um por POI).

```python
# poi2vec.py:438-444
poi_embedding[i] = fclass_embeddings[poi.fclass[i]]
```

**Consequência:** todo POI com o mesmo fclass recebe o **vetor idêntico**. O "McDonald's do centro" e o "McDonald's da zona sul" — ambos com fclass=Burgers — têm o mesmo embedding POI2Vec. Eles só se diferenciam depois, no estágio seguinte.

### Estágio 2 — HGI (Fase 5 do pipeline)

Recebe os vetores do POI2Vec como *features de entrada* e roda uma rede neural de grafos (GNN) sobre o grafo Delaunay dos POIs, misturando estrutura espacial (vizinhança, região, cidade) por cima do sinal de fclass.

**Consequência:** o embedding final do HGI **não é mais** puramente fclass — é fclass + estrutura de grafo. Dois McDonald's agora recebem vetores ligeiramente diferentes (porque seus vizinhos são diferentes), mas o "esqueleto" do vetor ainda vem do fclass.

### Por que isso importa para a auditoria

Como o fclass domina o sinal de entrada **e** fclass→category é determinístico, qualquer modelo que aprenda a distinguir fclasses no espaço vetorial **automaticamente** aprende a distinguir categories — sem nunca ter visto o rótulo de category durante o treino. O atalho está na estrutura dos dados, não no código.

---

## 3. O experimento do braço C, passo a passo

**Pergunta que o braço C responde:** "Se eu destruir a relação POI ↔ fclass, quanto o modelo ainda consegue? Se cair para chute aleatório, significa que o modelo estava 100% apoiado nessa relação."

### Protocolo

1. Pega a lista de POIs da Alabama. Cada POI tem: coordenadas, `fclass`, `category`.
2. **Embaralha a coluna `fclass`** com seed fixa (`shuffle_fclass_seed=42`). Agora o POI "McDonald's do centro" (originalmente fclass=Burgers) recebe aleatoriamente fclass=Church. O "McDonald's da zona sul" recebe fclass=Lake. É uma permutação — o conjunto de fclasses do dataset continua o mesmo, mas quem tem qual fclass foi embaralhado.
3. A coluna `category` fica **intacta** (o McDonald's continua marcado como Food no rótulo de treino).
4. Reroda tudo do zero: POI2Vec → HGI → gera inputs MTL → treina o MTL (1 fold, seed 42, config campeã DSelectK + aligned_mtl).
5. Mede Category F1 e Next-POI F1.

### Resultado

| métrica | baseline | braço C | Δ |
|---|---|---|---|
| Category F1 | 0.7855 | **0.1437** | **−64.19 p.p.** |
| Category accuracy | 0.8250 | 0.2623 | — |
| Next-POI F1 | 0.2383 | 0.1988 | −3.95 p.p. |

O chute aleatório para 7 classes (macro-F1) vale 1/7 ≈ **0.1429**. O modelo caiu exatamente nesse piso. A accuracy de 0.262 bate com a taxa da classe majoritária (Food ~32%), ou seja, o modelo virou um chute na classe mais comum.

### Interpretação

- **Category perdeu tudo.** Sem o fclass correto, o modelo vira random. Prova que Category estava 100% apoiada no atalho "qual fclass esse POI tem → tabela fclass→category → devolve category".
- **Next-POI mal se abalou.** Queda de apenas 3,95 p.p. mostra que essa tarefa **nunca** esteve apoiada no atalho do fclass. Ela usa outro sinal (padrões de mobilidade, sequência temporal, estrutura do grafo).

### Analogia das frutas

Imagine que você treinou um modelo para classificar frutas como "doce" ou "azeda", e como *feature de entrada* você deu o **nome da fruta** ("maçã", "limão", "morango", "uva"). O modelo acerta 99%.

Agora você embaralha os nomes — o que antes se chamava "maçã" agora se chama "limão", e vice-versa, aleatoriamente. O modelo cai para 50% (chute).

Isso **não prova** que o modelo era bom em identificar doçura. Prova que ele **apenas memorizou** um dicionário nome→sabor. A "features" carregavam a resposta de graça.

O braço C é exatamente isso: `fclass` é o "nome da fruta", `category` é "doce/azeda". Embaralha o nome, o modelo morre. Conclusão: nunca aprendeu nada além de consultar o dicionário.

---

## 4. O "path cosmético do `le_lambda`"

Dentro do POI2Vec existem **dois termos de perda** somados (`poi2vec.py:178`):

```python
return loss_graph + loss_hierarchy
```

### Termo 1 — `loss_graph` (skip-gram)

A parte que **realmente treina** os embeddings de fclass. Usa caminhadas (walks) no grafo Delaunay para puxar fclasses que co-ocorrem espacialmente para perto no espaço vetorial. É o objetivo principal, com gradientes de magnitude normal (~10⁰ a 10¹).

### Termo 2 — `loss_hierarchy` (o tal path do `le_lambda`)

Uma regularização que diz *"o vetor do fclass deve ficar perto do vetor da category dele"*. Concretamente (`poi2vec.py:164-174`):

```python
cat_embs    = self.in_embed(cat_idx)       # pega o vetor da category
fclass_embs = self.in_embed(fclass_idx)    # pega o vetor do fclass
diff = cat_embs - fclass_embs              # distância entre eles
loss_hierarchy = 0.5 * self.le_lambda * (diff * diff).sum()
```

**Em palavras:** para cada par `(category, fclass)` existente nos dados (ex.: `(Food, Burgers)`, `(Food, Pizza)`, `(Travel, Motel)`...), soma a distância ao quadrado entre os dois vetores, multiplica por `le_lambda` e joga na perda total. O otimizador, ao minimizar, puxa o vetor de "Burgers" para perto do vetor de "Food" — o que em tese **plantaria explicitamente informação de category dentro do embedding do fclass**.

### Por que é "cosmético"

O peso `le_lambda = 1e-8`. A outra perda (skip-gram) tem magnitude ~10⁰ a 10¹. Ou seja, o termo de hierarquia é **8 ordens de grandeza menor** — o gradiente dele é numericamente desprezível.

Analogia: é como temperar um caminhão de comida com um único grão de sal. Existe no código, mas não afeta o resultado.

### Prova empírica — o braço A

O braço A zerou `le_lambda` completamente (eliminando esse termo) e mediu o impacto:

| arm | Category F1 | Δ |
|---|---|---|
| baseline | 0.7855 | +0.00 |
| A (le_lambda=0) | 0.7992 | **+1.36 p.p.** |

Se fosse um canal de leakage real, remover faria o F1 **cair**. Como ele **subiu** 1,36 p.p., a conclusão é: esse path estava adicionando ruído mínimo, não vazando sinal útil de category para o embedding.

### Por que o código tem isso então?

É um resquício da formulação teórica do paper POI2Vec original (que inclui uma hierarquia `category ↔ sub-category`). Alguém manteve o termo mas com peso tão baixo que virou inerte. Pode ser removido sem efeito observável.

---

## 5. Por que isso NÃO é "data leakage clássico"

Data leakage clássico = rótulo do conjunto de validação escapando para o conjunto de treino. Aqui **nenhum rótulo vaza**:

- `data.y` (a coluna de category) nunca é lida durante o treino do HGI ou POI2Vec. Verificado por grep (§5 do audit).
- Não existe cabeça de classificação de category no HGI. O objetivo é puramente auto-supervisionado (contrastivo POI↔Região↔Cidade).
- Seleção do melhor modelo usa apenas a perda de treino — não métrica de validação.
- Os poucos pontos onde `category` aparece no código (hierarchy loss, decoração de DataFrames) foram auditados e são inócuos.

O que existe é um **atalho taxonômico no dataset**: `fclass` é um sub-tipo de `category`, então quem aprende fclass aprende category implicitamente. A informação do target já está estruturalmente codificada na *entrada*, não por bug, mas por propriedade do OSM.

**Impacto no paper é o mesmo que leakage clássico teria** — muda qual métrica ancora as afirmações — mas a natureza do problema é diferente. Dizer "há leakage no HGI" seria factualmente incorreto.

---

## 6. Outras perguntas frequentes

### "O embedding final do HGI é só o fclass?"
Não. É **fclass + estrutura de grafo**. O POI2Vec gera o vetor base (função do fclass), o HGI adiciona sinal espacial por cima via GNN. Mas o sinal do fclass domina a ponto de, quando você o remove (braço C), o Category F1 desabar — ou seja, a parte espacial sozinha carrega **essencialmente zero sinal discriminativo de category**.

### "Por que Next-POI sobrevive ao shuffle?"
Porque Next-POI prevê a category do **próximo** lugar na sequência de visitas do usuário — informação que não está no embedding do POI atual. O modelo precisa usar padrões de mobilidade (quem visita X tende a visitar Y depois), estrutura temporal e relações entre POIs no grafo. Esses sinais não dependem do mapeamento fclass→category do POI atual.

### "Isso invalida os resultados anteriores?"
Não. Os números de Category F1 continuam corretos e reproduzíveis. O que muda é a **interpretação**: Category F1 deixa de ser evidência de "o embedding aprendeu estrutura espacial/semântica" e passa a ser "o embedding preserva fielmente a identidade do fclass num espaço de 64 dimensões". Ainda é uma métrica útil, só mede outra coisa.

### "E o Check2HGI?"
Check2HGI concatena `category` one-hot diretamente nas features dos nós (`research/embeddings/check2hgi/preprocess.py:217-219`). Isso **seria** leakage clássico — o rótulo vira input. Por isso o Check2HGI precisa ser corrigido antes de entrar em qualquer fusão/experimento. Fora do escopo do audit do HGI, mas documentado em `CHECK2HGI_ENRICHMENT_PROPOSAL.md`.

### "E o DGI?"
Não auditado neste documento. DGI usa o mesmo POI2Vec como pre-step, então herda a mesma propriedade fclass-level. Deve ser auditado antes de qualquer conclusão DGI-específica.

### "O que é treino transdutivo? Isso é problema?"
Transdutivo = os nós do conjunto de validação **estão presentes no grafo** durante o treino (o HGI vê a *estrutura* dos POIs de validação, mas não os *rótulos* deles). É uma prática padrão em GNNs, geralmente defensável como caveat metodológico. Não foi testado diretamente na auditoria, mas considera-se efeito leve comparado ao atalho do fclass.

---

## 7. Implicações para o paper — versão resumida

1. **Next-POI F1** é a única métrica defensável de "qualidade de representação aprendida".
2. **Category F1** vira sanity check de fidelidade do embedding; comparações cross-engine em Category F1 devem ser renomeadas para "preservação de fclass".
3. **Parágrafo de caveat é obrigatório** na seção de avaliação: (i) determinismo fclass→category, (ii) resultado do braço C, (iii) reenquadramento da Category F1.
4. **Próximo passo:** replicar braço C em Florida antes da submissão BRACIS (~15 min) para confirmar que a propriedade vale lá também.

Ver §9 do `HGI_LEAKAGE_AUDIT.md` para o plano de ação completo e §8 para os números exatos de cada braço.

---

## 8. Referências internas

- `docs/issues/HGI_LEAKAGE_AUDIT.md` — documento técnico completo (código, linhas, severidade por canal).
- `docs/studies/results/P0/leakage_ablation/alabama/` — artefatos (JSONs por braço, README de resultados).
- `scripts/hgi_leakage_ablation.py` — driver do experimento dos 5 braços.
- `docs/issues/CHECK2HGI_ENRICHMENT_PROPOSAL.md` — landmine separada do Check2HGI.
- `docs/studies/CLAIMS_AND_HYPOTHESES.md` — quando C29 for incorporado, será a claim formal de refutação.
