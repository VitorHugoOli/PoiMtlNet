# Análise Crítica: Cálculo de Class Weights

## ✅ O que está CORRETO

### 1. Fórmula matemática
```python
weights = compute_class_weight('balanced', classes=cls, y=y_all)
# Fórmula: weight[i] = n_samples / (n_classes * n_samples_in_class[i])
```

**Verificação (dados reais):**
```
Class 0 (Community):     count=141,743 → weight=1.4077 ✅
Class 1 (Entertainment): count=112,614 → weight=1.7718 ✅
Class 2 (Food):          count=350,943 → weight=0.5686 ✅
Class 3 (Nightlife):     count= 62,248 → weight=3.2055 ✅
Class 4 (Outdoors):      count= 99,215 → weight=2.0111 ✅
Class 5 (Shopping):      count=299,917 → weight=0.6653 ✅
Class 6 (Travel):        count=330,061 → weight=0.6045 ✅
```

✅ **Pesos calculados corretamente!**

---

## ⚠️ PROBLEMAS ENCONTRADOS

### 1. ❌ Bug: Usando config errado (Category)
**Arquivo:** `src/train/category/cross_validation.py:37`

```python
# ERRADO ❌
cls = np.arange(CfgNextModel.NUM_CLASSES)  # ← Usando config do NEXT!

# CORRETO ✅
cls = np.arange(CfgCategoryModel.NUM_CLASSES)
```

**Impacto:**
- Funciona por acaso (ambos têm NUM_CLASSES=7)
- Mas é um **code smell** sério
- Se mudar NUM_CLASSES em um, quebra no outro

---

### 2. ⚠️ Potencial problema: DataLoader Iteration

**Código atual:**
```python
y_all = np.concatenate([y.numpy() for _, y in train_loader])
# Depois você usa train_loader novamente no treino
```

**Possíveis problemas:**

#### A) DataLoader consumido
- Se `train_loader` não tem `persistent_workers=False` ou você não o recria
- A iteração para pegar labels pode consumir o iterador
- Próximo uso pode falhar ou retornar dados errados

#### B) Ordem aleatória
- Se `shuffle=True`, cada iteração dá ordem diferente
- Não afeta os pesos (soma é a mesma)
- Mas é ineficiente: itera 2x o dataset

**Solução melhor:**
```python
# Opção 1: Iterar apenas 1 vez e cachear os dados
y_all = []
for _, y in train_loader:
    y_all.append(y.numpy())
y_all = np.concatenate(y_all)

# Opção 2: Se já tem os dados originais
y_all = dataset.targets  # Se dataset tiver .targets

# Opção 3: Usar um pequeno sample (se dataset muito grande)
# Pesos não mudam muito com sampling
```

---

### 3. ✅ Proteção contra classes ausentes

**Teste realizado:**
```python
# Se um fold não tiver uma classe (ex: Nightlife)
weights = compute_class_weight('balanced', classes=[0,1,2,3,4,5,6], y=y_without_class_3)
# Resultado: ValueError: classes should have valid labels that are in y
```

✅ **Bom!** sklearn falha rápido se classe está ausente
- Em K-fold, isso é raro mas pode acontecer
- Se acontecer, você vai saber imediatamente

---

### 4. ⚠️ Ordem das classes É IMPORTANTE

**Teste:**
```python
cls = np.arange(7)           # [0,1,2,3,4,5,6]
weights1 = compute_class_weight('balanced', classes=cls, y=y_all)
# [1.41, 1.77, 0.57, 3.21, 2.01, 0.67, 0.60]

cls_reverse = np.array([6,5,4,3,2,1,0])  # Ordem reversa
weights2 = compute_class_weight('balanced', classes=cls_reverse, y=y_all)
# [0.60, 0.67, 2.01, 3.21, 0.57, 1.77, 1.41]  ← ORDEM DIFERENTE!
```

**No seu código:**
```python
cls = np.arange(CfgNextModel.NUM_CLASSES)  # [0, 1, 2, 3, 4, 5, 6]
weights = compute_class_weight('balanced', classes=cls, y=y_all)
# weights[0] = peso da classe 0
# weights[1] = peso da classe 1
# ...
alpha = torch.tensor(weights, ...)
criterion = nn.CrossEntropyLoss(weight=alpha)
```

✅ **Correto!**
- `nn.CrossEntropyLoss` espera `weight[i]` = peso da classe `i`
- Você passa `cls = [0,1,2,3,4,5,6]` → ordem correta
- `alpha[0]` corresponde à classe 0 ✅

---

## 🔧 Recomendações

### Fix 1: Corrigir config (CRÍTICO)
```python
# src/train/category/cross_validation.py:37
cls = np.arange(CfgCategoryModel.NUM_CLASSES)  # Não CfgNextModel!
```

### Fix 2: Evitar iteração dupla do DataLoader
```python
# Opção A: Se dataset tem .targets
if hasattr(train_loader.dataset, 'targets'):
    y_all = np.array(train_loader.dataset.targets)

# Opção B: Se não, extrair antes
else:
    y_all = np.concatenate([y.numpy() for _, y in train_loader])
```

### Fix 3: Adicionar validação
```python
weights = compute_class_weight('balanced', classes=cls, y=y_all)
alpha = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

# Validação
assert len(alpha) == CfgCategoryModel.NUM_CLASSES, \
    f"Weight mismatch: got {len(alpha)}, expected {CfgCategoryModel.NUM_CLASSES}"
assert torch.all(alpha > 0), "All weights must be positive"

print(f"Class weights: {alpha.cpu().numpy()}")  # Log para debug
```

---

## 📊 Análise de Impacto

### Com os pesos atuais (assumindo correção do bug MTL):

**Nightlife (classe minoritária):**
- Sem weight: contribuição na loss = 4.46% (proporcional aos dados)
- Com weight (3.21): contribuição efetiva ≈ 14.3% ✅
- **3.2x mais importante que sem weight!**

**Food (classe majoritária):**
- Sem weight: contribuição na loss = 25.13%
- Com weight (0.57): contribuição efetiva ≈ 14.3% ✅
- **0.57x menos importante!**

**Resultado esperado:**
- F1 macro avg **aumenta** (classes balanceadas)
- Accuracy geral pode diminuir levemente (menos foco em majoritárias)
- Performance em Nightlife/Outdoors **melhora significativamente**

---

## ✅ Resumo Final

### Está CORRETO:
1. ✅ Fórmula de `compute_class_weight` está correta
2. ✅ Ordem das classes está correta (`cls = [0,1,2,3,4,5,6]`)
3. ✅ Conversão para tensor PyTorch está correta
4. ✅ Proteção contra classes ausentes (sklearn dá erro)

### Precisa CORRIGIR:
1. ❌ **Bug crítico:** Usar `CfgCategoryModel.NUM_CLASSES` em vez de `CfgNextModel` (category)
2. ❌ **Bug crítico:** Aplicar `weight=alpha_cat` em MTL category criterion (linha 341)
3. ⚠️ **Otimização:** Evitar iterar train_loader duas vezes (performance)
4. ⚠️ **Boas práticas:** Adicionar validação e logging dos weights

---

## 🎯 Ação Imediata

**Prioridade 1 (CRÍTICO):**
```python
# src/train/mtlnet/mtl_train.py:341
category_criterion = CrossEntropyLoss(reduction='mean', weight=alpha_cat)  # ← ADD
```

**Prioridade 2 (Code Quality):**
```python
# src/train/category/cross_validation.py:37
cls = np.arange(CfgCategoryModel.NUM_CLASSES)  # ← FIX
```

**Prioridade 3 (Performance):**
- Evitar iterar DataLoader 2x (caching ou usar dataset.targets)

**Prioridade 4 (Debug):**
- Adicionar logging dos weights calculados
- Adicionar assertions de validação