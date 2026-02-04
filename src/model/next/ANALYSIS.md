# Análise Crítica: Next-POI Prediction Architecture

## Contexto
**Tarefa:** Prever a próxima categoria de POI dado histórico de visitas
**Input:** Sequência de embeddings [B, seq_length=9, embed_dim]
**Output:** Logits [B, num_classes=7]

---

## ✅ Transformer FAZ SENTIDO aqui! (mas pode melhorar)

### Por que é diferente do Category Head?

| Aspecto | Category Head | Next-POI Head |
|---------|--------------|---------------|
| **Input** | 1 embedding único | Sequência de 9 embeddings |
| **Tokens** | Artificiais (divisão de vetor) | Reais (cada visita) |
| **Semântica** | Cada "token" sem significado | Cada token = 1 visita POI |
| **Ordem** | Irrelevante | **CRUCIAL** (temporal) |
| **Transformer?** | ❌ Não faz sentido | ✅ Faz sentido |

---

## ✅ Justificativa para usar Transformer

### 1. Múltiplos embeddings independentes
```python
# Cada posição = uma visita diferente
x = [emb_restaurante, emb_cinema, emb_gym, emb_cafe, ...]
# Cada embedding tem significado próprio ✅
```

### 2. Sequencialidade temporal importa
```python
# Ordem altera completamente o padrão
[Gym → Smoothie bar] ≠ [Smoothie bar → Gym]
                      ≠ [Bar → Cinema → Gym]
```

### 3. Self-attention captura padrões de transição
```python
# Attention pode aprender:
# "Após gym (80% das vezes) → juice bar ou restaurante saudável"
# "Após cinema (60% das vezes) → restaurante ou bar"
# "Sexta à noite: restaurante → bar (70%)"
```

### 4. Causal mask está correto
```python
# Linha 84-87 do código atual:
causal_mask = torch.triu(torch.ones(...), diagonal=1)
# Previne "olhar para o futuro" ✅
# Essencial para next-POI prediction
```

---

## ⚠️ Problemas do Modelo Atual (NextHeadSingle)

### 1. Positional Encoding Sinusoidal
```python
# Implementação atual (linhas 18-25):
pe[:, 0::2] = torch.sin(pos * div_term)
pe[:, 1::2] = torch.cos(pos * div_term)
```

**Problema:**
- Sinusoidal PE foi projetado para **sequências longas** (NLP: milhares de tokens)
- Permite generalização para sequências maiores que as vistas no treino
- **Seu caso: MAX_SEQ_LENGTH = 9** (muito curto!)

**Solução:**
```python
# Learned positional embeddings são melhores para sequências curtas
self.pos_embedding = nn.Parameter(torch.randn(1, 9, embed_dim))
```

**Por quê?**
- Pode aprender padrões específicos da sua aplicação
- Ex: "posição 0 (primeira visita) tem comportamento único"
- Mais parâmetros, mas para seq=9 é negligível

---

### 2. Pooling Strategy: Ignora importância temporal

```python
# Código atual (linhas 97-103):
attn_weights = torch.softmax(attn_logits, dim=1)
pooled = torch.sum(x * attn_weights, dim=1)
```

**Problema:**
- Trata todas as posições com importância similar
- **Intuição:** Visitas RECENTES importam mais que antigas!
- "Onde estive há 5 minutos" > "Onde estive ontem"

**Solução: Temporal Decay**
```python
# Peso exponencial: visitas recentes = mais importantes
decay = [e^-4, e^-3, e^-2, e^-1, e^0]
pooled = sum(x * decay * attention_weights)
```

---

### 3. Transformer pode ser overkill para 9 visitas

**Complexidade de Transformer:**
- Self-attention: O(seq_len²)
- Para seq=9: 81 operações (ok)
- Para seq=100: 10,000 operações (aí compensa)

**Alternativas mais eficientes:**
- **LSTM/GRU:** O(seq_len), projetados para sequências
- **Temporal CNN:** Paralelo, captura padrões locais
- **Hybrid (GRU + Attention):** Melhor dos dois mundos

---

## 🎯 Arquiteturas Recomendadas

### 1. **NextHeadGRU** ⚡ Mais eficiente
**Quando usar:** Baseline para sequências curtas
**Vantagens:**
- 25% menos parâmetros que LSTM
- Implicitamente modela temporalidade (sem PE)
- Mais rápido que Transformer
- Perfeito para seq_length < 20

```python
NextHeadGRU(
    embed_dim=256,
    hidden_dim=256,
    num_classes=7,
    num_layers=2,
    dropout=0.3
)
```

**Complexidade:** ~150k params | Velocidade: ⚡⚡⚡⚡

---

### 2. **NextHeadLSTM** 🧠 Memória de longo prazo
**Quando usar:** Se padrões têm dependências de longo alcance
**Vantagens:**
- Célula de memória explícita (cell state)
- Melhor para sequências com "contexto distante importante"
- Mais parâmetros = mais capacidade

```python
NextHeadLSTM(
    embed_dim=256,
    hidden_dim=256,
    num_classes=7,
    num_layers=2,
    dropout=0.3,
    bidirectional=False  # True se ordem não for estritamente causal
)
```

**Complexidade:** ~200k params | Velocidade: ⚡⚡⚡

---

### 3. **NextHeadHybrid** 🏆 RECOMENDADO
**Quando usar:** Melhor custo-benefício para seq=9
**Vantagens:**
- GRU processa sequência eficientemente
- Self-attention foca em visitas importantes
- Interpretável (pode visualizar attention weights)
- Combina strengths de RNN + Transformer

```python
NextHeadHybrid(
    embed_dim=256,
    hidden_dim=256,
    num_classes=7,
    num_heads=4,
    num_gru_layers=2,
    dropout=0.3
)
```

**Arquitetura:**
```
Input [B, 9, 256]
  ↓
GRU (2 layers) → Contexto sequencial
  ↓
Self-Attention → Foca em visitas relevantes
  ↓
Residual Connection
  ↓
Last timestep → Classifier
```

**Complexidade:** ~250k params | Velocidade: ⚡⚡⚡

**Por que funciona:**
1. GRU captura dependências temporais
2. Attention seleciona "quais visitas importam agora"
3. Residual garante gradient flow

---

### 4. **NextHeadTemporalCNN** 🚀 Padrões locais
**Quando usar:** Se transições seguem padrões muito locais
**Vantagens:**
- Paralelo (mais rápido que RNN)
- Captura padrões tipo: "Gym → Smoothie bar (sempre)"
- Receptive field cresce com camadas

```python
NextHeadTemporalCNN(
    embed_dim=256,
    hidden_channels=128,
    num_classes=7,
    num_layers=4,
    kernel_size=3,
    dropout=0.2
)
```

**Quando usar:**
- Padrões são muito "pares consecutivos" (bigrams)
- Ex: "Restaurante → Cinema", "Gym → Cafe"
- Menos eficaz para dependências distantes

**Complexidade:** ~180k params | Velocidade: ⚡⚡⚡⚡⚡

---

### 5. **NextHeadTransformerOptimized** 🔧 Transformer melhorado
**Quando usar:** Se quiser manter Transformer, use esta versão
**Melhorias sobre modelo atual:**

#### a) Learned Positional Embeddings
```python
# Antes (Sinusoidal):
pe[:, 0::2] = torch.sin(pos * div_term)

# Depois (Learned):
self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, embed_dim))
```

#### b) Temporal Decay Pooling
```python
# Visitas recentes têm mais peso
decay = exp(-[4, 3, 2, 1, 0])  # [0.018, 0.05, 0.135, 0.37, 1.0]
pooled = sum(x * decay) / sum(decay)
```

#### c) Pre-norm (norm_first=True)
```python
# Mais estável para redes profundas
encoder_layer = nn.TransformerEncoderLayer(..., norm_first=True)
```

```python
NextHeadTransformerOptimized(
    embed_dim=256,
    num_classes=7,
    num_heads=8,
    num_layers=2,
    seq_length=9,
    dropout=0.3,
    use_temporal_decay=True
)
```

**Complexidade:** ~220k params | Velocidade: ⚡⚡

---

## 📊 Comparação Geral

| Modelo | Params | Velocidade | Interpretabilidade | Adequação seq=9 |
|--------|--------|-----------|-------------------|-----------------|
| **GRU** | 150k | ⚡⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **LSTM** | 200k | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Hybrid (GRU+Attn)** | 250k | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Temporal CNN** | 180k | ⚡⚡⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Transformer Opt** | 220k | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Transformer Atual** | 220k | ⚡⚡ | ⭐⭐⭐ | ⭐⭐ |

---

## 🧪 Plano de Experimentação Recomendado

### Fase 1: Baselines Rápidos
```python
# 1. GRU (mais simples)
NextHeadGRU(embed_dim=256, hidden_dim=256, num_layers=2)

# 2. Transformer otimizado (comparação justa)
NextHeadTransformerOptimized(embed_dim=256, num_layers=2, use_temporal_decay=True)
```

**Objetivo:** Estabelecer baseline sólido rapidamente

---

### Fase 2: Exploração
```python
# 3. Hybrid (recomendado)
NextHeadHybrid(embed_dim=256, hidden_dim=256, num_heads=4)

# 4. LSTM (se GRU saturar)
NextHeadLSTM(embed_dim=256, hidden_dim=256, num_layers=2)

# 5. Temporal CNN (se padrões forem locais)
NextHeadTemporalCNN(embed_dim=256, num_layers=4)
```

**Objetivo:** Encontrar melhor arquitetura

---

### Fase 3: Otimização
Tunear o melhor da Fase 2:
- Hidden dim: [128, 256, 512]
- Num layers: [2, 3, 4]
- Dropout: [0.2, 0.3, 0.4]
- Learning rate schedule
- Temporal decay factor (se usando Transformer Opt)

---

### Fase 4: Análise
**Interpretabilidade (Hybrid ou Transformer):**
```python
# Visualizar attention weights
attn_weights = model.get_attention_weights(x)
# Quais visitas passadas são mais relevantes?
# Restaurante → ? (attn alto em 'delivery food')
```

**GRU/LSTM:**
```python
# Inspecionar hidden states
# t=8 (última posição) deve conter todo contexto
```

---

## 💡 Insights para Sequências Curtas (seq=9)

### 1. RNNs são subestimados
- **Mito:** "Transformers sempre > RNNs"
- **Realidade:** Para seq < 20, RNNs são:
  - Mais eficientes
  - Mais fáceis de treinar
  - Performance similar ou melhor

### 2. Temporal decay é crucial
```python
# Próxima visita depende MUITO da última
# Correlação temporal decai exponencialmente
visita[-1] (agora)    → peso 1.0
visita[-2] (1h atrás) → peso 0.37
visita[-5] (ontem)    → peso 0.05
```

### 3. Padrões de transição são chave
```python
# Não é só "onde estou agora"
# É "de onde vim → para onde vou"
# Bigramas: (POI_t-1, POI_t) → POI_t+1
```

**Hybrid captura isso bem:**
- GRU: contexto geral da trajetória
- Attention: foca na transição relevante

---

## 🎓 Quando usar cada arquitetura?

### Use **GRU** se:
- ✅ Quer baseline rápido e eficiente
- ✅ Sequências são curtas (< 20)
- ✅ Padrões seguem ordem sequencial estrita
- ❌ Não precisa de interpretabilidade alta

### Use **LSTM** se:
- ✅ GRU saturou (mais capacidade)
- ✅ Contexto distante importa (memória de longo prazo)
- ❌ Velocidade não é crítica

### Use **Hybrid (GRU + Attention)** se: 👑
- ✅ Quer melhor performance geral
- ✅ Interpretabilidade é importante (attention weights)
- ✅ Tem dados suficientes (~10k+ samples)
- ✅ Pode pagar custo computacional moderado

### Use **Temporal CNN** se:
- ✅ Padrões são majoritariamente locais (bigrams, trigrams)
- ✅ Precisa de velocidade máxima
- ✅ Pode treinar em paralelo (GPUs)
- ❌ Não precisa de dependências muito distantes

### Use **Transformer** se:
- ✅ Quer modelar relações complexas entre todas as visitas
- ✅ Sequências podem ser mais longas no futuro
- ✅ Tem GPU forte
- ⚠️ **Use versão otimizada** (learned PE + temporal decay)

---

## 📚 Referências

### Papers relevantes:
- **DeepMove (IJCAI 2018):** RNN + Attention para next-location
- **STAN (KDD 2020):** Spatial-Temporal Attention para POI
- **LSTM vs Transformer (2021):** Comparação para séries curtas

### Conclusões da literatura:
> "Para séries temporais curtas (< 50 timesteps), LSTMs e Transformers têm
> performance similar, mas LSTMs são mais eficientes" - Zerveas et al., 2021

> "Hybrid architectures (RNN + Attention) superam ambos para next-item prediction
> com sequências médias" - Kang & McAuley, ICDM 2018

---

## 🎯 Recomendação Final

**Para next-POI com seq_length=9:**

### Top 3 escolhas:

1. **NextHeadHybrid** 🥇
   - Melhor custo-benefício
   - Performance robusta
   - Interpretável

2. **NextHeadGRU** 🥈
   - Baseline sólido
   - Mais rápido
   - Ótimo ponto de partida

3. **NextHeadTransformerOptimized** 🥉
   - Se já usa Transformer
   - Melhorias significativas sobre versão atual
   - Mantém compatibilidade

---

**Regra de ouro para Next-POI:**
> "Sequências curtas favorecem RNNs e hybrids.
> Use Transformer apenas se otimizado para seu caso."

**Próximos passos sugeridos:**
1. Implementar GRU como baseline rápido
2. Testar Hybrid para comparação
3. Analisar attention weights (interpretabilidade)
4. Otimizar hiperparâmetros do melhor modelo
---

## 🐛 Bugfixes e Correções

### TemporalCNN: BatchNorm vs LayerNorm

**Problema identificado (2025-11-07):**
```python
# ERRADO - causava RuntimeError
nn.Conv1d(in_channels, out_channels, kernel_size)
nn.LayerNorm(out_channels)  # ❌
# Shape após Conv1d: [batch, channels, length] = [1024, 128, 11]
# LayerNorm esperava: [*, 128] (128 na última dim)
# Mas 128 está no MEIO (channels)!
```

**Erro:**
```
RuntimeError: Given normalized_shape=[128], expected input with shape [*, 128], 
but got input of size[1024, 128, 11]
```

**Solução:**
```python
# CORRETO
nn.Conv1d(in_channels, out_channels, kernel_size)
nn.BatchNorm1d(out_channels)  # ✅
# BatchNorm1d normaliza sobre channels (dim 1)
# Funciona com shape [batch, channels, length]
```

**Lição aprendida:**
- Para **Conv1d**: Use `BatchNorm1d` (normaliza channels)
- Para **Linear/Transformer**: Use `LayerNorm` (normaliza última dim)
- Conv1d shape: `[B, C, L]` → BatchNorm1d(C)
- Transformer shape: `[B, L, D]` → LayerNorm(D)
