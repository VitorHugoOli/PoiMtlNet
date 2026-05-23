# Recomendações para a Apresentação

## Princípio orientador

Slot apertado (~9 min de fala) e plateia heterogênea (computação urbana abrange redes, IoT, sensoriamento, ML — não só especialistas em representações de POI). O deck precisa **vender o "porquê" rápido**, mostrar **uma figura forte da arquitetura**, **um número que choca** (+20–24 pp em F1 categoria), e admitir limitações com honestidade (Travel no next-POI).

## Tempo & ritmo

- **Alvo**: 9 minutos de fala, 2 min Q&A
- **Slides**: 10 a 11 (incluindo capa e final) → **~50–60 s por slide** em média
- Slides densos (metodologia, resultados) podem usar 70–90 s; slides leves (capa, contribuições, conclusão) ficam em 30–40 s
- **Regra prática**: ensaiar 2x cronometrado. Se passar de 9:30, cortar bullets.

## O que enfatizar

1. **Problema + lacuna em 1 frase**: DGI codifica espaço/categoria juntos num só vetor; não há tempo explícito.
2. **A jogada central**: trocar o embedding monolítico (64-d DGI) por **três encoders desacoplados** (HGI + Time2Vec + Spatial) concatenados em 192-d.
3. **Resultado-âncora**: **+20–24 pp F1 em todas as 21 combinações categoria×estado** — esse é o número que o público vai lembrar.
4. **Nuance honesta**: no next-POI ganha em 16/21; perde em **Travel** porque relações topológicas de longa distância ainda são melhor capturadas pelo grafo DGI.
5. **SIREN vs Sphere2Vec-M**: não há vencedor único — depende da distribuição geográfica do estado (SIREN forte em FL/CA, Sphere2Vec-M no TX).

## O que NÃO colocar

- Equações longas (FiLM, NashMTL, contrastiva): no máximo 1–2 fórmulas curtas se realmente reforçarem a mensagem
- Tabela completa de resultados (21 linhas × 3 colunas): mostrar **subset** (3 categorias-chave por estado) ou um heatmap/barras
- História do MTLNet original — assumir que o público conhece o contexto MTL
- Detalhes de hiperparâmetro / treino — fica para Q&A

## Boas práticas de design

- **16:9 widescreen**
- **Fontes**: sans-serif (Inter, Roboto, Helvetica) ≥ 24pt no corpo, ≥ 36pt em títulos
- **Cores**: paleta sóbria + 1 cor de destaque (sugestão: verde-azulado #16A086 — bate com a paleta do artigo "first")
- **Logos**: UFV + NESPeD-LAB no rodapé; SBC/CoUrb na capa
- **Imagens em alta**: reaproveitar `imagens/arquitetura_modelo.png` (já no artigo) e gerar slide de distribuição espacial com `imagens/subáreas/distribuicao_estados.png`
- **Citações**: rodapé compacto, formato `[Autor, ANO]` — não atrapalhar a leitura

## Boas práticas de discurso

- **Capa**: pausar 3 s, olhar para a plateia, dizer título + filiação **em voz alta**
- **Não ler bullets** — bullets são para a plateia, falar com elas, não delas
- **Apontar para a figura** (laser/cursor) ao explicar a arquitetura
- **Resultado-âncora primeiro, detalhes depois**: nunca enterre o gancho
- **Slide final ≠ "Obrigado"**: termina com **takeaways + perguntas**; "obrigado" se diz com a voz

## Especificidades do contexto SBC/CoUrb

- **Português**: artigo está em PT-BR, slides em PT-BR também (consistente com a plateia local)
- Termos em inglês como *embedding*, *encoder*, *Multitask Learning* mantidos em itálico, alinhado ao artigo
- Plateia tem interesse aplicado (mobilidade urbana) — destacar **utilidade prática** dos ganhos (ex.: recomendação contextual, planejamento)
- Q&A típica em CoUrb: questões sobre **escalabilidade** (dataset, custo de treino), **generalização** (outros estados/países), **interpretabilidade** dos encoders. Preparar 3 respostas curtas.

## Backup slides (apêndice — NÃO mostrar por padrão)

Manter ao final, fora do fluxo principal, slides extras para Q&A:
1. Tabela completa F1 categoria (21 linhas)
2. Tabela completa F1 next-POI
3. Detalhe do encoder SIREN (fórmula + arquitetura)
4. Detalhe do encoder Sphere2Vec-M (multi-escala)
5. Função de perda contrastiva binária
6. Hiperparâmetros de treino + FLOPs
7. Configuração de hardware
