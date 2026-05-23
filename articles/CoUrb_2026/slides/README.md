# CoUrb 2026 — Slides ST-MTLNet

Pasta com o material para a apresentação oral do artigo **"ST-MTLNet: Representações Espaço-Temporais de Pontos de Interesse para Aprendizado Multitarefa"** no X Workshop de Computação Urbana (CoUrb 2026), SBRC 2026.

## Apresentação

- **Data**: 25/05/2026 (segunda-feira)
- **Local**: Praia do Forte, Bahia — SBRC 2026
- **Sessão**: Sessão Técnica 1 (11:15–12:00) — 4 artigos no slot
- **Tempo estimado por apresentação**: ~11 min totais (≈ 8–9 min de talk + 2–3 min de Q&A)
- **Apresentador**: Vitor Hugo de Oliveira Silva (2º autor)
- **Modalidade**: Presencial

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| [`EVENT_INFO.md`](EVENT_INFO.md) | Detalhes da edição, tempo, template, regras, links oficiais |
| [`RECOMMENDATIONS.md`](RECOMMENDATIONS.md) | Boas práticas de apresentação acadêmica + diretrizes do contexto SBC/SBRC |
| [`STRUCTURE.md`](STRUCTURE.md) | **v2** Roteiro slide-a-slide (10 + apêndice) com bullets, talking points e orçamento de tempo |
| [`DECK_CONTENT.md`](DECK_CONTENT.md) | ⭐ **Fonte única consolidada** — todo o conteúdo do deck (textos, layouts, cores, dados de gráfico, Q&A) pronto para gerar |
| [`CLAUDE_DESIGN_PROMPT.md`](CLAUDE_DESIGN_PROMPT.md) | ⭐ **Prompt para Claude Design** gerar artefato HTML standalone (cole numa nova sessão claude.ai + anexe `DECK_CONTENT.md` + 2 imagens) |
| [`FIGMA_PROMPT.md`](FIGMA_PROMPT.md) | Prompt alternativo para Figma MCP (carrega skill `figma-generate-design`) |
| [`FIGMA_VS_DIRECT.md`](FIGMA_VS_DIRECT.md) | Avaliação: Figma direto vs. prompt |
| [`advisor_feedback.md`](advisor_feedback.md) | Recomendações do agente sênior (nota 7,5/10 → 8,5–9 com 3 ajustes) |
| [`judge_feedback.md`](judge_feedback.md) | Auditoria crítica (nota 6,0/10 → 7,5–8 com correções P0; descobriu inconsistências numéricas no paper) |

## Caminho recomendado para gerar o deck visual

1. Abrir uma **nova conversa** em claude.ai (com artifacts ativados)
2. Anexar à mensagem: `DECK_CONTENT.md` + `imagens/arquitetura_modelo.png` + `imagens/subáreas/distribuicao_estados.png`
3. Colar o prompt completo de `CLAUDE_DESIGN_PROMPT.md`
4. Aguardar o artefato HTML standalone
5. Salvar como `.html`, abrir no navegador → `Cmd+P` → "Salvar como PDF"

## Status atual

- ✅ Conteúdo finalizado v2 (correções P0 aplicadas)
- ✅ Prompt pronto para Claude Design
- ⚠️ Limite MCP do Figma atingido nesta sessão; arquivo Figma vazio existe em [https://www.figma.com/slides/f1LNKjMvc8EM0aszOrfjC7](https://www.figma.com/slides/f1LNKjMvc8EM0aszOrfjC7)
- ⚠️ **CONFIRMAR antes da apresentação**: split de next-POI é por usuário/POI/janela? (Ver `STRUCTURE.md` slide 6 e Q&A Q7.)
