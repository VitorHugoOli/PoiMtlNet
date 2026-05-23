# Prompt para Claude Design — ST-MTLNet @ CoUrb 2026

> Cole este prompt **inteiro** numa nova conversa Claude (claude.ai). Anexe junto:
>
> 1. `DECK_CONTENT.md` (este repositório, mesma pasta)
> 2. `articles/CoUrb_2026/imagens/arquitetura_modelo.png`
> 3. `articles/CoUrb_2026/imagens/subáreas/distribuicao_estados.png`
>
> O Claude gera um **artefato HTML standalone** com 10 slides + apêndice, navegável por teclado, exportável como PDF.

---

## INÍCIO DO PROMPT (copie a partir daqui)

Você é um designer especialista em apresentações acadêmicas para conferências brasileiras de computação. Sua tarefa é construir um **deck de slides em HTML standalone** (um único arquivo `.html` autocontido com CSS inline), pronto para ser projetado em uma apresentação oral de 9 minutos no **X Workshop de Computação Urbana (CoUrb 2026)** em **25 de maio de 2026**, em Praia do Forte/BA.

### Especificação técnica do artefato

- **Output**: um único artefato HTML standalone, sem dependências externas (sem CDN, sem fontes web — pode usar `system-ui` ou Inter via `@import` opcional)
- **Tamanho dos slides**: 16:9, viewport 1920×1080
- **Navegação**: setas ←/→ para mudar de slide, `F` para fullscreen, `Esc` para sair, `P` para imprimir/PDF
- **Numeração**: rodapé mostra `N / 10` (apêndice numerado A1…A13)
- **Print-to-PDF**: media query `@media print` que coloca cada slide em uma página A4 landscape sem quebras
- **Slides ocultos**: apêndice (A1…A13) só acessível por hash `#a1`, `#a2`… ou tecla `B`; não aparece no fluxo normal de setas

### Conteúdo

O conteúdo completo está no arquivo anexo **`DECK_CONTENT.md`**. Use-o como **fonte de verdade**:
- Textos exatos (títulos, subtítulos, bullets, talking points NÃO entram no slide — só notas)
- Cores, tipografia, layout descrito por slide
- Posicionamento de imagens
- Dados do gráfico de barras (slide 7)
- Tabelas (slides 5, 6, e apêndices)

### Identidade visual (resumo)

- **Paleta**: 
  - texto `#1A1A1A`
  - destaque verde-azulado `#16A086`
  - números laranja `#E67E22`
  - limitações roxo `#663399`
  - perdas vermelho `#C0392B`
  - cinza médio `#7F8C8D`
  - fundo `#FFFFFF`, fundo de card `#F4F6F7`
- **Tipografia**: sans-serif (Inter ou system-ui)
- **Topo dos slides de conteúdo**: barra horizontal verde-azulada de 6 px
- **Capa**: barra vertical verde-azulada de 80 px à esquerda
- **Rodapé** (todos exceto capa): 3 zonas — "🎓 NESPeD-LAB · UFV" / "ST-MTLNet @ CoUrb 2026" / "N / 10" — fonte 14 pt cinza

### Imagens

Duas imagens anexas devem ser embutidas como **data URI base64** (`<img src="data:image/png;base64,...">`) para manter o HTML autocontido:

1. `arquitetura_modelo.png` → slide 4
2. `distribuicao_estados.png` → slide 6

Se for grande demais para inline (>2 MB), use placeholder com instrução: `<img src="arquitetura_modelo.png" alt="...">` e avise o usuário para salvar a imagem ao lado do HTML.

### Gráficos

**Slide 7** precisa de um **gráfico de barras agrupado** com os dados da seção "Slide 7" do `DECK_CONTENT.md`. Implementar:
- SVG inline (não Chart.js, não dependência externa)
- 12 grupos no eixo X (4 categorias × 3 estados)
- 2 barras por grupo (Baseline azul `#5DADE2` · ST-MTLNet `#16A086`)
- Eixo Y 0–80, gridlines suaves
- Labels diretamente nos eixos, legenda discreta

### Comportamento P0 — checagem de números

**OBRIGATÓRIO**: Antes de escrever o HTML, releia a seção "Slide 7" e "Slide 8" do `DECK_CONTENT.md` e confirme que estes números aparecem **literalmente** no deck:

- Slide 7: `≈ +21 pp F1` (NÃO "+20–24"), com breakdown `FL +20,2 · CA +20,9 · TX +22,0`
- Slide 8: `15 / 21 vitórias + 1 empate técnico` (NÃO "16/21"), `~72%` (NÃO "76%")
- Slide 10: idem ao slide 8

Esses números são o resultado de uma auditoria crítica das tabelas do paper. Não substitua por números do artigo original sem alerta.

### Slide-by-slide

Construa exatamente os 10 slides principais + 13 de apêndice descritos em `DECK_CONTENT.md`. Para cada slide:

1. Use o **título exato**
2. Reproduza o **layout descrito** (colunas, cards, posicionamento)
3. Coloque os **textos exatos** (não parafraseie)
4. Aplique a **paleta de cores** sugerida nos elementos certos
5. Coloque a **figura** no slide indicado
6. **Talking points** vão no atributo `data-notes` do slide ou em comentários HTML — NUNCA no slide visível

### Slides especiais

- **Slide 3**: diagrama "1 vetor → 3 vetores" — desenhar com `<div>` + `border-radius` + flexbox; não usar SVG complexo
- **Slide 7**: gráfico SVG inline (12 grupos × 2 barras)
- **Slide 8**: 3 cards lado a lado, cores diferentes por status (verde / amarelo / vermelho)
- **Apêndice A1, A2**: tabelas full F1 — recriar em HTML ou usar `<img>` da tabela exportada

### Modo speaker notes (opcional, bonus)

Se possível, incluir um modo `?notes=1` que mostra os talking points abaixo de cada slide para ensaio.

### Critérios de aceite

- [ ] Abre em qualquer browser moderno sem erros
- [ ] Setas ←/→ navegam slides 1–10
- [ ] Tecla `B` ou hash `#a1` acessa apêndice
- [ ] Tecla `P` imprime sem cortar slides
- [ ] Todos os números corrigidos (P0 da auditoria) presentes exatamente como descritos
- [ ] Paleta aplicada consistentemente
- [ ] 2 imagens carregam (inline ou referenciadas)
- [ ] Gráfico do slide 7 renderiza com os 12 grupos
- [ ] Speaker notes acessíveis (via `data-notes` ou `?notes=1`)

### Entrega

Entregue **um único artefato HTML**, com instrução no topo sobre como abrir, navegar e exportar PDF. Se ficar > 200 KB sem imagens (> 5 MB com imagens), divida em CSS externo numa segunda mensagem.

**Comece agora**: leia `DECK_CONTENT.md` integralmente primeiro, valide os números corrigidos (Slide 7, 8, 10), depois construa o HTML.

## FIM DO PROMPT
