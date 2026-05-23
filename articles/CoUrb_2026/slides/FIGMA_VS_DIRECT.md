# Posso usar Figma diretamente, ou você precisa passar o prompt?

## TL;DR

✅ **Posso usar o Figma MCP diretamente** desta sessão — o servidor está conectado e as ferramentas `mcp__plugin_figma_figma__*` estão disponíveis (use_figma, generate_figma_design, create_new_file, get_screenshot, etc.).

❗ **Mas há duas condições** antes de eu disparar a geração:

1. **Você precisa estar logado no Figma** na conta correta. O MCP usa as credenciais ativas no seu desktop. Verifique abrindo qualquer arquivo Figma em paralelo.
2. **Carregar a skill `figma-generate-design` antes**, que é o fluxo MANDATÓRIO segundo as instruções do servidor para gerar designs do zero. Posso carregar com `Skill`.

## O que eu faria, na prática

1. Carregar a skill `figma-generate-design` (passo obrigatório do servidor)
2. Chamar `whoami` para confirmar a conta logada (você confirma)
3. Chamar `create_new_file` para criar um novo arquivo "ST-MTLNet — CoUrb 2026"
4. Para cada um dos 10 slides do `STRUCTURE.md`, chamar `use_figma` / `generate_figma_design` com um prompt detalhado por slide (paleta, layout, tipografia, imagens)
5. Subir as duas figuras do paper (`arquitetura_modelo.png`, `subáreas/distribuicao_estados.png`) via `upload_assets`
6. Retornar o link do arquivo para você abrir/editar/exportar como PDF

**Tempo estimado**: ~5–10 minutos de tool calls em série.

## Quando NÃO usar o Figma MCP daqui

- Se você prefere PowerPoint/Keynote/Google Slides (mais fácil de editar no dia da apresentação) — nesse caso entrego só o conteúdo em texto e você monta
- Se você quiser controle 100% sobre o design (tipografia institucional UFV específica, etc.) — o MCP gera bem layouts, mas refinamento fino você quer fazer manualmente
- Se a conta Figma logada não for a sua

## Alternativa: prompt pronto

Se você preferir **rodar você mesmo** (em outra sessão Claude, em outra conta Figma, ou colando num assistente diferente), preparei [`FIGMA_PROMPT.md`](FIGMA_PROMPT.md) com o briefing completo. Cole no Claude com MCP Figma ativo, ou adapte para qualquer ferramenta de geração de slides (Beautiful.ai, Gamma, Tome).

## Recomendação

**Para uma apresentação na segunda-feira, sugiro o caminho híbrido**:

1. Eu gero o esqueleto no Figma (layouts + bullets + figuras já posicionadas)
2. Você abre no Figma, refina cores/tipografia institucional UFV, exporta PDF
3. Faz 2 ensaios cronometrados antes do voo

Isto economiza ~3–4 h de montagem manual e te dá liberdade de ajuste fino.

## Sua decisão

Diga uma das opções:

- **"vai no Figma"** → executo o fluxo MCP agora (preciso confirmar conta logada)
- **"PowerPoint/Keynote"** → fico só com markdown + você monta manualmente
- **"manda o prompt"** → uso o `FIGMA_PROMPT.md` para você rodar onde quiser
