# Checklist de submissão NetMob

## Antes de escrever

- [ ] Confirmar se será Main Conference ou Data Challenge.
- [ ] Reabrir a [página oficial 2026](https://netmob.org/) e confirmar datas.
- [ ] No Data Challenge, esclarecer por e-mail o conflito 20/08 versus 06/09 para o relatório final.
- [ ] Baixar/copiar a versão atual do template oficial; não adaptar template de 2023/2024.
- [ ] Confirmar compatibilidade com outro venue, caso o trabalho já tenha sido submetido/publicado.
- [ ] Congelar pergunta, hipótese, dataset, splits, métricas e baselines.

## Conteúdo científico mínimo

- [ ] Uma pergunta de pesquisa em uma frase.
- [ ] Uma contribuição principal e no máximo duas secundárias.
- [ ] Dados: origem, período, geografia, amostra, resolução e filtros.
- [ ] Split sem leakage e unidade correta de independência (usuário/tempo/região).
- [ ] Baselines fortes e comparáveis sob a mesma métrica.
- [ ] Média e dispersão/intervalo; não somente melhor seed.
- [ ] Delta absoluto e resultado absoluto.
- [ ] Pelo menos uma ablação ou teste de robustez que sustente o mecanismo.
- [ ] Viés de cobertura, representatividade, privacidade e validade externa.
- [ ] Linguagem causal somente se o desenho identificar efeito causal.

## Estrutura de duas páginas

- [ ] Título declara problema/descoberta; sigla do modelo é secundária.
- [ ] Primeiro parágrafo: contexto → lacuna → pergunta.
- [ ] Contribuições são verificáveis e correspondem às tabelas/figuras.
- [ ] Método pode ser entendido sem consultar outro paper.
- [ ] Resultado principal aparece na primeira metade da página 2, no máximo.
- [ ] Uma figura principal legível e autoexplicativa.
- [ ] Caption conclui, não apenas descreve eixos.
- [ ] Conclusão contém implicação e limite, não repete o resumo.
- [ ] Referências reduzidas às indispensáveis.

## Conformidade

- [ ] Exatamente duas páginas incluindo referências, figuras e tabelas.
- [ ] Sem apêndice.
- [ ] PDF abre corretamente e fontes estão incorporadas.
- [ ] Título, autores, afiliações e e-mails na primeira página.
- [ ] Template NetMob atual sem alteração de margens/fontes/espaçamento.
- [ ] Figuras vetoriais ou em resolução adequada; texto legível a 100%.
- [ ] Links, DOI, caracteres especiais e nomes de autores revisados.
- [ ] Metadados do sistema coincidem com o PDF.

## Revisão simulada (0–5)

| Pergunta | Nota |
|---|---:|
| dado móvel/comportamental é essencial à pergunta? | /5 |
| problema importa para ciência, cidade, sociedade ou indústria? | /5 |
| contribuição é nova e explícita? | /5 |
| desenho experimental sustenta o claim? | /5 |
| baselines e incerteza são suficientes? | /5 |
| insight é interpretável e acionável? | /5 |
| viés, privacidade e limites são tratados? | /5 |
| figura comunica a descoberta em 20 segundos? | /5 |

Critério interno: nenhum dos quatro primeiros itens abaixo de 4.

## Específico para `netcore`

- [ ] Não confundir next-POI, next-category e next-region.
- [ ] Usar “paridade” com margem explícita, não “vence”, para região.
- [ ] Resultados finais vêm do board com janelas sobrepostas, não números provisórios antigos.
- [ ] Mesmos estados, folds, seeds, loss e métricas nos lados STL e MTL.
- [ ] Explicar o confound de class weighting sem transformar o abstract em relatório de debugging.
- [ ] Figura Pareto: category macro-F1 × region Acc@10.
- [ ] Tabela compacta: STL category, STL region, MTL, ablação de substrato/confound.
- [ ] Headline: um modelo/um forward/duas tarefas; arquitetura é mecanismo, não o tema.

## Preparação da apresentação, se aceita

- [ ] Planejar 12 minutos e reservar a pergunta central para o primeiro minuto.
- [ ] Um slide de dados/viés; um de método; dois de resultados; um de limites/impacto.
- [ ] PDF de slides conforme instruções da edição.
- [ ] Bio curta do apresentador.
- [ ] Para poster, confirmar A0 vertical e responsabilidade de impressão.

