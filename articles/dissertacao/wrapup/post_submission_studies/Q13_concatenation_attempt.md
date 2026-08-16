# Q13 -- o controle de concatenacao na preparacao atual: tentativa de execucao

**Estado: NAO CONCLUIDO.** Registrado aqui porque a tentativa produziu um diagnostico util e porque
a proxima sessao nao deve repetir os mesmos quatro tropecos.

## O que o experimento responderia

A Tabela 9 do Cap. 5 compara a representacao por check-in contra o place embedding. O controle de
concatenacao junta ao place embedding as mesmas features por visita que o grafo le, e pergunta
quanto do intervalo isso fecha. O estudo que a dissertacao cita rodou em outra preparacao, com outro
harness, entao seus valores nao estao na escala da Tabela 9. Este rodaria na mesma escala, com tres
bracos por dataset e o place embedding sozinho como controle de fidelidade.

## Por que parou

O harness nao consegue montar o braco de place embedding com a mesma engine que produziu a coluna
reportada. Duas barreiras, ambas medidas:

**Primeira: a engine reportada nao tem a tabela que o carregador exige.** A coluna Place level foi
produzida com `hgi_dk_ovl`. Esse nome nao estava na lista de escolhas do harness; acrescentei. Mas o
carregador le `next_region.parquet` mesmo quando o alvo e categoria, e essa tabela nao foi construida
para essa engine. A mensagem e explicita: `next_region not yet built for EmbeddingEngine.HGI_DK_OVL`.

**Segunda: as engines vizinhas nao servem de substituto.** Medido em Alabama:

| engine | embeddings | janelas em `next.parquet` |
|---|---|---|
| `hgi` | 11.848 lugares | 12.709 |
| `hgi_dk_ovl` | 11.848 lugares | 96.326 |
| `check2hgi_dk_ovl` | 113.846 check-ins | 96.326 |

O protocolo reportado tem 96.326 janelas em Alabama, o mesmo numero do registro do piso de Markov e
da Tabela 1. Entao `hgi` tem os vetores certos com o janelamento errado, e `check2hgi_dk_ovl` tem o
janelamento certo com os vetores errados.

**Como o braco errado foi lancado, e isto e um erro meu que vale registrar.** Escrevi no driver que
`check2hgi_dk_ovl` resolve para os mesmos vetores que `hgi_dk_ovl`, citando 11.848 linhas identicas
em Alabama. A verificacao que produziu esse numero comparou `hgi` com `hgi_dk_ovl`, e nao envolveu
`check2hgi_dk_ovl` em momento nenhum. Transportei o resultado de uma comparacao para outra e a
escrevi como se tivesse sido medida. Medido depois: `hgi_dk_ovl` tem $(11.848, 66)$, vetores de
lugar; `check2hgi_dk_ovl` tem $(113.846, 68)$, vetores de check-in. Sao substratos diferentes, e o
nome parecido foi o que tornou o engano facil.

A consequencia foi imediata e o proprio experimento a detectou: o braco de fidelidade deu $32{,}75$
macro-F1 contra os $29{,}15$ reportados, porque rodou a representacao por check-in no lugar do place
embedding. **O controle de fidelidade falhou, que e exatamente o que ele existe para detectar.** Sem
ele passando, nenhuma comparacao entre os tres bracos teria valor. Se o braco de fidelidade nao
estivesse no desenho, o erro teria produzido tres numeros plausiveis e uma conclusao falsa.

**`hgi` tambem nao serve como substituto.** Testado depois: falha com
`FileNotFoundError: output/hgi/alabama/input/next_region.parquet`, a mesma barreira de `hgi_dk_ovl`.
O carregador exige essa tabela mesmo quando o alvo e categoria, e ela nao foi construida para
nenhuma das duas engines de place embedding.

**Terceira, e independente: o construtor de features rejeita o corpus.** Ele reconstroi as janelas a
partir dos check-ins e compara com a tabela de sequencias. Com `hgi` ele encontra 12.709 contra as
96.326 esperadas e aborta com `checkins_df is not the set these sequences were built from`. A
validacao esta correta; o problema e que ele foi escrito para o janelamento do estudo anterior.

## O que faltaria para concluir

Construir `next_region.parquet` para a engine de place embedding sob o janelamento atual, e adaptar
o construtor de features ao mesmo janelamento. E trabalho de engenharia de dados, nao um ajuste de
linha de comando, e muda arquivos de entrada que outros resultados leem. Por isso parei aqui em vez
de improvisar: um braco de fidelidade que nao reproduz a Tabela 9 nao pode fundamentar uma correcao
ao texto.

## O que isso NAO muda

A errata Q13 continua valida e nao depende deste experimento. Ela corrige um erro de **escopo**: a
frase depositada compara um ganho com um intervalo de outro estudo, em outra escala. Isso e
verificavel sem rodar nada, e a correcao e dizer a que escala os numeros pertencem. O experimento
fecharia o ponto de vez, permitindo uma afirmacao positiva sobre quanto a concatenacao fecha na
escala da Tabela 9; sem ele, a errata se limita a nao afirmar a separacao que o artigo recusa.

## Licoes pagas, para nao custarem de novo

1. A shell nao-interativa nao ativa o venv: usar o interpretador por caminho absoluto.
2. O braco de categoria nao leva `--input-type`; `--target category` ja escolhe a tabela.
3. `\include` no LaTeX exige que o subdiretorio de aux exista antes do build.
4. O nome de engine no sidecar de um resultado pode nao ser aceito pelo harness que o produziu,
   porque a lista de escolhas e mantida a mao.
5. Nomes de engine parecidos escondem substratos diferentes: `hgi_dk_ovl` e vetores de lugar,
   `check2hgi_dk_ovl` e vetores de check-in. Antes de trocar um pelo outro, ler as duas formas, e
   nao reaproveitar a verificacao de um par para justificar outro.
