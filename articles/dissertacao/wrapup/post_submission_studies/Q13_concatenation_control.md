# Q13 -- o controle de concatenacao: resultado

**Medido em 2026-08-16.** Semente 0, cinco dobras, engine e receita identicas entre os tres bracos.
O resultado contradiz a frase depositada, e por isso gerou errata.

## A pergunta

A Tabela 9 do Capitulo 5 compara a representacao por check-in contra o place embedding. A frase
depositada afirma que o ganho vem da representacao hierarquica, e nao de injecao de features, e
apoia isso num controle que teria fechado menos de um decimo do intervalo. Esse decimo vem de outro
estudo, medido em outra variante do grafo e com outro codigo de treino, entao seus valores nao estao
na escala da Tabela 9. Em Alabama aquele estudo reporta um ganho de $+2{,}0$ pontos, maior que o
intervalo inteiro da Tabela 9 ali, que e $+1{,}62$. Este estudo refaz o controle na mesma escala.

## O desenho

Tres bracos por dataset, mesma semente, mesmas dobras, mesma receita, variando apenas a entrada.

| braco | entrada | papel |
|---|---|---|
| place | place embedding, 9 blocos de 64 | **controle de fidelidade** |
| feat | place embedding mais as 11 features por visita, 9 blocos de 75 | o controle de concatenacao |
| checkin | representacao por check-in, 9 blocos de 64 | **controle de fidelidade** |

Os dois controles de fidelidade tem de reproduzir as colunas da Tabela 9. Se nao reproduzirem, os
tres bracos nao estao na escala da tabela e a comparacao nao vale.

## O controle de fidelidade tem de ser por fold, nao por media

A primeira onda usou uma unica taxa de aprendizado, $0{,}0025$, nos tres datasets. Pela **media** o
controle passou em todos: Arizona deu $31{,}9660$ contra $31{,}9278$ reportado, quatro centesimos de
diferenca. Mas a taxa **nao e uma so**: cada dataset tem a sua, registrada no proprio sidecar.

| dataset | alabama | arizona | florida | california | texas | istanbul |
|---|--:|--:|--:|--:|--:|--:|
| `max_lr` | 0,0025 | 0,0005 | 0,005 | 0,005 | 0,005 | 0,0005 |

$0{,}0025$ e a taxa de Alabama. Comparando **fold a fold**, a diferenca aparece:

| dataset | taxa usada | maior diferenca por fold | diferenca na media |
|---|---|--:|--:|
| Alabama | 0,0025, a correta | $0{,}0000$ nos cinco folds | $0{,}0000$ |
| Arizona | 0,0025, a de Alabama | $0{,}2128$, com troca de sinal | $+0{,}0382$ |
| Florida | 0,0025, a de Alabama | $0{,}0930$, com troca de sinal | $+0{,}0340$ |

Em Alabama, com a taxa certa, os cinco folds saem **exatamente iguais** aos reportados. Essa
igualdade e a evidencia de que o braco reproduz; media proxima nao e, porque uma media pode cair
perto por compensacao entre folds que discordam. Arizona e Florida foram refeitos com a taxa de cada
um.

### Arizona nao reproduz fold a fold, e a causa ainda nao esta estabelecida

Refeito com a taxa de Arizona, o braco `place` ainda **nao** reproduz fold a fold: a maior diferenca
e $0{,}19$ ponto, praticamente a mesma de antes. Entao a taxa nao era a unica causa. Comparando as
curvas de validacao epoca a epoca no fold 1:

| dataset | diferenca media ao longo das 50 epocas | maior diferenca |
|---|--:|--:|
| Alabama | $0{,}0000$ | $0{,}0000$ |
| Arizona | $+0{,}0613$ | $0{,}3463$ |

**Correcao a uma versao anterior deste documento.** Esta secao afirmava que o treino nao e
determinista e que a reproducao exata de Alabama foi sorte. Isso esta errado, e contradito pela
propria linha de Alabama na tabela acima: duas execucoes independentes, separadas por cinco dias,
concordando em quatro casas decimais em **todas as cinquenta epocas**, sao evidencia de que o
caminho **reproduz**. Nao-determinismo produziria discordancia nos dois datasets, nao em um so. A
evidencia que eu tinha reunido tambem era fraca: uma busca por chaves de determinismo no codigo,
que apenas mostra que nao ha uma chave explicita, e nao que o resultado varia.

**O que esta estabelecido:** Alabama reproduz exatamente; Arizona nao. **O que nao esta:** por que.
A hipotese de ruido de execucao e testavel com um controle direto, rodar a mesma configuracao de
Arizona duas vezes e medir a diferenca entre elas, e esse controle esta na fila. Se as duas
execucoes sairem identicas, a diferenca contra o valor reportado e uma diferenca real de entrada ou
de configuracao que eu ainda nao encontrei, e o braco `place` de Arizona nao esta verificado.
Enquanto o controle nao rodar, a fidelidade de Arizona esta **em aberto**, e nao explicada.

**O que isso afeta, e o que nao afeta.** O ganho da concatenacao em Arizona e de $1{,}70$ ponto,
uma ordem de grandeza acima da discrepancia de $0{,}19$, e os tres bracos do dataset rodaram na
mesma sessao com a mesma receita, de modo que qualquer efeito comum a eles se cancela no teste
pareado. A conclusao sobre a concatenacao nao depende da resolucao deste ponto. O que depende e a
afirmacao de que o braco de Arizona esta na escala exata da Tabela 9, e essa afirmacao fica
suspensa ate o controle responder.

## Os controles de fidelidade

| dataset | braco | medido | Tabela 9 | estado |
|---|---|--:|--:|---|
| Alabama | place | 29,1481 | 29,15 | **verificado**: igualdade exata fold a fold e epoca a epoca |
| Alabama | check-in | 30,7058 | 30,77 | dentro de $0{,}07$ |
| Arizona | place | 31,9953 | 31,93 | **em aberto**: media dentro de $0{,}07$, mas nao reproduz por fold |
| Arizona | check-in | 34,4991 | 34,51 | dentro de $0{,}02$ |
| Florida | place | 37,1524 | 37,13 | media dentro de $0{,}03$; por fold, maior diferenca $0{,}09$ |
| Florida | check-in | 37,3602 | 37,36 | dentro de $0{,}001$ |

A discrepancia por fold **nao cresce com o tamanho do dataset**: Florida tem treze vezes as janelas
de Alabama e seis vezes as de Arizona, e sua maior diferenca por fold ($0{,}09$) e menor que a de
Arizona ($0{,}19$). Nos dois casos o sinal alterna entre folds e a media fica perto de zero. Isso e
compativel com ruido de execucao, mas o controle direto e que decide.

## O resultado

| dataset | place | place + features | check-in | intervalo | ganho da concatenacao | fracao fechada |
|---|--:|--:|--:|--:|--:|--:|
| Alabama | 29,148 | 30,882 | 30,706 | $+1{,}558$ | $+1{,}734$ ($p = 0{,}003$) | **111 por cento** |
| Arizona | 31,995 | 33,697 | 34,499 | $+2{,}504$ | $+1{,}702$ ($p < 0{,}001$) | **68 por cento** |
| Florida | 37,152 | 38,171 | 37,360 | $+0{,}208$ | $+1{,}018$ ($p < 0{,}001$) | **490 por cento** |

Testes pareados sobre as cinco dobras. O ganho da concatenacao e unanime nas cinco dobras nos dois
datasets.

**Em Florida a concatenacao supera a representacao por check-in.** O intervalo entre as duas colunas
da tabela e de apenas $0{,}21$ ponto ali, o menor dos tres datasets, e a concatenacao sozinha ganha
$1{,}02$ ponto, ficando $0{,}81$ acima da representacao por check-in, unanime nas cinco dobras
($p = 0{,}001$). Os dois controles de fidelidade de Florida sao os mais proximos de todos:
$+0{,}022$ e $+0{,}000$ contra a tabela.

**Em Alabama a concatenacao alcanca a representacao por check-in.** Os dois bracos ficam a
$0{,}18$ ponto um do outro, e o teste pareado nao separa ($p = 0{,}53$). Em Arizona sobra intervalo:
a representacao por check-in fica $0{,}80$ ponto acima da concatenacao, unanime nas cinco dobras
($p = 0{,}03$).

## O que isso obriga a dizer

A frase depositada esta errada na direcao. Nao e verdade que o ganho de categoria venha da estrutura
hierarquica e nao das features: sao as features que carregam a maior parte dele, e em um dos dois
datasets carregam tudo. A errata Q13 substitui a frase.

**O que nao muda.** A conclusao central do capitulo e que a representacao de entrada domina a
arquitetura, e este resultado a apoia: injetar informacao por visita no place embedding move o
resultado muito mais do que qualquer diferenca de arquitetura medida na dissertacao. O que cai e uma
afirmacao mais fina, sobre qual parte da representacao carrega o ganho.

**O que fica em aberto.** Arizona mostra que sobra algo alem das features, e Alabama nao. Dois
datasets nao decidem se essa sobra cresce com o tamanho do vocabulario de regioes.

**Sobre Florida, e uma correcao ao que este documento afirmava.** A versao anterior dizia que
Florida nao rodou por falta de espaco em disco. Isso nunca foi medido, e esta errado. Medido em
2026-08-16: o disco tem 37 GB livres, e o `next.parquet` equivalente de Florida na engine de
check-in ocupa 4,0 GB. A causa real e mais simples: o `next.parquet` do place embedding para Florida
**nunca tinha sido construido**, e o script que o constroi existe. Florida entrou na onda depois
dessa medicao.

## Procedencia

- Receita, lida do sidecar do resultado reportado: `next_gru`, `bs 8192`, `max_lr 0,0025`,
  `logit_adjust_tau 0,5`, cinco dobras, 50 epocas, fp32, `compile+tf32`, semente 0.
- Selecao: macro-F1 na epoca de melhor macro-F1, dos CSVs de validacao por dobra.
- A receita custou a ser achada, e o controle de fidelidade e que revelou isso. Tres caminhos
  plausiveis nao reproduzem o valor reportado no fold 1 de Alabama (alvo $29{,}7261$):
  o driver que nomeia esta celula da $26{,}35$; o mesmo com `bs8192` da $26{,}77$; o harness de
  ablacao com a receita completa da $28{,}83$. Sem esse controle, qualquer um dos tres teria sido
  reportado como se fosse a coluna da tabela.

## O que foi construido, e o que nao foi tocado

**`next_region.parquet` para as engines de place embedding.** O carregador exige essa tabela mesmo
quando o alvo e categoria, e ela nao existia para elas. Copiada de `check2hgi_dk_ovl`, mesmas linhas
na mesma ordem, com guarda de alinhamento na escrita. E legitimo porque o carregador le dela apenas
`region_idx` e `last_region_idx`, rotulos derivados do mapa POI-para-regiao e do janelamento, nao do
substrato: verificado que duas engines com embeddings diferentes carregam esses rotulos identicos
linha a linha sob o mesmo janelamento. O proprio codigo ja usava esse argumento para outras engines.

**O construtor de features passou a aceitar o janelamento.** Ele reconstruia as janelas sempre com a
regra nao sobreposta, do estudo original, e abortava com razao ao encontrar 12.709 janelas onde as
sequencias atuais tem 96.326. Agora aceita `stride`, `min_sequence_length` e `emit_tail`, e os
defaults reproduzem o comportamento anterior byte a byte.

**O braco de concatenacao virou engine propria, `hgi_ovl_feat`.** Seu `next.parquet` ja traz o place
embedding com as features concatenadas. Assim o braco roda pelo caminho de treino normal e nenhum
codigo compartilhado foi alterado: a alternativa seria adicionar uma flag ao gerador de todos os
resultados da dissertacao.

## Uma nota de metodo

A primeira versao deste documento atribuiu a ausencia de Florida a falta de disco. Nenhuma medicao
sustentava isso, e o log do proprio driver reportava 37 GB livres em cada arm, acima do limite de
15 GB que ele mesmo impoe. A causa foi inventada para explicar uma ausencia cuja razao verdadeira
era um arquivo de entrada que nao tinha sido construido. Uma limitacao registrada com causa errada
e pior do que uma limitacao registrada sem causa: ela encerra a investigacao que teria resolvido o
ponto.
