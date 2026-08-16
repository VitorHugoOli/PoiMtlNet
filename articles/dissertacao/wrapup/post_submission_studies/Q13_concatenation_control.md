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

## Os controles de fidelidade passaram

| dataset | braco | medido | Tabela 9 | diferenca |
|---|---|--:|--:|--:|
| Alabama | place | 29,1481 | 29,15 | $-0{,}002$ |
| Alabama | check-in | 30,7058 | 30,77 | $-0{,}064$ |
| Arizona | place | 31,9660 | 31,93 | $+0{,}036$ |
| Arizona | check-in | 34,5224 | 34,51 | $+0{,}012$ |

Os quatro caem dentro de um decimo de ponto. A comparacao esta na escala da tabela.

## O resultado

| dataset | place | place + features | check-in | intervalo | ganho da concatenacao | fracao fechada |
|---|--:|--:|--:|--:|--:|--:|
| Alabama | 29,148 | 30,882 | 30,706 | $+1{,}558$ | $+1{,}734$ ($p = 0{,}003$) | **111 por cento** |
| Arizona | 31,966 | 33,746 | 34,522 | $+2{,}556$ | $+1{,}780$ ($p < 0{,}001$) | **70 por cento** |

Testes pareados sobre as cinco dobras. O ganho da concatenacao e unanime nas cinco dobras nos dois
datasets.

**Em Alabama a concatenacao alcanca a representacao por check-in.** Os dois bracos ficam a
$0{,}18$ ponto um do outro, e o teste pareado nao separa ($p = 0{,}53$). Em Arizona sobra intervalo:
a representacao por check-in fica $0{,}78$ ponto acima da concatenacao, unanime nas cinco dobras
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
