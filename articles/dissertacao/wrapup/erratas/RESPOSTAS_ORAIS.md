# Respostas orais das erratas

Uma pagina por errata. A frase e para ser dita em pe, sem consultar nada, e cada numero que ela
carrega esta verificado contra os artefatos (ver `VERIFICACAO.md`).

---

## Q13 -- o controle de concatenacao

**A pergunta que vem.** "O senhor afirma que o ganho vem da estrutura hierarquica e nao das
features. Como sabe?"

**A resposta, dita em pe.**

> Nao sabemos, e a afirmacao no texto esta errada. Refizemos o controle na escala da propria
> Tabela 9, e ele mostra o contrario: concatenar as features por visita ao place embedding fecha
> 111 por cento do intervalo em Alabama e 70 por cento em Arizona. Sao as features que carregam a
> maior parte do ganho de categoria. Ha errata escrita.
>
> O que a tabela estabelece continua de pe: a representacao de entrada domina a arquitetura. Injetar
> informacao por visita move o resultado muito mais do que qualquer diferenca de arquitetura que
> medimos. O que cai e a afirmacao mais fina sobre qual parte da representacao carrega o ganho.

**Se insistirem: entao a representacao por check-in nao serve para nada?**

> Serve, e Arizona mostra onde. La a representacao por check-in ainda fica 0,78 ponto acima da
> concatenacao, unanime nas cinco dobras, com p de 0,03. Em Alabama os dois empatam. Com dois
> datasets nao da para dizer se essa sobra cresce com o vocabulario de regioes, e nao vou afirmar
> que cresce.

**A procedencia, se perguntarem.** Os dois controles de fidelidade reproduzem a Tabela 9 dentro de
um decimo de ponto nos dois datasets. Sem isso a comparacao nao valeria, e de fato tres receitas
plausiveis nao reproduzem.


## Q14 · "O artigo lista o confundimento de capacidade como um dos cinco limites do resultado de
regiao. A lista da dissertacao nao o carrega. Por que ele saiu?"

**Ele nao deveria ter saido, entra como errata, e agora vem com a medicao que o artigo dizia faltar.**

> O artigo declara que o controle pareado por capacidade nao havia sido rodado. Eu o rodei. Dando ao
> modelo dedicado de regiao o orcamento inteiro de parametros do modelo conjunto, ele passa o
> conjunto em California, por quatro decimos de Acc@10, com a diferenca separada de zero em cinco
> dobras e direcao unanime, e o iguala em Texas, onde a diferenca nao se separa de zero. Um braco
> mais estreito, com cinquenta e sete por cento daquele orcamento, ja chega no mesmo nivel. Ou
> seja: a vantagem de regiao que eu reporto mede capacidade, nao troca entre as tarefas. O que
> sobrevive e a afirmacao operacional, um modelo produz as duas predicoes em uma passada sem custo
> mensuravel em nenhuma das duas tarefas, e essa continua de pe.

**Se perguntarem se isso derruba a tese.** Nao, e por uma razao que o documento ja declara: a tese
e sobre representacao, e vive no eixo de categoria. La o controle de capacidade aponta na direcao
oposta, multiplicar por seis e meio os parametros do dedicado **baixa** o macro-F1 dele. Os dois
ganhos de regiao sao resultados secundarios, fora do plano de analise registrado, e a p. 76 e a
p. 88 ja dizem isso.

**Se perguntarem por que reportar um resultado que enfraquece o proprio texto.** Porque a
alternativa e deixar a banca descobrir. O limite estava no artigo; retirá-lo da dissertacao sem
medi-lo seria a unica versao indefensavel dos tres caminhos.

---

## Q15 · o quarto fundamento de integridade descrito na errata do suplemento

**Nao e errata a escrever. E errata a corrigir, e a correcao e retirar.**

> Aquela linha da tabela de errata descreve uma auditoria de desenvolvimento feita sobre uma
> construcao anterior da representacao, e a propria linha declara os tres limites dela: uma sonda
> linear, Florida numa unica inicializacao aleatoria, e construcoes anteriores da representacao. O
> texto que eu depositei nao carrega esse fundamento, e nao deveria: ele mediria uma preparacao que
> nao e a que os resultados usam. A linha de errata e que esta sobredeclarando, e ela sai.

**A verificacao.** As expressoes "on three grounds", "fourth ground", "linear probe" e
"forward-edge" tem zero ocorrencias em prosa viva do volume principal; "linear probe" aparece uma
vez, dentro da propria tabela de errata. Medido em 2026-08-13.
