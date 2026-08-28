# CORTES_FALA.md — as cinco reescritas, prontas e NÃO aplicadas

> **Estado: aguardando o autor.** Escritas 2026-08-26, remedidas 2026-08-27 contra o deck vigente.
> **Nada aqui muda a TELA.** São cinco falas substituídas; nenhum slide sai, nenhum conteúdo de tela muda.

## Por que existem

Os quatro passos do protocolo levam **6:53** hoje — quase sete minutos explicando como se mede — e o
slide de resultado conjunto leva **3:15** sozinho. As reescritas chegam ao mesmo lugar com metade das
palavras. **Nenhum fato sai**; sai o rodeio.

## O que economizam

| | hoje | reescrita | corte |
|---|---:|---:|---:|
| **Protocolo 1 · the unit of data** | 176 pal · 75 s | **102 pal · 44 s** | −74 |
| **Protocolo 2 · what is measured** | 236 pal · 101 s | **113 pal · 48 s** | −123 |
| **Protocolo 3 · what is compared** | 231 pal · 99 s | **113 pal · 48 s** | −118 |
| **Protocolo 4 · how it is decided** | 321 pal · 138 s | **177 pal · 76 s** | −144 |
| **Result 2 · one model, two tasks** | 456 pal · 195 s | **231 pal · 99 s** | −225 |
| **TOTAL** | **1420 pal · 10:08** | **736 pal · 5:15** | **−684 pal · 4:53** |

## O efeito no relógio

| | palavras | tempo |
|---|---:|---:|
| fala hoje *(já sem o slide 38, migrado para a reserva)* | 7.346 | **52:28** |
| com os cinco cortes | 6662 | **47:35** |
| teto do Art. 23 | 7.000 | 50:00 |
| **margem** | **338** | **2:24** |

> **Margem real**, não margem de leitura corrida.

---

## ⚠ Duas armadilhas de contagem, as duas já disparadas

**1 · O `s44` ficou velho uma vez.** Foi escrito **antes** de o asterisco honesto entrar na fala
corrente, e o corte o perdia. **Reposto** — as 29 palavras finais. ⚠ **A tela do passo 4 carrega um
`*` no rodapé**: sem o asterisco na fala, a banca lê a marca e não recebe a explicação.

**2 · Eu contei o asterisco duas vezes** ao gerar este arquivo — a `ppt` já o tinha anexado ao
`s44.txt` e eu somei de novo, inflando o total em 29 palavras. **Peguei porque os totais não bateram
com os dela.** *Ler um arquivo que outra mão está editando e assumir que ele não mudou é a mesma
doença do número escrito à mão.*

🛑 **São DUAS perguntas, e a segunda foi descoberta em 27/08 depois de a primeira falhar sozinha.**

**Pergunta 1 — o que ENTROU no original depois que a reescrita nasceu?**
Foi ela que achou o asterisco honesto do `s44`, que a reescrita não tinha porque ainda não existia.

**Pergunta 2 — o que a reescrita AFIRMA que o deck já não sustenta?**
O `s46` abria com *"Terceiro resultado"*, escrito quando o slide se chamava `Result 3`. Aplicá-lo
**reverteu a renumeração já feita**: a tela dizia `Result 2` e a boca dizia "terceiro". **A pergunta 1
é cega para esta direção** — o asterisco foi apanhado porque **faltava**; este não, porque **sobrava**.

> **Um texto de substituição velho não só perde o que entrou depois — ele REINTRODUZ o que saiu.**
> A pergunta 2 só se responde lendo a reescrita **contra a TELA de hoje**, nunca contra a fala antiga.

⚠ **E não teste isto no PDF.** A fala vive em `% FALA:`, que **não renderiza** — procurar `"Terceiro
resultado"` no `pdftotext` dá ausente sempre, e ausente é o resultado que parece confirmar o
conserto. **Confira no `.tex` e no `SLIDES.md`.** *(Eu caí nisso ao verificar o próprio conserto.)*

🛑 **E a pergunta velha, que continua valendo:** a
similaridade entre uma fala e a reescrita dela é **0,20–0,37 por desenho**, e o `diff_fala.py` é
**estruturalmente cego** para essa classe. **A pergunta é: o que entrou no original DEPOIS que a
reescrita nasceu?**

---

## Os cinco textos

### Protocolo 1 · the unit of data  ·  `s41`

> O protocolo, em quatro passos, e de cada um eu digo a razão. Primeiro, a unidade de dado. Validação cruzada de cinco partições, disjunta por usuário: todas as janelas de uma pessoa ficam do mesmo lado da divisão. Isso é o reparo direto da limitação que o Capítulo 3 declarou, em que os check-ins de um mesmo usuário caíam dos dois lados. E uma ressalva que eu dou antes de alguém pedir: a partição retida é a que serve de validação, e eu não reservo uma terceira divisão. É dela que sai o segundo limite que eu apresento no fim desta seção.

### Protocolo 2 · what is measured  ·  `s42`

> Segundo, o que se mede. Em categoria, macro-F1, e a razão é a distribuição: na Flórida, um preditor que sempre responde a categoria mais comum acerta vinte e quatro vírgula sete por cento das visitas e ainda assim marca cinco vírgula sete de macro-F1. É por isso que acurácia simples não é a métrica aqui: ela premiaria exatamente esse preditor. Em região, acurácia em dez, a fração de visitas cuja região verdadeira está entre as dez mais pontuadas. E eu digo o que ela não faz: não separa o primeiro lugar do décimo. Região ausente do treino conta como erro. Os pontos de referência são o modelo dedicado e o piso de Markov.

### Protocolo 3 · what is compared  ·  `s43`

> Terceiro, o que se compara. O modelo conjunto contra os dedicados, com a mesma representação, as mesmas janelas e as mesmas partições. E a convenção que decide qual número eu reporto: os dois resultados saem de um único modelo salvo por partição, escolhido pela média geométrica das duas métricas. Eu digo isso com todas as letras porque ela me custa caro: a convenção alternativa, ler cada tarefa na melhor época dela, é mais favorável ao modelo conjunto, e transformaria mais quatro células de categoria e mais duas de região em melhorias que sobrevivem à mesma correção. Eu escolhi a que produz menos vitórias, porque é a única que um sistema implantado consegue servir.

### Protocolo 4 · how it is decided  ·  `s44`

> Quarto, como se decide, e o ponto é que um ganho afirmado e uma paridade afirmada exigem testes diferentes. Para próxima categoria, superioridade: eu pergunto se o conjunto é melhor. Para próxima região, não-inferioridade, com margem de dois pontos registrada antes de qualquer resultado ser lido: eu pergunto se ele não é pior. Isso importa porque ausência de significância não é evidência de igualdade — dizer 'não deu diferença, logo empatou' é formalmente inválido, e é a prática corrente na literatura de multitarefa. O plano foi escrito antes. Teste t pareado, intervalo de noventa por cento, correção de Holm sobre os seis conjuntos. E um desvio declarado: o plano registrava Wilcoxon, e com quatro sementes o Wilcoxon exato não desce abaixo de zero vírgula zero seiscentos e vinte e cinco. Ele não podia decidir nada. Continua reportado ao lado, como sensibilidade, com os dois testes no código publicado. E o asterisco que está na tela: o protocolo estatístico foi refinado depois, com base na literatura. É posterior ao que a banca recebeu, e não muda nenhum veredito.

### Result 2 · one model, two tasks  ·  `s46`

> Terceiro resultado, e é o que decide a tese. Duas tabelas, uma por tarefa: à esquerda a categoria, à direita a região, com três sistemas externos. Primeiro a comparação limpa. Em categoria, o conjunto fica pelo menos três vírgula zero seis pontos acima do POI-RGNN nos seis conjuntos, e o POI-RGNN é nativo da tarefa. Em região, fica acima do melhor externo de cada conjunto, também nos seis. Agora a ressalva de protocolo, porque os três não chegam em pé de igualdade. Só o HMT-GRN roda nos nossos dados, nas nossas partições e nas nossas inicializações. O STAN roda nas nossas partições mas constrói as próprias representações e as próprias sequências, e em dois conjuntos com partições incompletas. O ReHDM roda sob o protocolo publicado dele. E agora a coisa mais interessante do capítulo, e ela é contra eles, não a meu favor: o piso de Markov de primeira ordem, uma tabela de transição sem aprendizado nenhum, fica acima desses três sistemas na maioria dos conjuntos — acima do HMT-GRN nos seis. É por isso que eu trato o piso, e não os externos, como a referência que a próxima região tem de exceder. O conjunto excede o piso por quatro vírgula um a dez pontos. Os números vêm de quatro sementes por cinco partições; a dispersão e os intervalos estão no próximo slide, que é onde o veredito é decidido.

---

## O que foi conferido antes de aprovar

Cada coisa que sai da fala foi procurada **na tela**, não na memória:

| o que o corte remove | onde continua |
|---|---|
| *"os dois ganhos de região são resultados secundários, fora do plano"* | ✅ **na tela do slide 41**, verbatim |
| *"as cinco diferenças restantes são não resolvidas"* | ✅ **na tela do slide 41**, verbatim |
| a razão declarada da margem de dois pontos | ✅ na reserva, em *"Por que dois pontos?"* |
| **o asterisco honesto** | 🛑 **em lugar nenhum — por isso foi reposto** |

As duas primeiras cortam **bem**: aterrissam no slide onde o veredito de fato está, em vez de serem
ditas antes de os números aparecerem.
