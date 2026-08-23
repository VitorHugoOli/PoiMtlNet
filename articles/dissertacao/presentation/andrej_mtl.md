https://youtu.be/IHH47nZ7FZU?si=0nPuGwn7ibp1HJjn


Por volta do minuto 7, Andrej Karpathy aborda a complexidade de desenhar a arquitetura para redes de Múltiplas Tarefas
(Multi-Task Learning), pois ao contrário de uma rede focada em apenas uma tarefa, em MTL não é óbvio definir o quanto de
compartilhamento de parâmetros deve haver entre as tarefas, já que algumas se ajudam e outras acabam competindo (
"brigando") entre si pelos recursos da rede.

Aqui estão as falas exatas (quotes) onde ele ilustra essa complexidade:

"but it gets a bit more complicated because we don't just have one task we have multiple tasks so exactly how do you
layout this architecture how much feature sharing is there maybe the first three layers are shared but from then on they
split off into a different networks how much capacity do you allocate to any single different sub piece of this network
it's it's not really obvious how you actually go about exploring all these different choices" [07:23
]

"all these different tasks they're typically very heterogeneous so some of them will help each other and some of them
will hurt each other and so there's complicated binary matrix here in this paper of which tasks seems to be helping each
other in which tasks I seem to fight" [07:46
]

Para ilustrar abordagens que tentam lidar com esses desafios no design automático de arquiteturas e no agrupamento de
tarefas, ele cita e mostra imagens de dois artigos de pesquisa recentes na época:

Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation (Chenxi Liu et al., CVPR 2019) –
Citado por volta de [07:03
] como um exemplo de busca automática da arquitetura ideal (NAS).

Which Tasks Should Be Learned Together in Multi-task Learning? (Trevor Standley et al., ICML 2020) – Citado por volta
de [07:46
], este é o artigo que mostra a matriz complexa indicando quais tarefas melhoram o desempenho uma da outra quando
treinadas juntas e quais acabam prejudicando ("lutando" contra) o aprendizado.

---
Sim, na verdade, a maior parte da apresentação dele é justamente sobre o quão doloroso e complexo é fazer o Multi-Task
Learning (MTL) funcionar na prática em larga escala. Ele expande bastante essa discussão sobre se o MTL vale a pena (os
trade-offs da arquitetura) e o grande problema da **competição pela capacidade (parâmetros) da rede**.

Aqui estão os principais momentos onde ele aprofunda essa discussão:

### 1. O Dilema da Arquitetura: Compartilhamento vs. Redes Independentes

Por volta de, ele discute os dois extremos do design de arquitetura e por que o MTL se torna uma necessidade, mesmo com
seus problemas:

* **Extremo 1 (Redes Independentes para cada tarefa):** Se cada tarefa tiver sua própria rede, é ótimo para a equipe de
  engenharia porque o trabalho fica isolado (se você mexe em uma, não quebra a outra). No entanto, **não há
  compartilhamento de features**, o que prejudica tarefas com poucos dados, e o custo computacional no carro seria
  inviável.

> *"this will be very expensive at test time we have a finite compute budget on the car... moreover there's no feature
sharing and it's kind of a setup and so if your task one of the tasks as an example might not have enough data then
you're going to actually over fit"*

* **Extremo 2 (MTL Total - Um Backbone compartilhado com cabeças leves):** É muito mais barato e eficiente rodar no
  carro, mas as tarefas começam a "brigar" entre si pela capacidade da rede, e o trabalho da equipe fica totalmente
  acoplado.

> *"this would be significantly cheaper at test time because this backbone is basically multitasking... but there are
some downsides so as an example all these tasks will now fight for the same shared capacity sometimes they fight
sometimes they actually help each other"*

### 2. Quantidade de Parâmetros e Regularização por Tarefa

Mais adiante, ele discute como a quantidade de dados dita a quantidade de parâmetros que você pode alocar (o tamanho da
"cabeça" da rede) para uma tarefa específica dentro do modelo compartilhado. Você não pode simplesmente dar parâmetros
iguais para tudo:

> *"if I have a some task that has very few examples then of course I can't afford to train a very thick head for it or
I can't afford to use too many parameters there or I might want to regularize that piece of the network much more
strongly"*

### 3. A Competição Real por Capacidade (Parâmetros Finitos)

Na parte final da palestra, ele ilustra como o limite de parâmetros e o compartilhamento afetam a dinâmica da própria
equipe de inteligência artificial. Como **a capacidade da rede é finita**, quando um engenheiro tenta melhorar a sua
própria tarefa (ex: detectores de semáforo) aumentando a taxa de amostragem ou o peso da função de perda (loss
function), ele acaba "roubando" parâmetros e capacidade de aprendizado das outras tarefas (ex: objetos em movimento).

> *"what's going on here because their tasks will be starved of resources and the capacity of the network is finite and
so suddenly this is not okay."*

Ele conclui confessando que a comunidade acadêmica ainda não tem uma boa solução matemática ou teórica para alocar essa
capacidade finita de parâmetros de forma justa entre tantas tarefas heterogêneas:

> *"the higher point being that there's finite capacity to go around and a lot of people are trying to simultaneously
get their their tasks to work well... I have to somehow like allocate capacity to the tasks but there's no easy ways of
doing that... I don't really have language to describe how to correctly allocate capacity to tasks and interesting
weights to them"*

Em resumo, ele mostra que o MTL é **estritamente necessário** por questões de eficiência de processamento no veículo
(orçamento computacional) e para o compartilhamento de features base, mas que gerenciar a quantidade finita de
parâmetros contra 100 tarefas simultâneas é um pesadelo logístico, matemático e até de convivência entre a equipe.

---

2. O Problema da Competição de Gradientes (As Tarefas "Brigando")
Karpathy mencionou que ajustar a loss de uma tarefa pode destruir outra. A academia formalizou isso analisando os gradientes durante o backpropagation: se os gradientes de duas tarefas apontam para direções opostas, ocorre a interferência.

Gradient Surgery for Multi-Task Learning (PCGrad) (Yu et al., NeurIPS 2020)
Este artigo formaliza a "briga" geométrica entre as tarefas. O método projeta o gradiente de uma tarefa na normal do gradiente de outra tarefa sempre que eles entram em conflito (quando o produto escalar entre eles é negativo). Na prática, impede que uma tarefa apague o que a outra acabou de aprender nos parâmetros compartilhados.

GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks (Chen et al., ICML 2018)
Resolve exatamente o problema que Karpathy relatou sobre engenheiros multiplicando a loss por 10 manualmente. O GradNorm ajusta dinamicamente os pesos de cada tarefa (task weights) durante o treinamento, garantindo que todas as tarefas treinem em taxas semelhantes e que tarefas mais fáceis não dominem a capacidade da rede de forma prematura.