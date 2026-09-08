# Armadilhas de medição — como um agente se engana a medir

Destilado de um dia de execução (2026-08-27/28) em que **cinco instrumentos diferentes deram
resposta verde e errada**. Não é sobre slides. É sobre a classe de erro em que a ferramenta
funciona, não falha, não avisa — e responde a **outra pergunta**.

Cada item abaixo tem número medido, não impressão.

---

## 1 · O padrão-mãe: verde que responde a outra pergunta

> **Um resultado verde só vale se você souber que o vermelho podia ter aparecido.**

Se nunca viu o instrumento falhar, não sabe se ele mede. Antes de acreditar num verde: injete um
defeito conhecido e confirme que ele fica vermelho.

**Casos do dia:**

| instrumento | dizia | media na verdade |
|---|---|---|
| contagem de `Overfull` | "0 problemas" | nada em `columns` — **coluna transborda em silêncio** |
| detector de frame por string | "✅ cabe" | **o índice**, não o slide (rótulo de botão é igual ao código) |
| sonda com `str.replace` | "✅ cabe" | um ficheiro **sem** a alteração (replace não encontrou → no-op) |
| régua de tinta | "50 páginas carregadas" | **o número do frame**, que está lá por desenho |
| relógio da fala | "48:23" | uma fonte **dessincronizada** — o real era 49:58 |

---

## 2 · Sucesso PARCIAL é mais perigoso que falha total

Um aplicador de citações reportou **`39 aplicadas, 6 falhas`**. As 39 estavam em frames
**deslocados** — o contador de frames escapava a alguns e desalinhava tudo.

> **Uma falha total faz parar. Seis falhas convencem de que só há seis problemas** — e mandam olhar
> exatamente para o sítio errado.

**Regra:** quando um instrumento reporta sucesso parcial, **valide os sucessos**, não as falhas.

---

## 3 · Toda substituição leva asserção de contagem — inclusive em sonda

`str.replace(a, b)` devolve a string **intacta** quando `a` não existe. Sem erro, sem aviso.

```python
if s.count(a) != n:            # n esperado, explícito
    sys.exit(f"ABORTADO: {s.count(a)} != {n}")
s = s.replace(a, b)
```

⚠ **Eu aplicava isto no código de aplicação e não nas sondas** — e foram as sondas que mentiram.
A sonda mede o artefacto que vai decidir um corte: ela precisa da mesma disciplina.

---

## 4 · O comentário é invisível para o leitor e visível para a ferramenta

Três defeitos distintos, mesma raiz:

- um contador de `\begin{frame}` contava os que estavam **dentro de comentários** que explicavam a
  sintaxe → **2 aplicadas, 43 falhas**;
- substituir um intervalo de frames **apagou 57 palavras de fala** que viviam em comentários entre
  eles — e nenhuma verificação de PDF apanha, porque comentário não renderiza;
- uma varredura de termos proibidos sobre-reportava porque os comentários **citam verbatim** o que
  foi corrigido.

**Regra:** decida explicitamente se a ferramenta lê ou ignora comentários. **Nunca por omissão.**
E ao substituir um intervalo, **conte os comentários load-bearing antes e depois**.

---

## 5 · A unidade de custo é a LINHA, e linhas são quantizadas

Quatro medições contra a intuição de caracteres:

```
"+8 palavras"                        -> +27,6 pt   (uma instrucao de quebra forcava 2 linhas)
"corta as caixas 1 e 2 antes da 3"   ->  zero      (altura de minipage = MAXIMO das tres)
ordem de sacrificio numa coluna      ->  zero      (cortava a coluna MAIS CURTA)
citacao de 251 -> 84 caracteres      -> +42,4 pt   IDENTICO ao original
```

> **Caracteres não compram nada até removerem uma quebra.** E em layout de colunas ou caixas,
> **só a mais alta paga** — cortar da outra é trabalho perdido, por mais conteúdo que se sacrifique.

**Corolário de fluxo:** quem decide *o que* cortar não consegue prever *onde* o corte tem efeito.
Meça primeiro, corte depois.

---

## 6 · Nunca procure uma frase num texto extraído

`pdftotext` (e qualquer extractor) quebra linha na largura da coluna. **Um padrão que atravesse a
quebra dá ausente** — e ausente é sempre o resultado que parece confirmar que está tudo limpo.

> **Procure palavra única.** O falso negativo é sempre na direção de "está bem".

⚠ Aconteceu três vezes no mesmo dia, incluindo a mim **a verificar a minha própria aplicação**:
procurei a frase inteira, deu zero, e ela estava lá.

---

## 7 · Uma âncora ambígua devolve a primeira ocorrência, não a certa

Buscar por um rótulo (`B6-6`, `B-P1`) encontrava **o botão de um índice**, porque o índice vem
antes no ficheiro. Resultado: medições do frame errado reportadas como boas — **três vezes**.

**Regra:** ancore em algo estruturalmente único (aqui, o subtítulo **dentro** do intervalo do
frame), nunca numa string que também aparece em listas, índices ou comentários.

---

## 8 · Remover metade de um par simétrico não remove metade do efeito

Tirar só um `\vfill` de um par piorou de **+6,1 para +22,6 pt**: sem par, o outro empurrou tudo.
Vale para qualquer coisa que exista aos pares — margens, fills, delimitadores, guardas.

---

## 9 · Uma cópia é outro artefacto no instante em que a original muda

Duas ocorrências no mesmo dia:

- um agente extraiu o PDF para cache e raciocinou sobre ele **enquanto o outro editava** — concluiu
  que um slide tinha perdido o número do frame; não tinha, o instantâneo é que era anterior;
- um defeito foi reportado a partir de um PDF que vinha de uma **cópia de medição**, não do ficheiro
  de record.

**Regra:** ao reportar, diga **de que artefacto** e **de que momento**. E antes de escrever num
ficheiro partilhado, confirme pelo `stat` e por marcas conhecidas que ele ainda é o que você leu.

---

## 9b · O caso mais caro: um ficheiro protegido que o `git` dava como seguro

O autor mandou congelar uma cópia do trabalho. Ela ficou **gitignorada e não rastreada** — invisível
a `git status`, e um `git clean -xfd` apagava-a **sem diff e sem aviso**. O `.gitignore` da pasta
documenta que **o mesmo acidente já tinha acontecido** semanas antes, e deixa a instrução para não
repetir. Nunca executada.

⚠ **E o instrumento que se usaria para verificar mente:**

```
$ git check-ignore -v <caminho-ja-rastreado>
(nada)                                        exit=1   -> "nao e' ignorado"

$ git check-ignore -v --no-index <mesmo caminho>
.gitignore:19:*.pdf   <caminho>               exit=0   -> a regra APANHA-O
```

> **`git check-ignore` consulta o índice.** Para um caminho já rastreado responde *"não ignorado"* —
> que é verdade sobre o **estado**, e falso sobre a **regra**. Para testar a regra: `--no-index`.

**Sequência que funciona**, e cada passo verifica o anterior:

```
git add -f <caminho>                      # a regra continua la'; o -f e' que a vence
git show HEAD:<caminho> | md5             # o blob no git
md5 -q <caminho>                          # o disco
# os dois md5 TEM de bater -- so' isso prova que o conteudo foi mesmo guardado
```

**A lição de fundo:** *"eu criei uma cópia de segurança"* e *"a cópia está segura"* são afirmações
diferentes. A primeira é sobre uma ação; a segunda é sobre um estado, e precisa de ser **medida**.

---

## 10 · Um instrumento que acerta pela razão errada encerra a investigação

Procurando confirmar uma margem de 2 pp, encontrou-se `tost=0.02` no código. **O número batia — e a
chamada era de outra etapa.**

> **Um instrumento que erra chama atenção. Um que acerta por acidente fecha a pergunta.**

O que salvou foi alguém perguntar *"isto é mesmo esta etapa?"* **depois** de já ter a resposta certa.

---

## 11 · Declare o que a ferramenta NÃO alcança

Uma sonda que não localizava a página caía para "procurar em todas as páginas" — o que transforma
qualquer palavra comum num verde sem valor.

> **Uma ferramenta que sabe dizer `não sei` vale mais que uma que reporta limpo.**
> Uma que declara o seu alcance vale mais que uma que parece completo.

---

## 12 · Uma alegação vive em várias superfícies

Remover uma afirmação falsa de **um** sítio deixa-a viva nos outros. Um caso concreto: o nome saiu
da tela e **ficou na fala**, onde a frase era ainda mais categórica — o autor diria em voz alta
exatamente o que a tela deixara de afirmar.

**Regra:** ao remover uma **alegação** (não uma palavra), enumere primeiro **onde ela vive**.
Aqui eram três superfícies; a instrução cobria uma.

---

## 13 · O ponteiro que apodrece, e as duas edições que o produzem

Um documento superado engana quem o lê. **Um guarda superado engana quem o lê _e_ quem o obedece** —
ou gera desconfiança inútil de um ficheiro que está bem, ou manda "corrigir" o que está certo.

Em cinco horas de 2026-09-08 este ficheiro ganhou três casos, por três sessões diferentes.

**(a) A edição que não lê o que emenda.** Acrescentar uma linha a um cabeçalho sem ler a de cima
produziu duas afirmações contraditórias no mesmo bloco, **criadas pela mesma edição**. Não é
decadência: nasce partido.

**(b) A medição no ficheiro homónimo errado.** Existem dois `NORTH_STAR.md` neste repositório —
`docs/` (a receita campeã de MTL) e `articles/dissertacao/` (a tese e o mapa dos capítulos), 527 e
447 linhas, documentos diferentes. Uma sessão grepou o do `docs/` para responder a uma pergunta
sobre o da dissertação, concluiu **"zero ocorrências"**, e escreveu essa medição num guarda. As
frases existiam, **duas vezes cada**, no irmão — e correctamente tarjadas `[SUPERADO]`.

> ⚠ **Esta é a variante que nenhuma disciplina sobre grafias apanha.** As outras oito falhas desta
> família em 2026-09 foram buscas que **não podiam** encontrar (grafia errada, `[colchetes]` como
> classe de caracteres, `--include` sem aspas, um match cortado por `cut`). Esta é **a busca certa no
> sítio errado**. Contado no dia: **241** referências a `NORTH_STAR.md` sem caminho contra 67
> qualificadas. Os dois ficheiros levam agora tarja a dizer que o irmão existe.

**(c) O ponteiro que apodrece em sessenta minutos.** Um `canon.py:26` escrito à tarde apontava para a
linha 27 uma hora depois, porque **a própria edição que o escreveu** empurrou a constante. Escrito
por quem acabara de descobrir esta falha exacta.

**A regra, e é sobre onde se ancora:**

| âncora | dura? |
|---|---|
| número de linha | **não** — move-se com a edição seguinte, inclusive a sua |
| contagem que alguém tem de manter | **não** — a mesma classe (ver §3) |
| nome de ficheiro sem caminho | **não**, se houver homónimo — e há |
| **nome de símbolo** (`DEFAULT_CANON`, `\finalbuildfirstpage`) | **sim** — foi a que sobreviveu a todas as conversões |
| nome de estudo, geração, data, texto de um marcador | **sim** |

**Prefira o símbolo à frase.** Uma âncora em prosa move-se com cada errata; um nome de símbolo só
muda quando o código muda, e aí quebra ruidosamente.

**E verifique a âncora em cada ficheiro que ela reclama, não uma vez.** Um ponteiro que resolve num
sítio não resolve nos outros que o citam.

**O corolário que custou o dia:** ao substituir um guarda podre, **a medição que se escreve no
substituto é ela própria um guarda** — e pode ser pior que o original. Aqui foi: o guarda antigo só
podia causar desconfiança; o novo podia causar uma **deleção** de registo histórico marcado.

---

## 14 · A alegação citada de cor, e a reprodução que custa minutos

As nove falhas anteriores desta família tinham todas **um erro visível**: o ficheiro errado, a pasta
errada, a grafia errada, um `--include` sem aspas, um match cortado por `cut`. Esta não. **O padrão
estava bem construído — para a frase que quem o escreveu tinha na cabeça.**

**O caso (2026-09-08).** Um aviso apontava cinco linhas de um ficheiro como carregando a escada
superada. Ao verificá-lo, uma sessão procurou `category everywhere|region at four|\+28|place-level`
e encontrou **2 de 5**, concluindo que o ponteiro *nascera errado* em vez de ter apodrecido — e
propôs corrigir o diagnóstico registado.

O ficheiro escreve a mesma alegação de quatro maneiras:

| a linha diz | o padrão procurava |
|---|---|
| `category **outperforms** everywhere` | `category everywhere` |
| `region outperforms at **4 of 6**` (dígito) | `region at four` |
| `outperforms both dedicated models` | — |
| `category everywhere, region at four of six` | ✔ apanhada |

Recontando com as variantes: **4 de 5**. O ponteiro nasceu a ~80% e apodreceu com o ficheiro a
mexer-se, que era o diagnóstico original. **A correcção proposta teria substituído um diagnóstico
certo por um errado.**

> **A regra:** uma alegação não tem forma canónica. Citá-la de memória e procurar essa citação mede
> a memória de quem procura, não o ficheiro. Antes de concluir "não está lá", enumere **as formas em
> que a coisa se diz** — verbo intercalado, número por extenso e em dígito, sinónimo, ordem trocada.

### E o corolário, que é o mais valioso deste ficheiro

Esta refutação só existiu porque, ao receber a alegação de outra sessão, **pediu-se o commit e o
comando em vez de se aceitar o resultado** — e depois **correu-se**. A reprodução confirmou cada
passo e derrubou a conclusão: foi ao repetir a medição que o buraco no padrão apareceu.

> **Uma alegação de um par vale o que valer a sua reprodução, e reproduzir custa minutos.**

Não é desconfiança. É que ambas as sessões estavam de boa-fé e competentes, e mesmo assim a
correcção errada teria entrado **assinada por duas** e ficado meses. Nenhuma leitura atenta a
teria apanhado; só a re-execução.

*Caso e formulação: sessão `mobiwac`, que trouxe o commit, aceitou a refutação e retirou a
afirmação. A décima desta família em quatro dias, entre quatro sessões — nenhuma isenta.*

### A décima-primeira, no parágrafo a seguir a escrever as dez

Ao fechar a passada final escrevi *"nada meu por commitar"*, medido com `git status --porcelain`.
Verdadeiro — e a pergunta era outra. **Commitado e empurrado são estados diferentes**, e três
commits meus estavam a meio caminho. O `git status` responde sobre a **árvore de trabalho**; a
pergunta "está fechado?" é sobre o **remoto**.

    git status --porcelain          →  a árvore está limpa?
    git log origin/main..HEAD       →  há commits por empurrar?
    git status -sb                  →  "ahead N" na primeira linha

Não é uma variante nova: é §14 outra vez, **o comando certo para a pergunta ao lado**. Aconteceu no
mesmo dia a duas sessões — a outra teve dois commits de guardas por empurrar, e foi o autor a vê-los
no IDE antes de qualquer agente. **Antes de dizer "fechado", corra o segundo comando, não o
primeiro.**
