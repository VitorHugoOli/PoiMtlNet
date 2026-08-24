# HANDOFF.md — o que um agente novo precisa saber para continuar

> **Escrito 2026-08-24**, manhã do dia do primeiro ensaio. **Defesa: sexta, 28/08, 10:00, remota
> (Google Meet).** Banca: Fabrício A. Silva (orientador/presidente), Clayson S. F. de Sousa Celes
> (ITA, externo), Alex Borges.
>
> **Quem aprova é o autor.** Não existe portão com o orientador para a apresentação — foi decisão
> dele, registrada. Não reintroduza.
>
> **Este documento é orientação, não lei.** A lei da estrutura é o `PLANO_FLUXO_DEFESA.md`; a lei
> da palavra são os três arquivos em `../` (`WRITING_LAW.md`, `GLOSSARY.md`, `AGENT_GUARDRAILS.md`).
> Onde este arquivo divergir deles, eles vencem.

---

## 1 · Leia nesta ordem

| # | Arquivo | O que é |
|---|---|---|
| 1 | `../CLAUDE.md` **§0** | **De onde vem cada número.** Se você escrever um número sem ler isto, vai errar — foi o erro mais repetido do projeto |
| 2 | `PLANO_FLUXO_DEFESA.md` | A lei da estrutura: arco, seis seções, minutos, ledger de de-duplicação, e **16 regras em §8** que o deck obedece |
| 3 | `SLIDES.md` | O slide-a-slide: tela (inglês), fala (português), proveniência por número, proibições |
| 4 | `APRESENTACAO_DEFESA_GUIDE.md` | Logística, Art. 23, e a análise quadro a quadro de uma defesa real do mesmo programa |
| 5 | `../wrapup/open_points/ARGUICAO.md` | 23 perguntas de banca + 8 "não foi medido", com resposta preparada |

---

## 2 · Estado, em 24/08

| Artefato | Estado |
|---|---|
| `PLANO_FLUXO_DEFESA.md` | **Fechado e aprovado.** 48 min, seis seções, somas conferidas nos dois níveis |
| `SLIDES.md` | **Completo.** 54 slides de deck + 46 de reserva. Passou por 5 personas revisoras |
| `slides/main.tex` + `main.pdf` | **Compila: 111 páginas, 0 erros.** É o deck vivo — **e é o único**; o `slides_ux/` foi descartado pelo autor. ⚠ **Overfull não é mais zero, e isso é intencional** — ver §4 |
| `../src/banca.pdf` | **Congelado** — o que a banca recebeu. md5 `5be69d1b`, 119 pp. **Nunca reconstruir** |
| `../src/dissertacao.pdf` | O build corrente, **com a errata do Resumo aplicada**. md5 `d7e85bb7` |
| Ensaio nº 1 | **Hoje**, com amigos |

### O que NÃO está feito

- ~~Varredura visual das 110 páginas~~ — **FEITA 2026-08-24.** 80 achados, e a causa era em boa
  parte **uma só**: um bug do template. Ver §4.
- ~~A grafia de "Pedro Maia"~~ — **RESOLVIDO 2026-08-24 pelo autor: Pedro Augusto Maia Silva.**
  Está no slide de agradecimentos (S55). Era a única fonte possível: o nome não aparece em nenhum
  artigo, no texto entregue, nem em lugar nenhum do repositório.

---

## 3 · A armadilha central deste projeto

> **Um instrumento dizer "limpo" não é evidência de que está limpo.** Isto aconteceu **cinco vezes**
> nesta sequência, sempre com o mesmo formato: uma checagem passou porque media outra coisa.

| # | O instrumento disse | A verdade era |
|---|---|---|
| 1 | `make check` saía **0** e os commits diziam "gates verdes" | Saía **2**. O 0 era do shell; o `make` retornava erro. **Leia o código de saída, não a saída** |
| 2 | Busca pela frase do Resumo no build `academico`: **0 ocorrências** | Aquele build **começa na página 10** e não contém o Resumo. Zero do instrumento, não do texto |
| 3 | Avisos "No current point" do poppler apareciam **também no PDF oficial** → "ruído benigno" | Eram o sintoma de um bug real. **Uma referência que compartilha o defeito não é controle** |
| 4 | `pdflatex` compilou o deck: **42 páginas, 0 erros**, texto extraível completo | Capa e todos os divisores saíam **em branco** — texto branco sobre gradiente que não desenhou. **Um teste de texto teria aprovado** |
| 5 | `make all` no deck: **0 erros, 0 overfull** | A legenda **sobrepunha** a tabela do veredito. Colisão não gera aviso |

**A regra que decorre:** para qualquer coisa visual, **valide por renderização**. Para qualquer
coisa contada, valide **sobre o artefato final**, não sobre o relatório de quem o produziu (foi
assim que um slide duplicado passou pela minha conferência de ledger — eu contei os arrays que os
redatores devolveram, não o arquivo montado).

---

## 4 · A varredura visual — feita, e o que ela ensinou

**Rodada em 2026-08-24 sobre as 110 páginas renderizadas: 80 achados** (4 bloqueantes, 46 maiores,
30 menores). O tipo dominante era *sem espaço de respiro* (32), seguido de inconsistência (22),
ilegível (13) e sobreposição (10).

**A maior parte da sobreposição tinha uma causa única, no template:** `\beamerboxesframed` fixava
`width=\textwidth`. Dentro de uma `column`, `\textwidth` continua sendo a largura do **frame
inteiro** — então todo bloco em duas colunas era desenhado mais largo que a sua coluna e passava por
baixo do bloco vizinho, que o cobria. Uma linha (`\linewidth`) matou a classe inteira.

**São agora cinco os bugs corrigidos só na nossa cópia do template**, todos com errata no próprio
`.sty`: `\pagewidth`→`\paperwidth`; o `\autotocframe` que vazava o argumento; o `\decorationnet`
que nunca desenhava; o `width=\textwidth` acima; e o `\vskip-2mm` do `\titleframe`, que cortava o
topo dos dois cartões de logo na capa.

> ⚠ **`Overfull` deixou de ser zero de propósito — não "conserte" isso empurrando de volta.**
> O deck tinha 0 overfull porque os redatores usavam **31 `\vspace` negativos**. Eles não criavam
> espaço: puxavam o conteúdo para cima do elemento anterior. O log ficava limpo e a tela ficava
> sobreposta. Removidos, o LaTeX passou a declarar a verdade. Os estouros que restam foram
> **verificados por renderização** e ficam dentro da folga do beamer. **Se você reintroduzir
> `\vspace` negativo para zerar o log, você recria exatamente o defeito que esta varredura corrigiu.**

**A regra que decorre, e que vale para a próxima:** o log de compilação **não vê** colisão de blocos,
e um log limpo pode ser sintoma de cramming, não de saúde. Valide por renderização:

```bash
cd slides && pdftoppm -r 95 -png main.pdf /tmp/deck   # e olhar página a página
```

## 4b · A referência de densidade: a defesa do Henrique

**É uma GRAVAÇÃO, não um PDF** — e o caminho já estava no `APRESENTACAO_DEFESA_GUIDE.md:131,:317`.
Eu procurei um PDF, não achei, e escrevi "não está no repositório" **sem reler o meu próprio guia**.
Falha nº 2 do §3, outra vez. O arquivo:

```
/Volumes/linux/VIDEO/Screen Recording 2026-07-08 at 10.02.56.mov     # 7 GB, 1h01, 4096×2304
```

É a **sua tela** durante a defesa, então o Meet só está em foco parte do tempo (≈0–15 min e
≈35–60 min). Extraia com `ffmpeg -ss <seg> -i <arq> -frames:v 1` (o `-ss` ANTES do `-i` = seek por
keyframe, barato num arquivo de 7 GB).

**Ele usa o MESMO template NESPeD**, também é coletânea, também seis seções na barra. **64 slides em
≈48 min** — praticamente o nosso ritmo (54 em 48). O que difere não é a quantidade de slides:

| | Henrique | nosso deck |
|---|---|---|
| palavras na tela, mediana | **~25–30** | **119** |
| máximo | ~70 | **237** |
| slides com frase completa | poucos | **54 de 54** |

⚠ **As duas medidas não são diretamente comparáveis, e o erro faz cortar demais.** A contagem do
nosso deck lê o corpo do LaTeX e **não vê texto dentro de figura**; OCR sobre o vídeo **vê**. O slide
"só uma figura" dele dá 82 no OCR e ~9 de corpo. Compare corpo com corpo.

**E a contagem sozinha engana.** O slide mais denso dele tem ~70 palavras — é *uma pergunta de
pesquisa num bloco*, que a plateia lê de uma vez. O nosso de 237 são *quatro marcadores de ressalva*,
que a plateia lê quatro vezes e perde a fala. Mesma contagem, funções opostas. **Corte pela função,
não pelo número.**

> 🔴 **O deck viola a regra 10 do PLANO em 54 dos 54 slides.** A regra diz *"Marcadores por
> palavra-chave, **nunca parágrafos**"*. Medido: todo slide do deck tem ao menos uma frase completa
> longa na tela. Quem for enxugar **não está sobrepondo o plano — está aplicando uma regra que ele já
> continha**. Não é preciso licença para cortar; o mandato existe. Mas ver §4c antes.

## 4c · O que NÃO pode sair da tela, e o protocolo para o que pode

Do `PLANO_FLUXO_DEFESA.md` §8, classificado. **Mandato de tela** (cortar é regressão): **R1**
navegação/seis `\section` · **R5** carimbo `Next-POI Prediction = next category (Def. 2.7)` em arte
dos Caps. 3/4 · **R8** rodapé `pós-submissão` em todo slide da série B · **R10** numeração dos slides
· **R12** slides de transição, que o plano proíbe explicitamente que um corte de tempo remova ·
**R13** a contribuição duas vezes, **com redação idêntica** · **R14** "Apêndice B" sempre com o nome
do volume.

**Governam a palavra, não o volume** (comprimir é seguro): R2, R3, R7, R9, R11, R15. **R16 e R10
mandam cortar** — são aliados. **R4** (ledger) vive no `SLIDES.md`, **não na tela**.

**A armadilha é a R6**, *"ressalva antes da manchete, sempre"*. Foi ela que produziu os 119: os
redatores puseram a ressalva **na tela** porque a R6 exige que ela acompanhe o resultado. A leitura
que o S47 testou: a R6 exige que a ressalva **acompanhe**, não que esteja **projetada** — se a
manchete está na tela e a ressalva é dita no mesmo fôlego, está satisfeita. **Mas só com este
protocolo, e ele não é opcional:**

> Para cada cláusula que sai da tela: **(1)** localize-a no destino (a fala do MESMO slide, ou um
> slide de reserva) com grep; **(2)** se não existir no destino, **ela não sai** — escreva-a na fala
> primeiro; **(3)** registre no bloco do `SLIDES.md` onde ela ficou.
>
> Sem o passo 2 isso vira perda de honestidade com aparência de design. O **S47** é o molde: três
> itens saíram, cada um conferido no destino (B1-1, S3, S50) antes de sair, e a nota está no bloco.

## 4d · A fala: quem é canônico

**`SLIDES.md` é a fonte; o `% FALA:` do `main.tex` é o espelho.** Estabelecido por medida:
o `SLIDES.md` nasceu primeiro (`f9b8f82e`), o portão de revisão operou nele (`d1491956`), o deck veio
depois (`5c2ee121`), e a correção dos revisores **flui SLIDES.md → main.tex** (o texto corrigido está
no `.tex`, o antigo tem zero ocorrências lá). Só o `SLIDES.md` carrega `LEDGER`, `Proveniência` e
`Nunca dizer`.

Deriva atual entre as duas cópias: **120 palavras em 11.259 (1%)**. Pequena, mas real.

**Corte sempre no `SLIDES.md` primeiro, depois propague.** Nunca o contrário: o `% FALA:` não
renderiza, então um erro lá fica invisível até alguém abrir o `.tex`. **E não apague o `% FALA:` do
`main.tex` para "resolver" a duplicação** — ele existe para que quem edita o slide veja a fala na
mesma tela, e é o que mantém tela e voz sincronizadas.

⚠ **O deck não cabe em 50 min, e isso é independente da densidade de tela.** A trilha de fala do deck
principal tem **8.935 palavras** = ~64 min a 140 ppm; caberia em 48 min só a **186 ppm**, dizendo IC
e correção de Holm. O desencontro é **concentrado**, não espalhado: S43–S46 e S48 somam ~1.400
palavras de fala contra ~7 min orçados. Os `Tempo:` por slide foram estimados por peso de assunto e
**nunca reconciliados** com o texto que os preenche. **Decisão de corte de fala é do autor.**

## 5 · Decisões já tomadas — não reabra sem o autor

| Decisão | Ruling |
|---|---|
| **O vazamento do v18 sai da narrativa** | A dissertação **não o cita** (medido: `forward-only` tem zero ocorrências no PDF entregue). Narrá-lo poria na fala algo que a banca não acha no texto que julgou. **O que fica** é a direcionalidade, dita como **princípio de projeto**, nas palavras do próprio Cap. 5 |
| **Ordenação por LINHAGEM** | §2 só recebe o que é transversal **e não faz parte da herança que o arco narra**. MTLnet fica no Cap. 3 porque *"a mesma arquitetura, sem alterar uma linha"* é o argumento de controle do Cap. 4 |
| **Protocolo estatístico em 5.4**, não na §2 | Só o Cap. 5 o usa. Idem Acc@10 e joint-best |
| **Seções 3–5 levam o título do artigo**, não o veículo | Barra de navegação = a linhagem: MTLnet · ST-MTLNet · Check2HGI |
| **Seção 1 é genérica** | Sem "sete categorias", sem *mahalle*, sem nomes de estado — **exceto** a frase do veredito, que nomeia Flórida, Texas e Califórnia |
| **Karpathy não vai na conclusão** | Vai para a série B, como contexto ao oferecer o P1. *"Tasks fight for capacity"* é exagero na direção **oposta** à que a posição do tronco protege |
| **Q13/Q14/Q15** | Slides prontos para resposta **oral** |
| **NSO-46 fechado pela premissa inválida** | O parágrafo **não chega ao leitor** (zero aparições nos três builds). ⚠ **Não** por "não há vazamento no Cap. 3" — a auditoria mediu canal indireto e o confirmou por intervenção causal |

---

## 6 · Frases que não podem ser ditas

Estão espalhadas nas linhas `Nunca dizer:` do `SLIDES.md`. As que mais custam:

1. **`empata` / `matches` / `ties` / `em todos os conjuntos`** — nos dois eixos. O veredito é
   supera em **três células** e nada mais.
2. **A frase retratada:** *"o ganho vem da representação hierárquica e não da injeção de features"*.
   O controle refeito conclui que **a frase depositada está errada na direção**. Um revisor a
   encontrou no deck, entregue como fala, com os números da mesma tela a refutando.
3. **`meio ponto` no eixo de região.** É o limite da **categoria**. A margem derivada de região é
   **1,287 pp** — citar um no outro é exagero de três vezes.
4. **`o DGI não vaza`.** A auditoria mediu que vaza, de forma indireta. O que se diz é a diferença
   de mecanismo: **consulta exata no Cap. 4, média diluída a um salto no Cap. 3**.
5. **Qualquer macro-F1 de próxima categoria entre 54 e 80** — é número vazado pré-v18. A faixa
   entregue é **30–38**. ⚠ Escopado de propósito: Acc@10 de região vive em 59–77 e é legítimo.
6. **Misturar um número do `mtlcheck` com um da dissertação** na mesma frase. Protocolos diferentes.

---

## 7 · Como construir

```bash
cd slides
make check   # passe único: valida se compila. A CONTAGEM DE PÁGINAS DELE NÃO É A FINAL
make all     # 3 passes + bibtex. Use este para qualquer número que vá ser citado
```

- **O motor é `xelatex`.** `nesped.sty` carrega `fontspec`. Sob `pdflatex` o build "passa" e as
  telas com fundo saem **em branco** (caso 4 do §3).
- **O template tem cinco bugs corrigidos** só na nossa cópia — os três antigos
  (`\pagewidth`→`\paperwidth`, o `\autotocframe` que vazava o argumento, o `\decorationnet` que
  nunca desenhava) e os dois de 24/08 (`width=\textwidth`→`\linewidth` em `\beamerboxesframed`, e
  o `\vskip` do `\titleframe`). Cada um tem errata datada no `.sty`. O original de terceiros não
  foi tocado.
- **A série B usa `\miniframesoff`.** O número do frame **congela** ali — por isso cada slide B
  carrega o rótulo no conteúdo, não no rodapé.
- **Não rode `make` na pasta `../src/`** sem pensar: cinco alvos sobrescrevem o `dissertacao.pdf`,
  incluindo o `make` pelado. O `banca.pdf` está a salvo deles.

---

## 8 · Ambiguidades encontradas hoje, que precisam da palavra do autor

Trabalho não commitado apareceu depois do meu último commit. **Não sei quem o fez e não presumi.**

1. ~~**`slides_ux/`**~~ — **RESOLVIDO 2026-08-24: o autor decidiu ficar no `slides/`.** O
   `slides_ux/` é uma variante com fontes customizadas (Petrona + IBM Plex), 35 frames, incompleta.
   **Não é a direção.** Fica em disco como referência; **não construa a partir dele**.
2. ~~**`SLIDES_serieB.md`**~~ — **RESOLVIDO 2026-08-24: apagado, com a decisão do autor.** Era o
   rascunho do agente que escreveu a série B, **pré-revisão**. Estabelecido por medida, não por
   memória: dos 46 blocos, 43 têm corpo idêntico ao do `SLIDES.md`, **nenhum código existe só nele**,
   e os 3 que diferem são casos em que o `SLIDES.md` é a versão **posterior e mais cuidadosa** —
   inclusive `B3-7`, onde o rascunho ainda diz *"which **sits inside** the seed spread"*, a
   afirmação que `d1491956` ("os dois bloqueantes que os revisores acharam") trocou por *"of the
   order of"*. Manter era manter em circulação um texto que a revisão derrubou. **Recuperável:**
   `git show c7f0fd1f:articles/dissertacao/presentation/SLIDES_serieB.md`.
   **`SLIDES.md` é o canônico da série B — não recrie um segundo arquivo para ela.**

3. **`slides/_font_test.tex`** — teste de fonte, provavelmente descartável.

---

## 9 · Onde o conhecimento mora

| Assunto | Arquivo |
|---|---|
| De onde vem cada número entregue | `../CLAUDE.md` §0 |
| O que está aberto na dissertação | `../wrapup/open_points/LACUNAS.md` — 42 itens, 17 abertos |
| Perguntas de banca com resposta pronta | `../wrapup/open_points/ARGUICAO.md` |
| O que veio depois do envio | `../wrapup/` — a fronteira é declarada no README |
| O que ficou para trás | `../archive/` — **nada ali é fonte de nada** |
| A reescrita do sistema experimental | `../wrapup/NEW_VERSION.md` (`mtlcheck`) |
| Correções de nomenclatura estatística | `/Users/vitor/Desktop/mestrado/mtlcheck/mtlcheck/docs/NOMENCLATURE.md` |

---

## 10 · Como este autor trabalha

- **Ele aprova antes de qualquer mudança crítica.** Apresente opções com recomendação e espere.
- **Ele quer notificação** quando você precisar de decisão — não deixe pergunta parada em terminal.
- **Ele corrige o rumo com boas razões.** Duas das melhores mudanças do plano vieram de objeções
  dele: tirar o vazamento da narrativa, e recusar mostrar a Tabela 9 duas vezes.
- **Verifique a premissa dele antes de agir.** Uma vez a lembrança dele contradizia o registro (o
  vazamento do Cap. 3), e o registro existia porque ele mesmo pedira auditoria independente. Dizer
  isso claramente foi mais útil do que concordar.
