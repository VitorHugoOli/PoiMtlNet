# Advisor Feedback — Deck ST-MTLNet @ CoUrb 2026

> Avaliação sênior do deck de 9 min para o X Workshop de Computação Urbana (SBRC 2026, Praia do Forte, 25/05/2026). Apresentador: Vitor H. O. Silva (2º autor).

---

## 1. Visão geral em 3 linhas

O deck está **funcionalmente pronto** — estrutura de 10 slides, orçamento de tempo cabe (9:10 / 9:30), figura-mãe definida, número-âncora claro (+20–24 pp), limitação admitida (Travel). **Riscos principais**: (i) slide 4 está sobrecarregado de blocos novos (3 encoders + MTLNet + FiLM + ResBlocks) para 1:30 — vai estourar; (ii) ainda não há uma "frase-gancho" que ressoe com plateia de **redes/IoT urbano**, que é metade da sala. (iii) o título "ST-MTLNet" não comunica utilidade — para CoUrb conviria amarrar à mobilidade urbana já na capa. Com ajustes pontuais, é apresentação sólida.

---

## 2. Cinco recomendações específicas

**R1 — Reescrever o slide 4 em duas camadas, não uma.** *O que mudar:* dividir o slide 4 em **4a (figura limpa da arquitetura)** + **4b (apenas a equação de concatenação 64+64+64 → 192)**, ou então retirar dos bullets a menção a FiLM/ResBlocks (esse detalhe não muda entre baseline e proposta). *Por quê:* 1:30 não basta para apresentar três encoders novos + a espinha MTLNet inalterada; o slide vira "wall of text" e a figura perde força. *Onde:* Slide 4 (Proposta).

**R2 — Antecipar o número-âncora para o slide 1 ou 2, não esperar até o 7.** *O que mudar:* incluir uma linha de "preview de resultados" na capa ou logo após o contexto: *"Resultado deste trabalho: +20–24 pp F1 em 21/21 combinações de categoria."* *Por quê:* plateia de workshop ouve 4 talks seguidos; quem perde os primeiros 30 s do hook deixa de ouvir o gancho até 6:35 do talk. Resultado-âncora cedo prende atenção. *Onde:* slide 1 (subtítulo) ou slide 2 (faixa inferior).

**R3 — Reformular slide 3 com analogia visual da "decomposição".** *O que mudar:* substituir a lista de 3 X-vermelhos por **um diagrama 1:1** mostrando "1 vetor monolítico (DGI 64d)" → "3 vetores especializados (64+64+64)". *Por quê:* a tese central do paper é decomposição; a figura comunica isso em 2 s, a lista comunica em 30 s. *Onde:* Slide 3 (Lacuna+Hipótese).

**R4 — Tornar o slide 6 (Setup) mais visual / mostrar mapa.** *O que mudar:* aproveitar a figura `subáreas/distribuicao_estados.png` no slide 6 (não no 8), porque ela motiva a escolha geográfica dos 3 estados — não só explica o resultado. Mover ela para mais cedo libera o slide 8 de carregar duas mensagens. *Por quê:* plateia de computação urbana se conecta com mapas; é o tipo de slide que ganha pergunta "por que esses estados?". Responder de antemão visualmente. *Onde:* slides 6 e 8.

**R5 — Cortar slide 5 para 40 s e fundir parte com slide 4.** *O que mudar:* slide 5 (SIREN vs Sphere2Vec-M detalhado) é técnico demais para 1:00 num talk de 9:00. Reduzir para uma única linha de comparação ("**SIREN**: senoidal local · **Sphere2Vec-M**: esférica multi-escala") e deixar o detalhe completo no apêndice A3/A4. *Por quê:* a história que vence em CoUrb é "decomposição importa", não "qual senoide é melhor"; o detalhe de arquitetura espacial vai para Q&A. *Onde:* Slide 5 → 40 s; recuperar 20 s para slide 4 ou slide 8.

---

## 3. Três aberturas/encerramentos mais impactantes

**Hook A (slide 1 — pergunta urbana):** *"Quando você abre o Google Maps e ele sugere 'restaurante próximo às 19h', há três sinais distintos sendo combinados: onde você está, que horas são, e que tipo de lugar você gosta. Hoje vou mostrar que separar esses três sinais — em vez de fundi-los — melhora a predição em até 24 pontos."*

**Hook B (slide 1 — provocação técnica):** *"Vamos falar de uma escolha de projeto silenciosa que custa 20 pontos de F1: usar um embedding único para representar lugar, tempo e categoria ao mesmo tempo."*

**Encerramento C (slide 10 — fechamento aplicado):** terminar não com "Obrigado, perguntas?" mas com **uma frase de transferência para a plateia de CoUrb**: *"Para quem trabalha com mobilidade, IoT urbano ou recomendação contextual: a lição é que decompor a entrada por modalidade de sinal pode render mais ganho do que sofisticar a arquitetura central. Código aberto, dados abertos — venham conversar."*

---

## 4. Erros típicos a evitar em SBC/SBRC

- **Estourar o tempo** — chair em CoUrb costuma ser estrito (4 talks em 45 min); preparar versão de 7 min mental para caso a sessão atrase.
- **Ler bullets** — plateia brasileira sênior nota imediatamente; perde-se autoridade.
- **Excesso de jargão de ML puro** — metade da sala é de redes/IoT; explicar "embedding" em 1 frase no slide 2, não assumir conhecimento de NashMTL/FiLM.
- **Defender demais limitações** — admitir Travel logo e seguir. Não gastar 2 min justificando.
- **Mostrar tabela completa do paper** — 21 linhas em fonte 10pt é morte certa; usar gráfico de barras ou subset.
- **Português misturado com inglês solto** — manter consistência: termos técnicos em itálico (*embedding*, *encoder*), restante em PT-BR limpo.
- **"Obrigado" como último slide** — terminar com takeaway + call to action; "obrigado" se fala.

---

## 5. Cinco perguntas mais prováveis em Q&A + respostas curtas

**Q1: "Por que não isolaram a contribuição individual de cada encoder (ablation)?"**
*Resposta:* "Reconhecemos essa limitação no artigo — é o próximo passo natural. Hipótese atual é que o ganho vem majoritariamente do espacial, mas só ablation isolando temporal e categórico confirma. Está no roadmap."

**Q2: "Como esse modelo escala para datasets maiores ou para o Brasil?"**
*Resposta:* "Os três encoders são pré-treinados independentemente, então escalam linearmente com POIs — o gargalo é o MTLNet central, que treina em ~poucas horas em GPU única. Migração para Foursquare-BR ou InLoco é factível."

**Q3: "Vocês usam coordenadas contínuas — como tratam ruído de GPS / check-ins falsos?"**
*Resposta:* "Hoje confiamos no Gowalla cru; a loss contrastiva (par+ ≤10 km, par− ≥70 km) tolera ruído de algumas centenas de metros. Para dados mais ruidosos, suavização por kernel ou filtro Haversine seria adição direta."

**Q4: "Por que Time2Vec e não embeddings de hora/dia mais simples (one-hot, cíclico)?"**
*Resposta:* "Time2Vec aprende a periodicidade — fase e frequência são parâmetros, não fixos. Em Q&A profunda: cíclico (sin/cos de hora) é caso particular do Time2Vec com frequência fixa em 2π/24."

**Q5: "Travel piora — isso não invalida a tese?"**
*Resposta:* "Não. Travel envolve trajetórias longas onde a topologia regional do grafo DGI ainda é melhor — é um sinal estrutural diferente. Fusão híbrida grafo + coordenadas é o trabalho futuro mais promissor."

---

## 6. Veredito final

**Pode apresentar com esse deck? Sim.** A estrutura é sólida, o número-âncora é forte, limitações estão honestas, tempo cabe. **Prioridade de ajuste:** (1) aliviar o slide 4 (crítico), (2) antecipar o número-âncora para a capa, (3) cortar slide 5 para 40 s. Esses três ajustes valem 1 ponto na nota e ~30 s de folga no orçamento.

**Nota: 7,5 / 10.**
*Justificativa:* +2 pela estrutura clara e número-âncora, +2 pela honestidade sobre limitações, +1,5 pela figura-mãe forte, +1 pelo apêndice bem planejado, +1 pelos bullets enxutos no geral. Desconto de -1 pela sobrecarga do slide 4, -0,5 pela ausência de hook urbano explícito na abertura, -1 por ainda apresentar a comparação SIREN vs Sphere2Vec-M com peso desproporcional à mensagem central (decomposição > escolha de encoder).

Com os 3 ajustes prioritários, sobe para **8,5–9 / 10**.
