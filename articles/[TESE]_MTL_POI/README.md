# [TESE]_MTL_POI — Dissertação de Mestrado (UFV / PPGCC)

> Pasta de trabalho da **dissertação de mestrado** no formato **coletânea de artigos**
> (norma UFV §2.6 — "artigos científicos"). Esta pasta reúne o planejamento, as normas
> aplicáveis e o esqueleto do documento. Os três artigos que compõem a coletânea
> permanecem nas suas pastas de origem (`articles/CBIC___MTL/`, `articles/CoUrb_2026/`,
> `articles/[BRACIS]_Beyond_Cross_Task/`) e serão **reproduzidos/reformatados** aqui como
> capítulos.

---

## Decisões já tomadas (2026-06-05)

| Decisão | Valor | Justificativa |
|---|---|---|
| **Formato** | Coletânea de artigos (UFV §2.6) | 3 artigos sob um fio condutor único; encaixa no Art. 21 §1 (≥1 artigo publicado/aceito). |
| **Idioma da moldura** | **Português** (Introdução Geral + Conclusão Geral) | Padrão UFV; §1.3 permite PT/EN/ES. |
| **Idioma dos artigos** | Original de cada um (§2.6.3 permite idiomas mistos) | CBIC e BRACIS em inglês; CoUrb em português. |
| **Artigos da coletânea** | CBIC + CoUrb 2026 + BRACIS 2026 | `report_orientador` é relatório interno → fora. |
| **Candidato** | Vitor Hugo O. Silva *(confirmar nome completo)* | Autor central nos artigos. |
| **Orientador** | Fabrício A. Silva *(confirmar)* | Coautor recorrente nos três artigos. |
| **Programa** | PPGCC — Ciência da Computação, UFV (Campus Florestal / NESPeD-LAB) | — |

---

## Os três artigos (inventário)

| # | Sigla | Título | Idioma | Veículo / Status | 1º autor |
|---|---|---|---|---|---|
| 1 | **CBIC** | *An Investigation into Multi-Task Learning for Point-of-Interest Category Classification and Next-POI Prediction* | EN | CBIC — *(confirmar status: publicado/aceito)* | Vitor H. O. Silva |
| 2 | **CoUrb 2026** | *ST-MTLNet: Representações Espaço-Temporais de Pontos de Interesse para Aprendizado Multitarefa* | PT | CoUrb/SBC 2026 — *(confirmar status)* | Tarik S. Paiva ⚠ |
| 3 | **BRACIS 2026** | *Substrate Carries, Architecture Pays: Check-In-Level Embeddings for Multi-Task POI Prediction* | EN | BRACIS 2026 (LNCS) — submetido/em revisão | Vitor H. O. Silva |

> ⚠ **CoUrb 2026**: o candidato (Vitor) é **2º autor** (1º = Tarik S. Paiva). A norma UFV não
> exige primeira autoria, mas a inclusão de um artigo liderado por outro autor na coletânea
> **deve ser validada com a Comissão Orientadora**. Ver `NORMAS_UFV.md §Pontos a confirmar`.

---

## Fio condutor (a "história" da coletânea)

A coletânea não é "três artigos grampeados" — tem um arco único que a Introdução Geral
deve tornar explícito:

> **Pergunta central:** o aprendizado multitarefa (MTL) ajuda na predição de POI
> (categoria + próximo-POI), e do que depende esse ganho?

1. **CBIC (ponto de partida):** MTL ingênuo (embedding DGI + compartilhamento FiLM +
   NashMTL) **não supera** os baselines single-task. Conclusão da época: "MTL não ajuda"
   *para aquela configuração*.
2. **CoUrb (a representação importa):** ST-MTLNet investiga **representações
   espaço-temporais** de POIs — primeiro indício de que o gargalo é o *substrato/representação*,
   não o MTL em si.
3. **BRACIS (a correção):** **Check2HGI** (embeddings em nível de *check-in*) mostra que o
   **substrato carrega** a tarefa de categoria (+14 a +29 p.p. em todos os estados) e revela
   uma **assimetria de tarefa**; refina/corrige a conclusão do CBIC — MTL tem um trade-off
   honesto na tarefa de região, e o fator dominante é a qualidade da representação.

**Tese consolidada (para a Conclusão Geral):** em predição multitarefa de POI, o **substrato
de representação** é o fator dominante de desempenho; o MTL é viável e competitivo quando
pareado com o substrato certo, mas impõe um custo mensurável e sign-consistente na tarefa de
região — um trade-off de implantação, não uma falha fundamental do MTL.

---

## Estrutura do documento (UFV §2.6 — coletânea de artigos)

```
PRÉ-TEXTUAIS (contadas, não numeradas — §2.2 / §3.5)
  ├── Capa ............................ (responsabilidade da UFV/Gráfica — §2.1)
  ├── Folha de rosto .................. autoria, título, nota explicativa (PPGCC/UFV/grau),
  │                                     comissão orientadora, local e ano
  │     └── (verso) Ficha catalográfica  solicitar no site da BBT
  ├── Folha de aprovação .............. título, data, nomes+assinaturas (banca)
  ├── Dedicatória ..................... (opcional)
  ├── Agradecimentos .................. (opcional)
  ├── Biografia do autor .............. (opcional)
  ├── Listas (figuras/tabelas/símbolos) (opcional)
  ├── RESUMO (PT) ..................... com cabeçalho normalizado (§2.2.4)
  ├── ABSTRACT (EN) ................... com cabeçalho normalizado (§2.2.4)
  └── SUMÁRIO

TEXTUAIS (§2.6 — numeração arábica a partir daqui, canto sup. direito — §3.5)
  ├── 1. INTRODUÇÃO GERAL ............. [PT] contexto, problema, objetivos, fio condutor,
  │                                     contribuições, organização (bibliografia própria opc.)
  ├── 2. ARTIGO 1 — CBIC .............. [EN] reproduzido/reformatado (§2.6.4)
  ├── 3. ARTIGO 2 — CoUrb (ST-MTLNet) . [PT] reproduzido/reformatado
  ├── 4. ARTIGO 3 — BRACIS (Check2HGI)  [EN] reproduzido/reformatado
  └── 5. CONCLUSÃO GERAL ............. [PT] síntese transversal, limitações, trabalhos
                                        futuros (bibliografia própria opc.)

PÓS-TEXTUAIS (§2.8 — opcional)
  └── Apêndices / Anexos .............. material suplementar (versões publicadas dos artigos,
                                        provas, configurações, etc.)
```

> Ordem dos artigos = **cronológica = narrativa** (CBIC → CoUrb → BRACIS). Alternativa
> temática possível, mas a cronológica já é o arco da correção — recomendada.

---

## Arquivos desta pasta

| Arquivo | Conteúdo |
|---|---|
| `README.md` | Este guia — decisões, inventário, fio condutor, estrutura, próximos passos. |
| `NORMAS_UFV.md` | Extração autoritativa das normas UFV/PPGCC (com citações de artigo e links das fontes), regras de formatação, requisitos de defesa, template, **exemplos reais (precedentes)** e pontos a confirmar. |

> Esqueleto LaTeX (ecothesis/abnTeX2) **ainda não criado** — recomendado iniciar só após o
> orientador validar nome da tese, ordem e elegibilidade dos artigos. Ver `NORMAS_UFV.md §Template`.

---

## Próximos passos

- [ ] **Validar com o orientador:** título da tese, ordem dos artigos, e a inclusão do CoUrb
      (candidato é 2º autor).
- [ ] **Confirmar status de aceite** de cada artigo (Art. 21 §1 exige ≥1 publicado/aceito no
      Qualis Computação — CBIC já cobre, mas registrar o status dos três).
- [ ] Solicitar a **ficha catalográfica** no site da Biblioteca Central (BBT).
- [ ] Escolher a base: **LaTeX ecothesis** (recomendado, bibliografia por capítulo) ou
      **template Word oficial do PPGCC**.
- [ ] Escrever a **Introdução Geral** (PT) com o fio condutor acima.
- [ ] Inserir os três artigos como capítulos (reproduzidos do original, §2.6.4).
- [ ] Escrever a **Conclusão Geral** (PT).
- [ ] Preencher pré-textuais e rodar o **checklist de pré-defesa** do PPGCC.
