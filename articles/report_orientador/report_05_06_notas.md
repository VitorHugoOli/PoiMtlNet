# Notas de apoio — Relatório Check2HGI + MTL (05/06/2026)

> Companion do `report_05_06.tex` / `report_05_06.pdf`. Aqui ficam o **guia de leitura** e a
> **síntese para a fala**; o relatório em si (tabelas e números) está no PDF.

---

## Como ler o relatório

Este relatório registra, de forma **macro** (para apresentação falada) e **micro** (números
difíceis de lembrar de cabeça), o que mudou desde a submissão do artigo *Beyond Cross-Task*
(BRACIS 2026). A espinha da história é simples:

> **O artigo concluiu que o MTL "sacrificava" a tarefa de região (gap de −7 a −17 p.p. vs.
> single-task). Trabalho posterior mostrou que esse gap era, em boa parte, fruto de _dois defeitos
> metodológicos_ — a fórmula de seleção do checkpoint conjunto e o método de loss das cabeças.
> Corrigidos, um _único modelo MTL_ passa a empatar/superar tanto o teto single-task quanto a
> solução de dois modelos (composite), e o ganho de embedding (substrato) transfere para o MTL.
> A tarefa de categoria também melhora.**

### ⚠ Aviso de maturidade dos números (importante)

O achado central (§6 do relatório, "correção C25") é **muito recente (2026-06-05)** e está em
**re-validação**. Os números corrigidos do estado atual são **provisórios**, sob quatro ressalvas:

1. re-validação ainda em curso;
2. medidos em **AL / GE / FL** apenas (AZ / CA / TX pendentes);
3. usaram a receita `onecycle`, **não** a receita `B9` exata da §0.1 do paper — ainda não é
   substituição *drop-in*;
4. ainda não "congelados" numa versão canônica.

Nada disso invalida a direção; é **disciplina de rotulagem**.

---

## Síntese de uma página (para a fala)

1. **O artigo** mostrou Check2HGI ≫ HGI em *categoria* (+15 a +29 p.p.) e um MTL que *empata cat*
   mas *perde região* por −7 a −17 p.p.

2. **Dois bugs metodológicos** explicavam o gap de região:
   - a **fórmula de seleção do checkpoint conjunto** (seletor C21: média aritmética →
     `geom_simple = √(cat_F1 · reg_Acc@10)`; **+5,6 p.p.** ao corrigir);
   - o **loss ponderado por classe** desalinhado da métrica Acc@10 (C25; **~10–14 p.p.**).

3. **Nova metodologia L0–L3** de avaliação de *embedding* separou qualidade-de-substrato de
   capacidade-de-cabeça; revelou que o sinal de região vive em **log T** (não na geometria); e
   reabilitou o **Delaunay** → substrato **v14** (cat 67,4; fecha ~69% do gap de reg para o HGI).

4. **Corrigidos os bugs**, um **único modelo MTL** empata/supera o teto *single-task* **e** o
   *composite* de 2 modelos; o ganho de substrato **transfere** para o MTL; e a *categoria* sobe
   +2 a +3,5 p.p. No estado grande (FL), o último resíduo é fechado pelo **dual-tower**
   (73,06; −0,25 vs. teto) — as ordenações de arquitetura **inverteram** sob a correção (o
   dual-tower foi de pior a melhor).

5. **Ainda em aberto:** re-validar AZ/CA/TX, fechar/confirmar o resíduo de FL com o dual-tower,
   continuidade de receita (B9 exato da §0.1), re-pin da §0.1. *O estudo segue em execução — há
   margem para novas melhorias.*

---

## Mapa de números (anti-confusão)

Três conjuntos de **MTL-reg** distintos, separados por 10–14 p.p. — **não misturar**:

| conjunto | AL | (estado) | FL | o que é |
|---|---|---|---|---|
| §0.1 v11 (artigo) | 50,17 | AZ 40,78 | 63,27 | baseline — **confound de class-weighting** |
| §0.1 corrigido (subst. v11 GCN, não-pond.) | **62,60** | GE 56,34 | **70,74** | o número *do artigo* após o bug-fix (só o loss) |
| C25 atual (subst. v14, não-pond.) | **64,52** | GE 57,84 | **71,55** | bug-fix **+** ganho de substrato (empilha por cima) |
| board v14 antigo (pré-correção) | 50,14 | AZ 37,78 | 61,21 | **NÃO citar** — ponderado/confundido |
