#!/bin/sh
# gate_v17.sh -- o portao anti-v17 (CONSOLIDATION_PLAN item 3.62b).
# Corre de dentro de articles/[mobiwac]/src/. Precondicao da fusao final e do reenvio.
#
# HISTORICO DE FALHAS DO PROPRIO PORTAO (cada uma corrigida abaixo; nao as reintroduzir):
#  - proibia 64.54, que e a celula v17 de categoria de AL E a celula v18 legitima de regiao da CA
#    que a verificacao (c) exige presente: a lista contradizia-se a si propria;
#  - verificava frases no PDF linha a linha, e "sharing helps instead of hurting" atravessa uma
#    quebra de linha, portanto era invisivel;
#  - a verificacao da fonte nao lia figs/, onde vivia "fifth of a point";
#  - (c) so testava presenca, nunca o sinal;
#  - lia main.log como o encontrasse, portanto um log velho mudava o veredito;
#  - nao verificava o numero de paginas, com a decisao do autor fixada em 8.
set -u
rc=0
fail() { echo "  FALHA: $*"; rc=1; }

[ -f main.tex ] || { echo "correr de dentro de src/"; exit 2; }

# (0) o PDF tem de ser deste texto: reconstruir, nunca confiar no que esta em disco.
echo "[0] rebuild"
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
bibtex main >/dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
[ -f main.pdf ] || { echo "  FALHA: build nao produziu main.pdf"; exit 1; }

# texto do PDF numa so linha, para frases que atravessam quebras nao escaparem
pdftotext main.pdf - 2>/dev/null | tr '\n' ' ' | tr -s ' ' > /tmp/gate_pdf.txt
# fonte + comentarios, incluindo figs/ e tables/ -- e onde os numeros v17 se escondem
# DUAS VISTAS DA FONTE, e a distincao e deliberada:
#   TUDO (comentarios incluidos) -> para NUMEROS. Um numero v17 num comentario e dado que o proximo
#     agente copia. Provado: um comentario herdado afirmava que "the 65.69 percent ... are traced and
#     correct" enquanto a prosa imprimia, bem, 64.54 -- uma instrucao escrita a mandar repor o valor
#     errado. Mesma forma do bloco "Honesty rules (do NOT relax)" ja removido.
#   SO PROSA VIVA (comentarios retirados) -> para FRASES. Documentar uma remocao exige nomear o que
#     se removeu; proibir isso destruiria a proveniencia, que este projecto trata como lei.
#     Uma frase num comentario nao pode ser lida como dado.
cat sections/*.tex tables/*.tex figs/*.tex main.tex 2>/dev/null | tr '\n' ' ' | tr -s ' ' > /tmp/gate_src.txt
cat sections/*.tex tables/*.tex figs/*.tex main.tex 2>/dev/null | sed 's/%.*//' | tr '\n' ' ' | tr -s ' ' > /tmp/gate_live.txt

# (a) numeros da geracao com vazamento. Lista derivada de CAMERA_READY.md seccao 4.
#     NOTA: 64.54 NAO entra -- colide com a celula v18 legitima de regiao da California.
echo "[a] numeros v17"
for n in 63.32 63.33 64.51 65.79 65.83 65.84 79.84 79.85 77.24 77.23 77.05 77.04 \
         54.74 56.82 56.43 74.51 69.79 70.60 75.15 75.35 75.44 77.41 77.42 \
         69.70 69.80 59.46 59.56 63.49 64.95 67.06 67.07 65.69 70.11 76.70 \
         54.65 55.87 57.13 69.95 70.26 28.09 29.31 27.63 39.62 37.47 37.95 \
         26.56 29.50 35.53 32.48 32.31 34.46 38.96 66.06 65.68 62.37 \
         37.8 37.0 28.7 4.9 10.3 5.34 8.59 9.40 7.69 7.45 6.45 8.58 9.35 5.33; do
  grep -qF " $n" /tmp/gate_pdf.txt && fail "numero v17 no PDF -> $n"
  grep -qF " $n" /tmp/gate_src.txt && fail "numero v17 na fonte/comentarios -> $n"
done

# (b) frases retiradas.
#     ERRO DESTE PORTAO, CORRIGIDO 2026-09-06. A regra proibia a substring
#     "where the region task is hardest", e isso forcou a mutilacao da passagem P1 -- que o plano
#     manda PROTEGER. A forma morta e a asercao ISOLADA que estava no resumo ("the model DOES BETTER
#     where the region task is hardest", a leitura C4); a forma do capitulo -- "which is where the
#     region task is hardest and where the dedicated model has the most to gain" -- e LICENCIADA
#     pela frase seguinte, que nomeia os dois confundidores e declara o conjunto uma observacao e
#     nao uma lei. A cadeia nunca esteve na lista de nunca-citar do CAMERA_READY (zero ocorrencias);
#     foi inventada aqui. Um portao que destroi o que foi construido para proteger e pior do que
#     nenhum: a regra e agora a frase inteira, nao a substring. PDF colapsado (atravessam linhas) + fonte incluindo figs/.
echo "[b] frases retiradas"
for s in "has not been run" "several times the size" "seven datasets" "fifth of a point" \
         "sharing helps instead of hurting" "Honesty rules" "at least 4 Acc@10" \
         "at least 33 macro" "two answers at the price of one" "+5 percent" \
         "and learning rate were searched for the dedicated category model at every dataset" \
         "does better where the region task is hardest" "price worth paying"; do
  grep -qF "$s" /tmp/gate_pdf.txt && fail "frase retirada no PDF -> \"$s\""
  grep -qF "$s" /tmp/gate_live.txt && fail "frase retirada em prosa viva -> \"$s\""
done

# (c) as 12 celulas v18 tem de estar presentes E com o sinal certo (CAMERA_READY 3.1/3.2).
echo "[c] celulas v18 e sinais"
for n in 35.42 30.59 34.57 37.55 36.19 35.63 75.08 69.24 59.04 76.54 66.15 64.54; do
  grep -qF "$n" /tmp/gate_pdf.txt || fail "celula v18 EM FALTA no PDF -> $n"
done
# sinais: as unicas superioridades sao regiao TX/CA e categoria FL.
grep -qF "1.21" /tmp/gate_pdf.txt || fail "delta de regiao do Texas (+1.21) em falta"
grep -qF "1.06" /tmp/gate_pdf.txt || fail "delta de regiao da California (+1.06) em falta"
grep -qF "0.19" /tmp/gate_pdf.txt || fail "delta de categoria da Florida (+0.19) em falta"
# nenhum delta de categoria pode aparecer com magnitude de geracao antiga
for n in "+5." "+6." "+7." "+8." "+9."; do
  grep -qF "$n macro" /tmp/gate_pdf.txt && fail "delta de categoria com magnitude v17 -> $n"
done

# (d) integridade do build, lida do log que ESTE portao acabou de produzir.
echo "[d] build"
grep -q "Reference .* undefined" main.log && fail "referencia indefinida"
grep -q "Citation .* undefined" main.log && fail "citacao indefinida"
grep -q "Overfull" main.log && fail "overfull box"
grep -q "Rerun to get" main.log && fail "o build pede rerun"

# (e) orcamento de paginas (decisao do autor: 8 finais).
echo "[e] paginas"
PG=$(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/{print $2}')
echo "  paginas = ${PG}"
if [ "${GATE_PHASE:-1}" = "2" ]; then
  [ "$PG" = "8" ] || fail "fase 2: o alvo e 8 paginas, esta em ${PG}"
  grep -q "IEEEpubid" main.tex || fail "fase 2: \\IEEEpubid em falta (consome espaco na pagina 1)"
else
  echo "  (fase 1: paginas ainda nao sao criterio; correr com GATE_PHASE=2 para o corte)"
fi

[ $rc -eq 0 ] && echo "PORTAO VERDE"
exit $rc
