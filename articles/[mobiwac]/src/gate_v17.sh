#!/bin/sh
# gate_v17.sh -- o portao anti-v17 (CONSOLIDATION_PLAN item 3.62b).
# Corre a partir de articles/[mobiwac]/src/. Sai rc=1 se qualquer numero da geracao
# com vazamento sobreviver, no PDF ou na fonte (comentarios INCLUIDOS -- e la que
# vivem os blocos v17 herdados). Precondicao da fusao final e do reenvio.
set -u
rc=0
PDF=main.pdf
[ -f "$PDF" ] || { echo "FALHA: $PDF nao existe"; exit 1; }
pdftotext "$PDF" /tmp/gate_v17.txt 2>/dev/null || { echo "FALHA: pdftotext"; exit 1; }

# (a) celulas v17 de categoria e da representacao -- nunca no texto renderizado
# NOTA: 64.54 NAO entra nesta lista. E a celula v17 de categoria de Alabama E, ao mesmo tempo,
# a celula v18 legitima de regiao da California, que a verificacao (c) exige que esteja presente.
# Uma lista que a proibisse contradizia-se a si propria. Apanhado na primeira execucao do portao.
for n in 63.32 63.33 64.51 65.79 65.83 79.84 79.85 77.24 77.05 54.74 56.82 56.43 74.51 69.79 70.60 75.15 54.65 55.87 57.13 69.95 70.26 28.09 29.31 27.63 39.62 37.47 37.95 26.56 35.53 32.48 32.31 34.46 38.96 66.06 65.68 62.37 65.69; do
  if grep -qF "$n" /tmp/gate_v17.txt; then echo "PDF: numero v17 vivo -> $n"; rc=1; fi
done

# (b) frases retiradas -- PDF e fonte
for s in "has not been run" "several times the size" "seven datasets" "fifth of a point" \
         "sharing helps instead of hurting" "Honesty rules" "at least 4 Acc@10" \
         "at least 33 macro" "two answers at the price of one"; do
  grep -qF "$s" /tmp/gate_v17.txt && { echo "PDF: frase retirada viva -> \"$s\""; rc=1; }
  grep -rqF "$s" sections/ tables/ main.tex 2>/dev/null && { echo "FONTE: frase retirada viva -> \"$s\""; rc=1; }
done

# (c) as celulas v18 TEM de estar la, com o sinal certo (CAMERA_READY 3.1/3.2)
for n in 35.42 30.59 34.57 37.55 36.19 35.63 75.08 69.24 59.04 76.54 66.15 64.54; do
  grep -qF "$n" /tmp/gate_v17.txt || { echo "PDF: celula v18 EM FALTA -> $n"; rc=1; }
done

# (d) integridade do build
grep -q "Reference .* undefined" main.log 2>/dev/null && { echo "BUILD: referencia indefinida"; rc=1; }
grep -q "Citation .* undefined" main.log 2>/dev/null && { echo "BUILD: citacao indefinida"; rc=1; }
grep -q "Overfull" main.log 2>/dev/null && { echo "BUILD: overfull box"; rc=1; }

[ $rc -eq 0 ] && echo "PORTAO VERDE: nenhum numero ou frase v17 sobreviveu; as 12 celulas v18 estao presentes."
exit $rc
