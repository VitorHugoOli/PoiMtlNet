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
rm -f main.pdf   # nunca avaliar um PDF que este portao nao produziu
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
  # fronteira, e nao o prefixo " $n": um delta com sinal ("+9.35") escapava-lhe.
  grep -qE "(^|[^0-9.])${n}([^0-9]|$)" /tmp/gate_pdf.txt && fail "numero v17 no PDF -> $n"
  grep -qE "(^|[^0-9.])${n}([^0-9]|$)" /tmp/gate_src.txt && fail "numero v17 na fonte -> $n"
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
         "does better where the region task is hardest" "price worth paying" \
         "matches it" "non-inferior match" \
         "weekday\\\\ trained with no task labels"; do
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
grep -q '^!' main.log && fail "erro TeX que nao mata o build: derruba texto em silencio"

# (e) orcamento de paginas (decisao do autor: 8 finais).
echo "[e] paginas"
PG=$(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/{print $2}')
echo "  paginas = ${PG}"
if [ "${GATE_PHASE:-1}" = "2" ]; then
  [ "$PG" = "8" ] || fail "fase 2: o alvo e 8 paginas, esta em ${PG}"
  # \IEEEpubid REMOVIDO da fase 2 em 2026-09-06. O autor leu as normas do venue
  # (scomminc.com/pp/ieee/mobiwac26.htm) e confirmou: NAO ha bloco de copyright a acrescentar pelo
  # autor -- o formulario electronico e enviado ao autor de contacto DEPOIS do upload. Tambem nao ha
  # agradecimentos nem financiamento a declarar. Logo a pagina 1 nao perde espaco nenhum, e as 8
  # paginas passam a ser o unico criterio desta fase.
else
  echo "  (fase 1: paginas ainda nao sao criterio; correr com GATE_PHASE=2 para o corte)"
fi


# (f) FRASES PARTIDAS POR UM COMENTARIO. Classe nova, encontrada 2026-09-06 na verificacao da
#     Fase 1: dois blocos de comentario foram colados a MEIO de uma frase e engoliram-lhe a cauda.
#     O PDF passou a dizer "We remove We also run STAN" e "With frozen weights difference against
#     the values in Table III", perdendo pelo caminho a divulgacao de que o baseline externo
#     primario foi modificado e o resultado do CTLE com pesos congelados.
#     Nenhuma verificacao de numeros apanha isto; o build nao da erro; o portao estava VERDE atraves
#     das duas. O sinal e mecanico: uma linha de prosa viva que acaba sem pontuacao terminal e
#     imediatamente seguida por uma linha de comentario.
echo "[f] comentario colado a meio de uma frase"
#     REGRA: um comentario vive ENTRE frases, nunca a meio de uma. Quem escreve um comentario a
#     seguir a uma linha que acaba a meio de uma frase arrisca engolir-lhe a cauda -- foi assim que
#     o PDF passou a dizer "We remove We also run STAN", "The second control A final control uses" e
#     "With frozen weights difference against the values in Table III", perdendo pelo caminho a
#     divulgacao de que o baseline externo primario foi modificado e o resultado do CTLE congelado.
#     O portao estava VERDE atraves das tres. A deteccao e DELIBERADAMENTE larga: um falso positivo
#     custa ler uma linha, um falso negativo entrega texto partido. Para o limpar, mova-se o
#     comentario para depois do ponto final -- que e onde ele devia estar.
: > /tmp/gate_broken.txt
for f in sections/*.tex tables/*.tex main.tex; do
  awk -v F="$f" '
    { line=$0; sub(/[ \t]+$/,"",line); is_c = (line ~ /^[ \t]*%/)
      if (is_c && prev_live && prev !~ /[.:;,}%]$/ && prev !~ /\\$/) {
        tail=prev; if (length(tail)>56) tail=substr(tail,length(tail)-55)
        printf "  FALHA: %s:%d comentario a meio de frase -> ...%s\n", F, NR-1, tail
      }
      prev=line; prev_live=(!is_c && line!="")
    }' "$f" >> /tmp/gate_broken.txt
done

# (f2) A IMAGEM AO ESPELHO, encontrada 2026-09-06 depois de (f) ja existir. A verificacao (f) apanha
#      uma linha VIVA seguida de comentario. Falta o inverso: um comentario cuja ULTIMA linha absorveu
#      a cabeca da frase seguinte, deixando viva so a cauda. Foi assim que o PDF passou a imprimir
#      "...not a general rule. (Istanbul, Alabama, Arizona, and Florida), is equivalent to zero...".
#      O sinal: uma linha viva que comeca por algo que nao pode comecar uma frase -- minuscula,
#      parentese, virgula, fecho -- logo a seguir a um comentario.
for f in sections/*.tex tables/*.tex main.tex; do
  awk -v F="$f" '
    { line=$0; sub(/^[ \t]+/,"",line); sub(/[ \t]+$/,"",line)
      is_c = (line ~ /^%/)
      if (!is_c && line != "" && prev_comment) {
        if (line ~ /^[a-z(,)]/ && line !~ /^\\/) {
          head=line; if (length(head)>56) head=substr(head,1,56)
          printf "  FALHA: %s:%d frase comeca a meio, depois de um comentario -> \"%s...\"\n", F, NR, head
        }
      }
      prev_comment = is_c
    }' "$f" >> /tmp/gate_broken.txt
done
cat /tmp/gate_broken.txt
[ -s /tmp/gate_broken.txt ] && rc=1

[ $rc -eq 0 ] && echo "PORTAO VERDE"
exit $rc
