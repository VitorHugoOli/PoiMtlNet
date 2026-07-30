# Perguntas — Revisão do Artigo MobiWac

1. O que diz no artigo do Loreiro e como nos relacionamos a ele ?

2. O nosso MTL, é um MTL mesmo? O que estamos compartilhando ?
    1. Entender nossa arch a fundo.

3. O que a literatura quer ? Prever a categoria do proximo cehcking ou do poi(isso e a mesma coisa não ?)
    1. Quais tarefas são as mais usuais no contexto de POI ?
    2. Quais são as princiais entradas para next-POI ?

4. Me explique melhor a frase:
   > "to our knowledge, the first to treat fine-grained region as an end target of equal standing (Section II-B)"

   Isso e bom ? COmo a literatura lida com isso ?

5. Como funciona o TOST test ?

6. Como funciona DGI/HGI ?
    1. O que e Infomax ?

7. Na literatura a abordagem commum para se tentar prever é: Dado uma sequencia de checkings/POIs qual é o proximo POI?

8. Me explique os conceitos e afrase: :
   > "We fix the two-point margin in advance because a mobility-aware service acts on which region
   > will be busy, not on a single rank position (Section III), so a two-point shift in Acc@10 is
   > below the granularity at which such a service would behave differently. For scale, with 520
   > to 8,501 regions, a random top-ten guess includes the true region at most about two percent
   > of the time. The equivalence is well powered: the paired difference has a small standard
   > deviation (0.04 to 0.15 points at the four datasets with four seeds), so the procedure has
   > power near 1.0 to declare equivalence when the true difference is near zero; the reported
   > intervals pass a margin as small as half a point at Alabama and Arizona (Section VI-B)."

   Em detalhes.

9. Que teste de controle são esses?
   > "A controlled test isolates this: averaging the per-visit vectors into one vector per place
   > removes roughly 64 to 90 percent of the gain (state-dependent), so most comes from the
   > context that each visit carries, not from extra training signal."

   > "A feature-concat control (the place embedding joined with raw per-visit features, same
   > model) does not close the gap either, so the gain comes from the hierarchical per-visit
   > representation, not from contextualization alone or feature injection"

10. Eu quero entender melhor a questão das tarefas não estarem compartilhando conhecimento de acordo com o experimento
    do freezed de uma das tarefas. Minha duvida e quando, há compartilhamento ? O shared-trunk adiciona mais paramentros
    logo o ganho e por temros mais parametros e não por compartilhamento?

11. Como sabemso que os gradiente das taefas são ortogonais e por isso não usamos os MTL Optmizers ?
