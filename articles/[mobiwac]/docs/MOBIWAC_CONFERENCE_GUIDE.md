# Guia de submissão e leitura editorial da MobiWac

**Atualização:** 19 de junho de 2026  
**Janela analisada:** 2020, 2021, 2022, 2023 e 2025; regras correntes de 2026.  
**Nota de método:** a MobiWac não publica uma rubrica detalhada. Critérios além das
regras oficiais são inferências fundamentadas, não política formal da organização.

## 1. Identidade da conferência

A MobiWac é um simpósio internacional sobre **gestão de mobilidade e acesso sem fio**,
tradicionalmente realizado em conjunto com a MSWiM. O centro de gravidade não é
“mobilidade humana” em sentido amplo, mas problemas de redes e sistemas móveis:
protocolos de acesso, mobilidade e handover, 5G/6G, Wi-Fi/mmWave, IoT, redes veiculares,
localização, segurança, edge/fog, alocação de recursos, QoS e avaliação experimental.

Esse ponto é decisivo para o fit. Um trabalho de previsão de localização, mineração de
check-ins ou deep learning só é claramente MobiWac quando a contribuição responde a uma
decisão de rede ou sistema móvel: handover, seleção de AP/beam, alocação de recursos,
offloading, QoS, conectividade, segurança ou operação de infraestrutura.

Fontes: [CFP 2026](https://mobiwac-symposium.org/2026/cfp.html),
[histórico oficial](https://mobiwac-symposium.org/2026/past_editions.html).

## 2. Regras correntes — MobiWac 2026

### Papers regulares

- submissão em PDF pelo EDAS;
- **single blind**: nomes de autores podem e devem constar do manuscrito;
- máximo de **10 páginas**, duas colunas, incluindo figuras, tabelas e referências;
- tamanho regular de **8 páginas**; até 2 páginas extras mediante taxa;
- **template IEEE Proceedings** para o paper regular;
- trabalho original, não publicado e não simultaneamente submetido;
- ao menos um autor deve comparecer e apresentar;
- conferência: Paris, 26–30 de outubro de 2026;
- deadline publicado: **25 de junho de 2026**;
- notificação publicada: **31 de julho de 2026**;
- camera-ready: ainda TBA em 19/06/2026.

Fontes: [submissions 2026](https://mobiwac-symposium.org/2026/submissions.html) e
[página inicial 2026](https://mobiwac-symposium.org/2026/index.html).

### Inconsistência oficial relevante

A página de papers regulares exige **IEEE style**, mas a página de posters ainda diz
**ACM style** e limita posters a 4 páginas. Isso parece conteúdo residual da fase ACM.
Não reutilize automaticamente o template do paper regular para poster. Confirme no EDAS
ou com o TPC Chair antes de submeter poster.

Fonte: [posters 2026](https://mobiwac-symposium.org/2026/posters.html).

### Mudança de editora/formato

Até 2023, os proceedings recentes foram publicados pela ACM e os papers usavam ACM
double-column. Em 2025, os papers MobiWac apareceram como sessões dentro dos proceedings
IEEE de MSWiM 2025, com DOI IEEE. Em 2026, o CFP já pede explicitamente template IEEE.
Portanto, exemplos ACM de 2020–2023 são úteis para narrativa e densidade, mas **não são
modelo tipográfico para 2026**.

## 3. Processo de avaliação: fatos e inferências

### O que é oficial

- revisão por membros do Technical Program Committee e especialistas;
- avaliação de qualidade e relevância;
- em 2021, cada submissão recebeu pelo menos três revisões;
- em 2021, 20 papers foram aceitos, taxa oficial de 26%; três papers regulares foram
  finalistas do prêmio;
- o prêmio é decidido por uma comissão durante o simpósio, o que indica que a
  apresentação também pode influenciar o vencedor final;
- regular papers formam uma trilha única, com short papers/posters/demos intercalados.

Fontes: [MobiWac 2021](https://mobiwac-symposium.org/2021/index.html),
[CFP 2025](https://mobiwac-symposium.org/2025/cfp.html).

### Rubrica interna inferida

| Dimensão | O que um paper forte demonstra | Peso sugerido |
|---|---|---:|
| Fit | problema central de rede móvel/acesso sem fio | 20% |
| Novidade | mecanismo ou arquitetura nova, não só troca de modelo | 20% |
| Rigor | formulação correta, baselines fortes, ablação e incerteza | 20% |
| Realismo | traces, testbed, hardware ou simulador calibrado | 20% |
| Impacto | ganho em throughput, latência, energia, robustez, QoS ou segurança | 15% |
| Clareza | narrativa, figuras e conformidade | 5% |

Essa matriz não é oficial. Ela sintetiza os vencedores de 2020–2022 e destaques de
2023–2025.

## 4. O estilo que os papers fortes repetem

### 4.1 Problema operacional, não apenas tarefa de ML

O paper vencedor de 2022 não apresenta apenas um preditor: conecta previsão de qualidade
de enlace a alocação proativa em WLAN mmWave. O destaque de 2023 não apresenta apenas
GRU: seleciona entre beam switching, handoff e beam widening. A aprendizagem é meio;
a decisão de rede e sua consequência mensurável são o argumento principal.

### 4.2 Combinação de modelo e evidência

Os trabalhos fortes misturam duas ou mais camadas de validação:

- análise/formulação matemática + simulação;
- dataset sintético controlado + medição real;
- trace real + testbed;
- comparação contra políticas/baselines de rede;
- estudo de sensibilidade ou cenários fora do treino.

Um único split aleatório e uma única métrica de acurácia são fracos para este venue.

### 4.3 Claims quantitativos e ligados ao sistema

Prefira “reduz desperdício de recursos”, “aumenta dados transferidos”, “mantém QoS sob
blockage” ou “reduz erro de localização” a “melhora F1”. Métricas de ML devem ser
traduzidas em comportamento de rede, custo ou experiência do usuário.

### 4.4 Estrutura eficaz em oito páginas

1. **Introdução:** problema, consequência operacional, lacuna e 3–4 contribuições.
2. **Background/related work curto:** apenas o necessário para posicionar o gap.
3. **System/problem model:** entidades, hipóteses, decisão e objetivo.
4. **Método:** arquitetura/algoritmo com complexidade e integração no sistema.
5. **Metodologia experimental:** dados, cenários, baselines, métricas e hardware/software.
6. **Resultados:** pergunta por subseção; números, intervalos e ablações.
7. **Limitações/discussão:** onde falha, custo e generalização.
8. **Conclusão:** descoberta e implicação, sem recontar toda a introdução.

### 4.5 Figuras preferíveis

- uma visão geral do sistema no início;
- um diagrama do fluxo de decisão;
- resultados com baselines e variação, não apenas médias;
- uma figura que demonstre robustez a mobilidade, blockage, carga ou mudança de cenário;
- labels legíveis em duas colunas e captions que expressem a conclusão.

## 5. Temas recentes

Os programas 2022–2025 mostram recorrência em:

- mmWave, beamforming, beam alignment e blockage;
- 5G/6G, RAN slicing e alocação de recursos;
- localização indoor e radio maps;
- federated/distributed learning para redes;
- IoT e redes de baixa potência, especialmente LoRa/LoRaWAN;
- segurança, privacidade, ataques e detecção de anomalias;
- edge/fog, offloading e migração;
- redes veiculares, UAVs e gestão de mobilidade;
- uso de AI/ML quando integrado a uma ação de rede.

## 6. O que tende a enfraquecer uma submissão

- classificador genérico sem consequência para rede/acesso;
- ganho pequeno contra baselines fracos ou desatualizados;
- avaliação apenas sintética sem justificar o simulador;
- trace real sem protocolo contra leakage temporal/espacial;
- ausência de ablação, dispersão ou custo computacional;
- arquitetura descrita antes do problema;
- claim de “tempo real” sem latência, memória e plataforma;
- omitir limitações de mobilidade, cobertura, privacidade ou generalização;
- usar template ACM em 2026 por copiar um paper antigo.

## 7. Adequação do projeto deste repositório

O projeto trabalha com next-category/next-region e check-ins. O fit com MobiWac é
**condicional**. Uma narrativa puramente de recomendação ou previsão de POI é mais natural
em venues de mobilidade humana/mineração. Para MobiWac, a contribuição deve ser ligada a
uma decisão de sistema, por exemplo:

- previsão hierárquica para handover/AP selection;
- prefetch/offloading no edge guiado por categoria e região futuras;
- alocação antecipatória de recursos mantendo QoS;
- custo de erro espacial para conectividade e latência;
- robustez do sistema sob mudança geográfica.

Sem essa ponte, o risco principal é desk/TPC rejection por baixa relevância, mesmo com
bons resultados de ML.

## 8. Recomendação prática

Formule o paper como uma regra de desenho de sistema:

> prever conjuntamente semântica e região permite antecipar a próxima demanda de acesso,
> mas compartilhar toda a representação degrada a tarefa espacial; uma torre regional
> privada preserva a decisão de mobilidade enquanto o contexto compartilhado melhora a
> tarefa semântica.

Para sustentar isso na MobiWac, acrescente uma métrica operacional: handovers evitados,
cache hit, latência, custo de offloading, QoS ou utilização de recursos. Compare com uma
política reativa e faça ablação da previsão e da decisão de rede separadamente.

## 9. Organização e posicionamento em 2026

O General Chair é Jun Zhang (Southeast University) e o Program Chair é Rodolfo W. L.
Coutinho (Concordia University). O TPC inclui pesquisadores de redes sem fio, sistemas,
IoT, segurança e indústria (Cisco, Nokia, NEC Labs), além de participação brasileira de
UFMG, USP e UFFS. A composição reforça que o leitor primário é de networking/sistemas,
não de ML genérico.

Fonte: [comitê 2026](https://mobiwac-symposium.org/2026/committee.html).

Não confundir MobiWac com MobiCom ou MobiHoc. A MobiWac é um simpósio menor e
co-localizado; seus proceedings recentes são indexados no DBLP e foram publicados pela
ACM até 2023, migrando para o volume IEEE/MSWiM em 2025. Qualis, CORE e métricas de
ranking mudam com o tempo e não foram usados como proxy da qualidade editorial neste
dossiê.
