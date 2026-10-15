# Reprodução — *From COCOMO to GPT* (story points)

Reprodução do benchmark de **story points** de Alaswad, Poovammal & Aljaddouh,
*From COCOMO to GPT: A Comprehensive Evaluation of LLM-Based Software Effort
Estimation*, IEEE Access 14 (2026), DOI 10.1109/ACCESS.2026.3671204.

O repositório original (pasta acima desta) **não tem dados, não tem script de
LLM nem dos protocolos de generalização e robustez**, e os scripts de ML e de
Transformers usam splits diferentes dos descritos no artigo. Esta pasta oferece
duas leituras da reprodução, com custo zero:

- **Fiel ao artigo** (etapas 10, 20, 30): protocolo descrito no texto — split
  80/10/10 por projeto, avaliação em ~5% do dataset, TF-IDF de 30 mil termos.
- **Fiel ao código** (etapas 11 e 21, só Agile): executa o que os scripts
  publicados fazem. **É daqui que saem os números de ML da Tabela 14**: KNN e
  Regressão Linear batem nas três casas decimais (MAE 5,268 e 5,934) e a Random
  Forest e o XGBoost ficam a ≤ 0,02 (5,122 e 4,952, contra 5,112 e 4,971; a diferença vem
  do float32 usado para caber na memória).

A diferença entre as duas leituras é, por si só, um resultado de
reprodutibilidade: o protocolo descrito no artigo não é o que gerou as tabelas.

### Resultado da verificação do código original (Agile, ML)

| Modelo | MAE artigo | MAE código original (aqui) | Protocolo do artigo (aqui, test completo) |
|---|---|---|---|
| KNN | 5,268 | **5,268** | 4,357 |
| Regressão Linear | 5,934 | **5,934** | 25,77 sem SVD / 4,673 com SVD |
| Random Forest | 5,112 | **5,122** | 3,956 |
| XGBoost | 4,971 | **4,952** | 4,007 |
| SVR | 4,257 | pendente (> 1 h de CPU) | 3,590 |

O código original usa: TF-IDF com **93.192 termos** ajustado no dataset inteiro
(o artigo diz 30 mil); SVD de 3.000 componentes ajustada com o teste; split
**aleatório global 70/30**, ou seja, um teste de 6.994 issues (o artigo diz ~1.000).
Pelo protocolo do artigo, os erros dos modelos de ML ficam **menores** do que os
publicados.

### NEO, protocolo do artigo (subconjunto de avaliação, 1.022 issues)

| Modelo | MAE artigo (Tabela 15) | MAE aqui |
|---|---|---|
| KNN | 6,387 | 6,825 |
| Random Forest | 6,715 | 3,163 |
| XGBoost | 6,032 | 2,931 |
| Regressão Linear | 7,184 | 61,5 sem SVD (degenera) |
| SVR | 5,684 | pendente |

O código original não processa o NEO, então não há como saber de onde saíram os
números da Tabela 15.

## Baselines ingênuos — o teste que o artigo não faz

`12_run_baselines.py` gera quatro "modelos" que não leem o texto: média e
mediana do treino, e média e mediana **do próprio projeto**. Tawosi et al. (2022)
mostraram que a mediana costuma ser competitiva com o estado da arte, e o artigo
não compara contra ela. O resultado aqui confirma isso:

| Within (MAE) | Agile | NEO |
|---|---|---|
| Mediana do projeto (sem texto) | **3,92** | **2,64** |
| Melhor ML com texto | 3,67 (SVR) | 2,93 (XGBoost) |
| Mediana global | 4,30 | 4,52 |
| LLMs no artigo | 7,2 – 10,4 | 8,9 – 12,2 |

No Agile, só o SVR supera a mediana do projeto. No NEO, nenhum modelo de texto
supera. No cross-project, a mediana global (Agile 2,71; NEO 1,59) fica à frente
de quase todos os modelos.

## Extensão: características dos repositórios (só NEO)

O NEO traz um JSON por projeto com metadados da API do GitLab. `04_repo_features.py`
gera 8 features por issue: log de estrelas, forks e issues abertas, número de
tópicos, organização dona (GitLab, Minds, FPC) e **dias desde a primeira issue do
projeto**. O Agile não tem nada equivalente.

```bash
python repro/04_repo_features.py
python repro/10_run_ml.py --dataset neo --features repo        # só repositório, sem texto
python repro/10_run_ml.py --dataset neo --features text+repo   # texto + repositório
```

Os resultados vão para `{modelo}_repoonly.csv` e `{modelo}+repo.csv`, separados da replicação.

| NEO, MAE | within | cross |
|---|---|---|
| Mediana do projeto (baseline) | 2,64 | — |
| Mediana global (baseline) | 4,52 | 1,59 |
| SVR, só repositório | **2,61** | 1,72 |
| XGBoost, só repositório | 2,69 | **16,87** |
| XGBoost, só texto | 2,93 | 1,47 |
| XGBoost, texto + repositório | 2,79 | **1,47** |

Como ler:
- **No within, as características do repositório ajudam porque identificam o
  projeto.** Elas são constantes dentro de cada projeto (fora a idade), então o modelo
  aprende "qual time é este" e usa a escala de SP daquele time. Por isso o SVR só com
  repositório empata com a mediana do projeto: os dois capturam a mesma coisa.
- **No cross-project elas atrapalham as árvores.** Com só ~27 projetos de treino, a
  Random Forest e o XGBoost decoram a identidade dos projetos e aplicam a escala
  errada a projetos novos (MAE 17–19). Combinadas com o texto, o ganho é nulo.
- **Vazamento temporal:** estrelas, forks e issues abertas são um retrato de junho de
  2023 (data da mineração), posterior à maioria das issues. A única feature calculada
  só com o passado é a idade do projeto.

O NEO também traz os atributos completos de cada issue (labels, milestone,
responsáveis, tempo gasto). Ficaram de fora: quase todos são registrados *depois* da
estimativa (comentários, merge requests, data de fechamento) e o tempo gasto real só
existe em 834 das 40.027 issues (2%).

## Extensão: versões limpas (agile_limpo, neo_limpo)

Ao analisar a qualidade por projeto, encontramos problemas nos próprios dados:

- **NEO, 4 projetos cujo "story point" não é estimativa do time** (3.954 issues, 19%):
  - KiCad (15502567) e Duplicity (12450835) foram migrados do Launchpad: SP mínimo 4, com 64% e 53% das issues valendo exatamente 6, o que indica um peso padrão da migração;
  - Lazarus (28419588) e FPC Source (28644964) têm pesos de 100 a 300.
- **Agile: o projeto GHS (68 issues) é cópia exata de issues do JSW.** Isso explica os 17 códigos contra os 16 projetos declarados.
- **Duplicatas:** 129 grupos no Agile e 137 no NEO, alguns com SP diferente; **148 issues com SP = 0** no NEO.
- **Pré-processamento:** remover blocos de código, como o artigo descreve, apaga ~85% do texto das issues migradas do Launchpad, porque o relato original fica dentro de ```. Na versão limpa esses projetos saem de qualquer forma.

`05_clean_data.py` gera as versões limpas com os **mesmos splits** (cada issue continua no mesmo train/val/test). As originais ficam intactas para a reprodução.

| Regra | Agile | NEO |
|---|---|---|
| Projeto com SP inválido | 0 | −3.954 |
| SP = 0 | 0 | −148 |
| Duplicata com SP conflitante (grupo inteiro) | −67 | −39 |
| Duplicata idêntica (mantém uma) | −141 | −136 |
| Sem descrição (vazia, só URL ou só markup) | −2.219 | −1.684 |
| **Issues na versão limpa** | **20.886 (89,6%)** | **14.513 (70,9%)** |
| Test set / avaliação | 2.078 / 1.031 | 1.437 / 716 |
| SP média, mediana e desvio | 6,26 / 4 / 9,95 | 3,01 / 2 / 4,29 (antes 5,99 / 3 / 17,15) |

```bash
python repro/05_clean_data.py
python repro/50_quality_judge.py --engine jev --dataset agile_limpo neo_limpo --subset test   # reaproveita o já julgado
python repro/12_run_baselines.py
python repro/10_run_ml.py --dataset agile_limpo neo_limpo
python repro/90_evaluate.py
```

O log com cada issue removida e a regra que a removeu fica em `data/processed/limpeza_log.csv`.

## Extensão: qualidade da entrada (Jev / Laya)

Pergunta: as issues têm qualidade suficiente para alguém, humano ou modelo, estimar
o esforço a partir delas? São três camadas, da mais barata para a mais cara.

**1. Sinais objetivos** (sem modelo, segundos): `python repro/51_quality_report.py signals`

| % das issues | Agile | NEO |
|---|---|---|
| Descrição vazia | 9,6 | 8,0 |
| Menos de 15 palavras | 14,1 | 15,7 |
| No formato "As a … I want …" | 1,8 | 1,9 |
| Com "so that" | 3,4 | 2,6 |
| Com critérios de aceitação no texto | 0,4 | 19,7 |

Apesar do nome "user story dataset", **só ~2% das issues são user stories** no
formato clássico. O resto são bugs, tarefas técnicas e melhorias de Jira e GitLab.

**2. Juiz Jev ou Laya** (`50_quality_judge.py`). O Jev (TypeSafe AI) é o juiz de
decisões de Li et al., *JEV-as-a-Judge: Accept When Confident, Escalate When Unsure*
(arXiv 2609.26550). Ele não gera texto: devolve probabilidades para perguntas tipadas.
O Laya (Convai Innovations, Apache 2.0) é um modelo aberto com **a mesma API**, que
roda localmente e de graça. As perguntas ficam em `quality_criteria.json` (editável):
tipo da issue, clareza do objetivo, informação suficiente para estimar, tamanho do
escopo, critérios de aceitação e ambiguidade, inspiradas no framework QUS e no INVEST.

```bash
# Laya no Colab (recomendado: GPU, minutos) -> seção 5b do colab_reproducao.ipynb
# Laya local (CPU, ~1 s ou mais por issue). Num cmd separado:
pip install "laya[serve]"
set LAYA_MODELS=typed-decisions
laya-serve
# no cmd de trabalho:
python repro/50_quality_judge.py --engine laya --dataset agile --limit 20   # teste
python repro/50_quality_judge.py --engine laya                               # 2.188 issues

# Jev (API paga)
set TYPESAFE_API_KEY=sua_chave
python repro/50_quality_judge.py --engine jev
```

Por padrão o script julga só as 2.188 issues de avaliação, as mesmas em que medimos o
erro dos modelos. `--subset all` julga o dataset inteiro. O custo do Jev é desprezível:
pelo preço publicado no artigo (~US$ 0,042 por milhão de tokens de entrada), as 2.188 issues
dariam centavos de dólar. Mas exige conta na TypeSafe; confira o preço atual no site.

**Notas gerais.** Cada issue recebe duas notas de 0 a 100 (`see/quality.py`):
- `indice_qualidade`, composto e transparente: média de clareza, dá para estimar,
  critérios de aceitação e ausência de ambiguidade. Escopo e tipo não entram, porque
  descrevem a issue e não a qualidade do texto.
- `nota_geral_juiz`, avaliação holística do próprio juiz (pergunta `overall_quality`, 0 a 4).

O `report` mostra, por dataset, média, mediana e % de issues ruins (< 40), regulares
(40–70) e boas (≥ 70), além da correlação entre as duas notas. O componente "critérios de
aceitação" puxa o índice do Agile para baixo, porque quase nenhuma issue de lá os tem
(0,4%); isso é uma característica real dos dados, mas aparece no índice.

**3. Validação humana** (obrigatória antes de usar no TCC). O próprio artigo do Jev
mostra que esse tipo de juiz é bom quando a resposta "está escrita" no texto e fica perto
do acaso ao julgar a qualidade de prosa sem referência. Perguntas como "tem critérios de
aceitação?" são do primeiro tipo; "está claro?" fica perigosamente perto do segundo. O
artigo recomenda ~100 rótulos locais:

```bash
python repro/51_quality_report.py sample   # gera results/quality/rotulagem_humana.xlsx (100 issues)
#   -> preencha as colunas amarelas (há listas suspensas e uma aba de instruções)
python repro/51_quality_report.py report   # concordância (kappa), limiar de confiança e erro × qualidade
```

O `report` também cruza a qualidade com o erro dos modelos (`erro_por_qualidade.csv`).
Um primeiro resultado, só com os sinais objetivos: no Agile, as issues sem descrição têm
MAE maior (mediana do projeto 4,85 contra 3,80), como Abadeer & Sabetzadeh observaram.
No NEO acontece o contrário (1,77 contra 2,72), provavelmente porque lá as issues sem
descrição são tarefas pequenas. A coluna `SP_medio` ajuda a separar os dois efeitos.

## Como as etapas se encaixam

Cada etapa é um script independente. Elas se comunicam só por arquivos, então
cada uma pode rodar num computador diferente.

| Etapa | Script | Onde rodar | Tempo aproximado |
|---|---|---|---|
| 1. Baixar dados | `01_download_data.py` | qualquer um | 1 min |
| 2. Preparar | `02_prepare_data.py` | qualquer um | 1 min |
| 2b. Características dos repositórios (extensão, só NEO) | `04_repo_features.py` | qualquer um | segundos |
| 2c. Versões limpas (extensão) | `05_clean_data.py` | qualquer um | segundos |
| 3. Splits | `03_make_splits.py` | **já feito** (versionado em `data/splits/`) | — |
| 4. Textos perturbados | `40_perturb_texts.py` | **já feito** (versionado em `data/perturbed/`) | — |
| Baselines ingênuos (média/mediana) | `12_run_baselines.py` | qualquer um | segundos |
| Qualidade da entrada (extensão) | `50_quality_judge.py`, `51_quality_report.py` | Colab (Laya na GPU), notebook ou API do Jev | minutos a horas |
| ML clássico | `10_run_ml.py` | notebook (CPU) | ~1 h por dataset no within (SVR é o mais lento) |
| ML fiel ao código | `11_run_ml_original_code.py` | CPU com ≥ 4 GB de RAM livre | KNN/LR em 5 min; SVR > 1 h |
| Transformers | `20_run_transformers.py` | **Google Colab** (GPU grátis) — `colab_reproducao.ipynb` | horas |
| Transformers fiel ao código | `21_run_transformers_original_code.py` | Google Colab | horas |
| LLMs | `30_run_llm.py` | notebook, outro PC ou Colab | depende do provedor |
| Métricas | `90_evaluate.py` | qualquer um | segundos |
| LME, degradação, custos, figuras | `91_analysis.py` | qualquer um | segundos |

Todos os resultados vão para `results/predictions/{dataset}/{protocolo}/{modelo}.csv`,
sempre no mesmo formato (`issue_id, y_true, y_pred, ...`). As métricas são
calculadas **só** a partir desses arquivos.

## Primeiro uso

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r repro/requirements.txt
python -m nltk.downloader wordnet omw-1.4 averaged_perceptron_tagger_eng   # só se for refazer a etapa 40
python repro/01_download_data.py
python repro/02_prepare_data.py
python repro/04_repo_features.py      # extensão: características dos repositórios do NEO
python repro/12_run_baselines.py      # média e mediana (segundos)
```

## ML clássico

```bash
python repro/10_run_ml.py                                   # within, 5 modelos, 2 datasets
python repro/10_run_ml.py --protocol cross lopo perturb     # generalização e robustez
python repro/10_run_ml.py --svd 3000                        # variante do código original
```

## LLMs (sem custo)

O script grava cada resposta assim que ela chega (`results/llm_raw/*.jsonl`).
Se cair a internet, acabar a cota ou você fechar o notebook, **rode o mesmo comando
de novo** e ele continua de onde parou.

```bash
# Ver um prompt de exemplo, sem chamar nenhum modelo
python repro/30_run_llm.py --provider ollama --model x --dataset agile --dry-run

# Ollama local
ollama pull llama3.2:3b
python repro/30_run_llm.py --provider ollama --model llama3.2:3b

# API do Gemini (crie a chave em https://aistudio.google.com)
export GEMINI_API_KEY=...          # Windows PowerShell: $env:GEMINI_API_KEY="..."
python repro/30_run_llm.py --provider gemini --model gemini-2.5-flash --rpm 8

# Groq
export GROQ_API_KEY=...
python repro/30_run_llm.py --provider groq --model openai/gpt-oss-20b --rpm 25

# Teste rápido com 20 issues
python repro/30_run_llm.py --provider ollama --model llama3.2:3b --dataset agile --limit 20
```

Protocolos: `within` (principal), `cross`, `lopo`, `fewshot` (template FS1–FS3,
exemplos de outros projetos, como no artigo), `perturb` (textos perturbados).
Use `--template ZS1 ... ZS10` para o teste de sensibilidade ao prompt.
`--shots-from same_project` é uma variação nossa: few-shot com exemplos do
próprio projeto.

**Quantas chamadas cada experimento custa (por modelo):**

| Experimento | Agile | NEO | Total |
|---|---|---|---|
| within (principal) | 1.166 | 1.022 | 2.188 |
| cross | 488 | 425 | 913 |
| LOPO | 510 | 493 | 1.003 |
| few-shot | 100 | 100 | 200 |
| perturbação (3 seeds) | 3.498 | 3.066 | 6.564 |
| sensibilidade (10 templates) | 11.660 | 10.220 | 21.880 |

**Limites das camadas gratuitas** (conferidos em outubro de 2026; eles mudam com frequência):
- **OpenRouter** (modelos `:free`): 20 req/min e **só 50 req/dia** sem créditos comprados. Não serve para o experimento principal.
- **Groq**: a página de limites lista hoje modelos abertos como `openai/gpt-oss-20b`, `openai/gpt-oss-120b` e `qwen/qwen3.8-27b`, com cerca de 30 req/min e 1.000 req/dia. Não lista mais os Llama de chat. Confira em https://console.groq.com/docs/rate-limits
- **Gemini**: os limites gratuitos por modelo aparecem só no Google AI Studio, valem por projeto e zeram à meia-noite do horário do Pacífico.
- **Ollama** não tem limite: no notebook sem GPU, modelos de 1–3B são viáveis; modelos de ~8B rodam bem na GPU T4 do Colab (seção 5 do notebook).

**Juntando resultados de outro computador:** rode o script lá, faça commit da
pasta `results/llm_raw/` e dê `git pull` aqui (ou copie os `.jsonl` para a
mesma pasta). Depois:

```bash
python repro/30_run_llm.py --provider ollama --model llama3.2:3b --export-only
```

## Transformers (Colab)

Abra `repro/colab_reproducao.ipynb` no Colab, troque `REPO_URL` pelo seu fork e
rode as células. Checkpoints e resultados ficam no seu Google Drive: se a sessão
cair, basta rodar de novo.

## Métricas e análise

```bash
python repro/90_evaluate.py     # results/tables/metrics.csv
python repro/91_analysis.py     # LME, degradação, custos e figuras
```

## Decisões tomadas onde o artigo é omisso ou inconsistente

| Tema | Artigo | Código original | Aqui |
|---|---|---|---|
| Subconjunto do NEO | "curado", 20.479 stories, 25 projetos, critério não dito | não trata o NEO | NEO **sem o projeto GitLab (278964)**: dá exatamente 20.479 linhas e as estatísticas da Tabela 4, mas com 33 projetos. Removemos 5 issues sem SP, ficando 20.474 |
| Split within | 80/10/10 por projeto, "estratificação adaptativa" | aleatório global 70/30 (ML) e 80/20 (Transformers), sem validação | 80/10/10 por projeto, estratificado por SP (amostragem sistemática) |
| Subconjunto de avaliação | 50% do test (~5% do total) | não existe | 50% do test, estratificado por SP |
| Cross-project | 80/10/10 dos projetos; avalia em ~2% do total | não existe | igual; projetos sorteados com seed 42 |
| TF-IDF | 30k termos, uni+bigramas | matriz ajustada no **dataset inteiro** (vaza o teste) | ajustado **só no treino** |
| Redução de dimensionalidade | não menciona | SVD de 3.000 componentes | sem SVD por padrão; `--svd 3000` reproduz o código |
| Early stopping | "até 20 épocas com early stopping" | 20 épocas fixas, sem validação | MAE na validação, paciência 3 |
| GPT2SP | Tabela 3 quase igual ao GPT-2 | não implementado | GPT-2 + regressor MLP de 3 camadas (como no GPT2SP original) |
| Few-shot | 2–4 exemplos sorteados do treino (método) ou fixos (apêndice); só cross-project | templates com placeholders | 2–4 exemplos reais de outros projetos, sorteio estável por issue |
| Temperatura dos LLMs | "padrão" | — | 0 |
| Ruído de caractere | "mínimo" (texto) / 5% por caractere (Algoritmo 1) | — | 5% por caractere (`--char-p` muda) |
| PRED(25) com SP = 0 | não discute | divide por zero | conta como erro (como o código); afeta 148 issues do NEO |
| Resposta do LLM sem número | não discute | — | conta em `n_invalid` e recebe a mediana do treino |
| LME | sobre MAE, família fixa, modelo aleatório | **valores digitados à mão**, diferentes das Tabelas 14 e 15 | calculado das predições |

## Achados sobre o repositório original

1. `ml_for_story_point.py` termina com um dicionário "INSERT YOUR RESULTS HERE",
   cujos números não batem com as Tabelas 14 e 15. Nele os LLMs aparecem melhores
   que os Transformers (GPT-4-Turbo com MAE 3,742), o contrário da conclusão do artigo.
   Porém, alimentando o nosso `91_analysis.py` com os MAEs das Tabelas 14 e 15, o LME
   devolve **exatamente** os coeficientes publicados (μ = 5,754; β_Trans = −1,774;
   β_LLM = 3,955). Ou seja, o LME do artigo foi feito sobre os valores das tabelas, e o
   dicionário do repositório é um resto de rascunho. Nesse ajuste a variância do efeito
   aleatório de modelo sai 0: com só 2 observações por modelo, o LME equivale a uma
   regressão comum com 26 pontos.
2. `prepareData.py`, `helper.py`, `preprocessing.py` e a matriz TF-IDF são cópias
   do repositório de L. Truong (mrthlinh) e dependem de MongoDB e Google Drive.
3. O Agile tem 17 códigos de projeto, e não os 16 citados.
4. A regressão linear sobre as 30 mil features TF-IDF sem regularização é mal
   condicionada (mais features que exemplos) e chega a MAE ≈ 25. O MAE 5,93
   publicado só é plausível com a redução de dimensionalidade do código.
