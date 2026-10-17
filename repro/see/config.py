"""Configuração central da reprodução.

Tudo que define o experimento (caminhos, seeds, versões dos dados, tamanhos de
amostra) fica aqui, para que cada etapa possa ser rodada separadamente — e em
computadores diferentes — produzindo os mesmos arquivos.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Caminhos (relativos à pasta repro/)
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
RESULTS = ROOT / "results"
PREDICTIONS = RESULTS / "predictions"
LLM_RAW = RESULTS / "llm_raw"          # respostas brutas dos LLMs (JSONL, retomável)
TABLES = RESULTS / "tables"
PROMPTS_DIR = ROOT.parent / "Prompts"  # templates originais do repositório SEE-LLM

# --------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------
SEED = 42                          # seed principal (o artigo cita seed=42)
ROBUSTNESS_SEEDS = [42, 123, 2025]  # Algoritmo 1 do artigo

# --------------------------------------------------------------------------
# Fontes dos dados — fixadas por commit para a reprodução não mudar sozinha
# --------------------------------------------------------------------------
AGILE_REPO = "mrthlinh/Agile-User-Story-Point-Estimation"
AGILE_COMMIT = "0af824fd1594fa76f5566064ded8bcf2cfd57478"
AGILE_FILE = "data_csv/data"
AGILE_SHA256 = "ccc64d6b41852d3ed179cbbee17a6413b629d505442ace962849a1e53fe1e99f"

NEO_REPO = "giseldo/neodataset"
NEO_COMMIT = "35503adb4475ddf7fd31793f6971eb1be12e8226"
NEO_PROJECT_IDS = [
    "10152778", "10171263", "10171270", "10171280", "10174980", "12450835",
    "12584701", "12894267", "1304532", "14052249", "14976868", "15502567",
    "1714548", "19921167", "2009901", "21149814", "23285197", "250833",
    "2670515", "278964", "28419588", "28644964", "28847821", "3828396",
    "3836952", "4456656", "5261717", "6206924", "7071551", "7128869",
    "734943", "7603319", "7764", "7776928",
]
# O artigo usa um "subconjunto curado" de 20.479 stories sem dizer o critério.
# Removendo apenas o projeto GitLab (278964) obtemos exatamente 20.479 linhas,
# com média 5,99 / mediana 3 / desvio 17,15 — compatível com a Tabela 4.
# (São 33 projetos, e não 25 como o artigo declara.)
NEO_EXCLUDED_PROJECTS = ["278964"]

# --------------------------------------------------------------------------
# Protocolo de avaliação (Seção III-B do artigo)
# --------------------------------------------------------------------------
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.8, 0.1, 0.1
EVAL_SUBSET_FRAC = 0.5          # within: avalia em 50% do test set (~5% do total)
CROSS_EVAL_FRAC = 0.2           # cross-project: 20% do test set (~2% do total)
LOPO_SAMPLES = {"agile": 30, "neo": 15}   # amostras por projeto no LOPO
FEWSHOT_N_INSTANCES = 100       # few-shot: 100 instâncias do(s) projeto(s) de teste
FEWSHOT_K_RANGE = (2, 4)        # 2 a 4 exemplos por prompt

DATASETS = ["agile", "neo"]

# --------------------------------------------------------------------------
# EXTENSÃO — contexto do repositório (etapas 05–07; não faz parte do artigo)
# --------------------------------------------------------------------------
# CSVs originais de Choetkiertikul et al. (2019). Trazem a chave Jira de cada issue
# (MDL-1626, XD-1...), que o arquivo do mrthlinh usado na reprodução não tem.
AGILE_ORIG_REPO = "morakotch/datasets"
AGILE_ORIG_COMMIT = "133a5f59e7699a5612cbbc036b8d111844ad12cd"
AGILE_ORIG_DIR = "storypoint/IEEE TSE2018/dataset"
AGILE_ORIG_SHA256 = {
    "appceleratorstudio": "43f2de728c4251084782cc13bcc339e67e4cbbfc1d6f07be9d2be6b799bbf3b4",
    "aptanastudio": "c7af639b52a5a679dbfb29ac65d5c8558c18af45946932cd4e89a73c2ff34172",
    "bamboo": "90d64e163896168edad7ad5ed4411fcd0da103cf6d9c0bcde67fa97add2f1db9",
    "clover": "8218c05d864d2130e88ae24b6a90f318c5bba3e76692b132dbf51880f6c73736",
    "datamanagement": "5bed6be21916f86aeba375a06e4d2e3466257f5c2be78201e56effe26f0053a8",
    "duracloud": "ca9c2252eea0848cec3724e864ce782f4f1ac16838ebad19128bf67068f306cf",
    "jirasoftware": "2e6f525c0d88ed36985c287e9cdc42bf6800f66be0e4e900854368e98cd5c6aa",
    "mesos": "192814946b6d61e01a2d068cc957eade8402b40a864d6be42032f73aaa7f891e",
    "moodle": "5aff0df8cf96cdaa4295341c6e35fe20c550f088979ca07cee74bc9b8f3397ce",
    "mule": "a4e7cf46f94ba3ece298c24c49eee41c195326d65a6451f40fcf2f6fbe4759a2",
    "mulestudio": "e2a6019d1bcce88845667fe28d7c94212c181c030462b9805953490f756411cc",
    "springxd": "edbef829fbfb9dae56085db0804ce3077aa3f44ca65f95f80909750d301abbc5",
    "talenddataquality": "43d1cfab94d8fdcb6ddd85131a94957e18d3b0335b194fd4030a18e0c49bfaeb",
    "talendesb": "4929b8627e14c8fee520d35ec34c22f1cc0f407729837a4584ccbaad8029c322",
    "titanium": "a1ecb54d8a8a15b9c93f9fbdc5868404719c1df063504321375fdf98106aa9b5",
    "usergrid": "71fabca08e9172cc401ba57ad4dd138dbb17fbe46fcccd883d1452248aed08a6",
}
RAISE_REPOS = ROOT / "raise_repos.json"    # projeto do dataset -> repositórios do GitHub coletados no RAISE
CONTEXT_DIR = DATA_PROCESSED / "context"   # saídas das etapas 06 e 07
