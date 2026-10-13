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
