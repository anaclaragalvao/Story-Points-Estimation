"""Leitura e escrita dos arquivos compartilhados entre as etapas."""
import pandas as pd

from . import config


def load_processed(dataset: str) -> pd.DataFrame:
    path = config.DATA_PROCESSED / f"{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe. Rode antes: python repro/02_prepare_data.py")
    return pd.read_csv(path, dtype={"issue_id": str, "project": str}, keep_default_na=False,
                       na_values={"sp": [""], "created": [""]})


def load_splits(dataset: str) -> pd.DataFrame:
    path = config.SPLITS / f"{dataset}_splits.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe. Rode antes: python repro/03_make_splits.py")
    return pd.read_csv(path, dtype={"issue_id": str, "project": str})


def load_merged(dataset: str) -> pd.DataFrame:
    """Dados + colunas de split, numa tabela só."""
    data = load_processed(dataset)
    splits = load_splits(dataset).drop(columns=["project"])
    df = data.merge(splits, on="issue_id", how="inner", validate="one_to_one")
    assert len(df) == len(data), "splits e dados processados estão dessincronizados"
    return df


COST_FIELDS = ["timestamp", "machine", "dataset", "protocol", "family", "model",
               "n_train", "n_pred", "train_seconds", "infer_seconds", "notes"]


def log_cost(dataset: str, protocol: str, family: str, model: str, **fields):
    """Acrescenta uma linha em results/costs.csv (tempo de treino/inferência — RQ5)."""
    import csv
    import datetime as dt
    import platform
    unknown = set(fields) - set(COST_FIELDS)
    assert not unknown, f"campos desconhecidos: {unknown}"
    path = config.RESULTS / "costs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": dt.datetime.now().isoformat(timespec="seconds"), "machine": platform.node(),
           "dataset": dataset, "protocol": protocol, "family": family, "model": model, **fields}
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COST_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def predictions_path(dataset: str, protocol: str, model: str):
    return config.PREDICTIONS / dataset / protocol / f"{model}.csv"


def save_predictions(df: pd.DataFrame, dataset: str, protocol: str, model: str):
    """Formato único para todas as famílias: issue_id, y_true, y_pred (+ extras)."""
    required = {"issue_id", "y_true", "y_pred"}
    missing = required - set(df.columns)
    assert not missing, f"faltam colunas {missing}"
    path = predictions_path(dataset, protocol, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
