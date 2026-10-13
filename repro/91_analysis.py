"""Análise — LME por família (RQ1), degradação em generalização/robustez (RQ2–RQ4) e custos (RQ5).

    python repro/90_evaluate.py      # primeiro, gera results/tables/metrics.csv
    python repro/91_analysis.py

Saídas em results/tables/:
  lme_within.txt           modelo linear de efeitos mistos, equação (1) do artigo:
                           MAE_ij = mu + beta * Familia_i + (1 | Modelo_j) + e_ij
                           (uma observação por modelo × dataset; ML é a categoria de referência)
  degradation.csv          variação % de MAE e PRED(25) em cross, LOPO e perturbação,
                           relativa ao within, média por família (Tabelas 7 e 8)
  costs_summary.csv        tempos de treino e de inferência por família (Tabela 9)
  figures/*.png            barras por métrica (como a Figura 3), se o matplotlib estiver instalado

Diferente do repositório original, o LME aqui lê as métricas calculadas das
predições — nunca valores digitados à mão.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config  # noqa: E402

FAMILIES = ["ML", "Transformer", "LLM"]


def lme(metrics: pd.DataFrame, out_dir: Path):
    import statsmodels.formula.api as smf
    w = metrics[(metrics.protocol == "within") & metrics.family.isin(FAMILIES)].copy()
    lines = []
    present = [f for f in FAMILIES if f in set(w.family)]
    if len(present) < 2 or w.model.nunique() < 3:
        msg = f"LME precisa de pelo menos 2 famílias e 3 modelos no within (tem: {present}, {w.model.nunique()} modelos)."
        print(msg)
        (out_dir / "lme_within.txt").write_text(msg + "\n")
        return
    w["family"] = pd.Categorical(w.family, categories=present)
    lines.append(f"Observações: {len(w)} (modelos={w.model.nunique()}, datasets={w.dataset.nunique()})\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = smf.mixedlm("MAE ~ family", data=w, groups=w["model"]).fit(reml=False)
        null = smf.mixedlm("MAE ~ 1", data=w, groups=w["model"]).fit(reml=False)
    lines.append(str(res.summary()))
    lr = 2 * (res.llf - null.llf)
    from scipy.stats import chi2
    df_diff = len(present) - 1
    p_lr = chi2.sf(lr, df_diff)
    lines.append(f"\nTeste da razão de verossimilhança (efeito de família): LR={lr:.3f}, gl={df_diff}, p={p_lr:.4g}")
    lines.append(f"\nComparação com o artigo (story points, Seção IV-A.5): mu=5,754; beta_Trans=-1,774; beta_LLM=3,955")
    lines.append(f"Nesta reprodução: mu={res.params['Intercept']:.3f}"
                 + "".join(f"; beta_{f}={res.params.get(f'family[T.{f}]', float('nan')):.3f}"
                           f" (p={res.pvalues.get(f'family[T.{f}]', float('nan')):.4g})" for f in present[1:]))
    text = "\n".join(lines)
    (out_dir / "lme_within.txt").write_text(text + "\n")
    print(text)


def degradation(metrics: pd.DataFrame, out_dir: Path):
    base = metrics[metrics.protocol == "within"].set_index(["dataset", "model"])[["MAE", "PRED25"]]
    m = metrics.copy()
    m["group"] = np.where(m.protocol.str.startswith("perturb_s"), "perturb", m.protocol)
    m = m[m.group.isin(["cross", "lopo", "perturb"])]
    if m.empty:
        print("Ainda não há resultados de cross / lopo / perturb.")
        return
    m = m.groupby(["dataset", "group", "family", "model"], as_index=False)[["MAE", "PRED25"]].mean()
    m = m.join(base, on=["dataset", "model"], rsuffix="_within").dropna(subset=["MAE_within"])
    m["dMAE_%"] = 100 * (m.MAE - m.MAE_within) / m.MAE_within
    m["dPRED25_%"] = 100 * (m.PRED25 - m.PRED25_within) / m.PRED25_within.replace(0, np.nan)
    per_model = m.round(3)
    per_model.to_csv(out_dir / "degradation_per_model.csv", index=False)
    fam = m.groupby(["dataset", "group", "family"], as_index=False)[["dMAE_%", "dPRED25_%"]].mean()
    fam.to_csv(out_dir / "degradation.csv", index=False)
    print("\nDegradação relativa ao within (média por família) — compare com as Tabelas 7 e 8:")
    print(fam.round(1).to_string(index=False))


def costs(out_dir: Path):
    path = config.RESULTS / "costs.csv"
    if not path.exists():
        return
    c = pd.read_csv(path)
    summ = c.groupby(["family", "model"], as_index=False).agg(
        train_seconds=("train_seconds", "median"), infer_seconds=("infer_seconds", "median"), runs=("model", "size"))
    summ.to_csv(out_dir / "costs_summary.csv", index=False)
    print("\nCustos (mediana por execução; infer_seconds = por chamada no ML, por issue nos demais):")
    print(summ.to_string(index=False))


def figures(metrics: pd.DataFrame, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib não instalado: pulando figuras)")
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    w = metrics[metrics.protocol == "within"]
    colors = {"ML": "#4C72B0", "Transformer": "#55A868", "LLM": "#C44E52"}
    for ds, g in w.groupby("dataset"):
        g = g.assign(_o=g.family.map({f: i for i, f in enumerate(FAMILIES)}).fillna(9)).sort_values(["_o", "model"])
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        for ax, metric in zip(axes, ["MAE", "MdAE", "RMSE", "PRED25"]):
            ax.bar(g.model, g[metric], color=[colors.get(f, "gray") for f in g.family])
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=60, labelsize=8)
        fig.suptitle(f"{ds} — within-project (subconjunto de avaliação)")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{ds}_within.png", dpi=150)
        plt.close(fig)
    print(f"\nFiguras em {fig_dir}")


def main():
    path = config.TABLES / "metrics.csv"
    if not path.exists():
        sys.exit("Rode antes: python repro/90_evaluate.py")
    metrics = pd.read_csv(path)
    out_dir = config.TABLES
    print("=== LME (within) ===")
    lme(metrics, out_dir)
    degradation(metrics, out_dir)
    costs(out_dir)
    figures(metrics, out_dir)


if __name__ == "__main__":
    main()
