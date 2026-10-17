"""Etapa Transformers — BERT, RoBERTa, GPT-2 e GPT2SP com cabeça de regressão (Tabela 3).

Precisa de GPU: o caminho gratuito é o Google Colab (veja colab_transformers.ipynb).

    python repro/20_run_transformers.py                              # within, 4 modelos, 2 datasets
    python repro/20_run_transformers.py --models BERT --dataset agile
    python repro/20_run_transformers.py --protocol cross
    python repro/20_run_transformers.py --ckpt-dir /content/drive/MyDrive/see_ckpt   # Colab + Drive

Configuração (Tabela 3 e Seção III-B.2 do artigo):
  entrada  título + descrição, máx. 128 tokens ([CLS] título [SEP] descrição [SEP] nos encoders)
  treino   AdamW lr=2e-5, perda MSE, até 20 épocas com early stopping
  BERT     bert-base-uncased, pooling no [CLS], cabeça linear, batch 8
  RoBERTa  roberta-base, mean pooling, cabeça linear, batch 8, gradient clipping 1.0
  GPT-2    gpt2, pooling por média, cabeça linear, batch 16, pad = EOS
  GPT2SP   gpt2, pooling por média, regressor MLP de 3 camadas (como no GPT2SP original
           de Fu & Tantithamthavorn), batch 16. A Tabela 3 descreve GPT-2 e GPT2SP de
           forma quase idêntica; a cabeça MLP é o que os diferencia aqui.

Decisões que o artigo não especifica (registradas no README):
  - early stopping pela MAE na validação, paciência 3;
  - alvo padronizado (z-score do treino) durante o treino, como no código original do RoBERTa.

Retomável: se a sessão do Colab cair, rode o mesmo comando; ele continua da
última época salva e pula modelos que já têm predições.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config  # noqa: E402
from see.io import load_merged, log_cost, predictions_path, save_predictions  # noqa: E402
from see.text import clean_light  # noqa: E402

MODELS = {
    "BERT": dict(hf="bert-base-uncased", pooling="cls", head="linear", batch=8, clip=None),
    "RoBERTa": dict(hf="roberta-base", pooling="mean", head="linear", batch=8, clip=1.0),
    "GPT-2": dict(hf="gpt2", pooling="mean", head="linear", batch=16, clip=None),
    "GPT2SP": dict(hf="gpt2", pooling="mean", head="mlp3", batch=16, clip=None),
}


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------
def build_model(spec, hf_name):
    import torch
    from torch import nn
    from transformers import AutoModel

    class Regressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(hf_name)
            h = self.encoder.config.hidden_size
            if spec["head"] == "linear":
                self.head = nn.Linear(h, 1)
            else:  # mlp3
                self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.1),
                                          nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, 1))
            self.pooling = spec["pooling"]

        def forward(self, input_ids, attention_mask):
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            if self.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                m = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)
            return self.head(pooled).squeeze(-1)

    return Regressor()


def build_tokenizer(hf_name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_name)
    if tok.pad_token is None:  # GPT-2 não tem pad: usa EOS (Tabela 3)
        tok.pad_token = tok.eos_token
    return tok


def encode(tok, titles, descs, max_len):
    if tok.cls_token is not None:  # encoders: par de sentenças -> [CLS] t [SEP] d [SEP]
        return tok(list(titles), list(descs), truncation="longest_first", max_length=max_len,
                   padding="max_length", return_tensors="pt")
    joined = [f"{t}{tok.eos_token}{d}" for t, d in zip(titles, descs)]
    return tok(joined, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")


# --------------------------------------------------------------------------
# Treino / predição
# --------------------------------------------------------------------------
def batches(enc, y, batch, shuffle, rng):
    n = enc["input_ids"].shape[0]
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for s in range(0, n, batch):
        b = idx[s:s + batch]
        yield enc["input_ids"][b], enc["attention_mask"][b], (None if y is None else y[b])


def predict(model, enc, batch, device, use_amp):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for ids, mask, _ in batches(enc, None, batch * 4, False, None):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out.append(model(ids.to(device), mask.to(device)).float().cpu())
    return torch.cat(out).numpy()


def train_eval(spec, name, hf_name, train_df, val_df, pred_sets, args, ckpt_dir: Path, device):
    """Treina com early stopping e devolve {nome_do_conjunto: y_pred}, tempos."""
    import torch
    tok = build_tokenizer(hf_name)
    enc_tr = encode(tok, train_df.title_c, train_df.desc_c, args.max_len)
    enc_va = encode(tok, val_df.title_c, val_df.desc_c, args.max_len)
    y_tr = train_df.sp.to_numpy(dtype=np.float32)
    mu, sd = float(y_tr.mean()), float(y_tr.std() or 1.0)
    y_tr_z = torch.tensor((y_tr - mu) / sd)

    torch.manual_seed(config.SEED)
    model = build_model(spec, hf_name).to(device)
    if model.encoder.config.model_type == "gpt2":
        model.encoder.config.pad_token_id = tok.pad_token_id
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last, best = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    state = {"epoch": 0, "best_mae": math.inf, "bad": 0, "train_seconds": 0.0}
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        print(f"    retomando da época {state['epoch']} (melhor MAE val {state['best_mae']:.3f})", flush=True)

    rng = np.random.default_rng(config.SEED + state["epoch"])
    while state["epoch"] < args.epochs and state["bad"] < args.patience:
        t0 = time.perf_counter()
        model.train()
        total, nb = 0.0, 0
        for ids, mask, yb in batches(enc_tr, y_tr_z, spec["batch"], True, rng):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = torch.nn.functional.mse_loss(model(ids.to(device), mask.to(device)).float(), yb.to(device))
            scaler.scale(loss).backward()
            if spec["clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), spec["clip"])
            scaler.step(opt); scaler.update()
            total, nb = total + loss.item(), nb + 1
        val_pred = predict(model, enc_va, spec["batch"], device, use_amp) * sd + mu
        val_mae = float(np.mean(np.abs(val_pred - val_df.sp.to_numpy())))
        state["epoch"] += 1
        state["train_seconds"] += time.perf_counter() - t0
        if val_mae < state["best_mae"] - 1e-4:
            state["best_mae"], state["bad"] = val_mae, 0
            torch.save(model.state_dict(), best)
        else:
            state["bad"] += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "state": state}, last)
        print(f"    época {state['epoch']:>2}: loss={total / max(nb, 1):.4f}  MAE val={val_mae:.3f}"
              f"  ({time.perf_counter() - t0:.0f}s)", flush=True)

    model.load_state_dict(torch.load(best, map_location=device))
    preds, t_inf, n_inf = {}, 0.0, 0
    for key, pdf in pred_sets.items():
        enc = encode(tok, pdf.title_c, pdf.desc_c, args.max_len)
        t0 = time.perf_counter()
        preds[key] = predict(model, enc, spec["batch"], device, use_amp) * sd + mu
        t_inf += time.perf_counter() - t0
        n_inf += len(pdf)
    return preds, state, t_inf / max(n_inf, 1), mu


def with_clean_text(df):
    df = df.copy()
    df["title_c"] = df.title.map(clean_light)
    df["desc_c"] = df.description.map(clean_light)
    return df


def perturbed_sets(df, dataset):
    sets = {}
    for f in sorted((config.ROOT / "data" / "perturbed").glob(f"{dataset}_s*.csv")):
        p = pd.read_csv(f, dtype={"issue_id": str}, keep_default_na=False)
        p = p.merge(df[["issue_id", "project", "sp"]], on="issue_id")
        p["title_c"], p["desc_c"] = p.title, p.description
        sets[f"perturb_s{f.stem.split('_s')[-1]}"] = p
    return sets


def folds(df, protocol, lopo_projects):
    if protocol == "within":
        yield "all", df[df.within == "train"], df[df.within == "val"], df[df.within == "test"]
    elif protocol == "cross":
        yield "all", df[df.cross == "train"], df[df.cross == "val"], df[df.cross_eval == 1]
    elif protocol == "lopo":
        for p in lopo_projects or sorted(df.project.unique()):
            rest = df[df.project != p]
            yield p, rest[rest.within != "val"], rest[rest.within == "val"], df[(df.project == p) & (df.lopo_eval == 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=config.DATASETS, choices=config.ALL_DATASETS)
    ap.add_argument("--protocol", nargs="+", default=["within"], choices=["within", "cross", "lopo"])
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lopo-projects", nargs="*", help="limitar o LOPO a alguns projetos (é caro)")
    ap.add_argument("--ckpt-dir", default=str(config.RESULTS / "checkpoints"))
    ap.add_argument("--no-perturb", action="store_true", help="não prever os textos perturbados no within")
    ap.add_argument("--force", action="store_true", help="refazer mesmo se as predições já existirem")
    ap.add_argument("--hf-override", help="(teste) usar outro modelo do Hugging Face no lugar do original")
    ap.add_argument("--max-train", type=int, default=0, help="(teste) usar só N exemplos de treino")
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Dispositivo: {device}" + ("" if device.type != "cpu" else
          "  (sem GPU: o treino completo vai levar muitas horas; use o Colab)"))

    for dataset in args.dataset:
        df = with_clean_text(load_merged(dataset))
        for protocol in args.protocol:
            for name in args.models:
                spec = MODELS[name]
                hf_name = args.hf_override or spec["hf"]
                if predictions_path(dataset, protocol, name).exists() and not args.force:
                    print(f"== {dataset}/{protocol}/{name}: predições já existem, pulando (use --force)")
                    continue
                parts, perturb_parts = [], {}
                for fold, tr, va, pr in folds(df, protocol, args.lopo_projects):
                    if args.max_train:
                        tr = tr.sample(min(args.max_train, len(tr)), random_state=config.SEED)
                    print(f"== {dataset}/{protocol}/{name} fold={fold}: treino={len(tr)} val={len(va)} pred={len(pr)}",
                          flush=True)
                    sets = {"main": pr}
                    if protocol == "within" and not args.no_perturb:
                        sets.update(perturbed_sets(df, dataset))
                    ckpt = Path(args.ckpt_dir) / dataset / protocol / str(fold) / name
                    preds, state, t_inf, mu = train_eval(spec, name, hf_name, tr, va, sets, args, ckpt, device)
                    out = pr[["issue_id", "project", "sp"]].rename(columns={"sp": "y_true"}).copy()
                    out["y_pred"], out["fold"], out["family"], out["y_train_mean"] = preds["main"], fold, "Transformer", mu
                    out["best_epoch_val_mae"], out["epochs_run"] = state["best_mae"], state["epoch"]
                    parts.append(out)
                    for key, p in sets.items():
                        if key != "main":
                            o = p[["issue_id", "project", "sp"]].rename(columns={"sp": "y_true"}).copy()
                            o["y_pred"], o["fold"], o["family"] = preds[key], fold, "Transformer"
                            perturb_parts[key] = o
                    log_cost(dataset, protocol, "Transformer", name, n_train=len(tr), n_pred=len(pr),
                             train_seconds=round(state["train_seconds"], 1), infer_seconds=round(t_inf, 5),
                             notes=f"fold={fold}; device={device}; epochs={state['epoch']}; infer=s/issue")
                    if not args.max_train:  # libera espaço (o checkpoint do BERT tem ~1,3 GB)
                        for f in ckpt.glob("*.pt"):
                            f.unlink()
                path = save_predictions(pd.concat(parts), dataset, protocol, name)
                for key, o in perturb_parts.items():
                    save_predictions(o, dataset, key, name)
                print(f"  salvo: {path.relative_to(config.ROOT)}", flush=True)


if __name__ == "__main__":
    main()
