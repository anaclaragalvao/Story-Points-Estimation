"""Variante FIEL AO CÓDIGO — reproduz trans_based_for_story_point.py (só Agile, precisa de GPU).

    python repro/21_run_transformers_original_code.py --models BERT RoBERTa GPT-2

Mesma ideia de 11_run_ml_original_code.py: o objetivo é verificar se os números
de Transformers da Tabela 14 (BERT 4,112; RoBERTa 3,781; GPT-2 3,478) saem do
código publicado. O que o código faz, diferente do texto do artigo:

  - texto: coluna `concat` do arquivo original (título + descrição), sem limpeza;
  - split aleatório global 80/20 (random_state=42), teste com 4.663 issues;
  - 20 épocas fixas, sem validação nem early stopping;
  - BERT     [CLS], lr 2e-5, batch 8, rótulos brutos;
  - RoBERTa  mean pooling com máscara, lr **1e-5** (o artigo diz 2e-5), batch 8,
             rótulos padronizados com média/desvio do dataset INTEIRO;
  - GPT-2    média sobre TODOS os tokens, inclusive padding, lr 2e-5, batch 16, rótulos brutos;
  - GPT2SP   não existe no código; não há como reproduzir.

Treino em GPU não é determinístico: espere valores próximos, não idênticos.
Saída: results/predictions/agile/original_code/{modelo}.csv
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config  # noqa: E402
from see.io import load_processed, log_cost, predictions_path, save_predictions  # noqa: E402
from see.metrics import all_metrics  # noqa: E402

_spec = importlib.util.spec_from_file_location("tr", Path(__file__).with_name("20_run_transformers.py"))
tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tr)

ORIGINAL = {
    "BERT": dict(hf="bert-base-uncased", pooling="cls", lr=2e-5, batch=8, scale=False),
    "RoBERTa": dict(hf="roberta-base", pooling="mean", lr=1e-5, batch=8, scale=True),
    "GPT-2": dict(hf="gpt2", pooling="mean_unmasked", lr=2e-5, batch=16, scale=False),
}
PAPER_TABLE14 = {"BERT": 4.112, "RoBERTa": 3.781, "GPT-2": 3.478}


def build(spec, hf_name):
    import torch
    from torch import nn
    from transformers import AutoModel

    class Regressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(hf_name)
            self.head = nn.Linear(self.encoder.config.hidden_size, 1)

        def forward(self, ids, mask):
            h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            if spec["pooling"] == "cls":
                pooled = h[:, 0]
            elif spec["pooling"] == "mean":
                m = mask.unsqueeze(-1).to(h.dtype)
                pooled = (h * m).sum(1) / m.sum(1)
            else:  # mean_unmasked: bug do original, inclui padding
                pooled = h.mean(dim=1)
            return self.head(pooled).squeeze(-1)

    return Regressor()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(ORIGINAL), choices=list(ORIGINAL))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--ckpt-dir", default=str(config.RESULTS / "checkpoints"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--hf-override", help="(teste) outro modelo do Hugging Face")
    ap.add_argument("--max-train", type=int, default=0, help="(teste) só N exemplos de treino")
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_processed("agile")
    raw = pd.read_csv(config.DATA_RAW / "agile" / "data.csv", low_memory=False)
    assert len(raw) == len(df)
    texts = raw["concat"].astype(str).to_numpy()
    y_all = df["sp"].to_numpy(dtype=np.float32)
    i_tr, i_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    if args.max_train:
        i_tr = i_tr[: args.max_train]

    for name in args.models:
        spec = ORIGINAL[name]
        if predictions_path("agile", "original_code", name).exists() and not args.force:
            print(f"{name}: já existe, pulando (use --force)")
            continue
        hf_name = args.hf_override or spec["hf"]
        tok = tr.build_tokenizer(hf_name)
        enc = lambda t: tok(list(t), truncation=True, max_length=128, padding="max_length", return_tensors="pt")  # noqa: E731
        enc_tr, enc_te = enc(texts[i_tr]), enc(texts[i_te])
        mu, sd = (float(y_all.mean()), float(y_all.std())) if spec["scale"] else (0.0, 1.0)
        y_tr = torch.tensor((y_all[i_tr] - mu) / sd)

        torch.manual_seed(config.SEED)
        model = build(spec, hf_name).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"])
        ckpt = Path(args.ckpt_dir) / "agile" / "original_code" / name / "last.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        start, train_s = 0, 0.0
        if ckpt.exists():
            ck = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            start, train_s = ck["epoch"], ck["train_s"]
            print(f"{name}: retomando da época {start}")
        rng = np.random.default_rng(config.SEED + start)
        for epoch in range(start, args.epochs):
            t0 = time.perf_counter()
            model.train()
            total, nb = 0.0, 0
            for ids, mask, yb in tr.batches(enc_tr, y_tr, spec["batch"], True, rng):
                opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(ids.to(device), mask.to(device)), yb.to(device))
                loss.backward(); opt.step()
                total, nb = total + loss.item(), nb + 1
            train_s += time.perf_counter() - t0
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1,
                        "train_s": train_s}, ckpt)
            print(f"  {name} época {epoch + 1}/{args.epochs}: loss={total / max(nb, 1):.4f}", flush=True)

        t0 = time.perf_counter()
        y_hat = tr.predict(model, enc_te, spec["batch"], device, False) * sd + mu
        t_inf = (time.perf_counter() - t0) / len(i_te)
        out = df.iloc[i_te][["issue_id", "project"]].copy()
        out["y_true"], out["y_pred"], out["family"], out["fold"] = y_all[i_te], y_hat, "Transformer", "all"
        save_predictions(out, "agile", "original_code", name)
        log_cost("agile", "original_code", "Transformer", name, n_train=len(i_tr), n_pred=len(i_te),
                 train_seconds=round(train_s, 1), infer_seconds=round(t_inf, 5), notes=f"device={device}")
        m = all_metrics(y_all[i_te], y_hat)
        print(f"{name}: MAE={m['MAE']:.3f} MdAE={m['MdAE']:.3f} RMSE={m['RMSE']:.3f} PRED25={m['PRED25']:.3f}"
              f"   | artigo (Tabela 14): MAE={PAPER_TABLE14[name]}", flush=True)
        if not args.max_train:
            ckpt.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
