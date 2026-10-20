# converte cada linha do costs.csv para UTF-8 (algumas foram gravadas em cp1252 no Windows)
p = "repro/results/costs.csv"
linhas = []
for b in open(p, "rb").read().split(b"\n"):
    try:
        linhas.append(b.decode("utf-8"))
    except UnicodeDecodeError:
        linhas.append(b.decode("cp1252"))
open(p, "w", encoding="utf-8", newline="").write("\n".join(linhas))
print("ok:", len(linhas), "linhas")