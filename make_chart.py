"""Draw the README chart from results/*.csv (values transcribed from the saved notebook outputs)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t = pd.read_csv("results/titanic_prompting.csv")
h = pd.read_csv("results/rag_hyperparameter_search.csv")
s = pd.read_csv("results/rag_system_comparison.csv")
BLUE, ORANGE, GREEN, GREY = "#1d4ed8", "#c2410c", "#2a7f62", "#9ca3af"

fig, ax = plt.subplots(1, 3, figsize=(16, 5), dpi=150, gridspec_kw={"width_ratios": [1.15, 1, 1.15]})

a = ax[0]
strategies = list(t.strategy.unique()); x = np.arange(len(strategies)); w = .38
for off, m, col in [(-w / 2, "gpt-4o", BLUE), (w / 2, "gpt-4o-mini", GREY)]:
    d = t[t.model == m].set_index("strategy").loc[strategies]
    a.bar(x + off, d.accuracy, w, color=col, label=m)
    for xi, v in zip(x + off, d.accuracy): a.text(xi, v + .01, f"{v:.3f}", ha="center", fontsize=7)
a.set_xticks(x); a.set_xticklabels([s_.replace("-", "-\n", 1) if "-" in s_ else s_ for s_ in strategies], fontsize=8)
a.set_ylim(0, 1); a.set_title("Titanic survival, 500 passengers: accuracy", fontsize=10); a.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)

b = ax[1]
styles = {(300, 30): "--", (300, 60): "-", (500, 30): ":", (500, 60): "-."}
for (ch, ov), g in h.groupby(["chunk_words", "overlap_words"]):
    hi = (ch, ov) == (300, 60)
    b.plot(g.k, g.f1, marker="o", ls=styles[(ch, ov)], lw=2.2 if hi else 1.4, color=ORANGE if hi else "#6b7280", alpha=1,
           label=f"{ch} / {ov}" + ("  (selected)" if hi else ""))
b.set_xscale("log"); b.set_xticks([1, 3, 5, 10]); b.set_xticklabels([1, 3, 5, 10])
b.set_xlabel("k (retrieved chunks)"); b.set_ylabel("answer token-F1"); b.set_ylim(.4, .6)
b.set_title("Baseline RAG grid, 50 dev queries\n(spread across settings is small)", fontsize=10); b.legend(title="chunk / overlap (words)", frameon=False, fontsize=7, title_fontsize=7)

c = ax[2]
systems = ["Baseline RAG", "+ Semantic chunking", "+ Reranking", "+ Query rewriting"]
x = np.arange(4); w = .38
for off, ds, col in [(-w / 2, "WixQA-ExpertWritten", BLUE), (w / 2, "WixQA-Simulated", GREEN)]:
    d = s[s.dataset == ds].set_index("system").loc[systems]
    c.bar(x + off, d.factuality, w, color=col, label=ds.replace("WixQA-", ""))
    for xi, v in zip(x + off, d.factuality): c.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=7)
c.set_xticks(x); c.set_xticklabels(["Baseline", "Semantic\nchunking", "Reranking", "Query\nrewriting"], fontsize=8)
c.set_ylim(0, .5); c.set_title("RAG improvements: LLM-judged factuality\n(100 queries per dataset)", fontsize=10); c.legend(frameon=False, fontsize=8)

for q in ax: [q.spines[k].set_visible(False) for k in ("top", "right")]
plt.tight_layout(); plt.savefig("assets/results_overview.png")
