"""Reproducible figures for the project brief. Reads docs/results/metrics.json."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).parent
M = json.load(open(HERE.parent / "results" / "metrics.json"))
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
ACC, GREY, ORG = "#1f4e9e", "#8a8f98", "#d9822b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})
S = M["systems"]
BASE, LORA = "base few-shot (k=3)", "LoRA fine-tuned (0-shot)"
v = lambda sys, mode, t, m: S[sys]["entity"][mode][t][m]["value"]

def box(ax, x, y, w, h, text, fc="#eef3fb", ec=ACC, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#1b1b1b")

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.3))

# 1. architecture
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.set_xlim(0, 12); ax.set_ylim(0.3, 6.2); ax.axis("off")
box(ax, 0.2, 4.7, 2.6, 1.0, "BC5CDR\n(tner/bc5cdr, public)")
box(ax, 3.6, 4.7, 2.8, 1.0, "data.py\nspans/BIO -> target JSON")
box(ax, 7.2, 4.7, 2.0, 1.0, "prompts.py\nsame instruction")
box(ax, 10.0, 4.7, 1.8, 1.0, "train.py\nLoRA r=16,\nanswer-only loss")
arrow(ax, 2.8, 5.2, 3.6, 5.2); arrow(ax, 6.4, 5.2, 7.2, 5.2); arrow(ax, 9.2, 5.2, 10.0, 5.2)
box(ax, 3.6, 2.7, 2.8, 1.0, "inference.py\nbatched greedy decoding\n+ latency / token log")
box(ax, 7.2, 2.7, 2.0, 1.0, "Base model\n+ 3 shots", fc="#f4f4f4", ec=GREY)
box(ax, 10.0, 2.7, 1.8, 1.0, "Base model\n+ LoRA adapter\n(0-shot)", fc="#fdf0e2", ec=ORG)
arrow(ax, 8.2, 4.7, 8.2, 3.7); arrow(ax, 10.9, 4.7, 10.9, 3.7)
arrow(ax, 7.2, 3.2, 6.4, 3.2)  # few-shot base -> shared generator
# LoRA model -> shared generator, routed below the base-model box so it does not cross it
ax.plot([10.5, 10.5, 5.9], [2.7, 2.4, 2.4], color="#444", lw=1.3)
arrow(ax, 5.9, 2.4, 5.9, 2.7)
box(ax, 3.6, 0.9, 2.8, 1.0, "parsing.py\nextract -> parse -> repair\n-> pydantic schema")
arrow(ax, 5.0, 2.7, 5.0, 1.9)
box(ax, 7.2, 0.9, 2.0, 1.0, "metrics.py +\nevaluate.py\nstrict / relaxed F1,\nbootstrap CIs")
box(ax, 10.0, 0.9, 1.8, 1.0, "serve.py\nFastAPI /extract\nretry + repair")
arrow(ax, 6.4, 1.4, 7.2, 1.4)  # parsed outputs -> scorer
arrow(ax, 11.4, 2.7, 11.4, 1.9)  # the adapter is what the service loads
# the service reuses the same parser
ax.plot([4.4, 4.4, 10.6], [0.9, 0.55, 0.55], color="#444", lw=1.3)
arrow(ax, 10.6, 0.55, 10.6, 0.9)
ax.text(0.2, 5.95, "Data and training", fontsize=8, color=ACC, fontweight="bold", va="center")
ax.text(0.2, 3.2, "Two systems,\none generator", fontsize=8, color=ACC, fontweight="bold", va="center")
ax.text(0.2, 1.4, "Shared parsing,\nscoring and\nserving", fontsize=8, color=ACC, fontweight="bold", va="center")
fig.savefig(OUT / "architecture.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 2. precision vs recall
fig, axs = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True, sharex=True)
for ax, mode in zip(axs, ["strict", "relaxed"]):
    for sysn, c, mk in [(BASE, GREY, "o"), (LORA, ACC, "s")]:
        for t, fill in [("chemicals", True), ("diseases", False), ("overall", True)]:
            p, r = v(sysn, mode, t, "precision"), v(sysn, mode, t, "recall")
            big = t == "overall"
            ax.scatter(r, p, s=110 if big else 55, marker=mk, color=c,
                       facecolors=c if t != "diseases" else "white", linewidths=1.6, zorder=3)
    ax.set_title(f"{mode.capitalize()} match"); ax.set_xlabel("Recall")
    ax.set_xlim(0.15, 0.95); ax.set_ylim(0.4, 1.0); ax.grid(alpha=0.25)
axs[0].set_ylabel("Precision")
from matplotlib.lines import Line2D
axs[1].legend(handles=[Line2D([], [], marker="o", ls="", color=GREY, label="Base, few-shot (k=3)"),
                       Line2D([], [], marker="s", ls="", color=ACC, label="LoRA, 0-shot"),
                       Line2D([], [], marker="o", ls="", color="#555", ms=11, label="large = overall"),
                       Line2D([], [], marker="o", ls="", color="#555", ms=6, label="small filled = chemicals"),
                       Line2D([], [], marker="o", ls="", color="#555", mfc="white", ms=6, label="small hollow = diseases")],
              loc="lower right", frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT / "precision_recall.png", dpi=200); plt.close(fig)

# 3. F1 by type with CIs
fig, axs = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True)
types = ["chemicals", "diseases", "overall"]
for ax, mode in zip(axs, ["strict", "relaxed"]):
    for i, (sysn, c, lab) in enumerate([(BASE, GREY, "Base, few-shot"), (LORA, ACC, "LoRA")]):
        vals = [v(sysn, mode, t, "f1") for t in types]
        lo = [vals[j] - S[sysn]["entity"][mode][t]["f1"]["lo"] for j, t in enumerate(types)]
        hi = [S[sysn]["entity"][mode][t]["f1"]["hi"] - vals[j] for j, t in enumerate(types)]
        xs = [j + (i - 0.5) * 0.36 for j in range(3)]
        ax.bar(xs, vals, 0.34, color=c, label=lab, yerr=[lo, hi], capsize=3, ecolor="#333")
        for x, y in zip(xs, vals): ax.text(x, 0.03, f"{y:.2f}", ha="center", color="white", fontsize=8, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(types); ax.set_ylim(0, 1); ax.set_title(f"{mode.capitalize()} F1 (95% bootstrap CI)")
axs[0].set_ylabel("F1"); axs[0].legend(frameon=False, loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "f1_by_type.png", dpi=200); plt.close(fig)

# 4. cost profile: prompt tokens, latency
fig, axs = plt.subplots(1, 2, figsize=(8, 2.7))
names = ["Base,\nfew-shot", "LoRA"]
cols = [GREY, ACC]
for ax, vals, title, fmt in [
    (axs[0], [S[BASE]["tokens"]["prompt_mean"], S[LORA]["tokens"]["prompt_mean"]], "Prompt tokens per example", "{:.0f}"),
    (axs[1], [S[BASE]["latency_s"]["mean"], S[LORA]["latency_s"]["mean"]], "Mean latency per example (s)", "{:.3f}")]:
    ax.bar(names, vals, color=cols, width=0.55)
    for x, y in enumerate(vals): ax.text(x, y, fmt.format(y), ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=10); ax.set_ylim(0, max(vals) * 1.2)
fig.tight_layout(); fig.savefig(OUT / "cost_profile.png", dpi=200); plt.close(fig)
print("ok")
