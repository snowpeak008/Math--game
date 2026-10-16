"""生成改版前后节奏对比图。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False
OUT = Path("charts"); OUT.mkdir(exist_ok=True)
B = json.load(open("results/before.json")); A = json.load(open("results/after.json"))
INTENT = {"wheel": 1, "mill": 4, "granary": 10, "merchant": 25}  # 分钟内解锁
NAMES = {"paddy": "稻田", "wheel": "水车", "mill": "碾坊", "granary": "粮仓", "merchant": "米商"}

# 1. 第一轮产量曲线
fig, ax = plt.subplots(figsize=(10, 5.5))
for res, lab, c in [(B, "改版前", "tab:red"), (A, "改版后", "tab:blue")]:
    r = res["runs"][0]
    xs = [t / 60 for t, _, _ in r["series"]]; ys = [y for _, y, _ in r["series"]]
    ax.step(xs, ys, where="post", label=f"{lab}（第1轮，{r['prestige']['t']/60:.0f} 分钟转生）", color=c)
    fb = r["pacing"]["first_buy"]
    for g, nm in NAMES.items():
        if g in fb and g != "paddy":
            y = next(v for t, v, i in r["series"] if i == g)
            ax.scatter([fb[g] / 60], [y], color=c, zorder=5)
            ax.annotate(nm, (fb[g] / 60, y), textcoords="offset points", xytext=(4, -12), fontsize=8, color=c)
for g, m in INTENT.items():
    ax.axvline(m, color="gray", ls=":", lw=0.8); ax.text(m, ax.get_ylim()[0] if False else 1.5, f"目标:{NAMES[g]}≤{m}分", rotation=90, fontsize=7, color="gray", va="bottom")
ax.axvspan(45, 60, color="green", alpha=0.08, label="首次转生目标 45–60 分")
ax.set_yscale("log"); ax.set_xlabel("游戏时间（分钟）"); ax.set_ylabel("米粒产量 / 秒（对数）")
ax.set_title("米粒测试版：第一轮节奏曲线（贪心玩家，回本时间策略）"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "01_第一轮产量曲线_前后对比.png", dpi=130); plt.close(fig)

# 2. 首次购买时间 vs 目标
fig, ax = plt.subplots(figsize=(8, 4.5))
gs = list(INTENT); x = range(len(gs)); w = .28
for k, (res, lab, c) in enumerate([(B, "改版前", "tab:red"), (A, "改版后", "tab:blue")]):
    fb = res["runs"][0]["pacing"]["first_buy"]
    vals = [fb.get(g, float("nan")) / 60 for g in gs]
    bars = ax.bar([i + (k - .5) * w for i in x], vals, w, label=lab, color=c)
    ax.bar_label(bars, fmt="%.1f", fontsize=8)
ax.scatter(list(x), [INTENT[g] for g in gs], marker="_", s=900, color="black", label="目标上限", zorder=5)
ax.set_xticks(list(x), [NAMES[g] for g in gs]); ax.set_ylabel("首次购买（分钟）"); ax.set_title("各生产者解锁时间 vs 节奏目标（第1轮）")
ax.legend(); ax.grid(axis="y", alpha=.3); fig.tight_layout(); fig.savefig(OUT / "02_解锁时间_vs_目标.png", dpi=130); plt.close(fig)

# 3. 等待时间分布（第1轮前30分钟）
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, key, title in [(axs[0], "waits", "任意两次购买之间的等待"), (axs[1], "mwaits", "“有意义目标”之间的等待")]:
    for res, lab, c in [(B, "改版前", "tab:red"), (A, "改版后", "tab:blue")]:
        ws = res["runs"][0]["pacing_30min"][key]
        ax.hist(ws, bins=[0, 2, 5, 10, 20, 30, 60, 120, 240, 480, 960], alpha=.5, label=f"{lab}（n={len(ws)}, 最大{max(ws):.0f}s）", color=c)
    ax.set_xscale("log"); ax.set_xlabel("等待（秒，对数）"); ax.set_ylabel("次数"); ax.set_title(title); ax.legend(fontsize=8)
    ax.axvline(60 if key == "waits" else 120, color="k", ls="--", lw=.8)
fig.suptitle("第1轮前30分钟等待时间分布（虚线=目标上限）"); fig.tight_layout(); fig.savefig(OUT / "03_等待时间分布.png", dpi=130); plt.close(fig)

# 4. 4小时全程（含转生）
fig, ax = plt.subplots(figsize=(11, 5))
for res, lab, c in [(B, "改版前", "tab:red"), (A, "改版后", "tab:blue")]:
    for i, r in enumerate(res["runs"]):
        pts = [(t / 60, y) for t, y, _ in r["series"]]
        if pts:
            ax.step(*zip(*pts), where="post", color=c, lw=1, label=lab if i == 0 else None)
        if r["prestige"]:
            ax.axvline(r["prestige"]["t"] / 60, color=c, alpha=.35, ls="--")
            ax.text(r["prestige"]["t"] / 60, 2, f"+{r['prestige']['gain']:.0f}", color=c, fontsize=7)
ax.set_yscale("log"); ax.set_xlabel("游戏时间（分钟）"); ax.set_ylabel("米粒产量/秒（对数）")
ax.set_title(f"4 小时全程：转生（虚线，+稻魂）  改版前共 {B['souls_end']:.0f} 魂，改版后 {A['souls_end']:.0f} 魂")
ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / "04_四小时全程_转生.png", dpi=130); plt.close(fig)

# 5. 敏感性扫描
S = json.load(open("results/sweep.json"))
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
names = [s["name"] for s in S]
axs[0].barh(names, [s["first_prestige_min"] or 0 for s in S], color="tab:green")
axs[0].set_xlabel("首次转生时间（分钟）"); axs[0].axvspan(45, 60, color="green", alpha=.1); axs[0].invert_yaxis()
axs[0].set_title("首次转生时间（扫描用“峰值速率”策略）")
axs[1].barh(names, [s["lifetime_rice"] for s in S], color="tab:orange"); axs[1].set_xscale("log"); axs[1].invert_yaxis()
axs[1].set_xlabel("3 小时累计米粒（对数）"); axs[1].set_title("3 小时累计产出")
fig.suptitle("单参数敏感性扫描（基线 = 改版前）"); fig.tight_layout(); fig.savefig(OUT / "05_敏感性扫描.png", dpi=130); plt.close(fig)
print(sorted(p.name for p in OUT.iterdir()))
