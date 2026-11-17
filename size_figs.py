# -*- coding: utf-8 -*-
"""吸附機容量 ads_size_factor 多組比較圖（不改 model.py，沿用其 sweep 與 plot_energy_row）"""
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model as m
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = HERE / "outputs"
SF = [0.5, 1.0, 1.5, 2.0, 3.0]
pA = m.Params()
pB = replace(pA, base_supply_fixed=40.0)
wx = m.load_weather(pA)
T_grid = np.arange(40.0, 70.01, 0.5)

res = {}
for tag, p in (("A", pA), ("B", pB)):
    for sf in SF:
        pp = replace(p, ads_size_factor=sf)
        res[tag, sf] = (pp, *m.sweep(pp, wx, T_grid))
        print(tag, sf, "max", round(res[tag, sf][1]["saving_kW"].max(), 1), flush=True)

# ---- 圖 7：B 溫水基準，每個容量一列（fig1 同款） ----
fig, ax = plt.subplots(len(SF), 2, figsize=(14, 4.6 * len(SF)))
for r, sf in enumerate(SF):
    pp, df, base = res["B", sf]
    m.plot_energy_row(ax[r], df, base, pp, f"B 溫水基準・容量 {sf}×Q_air")
    ax[r, 0].set_ylim(10.85, 11.15)
    ax[r, 1].set_ylim(-260, 330)
fig.tight_layout()
m.save_fig(fig, OUT / "fig2_容量比較_B逐列.png")
plt.close(fig)

# ---- 圖 8：A / B 疊圖（每條線一個容量） ----
cmap = plt.get_cmap("viridis")
cols = {sf: cmap(i / (len(SF) - 1) * 0.9) for i, sf in enumerate(SF)}
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
for row, (tag, ttl) in enumerate((("A", "A 冷水基準"), ("B", "B 溫水基準（供水 40 °C）"))):
    a0, a1, a2 = axs[row]
    for sf in SF:
        pp, df, base = res[tag, sf]
        c = cols[sf]
        a0.plot(df["T_ret"], df["saving_kW"], color=c, lw=2, label=f"{sf}×Q_air")
        i = df["saving_kW"].idxmax()
        a0.plot(df.loc[i, "T_ret"], df.loc[i, "saving_kW"], "*", color=c, ms=11, mec="k", mew=0.5)
        a1.plot(df["T_ret"], df["chiller_saved_kW"], color=c, lw=2, label=f"冰機省電 {sf}×")
        a1.plot(df["T_ret"], -df["dP_aux_kW"], color=c, lw=1.5, ls="--")
        a2.plot(df["T_ret"], df["scn_ads_coverage"] * 100, color=c, lw=2, label=f"{sf}×Q_air")
    df0 = res[tag, SF[0]][1]
    a1.plot(df0["T_ret"], -df0["dP_IT_kW"], color=m.C_RED, lw=2.5, label="-ΔP_IT（與容量無關）")
    a0.axhline(0, color="gray", lw=0.8); a1.axhline(0, color="gray", lw=0.8)
    for a_ in (a0, a1, a2):
        a_.axvline(pA.T_supply_limit + pA.loop_dT, color="red", ls=":", lw=1)
        a_.set_xlabel("液冷回水溫度 T_ret (°C)"); a_.grid(alpha=0.3)
    a0.set_title(f"{ttl}：淨節省 ΔP vs. 容量（★ = 各容量最大值）", fontsize=10)
    a0.set_ylabel("年平均淨節省 (kW)"); a0.legend(fontsize=8, loc="lower left")
    a1.set_title(f"{ttl}：冰機省電（實線）與 -ΔP_aux（虛線）", fontsize=10)
    a1.set_ylabel("年平均功率 (kW)"); a1.legend(fontsize=7, loc="lower left", ncol=2)
    a2.set_title(f"{ttl}：吸附機冷量覆蓋率", fontsize=10)
    a2.set_ylabel("全年冷量覆蓋率 (%)"); a2.set_ylim(0, 105); a2.legend(fontsize=8, loc="upper left")
fig.tight_layout()
m.save_fig(fig, OUT / "fig3_容量比較_疊圖.png")
plt.close(fig)

for tag in "AB":
    for sf in SF:
        df = res[tag, sf][1]; i = df["saving_kW"].idxmax(); r = df.loc[i]
        print(f"{tag} {sf:>4}: max {r.saving_kW:7.1f} kW @ {r.T_ret:.1f}  冰機省 {r.chiller_saved_kW:6.1f}  ΔP_IT {r.dP_IT_kW:6.1f}  ΔP_aux {r.dP_aux_kW:6.1f}  覆蓋 {r.scn_ads_coverage*100:4.0f}%")
