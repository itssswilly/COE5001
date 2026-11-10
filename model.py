# -*- coding: utf-8 -*-
"""
AI 資料中心液冷廢熱驅動吸附式冰機 — 逐時(8760 h)能耗 + 經濟比較模型
參數來源見「參數與資料來源.txt」（參數名稱與本檔 Params 欄位一致）

執行： python model.py      → 產出 outputs/ 內的圖、CSV 與 summary.txt
"""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "PingFang TC",
                                   "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


# ======================================================================
# 參數
# ======================================================================
@dataclass(frozen=True)
class Params:
    # ---- 負載 ----
    P_IT: float = 10_000.0          # kW，基準 IT 功率
    f_liquid: float = 0.82          # 液冷帶走比例
    # ---- 氣象 ----
    weather_csv: str = ""           # 若有實測逐時資料：欄位 T_db, RH（8760 列）
    design_wb: float = 28.0         # °C 夏季設計濕球
    seed: int = 2026
    # ---- 液冷迴路 ----
    loop_dT: float = 10.0           # K，CDU 供回水溫差
    tower_approach: float = 4.0     # K，冷卻塔趨近溫度
    hx_approach: float = 2.0        # K，板式熱交換器趨近溫度
    base_supply_min: float = 25.0   # °C，Baseline 供水下限（防結露/控制）
    base_supply_fixed: float = 0.0  # °C，>0 表示 Baseline 已採溫水冷卻、供水固定於此值
    T_supply_limit: float = 45.0    # °C，W45 供水上限（超過只標示警告）
    k_leak: float = 0.0015          # 1/K，每 K 供水升溫 → 液冷 IT 功率增加比例
    # ---- 電動冰機 ----
    T_chw: float = 18.0             # °C 冰水出水
    carnot_eff: float = 0.48
    # ---- 吸附式冰機（ΔΔt 特性方程式）----
    ads_B: float = 1.5
    ads_ddt0: float = 2.0
    ads_TH_nom: float = 65.0
    ads_TM_nom: float = 30.0
    ads_TL_nom: float = 18.0
    ads_COP_max: float = 0.62
    ads_tau: float = 4.0
    ads_cf_max: float = 1.3         # 容量係數上限
    ads_TH_min: float = 50.0        # °C，最低驅動熱水溫度（廠商範圍 50–95 °C）
    ads_el_ratio: float = 0.02      # kW電 / kW冷（本體）
    ads_hw_pump: float = 0.008      # kW電 / kW驅動熱
    ads_size_factor: float = 1.5    # 額定容量 = 係數 × Q_air（額定點 65/30/18）
    # ---- 輔機 ----
    rej_hot_loop: float = 0.012     # kW/kW 排熱（液冷迴路）
    rej_cond_loop: float = 0.025    # kW/kW 排熱（冷凝迴路）
    chw_pump: float = 0.015
    cdu_pump: float = 0.010
    ups_loss: float = 0.04
    misc_fixed: float = 100.0
    # ---- 用水 ----
    evap_L_per_kWh: float = 1.3
    cycles: float = 4.0
    # ---- 碳排 ----
    emission_factor: float = 0.466  # kg CO2e / kWh（2025 產業係數）
    ef_decline: float = 0.0
    carbon_price: float = 0.0       # NT$/t（影子價格）
    # ---- 經濟 ----
    elec_price: float = 4.30        # NT$/kWh
    elec_escalation: float = 0.02
    water_price: float = 12.075     # NT$/m3
    water_escalation: float = 0.03
    ads_equip_cost: float = 16_000.0  # NT$/kW冷（額定）
    install_factor: float = 1.6
    fixed_integration: float = 10e6   # NT$
    om_rate: float = 0.025
    om_escalation: float = 0.02
    backup_credit: float = 0.0        # NT$/kW，抵減電動冰機容量（預設 0）
    discount_rate: float = 0.07
    life: int = 15


# ======================================================================
# 氣象
# ======================================================================
TAINAN = dict(  # 中央氣象署 臺南 1991–2020
    Tmax=[22.9, 24.0, 26.5, 29.3, 31.4, 32.4, 33.1, 32.6, 32.5, 30.8, 28.1, 24.4],
    Tmean=[17.8, 18.9, 21.6, 24.9, 27.5, 28.9, 29.4, 28.9, 28.6, 26.3, 23.4, 19.6],
    Tmin=[14.5, 15.5, 18.0, 21.7, 24.7, 26.3, 26.7, 26.3, 26.0, 23.3, 20.3, 16.3],
    RH=[75.0, 75.5, 73.8, 74.6, 75.3, 77.2, 76.1, 78.3, 75.2, 72.8, 74.5, 73.7],
)


def wet_bulb_stull(T, RH):
    """Stull (2011) 濕球溫度經驗式，T [°C]、RH [%]"""
    return (T * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
            + np.arctan(T + RH) - np.arctan(RH - 1.676331)
            + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH) - 4.686035)


def load_weather(p: Params) -> pd.DataFrame:
    if p.weather_csv:
        df = pd.read_csv(HERE / p.weather_csv)
        df = df.iloc[:8760].reset_index(drop=True)
    else:
        rng = np.random.default_rng(p.seed)
        hours = np.arange(8760)
        doy = hours // 24 + 0.5
        hod = hours % 24
        mid = np.array([15.5, 45, 74.5, 105, 135.5, 166, 196.5, 227.5, 258, 288.5, 319, 349.5])
        xp = np.concatenate([mid - 365, mid, mid + 365])

        def interp(key):
            v = np.array(TAINAN[key])
            return np.interp(doy, xp, np.tile(v, 3))

        Tmean, Tmax, Tmin, RHm = interp("Tmean"), interp("Tmax"), interp("Tmin"), interp("RH")
        # 日均溫 AR(1) 擾動：冬季大、夏季小
        n_day = 365
        sig = np.interp(np.arange(n_day) + 0.5, xp, np.tile(
            [1.8, 1.8, 1.6, 1.2, 0.9, 0.7, 0.7, 0.7, 0.8, 1.0, 1.3, 1.7], 3))
        e = np.zeros(n_day)
        for d in range(1, n_day):
            e[d] = 0.7 * e[d - 1] + np.sqrt(1 - 0.49) * rng.normal()
        dT_day = np.repeat(e * sig, 24)
        dRH_day = np.repeat(np.clip(rng.normal(0, 4, n_day), -10, 10), 24)
        diurnal = np.cos(2 * np.pi * (hod - 14) / 24)
        T = Tmean + dT_day + 0.5 * (Tmax - Tmin) * diurnal + rng.normal(0, 0.4, 8760)
        RH = np.clip(RHm + dRH_day - 12 * diurnal + rng.normal(0, 2, 8760), 35, 99)
        df = pd.DataFrame({"T_db": T, "RH": RH})
    df["T_wb"] = wet_bulb_stull(df["T_db"].values, df["RH"].values)
    df.index = pd.date_range("2025-01-01", periods=len(df), freq="h")
    return df


# ======================================================================
# 設備模型
# ======================================================================
def base_supply(p: Params, Twb):
    if p.base_supply_fixed > 0:
        return np.full_like(Twb, p.base_supply_fixed, dtype=float)
    return np.maximum(Twb + p.tower_approach + p.hx_approach, p.base_supply_min)


def cop_electric(p: Params, T_wb):
    T_evap = p.T_chw - 5.0
    T_cond = T_wb + p.tower_approach + 5.0 + 2.0  # 冷卻水進水 + 溫升 5 K + 趨近 2 K
    cop = p.carnot_eff * (T_evap + 273.15) / np.maximum(T_cond - T_evap, 5.0)
    return np.clip(cop, 3.0, 9.0)


def ads_ddt(p: Params, T_H, T_M, T_L):
    return (T_H - T_M) - p.ads_B * (T_M - T_L)


def ads_performance(p: Params, T_H, T_M):
    """回傳 (容量係數 cf, 熱力 COP)"""
    ddt = ads_ddt(p, T_H, T_M, p.T_chw)
    ddt_nom = ads_ddt(p, p.ads_TH_nom, p.ads_TM_nom, p.ads_TL_nom)
    x = ddt - p.ads_ddt0
    cf = np.clip(x / (ddt_nom - p.ads_ddt0), 0.0, p.ads_cf_max)
    cf = np.where(np.asarray(T_H) >= p.ads_TH_min, cf, 0.0)
    cop = np.where(x > 0, p.ads_COP_max * (1 - np.exp(-np.maximum(x, 0) / p.ads_tau)), 0.0)
    return cf, cop


# ======================================================================
# 兩情境逐時計算
# ======================================================================
def simulate(p: Params, wx: pd.DataFrame, T_ret: float | None):
    """T_ret=None → Baseline；否則 Scenario（液冷回水設定溫度 T_ret）。回傳逐時 DataFrame（kW）"""
    Twb = wx["T_wb"].values
    n = len(Twb)
    Q_air = p.P_IT * (1 - p.f_liquid) * np.ones(n)
    P_IT_liq0 = p.P_IT * p.f_liquid

    T_sup_base = base_supply(p, Twb)
    cop_e = cop_electric(p, Twb)

    if T_ret is None:
        T_sup = T_sup_base
        dP_IT = np.zeros(n)
        Q_ads = np.zeros(n); Q_drive = np.zeros(n); P_ads_el = np.zeros(n); cop_th = np.zeros(n)
    else:
        T_sup = np.full(n, T_ret - p.loop_dT)
        dP_IT = P_IT_liq0 * p.k_leak * np.maximum(T_sup - T_sup_base, 0.0)
        T_H = T_ret - p.hx_approach
        T_M = np.maximum(Twb + p.tower_approach, 20.0)
        cf, cop_th = ads_performance(p, T_H, T_M)
        rated = p.ads_size_factor * Q_air[0]
        Q_liq_avail = P_IT_liq0 + dP_IT
        cap = rated * cf
        Q_ads = np.minimum(cap, Q_air)
        with np.errstate(divide="ignore", invalid="ignore"):
            Q_drive = np.where(cop_th > 0, Q_ads / cop_th, 0.0)
        # 廢熱不足時（理論上不會發生）限制
        over = Q_drive > Q_liq_avail
        Q_drive = np.where(over, Q_liq_avail, Q_drive)
        Q_ads = np.where(over, Q_drive * cop_th, Q_ads)
        P_ads_el = p.ads_el_ratio * Q_ads + p.ads_hw_pump * Q_drive

    P_IT = p.P_IT + dP_IT
    Q_liq = P_IT_liq0 + dP_IT
    Q_ech = Q_air - Q_ads                      # 電動冰機補足
    P_chiller = Q_ech / cop_e
    # 排熱
    Q_rej_hot = Q_liq - Q_drive                # 液冷剩餘熱 → 冷卻塔
    Q_rej_ech = Q_ech + P_chiller              # 電動冰機冷凝
    Q_rej_ads = Q_ads + Q_drive                # 吸附機排熱 = 冷量 + 驅動熱
    P_tower = (p.rej_hot_loop * Q_rej_hot
               + p.rej_cond_loop * (Q_rej_ech + Q_rej_ads))
    P_pump = p.chw_pump * Q_air + p.cdu_pump * Q_liq
    P_other = p.ups_loss * P_IT + p.misc_fixed
    P_total = P_IT + P_chiller + P_ads_el + P_tower + P_pump + P_other
    Q_rej = Q_rej_hot + Q_rej_ech + Q_rej_ads
    water_Lph = Q_rej * p.evap_L_per_kWh * (1 + 1 / (p.cycles - 1))

    return pd.DataFrame(dict(
        T_wb=Twb, T_sup=T_sup, cop_e=cop_e, cop_th=cop_th,
        P_IT=P_IT, dP_IT=dP_IT, Q_ads=Q_ads, Q_drive=Q_drive,
        P_chiller=P_chiller, P_ads_el=P_ads_el, P_tower=P_tower, P_pump=P_pump,
        P_other=P_other, P_total=P_total, Q_rej=Q_rej, water_Lph=water_Lph,
    ), index=wx.index)


def annual(p: Params, h: pd.DataFrame) -> dict:
    E_tot = h["P_total"].sum()              # kWh
    E_IT = h["P_IT"].sum()
    water = h["water_Lph"].sum()            # L
    Q_air_E = p.P_IT * (1 - p.f_liquid) * len(h)
    return dict(
        E_total_MWh=E_tot / 1e3, E_IT_MWh=E_IT / 1e3,
        P_avg_kW=E_tot / len(h), PUE=E_tot / E_IT,
        water_m3=water / 1e3, WUE_L_per_kWh=water / E_IT,
        CO2_t=E_tot * p.emission_factor / 1e3,
        ads_coverage=h["Q_ads"].sum() / Q_air_E,
        dP_IT_avg_kW=h["dP_IT"].mean(),
        over_limit_h=int((h["T_sup"] > p.T_supply_limit + 1e-9).sum()),
    )


# ======================================================================
# 經濟
# ======================================================================
def economics(p: Params, base: dict, scen: dict) -> dict:
    rated = p.ads_size_factor * p.P_IT * (1 - p.f_liquid)
    equip = rated * p.ads_equip_cost
    capex = equip * p.install_factor + p.fixed_integration - rated * p.backup_credit
    dE = (base["E_total_MWh"] - scen["E_total_MWh"]) * 1e3   # kWh/yr 節省
    dW = base["water_m3"] - scen["water_m3"]                  # m3/yr 節省
    years = np.arange(p.life + 1)
    cf = np.zeros(p.life + 1)
    cf[0] = -capex
    rows = []
    for y in years[1:]:
        elec = dE * p.elec_price * (1 + p.elec_escalation) ** (y - 1)
        water = dW * p.water_price * (1 + p.water_escalation) ** (y - 1)
        co2_t = dE * p.emission_factor * (1 - p.ef_decline) ** (y - 1) / 1e3
        carbon = co2_t * p.carbon_price
        om = -equip * p.om_rate * (1 + p.om_escalation) ** (y - 1)
        cf[y] = elec + water + carbon + om
        rows.append(dict(year=y, elec=elec, water=water, carbon=carbon, om=om, net=cf[y]))
    disc = (1 + p.discount_rate) ** -years
    dcf = cf * disc
    npv = dcf.sum()
    cum, dcum = np.cumsum(cf), np.cumsum(dcf)

    def payback(c):
        idx = np.where(c >= 0)[0]
        if len(idx) == 0 or idx[0] == 0:
            return np.nan
        k = idx[0]
        return k - 1 + (-c[k - 1]) / (c[k] - c[k - 1])

    irr = np.nan
    if cf[1:].sum() > -cf[0]:
        lo, hi = -0.99, 2.0
        f = lambda r: (cf * (1 + r) ** -years).sum()
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        irr = mid
    return dict(rated_kW=rated, capex=capex, equip=equip, dE_kWh=dE, dW_m3=dW,
                npv=npv, irr=irr, payback=payback(cum), dpayback=payback(dcum),
                cf=cf, dcf=dcf, cum=cum, dcum=dcum, table=pd.DataFrame(rows))


# ======================================================================
# 掃描與分析
# ======================================================================
def run_case(p: Params, wx, T_ret, base_ann=None):
    if base_ann is None:
        base_ann = annual(p, simulate(p, wx, None))
    h = simulate(p, wx, T_ret)
    s = annual(p, h)
    eco = economics(p, base_ann, s)
    return h, s, eco, base_ann


def sweep(p: Params, wx, T_grid):
    base_ann = annual(p, simulate(p, wx, None))
    rows = []
    for T in T_grid:
        h, s, eco, _ = run_case(p, wx, T, base_ann)
        rows.append(dict(T_ret=T, **{f"scn_{k}": v for k, v in s.items()},
                         saving_kW=base_ann["P_avg_kW"] - s["P_avg_kW"],
                         chiller_saved_kW=(simulate_cache(p, wx)["P_chiller"].mean() - h["P_chiller"].mean()),
                         dP_IT_kW=h["dP_IT"].mean(),
                         npv=eco["npv"], irr=eco["irr"], payback=eco["payback"],
                         capex=eco["capex"]))
    df = pd.DataFrame(rows)
    # 輔機 + 其他（UPS 損失等）的淨增量 = 總差 − 冰機省 + ΔP_IT
    df["dP_aux_kW"] = df["chiller_saved_kW"] - df["dP_IT_kW"] - df["saving_kW"]
    return df, base_ann


_cache = {}


def simulate_cache(p, wx):
    key = (p, id(wx))
    if key not in _cache:
        _cache[key] = simulate(p, wx, None)
    return _cache[key]


def dynamic_case(p: Params, wx, T_grid, base_ann=None):
    """逐日動態控制：每天選一個回水溫度（或不啟用吸附機、維持 Baseline 供水），使當日總耗電最低。"""
    hb = simulate(p, wx, None)
    if base_ann is None:
        base_ann = annual(p, hb)
    cands = [hb] + [simulate(p, wx, T) for T in T_grid]
    labels = np.array([np.nan] + list(T_grid))
    day = np.arange(len(hb)) // 24
    daily = np.array([c["P_total"].groupby(day).sum().values for c in cands])  # (n_cand, 365)
    pick = daily.argmin(axis=0)
    pick_h = pick[day]
    cols = cands[0].columns
    arr = np.stack([c.values for c in cands])                                 # (n_cand, 8760, ncol)
    h = pd.DataFrame(arr[pick_h, np.arange(len(hb))], index=hb.index, columns=cols)
    h["T_ret_set"] = labels[pick_h]
    s = annual(p, h)
    eco = economics(p, base_ann, s)
    return h, s, eco, base_ann


def zero_crossings(x, y):
    out = []
    for i in range(len(x) - 1):
        if np.sign(y[i]) != np.sign(y[i + 1]) and not np.isnan(y[i] + y[i + 1]):
            out.append(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return out


# ======================================================================
# 主程式
# ======================================================================
C_BASE, C_SCN, C_ACC, C_RED = "#4C72B0", "#DD8452", "#55A868", "#C44E52"


def save_fig(fig, path, dpi=150, tries=5):
    """Windows 上圖檔若被預覽/同步程式暫時鎖住會寫入失敗，重試幾次"""
    import time
    for k in range(tries):
        try:
            fig.savefig(path, dpi=dpi)
            return
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(1.0)


def fmt_irr(e):
    return f"{e['irr']*100:.1f}%" if np.isfinite(e["irr"]) else "—（不回本）"


def fmt_pb(e, life):
    return f"{e['dpayback']:.1f} 年" if np.isfinite(e["dpayback"]) else f"> {life} 年"


def econ_study(p: Params, wx, size_grid, T_coarse, T_dyn):
    """固定回水（溫度 × 容量 NPV 地圖）與逐日動態兩種策略的經濟最佳點"""
    npv_map = np.zeros((len(size_grid), len(T_coarse)))
    dyn_rows = []
    for i, sf in enumerate(size_grid):
        pi = replace(p, ads_size_factor=sf)
        b = annual(pi, simulate(pi, wx, None))
        for j, T in enumerate(T_coarse):
            npv_map[i, j] = run_case(pi, wx, T, b)[2]["npv"]
        _, sd, ed, _ = dynamic_case(pi, wx, T_dyn, b)
        pv_benefit = ed["npv"] + ed["capex"]          # 不含投資的淨效益現值
        dyn_rows.append(dict(size_factor=sf, npv=ed["npv"], saving_kW=ed["dE_kWh"] / 8760,
                             coverage=sd["ads_coverage"], pv_benefit=pv_benefit,
                             allow_cost_per_kW=(pv_benefit - p.fixed_integration) / ed["rated_kW"]))
    ii, jj = np.unravel_index(np.argmax(npv_map), npv_map.shape)
    r = dict(p=p, npv_map=npv_map, size_grid=size_grid, T_coarse=T_coarse, T_dyn=T_dyn)
    r["sf_fix"], r["T_fix"] = size_grid[ii], T_coarse[jj]
    r["p_fix"] = replace(p, ads_size_factor=r["sf_fix"])
    r["h_fix"], r["s_fix"], r["eco_fix"], r["base"] = run_case(r["p_fix"], wx, r["T_fix"])
    r["dyn_df"] = pd.DataFrame(dyn_rows)
    r["sf_dyn"] = r["dyn_df"].loc[r["dyn_df"]["npv"].idxmax(), "size_factor"]
    r["p_dyn"] = replace(p, ads_size_factor=r["sf_dyn"])
    r["h_dyn"], r["s_dyn"], r["eco_dyn"], _ = dynamic_case(r["p_dyn"], wx, T_dyn)
    r["on_days"] = int(r["h_dyn"]["T_ret_set"].resample("D").first().notna().sum())
    r["best"] = "dyn" if r["eco_dyn"]["npv"] > r["eco_fix"]["npv"] else "fix"
    return r


def best_of(r):
    k = r["best"]
    return r[f"p_{k}"], r[f"s_{k}"], r[f"eco_{k}"], k


def plot_energy_row(ax, df, base_ann, p, title):
    be = zero_crossings(df["T_ret"].values, df["saving_kW"].values)
    ax[0].axhline(base_ann["P_avg_kW"] / 1e3, color=C_BASE, lw=2, label="Baseline（冷卻塔 + 電動冰機）")
    ax[0].plot(df["T_ret"], df["scn_P_avg_kW"] / 1e3, color=C_SCN, lw=2, label="Scenario（吸附式冰機）")
    box = dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.95)
    for n_, t in enumerate(be):
        ax[0].axvline(t, color="gray", ls="--", lw=1, label="損益兩平溫度" if n_ == 0 else None)
    ax[0].axvline(p.T_supply_limit + p.loop_dT, color="red", ls=":", lw=1, label="供水 45 °C 上限")
    be_txt = ("損益兩平回水溫度：\n" + "、".join(f"{t:.1f} °C" for t in be)) if be else "無損益兩平點\n（全範圍 Scenario 較耗電）"
    ax[0].text(0.98, 0.04, be_txt, transform=ax[0].transAxes, ha="right", va="bottom",
               fontsize=9, bbox=box, zorder=5)
    ax[0].set_ylabel("年平均總耗電 (MW)")
    ax[0].set_title(f"{title}：回水溫度 vs. 總耗電", fontsize=10)
    ax[0].legend(fontsize=8, loc="upper left")

    i = df["saving_kW"].idxmax()
    ax[1].plot(df["T_ret"], df["chiller_saved_kW"], color=C_ACC, lw=2, label="電動冰機省電 Q_air/COP_e")
    ax[1].plot(df["T_ret"], -df["dP_IT_kW"], color=C_RED, lw=2, label="-ΔP_IT（晶片升溫）")
    ax[1].plot(df["T_ret"], -df["dP_aux_kW"], color="#8172B3", lw=2, label="-ΔP_aux（輔機/UPS）")
    ax[1].plot(df["T_ret"], df["saving_kW"], color="k", lw=2.5, label="淨節省 ΔP")
    ax[1].axhline(0, color="gray", lw=0.8)
    ax[1].plot(df.loc[i, "T_ret"], df.loc[i, "saving_kW"], "k*", ms=12)
    ax[1].annotate(f"最大淨節省 {df.loc[i,'saving_kW']:.0f} kW\n@ {df.loc[i,'T_ret']:.1f} °C",
                   (df.loc[i, "T_ret"], df.loc[i, "saving_kW"]),
                   xytext=(0.04, 0.95), textcoords="axes fraction", ha="left", va="top",
                   fontsize=9, bbox=box, zorder=5,
                   arrowprops=dict(arrowstyle="->", color="k", lw=1))
    a2 = ax[1].twinx()
    a2.plot(df["T_ret"], df["scn_ads_coverage"] * 100, color=C_ACC, ls="--", lw=1.2)
    a2.set_ylabel("冷量覆蓋率 %（虛線）", color=C_ACC, fontsize=9)
    a2.set_ylim(0, 105)
    ax[1].set_ylabel("年平均功率 (kW)")
    ax[1].set_title(f"{title}：節能分解（容量 {p.ads_size_factor}×Q_air，k_leak {p.k_leak*100:.2f}%/K）",
                    fontsize=10)
    ax[1].legend(fontsize=7, loc="lower left")
    for a_ in ax:
        a_.set_xlabel("液冷回水溫度 T_ret (°C)")
        a_.grid(alpha=0.3)
    return be


def cashflow_panel(ax, eco, title, life):
    cf, dcum, cum = eco["cf"], eco["dcum"], eco["cum"]
    yrs = np.arange(len(cf))
    sc = 1e6
    span = max(np.abs(cf).max(), np.abs(dcum).max(), np.abs(cum).max()) / sc
    for y, v in zip(yrs, cf):
        col = C_ACC if v >= 0 else C_RED
        ax.annotate("", xy=(y, v / sc), xytext=(y, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2, mutation_scale=14))
        ax.text(y + 0.18, v / sc + (0.03 if v >= 0 else -0.07) * span,
                f"{v/sc:.1f}", ha="center", fontsize=7, color=col)
    ax.axhline(0, color="k", lw=1.2)
    ax.plot(yrs, dcum / sc, "o-", color=C_BASE, lw=1.8, ms=4, label="累積折現現金流")
    ax.plot(yrs, cum / sc, "s--", color="gray", lw=1, ms=3, label="累積未折現現金流")
    ax.set_xticks(yrs)
    ax.set_xlabel("年")
    ax.set_ylabel("百萬 NT$")
    ax.set_title(f"{title}\nCAPEX {eco['capex']/1e6:.1f} 百萬｜年節電 {eco['dE_kWh']/1e3:,.0f} MWh｜"
                 f"NPV {eco['npv']/1e6:.1f} 百萬｜IRR {fmt_irr(eco)}｜折現回收 {fmt_pb(eco, life)}",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")


def strat_label(r):
    if r["best"] == "dyn":
        return (f"逐日動態回水（{r['T_dyn'][0]:.0f}–{r['T_dyn'][-1]:.0f} °C／不啟用），"
                f"容量 {r['eco_dyn']['rated_kW']:.0f} kW 冷，啟用 {r['on_days']} 天")
    return f"全年固定回水 {r['T_fix']:.0f} °C，容量 {r['eco_fix']['rated_kW']:.0f} kW 冷"


def main():
    OUT.mkdir(exist_ok=True)
    pA = Params()                                  # A：題目設定，冷卻水追蹤濕球
    pB = replace(pA, base_supply_fixed=40.0)       # B：基準已採溫水冷卻（供水 40 °C）
    wx = load_weather(pA)
    wx[["T_db", "RH", "T_wb"]].round(2).to_csv(OUT / "weather_8760.csv", encoding="utf-8-sig")
    T_grid = np.arange(40.0, 70.01, 0.5)
    size_grid = np.round(np.arange(0.5, 3.01, 0.25), 2)
    T_coarse = np.arange(44.0, 70.01, 1.0)
    T_dyn = np.arange(50.0, 66.01, 1.0)

    # ---------- 1. 溫度掃描 ----------
    dfA, baseA = sweep(pA, wx, T_grid)
    dfB, baseB = sweep(pB, wx, T_grid)
    dfA.to_csv(OUT / "sweep_A_冷水基準.csv", index=False, encoding="utf-8-sig")
    dfB.to_csv(OUT / "sweep_B_溫水基準.csv", index=False, encoding="utf-8-sig")
    warm_curves = {
        "A：追蹤濕球（約 25–34 °C）": dfA,
        "固定供水 35 °C": sweep(replace(pA, base_supply_fixed=35.0), wx, T_grid)[0],
        "B：固定供水 40 °C": dfB,
    }

    # ---------- 2. k_leak 敏感度 ----------
    k_list = [0.0005, 0.001, 0.0015, 0.0025, 0.0035, 0.005]
    k_A = {k: sweep(replace(pA, k_leak=k), wx, T_grid)[0] for k in k_list}
    k_B = {k: sweep(replace(pB, k_leak=k), wx, T_grid)[0] for k in k_list}

    # ---------- 3. 經濟 ----------
    rA = econ_study(pA, wx, size_grid, T_coarse, T_dyn)
    rB = econ_study(pB, wx, size_grid, T_coarse, T_dyn)
    for tag, r in (("A", rA), ("B", rB)):
        best_of(r)[2]["table"].to_csv(OUT / f"cashflow_{tag}_best.csv", index=False, encoding="utf-8-sig")

    # ---------- 4. NPV 龍捲風（B 最佳點）----------
    pBb, _, ecoBb, kB = best_of(rB)

    def npv_of(pp):
        if kB == "dyn":
            return dynamic_case(pp, wx, T_dyn)[2]["npv"]
        return run_case(pp, wx, rB["T_fix"])[2]["npv"]

    npv0 = ecoBb["npv"]
    tornado_spec = [
        ("k_leak 0.05 / 0.50 %/K", dict(k_leak=0.0005), dict(k_leak=0.005)),
        ("設備單價 €220 / €1018 per kW", dict(ads_equip_cost=7_800), dict(ads_equip_cost=36_100)),
        ("電價 3.5 / 5.5 NT$/kWh", dict(elec_price=3.5), dict(elec_price=5.5)),
        ("折現率 4% / 10%", dict(discount_rate=0.04), dict(discount_rate=0.10)),
        ("吸附 COP_max 0.50 / 0.70", dict(ads_COP_max=0.50), dict(ads_COP_max=0.70)),
        ("年限 10 / 20 年", dict(life=10), dict(life=20)),
        ("安裝係數 1.3 / 2.0", dict(install_factor=1.3), dict(install_factor=2.0)),
        ("碳價 0 / 300 NT$/t", dict(carbon_price=0), dict(carbon_price=300)),
        ("最低驅動溫度 55 / 45 °C", dict(ads_TH_min=55.0), dict(ads_TH_min=45.0)),
    ]
    tornado = [(n, npv_of(replace(pBb, **lo)), npv_of(replace(pBb, **hi))) for n, lo, hi in tornado_spec]

    # ======================================================================
    # 繪圖
    # ======================================================================
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    beA = plot_energy_row(ax[0], dfA, baseA, pA, "A 冷水基準")
    beB = plot_energy_row(ax[1], dfB, baseB, pB, "B 溫水基準(供水40°C)")
    fig.tight_layout()
    save_fig(fig, OUT / "fig1_溫度vs總耗電.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    cmap = plt.get_cmap("viridis")
    for a_, kc, ttl in ((axs[0], k_A, "(a) A 冷水基準：k_leak 敏感度"),
                        (axs[1], k_B, "(b) B 溫水基準：k_leak 敏感度")):
        for i, (k, d) in enumerate(kc.items()):
            a_.plot(d["T_ret"], d["saving_kW"], color=cmap(i / (len(kc) - 1)), lw=2,
                    label=f"k_leak = {k*100:.2f} %/K")
        a_.set_title(ttl)
    for (name, d), c in zip(warm_curves.items(), [C_BASE, C_SCN, C_RED]):
        axs[2].plot(d["T_ret"], d["saving_kW"], color=c, lw=2, label=f"Baseline 供水：{name}")
    axs[2].set_title(f"(c) Baseline 供水溫度的影響（k_leak {pA.k_leak*100:.2f}%/K）")
    for a_ in axs:
        a_.axhline(0, color="gray", lw=0.8)
        a_.axvline(pA.T_supply_limit + pA.loop_dT, color="red", ls=":", lw=1)
        a_.set_xlabel("液冷回水溫度 T_ret (°C)")
        a_.set_ylabel("年平均淨節省 (kW)")
        a_.legend(fontsize=8)
        a_.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, OUT / "fig2_敏感度.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    vmax = max(np.abs(rA["npv_map"]).max(), np.abs(rB["npv_map"]).max()) / 1e6
    for a_, r, ttl in ((axs[0], rA, "A 冷水基準"), (axs[1], rB, "B 溫水基準")):
        im = a_.pcolormesh(T_coarse, size_grid, r["npv_map"] / 1e6, cmap="RdBu", vmin=-vmax, vmax=vmax,
                           shading="nearest")
        if r["npv_map"].max() > 0 > r["npv_map"].min():
            cs = a_.contour(T_coarse, size_grid, r["npv_map"] / 1e6, levels=[0], colors="k", linewidths=1.5)
            a_.clabel(cs, fmt="NPV=0", fontsize=8)
        a_.plot(r["T_fix"], r["sf_fix"], "k*", ms=14)
        a_.annotate(f"最大 NPV {r['eco_fix']['npv']/1e6:.1f} 百萬\n@ {r['T_fix']:.0f} °C, {r['sf_fix']}×Q_air",
                    (r["T_fix"], r["sf_fix"]), xytext=(8, 8), textcoords="offset points", fontsize=9)
        a_.set_xlabel("液冷回水溫度 T_ret (°C)（全年固定）")
        a_.set_ylabel("吸附機額定容量 / Q_air")
        a_.set_title(f"{ttl}：NPV 地圖（折現率 {pA.discount_rate*100:.0f}%，{pA.life} 年）")
        fig.colorbar(im, ax=a_, label="NPV（百萬 NT$）")
    fig.tight_layout()
    save_fig(fig, OUT / "fig3_NPV地圖.png")
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(11, 10))
    cashflow_panel(axs[0], best_of(rA)[2], f"(a) A 冷水基準最佳：{strat_label(rA)}", pA.life)
    cashflow_panel(axs[1], best_of(rB)[2], f"(b) B 溫水基準最佳：{strat_label(rB)}", pA.life)
    fig.tight_layout()
    save_fig(fig, OUT / "fig4_現金流量圖.png")
    plt.close(fig)

    tornado.sort(key=lambda t: abs(t[2] - t[1]))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (name, lo, hi) in enumerate(tornado):
        ax.barh(i, (lo - npv0) / 1e6, left=npv0 / 1e6, color=C_RED, alpha=0.8)
        ax.barh(i, (hi - npv0) / 1e6, left=npv0 / 1e6, color=C_ACC, alpha=0.8)
    ax.set_yticks(range(len(tornado)))
    ax.set_yticklabels([t[0] for t in tornado], fontsize=9)
    ax.axvline(npv0 / 1e6, color="k", lw=1)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("NPV（百萬 NT$）　紅=前值 / 綠=後值")
    ax.set_title(f"NPV 敏感度（B 溫水基準最佳點：{strat_label(rB)}）", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    save_fig(fig, OUT / "fig5_NPV龍捲風.png")
    plt.close(fig)

    h_dyn = rB["h_dyn"]
    hb = simulate(rB["p_dyn"], wx, None)
    daily = lambda s: s.resample("D").mean()
    fig, ax = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    ax[0].plot(wx.index, wx["T_wb"], lw=0.5, color="gray")
    ax[0].set_ylabel("濕球 (°C)")
    ax[0].set_title(("合成臺南逐時氣象" if not pA.weather_csv else "實測逐時氣象")
                    + f"｜B 溫水基準・逐日動態策略（容量 {rB['eco_dyn']['rated_kW']:.0f} kW 冷）")
    ax[1].plot(daily(h_dyn["Q_ads"]).index, daily(h_dyn["Q_ads"]), color=C_ACC, label="吸附機冷量")
    ax[1].axhline(pA.P_IT * (1 - pA.f_liquid), color="k", ls="--", lw=1, label="氣冷負載 Q_air")
    ax[1].set_ylabel("kW（日平均）")
    ax[1].set_ylim(0, pA.P_IT * (1 - pA.f_liquid) * 1.35)
    ax[1].legend(fontsize=8, loc="upper center", ncol=2, framealpha=0.95)
    axr = ax[1].twinx()
    axr.plot(h_dyn["T_ret_set"].resample("D").first(), ".", ms=3, color=C_SCN)
    axr.set_ylabel("當日回水設定 °C（點）", color=C_SCN)
    ax[2].plot(daily(hb["P_total"]).index, daily(hb["P_total"]) / 1e3, color=C_BASE, label="Baseline B")
    ax[2].plot(daily(h_dyn["P_total"]).index, daily(h_dyn["P_total"]) / 1e3, color=C_SCN, label="Scenario（動態）")
    ax[2].set_ylabel("總耗電 MW（日平均）")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, OUT / "fig6_逐時年度.png")
    plt.close(fig)

    # ======================================================================
    # 文字摘要
    # ======================================================================
    lines = []
    L = lines.append
    L("=" * 70)
    L(" 結果摘要（自動產生；改參數請改 model.py 並同步更新 參數與資料來源.txt）")
    L("=" * 70)
    L(f"氣象：{'實測 ' + pA.weather_csv if pA.weather_csv else '合成臺南逐時資料'}；"
      f"年均濕球 {wx['T_wb'].mean():.1f} °C，0.4% 超越濕球 {np.percentile(wx['T_wb'], 99.6):.1f} °C"
      f"（設計值 {pA.design_wb}）")
    L("A 冷水基準 = 題目設定：液冷供水 = 濕球 + 6 K（≥25 °C）")
    L("B 溫水基準 = 對照組：液冷供水固定 40 °C（GB300 溫水冷卻設計常見）")
    L("  註：兩情境的 ΔP_IT 都以各自 Baseline 供水為參考，Baseline 本身 IT 功率不另計溫度影響，")
    L("      故 A、B 的 Baseline 耗電數字相同；B 若改用乾式冷卻器排熱，Baseline WUE 會更低（未模擬）。")
    p_unit = pA.ads_equip_cost * pA.install_factor
    for tag, df, base, kc, r, be in (("A", dfA, baseA, k_A, rA, beA), ("B", dfB, baseB, k_B, rB, beB)):
        L("")
        L(f"────────── 情境 {tag} ──────────")
        L(f"Baseline：年平均 {base['P_avg_kW']/1e3:.3f} MW，PUE {base['PUE']:.4f}，年耗電 "
          f"{base['E_total_MWh']:,.0f} MWh，碳排 {base['CO2_t']:,.0f} t，WUE {base['WUE_L_per_kWh']:.3f} L/kWh")
        i = df["saving_kW"].idxmax()
        L(f"[固定回水掃描] 容量 {pA.ads_size_factor}×Q_air，k_leak {pA.k_leak*100:.2f}%/K")
        L(f"  最大淨節省 {df.loc[i,'saving_kW']:.0f} kW @ {df.loc[i,'T_ret']:.1f} °C（冰機省 "
          f"{df.loc[i,'chiller_saved_kW']:.0f}、ΔP_IT {df.loc[i,'dP_IT_kW']:.0f}、ΔP_aux {df.loc[i,'dP_aux_kW']:.0f} kW；"
          f"冷量覆蓋 {df.loc[i,'scn_ads_coverage']*100:.0f}%；ΔPUE {df.loc[i,'scn_PUE']-base['PUE']:+.4f}）")
        L("  損益兩平回水溫度：" + (", ".join(f"{t:.1f} °C" for t in be) if be else "無（全範圍不划算）"))
        L("  k_leak 敏感度（最大節省 @ 溫度；節能為正區間）")
        for k, d in kc.items():
            j = d["saving_kW"].idxmax()
            pz = d[d["saving_kW"] > 0]["T_ret"]
            L(f"    {k*100:.2f} %/K : {d.loc[j,'saving_kW']:6.0f} kW @ {d.loc[j,'T_ret']:.1f} °C；正區間 "
              + (f"{pz.min():.1f}–{pz.max():.1f} °C" if len(pz) else "無"))
        for key, name in (("fix", "固定回水"), ("dyn", "逐日動態")):
            e, s = r[f"eco_{key}"], r[f"s_{key}"]
            extra = f"T_ret {r['T_fix']:.0f} °C" if key == "fix" else f"啟用 {r['on_days']} 天"
            L(f"[經濟・{name}最佳] {extra}，容量 {e['rated_kW']:.0f} kW 冷（{r['sf_'+key]}×Q_air）")
            L(f"  CAPEX {e['capex']/1e6:.1f} 百萬（設備 {e['equip']/1e6:.1f}）；年節電 {e['dE_kWh']/1e3:,.0f} MWh"
              f"（{e['dE_kWh']/8760:.0f} kW）；年節水 {e['dW_m3']:,.0f} m3；年減碳 {e['dE_kWh']*pA.emission_factor/1e3:,.0f} t")
            L(f"  PUE {r['base']['PUE']:.4f} → {s['PUE']:.4f}；WUE {r['base']['WUE_L_per_kWh']:.3f} → "
              f"{s['WUE_L_per_kWh']:.3f}；NPV {e['npv']/1e6:.1f} 百萬；IRR {fmt_irr(e)}；折現回收 {fmt_pb(e, pA.life)}"
              + (f"；⚠ 供水 > 45 °C 共 {s['over_limit_h']} h" if s["over_limit_h"] else ""))
        L("  [損益兩平投資額]（逐日動態策略；NPV=0 時可接受的安裝後單價 = (效益現值−固定整合費)/額定容量）")
        for _, row in r["dyn_df"].iterrows():
            L(f"    容量 {row['size_factor']:.2f}×Q_air：節省 {row['saving_kW']:5.0f} kW，效益現值 "
              f"{row['pv_benefit']/1e6:6.1f} 百萬 → 可接受 {row['allow_cost_per_kW']:8,.0f} NT$/kW"
              f"（目前假設 {p_unit:,.0f}）")
        if r["on_days"]:
            mon = r["h_dyn"]["T_ret_set"].notna().groupby(r["h_dyn"].index.month).mean()
            L("  動態策略各月啟用比例：" + " ".join(f"{m}月{v*100:.0f}%" for m, v in mon.items()))
    L("")
    L(f"[NPV 敏感度・B 最佳點：{strat_label(rB)}]（百萬 NT$：前值 / 後值；基準 {npv0/1e6:.1f}）")
    for name, lo, hi in sorted(tornado, key=lambda t: -abs(t[2] - t[1])):
        L(f"  {name:<26s} {lo/1e6:8.1f} / {hi/1e6:8.1f}")
    L("")
    L("[參數（情境 A；情境 B 僅 base_supply_fixed = 40）]")
    for k, v in asdict(pA).items():
        L(f"  {k} = {v}")
    txt = "\n".join(lines)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
