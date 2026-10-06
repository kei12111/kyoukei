"""キャリー×モメンタム 月次ローテーションの検証 (ルールは RULES.md で事前固定)

入力: data.csv (行形式 kind,code,YYYY-MM,value)
  FX,<CUR>,YYYY-MM,<CUR per EUR 月末値>  (ECB EXR M.<CUR>.EUR.SP00.E)
  IR,<AREA>,YYYY-MM,<政策金利 %>          (BIS WS_CBPOL)
使い方: python carry_mom.py data.csv
"""
import sys

import numpy as np
import pandas as pd

CURS = ["EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF", "NOK", "SEK"]
AREA = {"EUR": "XM", "GBP": "GB", "JPY": "JP", "AUD": "AU", "NZD": "NZ",
        "CAD": "CA", "CHF": "CH", "NOK": "NO", "SEK": "SE", "USD": "US"}
COST_ONE_WAY = 0.0003
SWAP_MARKUP = 0.005
START, END = "2010-01", "2025-12"


def load(path):
    df = pd.read_csv(path, header=None, names=["kind", "code", "month", "value"])
    df = df[df.kind.isin(["FX", "IR"])]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    fx = df[df.kind == "FX"].pivot(index="month", columns="code", values="value")
    ir = df[df.kind == "IR"].pivot(index="month", columns="code", values="value")
    # 日銀は2013-05〜2016-08に政策金利の目標がなく BIS に値がないため直前値で埋める
    ir = ir.sort_index().ffill()
    # 外貨1単位あたりのUSD (外貨買い/USD売りの価格)
    px = pd.DataFrame(index=fx.index)
    px["EUR"] = fx["USD"]
    for c in CURS[1:]:
        px[c] = fx["USD"] / fx[c]
    diff = pd.DataFrame({c: ir[AREA[c]] - ir["US"] for c in CURS}) / 100.0
    return px.sort_index(), diff.sort_index()


def backtest(px, diff, n=3, mom_len=12, use_carry=True, use_mom=True):
    spot_ret = px.pct_change()
    mom = px / px.shift(mom_len) - 1
    months = [m for m in px.index if START <= m <= END]
    prev_w = pd.Series(0.0, index=CURS)
    rets = {}
    for m in months:
        i = px.index.get_loc(m)
        sig_m = px.index[i - 1]
        score = pd.Series(0.0, index=CURS)
        if use_carry:
            score += diff.loc[sig_m].rank()
        if use_mom:
            score += mom.loc[sig_m].rank()
        # 同点はキャリー優先
        order = sorted(CURS, key=lambda c: (score[c], diff.loc[sig_m, c]), reverse=True)
        w = pd.Series(0.0, index=CURS)
        w[order[:n]] = 1.0 / (2 * n)
        w[order[-n:]] = -1.0 / (2 * n)
        gross = (w * (spot_ret.loc[m] + diff.loc[sig_m] / 12)).sum()
        cost = (w - prev_w).abs().sum() * COST_ONE_WAY + w.abs().sum() * SWAP_MARKUP / 12
        rets[m] = gross - cost
        # 月中の値動きでウェイトがずれる分は無視(毎月末に等ウェイトへ戻す)
        prev_w = w
    return pd.Series(rets)


def stats(r):
    ann_ret = r.mean() * 12
    ann_vol = r.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    scaled = r * (0.10 / ann_vol)
    eq = (1 + scaled).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
            "mdd_at_10vol": mdd, "months": len(r)}


def main(path):
    px, diff = load(path)
    missing = px.loc["2009-01":END].isna().sum().sum() + diff.loc["2009-01":END].isna().sum().sum()
    print(f"欠損セル数(2009-01〜{END}): {missing}")
    base = backtest(px, diff)
    full = stats(base)
    h1 = stats(base[base.index <= "2017-12"])
    h2 = stats(base[base.index >= "2018-01"])
    print("\n== 基本ルール (上位3/下位3, モメンタム12か月, キャリー+モメンタム) ==")
    for k, s in [("全期間", full), ("前半2010-17", h1), ("後半2018-25", h2)]:
        print(f"{k}: 年率リターン {s['ann_ret']:.2%}  年率ボラ {s['ann_vol']:.2%}  "
              f"シャープ {s['sharpe']:.2f}  10%ボラ換算MDD {s['mdd_at_10vol']:.1%}")
    yearly = base.groupby(base.index.str[:4]).apply(lambda x: (1 + x).prod() - 1)
    print("\n年別リターン(グロス1倍):")
    print(" ".join(f"{y}:{v:+.1%}" for y, v in yearly.items()))

    variants = {
        "上位/下位2": dict(n=2), "上位/下位4": dict(n=4),
        "モメンタム6か月": dict(mom_len=6), "モメンタム3か月": dict(mom_len=3),
        "キャリーのみ": dict(use_mom=False), "モメンタムのみ": dict(use_carry=False),
    }
    print("\n== 近傍パラメータ ==")
    vs = {}
    for name, kw in variants.items():
        vs[name] = stats(backtest(px, diff, **kw))["sharpe"]
        print(f"{name}: シャープ {vs[name]:.2f}")

    checks = [
        ("1. 全期間シャープ ≥ 0.5", full["sharpe"] >= 0.5),
        ("2. 前半・後半ともシャープ > 0", h1["sharpe"] > 0 and h2["sharpe"] > 0),
        ("3. 10%ボラ換算MDD ≤ 30%", full["mdd_at_10vol"] >= -0.30),
        ("4. 近傍6通り中4通り以上でシャープ > 0.3", sum(v > 0.3 for v in vs.values()) >= 4),
    ]
    print("\n== 判定 ==")
    for name, ok in checks:
        print(f"{'合格' if ok else '不合格'}  {name}")
    print("総合:", "合格" if all(ok for _, ok in checks) else "不合格")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data.csv")
