#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import sys

REQUIRED_PKGS = [
    ("pandas", "pandas"),
    ("yfinance", "yfinance"),
    ("requests", "requests"),
    ("bs4", "beautifulsoup4"),
    ("plotly", "plotly"),
]

_missing = []
for import_name, pip_name in REQUIRED_PKGS:
    try:
        __import__(import_name)
    except Exception:
        _missing.append(pip_name)

if _missing:
    print("\n[!] Missing packages:", ", ".join(_missing))
    print("Run and retry:\n  pip3 install -U " + " ".join(_missing))
    sys.exit(1)

import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def ensure_output_dir(dirname: str = "outputs") -> str:
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    return dirname


def clean_revenue_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", "Revenue"])

    cols_lower = [str(c).lower() for c in df.columns]
    date_idx = next((i for i, c in enumerate(cols_lower) if "date" in c or "period" in c), 0)
    rev_idx = next((i for i, c in enumerate(cols_lower) if "revenue" in c), 1 if df.shape[1] > 1 else 0)

    df = df.iloc[:, [date_idx, rev_idx]].copy()
    df.columns = ["Date", "Revenue"]

    df = df.dropna(how="all")
    df["Revenue"] = (
        df["Revenue"].astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df = df[df["Revenue"] != ""]
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
    df = df.dropna(subset=["Revenue"])

    def _norm_date(s: str) -> str:
        s = str(s)
        m = re.search(r"(\d{4}-\d{2}-\d{2}|\d{4}-\d{2}|\d{4})", s)
        return m.group(1) if m else s

    df["Date"] = df["Date"].astype(str).map(_norm_date)
    return df.reset_index(drop=True)


def get_revenue_table(url: str) -> pd.DataFrame:
    """Robust parser for Macrotrends revenue page that tolerates header/row mismatch."""
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X)"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", {"class": "table"})
    if table is not None:
        ths = [th.get_text(strip=True) for th in table.find_all("th")]
        rows = [[td.get_text(strip=True) for td in tr.find_all("td")] for tr in table.find_all("tr")]
        rows = [r for r in rows if r]
        df = pd.DataFrame(rows)
        if ths and len(ths) == df.shape[1]:
            df.columns = ths
        return clean_revenue_table(df)

    all_tables = pd.read_html(r.text)
    candidates = []
    for t in all_tables:
        if t.shape[1] >= 2 and any("revenue" in str(c).lower() for c in t.columns):
            candidates.append(t)
    if not candidates and all_tables:
        candidates = [max(all_tables, key=lambda t: t.shape[1])]
    if not candidates:
        return pd.DataFrame(columns=["Date", "Revenue"])
    return clean_revenue_table(candidates[0])


def get_stock_data(ticker: str) -> pd.DataFrame:
    tkr = yf.Ticker(ticker)
    df = tkr.history(period="max")
    df.reset_index(inplace=True)
    keep_cols = [c for c in ["Date", "Close"] if c in df.columns]
    return df[keep_cols].copy()


def get_tesla_revenue() -> pd.DataFrame:
    return get_revenue_table("https://www.macrotrends.net/stocks/charts/TSLA/tesla/revenue")


def get_gme_revenue() -> pd.DataFrame:
    return get_revenue_table("https://www.macrotrends.net/stocks/charts/GME/gamestop/revenue")


def make_graph(stock_data: pd.DataFrame, revenue_data: pd.DataFrame, stock: str, title: str | None = None):
    if title is None:
        title = f"{stock} Stock Price vs. Revenue"

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=(f"{stock} Historical Close Price", f"{stock} Quarterly Revenue")
    )

    fig.add_trace(
        go.Scatter(x=stock_data["Date"], y=stock_data["Close"], mode="lines", name=f"{stock} Close"),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=revenue_data["Date"], y=revenue_data["Revenue"], name=f"{stock} Revenue"),
        row=2, col=1
    )

    fig.update_layout(height=700, title_text=title, xaxis_rangeslider_visible=False, template="plotly_white")
    return fig


def main():
    outdir = ensure_output_dir()

    tesla_data = get_stock_data("TSLA")
    print("\nQ1: Tesla Stock Data (head):\n", tesla_data.head())
    tesla_data.head().to_csv(os.path.join(outdir, "Q1_tesla_data_head.csv"), index=False)

    tesla_revenue = get_tesla_revenue()
    print("\nQ2: Tesla Revenue (tail):\n", tesla_revenue.tail())
    tesla_revenue.tail().to_csv(os.path.join(outdir, "Q2_tesla_revenue_tail.csv"), index=False)

    gme_data = get_stock_data("GME")
    print("\nQ3: GameStop Stock Data (head):\n", gme_data.head())
    gme_data.head().to_csv(os.path.join(outdir, "Q3_gme_data_head.csv"), index=False)

    gme_revenue = get_gme_revenue()
    print("\nQ4: GameStop Revenue (tail):\n", gme_revenue.tail())
    gme_revenue.tail().to_csv(os.path.join(outdir, "Q4_gme_revenue_tail.csv"), index=False)

    fig_tsla = make_graph(tesla_data, tesla_revenue, stock="Tesla", title="Tesla — Stock Price & Revenue")
    fig_tsla.write_html(os.path.join(outdir, "Q5_tesla_dashboard.html"), include_plotlyjs="cdn")
    try:
        import plotly.io as pio
        pio.write_image(fig_tsla, os.path.join(outdir, "Q5_tesla_dashboard.png"), scale=2, width=1000, height=700)
    except Exception as e:
        print("PNG export skipped (install kaleido to enable):", e)

    fig_gme = make_graph(gme_data, gme_revenue, stock="GameStop", title="GameStop — Stock Price & Revenue")
    fig_gme.write_html(os.path.join(outdir, "Q6_gamestop_dashboard.html"), include_plotlyjs="cdn")
    try:
        import plotly.io as pio
        pio.write_image(fig_gme, os.path.join(outdir, "Q6_gamestop_dashboard.png"), scale=2, width=1000, height=700)
    except Exception as e:
        print("PNG export skipped (install kaleido to enable):", e)

    print("\nAll done! Open the 'outputs' folder for CSVs and dashboards.")

if __name__ == "__main__":
    main()
