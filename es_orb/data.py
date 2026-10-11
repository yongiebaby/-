"""ES 5分足CSV (ATASエクスポート) の読み込みと、RTH日単位の配列への整形。"""
import numpy as np
import pandas as pd

TICK = 0.25
OR_MINUTES = (5, 15, 30, 60)
MAX_BARS = 78  # 9:30-16:00 ET の5分足本数
MIN_RTH_BARS = 70  # これ未満の日(祝日・短縮取引日)は除外

USECOLS = ["DateTime_UTC", "IsRTH", "Open", "High", "Low", "Close", "ATR20Ticks"]


def _read_one(path):
    if str(path).lower().endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(path) as z:
            name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            with z.open(name) as f:
                df = pd.read_csv(f, sep=";", usecols=USECOLS)
    else:
        df = pd.read_csv(path, sep=";", usecols=USECOLS)
    return df


def load_csv(paths):
    if isinstance(paths, (str, bytes)) or not hasattr(paths, "__iter__"):
        paths = [paths]
    df = pd.concat([_read_one(p) for p in paths], ignore_index=True)
    df["DateTime_UTC"] = pd.to_datetime(df["DateTime_UTC"])
    df = df.sort_values("DateTime_UTC").drop_duplicates("DateTime_UTC").reset_index(drop=True)
    df["et"] = df["DateTime_UTC"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    return df


def build_days(df):
    """RTH日ごとの配列を作る。

    直近3足のhigh/low(ブレイク足の前の3本)とATR(ブレイク足の1本前の値)は
    ETHを含む連続系列から計算するので、OR5でもRTH前の足を参照できる。
    """
    low, high = df["Low"].to_numpy(float), df["High"].to_numpy(float)
    p3l = pd.Series(low).shift(1).rolling(3).min().to_numpy()
    p3h = pd.Series(high).shift(1).rolling(3).max().to_numpy()
    atr = (df["ATR20Ticks"].shift(1) * TICK).to_numpy(float)

    rth = df["IsRTH"].to_numpy() == 1
    et = df["et"]
    minute = ((et.dt.hour * 60 + et.dt.minute) - (9 * 60 + 30)).to_numpy()
    date = et.dt.date.to_numpy()

    idx = np.flatnonzero(rth)
    groups = pd.Series(idx).groupby(date[idx])

    days, excluded = [], []
    rows = []
    for d, g in groups:
        ii = g.to_numpy()
        n = len(ii)
        info = dict(date=d, bars=n, first=int(minute[ii[0]]), last=int(minute[ii[-1]]))
        if n < MIN_RTH_BARS or minute[ii[0]] != 0 or minute[ii[-1]] != 385:
            info["reason"] = "holiday/early close" if n < MIN_RTH_BARS else "irregular session"
            excluded.append(info)
            continue
        ii = ii[(minute[ii] >= 0) & (minute[ii] < 390)][:MAX_BARS]
        days.append(info)
        rows.append(ii)

    D = len(rows)
    shape = (D, MAX_BARS)
    arr = {k: np.full(shape, np.nan) for k in ("O", "H", "L", "C", "P3L", "P3H", "ATR")}
    MIN = np.full(shape, 9999, dtype=np.int64)
    NB = np.zeros(D, dtype=np.int64)
    src = dict(O=df["Open"].to_numpy(float), H=high, L=low, C=df["Close"].to_numpy(float),
               P3L=p3l, P3H=p3h, ATR=atr)
    for di, ii in enumerate(rows):
        n = len(ii)
        NB[di] = n
        MIN[di, :n] = minute[ii]
        for k in arr:
            arr[k][di, :n] = src[k][ii]

    ORH = np.full((len(OR_MINUTES), D), np.nan)
    ORL = np.full((len(OR_MINUTES), D), np.nan)
    for oi, m in enumerate(OR_MINUTES):
        for di in range(D):
            n = NB[di]
            sel = MIN[di, :n] < m
            # ORを構成する足が欠けている日はそのORを無効にする
            if sel.sum() == m // 5:
                ORH[oi, di] = arr["H"][di, :n][sel].max()
                ORL[oi, di] = arr["L"][di, :n][sel].min()

    dates = pd.to_datetime([r["date"] for r in days])
    return dict(**arr, MIN=MIN, NB=NB, ORH=ORH, ORL=ORL,
                dates=dates, year=dates.year.to_numpy(np.int64),
                days=pd.DataFrame(days), excluded=pd.DataFrame(excluded))


def load(paths):
    return build_days(load_csv(paths))
