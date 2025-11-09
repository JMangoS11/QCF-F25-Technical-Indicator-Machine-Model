# !pip install pandas numpy scikit-learn yfinance matplotlib
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

@dataclass
class Config:
    ticker: str = "SPY"
    period: str = "10y"
    start: str = None
    end: str = "2025-11-02"  # Cap testing to last week
    train_end: str = "2024-10-31"  # Leave room for test data until Nov 2
    proba_threshold: float = 0.55
    fee_bps: float = 1.0
    model: str = "rf"
    random_state: int = 42
CFG = Config()

def fetch_prices(ticker, start=None, end=None, period="10y"):
    if start and end:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    elif end and not start:
        # If only end is specified, get 10 years of data up to end date
        import datetime
        end_dt = pd.to_datetime(end)
        start_dt = end_dt - pd.DateOffset(years=10)
        df = yf.download(ticker, start=start_dt.strftime('%Y-%m-%d'), end=end, auto_adjust=True, progress=False)
    elif start and not end:
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    else:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    # Flatten MultiIndex columns and rename
    df.columns = [col[0].lower() for col in df.columns]
    df["ret1"] = df["close"].pct_change()
    return df.dropna()

def rsi(close, window=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100/(1+rs))

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def bollinger(close, n=20, k=2.0):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    up = ma + k*sd
    low = ma - k*sd
    width = (up - low) / (ma + 1e-12)
    pctb = (close - low) / ((up - low) + 1e-12)
    return ma, up, low, width, pctb

def make_features(df):
    out = pd.DataFrame(index=df.index)
    c = df["close"]
    out["rsi"] = rsi(c, 14)
    ml, ms, mh = macd(c)
    out["macd_line"], out["macd_sig"], out["macd_hist"] = ml, ms, mh
    ma, up, low, width, pctb = bollinger(c, 20, 2.0)
    out["bb_pctb"], out["bb_width"], out["bb_ma"] = pctb, width, ma
    for w in [5,10,20,50,100,200]:
        out[f"sma_{w}"] = c.rolling(w).mean()
        out[f"slope_{w}"] = out[f"sma_{w}"].pct_change(5)
    out["mom_10"], out["mom_20"] = c.pct_change(10), c.pct_change(20)
    return out.dropna()

def make_labels(df, horizon=1):
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    return (fwd > 0).astype(int)

px = fetch_prices(CFG.ticker, CFG.start, CFG.end, CFG.period)
X_all = make_features(px)
y_all = make_labels(px)
idx = X_all.index.intersection(y_all.index)
X_all = X_all.loc[idx].shift(1).dropna()
y_all = y_all.loc[X_all.index]

# Split data into train and test sets
train_mask = X_all.index <= CFG.train_end
X_train, X_test = X_all[train_mask], X_all[~train_mask]
y_train, y_test = y_all[train_mask], y_all[~train_mask]

def make_model(name):
    if name == "rf":
        model = RandomForestClassifier(random_state=CFG.random_state, n_jobs=-1)
        grid = {
            "model__n_estimators": [200, 400, 800],
            "model__max_depth": [4, 6, 8, None],
            "model__min_samples_leaf": [5, 10, 20],
        }
    elif name == "gb":
        model = GradientBoostingClassifier(random_state=CFG.random_state)
        grid = {
            "model__n_estimators": [200, 400, 800],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        }
    else:
        raise ValueError("unknown model")
    pipe = Pipeline([("model", model)])
    return pipe, grid

pipe, grid = make_model(CFG.model)
tscv = TimeSeriesSplit(n_splits=5)
gcv = GridSearchCV(pipe, grid, cv=tscv, scoring="roc_auc", n_jobs=-1, verbose=1)
gcv.fit(X_train, y_train)
print("Best params:", gcv.best_params_)
print("Best CV AUC:", gcv.best_score_)

best = gcv.best_estimator_
best.fit(X_train, y_train)
if len(X_test) > 0:
    proba = best.predict_proba(X_test)
    pos_col = list(best.named_steps["model"].classes_).index(1)
    p_up = pd.Series(proba[:, pos_col], index=X_test.index)
    signal = (p_up >= CFG.proba_threshold).astype(int)
else:
    p_up = pd.Series(dtype=float)
    signal = pd.Series(dtype=int)
p_up.head()



def backtest(close, signal, fee_bps=1.0):
    close = close.loc[signal.index]
    ret1 = close.pct_change().fillna(0.0)
    sig = signal.reindex(close.index).fillna(0.0).astype(float)
    turnover = sig.diff().abs().fillna(0.0)
    fee = turnover * (fee_bps/10000.0)
    strat_ret = sig.shift(1).fillna(0.0) * ret1 - fee
    eq = (1 + strat_ret).cumprod()
    bh = (1 + ret1).cumprod()
    return strat_ret, eq, bh

strat_ret, eq, bh = backtest(px.loc[signal.index, "close"], signal, fee_bps=CFG.fee_bps)
def perf(strat_ret):
    ann = 252
    if len(strat_ret) == 0:
        return {k: np.nan for k in ["CAGR","Vol","Sharpe","MaxDD","HitRate"]}
    eq = (1 + strat_ret).cumprod()
    cagr = eq.iloc[-1]**(ann/len(eq)) - 1
    vol = strat_ret.std() * np.sqrt(ann)
    sharpe = (strat_ret.mean()/(strat_ret.std()+1e-12)) * np.sqrt(ann)
    mdd = (eq/eq.cummax() - 1).min()
    hit = (strat_ret > 0).mean()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": mdd, "HitRate": hit}
metrics = perf(strat_ret)
print(pd.Series(metrics))
plt.figure(figsize=(10,5))
plt.plot(eq.index, eq.values, label="Strategy")
plt.plot(bh.index, bh.values, label="Buy & Hold")
plt.title(f"Test Equity Curve ({CFG.ticker})")
plt.legend()
plt.show()


import os

out = pd.DataFrame({
    "proba_up": p_up,
    "signal": signal,
    "ret1": px.loc[signal.index, "ret1"],
})
out["strategy_ret"] = out["signal"].shift(1).fillna(0.0) * out["ret1"]
csv_path = "./test_set_predictions_signals.csv"

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

out.to_csv(csv_path)
csv_path
