import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import least_squares
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf


# Define uma data de Início e Fim
start_dt = datetime(year=2015, month=11, day=1)
end_dt = datetime(year=2021, month=5, day=1)

# Lê o arquivo CSV do BVSP
bvsp_df = pd.read_csv("bvsp.csv")
bvsp_df["Date"] = pd.to_datetime(bvsp_df["Date"])
bvsp_df = bvsp_df[bvsp_df["Date"] > start_dt]
bvsp_df = bvsp_df[bvsp_df["Date"] < end_dt]
bvsp_df = bvsp_df.set_index("Date")

# # Baixa os dados do DOW JONES usando a API yfinance
df = yf.download(["^HSI"], start=start_dt, end=end_dt)
  
# # As colunas estão em formato MultiIndex, converte para formato normal
df.columns = [x[0] for x in df.columns]

df = bvsp_df
sns.lineplot(x=np.arange(len(df)), y=df["Close"])


def model(t, params):
    A, B, D, C, W, T, F = params
    return A + (B*(T-t)**D)*(1+C*np.cos(W*np.log(T-t) + F))


def resid(params, t, y):
    return y - model(t, params)


t = np.arange(len(df[:1000]))
y = np.log(df[:1000]["Close"]).to_numpy()


A0  = np.mean(y)
B0  = np.ptp(y)
T0 = t.max() + 0.1*t.max()
D0  = 0.5
C0  = 0.2
W0  = 10.0
F0  = 0.2

x0 = [A0, B0, D0, C0, W0, T0, F0]

bounds = ([-np.inf, -np.inf, 0.0, -np.inf, 5.0, t.max()+10e-6, -np.inf],
          [np.inf, np.inf, 1.0, np.inf, 15.0, np.inf, np.inf])

res = least_squares(
    fun = resid,
    args = (t, y),
    x0 = x0,
    method="trf",
    bounds=bounds,
    max_nfev=100000
)
res.success
params = res.x

t = np.arange(len(df))
y = np.log(df["Close"]).to_numpy()

tt = np.linspace(0, params[5], 1000)
ypred = model(tt, params)
sns.lineplot(x=t, y=y)
sns.lineplot(x=tt, y=ypred)