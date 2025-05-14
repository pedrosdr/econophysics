import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import least_squares
import itertools
import plotnine as gg
import yfinance as yf

# carrega série BVSP entre start_dt e end_dt
start_dt = datetime(2015,11,1)
end_dt = datetime(2021,5,1)

# start_dt = datetime(2004,11,1)
# end_dt = datetime(2010,5,1)

# start_dt = datetime(2002,11,1)
# end_dt = datetime(2010,5,1)

# Ibovespa
bvsp_df = (pd.read_csv("bvsp.csv", parse_dates=["Date"])
             .query("Date > @start_dt and Date < @end_dt")
             .set_index("Date"))
df = bvsp_df.copy()

# # Dow Jones
# df = yf.download(["^DJI"], start = start_dt, end = end_dt)
# df.columns = [x[0] for x in df.columns]

# # Nasdaq
# df = yf.download(["^IXIC"], start = start_dt, end = end_dt)
# df.columns = [x[0] for x in df.columns]

y_full = np.log(df["Close"].to_numpy())
t_full = np.arange(len(df))

import seaborn as sns
sns.lineplot(x=t_full, y=y_full)


# define t e y para a fase de busca inicial
t = np.arange(850)
y = y_full[:850]


# modelo e função de resíduos
def model(t, params):
    A, B, D, C, W, T, F = params
    return A + (B*(T-t)**D)*(1 + C*np.cos(W*np.log(T-t) + F))


def resid(params, t, y):
    return y - model(t, params)


# grade de chute inicial (x0)
grid = {
    "A": [np.mean(y), y[0], y.max()],
    "B": [np.ptp(y), 10.0, -10.0],
    "D": [0.1, 0.5],
    "C": [1.0, 10.0],
    "W": [5.0, 10.0],
    "T": [t.max()*1.05, t.max()*1.10, t.max()*1.50],
    "F": [1.0, 10.0]
}

# gera lista de tuplas com todas as combinações de (A,B,D,C,W,T,F)
x0s = list(itertools.product(*grid.values()))

# limites para o least_squares
bounds = (
    [-np.inf, -np.inf, 0.0, -np.inf, 5.0, t.max()+1e-6, -np.inf],
    [ np.inf,  np.inf, 1.0,  np.inf, 15.0,      np.inf,  np.inf]
)

# percorre todas as chutes iniciais, guarda (mse, resultado)
# True Region
# Levenberg–Marquardt
# resultados = []
# for i in range(len(x0s)):
#     x0 = x0s[i]
#     res = least_squares(
#         resid, x0, args=(t,y),
#         bounds=bounds, 
#         method="trf", max_nfev=100_000)
#     mse = np.mean((y - model(t, res.x))**2)
#     resultados.append((mse, res))
#     print(f"Época: {i}")

# # extrai o ajuste com menor MSE
# best_mse, best_res = min(resultados, key=lambda x: x[0])
# best_params = best_res.x

# print(f"Melhor MSE encontrado: {best_mse:.6f}")
# print("Parâmetros ótimos:", best_params)

# Pandemia - True Region (2015-2021) Treinado com 850 observações
best_params = [ 1.17471997e+01, -8.43967769e-04,  1.00000000e+00,  
                1.00118380e-01, 1.50000000e+01,  1.16543703e+03, 
                -6.12327663e+01]

# # Crise 2008 - True Region (2004-2010) Treinado com 850 observações
# best_params = [ 1.15821767e+01, -6.74477044e-03,  7.75165631e-01, 
#                 -5.98349787e-02, 1.50000000e+01,  1.07970040e+03, 
#                 -3.62970531e+01]

# Crise 2008 - Levenberg–Marquardt (2004-2010) Treinado com 850 observações
# best_params = [ 5.65788741e+00,  3.06094534e+18, -4.21427903e+00, 
#                 1.34300430e-02, 2.96418839e+02,  1.71288502e+04, 
#                 -2.85235701e+03]

# Crise 2008 - Dow Jones (True Region) (2004-2010) Treinado com 850 observações
# best_params = [ 1.60640712e+02, -1.50827850e+02,  4.30261945e-04, 
#                -6.78770278e-04, 5.00000000e+00,  1.26121442e+03,  
#                4.99871983e+01]

# Crise 2008 - Nasdaq (True Region) (2002-2010) Treinado com 1250 observações
# best_params = [ 7.90893465e+00, -4.12163503e-04,  1.00000000e+00, 
#                -2.45553479e-01, 1.00537646e+01,  1.31678191e+03,  
#                1.20894409e-01]

# aplica o modelo a toda a série
y_pred_full = model(t_full, best_params)

# plota os gráficos
dfplot = pd.DataFrame({
    "ln_s": y_full,
    "ln_s_pred": y_pred_full,
    "date": df.index    
})

gg.ggplot(dfplot) + gg.theme_light() +\
    gg.geom_line(
        mapping = gg.aes(
            x="date",
            y="ln_s"
        ),
        size=0.3
    ) +\
    gg.geom_line(
        mapping = gg.aes(
            x="date",
            y="ln_s_pred"
        ),
        size=1
    ) +\
    gg.scale_x_datetime(
        labels=lambda x: [dt.strftime("%Y") for dt in x]
    ) +\
    gg.ggtitle("Ajuste do Modelo Log-Periódico") +\
    gg.labs(y="log(preço)") +\
    gg.theme(
        panel_grid=gg.element_blank(),
        axis_title_x=gg.element_blank(),
        plot_title=gg.element_text(hjust=0.5)
    )