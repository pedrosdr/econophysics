import pandas as pd
import numpy as np
import yfinance as yf
import plotnine as gg
from datetime import datetime, timedelta
from typing import List
from scipy.stats import kstest
import scipy as sp
import seaborn as sns

# Define uma data de Início e Fim
start_dt = datetime(year=2000, month=1, day=1)
end_dt = datetime(year=2025, month=5, day=11)


# # Baixa os dados do Ibovespa usando a API yfinance
# bvsp_df = yf.download(["^BVSP"], start=start_dt, end=end_dt)
  
# # As colunas estão em formato MultiIndex, converte para formato normal
# bvsp_df.columns = [x[0] for x in bvsp_df.columns]
# bvsp_df.to_csv("bvsp.csv")

# Lê o arquivo CSV do BVSP
bvsp_df = pd.read_csv("bvsp.csv")
bvsp_df["Date"] = pd.to_datetime(bvsp_df["Date"])
bvsp_df = bvsp_df[bvsp_df["Date"] > start_dt]
bvsp_df = bvsp_df[bvsp_df["Date"] < end_dt]
bvsp_df = bvsp_df.set_index("Date")

# Mantém apenas os valores de fechamento e renomeia para 's'
bvsp_df = bvsp_df.loc[:,["Close"]]
bvsp_df = bvsp_df.rename(columns={"Close": "s"})

# Calcula o retorno log-preço (nominal)
bvsp_df["r"] = np.log(bvsp_df["s"] / bvsp_df["s"].shift(1))

# Calcula a volatilidade nominal (módulo do retorno nominal)
bvsp_df["v"] = np.abs(bvsp_df["r"])

# Remove os valores nulos (primeira linha ficou nula)
bvsp_df = bvsp_df.dropna()

# Reseta o index para obter a coluna de datas
bvsp_df = bvsp_df.reset_index()

# Deixa os nomes das colunas minúsculos
bvsp_df.columns = bvsp_df.columns.str.lower()

df = bvsp_df.copy()


def prices_from_returns(returns, initial_price=1.0, return_initial=True):
    """
    Recebe uma sequência de retornos logarítmicos `returns` e um preço inicial,
    e retorna a série de preços correspondentes.
    
    Parâmetros:
    - returns: iterable de retornos log (float)
    - initial_price: preço inicial no tempo t=0 (default=1.0)
    - return_initial: se True, inclui o preço inicial na série; caso contrário,
      retorna apenas preços a partir do primeiro retorno.
    """
    prices = [initial_price] # lista que armazenará os preços, já iniciada com initial_price
    for r in returns:
        # exp(r) aplica a transformação inversa do log e multiplica pelo último preço
        price = np.exp(r) * prices[len(prices)-1]
        prices.append(price)
    # Retorna com ou sem o preço inicial, conforme flag
    return prices if return_initial else prices[1:]


# Define uma função para plotar os gráficos usanso ggplot
def chart(
        data: pd.DataFrame, 
        x: str, 
        y: str, 
        title: str = "Title", 
        ylab: str ="y", 
        hline: bool =False, 
        yfmt=None) -> gg.ggplot:
    """
    Função que plota os gráficos usando ggplot (plotnine)
    """
    
    # Cria um gráfico de linha usando ggplot, formatando o datetime do eixo x
    fig = gg.ggplot(data=data) + gg.theme_light() +\
        gg.geom_line(mapping=gg.aes(y=y, x=x), size=0.5) +\
        gg.scale_x_datetime(labels=lambda x: [dt.strftime("%Y") for dt in x])
    
    # Adiciona uma linha pontilhada em y=0.0 se hline for True
    if hline:
        fig = fig + gg.geom_hline(yintercept=0.0, linetype="dashed", size=1.3)
        
    # Formata o texto do eixo y se yfmt não for None
    if yfmt:
        fig = fig + gg.scale_y_continuous(
                labels=lambda y: [yfmt(yi) for yi in y]
        )
    
    # Adiciona os títulos e tema do gráfico (tira as linhas de grid, ajusta
    # o ângulo do eixo x, etc)
    fig = fig + gg.ggtitle(title) +\
        gg.labs(
            y=ylab
        ) +\
        gg.theme(
           panel_grid=gg.element_blank(),
           axis_title_x=gg.element_blank(),
           plot_title=gg.element_text(hjust=0.5)
        )
    return fig


# Converte a data de início e fim da análise para string
start_str = start_dt.strftime("%b/%Y")
end_str = end_dt.strftime("%b/%Y")

# Plota os gráficos (são autoexplicativos)
g_s = chart(
      df, 
      "date", 
      "s", 
      title=f"Fechamento do IBovespa de {start_str} a {end_str}",
      ylab="Preço"
)

g_r = chart(
      df, 
      "date", 
      "r", 
      title=f"Retornos Log-Preço do IBovespa de {start_str} a {end_str}",
      ylab="Retorno Log-Preço"
)

g_v = chart(
      df, 
      "date", 
      "v", 
      title=f"Volatilidade do IBovespa de {start_str} a {end_str}",
      ylab="Volatilidade"
)


# Cria uma função que calcula as autocorrelações
def autocorr(returns: pd.Series, n_lags:int = 100) -> List[float]:
    """
    Função que calcula as autocorrelações dos retornos
    """
    
    # Cria uma lista de correlações
    correlations = []
    for tau in range(n_lags):
        # Para cada tau, cria um dataframe contendo o valor dos retornos e o
        # valor dos retornos com um lag de -tau (t + tau)
        r_df = pd.DataFrame({
            "r": returns,
            "rlag": returns.shift(-tau)    
        })
        
        # Remove os valores que ficaram nulos
        r_df = r_df.dropna()
        
        # Calcula a covariância amostral 
        # cov(x,y) = soma((x-xm)*(y-ym)) / (n-1)
        cov = (r_df["r"] - r_df["r"].mean()) * (r_df["rlag"] - r_df["rlag"].mean())
        cov = cov.sum() / (len(cov) - 1.0)
        
        # Calcula a autocorrelação normalizada (coeficiente de correlação)
        # R(x,y) = cov(x,y) / (dp(x) * dp(y))
        corr = cov / (r_df["r"].std() * r_df["rlag"].std())
        
        # Adiciona o valor da correlação à lista de correlações
        correlations.append(float(corr))
    return correlations


# Cria uma função para plotar o gráfico da autocorrelação
def autocorr_chart(
        returns: pd.Series, 
        n_lags:int = 100,
        start:int = 0,
        title: str = "Title",
        ylab: str = "Autocorrelação",
        log=False) -> gg.ggplot:
    """
    Função que plota o gráfico das autocorrelações
    """
    
    # Se a escala é logarítmica o lag mínimo deve ser 1
    if log and start == 0:
        start = 1
    
    lags = list(range(n_lags))[start:]
    correlations = autocorr(returns, n_lags)[start:]
    
    # Se a escala é logarítmica, aplica log aos eixos x e y
    # não mostra a linha pontilhada em 0 se a escala é logarítmica
    if log:
        hline=False
    else:
        hline=True
        
    x = lags
    y = correlations
    
    # Cria um gráfico de linhas usando ggplot 
    fig = gg.ggplot() + gg.theme_light() +\
        gg.geom_line(
            mapping=gg.aes(x=x, y=y), 
            size=0.5
        )
    
    # Se a escala não é logarítmica, adiciona uma linha pontilhada passando por
    # y = 0
    if hline:
        fig = fig + gg.geom_hline(
            yintercept=0.0, 
            linetype="dashed", 
            size=0.5, 
            color="#383838"
        )  
    
    # Se a escala é logarítmica, exponenciaa os eixos x e y para mostrar os
    # valores originas do lag e da correlação
    if log:
        fig = fig + gg.scale_y_log10() + gg.scale_x_log10()
    
    # Define os títulos dos eixos e estilos do gráfico
    fig = fig + gg.ggtitle(title) +\
        gg.labs(
            x="Tempo (dias úteis)",
            y=ylab
        ) +\
        gg.theme(
           panel_grid=gg.element_blank(),
           plot_title=gg.element_text(hjust=0.5)
        )
    return fig


# Autocorrelações do retorno log-preço real
g_ac_v = autocorr_chart(
    df["v"], 
    log=False, 
    n_lags=200,
    title=f"Autocorrelação normalizada da volatilidade do Ibovespa\n{start_str} a {end_str}",
    ylab="Autocorrelação do retorno real do Ibovespa (r)")


# Dados Simulados a partir de uma Normal
df_simul = pd.DataFrame({
    "r": np.random.randn(len(df)) * df["r"].std() + df["r"].mean(),
    "date": df["date"]
})

df_simul["s"] = prices_from_returns(df_simul["r"], return_initial=False)
df_simul["v"] = np.abs(df_simul["r"])


g_s_simul = chart(
      df_simul, 
      "date", 
      "s", 
      title="Preços simulados",
      ylab="Preço"
)

g_r_simul = chart(
      df_simul, 
      "date", 
      "r", 
      title="Retornos Log-Preço simulados",
      ylab="Retorno Log-Preço"
)

g_v_simul = chart(
      df_simul, 
      "date", 
      "v", 
      title="Volatilidade simulada",
      ylab="Volatilidade"
)

g_ac_v_simul = autocorr_chart(
    df_simul["v"], 
    log=False, 
    n_lags=200,
    title="Autocorrelação normalizada da volatilidade simulada",
    ylab="Autocorrelação do retorno real do Ibovespa (r)")


# Plotando os gráficos para comparação
print(g_s)
print(g_r)
print(g_v)
print(g_ac_v)

print(g_s_simul)
print(g_r_simul)
print(g_v_simul)
print(g_ac_v_simul)


# Plotando a distribuição acumulada da volatilidade
counts, edges = np.histogram(df["v"], bins=10000)
prob = counts / counts.sum()
x = edges[1:]
cdf = np.flip(np.flip(prob).cumsum())

df_cdf = pd.DataFrame({
    "x": x,
    "cdf": cdf    
})

df_cdf = df_cdf[df_cdf["x"] > 0.0001]


def plot_cdf(df_cdf, log_x = False, log_y = False):
    fig = gg.ggplot(df_cdf) + gg.theme_light() +\
        gg.geom_line(mapping = gg.aes(
            x="x",
            y="cdf"
        )) +\
        gg.theme(
            panel_grid=gg.element_blank(),
            plot_title=gg.element_text(hjust=0.5)
        ) +\
        gg.labs(
            x="Volatilidade",
            y="P(|r| > |r(t, 1day)|)"
        )
        
    if log_x:
        fig = fig + gg.scale_x_log10()
    
    if log_y:
        fig = fig + gg.scale_y_log10()
        
    return fig


plot_cdf(df_cdf)
plot_cdf(df_cdf, log_y=True)
plot_cdf(df_cdf, log_y=True, log_x=True)

df_cdf["logx"] = np.log(df_cdf["x"])
df_cdf["logcdf"] = np.log(df_cdf["cdf"])

df_filt = df_cdf[df_cdf["x"] > 0.015]
plot_cdf(df_filt, log_x=True, log_y=True)

d_x = df_filt["logx"].iloc[0] - df_filt["logx"].iloc[-1]
d_cdf = df_filt["logcdf"].iloc[0] - df_filt["logcdf"].iloc[-1]
    
coef = d_cdf / d_x
print("Expoente da Lei de Potência ~", coef)
