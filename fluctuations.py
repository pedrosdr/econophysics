import pandas as pd
import numpy as np
import yfinance as yf
import plotnine as gg
from datetime import datetime
from typing import List

# Define uma data de Início e Fim
start_dt = datetime(year=1993, month=1, day=1)
end_dt = datetime(year=2012, month=4, day=9)

# Lê a base da taxa selic
selic_df = pd.read_csv("selic.csv", sep=";")

# renomeia as colunas
selic_df.columns = ["date", "selic"]

# Remove o texto do footer na última linha
selic_df = selic_df.iloc[:-1,:]

# Converte a coluna data de string para datetime
selic_df["date"] = pd.to_datetime(selic_df["date"], format=("%d/%m/%Y"))

# Remove os separadores de milhar da selic e converte para float64
selic_df["selic"] = selic_df["selic"].str.replace(",","").astype("float64")

# A taxa selic está em percentual, divide por 100 para obter o valor real
selic_df["selic"] = selic_df["selic"] / 100.0

# Calcula o retorno log-preço da selic e converte para taxa diária
selic_df["rf"] = np.log(selic_df["selic"] + 1.0) / 252.0


# Baixa os dados do Ibovespa usando a API yfinance
bvsp_df = yf.download(["^BVSP"], start=start_dt, end=end_dt)

# As colunas estão em formato MultiIndex, converte para formato normal
bvsp_df.columns = [x[0] for x in bvsp_df.columns]

# Mantém apenas os valores de fechamento e renomeia para 's'
bvsp_df = bvsp_df.loc[:,["Close"]]
bvsp_df = bvsp_df.rename(columns={"Close": "s"})

# Calcula o retorno log-preço (nominal)
bvsp_df["rn"] = np.log(bvsp_df["s"] / bvsp_df["s"].shift(1))

# Calcula a volatilidade nominal (módulo do retorno nominal)
bvsp_df["vn"] = np.abs(bvsp_df["rn"])

# Remove os valores nulos (primeira linha ficou nula)
bvsp_df = bvsp_df.dropna()

# Reseta o index para obter a coluna de datas
bvsp_df = bvsp_df.reset_index()

# Deixa os nomes das colunas minúsculos
bvsp_df.columns = bvsp_df.columns.str.lower()


# Faz um left join do dataframe do IBovespa com o dataframe da taxa Selic
# ligando os dois pela data
df = pd.merge(bvsp_df, selic_df, on="date", how="left")

# Datas que não estavam presentes na tabela da taxa selic:
not_matched_values = df[df.isna().any(axis=1)]

# Remove os valores nulos
df = df.dropna()


# Calcula os retornos reais (prêmio de risco)
df["r"] = df["rn"] - df["rf"]

# Calcula a volatilidade real (módulo do retorno real)
df["v"] = np.abs(df["r"])

# Calcula a volatilidade realizada para uma janela de um mês (22 dias úteis)
df["sd"] = df["r"].shift(1).rolling(22).std()


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
    fig = gg.ggplot(data=df) + gg.theme_light() +\
        gg.geom_line(mapping=gg.aes(y=y, x=x), size=0.5, color="#4d58bf") +\
        gg.scale_x_datetime(labels=lambda x: [dt.strftime("%b/%Y") for dt in x])
    
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
           panel_border=gg.element_blank(),
           panel_grid=gg.element_blank(),
           axis_ticks_major_y=gg.element_blank(),
           axis_text_x=gg.element_text(angle=45),
           axis_title_x=gg.element_blank(),
           plot_title=gg.element_text(hjust=0.5)
        )
    return fig


# Converte a data de início e fim da análise para string
start_str = start_dt.strftime("%b/%Y")
end_str = end_dt.strftime("%b/%Y")

# Plota os gráficos (são autoexplicativos)
chart(
      df, 
      "date", 
      "s", 
      title=f"Fechamento do IBovespa de {start_str} a {end_str}",
      ylab="Preço"
)

chart(
      df, 
      "date", 
      "selic", 
      title=f"Taxa Selic de {start_str} a {end_str}",
      ylab="Taxa Selic (base 252 dias)",
      yfmt=lambda x: f"{x*100:.2f}%"
)

chart(
      df, 
      "date", 
      "rn", 
      title=f"Retorno nominal de {start_str} a {end_str}",
      ylab="Retorno nominal (r + rf)",
      hline=True
)

chart(
      df, 
      "date", 
      "r", 
      title=f"Retorno real de {start_str} a {end_str}",
      ylab="Retorno real (r)",
      hline=True
)

chart(
      df, 
      "date", 
      "vn", 
      title=f"Volatilidade nominal de {start_str} a {end_str}",
      ylab="Volatilidade nominal |r + rf|",
      hline=True
)

chart(
      df, 
      "date", 
      "v", 
      title=f"Volatilidade real de {start_str} a {end_str}",
      ylab="Volatilidade real |r|",
      hline=True
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
        x = np.log(lags)
        y = np.log(correlations)
        hline=False
    else:
        x = lags
        y = correlations
        hline=True
    
    
    # Cria um gráfico de linhas usando ggplot 
    fig = gg.ggplot() + gg.theme_light() +\
        gg.geom_line(
            mapping=gg.aes(x=x, y=y), 
            size=0.5, 
            color="#4d58bf"
        )
    
    # Se a escala não é logarítmica, adiciona uma linha pontilhada passando por
    # y = 0
    if hline:
        fig = fig + gg.geom_hline(
            yintercept=0.0, 
            linetype="dashed", 
            size=1.1, 
            color="#383838"
        )  
    
    # Se a escala é logarítmica, exponenciaa os eixos x e y para mostrar os
    # valores originas do lag e da correlação
    if log:
        fig = fig + gg.scale_x_continuous(
            labels = lambda breaks: [f"{np.exp(br):.1f}" for br in breaks]
        ) +\
        gg.scale_y_continuous(
            labels = lambda breaks: [f"{np.exp(br):.2f}" for br in breaks]
        )
    
    # Define os títulos dos eixos e estilos do gráfico
    fig = fig + gg.ggtitle(title) +\
        gg.labs(
            x="Tempo (dias úteis)",
            y=ylab
        ) +\
        gg.theme(
           panel_border=gg.element_blank(),
           panel_grid=gg.element_blank(),
           axis_ticks_major_y=gg.element_blank(),
           plot_title=gg.element_text(hjust=0.5)
        )
    return fig


# Autocorrelações do retorno log-preço real
autocorr_chart(
    df["r"], 
    log=False, 
    n_lags=100,
    title=f"Autocorrelação do retorno real de {start_str} a {end_str}",
    ylab="Autocorrelação do retorno real (r)")

# Autocorrelações do retorno log-preço nominal
autocorr_chart(
    df["rn"], 
    log=False, 
    n_lags=100,
    title=f"Autocorrelação do retorno nominal de {start_str} a {end_str}",
    ylab="Autocorrelação do retorno nominal (r + rf)")

# Autocorrelaçoes da volatilidade real (módulo de r)
autocorr_chart(
    df["v"], 
    log=True, 
    n_lags=100,
    title=f"Autocorrelação da volatilidade real de {start_str} a {end_str} (log-log)",
    ylab="Autocorrelação da volatilidade real |r|")

# Autocorrelaçoes da volatilidade nominal (módulo de rn)
autocorr_chart(
    df["vn"], 
    log=True, 
    n_lags=100,
    title=f"Autocorrelação da volatilidade nominal de {start_str} a {end_str} (log-log)",
    ylab="Autocorrelação da volatilidade nominal |r + rf|")
